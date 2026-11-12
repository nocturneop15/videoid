"""GPU OCR: PaddleOCR PP-OCRv5 models via RapidOCR on ONNX Runtime + DirectML (any DX12 GPU).

Multi-language: text DETECTION runs once per frame (script independent); every configured
RECOGNITION model then reads the detected boxes (batched) and the best-scoring reading wins.
"""
import logging
import os
import threading
import unicodedata

import numpy as np

from .log import log
from .textutil import strip_accents

# ---------------------------------------------------------------------------------------------
# Thread safety (several OCR objects run in parallel worker threads, see pipeline.OcrRunner):
#  * one OCR object is used by ONE thread at a time - RapidOCR mutates its own state on every call
#  * every ONNX session is created and warmed up in __init__ (RapidOCR would otherwise create the
#    rec session lazily inside a worker; on DirectML that crashed while another session was running)
#  * DirectML: Run() calls from several threads at once hung / crashed (Radeon 780M), so sessions
#    of a locked OCR share one process-wide lock around the GPU call only - CPU pre/post-processing
#    of the other workers still overlaps it. CUDA runs concurrent sessions natively.
# ---------------------------------------------------------------------------------------------
GPU_LOCK = threading.Lock()

# Recognition input "buckets": RapidOCR feeds the rec model (batch 1..6, 3, 48, W) with W = the widest crop
# of the batch - nearly every batch on text-heavy credits is a shape never seen before. DirectML prepares
# the model again for every new input shape; on an RTX 3070 credits took 1.6 s/frame vs 0.28 s on light
# frames. Padding to a few fixed shapes (zeros = RapidOCR's own padding value) means each is prepared once.
REC_WIDTH_STEP = 160


def _pad_rec_input(x, batch, step):
    b, c, h, w = x.shape
    wb = max(w, -(-w // step) * step)
    bb = max(b, batch)
    if (bb, wb) == (b, w):
        return x
    out = np.zeros((bb, c, h, wb), dtype=x.dtype)
    out[:b, :, :, :w] = x
    return out


def _install_session_lock():
    """Hook OrtInferSession: an optional per-session lock and optional rec input padding."""
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
    if getattr(OrtInferSession, "_videoid_lockable", False):
        return
    OrtInferSession._videoid_run = OrtInferSession.__call__     # the real GPU call (tests wrap it)

    def __call__(self, x):
        pad = getattr(self, "_videoid_rec_pad", None)
        b = w = None
        if pad is not None and getattr(x, "ndim", 0) == 4:
            b, w = x.shape[0], x.shape[3]
            x = _pad_rec_input(x, *pad)
        lock = getattr(self, "_videoid_lock", None)
        if lock is None:
            y = OrtInferSession._videoid_run(self, x)
        else:
            with lock:
                y = OrtInferSession._videoid_run(self, x)
        if b is not None and (x.shape[0], x.shape[3]) != (b, w):
            # back to what RapidOCR expects: its batch, and time steps for its width (T scales with W)
            t = int(round(y.shape[1] * w / x.shape[3]))
            y = y[:b, :t]
        return y

    OrtInferSession.__call__ = __call__
    OrtInferSession._videoid_lockable = True


def onnx_error_text(exc):
    """ONNX Runtime on non-English Windows: the native error text (e.g. a DirectML/HRESULT message) is in the
    ANSI code page, the Python binding decodes it as UTF-8 and fails, so only a UnicodeDecodeError is left.
    Its .object still holds the original bytes - decode them properly. Returns None if not that case."""
    seen, x = set(), exc
    while x is not None and id(x) not in seen:
        seen.add(id(x))
        if isinstance(x, UnicodeDecodeError) and isinstance(x.object, (bytes, bytearray)):
            raw = bytes(x.object)
            for enc in (("mbcs", "oem") if os.name == "nt" else ()) + ("latin-1",):
                try:
                    return " ".join(raw.decode(enc).split())
                except (UnicodeDecodeError, LookupError):
                    continue
        x = x.__cause__ or x.__context__
    return None


def ocr_workers(cfg, device_is_gpu=True):
    """ocr.workers: "auto" = 2 on a GPU, 1 on CPU (the CPU engine is multi-threaded already)."""
    w = cfg.get("workers", "auto")
    if w in (None, "", "auto"):
        return 2 if device_is_gpu else 1
    return max(1, int(w))


def _accents(s):
    return sum(1 for c in s if strip_accents(c) != c)


def _better(new, cur):
    """new/cur: (text, score, lang). Two models agreeing on the letters but differing only in accents:
    trust the one that CAN write accents (an ASCII-only model is no evidence against them)."""
    t, s, _ = new
    if not t:
        return False
    if not cur[0]:
        return True
    if strip_accents(t).casefold() == strip_accents(cur[0]).casefold():
        a, b = _accents(t), _accents(cur[0])
        if a != b:
            return a > b
    return s > cur[1] + 0.02

# user-facing names -> PP-OCRv5 recognition model
LANG_ALIASES = {
    "latin": "latin", "cs": "latin", "sk": "latin", "de": "latin", "fr": "latin", "es": "latin", "it": "latin",
    "pl": "latin", "hu": "latin", "pt": "latin", "nl": "latin", "tr": "latin", "vi": "latin", "id": "latin",
    "en": "en", "english": "en",
    "ch": "ch", "zh": "ch", "chinese": "ch", "ja": "ch", "japanese": "ch",      # ch model reads zh + ja + en
    "ko": "korean", "korean": "korean",
    "hi": "devanagari", "hindi": "devanagari", "mr": "devanagari", "marathi": "devanagari",
    "ne": "devanagari", "nepali": "devanagari", "devanagari": "devanagari",
    "ta": "ta", "tamil": "ta", "te": "te", "telugu": "te",
    "th": "th", "thai": "th",
    "ar": "arabic", "arabic": "arabic", "fa": "arabic", "ur": "arabic",
    "ru": "eslav", "uk": "eslav", "be": "eslav", "russian": "eslav", "ukrainian": "eslav", "eslav": "eslav",
    "bg": "cyrillic", "sr": "cyrillic", "mk": "cyrillic", "cyrillic": "cyrillic",
    "el": "el", "greek": "el",
}
MODELS = sorted(set(LANG_ALIASES.values()))
# Wikidata label languages worth fetching for each OCR model (native-script names of people/titles)
WIKIDATA_LANGS = {
    "ch": ["zh", "ja", "zh-hant", "zh-hans"], "korean": ["ko"], "devanagari": ["hi", "mr", "ne"],
    "ta": ["ta"], "te": ["te"], "th": ["th"], "arabic": ["ar", "fa", "ur"], "eslav": ["ru", "uk", "be"],
    "cyrillic": ["ru", "bg", "sr", "mk"], "el": ["el"],
}


def resolve_languages(langs):
    out = []
    for l in langs or ["latin"]:
        m = LANG_ALIASES.get(str(l).strip().lower())
        if not m:
            raise ValueError(f"unknown OCR language '{l}'. Known: {', '.join(sorted(LANG_ALIASES))}")
        if m not in out:
            out.append(m)
    return out


def wikidata_langs(langs):
    out = []
    for m in resolve_languages(langs):
        out += WIKIDATA_LANGS.get(m, [])
    return out


class OCR:
    def __init__(self, cfg):
        import onnxruntime as ort
        from rapidocr import RapidOCR

        self.langs = resolve_languages(cfg.get("languages") or ["latin"])
        providers = ort.get_available_providers()
        want_gpu = cfg.get("gpu", True)
        workers = ocr_workers(cfg, want_gpu)
        self.device = "CPU"
        base = {
            "Global.log_level": "error",
            "Global.use_cls": False,        # credits are never upside down
            # with several languages the primary model must not drop boxes it cannot read itself
            "Global.text_score": 0.0 if len(self.langs) > 1 else 0.5,
            # RapidOCR shrinks anything longer than 2000 px before detection - frames are sized by us
            "Global.max_side_len": 4096,
            # several workers x all cores (ONNX default) would oversubscribe the CPU
            **({"EngineConfig.onnxruntime.intra_op_num_threads": max(1, (os.cpu_count() or 2) // workers),
                "EngineConfig.onnxruntime.inter_op_num_threads": 1} if workers > 1 else {}),
            "Det.ocr_version": "PP-OCRv5",
            "Det.model_type": cfg.get("det_model", "server"),
            "Rec.ocr_version": "PP-OCRv5",
            "Rec.model_type": "mobile",
        }
        if want_gpu and "DmlExecutionProvider" in providers:
            base["EngineConfig.onnxruntime.use_dml"] = True
            self.device = "GPU (DirectML)"
        elif want_gpu and "CUDAExecutionProvider" in providers:
            base["EngineConfig.onnxruntime.use_cuda"] = True
            self.device = "GPU (CUDA)"
        for name in ("RapidOCR", "rapidocr"):
            logging.getLogger(name).setLevel(logging.ERROR)
        self.engines = [self._build(RapidOCR, dict(base, **{"Rec.lang_type": l})) for l in self.langs]
        self.is_gpu = self.device != "CPU"
        mode = str(cfg.get("gpu_lock", "auto")).lower()
        self.gpu_lock = self.is_gpu and (mode in ("true", "1", "yes") or (mode == "auto" and "DirectML" in self.device))
        if self.gpu_lock:
            _install_session_lock()
            for s in self._sessions():
                s._videoid_lock = GPU_LOCK
        bmode = str(cfg.get("rec_buckets", "auto")).lower()
        self.rec_buckets = bmode in ("true", "1", "yes") or (bmode == "auto" and "DirectML" in self.device)
        if self.rec_buckets:
            _install_session_lock()
            for eng in self.engines:
                rec = getattr(eng, "text_rec", None)
                if rec is not None and getattr(rec, "session", None) is not None:
                    rec.session._videoid_rec_pad = (int(getattr(rec, "rec_batch_num", 6) or 6), REC_WIDTH_STEP)
        self.min_conf = cfg.get("min_confidence", 0.6)
        self.device += f", languages: {', '.join(self.langs)}, detector: {base['Det.model_type']}"
        if "latin" not in self.langs and any(l in ("en", "ch") for l in self.langs):
            log("  note: no 'latin' OCR language - accented letters (č ř ů é ä ß ...) will be read without accents", 1)

    def _build(self, cls, params):
        from rapidocr import LangRec, ModelType, OCRVersion
        p = dict(params)
        p["Rec.lang_type"] = LangRec(p["Rec.lang_type"])
        for k in ("Rec.ocr_version", "Det.ocr_version"):
            p[k] = OCRVersion(p[k])
        for k in ("Det.model_type", "Rec.model_type"):
            p[k] = ModelType(p[k])
        try:
            return self._warm_up(cls(params=p))
        except Exception as e:  # GPU init can fail on exotic drivers - fall back to CPU
            if p.pop("EngineConfig.onnxruntime.use_dml", None) or p.pop("EngineConfig.onnxruntime.use_cuda", None):
                log(f"  GPU OCR init failed ({e}); using CPU", 0)
                self.device = "CPU"
                return self._warm_up(cls(params=p))
            raise

    @staticmethod
    def _warm_up(eng):
        """Create every ONNX session now, in the constructing thread (RapidOCR loads them lazily)."""
        if hasattr(eng, "_load_det_model") and hasattr(eng, "_load_rec_model"):
            eng._load_det_model()
            eng._load_rec_model()
        else:   # other RapidOCR versions: a frame with text loads detection + recognition
            import cv2
            img = np.zeros((96, 480, 3), np.uint8)
            cv2.putText(img, "Warm up 123", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            eng(img)
        if getattr(eng, "text_det", None) is None or getattr(eng, "text_rec", None) is None:
            raise RuntimeError("OCR warm-up did not create all models (unsupported RapidOCR version?)")
        return eng

    def _sessions(self):
        for eng in self.engines:
            for part in (getattr(eng, "text_det", None), getattr(eng, "text_rec", None),
                         getattr(eng, "text_cls", None)):
                s = getattr(part, "session", None)
                if s is not None:
                    yield s

    @staticmethod
    def _crop(img, pts):
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
        h, w = img.shape[:2]
        c = img[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
        return c if c.size else None

    def run(self, img):
        """img: BGR ndarray. Returns list of dict(text, conf, x, y, w, h, lang) in relative coords."""
        try:
            return self._run(img)
        except Exception as e:
            readable = onnx_error_text(e)
            if readable:
                raise RuntimeError(f"OCR failed on {self.device.split(',')[0]}: {readable}") from e
            raise

    def _run(self, img):
        H, W = img.shape[:2]
        if len(self.engines) == 1:
            res = self.engines[0](img)
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []
            boxes = [np.asarray(b, dtype=float) for b in boxes]
            best = [(t or "", float(s), self.langs[0]) for t, s in zip(res.txts, res.scores)]
        else:
            # detect once (script independent), then every recognition model reads all boxes;
            # RapidOCR's own pipeline would drop boxes the first model cannot read
            det = self.engines[0](img, use_det=True, use_cls=False, use_rec=False)
            boxes = getattr(det, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []
            boxes = [np.asarray(b, dtype=float) for b in boxes]
            crops = [self._crop(img, b) for b in boxes]
            idx = [i for i, c in enumerate(crops) if c is not None]
            best = [("", 0.0, "")] * len(boxes)
            for lang, eng in zip(self.langs, self.engines):
                try:
                    rr = eng.recognize_txt([crops[i] for i in idx])
                except Exception as e:
                    log(f"  {lang} recognition failed: {onnx_error_text(e) or e}", 2)
                    continue
                for i, t, s in zip(idx, rr.txts, rr.scores):
                    cand = (t or "", float(s), lang)
                    if _better(cand, best[i]):
                        # agreeing readings back each other up - keep the higher confidence
                        same = best[i][0] and strip_accents(cand[0]).casefold() == strip_accents(best[i][0]).casefold()
                        best[i] = (cand[0], max(cand[1], best[i][1]) if same else cand[1], lang)
        out = []
        for pts, (txt, sc, lang) in zip(boxes, best):
            txt = unicodedata.normalize("NFC", txt or "")   # one code point per accented letter
            if sc < self.min_conf or not txt.strip():
                continue
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            out.append({"text": txt.strip(), "conf": sc, "x": float(x0 / W), "y": float(y0 / H),
                        "w": float((x1 - x0) / W), "h": float((y1 - y0) / H), "lang": lang})
        return out

"""GPU OCR: PaddleOCR PP-OCRv5 models via RapidOCR on ONNX Runtime + DirectML (any DX12 GPU).

Multi-language: text DETECTION runs once per frame (script independent); every configured
RECOGNITION model then reads the detected boxes (batched) and the best-scoring reading wins.
"""
import logging
import unicodedata

import numpy as np

from .log import log
from .textutil import strip_accents


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
        self.device = "CPU"
        base = {
            "Global.log_level": "error",
            "Global.use_cls": False,        # credits are never upside down
            # with several languages the primary model must not drop boxes it cannot read itself
            "Global.text_score": 0.0 if len(self.langs) > 1 else 0.5,
            # RapidOCR shrinks anything longer than 2000 px before detection - frames are sized by us
            "Global.max_side_len": 4096,
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
            return cls(params=p)
        except Exception as e:  # GPU init can fail on exotic drivers - fall back to CPU
            if p.pop("EngineConfig.onnxruntime.use_dml", None) or p.pop("EngineConfig.onnxruntime.use_cuda", None):
                log(f"  GPU OCR init failed ({e}); using CPU", 0)
                self.device = "CPU"
                return cls(params=p)
            raise

    @staticmethod
    def _crop(img, pts):
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
        h, w = img.shape[:2]
        c = img[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
        return c if c.size else None

    def run(self, img):
        """img: BGR ndarray. Returns list of dict(text, conf, x, y, w, h, lang) in relative coords."""
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
                    log(f"  {lang} recognition failed: {e}", 2)
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

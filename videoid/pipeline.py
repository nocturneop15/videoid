"""Per-file pipeline: local evidence -> OCR of titles/credits -> people -> database matching -> store."""
import json
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import localinfo, media
from .credits import CreditCollector, llm_extract
from .db import now
from .film import Film
from .log import log


def iter_videos(paths, cfg, recursive=True):
    exts = {"." + e.lower() for e in cfg["scan"]["extensions"]}
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p.resolve()
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in exts:
                    yield f.resolve()
        else:
            log(f"not found: {p}", 0)


class Context:
    def __init__(self, cfg, db, http, matcher, ocr_factory, llm):
        self.cfg, self.db, self.http, self.matcher = cfg, db, http, matcher
        self._ocr_factory, self._engines = ocr_factory, None
        self.llm = llm

    def ocr_engines(self):
        """ocr.workers independent OCR objects (each with its own ONNX sessions). Built one after another
        in this thread - creating sessions concurrently is not safe on DirectML."""
        if self._engines is None:
            from .ocr import ocr_workers
            first = self._ocr_factory()
            n = ocr_workers(self.cfg["ocr"], getattr(first, "is_gpu", False))
            self._engines = [first] + [self._ocr_factory() for _ in range(n - 1)]
            lock = " (GPU calls serialized)" if n > 1 and getattr(first, "gpu_lock", False) else ""
            log(f"OCR engine ready on {first.device}, {n} worker(s){lock}", 1)
        return self._engines

    @property
    def ocr(self):
        return self.ocr_engines()[0]


class OcrRunner:
    """Runs OCR on several engines in parallel threads. Each engine is checked out by exactly one thread at a
    time; futures are returned in submission order. With one engine everything runs inline (no threads)."""

    def __init__(self, engines):
        self.engines = list(engines)
        self.parallel = len(self.engines) > 1
        self._pool = ThreadPoolExecutor(len(self.engines), thread_name_prefix="ocr") if self.parallel else None
        self._free = queue.SimpleQueue()
        for e in self.engines:
            self._free.put(e)

    def _job(self, img):
        eng = self._free.get()
        try:
            return eng.run(img)
        finally:
            self._free.put(eng)

    def submit(self, img):
        if self._pool:
            return self._pool.submit(self._job, img)
        f = Future()
        try:
            f.set_result(self.engines[0].run(img))
        except Exception as e:
            f.set_exception(e)
        return f

    @property
    def max_in_flight(self):
        return 2 * len(self.engines)      # bounds memory: at most this many frames wait for OCR

    def close(self, cancel=False):
        """Wait for running jobs; with cancel, drop the queued ones (error / Ctrl+C)."""
        if self._pool:
            self._pool.shutdown(wait=True, cancel_futures=cancel)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(cancel=exc_type is not None)


def _prefetch(gen, size=4):
    """Decode frames in a background thread (bounded queue), so ffmpeg works while the GPU reads.
    Closing the returned generator stops the thread and the ffmpeg process."""
    q, stop, end = queue.Queue(maxsize=size), threading.Event(), object()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def pump():
        try:
            for item in gen:
                if not put(item):
                    break
        except BaseException as e:          # handed to the consumer
            put(e)
        finally:
            gen.close()                     # stops ffmpeg (media.frames' finally)
            put(end)

    th = threading.Thread(target=pump, daemon=True, name="decode")
    th.start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        th.join()


def _segments(duration, sc):
    # ffmpeg reads "-t 0" as "no limit", so a disabled (0 minute) segment must not be passed on
    return [s for s in _segments_raw(duration, sc) if s[2] > 0]


def _segments_raw(duration, sc):
    o, e = sc["open_minutes"] * 60, sc["end_minutes"] * 60
    if not duration:
        return [("open", 0, o, sc["fps_open"])]
    if duration <= o + e:
        return [("open", 0, duration / 2, sc["fps_open"]), ("end", duration / 2, duration / 2, sc["fps_end"])]
    return [("open", 0, o, sc["fps_open"]), ("end", max(0, duration - e), e, sc["fps_end"])]


def run_ocr(ctx, path, info):
    """OCR the opening/end segments. Returns [(segment, t, boxes)] for frames that contain text."""
    sc = ctx.cfg["scan"]
    t0 = time.time()
    stats = {"frames": 0, "ocr_frames": 0}
    out = []
    with OcrRunner(ctx.ocr_engines()) as runner:
        for seg, start, length, fps in _segments(info["duration"], sc):
            frames = media.frames(path, start, length, fps, info, sc)
            if runner.parallel:
                frames = _prefetch(frames)
            ocr_frames(runner, seg, frames, int(length * fps), stats, out)
            log("", 1)
    stats["seconds"] = round(time.time() - t0, 1)
    for k in ("decode_wait", "ocr_wait"):
        stats[k] = round(stats.get(k, 0.0), 1)
    log(f"  OCR: {stats['ocr_frames']} frames in {stats['seconds']}s, {len(out)} with text "
        f"(waited {stats['decode_wait']}s for decoding, {stats['ocr_wait']}s for OCR)", 1)
    return out, stats


def ocr_frames(runner, seg, frames, total, stats, out):
    """OCR one segment's (t, img) frames, appending [seg, t, boxes, weight] to out.

    Frames are submitted to the runner in order and their results applied strictly in that order, so the
    output is identical to sequential OCR: a static card that stays on screen adds weight to the last OCR'd
    frame only if that frame turned out to have text."""
    last = None
    pending = deque()          # in-flight frames, oldest first: [t, future, repeats, out entry, done]
    last_ev = None

    def resolve(ev):
        boxes = ev[1].result()             # re-raises a worker's exception here, in the main thread
        ev[4] = True
        if boxes:
            ev[3] = [seg, round(ev[0], 2), boxes, 1 + ev[2]]
            out.append(ev[3])

    stats.setdefault("decode_wait", 0.0)
    stats.setdefault("ocr_wait", 0.0)
    clock = time.perf_counter
    it = iter(frames)
    try:
        while True:
            t0 = clock()
            try:
                t, img = next(it)
            except StopIteration:
                break
            finally:
                stats["decode_wait"] += clock() - t0          # time spent waiting for ffmpeg
            stats["frames"] += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.max() < 60:                       # black frame
                continue
            thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
            if last is not None and np.abs(thumb - last).mean() < 1.5:   # static card already read
                if last_ev is not None:   # ...but it stays on screen longer = more evidence (if it has text)
                    if not last_ev[4]:
                        last_ev[2] += 1
                    elif last_ev[3] is not None:
                        last_ev[3][3] += 1
                continue
            last = thumb
            t0 = clock()
            last_ev = [t, runner.submit(img), 0, None, False]     # inline OCR when there is one worker
            pending.append(last_ev)
            stats["ocr_frames"] += 1
            while len(pending) > runner.max_in_flight or (pending and pending[0][1].done()):
                resolve(pending.popleft())
            stats["ocr_wait"] += clock() - t0                     # time the main thread waited for OCR
            if stats["frames"] % 10 == 0:
                log(f"\r  [{seg}] frame {stats['frames']}/{total}  ocr'd {stats['ocr_frames']}  "
                    f"with text {len(out)}   ", 1, end="")
        t0 = clock()
        while pending:
            resolve(pending.popleft())
        stats["ocr_wait"] += clock() - t0
    finally:
        frames.close()                     # stop decoding / ffmpeg on errors and Ctrl+C too


def parse_frames(ctx, frames, stats=None):
    """Credits parsing over stored/fresh OCR frames -> unique lines, people, meta (+ optional LLM)."""
    col = CreditCollector()
    for fr in frames:
        seg, t, boxes = fr[:3]
        col.feed(seg, t, boxes, fr[3] if len(fr) > 3 else 1)
    lines = col.unique_lines()
    people = col.result_people()
    meta = col.meta()
    if stats:
        meta["stats"] = stats
    log(f"  credits: {len(lines)} unique strings -> {len(people)} people", 1)
    if ctx.llm and lines:
        log(f"  LLM ({ctx.llm['model']}) reading {len(lines)} unique strings ...", 1)
        lp, lt, ly = llm_extract(ctx.http, ctx.llm, lines)
        people += lp
        if lt:
            meta["title_candidates"].insert(0, lt)
        if ly and isinstance(ly, int):
            meta["copyright_years"].insert(0, ly)
        meta["llm"] = ctx.llm["model"]
        log(f"  LLM: {len(lp)} people", 1)
    return lines, people, meta


def build_evidence(local, info, people, meta):
    titles = list(local["titles"])
    for t in meta.get("title_candidates", [])[:3]:
        titles.append((t, "ocr"))
    years = list(local["years"]) + [(y, "ocr") for y in meta.get("copyright_years", [])]
    # merge people: same name may come from OCR, LLM and NFO
    merged = {}
    for p in list(people) + list(local.get("people", [])):
        k = (p["name"].lower(), p["role"])
        if k not in merged:
            merged[k] = dict(p)
    return {
        "titles": titles, "years": years, "ids": local.get("ids", {}), "people": list(merged.values()),
        "runtime": (info["duration"] / 60) if info.get("duration") else None,
    }


def process_file(ctx, path, opts):
    db, cfg = ctx.db, ctx.cfg
    st = path.stat()
    if st.st_size < cfg["scan"]["min_size_mb"] * 1024 * 1024 and not opts.all_sizes:
        log(f"skip (small): {path}", 2)
        return None
    if "sample" in path.stem.lower() and not opts.all_sizes:
        log(f"skip (sample): {path}", 2)
        return None
    h = localinfo.oshash(path)
    row = db.file_lookup(str(path), st.st_size, h)
    unchanged = row is not None and row["size"] == st.st_size
    if unchanged and row["matched_at"] and not (opts.force or opts.rematch or opts.reparse):
        log(f"\n== {path}  (cached, use --rematch / --reparse / --force to redo)", 0)
        return row["id"]

    log(f"\n== {path}", 0)
    t_start = time.time()
    info = media.probe(path)
    local = localinfo.gather(path, info)
    if opts.no_nfo:      # testing: identify from name + credits only
        local.update(ids={}, people=[], nfo_files=[])
        local["titles"] = [t for t in local["titles"] if t[1] != "nfo"]
        local["years"] = [y for y in local["years"] if y[1] != "nfo"]
    log(f"  local: titles={[t for t, _ in local['titles']]} years={[y for y, _ in local['years']]} "
        f"ids={local['ids']}" + (f" nfo={len(local['nfo_files'])}" if local["nfo_files"] else ""), 1)
    fid = db.file_upsert(str(path), size=st.st_size, mtime=st.st_mtime, oshash=h, duration=info["duration"],
                         width=info["width"], height=info["height"], local_json=json.dumps(local, ensure_ascii=False))

    people, meta = [], {}
    if opts.ocr:
        stored = db.frames(fid) if unchanged and row["ocr_done"] and not opts.force else []
        if stored and opts.reparse:
            log(f"  re-parsing {len(stored)} stored OCR frames", 1)
            old = json.loads(row["ocr_meta_json"] or "{}")
            lines, people, meta = parse_frames(ctx, stored, old.get("stats"))
            db.save_ocr(fid, lines, people, meta)
        elif unchanged and row["ocr_done"] and not opts.force and not opts.reparse:
            people = [p for p in db.people(fid) if p["source"] in ("ocr", "llm")]
            meta = json.loads(row["ocr_meta_json"] or "{}")
            log(f"  reusing stored OCR: {len(people)} people", 1)
        else:
            frames, stats = run_ocr(ctx, path, info)
            db.save_frames(fid, frames)
            lines, people, meta = parse_frames(ctx, frames, stats)
            db.save_ocr(fid, lines, people, meta)

    ev = build_evidence(local, info, people, meta)
    ranked = ctx.matcher.identify(ev)
    store = [{"film_key": r["film_key"], "score": r["score"], "confidence": r.get("confidence", 0),
              "verdict": r.get("verdict", "POSSIBLE"), "evidence": r["evidence"]} for r in ranked[:5]]
    db.save_matches(fid, store)
    if ranked:
        b = ranked[0]
        f = b["film"]
        db.file_upsert(str(path), matched_at=now(), film_key=b["film_key"], title=f.title, year=f.year,
                       verdict=b["verdict"], confidence=b["confidence"], ids_json=json.dumps(f.ids))
    else:
        db.file_upsert(str(path), matched_at=now(), film_key=None, title=None, year=None, verdict="UNKNOWN",
                       confidence=0, ids_json="{}")
    ocr_s = (meta.get("stats") or {}).get("seconds") if opts.ocr else None
    log(f"  total time: {fmt_duration(time.time() - t_start)}"
        + (f" (OCR {fmt_duration(ocr_s)})" if ocr_s else ""), 0)
    return fid


def fmt_duration(s):
    s = int(round(s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def describe(db, fid, top=3, show_people=True):
    row = db.file_get(fid)
    out = [f"\n>> {row['path']}"]
    ms = db.matches(fid)
    if not ms:
        out.append("   no match found")
        return "\n".join(out)
    for m in ms[:top]:
        fj = db.film_json(m["film_key"]) or {}
        f = Film.from_dict(fj)
        ev = json.loads(m["evidence_json"])
        star = "*" if m["rank"] == 0 else " "
        out.append(f" {star} [{m['verdict']:<8} {m['confidence'] * 100:5.1f}%  score {m['score']:6.1f}]  "
                   f"{f.title} ({f.year or '?'})" + (f"  {f.runtime} min" if f.runtime else ""))
        if m["rank"] == 0:
            orig = [t for t in f.titles if t != f.title][:3]
            if orig:
                out.append(f"      also known as: {' / '.join(orig)}")
            for k, u in f.urls().items():
                out.append(f"      {k:<9} {u}")
            if show_people and ev["people"]:
                out.append(f"      matched {len(ev['people'])} people:")
                for p in ev["people"][:25]:
                    out.append(f"        {p['role']:<20} {p['db']}" + (f"   (read: {p['ocr']})" if p["ocr"] != p["db"] else ""))
                if len(ev["people"]) > 25:
                    out.append(f"        ... +{len(ev['people']) - 25} more")
            bits = []
            if ev.get("ids"):
                bits.append("NFO id " + ",".join(ev["ids"]))
            if ev.get("title") and ev["title"].get("pair"):
                bits.append(f"title {ev['title']['sim']}% via {ev['title']['pair'][2]}")
            if ev.get("year"):
                bits.append(f"year diff {ev['year']['diff']}")
            if ev.get("runtime"):
                bits.append(f"runtime diff {ev['runtime']['diff']} min")
            if bits:
                out.append("      evidence: " + "; ".join(bits))
            for w in ev.get("warnings") or []:
                out.append(f"      WARNING: {w}")
    return "\n".join(out)

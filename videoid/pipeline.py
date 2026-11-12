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

from . import localinfo, media, subtitles
from .credits import CreditCollector, llm_extract
from .db import now
from .film import Film
from .textutil import title_sim
from . import ui
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
        self.cached = False         # set by process_file: the last file was skipped as already analysed

    def ocr_engines(self):
        """ocr.workers independent OCR objects (each with its own ONNX sessions). Built one after another
        in this thread - creating sessions concurrently is not safe on DirectML."""
        if self._engines is None:
            from .ocr import ocr_workers
            log(ui.step("Starting the text recognition (OCR) engine - loads the models once per run ..."), 1)
            first = self._ocr_factory()
            n = ocr_workers(self.cfg["ocr"], getattr(first, "is_gpu", False))
            share = getattr(first, "shares_sessions", False)
            # serialized GPU calls: workers share one set of sessions (one copy of the models in GPU memory)
            self._engines = [first] + [first.clone() if share else self._ocr_factory() for _ in range(n - 1)]
            how = (", shared GPU sessions, GPU calls serialized" if share else
                   ", GPU calls serialized" if getattr(first, "gpu_lock", False) else "") if n > 1 else ""
            log(ui.detail(f"OCR engine ready: {first.device}, {n} worker(s){how}"), 1)
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


def run_ocr(ctx, path, info, segments=None):
    """OCR the opening/end segments (or the given [(segment, start, length, fps)]).
    Returns [(segment, t, boxes)] for frames that contain text."""
    sc = ctx.cfg["scan"]
    t0 = time.time()
    stats = {"frames": 0, "ocr_frames": 0}
    out = []
    with OcrRunner(ctx.ocr_engines()) as runner:
        for seg, start, length, fps in segments or _segments(info["duration"], sc):
            what = "opening titles" if seg == "open" else "end credits"
            rate = f"1 frame/s" if fps >= 0.999 else f"1 frame every {1 / fps:g} s"
            if fps > 1.001:
                rate = f"{fps:g} frames/s"
            log(ui.step(f"Reading the {what} with OCR: {_clock(start)}-{_clock(start + length)} of the film, "
                        f"{rate}"), 1)
            frames = media.frames(path, start, length, fps, info, sc)
            if runner.parallel:
                frames = _prefetch(frames)
            ocr_frames(runner, seg, frames, int(length * fps), stats, out)
    stats["seconds"] = round(time.time() - t0, 1)
    for k in ("decode_wait", "ocr_wait"):
        stats[k] = round(stats.get(k, 0.0), 1)
    log(ui.detail(f"text found on {len(out)} of {stats['ocr_frames']} frames read, "
                  f"in {fmt_duration(stats['seconds'])}"), 1)
    log(ui.detail(f"(skipped {stats['frames'] - stats['ocr_frames']} black or unchanged frames; waited "
                  f"{stats['decode_wait']}s for decoding, {stats['ocr_wait']}s for OCR)"), 2)
    return out, stats


def _clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


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
    seg_start, frames0, text0 = clock(), stats["frames"], len(out)

    def progress(done=False):
        n = stats["frames"] - frames0
        bar_w = 24
        fill = min(bar_w, int(bar_w * n / max(1, total)))
        bar = ui.c("#" * fill, ui.CYAN) + ui.c("." * (bar_w - fill), ui.DIM)
        tail = "done" if done else ""
        if not done and n:
            left = (clock() - seg_start) / n * max(0, total - n)
            tail = f"about {fmt_duration(left)} left"
        log(f"\r    [{bar}] {min(n, total)}/{total} frames, text on {len(out) - text0}  {tail}     ", 1,
            end="\n" if done else "")
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
                progress()
        t0 = clock()
        while pending:
            resolve(pending.popleft())
        stats["ocr_wait"] += clock() - t0
        progress(done=True)
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
    log(ui.step(f"Finding names and roles in the text: {len(lines)} different lines -> {len(people)} people"), 1)
    if ctx.llm and lines:
        log(ui.step(f"Local AI ({ctx.llm['model']}) reading the {len(lines)} lines for more names ..."), 1)
        lp, lt, ly = llm_extract(ctx.http, ctx.llm, lines)
        people += lp
        if lt:
            meta["title_candidates"].insert(0, lt)
        if ly and isinstance(ly, int):
            meta["copyright_years"].insert(0, ly)
        meta["llm"] = ctx.llm["model"]
        log(ui.detail(f"the local AI found {len(lp)} people"), 1)
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
    index, total = getattr(ctx, "position", (None, None))
    log(ui.header(path.name, index, total), 0)
    log(ui.detail(f"in {path.parent}"), 1)
    if unchanged and row["matched_at"] and not (opts.force or opts.rematch or opts.reparse):
        log(ui.detail("already identified - showing the stored result (--rematch / --reparse / --force redo it)"), 0)
        ctx.cached = True
        return row["id"]

    t_start = time.time()
    log(ui.step("Reading the file name, folder name, NFO and the file's own metadata"), 1)
    info = media.probe(path)
    local = localinfo.gather(path, info)
    if opts.no_nfo:      # testing: identify from name + credits only
        local.update(ids={}, people=[], nfo_files=[])
        local["titles"] = [t for t in local["titles"] if t[1] != "nfo"]
        local["years"] = [y for y in local["years"] if y[1] != "nfo"]
    got = [f"titles {', '.join(repr(t) for t, _ in local['titles']) or '-'}",
           f"years {', '.join(str(y) for y, _ in local['years']) or '-'}"]
    if info.get("duration"):
        got.append(f"length {fmt_duration(info['duration'])}")
    if local["nfo_files"]:
        got.append(f"NFO with {', '.join(f'{k} {v}' for k, v in local['ids'].items()) or 'no ids'}")
    log(ui.detail("found: " + "; ".join(got)), 1)
    fid = db.file_upsert(str(path), size=st.st_size, mtime=st.st_mtime, oshash=h, duration=info["duration"],
                         width=info["width"], height=info["height"], local_json=json.dumps(local, ensure_ascii=False))

    people, meta, ranked = [], {}, None
    if opts.ocr:
        stored = db.frames(fid) if unchanged and row["ocr_done"] and not opts.force else []
        if stored and opts.reparse:
            log(ui.step(f"Reading names again from the {len(stored)} frames of text stored by an earlier scan"), 1)
            old = json.loads(row["ocr_meta_json"] or "{}")
            lines, people, meta = parse_frames(ctx, stored, old.get("stats"))
            db.save_ocr(fid, lines, people, meta)
        elif unchanged and row["ocr_done"] and not opts.force and not opts.reparse:
            people = [p for p in db.people(fid) if p["source"] in ("ocr", "llm")]
            meta = json.loads(row["ocr_meta_json"] or "{}")
            log(ui.step(f"Reusing the credits read in an earlier scan: {len(people)} people"), 1)
        else:
            ranked = subtitle_fast_track(ctx, path, info, local, opts)      # cheapest first: no OCR at all
            if ranked is None:
                ranked = quick_nfo_check(ctx, path, info, local, fid, opts)
                if ranked is None:
                    frames, stats = run_ocr(ctx, path, info)
                    db.save_frames(fid, frames)
                    lines, people, meta = parse_frames(ctx, frames, stats)
                    db.save_ocr(fid, lines, people, meta)
                else:
                    meta = json.loads(db.file_get(fid)["ocr_meta_json"] or "{}")

    if ranked is None:
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
    log(ui.detail(f"finished in {fmt_duration(time.time() - t_start)}"
                  + (f" (reading text {fmt_duration(ocr_s)})" if ocr_s else "")), 0)
    return fid


def subtitle_fast_track(ctx, path, info, local, opts):
    """Text subtitles: the film's character names in the dialogue confirm a candidate found by its title, so
    no OCR is needed. Returns the ranked matches when that is conclusive, else None (-> NFO check / OCR)."""
    sc = ctx.cfg["scan"]
    need = int(sc.get("subtitle_characters", 3) or 0)
    if need <= 0 or not getattr(opts, "subs", True):
        return None
    if not subtitles.text_tracks(info) and not subtitles.sidecars(path):
        if info.get("subtitles"):
            log(ui.detail("the subtitles are pictures (PGS / VobSub) - they would need OCR, skipped"), 1)
        return None
    log(ui.step("The film has text subtitles - looking for its characters' names in the dialogue first "
                "(much faster than reading the credits)"), 0)
    t0 = time.time()
    lines, sources = subtitles.read(path, info)
    words = subtitles.name_words(lines)
    log(ui.detail(f"{len(lines)} subtitle lines from {', '.join(sources) or 'nothing readable'}, "
                  f"{len(words)} different capitalised words, in {fmt_duration(time.time() - t0)}"), 1)
    if len(lines) < 50:
        log(ui.warn("too little subtitle text to go on - reading the credits instead"), 0)
        return None
    ev = build_evidence(local, info, [], {})
    ranked = ctx.matcher.identify(ev)
    if not ranked:
        log(ui.warn("no candidate film from the file name - reading the credits instead"), 0)
        return None
    log(ui.step(f"Comparing the dialogue with the character names of {min(6, len(ranked))} candidate films"), 1)
    for r in ranked[:6]:
        r["characters"] = subtitles.character_hits(ctx.matcher.characters(r["film"]), words)
        r["score"] += min(10, 2 * len(r["characters"]))
    ranked[:6] = sorted(ranked[:6], key=lambda r: -r["score"])
    top = ranked[0]
    e, f, hits = top["evidence"], top["film"], top.get("characters", [])
    name = f"{f.title} ({f.year or '?'})"
    titled = bool(e.get("ids")) or (e.get("title") or {}).get("sim", 0) >= 92
    rivals = [r["score"] for r in ranked[1:6] if len(r.get("characters", [])) >= need]
    margin = top["score"] - max(rivals) if rivals else 99
    # a misnamed file (the sequel's name on the original) would have the same characters - the length must fit
    length_ok = not (e.get("runtime") and e["runtime"]["diff"] > 10)
    if titled and len(hits) >= need and margin >= 6 and length_ok and not e.get("warnings"):
        e["subtitles"] = {"characters": hits[:12], "sources": sources}
        top["verdict"], top["confidence"] = "CERTAIN", max(top.get("confidence", 0), 0.99)
        log(ui.good(f"Subtitles confirm {name}: {len(hits)} of its characters are named in the dialogue "
                    f"({', '.join(hits[:6])}{', ...' if len(hits) > 6 else ''}) - no OCR needed"), 0)
        return ranked
    if not titled:
        why = f"the file name doesn't name {name} clearly"
    elif len(hits) < need:
        why = f"only {len(hits)} of {name}'s known characters named in the dialogue"
    elif not length_ok:
        why = f"{name} is {e['runtime']['film']} min long, the file {e['runtime']['file']} min"
    else:
        why = "another film with the same characters scores almost as high (sequel or remake?)"
    log(ui.warn(f"Subtitles not conclusive ({why}) - reading the credits instead"), 0)
    return None


def quick_nfo_check(ctx, path, info, local, fid, opts):
    """A foreign NFO next to the movie names the film: confirm it cheaply from the end credits only, sampled
    every few seconds. The film is identified from the credits ALONE (no NFO, no file name - films sharing actors
    must not confirm each other) and the NFO is accepted only when that lands on the NFO's film.
    Returns the ranked matches (full evidence) when verified, else None (-> normal full scan)."""
    sc = ctx.cfg["scan"]
    every = float(sc.get("nfo_check_every", 3) or 0)
    nfo_titles = [t for t, src in local.get("titles", []) if src == "nfo"]
    if (not local.get("nfo_files") or not (local.get("ids") or nfo_titles) or every <= 0
            or not getattr(opts, "quick_nfo", True) or not info.get("duration")):
        return None
    end = [s for s in _segments(info["duration"], sc) if s[0] == "end"]
    if not end:
        return None
    seg, start, length, _ = end[0]
    log(ui.step(f"An NFO names the film - checking it quickly against the end credits "
                f"(one frame every {every:g} s) instead of the full scan"), 0)
    frames, stats = run_ocr(ctx, path, info, [(seg, start, length, 1.0 / every)])
    lines, people, meta = parse_frames(ctx, frames, stats)
    need = int(sc.get("nfo_check_people", 3))
    why = f"only {len(people)} people read from the credits"
    if len(people) >= need:
        log(ui.step("Identifying the film from the credits alone, ignoring the NFO and the file name"), 1)
        blank = {"titles": [], "years": [], "ids": {}, "people": []}
        alone = ctx.matcher.identify(build_evidence(blank, info, people, meta))
        if not alone:
            why = "the credits alone match no film"
        else:
            b = alone[0]
            f, n = b["film"], len(b["evidence"]["people"])
            name = f"{f.title} ({f.year or '?'})"
            ids = local.get("ids") or {}
            same = (any(f.ids.get(k) == str(v) for k, v in ids.items()) if ids else
                    max((title_sim(t, ft) for t in nfo_titles for ft in f.titles), default=0) >= 90)
            if b["verdict"] not in ("CERTAIN", "PROBABLE") or n < need:
                why = f"credits alone not conclusive: {name}, {b['verdict']}, {n} people"
            elif not same:
                why = f"the credits are {name} ({n} people), not the NFO's film"
            else:
                ranked = ctx.matcher.identify(build_evidence(local, info, people, meta))
                if ranked and ranked[0]["film_key"] == b["film_key"]:
                    log(ui.good(f"NFO verified: the end credits alone identify {name} "
                                f"({n} people, {b['verdict']})"), 0)
                    meta["nfo_verified"] = True
                    ctx.db.save_frames(fid, frames)
                    ctx.db.save_ocr(fid, lines, people, meta)
                    return ranked
                why = "NFO and credits together point elsewhere"
    log(ui.warn(f"NFO NOT verified ({why}) - doing the full scan"), 0)
    return None


def fmt_duration(s):
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


def describe(db, fid, top=3, show_people=True):
    row = db.file_get(fid)
    ms = db.matches(fid)
    out = ["", ui.result_badge(row["verdict"] if ms else "UNKNOWN") + " " + ui.c(row["path"], ui.BOLD)]
    if not ms:
        out.append(ui.c("   no matching film found", ui.RED))
        return "\n".join(out)
    if row["verdict"] == "UNKNOWN":
        out.append(ui.c("   you marked these candidates as wrong (--interactive)", ui.RED))
    for m in ms[:top]:
        fj = db.film_json(m["film_key"]) or {}
        f = Film.from_dict(fj)
        ev = json.loads(m["evidence_json"])
        first = m["rank"] == 0
        name = f"{f.title} ({f.year or '?'})"
        out.append((ui.c(" * ", ui.BOLD) if first else "   ")
                   + f"[{ui.verdict(m['verdict'], 8)} {m['confidence'] * 100:5.1f}%  score {m['score']:6.1f}]  "
                   + (ui.c(name, ui.BOLD + ui.WHITE) if first else ui.c(name, ui.DIM))
                   + (f"  {f.runtime} min" if f.runtime else ""))
        if m["rank"] == 0:
            orig = [t for t in f.titles if t != f.title][:3]
            if orig:
                out.append(f"      also known as: {' / '.join(orig)}")
            for k, u in f.urls().items():
                out.append(f"      {k:<9} " + ui.c(u, ui.BLUE))
            if show_people and ev["people"]:
                out.append(f"      matched {len(ev['people'])} people:")
                for p in ev["people"][:25]:
                    out.append(f"        {p['role']:<20} {p['db']}" + (f"   (read: {p['ocr']})" if p["ocr"] != p["db"] else ""))
                if len(ev["people"]) > 25:
                    out.append(f"        ... +{len(ev['people']) - 25} more")
            bits = ["chosen by you (--interactive)"] if ev.get("chosen_by_user") else []
            if ev.get("subtitles"):
                ch = ev["subtitles"]["characters"]
                bits.append(f"{len(ch)} characters named in the subtitles ({', '.join(ch[:4])}"
                            f"{', ...' if len(ch) > 4 else ''})")
            if ev.get("ids"):
                bits.append("NFO id " + ",".join(ev["ids"]))
            if ev.get("title") and ev["title"].get("pair"):
                bits.append(f"title {ev['title']['sim']}% via {ev['title']['pair'][2]}")
            if ev.get("name_year"):
                bits.append("title + year match the " + {"filename": "file name", "folder": "folder name", "nfo": "NFO",
                                                        "embedded": "file's metadata", "user": "name you typed"}
                            .get(ev["name_year"], ev["name_year"]))
            elif ev.get("year"):
                bits.append(f"year diff {ev['year']['diff']}")
            if ev.get("runtime"):
                bits.append(f"runtime diff {ev['runtime']['diff']} min")
            if bits:
                out.append(ui.c("      evidence: " + "; ".join(bits), ui.DIM))
            for w in ev.get("warnings") or []:
                out.append("      " + ui.badge("WARNING", ui.ON_YELLOW) + " " + ui.c(w, ui.YELLOW))
    return "\n".join(out)

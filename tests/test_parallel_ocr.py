"""Tests for parallel OCR (pipeline.OcrRunner / ocr_frames / _prefetch, ocr.GPU_LOCK).

    runtime\\python\\python.exe tests\\test_parallel_ocr.py                       # logic tests, no GPU (seconds)
    runtime\\python\\python.exe tests\\test_parallel_ocr.py --video FILM.mkv      # + real OCR on the end credits
    runtime\\python\\python.exe tests\\test_parallel_ocr.py --video FILM.mkv --workers 1,2,3,4 --frames 30

The real-OCR part checks that every worker count gives exactly the sequential result and prints the speed of
each, so it doubles as the benchmark for choosing ocr.workers / ocr.gpu_lock on a given GPU.
"""
import argparse
import copy
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from videoid import config, media, log as logmod  # noqa: E402
from videoid import pipeline  # noqa: E402
from videoid.pipeline import OcrRunner, _prefetch, ocr_frames  # noqa: E402

logmod.VERBOSITY = 0
FAILED = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def reference(engine, seg, frames, out):
    """The original sequential loop (before parallel OCR) - the behaviour to preserve."""
    last, last_had_text = None, False
    for t, img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.max() < 60:
            continue
        thumb = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)
        if last is not None and np.abs(thumb - last).mean() < 1.5:
            if last_had_text:
                out[-1][3] += 1
            continue
        last = thumb
        boxes = engine.run(img)
        last_had_text = bool(boxes)
        if boxes:
            out.append([seg, round(t, 2), boxes, 1])


def gen(items, closed=None):
    try:
        yield from items
    finally:
        if closed is not None:
            closed.append(True)


# --------------------------------------------------------------------------------------------- fakes
class FakeEngine:
    """Deterministic 'OCR': text depends only on the frame; random latency scrambles completion order.
    Detects being used by two threads at once."""
    in_use_violations = 0

    def __init__(self, fail_on=None, max_sleep=0.004):
        self.busy = threading.Lock()
        self.fail_on, self.max_sleep, self.calls = fail_on, max_sleep, 0

    def run(self, img):
        if not self.busy.acquire(blocking=False):
            FakeEngine.in_use_violations += 1
            self.busy.acquire()
        try:
            self.calls += 1
            h = int(img[::7, ::11].sum()) * 2654435761 % 1000003
            if self.fail_on is not None and h == self.fail_on:
                raise RuntimeError("injected OCR failure")
            time.sleep(random.random() * self.max_sleep)
            return [] if h % 3 == 0 else [{"text": f"line {h}", "conf": 0.9}]
        finally:
            self.busy.release()


def fake_frames(n, seed):
    rng = np.random.default_rng(seed)
    frames, prev = [], None
    for i in range(n):
        r = rng.random()
        if r < 0.1:
            img = np.zeros((54, 96, 3), np.uint8)                       # black
        elif r < 0.4 and prev is not None:
            img = prev                                                   # static card repeats
        else:
            img = rng.integers(60, 255, (54, 96, 3), dtype=np.uint8)     # new card
        frames.append((float(i), img))
        prev = img
    return frames


def ocr_threads():
    return [t for t in threading.enumerate() if t.name.startswith(("ocr", "decode"))]


def logic_tests():
    print("\n== logic tests (fake engines)")
    random.seed(1)
    mismatches = 0
    for seed in range(40):
        frames = fake_frames(120, seed)
        ref = []
        reference(FakeEngine(max_sleep=0), "end", frames, ref)
        for workers in (1, 2, 3, 4, 8):
            out, stats = [], {"frames": 0, "ocr_frames": 0}
            with OcrRunner([FakeEngine() for _ in range(workers)]) as runner:
                src = gen(frames)
                ocr_frames(runner, "end", _prefetch(src) if runner.parallel else src, len(frames), stats, out)
            if out != ref:
                mismatches += 1
    check(mismatches == 0, f"output identical to the sequential loop (40 random sequences x 1/2/3/4/8 workers)")
    check(FakeEngine.in_use_violations == 0, "no engine was ever used by two threads at once")
    check(not ocr_threads(), "no worker / decode threads left running")

    # a failing frame: the error reaches the caller, decoding stops, queued work is cancelled
    frames = fake_frames(300, 99)
    target = None
    for _, img in frames[150:]:
        if img.max() >= 60:
            target = int(img[::7, ::11].sum()) * 2654435761 % 1000003
            break
    engines = [FakeEngine(fail_on=target) for _ in range(3)]
    closed, err = [], None
    try:
        with OcrRunner(engines) as runner:
            ocr_frames(runner, "end", _prefetch(gen(frames, closed)), len(frames), {"frames": 0, "ocr_frames": 0}, [])
    except RuntimeError as e:
        err = e
    check(err is not None and "injected" in str(err), "worker exception is re-raised in the main thread")
    check(closed == [True], "frame source closed after the error (ffmpeg would be stopped)")
    check(sum(e.calls for e in engines) < 200, f"queued frames cancelled ({sum(e.calls for e in engines)} of 300 OCR'd)")
    check(not ocr_threads(), "no threads left after the error")

    # consumer stops early (Ctrl+C / error in the main thread): decode thread and source stop
    closed = []
    p = _prefetch(gen(fake_frames(500, 5), closed))
    for i, _ in enumerate(p):
        if i == 3:
            break
    p.close()
    check(closed == [True] and not ocr_threads(), "closing the prefetcher early stops its thread and the source")

    # an error inside the frame source (ffmpeg failure) surfaces in the consumer
    def bad():
        yield 0.0, np.full((54, 96, 3), 200, np.uint8)
        raise OSError("decoder broke")
    try:
        list(_prefetch(bad()))
        got = None
    except OSError as e:
        got = e
    check(got is not None and "decoder" in str(got) and not ocr_threads(), "decoder error re-raised, thread ended")


# --------------------------------------------------------------------------------------------- real OCR
def same_output(a, b):
    """Frames, weights, texts and boxes must be identical. Confidences may differ in the last digits: with more
    workers each engine gets fewer CPU threads, which changes floating-point summation order (verified: with
    equal thread counts parallel output is bit-identical)."""
    if len(a) != len(b):
        return False
    for fa, fb in zip(a, b):
        if fa[:2] != fb[:2] or fa[3] != fb[3] or len(fa[2]) != len(fb[2]):
            return False
        for x, y in zip(fa[2], fb[2]):
            if (x["text"], x["x"], x["y"], x["w"], x["h"]) != (y["text"], y["x"], y["y"], y["w"], y["h"]) \
                    or abs(x["conf"] - y["conf"]) > 1e-3:
                return False
    return True


def real_tests(video, workers_list, n_frames):
    from videoid.ocr import OCR
    print(f"\n== real OCR: {video}")
    cfg = config.load()
    info = media.probe(video)
    start = max(0, info["duration"] - 150)
    frames = [(t, img.copy()) for t, img in media.frames(video, start, n_frames, 1.0, info, cfg["scan"])]
    print(f"  {len(frames)} frames {frames[0][1].shape[1]}x{frames[0][1].shape[0]} from {start / 60:.1f} min")

    results, timings = {}, {}
    for w in workers_list:
        ocfg = dict(cfg["ocr"], workers=w)
        t = time.time()
        engines = [OCR(ocfg) for _ in range(w)]
        init = time.time() - t
        # count how many ONNX calls are inside the GPU at the same time (measured under the lock, if any)
        from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
        from videoid.ocr import _install_session_lock
        _install_session_lock()
        orig, state, mtx = OrtInferSession._videoid_run, {"now": 0, "max": 0}, threading.Lock()

        def counting(self, x):
            with mtx:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            try:
                return orig(self, x)
            finally:
                with mtx:
                    state["now"] -= 1
        OrtInferSession._videoid_run = counting
        try:
            out, stats = [], {"frames": 0, "ocr_frames": 0}
            t = time.time()
            with OcrRunner(engines) as runner:
                src = gen(frames)
                ocr_frames(runner, "end", _prefetch(src) if runner.parallel else src, len(frames), stats, out)
            dt = time.time() - t
        finally:
            OrtInferSession._videoid_run = orig
        results[w], timings[w] = out, dt
        lock = "locked" if engines[0].gpu_lock else "unlocked"
        print(f"  workers={w} ({lock}, {engines[0].device.split(',')[0]}): {dt:6.1f}s  "
              f"{dt / max(1, stats['ocr_frames']):.3f} s/frame  ({stats['ocr_frames']} OCR'd, init {init:.0f}s)  "
              f"max concurrent ONNX calls {state['max']}", flush=True)
        if engines[0].gpu_lock:
            check(state["max"] <= 1, f"workers={w}: GPU calls never overlapped while locked")
        elif w > 1:
            check(state["max"] > 1, f"workers={w}: unlocked calls really ran concurrently")
        del engines
    for w in workers_list[1:]:
        check(same_output(results[workers_list[0]], results[w]),
              f"workers={w}: identical OCR output to workers={workers_list[0]}")
    base = timings[workers_list[0]]
    print("  speed-up vs workers=%d: " % workers_list[0]
          + "  ".join(f"{w}: {base / timings[w]:.2f}x" for w in workers_list))
    check(not ocr_threads(), "no threads left")

    # end to end through ffmpeg (run_ocr) with 2 workers vs 1
    print("\n== end-to-end run_ocr (ffmpeg -> prefetch -> workers)")
    outs = {}
    for w in (1, 2):
        c = copy.deepcopy(cfg)
        c["ocr"]["workers"] = w
        c["scan"].update(open_minutes=0, end_minutes=0.4, fps_end=1.0)
        ctx = pipeline.Context(c, None, None, None, lambda c=c: OCR(c["ocr"]), None)
        outs[w], st = pipeline.run_ocr(ctx, video, info)
        print(f"  workers={w}: {st['ocr_frames']} frames in {st['seconds']}s")
    check(same_output(outs[1], outs[2]), "run_ocr: 2 workers == 1 worker")
    left = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ffmpeg*"], capture_output=True, text=True).stdout
    check("ffmpeg" not in left.lower() and not ocr_threads(), "no ffmpeg process or threads left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--workers", default="1,2,3")
    ap.add_argument("--frames", type=int, default=20, help="seconds of end credits (1 fps) for the real test")
    a = ap.parse_args()
    logic_tests()
    if a.video:
        real_tests(a.video, [int(x) for x in a.workers.split(",")], a.frames)
    print("\n" + ("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED:\n  " + "\n  ".join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

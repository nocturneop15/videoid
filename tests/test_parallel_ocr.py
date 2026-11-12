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
# Every measurement runs in a FRESH child process: OCR engines left alive in the same process (their GPU
# sessions and memory) skewed earlier timings on the RTX 3070 by 2-3x.
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


def gpu_memory_mb():
    """Used memory of the first NVIDIA GPU (nvidia-smi ships with the driver), else None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        used, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def child(spec):
    """One measurement: full path ffmpeg -> prefetch -> workers on one window. Prints one JSON line."""
    import json
    from videoid.ocr import OCR, _install_session_lock, _runs_on_gpu
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
    cfg = config.load()
    cfg["ocr"].update(spec["ocr"])
    cfg["scan"].update(spec.get("scan", {}))
    info = media.probe(spec["video"])
    start = max(0.0, info["duration"] - spec["from_end"])
    mem0 = gpu_memory_mb()
    t = time.time()
    ctx = pipeline.Context(cfg, None, None, None, lambda: OCR(cfg["ocr"]), None)
    engines = ctx.ocr_engines()
    init = time.time() - t
    mem1 = gpu_memory_mb()
    first = engines[0]

    _install_session_lock()
    orig, state, mtx = OrtInferSession._videoid_run, {"now": 0, "max": 0}, threading.Lock()

    def counting(self, x):
        if first.is_gpu and not _runs_on_gpu(self):     # CPU recognition may overlap GPU detection
            return orig(self, x)
        with mtx:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        try:
            return orig(self, x)
        finally:
            with mtx:
                state["now"] -= 1
    OrtInferSession._videoid_run = counting

    out, stats = [], {"frames": 0, "ocr_frames": 0}
    t = time.time()
    with OcrRunner(engines) as runner:
        frames = media.frames(spec["video"], start, spec["seconds"], 1.0, info, cfg["scan"])
        if runner.parallel:
            frames = _prefetch(frames)
        ocr_frames(runner, "end", frames, spec["seconds"], stats, out)
    dt = time.time() - t
    left = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ffmpeg*"], capture_output=True, text=True).stdout
    print("RESULT " + json.dumps({
        "seconds": dt, "init": init, "ocr_frames": stats["ocr_frames"], "frames": stats["frames"],
        "decode_wait": stats["decode_wait"], "ocr_wait": stats["ocr_wait"], "max_gpu_calls": state["max"],
        "locked": bool(first.gpu_lock), "gpu": bool(first.is_gpu), "rec_on_cpu": bool(first.rec_on_cpu),
        "shared": bool(len(engines) > 1 and engines[1].engines[0].text_det.session is
                       first.engines[0].text_det.session),
        "device": first.device.split(",")[0], "mem_before": mem0, "mem_after_init": mem1,
        "clean": "ffmpeg" not in left.lower() and not ocr_threads(), "out": out}, ensure_ascii=False), flush=True)


def run_child(spec):
    import json
    r = subprocess.run([sys.executable, "-X", "faulthandler", __file__, "--child", json.dumps(spec)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
        return {"error": f"exit code {r.returncode}: " + " | ".join(tail)}
    return json.loads(line[7:])


def real_tests(video, workers_list, n_frames):
    print(f"\n== real OCR, each run in a fresh process: {video}")
    info = media.probe(video)
    r = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-i", video], capture_output=True, text=True,
                       encoding="utf-8", errors="replace").stderr
    print("  " + next((l.strip() for l in r.splitlines() if "Video:" in l), "?")[:140])
    mem = gpu_memory_mb()
    if mem:
        print(f"  GPU memory before: {mem[0]} of {mem[1]} MB used")
    windows = [("little text", 150, n_frames), ("end credits", 24, 24)]
    for wname, from_end, secs in windows:
        print(f"\n  -- {wname}: {secs} s from {max(0, info['duration'] - from_end) / 60:.1f} min")
        res = {}
        for w in workers_list:
            x = run_child({"video": video, "from_end": from_end, "seconds": secs, "ocr": {"workers": w}})
            res[w] = x
            if "error" in x:
                check(False, f"{wname}, workers={w}: {x['error']}")
                continue
            how = ("shared sessions" if x["shared"] else "own sessions") if w > 1 else ""
            where = x["device"] + (", recognition on CPU" if x["rec_on_cpu"] else "")
            gm = (f"  GPU mem +{x['mem_after_init'][0] - x['mem_before'][0]} MB"
                  if x["mem_before"] and x["mem_after_init"] else "")
            print(f"  workers={w} ({where}{', ' + how if how else ''}): "
                  f"{x['seconds'] / max(1, x['ocr_frames']):.3f} s/frame  ({x['ocr_frames']} OCR'd in "
                  f"{x['seconds']:.1f}s; waited decoding {x['decode_wait']:.1f}s, OCR {x['ocr_wait']:.1f}s; "
                  f"init {x['init']:.1f}s){gm}", flush=True)
            if x["locked"]:
                check(x["max_gpu_calls"] <= 1, f"{wname}, workers={w}: GPU calls never overlapped while locked")
            check(x["clean"], f"{wname}, workers={w}: no ffmpeg process or threads left")
        ok = [w for w in workers_list if "error" not in res[w]]
        if ok:
            base = res[ok[0]]
            for w in ok[1:]:
                check(same_output(base["out"], res[w]["out"]), f"{wname}, workers={w}: same OCR output as {ok[0]}")
            print("  speed-up vs workers=%d: " % ok[0] + "  ".join(
                f"{w}: {base['seconds'] / res[w]['seconds']:.2f}x" for w in ok))

    # where text recognition runs (end credits = text-heavy), 1 worker, each in a fresh process
    print("\n  -- text recognition on GPU vs CPU (end credits, 1 worker)")
    outs = {}
    for mode in ("gpu", "cpu", "auto"):
        x = run_child({"video": video, "from_end": 24, "seconds": 24, "ocr": {"workers": 1, "rec_device": mode}})
        if "error" in x:
            check(False, f"rec_device={mode}: {x['error']}")
            continue
        if not x["gpu"]:
            print("  (OCR runs on the CPU - nothing to compare)")
            break
        outs[mode] = x["out"]
        print(f"  rec_device={mode:4}: {x['seconds'] / max(1, x['ocr_frames']):.3f} s/frame  -> recognition on "
              f"{'CPU' if x['rec_on_cpu'] else 'GPU'}", flush=True)
    if "gpu" in outs and "cpu" in outs:
        same = sum(1 for a, b in zip(outs["gpu"], outs["cpu"])
                   if [y["text"] for y in a[2]] == [y["text"] for y in b[2]])
        print(f"  text identical in {same} of {len(outs['gpu'])} frames with text")


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--workers", default="1,2,3")
    ap.add_argument("--frames", type=int, default=20, help="seconds of the little-text window (1 fps)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(json.loads(a.child))
        return
    logic_tests()
    if a.video:
        real_tests(a.video, [int(x) for x in a.workers.split(",")], a.frames)
    print("\n" + ("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED:\n  " + "\n  ".join(FAILED)))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()

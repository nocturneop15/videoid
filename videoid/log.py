import sys
import threading
from contextlib import contextmanager

VERBOSITY = 1
_capture = None          # lines collected for the per-movie log file (see capture())
_lock = threading.Lock()


def log(msg, level=1, end="\n"):
    """level 0 = always, 1 = normal, 2 = verbose/debug."""
    if level <= VERBOSITY:
        print(msg, file=sys.stderr, end=end, flush=True)
    _keep(msg, level, end)


def out(msg):
    """Result text (stdout) that also belongs in the per-movie log file."""
    print(msg, flush=True)
    _keep(msg, 0, "\n")


def _keep(msg, level, end):
    # progress lines (end="" / "\r...") are console-only; the file gets normal-level messages
    if _capture is not None and level <= 1 and end == "\n":
        with _lock:
            _capture.extend(str(msg).lstrip("\r").splitlines() or [""])


@contextmanager
def capture():
    """Collect everything logged inside the block into a list (for the movie's log file)."""
    global _capture
    prev, buf = _capture, []
    _capture = buf
    try:
        yield buf
    finally:
        _capture = prev

import sys
import threading
from contextlib import contextmanager

from .ui import color_on, plain

VERBOSITY = 1
_capture = None          # lines collected for the per-movie log file (see capture())
_lock = threading.Lock()


def _print(msg, stream, end="\n"):
    msg = str(msg)
    print(msg if color_on(stream) else plain(msg), file=stream, end=end, flush=True)


def log(msg, level=1, end="\n"):
    """level 0 = always, 1 = normal, 2 = verbose/debug."""
    if level <= VERBOSITY:
        _print(msg, sys.stderr, end)
    _keep(msg, level, end)


def out(msg):
    """Result text (stdout) that also belongs in the per-movie log file."""
    _print(msg, sys.stdout)
    _keep(msg, 0, "\n")


def _keep(msg, level, end):
    # progress lines (end="" / "\r...") are console-only; the file gets normal-level messages, without colours
    if _capture is not None and level <= 1 and end == "\n":
        with _lock:
            _capture.extend(plain(msg).lstrip("\r").splitlines() or [""])


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

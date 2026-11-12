"""Terminal styling: colours for results, prompts and progress steps.

Strings carry ANSI codes; log.py strips them for log files and for output that is not a colour terminal
(redirected to a file, NO_COLOR set). Only ASCII symbols - a cp1250 console cannot print check marks."""
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")

RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = (f"\x1b[{n}m" for n in (91, 92, 93, 94, 95, 96, 97))
ON_RED, ON_GREEN, ON_YELLOW, ON_BLUE = "\x1b[97;41m", "\x1b[30;42m", "\x1b[30;43m", "\x1b[97;44m"

VERDICT = {"CERTAIN": BOLD + GREEN, "PROBABLE": CYAN, "POSSIBLE": YELLOW, "UNLIKELY": RED, "UNKNOWN": RED}


def _enable_windows_vt():
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        ok = False
        for h in (-11, -12):                       # stdout, stderr
            handle = k.GetStdHandle(h)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(handle, ctypes.byref(mode)):
                ok = bool(k.SetConsoleMode(handle, mode.value | 0x0004)) or ok   # VIRTUAL_TERMINAL_PROCESSING
        return ok
    except Exception:
        return False


_VT = _enable_windows_vt()


def color_on(stream):
    """Colours for this stream: a terminal, NO_COLOR not set (FORCE_COLOR forces them)."""
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return _VT and stream.isatty()
    except Exception:
        return False


def plain(text):
    return ANSI.sub("", str(text))


def c(text, style):
    return f"{style}{text}{RESET}"


def badge(text, style):
    return c(f" {text} ", style)


def verdict(v, width=0):
    v = v or "UNKNOWN"
    return c(v.ljust(width), VERDICT.get(v, ""))


# ---- what the engine is doing ----
def header(text, index=None, total=None):
    pos = f"[{index}/{total}] " if index and total and total > 1 else ""
    return "\n" + c(f"=== {pos}", BOLD + BLUE) + c(text, BOLD + WHITE)


def step(text):
    return c("  > ", BOLD + CYAN) + text


def detail(text):
    return c(f"    {text}", DIM)


def good(text):
    return c("  + ", BOLD + GREEN) + c(text, GREEN)


def warn(text):
    return c("  ! ", BOLD + YELLOW) + c(text, YELLOW)


def bad(text):
    return badge("ERROR", ON_RED) + " " + c(text, RED)


def status(label, text):
    """'kodi: wrote ...' / 'rename: skipped ...' - green when something was done, yellow when skipped."""
    done = text.startswith(("wrote", "renamed"))
    return c(f"   {label}: ", BOLD) + c(text, GREEN if done else YELLOW)


def result_badge(v):
    if v == "CERTAIN":
        return badge("IDENTIFIED", ON_GREEN)
    if v in ("PROBABLE", "POSSIBLE"):
        return badge("NOT SURE", ON_YELLOW)
    return badge("NOT IDENTIFIED", ON_RED)


# ---- questions ----
def question(title):
    bar = f" ? {title} "
    return "\n" + c(bar.ljust(max(len(bar), 72)), BOLD + "\x1b[30;43m")


def option(n, text):
    return c(f"   {n:>2}) ", BOLD + YELLOW) + text


def ask(text):
    return c("   >> ", BOLD + YELLOW) + c(text, BOLD + WHITE) + " "

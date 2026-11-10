import sys

VERBOSITY = 1


def log(msg, level=1, end="\n"):
    """level 0 = always, 1 = normal, 2 = verbose/debug."""
    if level <= VERBOSITY:
        print(msg, file=sys.stderr, end=end, flush=True)

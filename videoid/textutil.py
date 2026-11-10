"""Text normalisation and fuzzy matching helpers."""
import re
import unicodedata
from functools import lru_cache

from rapidfuzz import fuzz

_NON_WORD = re.compile(r"[\W_]+")


def _is_latinish(c):
    return ord(c) < 0x0250 or 0x1E00 <= ord(c) <= 0x1EFF        # Latin incl. extended + Vietnamese


def strip_accents(s):
    """Remove diacritics from Latin letters only - combining marks in Devanagari, Thai, etc. are
    vowel signs and must stay."""
    out = []
    for c in s:
        d = unicodedata.normalize("NFKD", c)
        if d and _is_latinish(d[0]):
            out.append("".join(x for x in d if not unicodedata.combining(x)))
        else:
            out.append(c)
    return "".join(out)


def fold(s):
    """Accent-stripped lowercase, SAME LENGTH as the input (so regex positions map back)."""
    out = []
    for c in s:
        d = unicodedata.normalize("NFKD", c)
        base = d[0] if d and _is_latinish(d[0]) else c
        if ord(c) >= 0x0400 and not _is_latinish(c):     # Cyrillic/Greek: drop accents but keep the letter
            base = d[0] if d and not unicodedata.combining(d[0]) else c
        out.append(base.lower()[:1] or " ")
    return "".join(out)


@lru_cache(maxsize=200_000)
def norm(s):
    return _NON_WORD.sub(" ", strip_accents(s or "").casefold()).strip()


SCRIPTS = (("latin", _is_latinish), ("cyrillic", lambda c: 0x0400 <= ord(c) <= 0x052F),
           ("greek", lambda c: 0x0370 <= ord(c) <= 0x03FF), ("arabic", lambda c: 0x0600 <= ord(c) <= 0x06FF),
           ("devanagari", lambda c: 0x0900 <= ord(c) <= 0x097F), ("tamil", lambda c: 0x0B80 <= ord(c) <= 0x0BFF),
           ("telugu", lambda c: 0x0C00 <= ord(c) <= 0x0C7F), ("thai", lambda c: 0x0E00 <= ord(c) <= 0x0E7F),
           ("hangul", lambda c: 0xAC00 <= ord(c) <= 0xD7AF or 0x1100 <= ord(c) <= 0x11FF or 0x3130 <= ord(c) <= 0x318F),
           ("cjk", lambda c: 0x3040 <= ord(c) <= 0x30FF or 0x3400 <= ord(c) <= 0x4DBF or 0x4E00 <= ord(c) <= 0x9FFF
                             or 0xF900 <= ord(c) <= 0xFAFF))
CASED = {"latin", "cyrillic", "greek"}


def script_of(s):
    """Dominant script of the letters in s."""
    counts = {}
    for c in s:
        if not c.isalpha():
            continue
        for name, test in SCRIPTS:
            if test(c):
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "latin"


def compact(s):
    return norm(s).replace(" ", "")


def name_sim(a, b):
    """Similarity 0..100 of two person names, tolerant to OCR spacing/diacritics/order."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    return max(fuzz.ratio(na.replace(" ", ""), nb.replace(" ", "")), fuzz.token_sort_ratio(na, nb))


_ARTICLES = re.compile(r"^(the|a|an|der|die|das|le|la|les|el|il)\s+")


def title_sim(a, b):
    na, nb = _ARTICLES.sub("", norm(a)), _ARTICLES.sub("", norm(b))
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    return max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb) - 3)


def split_camel(s):
    """'JohnSmith' -> 'John Smith' (OCR often drops spaces)."""
    return re.sub(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ])", " ", s)


ROMAN = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


def roman_to_int(s):
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = ROMAN.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total

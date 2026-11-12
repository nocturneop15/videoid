"""Subtitles as quick evidence: character names in the dialogue confirm a film without OCR.

Text subtitle tracks inside the video (SRT, ASS, mov_text, WebVTT) and subtitle files next to it are read;
image subtitles (PGS, VobSub) would need OCR themselves and are skipped. The capitalised words of the
dialogue are compared with the candidate films' character names (TMDB, Wikidata)."""
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .localinfo import _decode
from .media import CREATE_NO_WINDOW, ffmpeg_exe
from .textutil import norm

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
SIDE_EXT = {".srt", ".ass", ".ssa", ".vtt"}
MAX_TRACKS = 3
# character "names" that are roles, not names - they would match ordinary words of the dialogue
GENERIC = set("""self himself herself themselves narrator voice uncredited mother father mom dad mum son daughter
brother sister wife husband boy girl man woman men women kid child baby doctor nurse police policeman officer
cop detective agent guard soldier waiter waitress driver bartender barman priest judge lawyer teacher student
boss manager guest customer reporter singer dancer friend neighbor neighbour old young little big mister miss
mrs lady sir king queen prince princess captain sergeant general president killer man's woman's dead""".split())


def text_tracks(info):
    """Text subtitle streams, full ones before 'forced' (those only translate a few foreign lines)."""
    tr = [s for s in info.get("subtitles") or [] if s["codec"].lower() in TEXT_CODECS]
    return sorted(tr, key=lambda s: s["forced"])[:MAX_TRACKS]


def sidecars(video):
    video = Path(video)
    stem = video.stem.lower()
    try:
        return [f for f in video.parent.iterdir() if f.is_file() and f.suffix.lower() in SIDE_EXT
                and (f.stem.lower() == stem or f.stem.lower().startswith(stem + "."))]
    except OSError:
        return []


def read(video, info, timeout=300):
    """All subtitle text: embedded text tracks (one ffmpeg pass) + subtitle files next to the video.
    Returns (lines, sources) - sources describe where they came from, e.g. ['cze track', 'film.cs.srt']."""
    lines, sources = [], []
    tracks = text_tracks(info)
    if tracks:
        with tempfile.TemporaryDirectory(prefix="videoid-subs-") as tmp:
            cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video)]
            outs = []
            for i, t in enumerate(tracks):
                out = Path(tmp) / f"{i}.srt"
                cmd += ["-map", f"0:{t['stream']}", "-c:s", "srt", "-f", "srt", str(out)]
                outs.append((t, out))
            try:
                subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW)
            except subprocess.TimeoutExpired:
                pass
            for t, out in outs:
                if out.exists() and out.stat().st_size:
                    lines += parse(_decode(out.read_bytes()))
                    sources.append(f"{t['lang'] or 'unknown language'} track")
    for f in sidecars(video):
        try:
            lines += parse(_decode(f.read_bytes()))
            sources.append(f.name)
        except OSError:
            pass
    return lines, sources


_TIME = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->")


def parse(text):
    """SRT / WebVTT / ASS text -> dialogue lines without numbers, times, tags and styling."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Dialogue:"):                     # ASS: text is the 10th field
            parts = line.split(",", 9)
            line = parts[9] if len(parts) == 10 else ""
            line = line.replace("\\N", " ").replace("\\n", " ")
        if not line or line.isdigit() or _TIME.match(line) or line.upper().startswith(("WEBVTT", "[", "NOTE")):
            continue
        line = re.sub(r"\{[^}]*\}|<[^>]+>", "", line).strip()
        if line:
            out.append(line)
    return out


_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.U)


def name_words(lines):
    """Capitalised words of the dialogue, normalised (lower case, no accents) -> how often they occur."""
    c = Counter()
    for line in lines:
        for w in _WORD.findall(line):
            if w[0].isupper() and not w.isupper() or (w.isupper() and len(w) >= 4):
                c[norm(w)] += 1
    return c


def _tokens(character):
    """'Mark "Rent Boy" Renton / Narrator' -> distinctive name words ['mark', 'rent', 'renton']."""
    out = []
    for part in re.split(r"\s*/\s*|\s*\(.*?\)\s*", character or ""):
        for w in norm(part).split():
            if len(w) >= 4 and w not in GENERIC:
                out.append(w)
    return out


def _seen(token, words):
    """How often a character's name word occurs, allowing Czech/Slovak endings: Renton -> Rentone, Rentonovi;
    Laura -> Lauro. Only prefixes of 4+ letters."""
    stems = {token} | ({token[:-1]} if len(token) >= 5 else set())
    return sum(n for w, n in words.items() if any(w.startswith(s) and len(w) - len(s) <= 4 for s in stems))


def character_hits(characters, words, min_seen=2):
    """Distinct characters whose name occurs at least min_seen times in the dialogue."""
    hits, used = [], set()
    for ch in characters:
        toks = [t for t in _tokens(ch) if t not in used]
        best = max(((t, _seen(t, words)) for t in toks), key=lambda x: x[1], default=(None, 0))
        if best[1] >= min_seen:
            used.add(best[0])                  # "Mark Renton" and "Renton" are the same person
            hits.append(ch)
    return hits

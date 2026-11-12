"""Evidence available locally: file name / folder name (guessit), NFO files, container metadata."""
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from .log import log

IMDB_RE = re.compile(r"\b(tt\d{7,9})\b")
TMDB_RE = re.compile(r"themoviedb\.org/movie/(\d+)")
CSFD_RE = re.compile(r"csfd\.(?:cz|sk)/film/(\d+)")
YEAR_RE = re.compile(r"\b(18[89]\d|19\d\d|20\d\d)\b")
GENERIC_NAMES = {"movie", "film", "video", "main", "feature", "title", "vts", "video_ts", "bdmv", "stream",
                 "disc", "cd1", "cd2", "part1", "part2", "sample"}


def oshash(path):
    """OpenSubtitles hash: size + 64-bit sums of first and last 64 KiB. Stable across renames."""
    try:
        size = path.stat().st_size
        if size < 131072:
            return None
        h = size
        with open(path, "rb") as f:
            for off in (0, size - 65536):
                f.seek(off)
                buf = f.read(65536)
                for (v,) in struct.iter_unpack("<Q", buf):
                    h = (h + v) & 0xFFFFFFFFFFFFFFFF
        return f"{h:016x}"
    except OSError:
        return None


_EP_MARK = re.compile(r"(?i)(?:^|[^a-z0-9])(?:s\d{1,2}[ ._-]?e\d{1,3}|\d{1,2}x\d{1,3}|e\d{1,3}|ep\.?|episode|"
                      r"epizoda|d[ií]l|[čc][aá]st|folge|serie|season|s[eé]rie|series)(?=[^a-z]|$)")


def episode_marked(name, season=None):
    """Only a season or an explicit marker (S01E02, 1x02, E05, Episode, díl, část, ...) makes a file a TV
    episode; guessit alone also calls "35 let ..." or "2001 ..." episode 35 / 2001."""
    return season is not None or bool(_EP_MARK.search(name or ""))


def is_episode(local, video_name=""):
    """For files scanned before this rule: the stored guess is re-checked against the file name."""
    if local.get("type") != "episode":
        return False
    return local.get("nfo_episode") or episode_marked(video_name, local.get("season"))


def guess_from_name(path):
    out = {"titles": [], "years": [], "type": None, "season": None, "episode": None}
    try:
        from guessit import guessit
    except ImportError:
        guessit = None
    candidates = [path.name]
    parent = path.parent.name
    if parent and parent.lower() not in GENERIC_NAMES:
        candidates.append(parent)
    for i, name in enumerate(candidates):
        src = "filename" if i == 0 else "folder"
        if guessit:
            try:
                g = guessit(name)
            except Exception:  # guessit can choke on odd names
                g = {}
            t = g.get("title")
            if t and t.lower() not in GENERIC_NAMES:
                out["titles"].append((str(t), src))
                if g.get("alternative_title"):
                    alt = g["alternative_title"]
                    for a in (alt if isinstance(alt, list) else [alt]):
                        out["titles"].append((str(a), src))
            if g.get("year"):
                out["years"].append((int(g["year"]), src))
            if i == 0:
                out["type"] = g.get("type")
                out["season"], out["episode"] = g.get("season"), g.get("episode")
                if out["type"] == "episode" and not episode_marked(name, out["season"]):
                    # "35 let Spirituál kvintetu": a bare number is not an episode - read it as a film title
                    out.update(type="movie", season=None, episode=None)
                    try:
                        mt = guessit(name, {"type": "movie"}).get("title")
                    except Exception:
                        mt = None
                    if mt and t and str(mt) != str(t):
                        out["titles"] = [(str(mt), src) if x == (str(t), src) else x for x in out["titles"]]
        else:
            stem = re.sub(r"[._]+", " ", Path(name).stem)
            m = YEAR_RE.search(stem)
            if m:
                out["years"].append((int(m.group(1)), src))
                stem = stem[:m.start()]
            stem = stem.strip(" -([")
            if stem:
                out["titles"].append((stem, src))
    return out


def _decode(raw):
    for enc in ("utf-8-sig", "cp1250", "cp437"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


MARKER = "created by VideoID"     # NFOs written by `videoid kodi` - not evidence for a new scan


def is_ours(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return MARKER in fh.read(4096)
    except OSError:
        return False


def find_nfos(video):
    folder = video.parent
    nfos = sorted(n for n in folder.glob("*.nfo") if not is_ours(n))
    if not nfos:
        return []
    same = [n for n in nfos if n.stem.lower() == video.stem.lower()]
    generic = [n for n in nfos if n.stem.lower() in ("movie", "film")]
    videos_in_folder = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in
                        (".mkv", ".mp4", ".avi", ".m4v", ".mov", ".wmv", ".ts", ".m2ts", ".mpg")]
    if same or generic:
        return same + generic
    # a lone video in the folder owns every NFO there (scene releases)
    return nfos if len(videos_in_folder) <= 1 else []


def parse_nfo(path):
    text = _decode(path.read_bytes())
    out = {"ids": {}, "titles": [], "years": [], "people": [], "file": str(path)}
    if m := IMDB_RE.search(text):
        out["ids"]["imdb"] = m.group(1)
    if m := TMDB_RE.search(text):
        out["ids"]["tmdb"] = m.group(1)
    if m := CSFD_RE.search(text):
        out["ids"]["csfd"] = m.group(1)

    xml_start = text.find("<movie")
    if xml_start < 0:
        xml_start = text.find("<episodedetails")
        out["episode_nfo"] = xml_start >= 0
    if xml_start >= 0:
        _parse_kodi(text[xml_start:], out)
    else:
        # free-text (scene) NFO: "Title: ..." / "Year: ..." / "Director: ..."
        for key, val in re.findall(r"^\s*([A-Za-z .]{2,20}?)\s*[:\.]{1,}\s*(.+?)\s*$", text, re.M):
            k = key.strip().lower()
            v = val.strip()
            if k in ("title", "movie", "name", "film", "release name", "original title", "nazev", "název"):
                out["titles"].append((v, "nfo"))
            elif k in ("year", "rok", "released", "release date"):
                if m := YEAR_RE.search(v):
                    out["years"].append((int(m.group(1)), "nfo"))
            elif k in ("director", "directed by", "režie", "rezie"):
                out["people"] += [{"name": n.strip(), "role": "director", "source": "nfo"}
                                  for n in re.split(r"[,/&]", v) if n.strip()]
            elif k in ("cast", "starring", "actors", "hrají", "hraji"):
                out["people"] += [{"name": n.strip(), "role": "cast", "source": "nfo"}
                                  for n in re.split(r"[,/&]", v) if n.strip()]
    return out


def _parse_kodi(xml, out):
    end = max(xml.rfind("</movie>"), xml.rfind("</episodedetails>"))
    if end > 0:
        xml = xml[: xml.find(">", end) + 1]
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        for tag in ("title", "originaltitle"):
            for v in re.findall(rf"<{tag}>([^<]+)</{tag}>", xml):
                out["titles"].append((v.strip(), "nfo"))
        if m := re.search(r"<year>(\d{4})</year>", xml):
            out["years"].append((int(m.group(1)), "nfo"))
        return
    for tag in ("title", "originaltitle", "sorttitle"):
        for e in root.findall(tag):
            if e.text and e.text.strip():
                out["titles"].append((e.text.strip(), "nfo"))
    for tag in ("year", "premiered", "releasedate"):
        e = root.find(tag)
        if e is not None and e.text and (m := YEAR_RE.search(e.text)):
            out["years"].append((int(m.group(1)), "nfo"))
            break
    for u in root.findall("uniqueid"):
        typ = (u.get("type") or "").lower()
        if u.text and typ in ("imdb", "tmdb", "csfd"):
            out["ids"][typ] = u.text.strip()
    for tag, key in (("id", None), ("imdbid", "imdb"), ("tmdbid", "tmdb")):
        e = root.find(tag)
        if e is not None and e.text:
            v = e.text.strip()
            if key:
                out["ids"].setdefault(key, v)
            elif IMDB_RE.fullmatch(v):
                out["ids"].setdefault("imdb", v)
    for tag, role in (("director", "director"), ("credits", "writer"), ("composer", "composer"),
                      ("producer", "producer")):
        for e in root.findall(tag):
            if e.text and e.text.strip():
                out["people"].append({"name": e.text.strip(), "role": role, "source": "nfo"})
    for a in root.findall("actor"):
        n = a.find("name")
        if n is not None and n.text:
            out["people"].append({"name": n.text.strip(), "role": "cast", "source": "nfo"})


def gather(path, probe):
    """Collect all local evidence for a video file."""
    g = guess_from_name(path)
    ev = {"titles": list(g["titles"]), "years": list(g["years"]), "ids": {}, "people": [],
          "type": g["type"], "season": g["season"], "episode": g["episode"], "nfo_files": []}
    for nfo in find_nfos(path):
        try:
            n = parse_nfo(nfo)
        except OSError as e:
            log(f"  cannot read {nfo}: {e}", 1)
            continue
        ev["nfo_files"].append(n["file"])
        for k, v in n["ids"].items():
            ev["ids"].setdefault(k, v)
        ev["titles"] += n["titles"]
        ev["years"] += n["years"]
        ev["people"] += n["people"]
        if n.get("episode_nfo"):
            ev.update(type="episode", nfo_episode=True)
    if probe.get("title"):
        t = probe["title"]
        # container titles are often release names - run them through guessit too
        try:
            from guessit import guessit
            gt = guessit(t)
            if gt.get("title"):
                ev["titles"].append((str(gt["title"]), "embedded"))
            if gt.get("year"):
                ev["years"].append((int(gt["year"]), "embedded"))
        except Exception:
            ev["titles"].append((t, "embedded"))
    # dedupe preserving order
    seen = set()
    ev["titles"] = [x for x in ev["titles"] if not (x[0].lower() in seen or seen.add(x[0].lower()))]
    ev["years"] = list(dict.fromkeys(ev["years"]))
    return ev

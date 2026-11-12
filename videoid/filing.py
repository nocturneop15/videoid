"""Files next to the movie: its log file and renaming it (with its companions) to the identified title."""
import json
import re
import time
from pathlib import Path

from .film import Film
from .kodi import VERDICT_RANK, _pick, refresh_details

ART = ("poster", "fanart", "thumb", "landscape", "banner", "clearlogo", "clearart", "discart", "logo", "disc")
VIDEO_EXT = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".vob",
             ".webm", ".flv", ".ogm", ".divx", ".3gp", ".rmvb"}


def log_path(video):
    return Path(video).with_suffix(".log")


def write_log(video, lines):
    """Append this run's messages to <movie>.log next to the movie."""
    target = log_path(video)
    head = f"===== VideoID {time.strftime('%Y-%m-%d %H:%M:%S')} ====="
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write("\n".join([head] + [l for l in lines if l.strip() or l == ""]).rstrip() + "\n\n")
        return target
    except OSError:
        return None


def clean_name(text):
    """Make a title safe as a Windows file name."""
    text = re.sub(r"\s*:\s*", " - ", text)
    text = re.sub(r'[<>"/\\|?*\x00-\x1f]', "", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(". ")
    return text[:180]


def new_stem(film, pattern, prefer):
    """The title Kodi gets in the NFO (same source order, kodi.prefer), so file name and library agree."""
    title = _pick(film, "title", prefer)[0] or film.title
    if not title:
        return None
    original = _pick(film, "originaltitle", prefer)[0] or film.title or title
    stem = pattern.format(title=title, originaltitle=original, year=film.year or "")
    return clean_name(re.sub(r"\(\s*\)|\[\s*\]", "", stem)) or None


def companions(video):
    """Files that belong to the movie: same stem + '.something' (subtitles, nfo, log) or '-poster.jpg' etc."""
    video = Path(video)
    stem = video.stem.lower()
    out = []
    for f in video.parent.iterdir():
        if f == video or not f.is_file():
            continue
        n = f.name.lower()
        if n.startswith(stem + "."):
            if f.suffix.lower() in VIDEO_EXT:          # another video (e.g. "Film.part2.mkv") is not ours
                continue
            out.append((f, f.name[len(video.stem):]))
        elif any(n.startswith(f"{stem}-{a}.") for a in ART):
            out.append((f, f.name[len(video.stem):]))
    return out


def rename_movie(db, fid, matcher, rcfg, prefer):
    """Rename the movie file and its companions to "Title (Year)". Returns (new path or None, status text)."""
    row = db.file_get(fid)
    video = Path(row["path"])
    min_rank = VERDICT_RANK.get(str(rcfg.get("min_verdict", "PROBABLE")).upper(), 2)
    if VERDICT_RANK.get(row["verdict"] or "", 0) < min_rank:
        return None, f"not renamed ({row['verdict'] or 'not identified'})"
    from .localinfo import is_episode
    if is_episode(json.loads(row["local_json"] or "{}"), video.name):
        return None, "not renamed (TV episode)"
    if not video.exists():
        return None, "not renamed (file not found)"
    fj = db.film_json(row["film_key"])
    if not fj:
        return None, "not renamed (no film record)"
    film = refresh_details(Film.from_dict(fj), matcher, db)
    stem = new_stem(film, rcfg.get("pattern") or "{title} ({year})", list(prefer or ["tmdb", "csfd"]))
    if not stem:
        return None, "not renamed (no title)"
    if stem == video.stem:
        return None, "name already correct"
    target = video.with_name(stem + video.suffix)
    moves = [(video, target)] + [(f, video.with_name(stem + rest)) for f, rest in companions(video)]
    if target.exists() and target.name.lower() != video.name.lower():   # case-only change is fine on Windows
        return None, f"not renamed ({target.name} already exists)"
    done = []
    for src, dst in moves:
        if dst.exists() and dst.name.lower() != src.name.lower():
            continue                                   # never overwrite a companion that is already there
        try:
            src.rename(dst)
            done.append((src, dst))
        except OSError as e:
            if src == video:
                return None, f"not renamed ({e.strerror or e})"
    db.file_move(fid, str(target))
    extra = f" (+{len(done) - 1} companion file{'s' if len(done) != 2 else ''})" if len(done) > 1 else ""
    return target, f"renamed {video.name} -> {target.name}{extra}"

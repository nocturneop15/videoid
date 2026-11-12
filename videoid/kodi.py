"""Kodi-compatible metadata: <video>.nfo (movie XML) plus <video>-poster.jpg / <video>-fanart.jpg.

Kodi reads these with the "Local information only" scraper or any scraper with "use local
information" (the default). Format: https://kodi.wiki/view/NFO_files/Movies
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from .film import Film
from .localinfo import MARKER, is_ours
from .log import log
from .net import TOOL_UA

VERDICT_RANK = {"CERTAIN": 3, "PROBABLE": 2, "POSSIBLE": 1}
RATING_NAME = {"tmdb": "themoviedb", "csfd": "csfd", "imdb": "imdb"}
MAX_ACTORS = 40


def refresh_details(film, matcher, db, force=False):
    """Films matched before details were stored (or with --refresh): re-fetch plot, genres, artwork...
    TMDB / CSFD responses usually come straight from the HTTP cache."""
    have = set(film.details)
    stale = set()
    if matcher.tmdb and "tmdb" in have and film.details["tmdb"].get("lang", "en-US") != matcher.tmdb.tag:
        stale.add("tmdb")                  # fetched in another language before scrape_language changed
    for name, src in (("tmdb", matcher.tmdb), ("csfd", matcher.csfd)):
        if src and name in film.ids and (force or name not in have or name in stale):
            if name in stale:
                film.details.pop(name, None)
            try:
                f = src.fetch(film.ids[name])
            except Exception as e:            # source down / blocked: write what we have
                log(f"  {name}: {e}", 1)
                f = None
            if f:
                film.merge(f)
    if set(film.details) != have or force or stale:
        db.save_film(film)
    return film


def _pick(film, field, prefer):
    for src in prefer + [s for s in film.details if s not in prefer]:
        v = film.details.get(src, {}).get(field)
        if v:
            return v, src
    return None, None


def _distinct_people(names):
    """Drop abbreviated duplicates: "R. Wright Campbell" next to "Robert Wright Campbell"."""
    out = []
    for n in names:
        t = n.replace(".", ". ").split()
        dup = False
        for o in out:
            u = o.replace(".", ". ").split()
            if t[-1:] == u[-1:] and len(t) == len(u) and all(
                    a == b or (a.endswith(".") and b.startswith(a[:-1])) or (b.endswith(".") and a.startswith(b[:-1]))
                    for a, b in zip(t, u)):
                dup = True
                break
        if not dup:
            out.append(n)
    return out


def _sub(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if text is not None:
        e.text = str(text)
    return e


def build_nfo(film, prefer, duration=None):
    """Kodi <movie> XML for a Film."""
    root = ET.Element("movie")
    title, _ = _pick(film, "title", prefer)
    title = title or film.title
    original, _ = _pick(film, "originaltitle", prefer)
    _sub(root, "title", title)
    _sub(root, "originaltitle", original or film.title)

    ratings = [(s, d["rating"]) for s in prefer + [s for s in film.details if s not in prefer]
               if (d := film.details.get(s)) and d.get("rating") and d["rating"].get("value") is not None]
    if ratings:
        rs = _sub(root, "ratings")
        for i, (src, r) in enumerate(ratings):
            e = _sub(rs, "rating", name=RATING_NAME.get(src, src), max=r.get("max", 10),
                     default="true" if i == 0 else "false")
            _sub(e, "value", r["value"])
            if r.get("votes"):
                _sub(e, "votes", r["votes"])

    plot, _ = _pick(film, "plot", prefer)
    if plot:
        _sub(root, "plot", plot)
        _sub(root, "outline", plot.split(". ")[0].rstrip(".") + "." if len(plot) > 300 else plot)
    tagline, _ = _pick(film, "tagline", prefer)
    if tagline:
        _sub(root, "tagline", tagline)
    runtime = film.runtime or (round(duration / 60) if duration else None)
    if runtime:
        _sub(root, "runtime", runtime)

    poster, _ = _pick(film, "poster", prefer)
    fanart, _ = _pick(film, "fanart", prefer)
    if poster:
        _sub(root, "thumb", poster, aspect="poster")
    if fanart:
        _sub(_sub(root, "fanart"), "thumb", fanart)

    default = next((k for k in ("imdb", "tmdb", "csfd", "wikidata") if k in film.ids), None)
    for k in ("imdb", "tmdb", "csfd", "wikidata"):
        if k in film.ids:
            _sub(root, "uniqueid", film.ids[k], type=k, default="true" if k == default else "false")

    for field, tag in (("genres", "genre"), ("countries", "country")):
        vals, _ = _pick(film, field, prefer)
        for v in vals or []:
            _sub(root, tag, v)
    for n in _distinct_people(film.credits.get("writer", [])):
        _sub(root, "credits", n)
    for n in _distinct_people(film.credits.get("director", [])):
        _sub(root, "director", n)
    premiered, _ = _pick(film, "premiered", prefer)
    if premiered:
        _sub(root, "premiered", premiered)
    if film.year:
        _sub(root, "year", film.year)
    studios, _ = _pick(film, "studios", prefer)
    for v in studios or []:
        _sub(root, "studio", v)
    tags, _ = _pick(film, "tags", prefer)
    for v in tags or []:
        _sub(root, "tag", v)

    chars, thumbs = {}, {}
    for d in film.details.values():
        chars = {**d.get("characters", {}), **chars}
        thumbs = {**d.get("thumbs", {}), **thumbs}
    for i, n in enumerate(film.credits.get("cast", [])[:MAX_ACTORS]):
        a = _sub(root, "actor")
        _sub(a, "name", n)
        if chars.get(n):
            _sub(a, "role", chars[n])
        _sub(a, "order", i)
        if thumbs.get(n):
            _sub(a, "thumb", thumbs[n])

    ET.indent(root, "  ")
    srcs = ", ".join(sorted(film.sources | set(film.details)))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n'
            f"<!-- {MARKER} from {srcs} -->\n" + ET.tostring(root, encoding="unicode") + "\n")


def _download(url, target):
    try:
        r = requests.get(url, headers={"User-Agent": TOOL_UA}, timeout=60)
        r.raise_for_status()
        if not r.headers.get("Content-Type", "image/").startswith("image/"):
            raise ValueError(f"not an image: {r.headers.get('Content-Type')}")
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(target)
        return True
    except (requests.RequestException, ValueError, OSError) as e:
        log(f"  artwork {url}: {e}", 1)
        return False


def write_for_file(db, fid, matcher, kcfg, overwrite=False, dry_run=False, refresh=False):
    """Write NFO (+ artwork) next to one analysed file. Returns a short status string."""
    row = db.file_get(fid)
    video = Path(row["path"])
    min_rank = VERDICT_RANK.get(str(kcfg.get("min_verdict", "CERTAIN")).upper(), 3)
    if VERDICT_RANK.get(row["verdict"] or "", 0) < min_rank:
        return f"skipped ({row['verdict'] or 'not identified'})"
    local = json.loads(row["local_json"] or "{}")
    from .localinfo import is_episode
    if is_episode(local, video.name):
        return "skipped (TV episode - only movies are written)"
    if not video.exists():
        return "skipped (file not found)"
    fj = db.film_json(row["film_key"])
    if not fj:
        return "skipped (no film record)"
    film = refresh_details(Film.from_dict(fj), matcher, db, force=refresh)

    # Kodi reads <video>.nfo, else movie.nfo - never replace one we did not write without --overwrite
    target = video.with_suffix(".nfo")
    foreign = [n for n in (target, video.with_name("movie.nfo")) if n.exists() and not is_ours(n)]
    if foreign and not overwrite:
        if foreign[0] == target:
            return f"skipped (existing {target.name}; --overwrite replaces it and keeps a .bak copy)"
        return f"skipped (existing movie.nfo; --overwrite writes {target.name}, which Kodi reads instead)"
    xml = build_nfo(film, list(kcfg.get("prefer") or ["tmdb", "csfd"]), row["duration"])
    if dry_run:
        print(xml)
        return f"would write {target.name}"
    done = []
    if target in foreign:
        bak = target.with_name(target.name + ".bak")
        target.replace(bak)
        done.append(f"(old one kept as {bak.name})")
    target.write_text(xml, encoding="utf-8")
    done.insert(0, target.name)

    if kcfg.get("artwork", True):
        prefer = list(kcfg.get("prefer") or ["tmdb", "csfd"])
        for field in ("poster", "fanart"):
            url, _ = _pick(film, field, prefer)
            if not url:
                continue
            ext = Path(url.split("?")[0]).suffix.lower()
            art = video.with_name(f"{video.stem}-{field}{ext if ext in ('.jpg', '.jpeg', '.png') else '.jpg'}")
            if art.exists() and not overwrite:
                continue
            if _download(url, art):
                done.append(art.name)
    return "wrote " + ", ".join(done)

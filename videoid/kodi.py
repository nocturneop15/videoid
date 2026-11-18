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
    """Complete the film from every source before an NFO is written - also on --dry-run, which writes no files
    but keeps everything fetched: ids missing for a database are looked up (Wikidata / ČSFD / TMDB / IMDb),
    plot, genres, artwork... are fetched where missing, and the result is stored in the database (full records
    also in the permanent web cache), so the next run needs no network for it. --refresh fetches again."""
    before = json.dumps(film.to_dict(), sort_keys=True, ensure_ascii=False)
    stale = set()
    if matcher.tmdb and "tmdb" in film.details and film.details["tmdb"].get("lang", "en-US") != matcher.tmdb.tag:
        stale.add("tmdb")                  # fetched in another language before scrape_language changed
    matcher.http.bypass = force            # --refresh: fetch again, even records kept forever
    try:
        try:
            matcher._finalize(film, {})    # ids for the databases the record lacks (their full records merge in)
        except Exception as e:             # source down / blocked: go on with what we have
            log(f"  completing ids: {e}", 1)
        for name, src in (("tmdb", matcher.tmdb), ("csfd", matcher.csfd)):
            if src and name in film.ids and (force or name not in film.details or name in stale):
                if name in stale:
                    film.details.pop(name, None)
                try:
                    f = src.fetch(film.ids[name])
                except Exception as e:
                    log(f"  {name}: {e}", 1)
                    f = None
                if f:
                    film.merge(f)
    finally:
        matcher.http.bypass = False
    if force or json.dumps(film.to_dict(), sort_keys=True, ensure_ascii=False) != before:
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


# ---------------------------------------------------------------- completing an existing (foreign) NFO
COMPLETED = "completed by VideoID"          # must not contain MARKER: the NFO stays the user's own evidence
SINGLE = ("title", "originaltitle", "plot", "outline", "tagline", "runtime", "premiered", "year")
MULTI = ("genre", "country", "credits", "director", "studio", "tag")


def _key(s):
    return " ".join((s or "").split()).casefold()


def _blank(e):
    return e is not None and not (e.text or "").strip() and len(e) == 0


def nfo_matches(nfo, film):
    """Does a parsed NFO (localinfo.parse_nfo) name this film? Shared ids must agree; without ids the title
    must fit and the year be within a year."""
    from .textutil import title_sim
    shared = [k for k in nfo["ids"] if k in film.ids]
    if shared:
        return all(film.ids[k] == str(nfo["ids"][k]) for k in shared)
    if not nfo["titles"]:
        return False
    if max((title_sim(t, ft) for t, _ in nfo["titles"] for ft in film.titles), default=0) < 90:
        return False
    years = [y for y, _ in nfo["years"]]
    return not (years and film.year and min(abs(y - film.year) for y in years) > 1)


def verify_against_video(path, nfo, db, fid):
    """Check an existing NFO against what was read from the video itself: its real length, the names read
    from its credits (OCR / local AI), the character names heard in its subtitles, and the quick NFO check of
    the scan. Returns (support, contradictions) - two lists of short texts."""
    import re
    from rapidfuzz import fuzz, process
    from .localinfo import _decode
    from .textutil import compact
    row = db.file_get(fid)
    text = _decode(path.read_bytes())
    support, contra = [], []

    # length: the NFO's <runtime> (minutes) or its stream details (seconds) against the file's duration
    nfo_min = None
    if m := re.search(r"<durationinseconds>\s*(\d+)", text):
        nfo_min = int(m.group(1)) / 60
    elif m := re.search(r"<runtime>\s*(\d+)", text):
        nfo_min = int(m.group(1))
    if nfo_min and row["duration"]:
        file_min = row["duration"] / 60
        d = abs(nfo_min - file_min)
        msg = f"NFO {nfo_min:.0f} min, video {file_min:.0f} min"
        if d <= 8:
            support.append(f"length fits ({msg})")
        elif d > 15:                        # more than a different cut / PAL speed-up explains
            contra.append(f"length differs ({msg})")

    # names read from the video's credits against the NFO's directors, writers, actors...
    read = [p["name"] for p in db.people(fid) if p["source"] in ("ocr", "llm")]
    listed = list(dict.fromkeys(compact(p["name"]) for p in nfo["people"] if compact(p["name"])))
    if read and listed:
        hits = {h[0] for n in read if len(compact(n)) >= 5
                for h in [process.extractOne(compact(n), listed, scorer=fuzz.ratio, score_cutoff=88)] if h}
        if hits:
            support.append(f"{len(hits)} of the NFO's people read in the video's credits")
        elif len(read) >= 3 and len(listed) >= 3:
            contra.append(f"none of the NFO's {len(listed)} people among the {len(read)} names read in the credits")

    # character names heard in the subtitles against the NFO's actor roles
    ms = db.matches(fid)
    chars = (json.loads(ms[0]["evidence_json"]).get("subtitles") or {}).get("characters", []) if ms else []
    roles = " ".join(re.findall(r"<role>([^<]+)</role>", text)).casefold()
    if chars and roles:
        heard = [c for c in chars if c.casefold() in roles]
        if heard:
            support.append(f"{len(heard)} of its characters named in the subtitles ({', '.join(heard[:3])})")

    meta = json.loads(row["ocr_meta_json"] or "{}")
    if meta.get("nfo_verified"):
        support.append("confirmed by the end credits during the scan")
    return support, contra


def _split_xml(text):
    """(text before <movie>, the <movie> element, text after it) - scene NFOs wrap the XML in other text."""
    start = text.find("<movie")
    end = text.rfind("</movie>")
    if start < 0 or end < 0:
        return None
    end += len("</movie>")
    head = text[:start]
    if head.lstrip().startswith("<?xml"):                    # the declaration is written again
        head = head[head.find("?>") + 2:] if "?>" in head else ""
    return head, text[start:end], text[end:]


def complete_nfo(root, fresh):
    """Add to the existing <movie> element `root` what the freshly built one `fresh` has and it lacks.
    Nothing already there is changed or removed. Returns the names of the fields added."""
    added = []

    def put(e):
        """after the last element with the same tag, else at the end"""
        same = [i for i, c in enumerate(root) if c.tag == e.tag]
        root.insert(same[-1] + 1 if same else len(root), e)
        added.append(e.tag)

    for e in fresh:
        tag = e.tag
        if tag in SINGLE:
            old = root.find(tag)
            if old is None:
                put(e)
            elif _blank(old):
                old.text = e.text
                added.append(tag)
        elif tag in MULTI:
            if _key(e.text) not in {_key(o.text) for o in root.findall(tag)}:
                put(e)
        elif tag == "uniqueid":
            if (e.get("type") or "").lower() not in {(o.get("type") or "").lower() for o in root.findall(tag)}:
                e.set("default", "false")                    # the NFO's own default id stays the default
                put(e)
        elif tag == "ratings":
            old = root.find(tag)
            if old is None:
                put(e)
            else:
                have = {(r.get("name") or "").lower() for r in old.findall("rating")}
                for r in e.findall("rating"):
                    if (r.get("name") or "").lower() not in have:
                        r.set("default", "false")
                        old.append(r)
                        added.append("rating")
        elif tag == "thumb":                                  # poster
            if not any((t.get("aspect") or "poster") == "poster" and (t.text or "").strip()
                       for t in root.findall("thumb")):
                put(e)
        elif tag == "fanart":
            old = root.find(tag)
            if old is None or not any((t.text or "").strip() for t in old.findall("thumb")):
                if old is not None:
                    root.remove(old)
                put(e)
        elif tag == "actor":
            name = _key(e.findtext("name"))
            old = next((a for a in root.findall("actor") if _key(a.findtext("name")) == name), None)
            if old is None:
                n = len(root.findall("actor"))
                if (o := e.find("order")) is not None:
                    o.text = str(n)                          # after the NFO's own actors
                put(e)
            else:
                for sub in ("role", "thumb"):
                    if (old.find(sub) is None or _blank(old.find(sub))) and e.find(sub) is not None:
                        if old.find(sub) is not None:
                            old.remove(old.find(sub))
                        old.append(e.find(sub))
                        added.append(f"actor {sub}")
    return added


def _complete_file(path, film, xml, dry_run):
    """Fill in the foreign Kodi NFO at `path` from the freshly built `xml`. Returns (changed, status), or None
    when the NFO is not a Kodi <movie> NFO (free text: it can't be completed)."""
    from .localinfo import _decode
    text = _decode(path.read_bytes())
    parts = _split_xml(text)
    if not parts:
        return None
    head, body, tail = parts
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        root = ET.fromstring(body, parser=parser)
    except ET.ParseError:
        return None
    fresh = ET.fromstring(xml[xml.find("<movie"):])
    added = complete_nfo(root, fresh)
    if not added:
        return False, f"existing {path.name} already has everything"
    counts = {}
    for a in added:
        counts[a] = counts.get(a, 0) + 1
    what = ", ".join(f"{n} {t}" if n > 1 else t for t, n in counts.items())
    for c in list(root):                                     # one note, replaced on every completion
        if c.tag is ET.Comment and COMPLETED in (c.text or ""):
            root.remove(c)
    srcs = ", ".join(sorted(film.sources | set(film.details)))
    root.insert(0, ET.Comment(f" missing fields {COMPLETED} from {srcs} "))
    ET.indent(root, "  ")
    out = ('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n' + head.lstrip("\r\n")
           + ET.tostring(root, encoding="unicode") + tail)
    if dry_run:
        print(out)
        return True, f"missing fields into {path.name} ({what})"
    bak = path.with_name(path.name + ".bak")
    if not bak.exists():                                     # the original as it was, kept once
        bak.write_bytes(path.read_bytes())
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(out, encoding="utf-8")
    tmp.replace(path)
    return True, f"missing fields into {path.name} ({what}; original kept as {bak.name})"


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
    xml = build_nfo(film, list(kcfg.get("prefer") or ["tmdb", "csfd"]), row["duration"])
    done = []
    if foreign and not overwrite:
        # an NFO naming the identified film AND fitting the video itself (length, credits, subtitles) is kept and
        # only completed with what the databases add
        from .localinfo import parse_nfo
        notes, kept, changed = [], [], []
        for n in foreign:
            res, fits = None, True
            try:
                nfo = parse_nfo(n)
                same = nfo_matches(nfo, film)
                if same and kcfg.get("complete_nfo", True):
                    support, contra = verify_against_video(n, nfo, db, fid)
                    fits = bool(support) and not contra
                    if fits:
                        res = _complete_file(n, film, xml, dry_run)
                        if res:
                            res = (res[0], res[1] + "; verified against the video: " + ", ".join(support))
                    else:
                        why = "; ".join(contra) or "nothing read from the video to check it against"
                        notes.append(f"existing {n.name} not verified against the video ({why}) - left alone")
            except OSError as e:
                res, same = (False, f"{n.name}: {e}"), True
            if res:
                (changed if res[0] else kept).append(res[1])
            elif not fits:
                pass
            elif not same:
                notes.append(f"existing {n.name} names a different film - left alone")
            elif kcfg.get("complete_nfo", True):
                notes.append(f"existing {n.name} is not a Kodi NFO - can't be completed")
            else:
                notes.append(f"existing {n.name} kept (kodi.complete_nfo is off)")
        if not (changed or kept):
            notes.append("--overwrite replaces it and keeps a .bak copy" if target in foreign else
                         f"--overwrite writes {target.name}, which Kodi reads instead")
            return "skipped (" + "; ".join(notes) + ")"
        if dry_run:
            return ("would write " if changed else "") + "; ".join(changed + kept + notes) + \
                " (dry run: nothing written next to the video; the data fetched is kept in the database)"
        if not changed:
            done.append("; ".join(kept + notes))
            done = _artwork(video, film, kcfg, overwrite, done)
            return done[0] if len(done) == 1 else "wrote " + ", ".join(done[1:]) + f" ({done[0]})"
        done.append("; ".join(changed + kept + notes))
    else:
        if dry_run:
            print(xml)
            return f"would write {target.name} (dry run: the data fetched is kept in the database)"
        if target in foreign:
            bak = target.with_name(target.name + ".bak")
            target.replace(bak)
            done.append(f"(old one kept as {bak.name})")
        target.write_text(xml, encoding="utf-8")
        done.insert(0, target.name)
    return "wrote " + ", ".join(_artwork(video, film, kcfg, overwrite, done))


def _artwork(video, film, kcfg, overwrite, done):
    """Download <video>-poster / -fanart (missing ones only, unless overwrite); their names join `done`."""
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
    return done

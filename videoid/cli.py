"""Command line interface."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from . import __version__, config, log as logmod, ui
from .film import Film
from .log import log


def _repair_argv(argv):
    """Undo the Windows trailing-backslash quoting trap.

    PowerShell passes 'X:\\My films\\' to a program as "X:\\My films\\" and the C runtime reads the closing \\" as a
    literal quote, so the path and every argument after it arrive as ONE argument: 'X:\\My films" --force --kodi'.
    A real Windows path can never contain '"', so split there: the path gets its backslash back and the rest is
    split into the arguments it was meant to be."""
    if sys.platform != "win32":
        return list(argv)
    out = []
    for a in argv:
        if '"' in a:
            head, _, rest = a.partition('"')
            out.append(head + "\\")
            out.extend(_repair_argv(rest.split()))
        else:
            out.append(a)
    return out


def _ctx(cfg, args, need_ocr=True):
    from .credits import detect_llm
    from .db import DB
    from .matcher import Matcher
    from .net import Http
    from .ocr import OCR
    from .pipeline import Context

    db = DB(config.db_path(cfg))
    http = Http(db, cfg["cache_days"], offline=getattr(args, "offline", False))
    llm = None
    if need_ocr and getattr(args, "llm", None) is not False:
        lcfg = dict(cfg["llm"])
        if getattr(args, "llm", None) is True:
            lcfg["enabled"] = True
        if lcfg["enabled"] in (True, "auto"):
            llm = detect_llm(http, lcfg)
            if llm:
                log(f"Local LLM: {llm['model']} @ {llm['url']}", 1)
    return Context(cfg, db, http, Matcher(http, cfg, db), lambda: OCR(cfg["ocr"]), llm)


def cmd_scan(cfg, args):
    import time
    from .pipeline import describe, fmt_duration, iter_videos, process_file
    sc = cfg["scan"]
    t_all = time.time()
    for k in ("open_minutes", "end_minutes"):
        v = getattr(args, k)
        if v is not None:
            sc[k] = v
    if args.fps:
        sc["fps_open"] = sc["fps_end"] = args.fps
    if args.cpu:
        cfg["ocr"]["gpu"] = False
    if args.lang:
        from .ocr import resolve_languages
        cfg["ocr"]["languages"] = [l.strip() for l in args.lang.split(",") if l.strip()]
        resolve_languages(cfg["ocr"]["languages"])          # validate early
    if not cfg["tmdb_api_key"]:
        log(ui.warn("no TMDB key - cast and crew come from Wikidata, ČSFD and IMDb's top stars only "
                    "('videoid keys' adds one)"), 1)
    ctx = _ctx(cfg, args, need_ocr=args.ocr)
    results, errors = [], 0
    write_logs = args.log_file and sc.get("log_file", True)
    files = list(iter_videos(args.paths, cfg, recursive=not args.no_recursive))
    if len(files) > 1:
        log(ui.c(f"{len(files)} video files to check", ui.BOLD), 0)
    for i, path in enumerate(files, 1):
        ctx.position = (i, len(files))
        with logmod.capture() as lines:
            fid, worth_logging = _scan_one(ctx, cfg, path, args, describe, process_file)
        if fid:
            results.append(fid)
        elif worth_logging:
            errors += 1
        if write_logs and worth_logging:
            from .filing import write_log
            write_log(Path(ctx.db.file_get(fid)["path"]) if fid else path, lines)   # follows a rename
    if len(results) + errors > 1:
        v = [ctx.db.file_get(f)["verdict"] for f in results]
        sure = sum(x == "CERTAIN" for x in v)
        unsure = sum(x in ("PROBABLE", "POSSIBLE") for x in v)
        parts = [ui.c(f"{sure} identified", ui.GREEN), ui.c(f"{unsure} not sure", ui.YELLOW),
                 ui.c(f"{len(v) - sure - unsure} not identified", ui.RED)]
        if errors:
            parts.append(ui.c(f"{errors} failed", ui.BOLD + ui.RED))
        log("\n" + ui.c(f"=== {len(results) + errors} files in {fmt_duration(time.time() - t_all)}: ", ui.BOLD)
            + ", ".join(parts), 0)
    if args.json:
        Path(args.json).write_text(json.dumps([_file_record(ctx.db, f) for f in results], indent=2,
                                              ensure_ascii=False), encoding="utf-8")
        log(ui.good(f"wrote {args.json}"), 0)
    return 0


def _scan_one(ctx, cfg, path, args, describe, process_file):
    """Identify one movie, print the result, rename it and write Kodi files.
    Returns (database id, whether anything happened that belongs in the movie's log file)."""
    ctx.cached = False
    try:
        fid = process_file(ctx, path, args)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log(ui.bad(f"{path}: {e!r}"), 0)
        if logmod.VERBOSITY >= 2:
            import traceback
            traceback.print_exc()
        return None, True
    if not fid:
        return None, False
    logmod.out(describe(ctx.db, fid, top=args.top))
    changed = not ctx.cached                 # a cached result was logged when it was made
    if args.interactive and _ask_match(ctx, fid):
        changed = True
    if args.rename:
        from .filing import rename_movie
        new, status = rename_movie(ctx.db, fid, ctx.matcher, cfg["rename"], cfg["kodi"].get("prefer"))
        logmod.out(ui.status("rename", status))
        changed = changed or new is not None
    if args.kodi:
        from .kodi import write_for_file
        status = write_for_file(ctx.db, fid, ctx.matcher, cfg["kodi"], overwrite=args.overwrite_nfo)
        logmod.out(ui.status("kodi", status))
        changed = changed or status.startswith("wrote")
    return fid, changed


def _ask_match(ctx, fid, limit=6):
    """scan --interactive: when the result is not CERTAIN, let the user pick the film (or confirm the only one),
    or type the film's name / IMDb id / database link to search again. Returns True when the stored result changed."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    db = ctx.db
    ms = db.matches(fid)
    if ms and ms[0]["verdict"] == "CERTAIN":
        return False
    if ms and json.loads(ms[0]["evidence_json"]).get("rejected_by_user"):
        return False
    row = db.file_get(fid)
    file_min = f", file {row['duration'] / 60:.0f} min" if row["duration"] else ""
    changed = False
    while True:
        ms = db.matches(fid)[:limit]
        if not ms:
            logmod.out(ui.question(f"Not identified{file_min}. Do you know which film it is?"))
        else:
            logmod.out(ui.question(f"Not sure{file_min}. " + ("Which film is it?" if len(ms) > 1 else "Is it this film?")))
            _show_options(db, ms)
        ans = _read_answer(ms)
        if ans == "":
            logmod.out(ui.warn("left as it is"))
            return changed
        if ans.isdigit() and int(ans) <= len(ms):
            break
        if ans.isdigit() and len(ans) < 3:          # "1917" is a film, "7" is a mistyped choice
            logmod.log(ui.warn(f"type a number from 0 to {len(ms)}, a film name, or press Enter"), 0)
            continue
        if _search_typed(ctx, fid, row, ans):
            changed = True
    if ans == "0":
        db.choose_match(fid, None)
        logmod.out(ui.warn("you chose: none of these - marked UNKNOWN"))
        return True
    pick = db.choose_match(fid, int(ans) - 1)
    f = db.film_json(pick["film_key"]) or {}
    logmod.out(ui.good(f"you chose: {f.get('title')} ({f.get('year') or '?'}) - now CERTAIN"))
    return True


def _show_options(db, ms):
    for i, m in enumerate(ms, 1):
        f = Film.from_dict(db.film_json(m["film_key"]) or {})
        ev = json.loads(m["evidence_json"])
        aka = [t for t in f.titles if t != f.title][:2]
        n = len(ev.get("people") or [])
        bits = [f"{f.runtime} min" if f.runtime else None, f"{n} {'person' if n == 1 else 'people'} matched",
                ("also " + " / ".join(aka)) if aka else None]
        logmod.out(ui.option(i, ui.c(f"{f.title} ({f.year or '?'})", ui.BOLD + ui.WHITE)) + "  "
                   + ui.verdict(m["verdict"]) + "  " + ", ".join(b for b in bits if b))
        for k in ("imdb", "csfd", "tmdb"):
            if k in f.urls():
                logmod.out("        " + ui.c(f.urls()[k], ui.BLUE))


def _read_answer(ms):
    if not ms:
        pick = ""
    elif len(ms) == 1:
        pick = "1 = yes, 0 = no, "
    else:
        pick = f"1-{len(ms)} = this film, 0 = none of these, "
    prompt = ui.ask(f"{pick}or type the film's name (or IMDb/CSFD link), Enter = leave as it is:")
    try:
        return input(prompt if ui.color_on(sys.stdout) else ui.plain(prompt)).strip()
    except EOFError:
        return ""


def _parse_typed(text):
    """'Pelíšky 1999' / 'Pelisky (1999)' / 'tt0167116' / 'https://www.csfd.cz/film/5954-pelisky/'
    -> (titles to search, year, ids)."""
    from .localinfo import CSFD_RE, IMDB_RE, TMDB_RE
    ids = {}
    for key, rx in (("imdb", IMDB_RE), ("tmdb", TMDB_RE), ("csfd", CSFD_RE)):
        if m := rx.search(text):
            ids[key] = m.group(1)
    if ids:
        return [], None, ids
    # a trailing year may be part of the title ("Blade Runner 2049"): search both, the year only helps
    titles, year = [text.strip()], None
    if m := re.match(r"^(.*?\S)[\s,]*[(\[]?((?:18|19|20)\d\d)[)\]]?$", text.strip()):
        titles.append(m.group(1).strip(" ,-"))
        year = int(m.group(2))
    return titles, year, ids


def _search_typed(ctx, fid, row, text):
    """Search the databases for what the user typed, keeping the credits read from the video as evidence,
    and store the result as the new candidate list. True when something was found."""
    titles, year, ids = _parse_typed(text)
    people = [p for p in ctx.db.people(fid) if p["source"] in ("ocr", "llm")]
    ev = {"titles": [(t, "user") for t in titles], "years": [(year, "user")] if year else [],
          "ids": ids, "people": people, "runtime": row["duration"] / 60 if row["duration"] else None}
    logmod.out(ui.step(f"Searching for what you typed: {text}"))
    try:
        ranked = ctx.matcher.identify(ev)
    except Exception as e:
        logmod.out(ui.bad(f"search failed: {e}"))
        return False
    if not ranked:
        logmod.out(ui.warn("nothing found - try another spelling, the original title, or an IMDb/CSFD link"))
        return False
    ctx.db.save_matches(fid, [{"film_key": r["film_key"], "score": r["score"],
                               "confidence": r.get("confidence", 0), "verdict": r.get("verdict", "POSSIBLE"),
                               "evidence": dict(r["evidence"], typed_by_user=text)} for r in ranked[:5]])
    b = ranked[0]
    ctx.db.file_upsert_id(fid, film_key=b["film_key"], title=b["film"].title, year=b["film"].year,
                          verdict=b.get("verdict", "POSSIBLE"), confidence=b.get("confidence", 0),
                          ids_json=json.dumps(b["film"].ids))
    return True


def _file_record(db, fid):
    row = db.file_get(fid)
    ms = db.matches(fid)
    best = db.film_json(ms[0]["film_key"]) if ms else None
    return {
        "path": row["path"], "verdict": row["verdict"], "confidence": row["confidence"],
        "film": best, "urls": Film.from_dict(best).urls() if best else {},
        "evidence": json.loads(ms[0]["evidence_json"]) if ms else None,
        "alternatives": [{"film_key": m["film_key"], "score": m["score"], "verdict": m["verdict"]} for m in ms[1:]],
        "ocr_people": db.people(fid),
        "local": json.loads(row["local_json"] or "{}"),
    }


def _find_file(db, ref):
    if ref.isdigit():
        return int(ref)
    p = str(Path(ref).resolve())
    r = db.con.execute("SELECT id FROM files WHERE path=?", (p,)).fetchone()
    if not r:
        r = db.con.execute("SELECT id FROM files WHERE path LIKE ? ORDER BY path LIMIT 1", (f"%{ref}%",)).fetchone()
    return r["id"] if r else None


def cmd_show(cfg, args):
    from .db import DB
    from .pipeline import describe
    db = DB(config.db_path(cfg))
    fid = _find_file(db, args.file)
    if not fid:
        log("not in database", 0)
        return 1
    logmod.out(describe(db, fid, top=args.top))
    if args.ocr:
        print("\n   unique OCR strings:")
        for l in db.ocr_lines(fid):
            print(f"    [{l['segment']:<4} {l['t']:7.1f}s x{l['seen']:<3}] {l['text']}")
    if args.people:
        print("\n   people read from credits:")
        for p in sorted(db.people(fid), key=lambda p: (p["role"], p["name"])):
            print(f"    {p['role']:<20} {p['name']}  ({p['source']}, x{p['seen']})")
    return 0


def cmd_list(cfg, args):
    from .db import DB
    db = DB(config.db_path(cfg))
    for r in db.files():
        if args.verdict and (r["verdict"] or "").upper() != args.verdict.upper():
            continue
        ids = json.loads(r["ids_json"] or "{}")
        print(f"{r['id']:>4}  {(r['verdict'] or '-'):<9} {(r['confidence'] or 0) * 100:5.1f}%  "
              f"{(r['title'] or '?')[:40]:<40} {r['year'] or '':<5} {ids.get('imdb', ''):<11} {r['path']}")
    return 0


def cmd_export(cfg, args):
    from .db import DB
    db = DB(config.db_path(cfg))
    recs = [_file_record(db, r["id"]) for r in db.files() if r["matched_at"]]
    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["path", "verdict", "confidence", "title", "year", "imdb", "tmdb", "csfd", "wikidata",
                        "director", "matched_people"])
            for r in recs:
                f = r["film"] or {}
                ids = f.get("ids", {})
                w.writerow([r["path"], r["verdict"], r["confidence"], f.get("title"), f.get("year"),
                            ids.get("imdb"), ids.get("tmdb"), ids.get("csfd"), ids.get("wikidata"),
                            ", ".join((f.get("credits") or {}).get("director", [])),
                            len((r["evidence"] or {}).get("people", []))])
    else:
        out.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"exported {len(recs)} files to {out}", 0)
    return 0


def _select_files(db, refs):
    """Database file ids for ids, files or folders (everything analysed below a folder)."""
    out = []
    rows = db.files()
    for ref in refs:
        if ref.isdigit() and not Path(ref).exists():
            out.append(int(ref))
            continue
        forms = {str(p).lower().rstrip("\\/") for p in (Path(ref).absolute(), Path(ref).resolve())}
        hit = [r["id"] for r in rows
               if any(r["path"].lower() == f or r["path"].lower().startswith(f + "\\") for f in forms)]
        if not hit:
            log(f"{ref}: nothing analysed there yet (run 'videoid scan' first)", 0)
        out += hit
    return list(dict.fromkeys(out))


def cmd_kodi(cfg, args):
    from .db import DB
    from .kodi import write_for_file
    from .matcher import Matcher
    from .net import Http
    if args.min_verdict:
        cfg["kodi"]["min_verdict"] = args.min_verdict.upper()
    if args.no_artwork:
        cfg["kodi"]["artwork"] = False
    if args.prefer:
        cfg["kodi"]["prefer"] = [s.strip() for s in args.prefer.split(",") if s.strip()]
    db = DB(config.db_path(cfg))
    matcher = Matcher(Http(db, cfg["cache_days"], offline=args.offline), cfg, db)
    fids = _select_files(db, args.paths)
    written = 0
    for fid in fids:
        try:
            status = write_for_file(db, fid, matcher, cfg["kodi"], overwrite=args.overwrite,
                                    dry_run=args.dry_run, refresh=args.refresh)
        except OSError as e:
            status = f"ERROR {e}"
        written += status.startswith(("wrote", "would write"))
        print(f"{db.file_get(fid)['path']}\n   {status}", flush=True)
    log(f"\n{written} of {len(fids)} files got Kodi metadata", 0)
    return 0


def cmd_config(cfg, args):
    if args.key and args.value is not None:
        v = config.set_value(args.key, args.value)
        print(f"{args.key} = {v!r}  (saved to {config.CONFIG_PATH})")
        return 0
    shown = json.loads(json.dumps(cfg))
    if shown.get("tmdb_api_key"):
        shown["tmdb_api_key"] = shown["tmdb_api_key"][:4] + "..."
    if args.key:
        node = shown
        for part in args.key.split("."):
            node = node[part]
        print(json.dumps(node, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(shown, indent=2, ensure_ascii=False))
    return 0


def cmd_doctor(cfg, args):
    import platform
    ok = True
    print(f"VideoID {__version__}  Python {platform.python_version()}  {sys.executable}")
    try:
        from . import media
        print(f"ffmpeg:      {media.ffmpeg_exe()}")
    except Exception as e:
        ok = False
        print(f"ffmpeg:      MISSING ({e})")
    try:
        import onnxruntime as ort
        print(f"onnxruntime: {ort.__version__} providers={ort.get_available_providers()}")
    except Exception as e:
        ok = False
        print(f"onnxruntime: MISSING ({e})")
    try:
        import time
        import numpy as np
        from .ocr import OCR
        o = OCR(cfg["ocr"])
        img = np.zeros((720, 1280, 3), np.uint8)
        import cv2
        cv2.putText(img, "Directed by Test Person", (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        o.run(img)
        t = time.time()
        res = o.run(img)
        print(f"OCR:         {o.device}, {1000 * (time.time() - t):.0f} ms/frame, read: {[b['text'] for b in res]}")
    except Exception as e:
        ok = False
        print(f"OCR:         FAILED ({e!r})")
    from .credits import detect_llm
    from .db import DB
    from .net import Http
    db = DB(config.db_path(cfg))
    http = Http(db, cfg["cache_days"])
    llm = detect_llm(http, cfg["llm"])
    print(f"local LLM:   {llm['model'] + ' @ ' + llm['url'] if llm else 'none (optional: Ollama / LM Studio)'}")
    print(f"TMDB key:    {'set' if cfg['tmdb_api_key'] else 'not set (optional, run: videoid keys)'}")
    from .sources import csfd as csfd_src
    print(f"CSFD:        {'node-csfd-api via ' + str(csfd_src.NODE) if csfd_src.available() else 'unavailable (run setup.cmd)'}")
    n = db.con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    nf = db.con.execute("SELECT COUNT(*) FROM films").fetchone()[0]
    print(f"database:    {config.db_path(cfg)}  ({n} files, {nf} films)")
    return 0 if ok else 1


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(prog="videoid", description="Identify movies from file names, NFOs and OCR'd credits.")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="analyse video files / folders")
    s.add_argument("paths", nargs="+")
    s.add_argument("--force", action="store_true", help="redo OCR even if stored")
    s.add_argument("--rematch", action="store_true", help="reuse stored OCR, redo database matching")
    s.add_argument("--reparse", action="store_true",
                   help="re-run credits parsing on stored OCR frames (no OCR), then match")
    s.add_argument("--no-nfo", action="store_true", help="ignore NFO files (testing credit-only identification)")
    s.add_argument("--no-ocr", dest="ocr", action="store_false", help="only file name / NFO")
    s.add_argument("--llm", dest="llm", action="store_true", default=None, help="force local LLM")
    s.add_argument("--no-llm", dest="llm", action="store_false", help="disable local LLM")
    s.add_argument("--cpu", action="store_true", help="run OCR on CPU")
    s.add_argument("--lang", help="OCR languages, comma separated, e.g. latin,hi,ja,ko,zh,ta,te,th,ar,ru "
                                  "(default from config ocr.languages)")
    s.add_argument("--open", dest="open_minutes", type=float, help="minutes of opening titles to scan")
    s.add_argument("--end", dest="end_minutes", type=float, help="minutes of end credits to scan")
    s.add_argument("--fps", type=float, help="frames read per second (default 0.5 = one every 2 s; 1 for very fast credits)")
    s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--all-sizes", action="store_true", help="include samples and small files")
    s.add_argument("--offline", action="store_true", help="use only cached web responses")
    s.add_argument("--top", type=int, default=3, help="candidates to print")
    s.add_argument("--json", help="also write results to this JSON file")
    s.add_argument("--kodi", action="store_true", help="also write Kodi .nfo + poster/fanart for identified files")
    s.add_argument("--overwrite-nfo", action="store_true", help="with --kodi: replace NFOs not written by VideoID")
    s.add_argument("-i", "--interactive", action="store_true",
                   help="when a film is not CERTAIN, ask which film it is (only CERTAIN films get Kodi files)")
    s.add_argument("--rename", action="store_true",
                   help="rename identified movies (and their subtitles, NFO, artwork) to 'Title (Year)'")
    s.add_argument("--no-log", dest="log_file", action="store_false",
                   help="do not write <movie>.log next to each movie")
    s.add_argument("--no-subs", dest="subs", action="store_false",
                   help="don't try to identify films from their subtitles first")
    s.add_argument("--no-quick", dest="quick_nfo", action="store_false",
                   help="always do the full scan, even when an NFO names the film")

    s = sub.add_parser("show", help="show stored result for a file (path, part of path or id)")
    s.add_argument("file")
    s.add_argument("--ocr", action="store_true", help="print unique OCR strings")
    s.add_argument("--people", action="store_true", help="print people read from credits")
    s.add_argument("--top", type=int, default=5)

    s = sub.add_parser("list", help="list all files in the database")
    s.add_argument("--verdict", help="filter: CERTAIN, PROBABLE, POSSIBLE, UNKNOWN")

    s = sub.add_parser("export", help="export database to .json or .csv")
    s.add_argument("output")

    s = sub.add_parser("kodi", aliases=["nfo"], help="write Kodi .nfo + poster/fanart next to identified files")
    s.add_argument("paths", nargs="+", help="files, folders (everything analysed below) or database ids")
    s.add_argument("--min-verdict", help="CERTAIN (default), PROBABLE or POSSIBLE")
    s.add_argument("--prefer", help="metadata source order, e.g. csfd,tmdb (default from config kodi.prefer)")
    s.add_argument("--no-artwork", action="store_true", help="NFO only, no poster/fanart download")
    s.add_argument("--overwrite", action="store_true",
                   help="replace NFOs/artwork not written by VideoID (old NFO kept as .nfo.bak)")
    s.add_argument("--refresh", action="store_true", help="re-fetch plot/genres/artwork from TMDB/CSFD")
    s.add_argument("--dry-run", action="store_true", help="print the NFO, write nothing")
    s.add_argument("--offline", action="store_true", help="use only cached web responses")

    s = sub.add_parser("config", help="show or set configuration, e.g. config tmdb_api_key XXXX")
    s.add_argument("key", nargs="?")
    s.add_argument("value", nargs="?")

    sub.add_parser("doctor", help="check ffmpeg, GPU OCR, LLM and database")
    sub.add_parser("keys", help="enter / change API keys (also re-asks skipped ones)")

    # "scan" is the default command: `videoid D:\Movies --kodi` == `videoid scan D:\Movies --kodi`
    argv = _repair_argv(sys.argv[1:]) if argv is None else list(argv)
    first = next((i for i, a in enumerate(argv) if not a.startswith("-")), None)
    if first is not None and argv[first] not in sub.choices:
        argv.insert(first, "scan")
    args = ap.parse_args(argv)
    logmod.VERBOSITY = 2 if args.verbose else 0 if args.quiet else 1
    cfg = config.load()
    if args.cmd == "keys":
        config.ask_keys(cfg, force=True)
        return 0
    if args.cmd in ("scan", "doctor") and not getattr(args, "offline", False):
        cfg = config.ask_keys(cfg)            # first start: ask for missing keys, skippable if non-essential
    return {"scan": cmd_scan, "kodi": cmd_kodi, "nfo": cmd_kodi, "show": cmd_show, "list": cmd_list, "export": cmd_export,
            "config": cmd_config, "doctor": cmd_doctor}[args.cmd](cfg, args)

"""Evaluate VideoID against a scrambled test set.

The mapping file (from the scrambling step) holds: type<TAB>original rel path<TAB>scrambled rel path.
Original video names end with " - <tmdb id>" or " - csfd<csfd id>", which is the ground truth.

    runtime\\python\\python.exe tests\\evaluate.py --limit 2
    runtime\\python\\python.exe tests\\evaluate.py --start 2 --limit 20 -- --rematch
    runtime\\python\\python.exe tests\\evaluate.py --no-scan          # only score what is in the DB
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from videoid import config  # noqa: E402
from videoid.db import DB  # noqa: E402

VIDEO = {".mp4", ".mkv", ".avi", ".m4v", ".mov", ".wmv", ".ts", ".m2ts", ".mpg", ".mpeg", ".webm", ".braw"}
TRUTH_RE = re.compile(r" - (csfd)?(\d+)$")


def load_cases(mapping, root):
    cases = []
    for line in Path(mapping).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        typ, orig, scr = line.split("\t")
        if typ != "F" or Path(orig).suffix.lower() not in VIDEO:
            continue
        m = TRUTH_RE.search(Path(orig).stem)
        truth = ({"csfd": m.group(2)} if m.group(1) else {"tmdb": m.group(2)}) if m else {}
        cases.append({"orig": orig, "path": str(Path(root) / scr), "truth": truth})
    cases.sort(key=lambda c: c["orig"].lower())
    return cases


def judge(truth, ids, wd):
    if not truth:
        return "no-truth"
    if not ids:
        return "NO MATCH"
    k, v = next(iter(truth.items()))
    if k in ids:
        return "OK" if str(ids[k]) == v else "WRONG"
    # result lacks that database's id - cross-check through Wikidata (tmdb P4947 / csfd P2529)
    from videoid.film import Film
    f = Film(ids={k: v})
    wd.resolve([f])
    q = f.ids.get("wikidata")
    if q:
        if ids.get("wikidata"):
            return "OK" if ids["wikidata"] == q else "WRONG"
        full = wd.fetch([q])
        if full:
            for key in ("imdb", "tmdb", "csfd"):
                if key in full[0].ids and key in ids:
                    return "OK" if full[0].ids[key] == str(ids[key]) else "WRONG"
    return "unverifiable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True, help="mapping file written by the scrambling step")
    ap.add_argument("--root", required=True, help="folder with the scrambled files")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--report", help="write per-file results as JSON")
    ap.add_argument("scan_args", nargs="*", help="extra args for 'videoid scan' (after --)")
    a = ap.parse_args()

    cases = load_cases(a.mapping, a.root)[a.start:]
    if a.limit:
        cases = cases[:a.limit]
    if not a.no_scan:
        for c in cases:
            cmd = [sys.executable, "-m", "videoid", "scan", c["path"], "--all-sizes", *a.scan_args]
            subprocess.run(cmd, cwd=ROOT)

    from videoid.net import Http
    from videoid.sources.wikidata import Wikidata
    cfg = config.load()
    db = DB(config.db_path(cfg))
    wd = Wikidata(Http(db, cfg["cache_days"]))
    rows, counts = [], {}
    for c in cases:
        r = db.con.execute("SELECT * FROM files WHERE path=?", (c["path"],)).fetchone()
        ids = json.loads(r["ids_json"] or "{}") if r else {}
        res = judge(c["truth"], ids, wd) if r else "not scanned"
        if res == "unverifiable" and r["title"]:
            # last resort: original folder name "YEAR - Title" vs found title/year
            from videoid.textutil import title_sim
            film = db.film_json(r["film_key"]) or {}
            m = re.match(r"_?(\d{4}) - (.+)$", Path(c["orig"]).parts[0])
            if m:
                sim = max((title_sim(m.group(2), t) for t in film.get("titles") or [r["title"]]), default=0)
                yok = not r["year"] or abs(int(r["year"]) - int(m.group(1))) <= 1
                res = "OK(title)" if sim >= 90 and yok else "WRONG(title)" if sim < 60 else res
        counts[res] = counts.get(res, 0) + 1
        rows.append({**c, "result": res, "verdict": r["verdict"] if r else None,
                     "confidence": r["confidence"] if r else None,
                     "found": f"{r['title']} ({r['year']})" if r and r["title"] else None, "ids": ids})

    print("\n" + "=" * 100)
    for x in rows:
        exp = ",".join(f"{k}:{v}" for k, v in x["truth"].items())
        conf = f"{(x['confidence'] or 0) * 100:5.1f}%" if x["confidence"] is not None else "     "
        print(f"{x['result']:<13} {x['verdict'] or '-':<9} {conf}  expected {exp:<16} found {x['found'] or '-'}")
        print(f"{'':>13} {x['orig']}")
    print("-" * 100)
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())), f"  (total {len(rows)})")
    certain_wrong = [x for x in rows if x["verdict"] == "CERTAIN" and x["result"] == "WRONG"]
    if certain_wrong:
        print(f"!! {len(certain_wrong)} CERTAIN results are WRONG")
    if a.report:
        Path(a.report).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

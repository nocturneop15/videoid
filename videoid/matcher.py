"""Candidate generation across databases, cross-linking, enrichment and evidence scoring."""
import math

from rapidfuzz import fuzz, process

from .credits import ROLE_PRIORITY, is_verified_name
from .film import Film, merge_all
from .log import log
from .sources.csfd import CSFD
from .sources.imdb import IMDb
from .sources.tmdb import TMDB
from .sources.wikidata import Wikidata
from .textutil import CASED, compact, norm, script_of, title_sim

ROLE_WEIGHT = {"director": 10, "composer": 6, "dop": 4, "writer": 4, "editor": 3, "producer": 3,
               "exec_producer": 2, "production_designer": 2.5, "costume": 2, "casting": 2, "sound": 1.5,
               "dramaturg": 2, "cast": 2, "crew": 1}
KEY_ROLES = {"director", "composer", "dop", "writer", "editor"}


class Matcher:
    def __init__(self, http, cfg, db):
        self.db = db
        s = cfg["sources"]
        from .ocr import wikidata_langs
        self.wd = Wikidata(http, wikidata_langs(cfg["ocr"].get("languages"))) if s.get("wikidata", True) else None
        tm = TMDB(http, cfg.get("tmdb_api_key"), cfg.get("tmdb_language", "en-US"))
        self.tmdb = tm if (s.get("tmdb", True) and tm.enabled) else None
        self.imdb = IMDb(http) if s.get("imdb", True) else None
        self.csfd = CSFD(http) if s.get("csfd", True) else None

    # ------------------------------------------------------------------ public
    def identify(self, ev):
        """ev: dict(titles=[(t,src)], years=[(y,src)], ids={}, people=[{name,role,..}], runtime=min)"""
        cands = self._candidates(ev)
        log(f"  {len(cands)} candidate films", 1)
        if not cands:
            return []
        cands = self._enrich(cands, ev)
        ranked = self._rank(cands, ev)
        if ranked:
            best = ranked[0]["film"]
            self._finalize(best, ev)              # fill missing ids (CSFD/TMDB/IMDb) for the winner
            ranked[0].update(self.score(best, ev))
            ranked.sort(key=lambda r: -r["score"])
            self._verdicts(ranked, ev)
        for r in ranked:
            r["film_key"] = self.db.save_film(r["film"])
        return ranked

    # ---------------------------------------------------------- candidates
    def _queries(self, ev):
        qs, seen = [], set()
        for t, src in ev["titles"]:
            k = norm(t)
            if k and k not in seen and len(k) >= 2:
                seen.add(k)
                qs.append((t, src))
        return qs[:6]

    def _best_year(self, ev):
        for y, src in ev["years"]:
            if src in ("nfo", "filename", "folder", "embedded"):
                return y
        return ev["years"][0][0] if ev["years"] else None

    def _key_people(self, ev, limit=15):
        prio = {r: i for i, r in enumerate(ROLE_PRIORITY)}
        # names that start with a known first name first, then by role importance and how often seen
        ppl = sorted(ev["people"], key=lambda p: (not is_verified_name(p["name"]), prio.get(p["role"], 99),
                                                  -p.get("seen", 1)))
        out, seen = [], []
        for p in ppl:
            k = compact(p["name"])
            single_ok = script_of(p["name"]) not in CASED          # "张艺谋" is one token
            if (len(p["name"].split()) < 2 and not single_ok) or \
                    process.extractOne(k, seen, scorer=fuzz.ratio, score_cutoff=85):
                continue      # OCR variant of a name already chosen ("Jirf Šust" ~ "Jiří Šust")
            seen.append(k)
            out.append(p["name"])
            if len(out) >= limit:
                break
        return out

    def _candidates(self, ev):
        films = []
        ids = ev.get("ids") or {}
        # 1) explicit ids from NFO
        if ids:
            log(f"  NFO ids: {ids}", 1)
            films.append(Film(ids=ids, source="nfo"))
        year = self._best_year(ev)
        # 2) title searches
        for q, src in self._queries(ev):
            log(f"  searching title '{q}' ({src})", 1)
            for s in (self.wd, self.tmdb, self.imdb, self.csfd):
                if not s:
                    continue
                try:
                    if s is self.wd:
                        films += [Film(ids={"wikidata": x}, source="wikidata") for x in s.search_films(q)]
                    else:
                        films += s.search(q, year)
                except Exception as e:  # one broken source must not stop the others
                    log(f"    {s.name} search failed: {e}", 1)
        # 3) reverse lookup: films shared by the people seen in the credits
        names = self._key_people(ev)
        if len(names) >= 2:
            log(f"  reverse lookup by {len(names)} people: {', '.join(names[:6])}...", 1)
            if self.wd:
                try:
                    for q, n in self.wd.films_by_people(names):
                        f = Film(ids={"wikidata": q}, source="wikidata")
                        f.hits = n
                        films.append(f)
                except Exception as e:
                    log(f"    wikidata people lookup failed: {e}", 1)
            if self.tmdb:
                try:
                    for mid, n in self.tmdb.films_by_people(names):
                        f = Film(ids={"tmdb": mid}, source="tmdb")
                        f.hits = n
                        films.append(f)
                except Exception as e:
                    log(f"    tmdb people lookup failed: {e}", 1)
            if self.csfd and not self.csfd.blocked:
                try:
                    verified = [n for n in names if is_verified_name(n)] or names
                    for cid, n in self.csfd.films_by_people(verified):
                        f = Film(ids={"csfd": cid}, source="csfd")
                        f.hits = n
                        films.append(f)
                except Exception as e:
                    log(f"    csfd people lookup failed: {e}", 1)
        return merge_all(films)

    # ------------------------------------------------------------- enrich
    def _prelim(self, f, ev):
        s = f.hits * 8
        best = max((title_sim(t, ft) for t, _ in ev["titles"] for ft in f.titles), default=0)
        s += 10 if best >= 90 else 4 if best >= 80 else 0
        yrs = {y for y, _ in ev["years"]}
        if f.year and yrs and min(abs(f.year - y) for y in yrs) <= 1:
            s += 5
        if any(f.ids.get(k) == v for k, v in (ev.get("ids") or {}).items()):
            s += 50
        return s

    def _enrich(self, cands, ev):
        # Wikidata links everything: resolve QIDs, then fetch full records with credits
        if self.wd:
            try:
                self.wd.resolve(cands)
                cands = merge_all(cands)
                cands = merge_all(cands + self.wd.fetch([c.ids["wikidata"] for c in cands if "wikidata" in c.ids]))
            except Exception as e:
                log(f"    wikidata enrich failed: {e}", 1)
        cands.sort(key=lambda f: -(self._prelim(f, ev) + self._quick_people(f, ev)))
        extra = []
        for i, f in enumerate(cands[:12]):
            extra += self._fetch_more(f, deep=i < 6)
        cands = merge_all(cands + extra)
        return self._merge_lookalikes(cands)

    @staticmethod
    def _merge_lookalikes(cands):
        """Records from databases that share no id (e.g. CSFD vs IMDb for a film missing in Wikidata):
        same title + year +-1 + at least one shared person = same film."""
        out = []
        for f in cands:
            twin = None
            for o in out:
                if any(k in o.ids and k in f.ids and o.ids[k] != f.ids[k] for k in ("imdb", "tmdb", "csfd")):
                    continue        # conflicting ids - different films
                if f.year and o.year and abs(f.year - o.year) > 1:
                    continue
                if not any(title_sim(a, b) >= 95 for a in f.titles for b in o.titles):
                    continue
                pf = {c for c, *_ in f.people_index()}
                po = {c for c, *_ in o.people_index()}
                if pf & po:
                    twin = o
                    break
            if twin:
                twin.merge(f)
            else:
                out.append(f)
        return out

    def _fetch_more(self, f, deep=False):
        out = []
        try:
            if self.tmdb:
                tid = f.ids.get("tmdb") or (self.tmdb.find_imdb(f.ids["imdb"]) if "imdb" in f.ids else None)
                if tid:
                    out.append(self.tmdb.fetch(tid))
            if self.csfd and "csfd" in f.ids and (deep or f.n_people() < 10):
                out.append(self.csfd.fetch(f.ids["csfd"]))
            if self.imdb and "imdb" in f.ids and (deep or f.n_people() < 10) and not self.tmdb:
                out.append(self.imdb.fetch(f.ids["imdb"]))
        except Exception as e:
            log(f"    enrich {f} failed: {e}", 1)
        return [x for x in out if x]

    def _finalize(self, f, ev):
        """Make sure the winner carries ids for every database."""
        if self.wd and "wikidata" not in f.ids:
            self.wd.resolve([f])
            if "wikidata" in f.ids:
                for x in self.wd.fetch([f.ids["wikidata"]]):
                    f.merge(x)
        if self.csfd and "csfd" not in f.ids:
            for t in f.titles[:3]:
                hits = [h for h in self.csfd.search(t, f.year) if not f.year or not h.year or abs(h.year - f.year) <= 1]
                for h in hits[:2]:
                    full = self.csfd.fetch(h.ids["csfd"])
                    if full and (full.ids.get("imdb") == f.ids.get("imdb") or self._same_people(f, full)):
                        f.merge(full)
                        break
                if "csfd" in f.ids:
                    break
        if self.imdb and "imdb" in f.ids and f.n_people() < 10:
            full = self.imdb.fetch(f.ids["imdb"])
            if full:
                f.merge(full)
        if self.tmdb and "tmdb" not in f.ids and "imdb" in f.ids:
            tid = self.tmdb.find_imdb(f.ids["imdb"])
            if tid and (full := self.tmdb.fetch(tid)):
                f.merge(full)

    @staticmethod
    def _same_people(a, b):
        da = {compact(n) for n in a.credits.get("director", [])}
        db = {compact(n) for n in b.credits.get("director", [])}
        return bool(da & db)

    # -------------------------------------------------------------- scoring
    def _quick_people(self, f, ev):
        idx = {c for c, *_ in f.people_index()}
        return sum(2 for p in ev["people"] if compact(p["name"]) in idx)

    def score(self, f, ev):
        s = 0.0
        det = {"people": [], "title": None, "year": None, "runtime": None, "ids": []}
        for k, v in (ev.get("ids") or {}).items():
            if f.ids.get(k) == str(v):
                s += 40
                det["ids"].append(k)
            elif k in f.ids:
                s -= 15
        best_t, best_pair = 0, None
        for t, src in ev["titles"]:
            for ft in f.titles:
                sim = title_sim(t, ft)
                if sim > best_t:
                    best_t, best_pair = sim, (t, ft, src)
        if best_t >= 92:
            s += 12
        elif best_t >= 82:
            s += 6
        det["title"] = {"sim": round(best_t), "pair": best_pair}
        yrs = [(y, src) for y, src in ev["years"]]
        if f.year and yrs:
            d, src = min((abs(f.year - y), src) for y, src in yrs)
            s += 6 if d == 0 else 3 if d == 1 else (-6 if d > 2 and src in ("nfo", "filename") else 0)
            det["year"] = {"film": f.year, "diff": d, "src": src}
        if f.runtime and ev.get("runtime"):
            d = abs(f.runtime - ev["runtime"])
            s += 4 if d <= 3 else 2 if d <= 8 else -4 if d > 25 else 0
            det["runtime"] = {"film": f.runtime, "file": round(ev["runtime"]), "diff": round(d)}

        idx = f.people_index()
        compacts = [x[0] for x in idx]
        norms = [x[1] for x in idx]
        used = set()
        for p in ev["people"]:
            pc, pn = compact(p["name"]), norm(p["name"])
            uncased = script_of(p["name"]) not in CASED      # CJK names are 2-4 characters
            if len(pc) < (2 if uncased else 5):
                continue
            cut = (100 if len(pc) <= 3 else 85) if uncased else 88
            hit = process.extractOne(pc, compacts, scorer=fuzz.ratio, score_cutoff=cut)
            if not hit and not uncased:
                hit = process.extractOne(pn, norms, scorer=fuzz.token_sort_ratio, score_cutoff=90)
            if not hit:
                continue
            _, _, fname, frole, order = idx[hit[2]]
            if (fname, frole) in used:                 # same person reached via an alias
                continue
            used.add((fname, frole))
            w = ROLE_WEIGHT.get(frole, 1)
            if frole == "cast":
                w = 3 if order < 10 else 1.5
            if p["role"] == frole:
                w *= 1.25
            s += w
            det["people"].append({"ocr": p["name"], "db": fname, "role": frole, "sim": round(hit[1]),
                                  "src": p.get("source", "ocr")})
        return {"score": round(s, 2), "evidence": det}

    def _rank(self, cands, ev):
        ranked = []
        for f in cands:
            r = self.score(f, ev)
            r["film"] = f
            ranked.append(r)
        ranked.sort(key=lambda r: -r["score"])
        return ranked[:10]

    def _verdicts(self, ranked, ev):
        s2 = ranked[1]["score"] if len(ranked) > 1 else 0
        for i, r in enumerate(ranked):
            e = r["evidence"]
            ppl = e["people"]
            key = {p["role"] for p in ppl} & KEY_ROLES
            dir_hit = any(p["role"] == "director" for p in ppl)
            n = len(ppl)
            exact_ty = (e["title"]["sim"] >= 92 and e["year"] and e["year"]["diff"] == 0)
            strong = ((dir_hit and n >= 3) or n >= 5 or (len(key) >= 2 and n >= 3)
                      or (e["ids"] and (n >= 2 or exact_ty)))
            s1 = r["score"]
            other = s2 if i == 0 else ranked[0]["score"]
            margin = s1 - other
            if i == 0 and strong and (other < 0.6 * s1 or margin >= 15):
                v = "CERTAIN"
            elif i == 0 and s1 >= 18 and margin >= 6:
                v = "PROBABLE"
            elif s1 > 0:
                v = "POSSIBLE"
            else:
                v = "UNLIKELY"
            conf = 1 - math.exp(-max(s1, 0) / 20)
            if i == 0 and s1 > 0:
                conf *= min(1.0, 0.5 + margin / max(s1, 1))
            else:
                conf *= 0.5
            if v == "CERTAIN":
                conf = max(conf, 0.99)
            r["verdict"], r["confidence"] = v, round(min(conf, 0.999), 3)
            if i == 0:
                e["warnings"] = self._warnings(r["film"], ev)

    @staticmethod
    def _warnings(f, ev):
        w = []
        for k, v in (ev.get("ids") or {}).items():
            if k in f.ids and f.ids[k] != str(v):
                w.append(f"NFO says {k} {v}, but the credits point to {k} {f.ids[k]}")
        named = [t for t, src in ev["titles"] if src in ("filename", "nfo", "embedded")]
        if named and f.titles and max(title_sim(t, ft) for t in named for ft in f.titles) < 60 and ev["people"]:
            w.append(f"file/NFO title {named[0]!r} does not match the identified film")
        yrs = [y for y, src in ev["years"] if src in ("filename", "nfo")]
        if f.year and yrs and min(abs(f.year - y) for y in yrs) > 2:
            w.append(f"year in file name/NFO {yrs[0]} differs from film year {f.year}")
        return w

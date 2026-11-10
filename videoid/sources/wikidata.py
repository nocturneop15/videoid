"""Wikidata: free, no key, and the hub that links IMDb / TMDB / CSFD identifiers together."""
from collections import defaultdict

from ..film import Film
from ..log import log

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"

ROLE_PROPS = {
    "P57": "director", "P162": "producer", "P1431": "exec_producer", "P86": "composer", "P161": "cast",
    "P725": "cast", "P58": "writer", "P344": "dop", "P1040": "editor", "P2554": "production_designer",
    "P2515": "costume", "P1809": "crew", "P3092": "crew", "P4608": "crew",
}
EXT_PROPS = {"imdb": "P345", "tmdb": "P4947", "csfd": "P2529"}
LABEL_LANGS = "en,mul,cs,sk,de,fr,it,es,pl,hu,ru,ja"


def _lit(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


class Wikidata:
    name = "wikidata"

    def __init__(self, http, extra_langs=()):
        self.http = http
        # native-script labels for OCR languages (Hindi, Chinese, ...) are fetched as name aliases
        self.langs = list(dict.fromkeys(["en", "mul", "cs", "sk", "de", "fr", *extra_langs]))

    def sparql(self, q):
        r = self.http.get(SPARQL, params={"query": q, "format": "json"},
                          headers={"Accept": "application/sparql-results+json"})
        return (r or {}).get("results", {}).get("bindings", [])

    def _search(self, query, limit):
        r = self.http.get(API, params={"action": "query", "list": "search", "srsearch": query,
                                       "srlimit": limit, "srnamespace": 0, "format": "json"})
        return [h["title"] for h in (r or {}).get("query", {}).get("search", [])]

    # ----- candidate generation -----
    def search_films(self, title, limit=8):
        # has an IMDb title id == film / series / episode
        return self._search(f"{title} haswbstatement:P345 -haswbstatement:P31=Q5", limit)   # not people

    def search_people(self, name, limit=3):
        return self._search(f"{name} haswbstatement:P31=Q5 haswbstatement:P345", limit)

    def films_by_people(self, names, min_hits=2):
        persons = {}
        for n in names:
            for q in self.search_people(n):
                persons.setdefault(q, n)
        if not persons:
            return []
        props = " ".join(f"wdt:{p}" for p in ROLE_PROPS)
        vals = " ".join(f"wd:{q}" for q in persons)
        rows = self.sparql(f"""
            SELECT ?film (COUNT(DISTINCT ?p) AS ?n) WHERE {{
              VALUES ?p {{ {vals} }} VALUES ?prop {{ {props} }}
              ?film ?prop ?p . ?film wdt:P345 ?imdb .
            }} GROUP BY ?film HAVING(COUNT(DISTINCT ?p) >= {min_hits}) ORDER BY DESC(?n) LIMIT 40""")
        out = []
        for r in rows:
            q = r["film"]["value"].rsplit("/", 1)[-1]
            out.append((q, int(r["n"]["value"])))
        return out

    def resolve(self, films):
        """Attach Wikidata QIDs to films that only have imdb/tmdb/csfd ids (one batched query)."""
        pairs = []
        for f in films:
            if "wikidata" in f.ids:
                continue
            for k, p in EXT_PROPS.items():
                if k in f.ids:
                    pairs.append((p, f.ids[k]))
        if not pairs:
            return {}
        union = " UNION ".join(f"{{ ?f wdt:{p} {_lit(v)} . BIND({_lit(p + ':' + v)} AS ?k) }}" for p, v in pairs[:150])
        rows = self.sparql(f"SELECT ?f ?k WHERE {{ {union} }}")
        out = {}
        for r in rows:
            p, v = r["k"]["value"].split(":", 1)
            out[(p, v)] = r["f"]["value"].rsplit("/", 1)[-1]
        for f in films:
            for k, p in EXT_PROPS.items():
                if k in f.ids and (p, f.ids[k]) in out:
                    f.ids.setdefault("wikidata", out[(p, f.ids[k])])
        return out

    # ----- full records -----
    def fetch(self, qids):
        qids = [q for q in dict.fromkeys(qids) if q and q.startswith("Q")]
        films = {}
        for i in range(0, len(qids), 40):
            films.update(self._fetch(qids[i:i + 40]))
        return list(films.values())

    def _fetch(self, qids):
        vals = " ".join(f"wd:{q}" for q in qids)
        films = {q: Film(ids={"wikidata": q}, source="wikidata") for q in qids}
        rows = self.sparql(f"""
            SELECT ?film ?imdb ?tmdb ?csfd ?date ?dur ?title ?lbl WHERE {{
              VALUES ?film {{ {vals} }}
              OPTIONAL {{ ?film wdt:P345 ?imdb }} OPTIONAL {{ ?film wdt:P4947 ?tmdb }}
              OPTIONAL {{ ?film wdt:P2529 ?csfd }} OPTIONAL {{ ?film wdt:P577 ?date }}
              OPTIONAL {{ ?film wdt:P2047 ?dur }} OPTIONAL {{ ?film wdt:P1476 ?title }}
              OPTIONAL {{ ?film rdfs:label ?lbl FILTER(LANG(?lbl) IN ({", ".join(_lit(l) for l in self.langs)})) }}
            }}""")
        years = defaultdict(list)
        for r in rows:
            q = r["film"]["value"].rsplit("/", 1)[-1]
            f = films[q]
            for k in ("imdb", "tmdb", "csfd"):
                if k in r:
                    f.ids.setdefault(k, r[k]["value"])
            for k in ("title", "lbl"):
                if k in r:
                    f.add_title(r[k]["value"])
            if "date" in r:
                try:
                    years[q].append(int(r["date"]["value"][:4]))
                except ValueError:
                    pass
            if "dur" in r and not f.runtime:
                try:
                    f.runtime = int(float(r["dur"]["value"]))
                except ValueError:
                    pass
        for q, ys in years.items():
            films[q].year = min(ys)   # earliest release = production year
        props = " ".join(f"wdt:{p}" for p in ROLE_PROPS)
        langs = ", ".join(_lit(l) for l in self.langs)
        rows = self.sparql(f"""
            SELECT ?film ?prop ?person ?lbl WHERE {{
              VALUES ?film {{ {vals} }} VALUES ?prop {{ {props} }}
              ?film ?prop ?person .
              ?person rdfs:label ?lbl FILTER(LANG(?lbl) IN ({langs}))
            }}""")
        # one row per label language: first label (by our language order) is canonical, others are aliases
        rank = {l: i for i, l in enumerate(self.langs)}
        people = {}
        for r in rows:
            q = r["film"]["value"].rsplit("/", 1)[-1]
            prop = r["prop"]["value"].rsplit("/", 1)[-1]
            pid = r["person"]["value"]
            lbl = r["lbl"]
            people.setdefault((q, prop, pid), []).append((rank.get(lbl.get("xml:lang"), 99), lbl["value"]))
        for (q, prop, pid), labels in people.items():
            labels.sort()
            name = labels[0][1]
            role = ROLE_PROPS.get(prop, "crew")
            films[q].add_person(name, role)
            for _, alias in labels[1:]:
                films[q].add_alias(name, alias)
        log(f"    wikidata: fetched {len(qids)} films", 2)
        return films

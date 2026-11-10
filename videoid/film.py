"""Film record merged from several databases."""
from .textutil import compact, norm

ID_ORDER = ("imdb", "wikidata", "tmdb", "csfd")
URLS = {
    "imdb": "https://www.imdb.com/title/{}/",
    "tmdb": "https://www.themoviedb.org/movie/{}",
    "csfd": "https://www.csfd.cz/film/{}/",
    "wikidata": "https://www.wikidata.org/wiki/{}",
}


class Film:
    def __init__(self, ids=None, title=None, titles=None, year=None, runtime=None, credits=None, source=None):
        self.ids = {k: str(v) for k, v in (ids or {}).items() if v}
        self.title = title
        self.titles = []
        for t in ([title] if title else []) + list(titles or []):
            self.add_title(t)
        self.year = year
        self.runtime = runtime
        self.credits = {}
        for role, names in (credits or {}).items():
            for n in names:
                self.add_person(n, role)
        self.sources = {source} if source else set()
        self.hits = 0            # how many reverse-lookup people pointed to this film
        self.aliases = {}        # canonical name -> other spellings/scripts ("Zhang Yimou" -> ["张艺谋"])

    def add_alias(self, name, alias):
        if alias and compact(alias) != compact(name):
            lst = self.aliases.setdefault(name, [])
            if all(compact(a) != compact(alias) for a in lst):
                lst.append(alias)

    def key(self):
        for k in ID_ORDER:
            if k in self.ids:
                return f"{k}:{self.ids[k]}"
        return f"title:{norm(self.title or '')}:{self.year or ''}"

    def add_title(self, t):
        t = (t or "").strip()
        if not t or t.lower() in (x.lower() for x in self.titles):
            return
        # same title without diacritics ("Haskovy povidky") -> keep the accented spelling
        for i, x in enumerate(self.titles):
            if norm(x) == norm(t):
                if sum(ord(c) > 127 for c in t) > sum(ord(c) > 127 for c in x):
                    self.titles[i] = t
                    if self.title == x:
                        self.title = t
                return
        self.titles.append(t)
        if not self.title:
            self.title = t

    def add_person(self, name, role):
        name = (name or "").strip()
        if not name:
            return
        lst = self.credits.setdefault(role, [])
        c = compact(name)
        if all(compact(x) != c for x in lst):
            lst.append(name)

    def shares_id(self, other):
        return any(k in other.ids and other.ids[k] == v for k, v in self.ids.items())

    def merge(self, o):
        for k, v in o.ids.items():
            self.ids.setdefault(k, v)
        for t in o.titles:
            self.add_title(t)
        self.year = self.year or o.year
        self.runtime = self.runtime or o.runtime
        for role, names in o.credits.items():
            for n in names:
                self.add_person(n, role)
        self.sources |= o.sources
        self.hits = max(self.hits, o.hits)
        for n, al in o.aliases.items():
            for a in al:
                self.add_alias(n, a)
        return self

    def n_people(self):
        return sum(len(v) for v in self.credits.values())

    def people_index(self):
        """[(compact, norm, name, role, order)] - used for fuzzy matching. Aliases point to the canonical name."""
        out = []
        for role, names in self.credits.items():
            for i, n in enumerate(names):
                out.append((compact(n), norm(n), n, role, i))
                for a in self.aliases.get(n, []):
                    out.append((compact(a), norm(a), n, role, i))
        return out

    def urls(self):
        return {k: URLS[k].format(v) for k, v in self.ids.items() if k in URLS}

    def to_dict(self):
        return {"ids": self.ids, "title": self.title, "titles": self.titles, "year": self.year,
                "runtime": self.runtime, "credits": self.credits, "aliases": self.aliases,
                "sources": sorted(self.sources)}

    @classmethod
    def from_dict(cls, d):
        f = cls(ids=d.get("ids"), title=d.get("title"), titles=d.get("titles"), year=d.get("year"),
                runtime=d.get("runtime"), credits=d.get("credits"))
        f.sources = set(d.get("sources") or [])
        f.aliases = d.get("aliases") or {}
        return f

    def __repr__(self):
        return f"<Film {self.title} ({self.year}) {self.ids}>"


def merge_all(films):
    out = []
    for f in films:
        if f is None:
            continue
        hit = [o for o in out if o.shares_id(f)]
        if not hit:
            out.append(f)
            continue
        base = hit[0].merge(f)
        for o in hit[1:]:
            base.merge(o)
            out.remove(o)
    return out

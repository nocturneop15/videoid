"""The Movie Database (needs a free API key or read-access token)."""
from collections import Counter

from ..film import Film

BASE = "https://api.themoviedb.org/3"
JOB_ROLE = {
    "director": "director", "producer": "producer", "executive producer": "exec_producer",
    "co-producer": "producer", "original music composer": "composer", "music": "composer", "composer": "composer",
    "director of photography": "dop", "cinematography": "dop", "editor": "editor", "screenplay": "writer",
    "writer": "writer", "novel": "writer", "story": "writer", "author": "writer", "characters": "writer",
    "production design": "production_designer", "art direction": "production_designer",
    "costume design": "costume", "casting": "casting", "sound designer": "sound", "music supervisor":
    "music_supervisor", "makeup artist": "makeup", "visual effects supervisor": "vfx",
}


class TMDB:
    name = "tmdb"

    def __init__(self, http, key, language="en-US"):
        self.http = http
        self.key = key or ""
        self.lang = language
        self.enabled = bool(self.key)

    def _get(self, path, **params):
        headers = None
        if len(self.key) > 40:          # v4 read access token (JWT)
            headers = {"Authorization": f"Bearer {self.key}"}
        else:
            params["api_key"] = self.key
        return self.http.get(BASE + path, params=params, headers=headers)

    @staticmethod
    def _year(d):
        try:
            return int((d or "")[:4]) or None
        except ValueError:
            return None

    def search(self, title, year=None, limit=6):
        out = []
        for y in ([year, None] if year else [None]):
            params = {"query": title, "language": self.lang, "include_adult": "true"}
            if y:
                params["year"] = y
            r = self._get("/search/movie", **params) or {}
            for m in r.get("results", [])[:limit]:
                out.append(Film(ids={"tmdb": m["id"]}, title=m.get("title"),
                                titles=[m.get("original_title")], year=self._year(m.get("release_date")),
                                source="tmdb"))
            if out:
                break
        return out

    def find_imdb(self, tt):
        r = self._get(f"/find/{tt}", external_source="imdb_id") or {}
        res = r.get("movie_results") or []
        return res[0]["id"] if res else None

    def fetch(self, tmdb_id):
        m = self._get(f"/movie/{tmdb_id}", language=self.lang,
                      append_to_response="credits,external_ids,alternative_titles")
        if not m:
            return None
        ext = m.get("external_ids") or {}
        f = Film(ids={"tmdb": m["id"], "imdb": ext.get("imdb_id") or m.get("imdb_id"),
                      "wikidata": ext.get("wikidata_id")},
                 title=m.get("title"), titles=[m.get("original_title")],
                 year=self._year(m.get("release_date")), runtime=m.get("runtime") or None, source="tmdb")
        for t in (m.get("alternative_titles") or {}).get("titles", []):
            f.add_title(t.get("title"))
        cr = m.get("credits") or {}
        for c in sorted(cr.get("cast", []), key=lambda c: c.get("order", 999)):
            f.add_person(c.get("name"), "cast")
            f.add_alias(c.get("name"), c.get("original_name"))       # native-script name
        for c in cr.get("crew", []):
            f.add_person(c.get("name"), JOB_ROLE.get((c.get("job") or "").lower(), "crew"))
            f.add_alias(c.get("name"), c.get("original_name"))
        return f

    def films_by_people(self, names, min_hits=2):
        counts = Counter()
        for n in names:
            r = self._get("/search/person", query=n, include_adult="true") or {}
            seen = set()
            for p in r.get("results", [])[:2]:
                cr = self._get(f"/person/{p['id']}/movie_credits") or {}
                for c in cr.get("cast", []) + cr.get("crew", []):
                    seen.add(c["id"])
            counts.update(seen)
        return [(mid, n) for mid, n in counts.most_common(30) if n >= min_hits]

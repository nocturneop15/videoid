"""IMDb: public suggestion API for search, title page JSON-LD + full credits page for people."""
import json
import re
from urllib.parse import quote

from bs4 import BeautifulSoup

from ..film import Film
from .. import ui
from ..log import log

SECTION_ROLE = {
    "director": "director", "writer": "writer", "cast": "cast", "producer": "producer", "composer": "composer",
    "cinematographer": "dop", "editor": "editor", "casting_director": "casting",
    "production_designer": "production_designer", "art_director": "production_designer",
    "costume_designer": "costume", "make_up_department": "makeup", "sound_department": "sound",
    "visual_effects": "vfx", "music_department": "music_supervisor", "assistant_director": "first_ad",
}


class IMDb:
    name = "imdb"

    def __init__(self, http):
        self.http = http
        self.pages_blocked = False   # imdb.com HTML answers bots with an AWS WAF challenge (HTTP 202)

    def search(self, title, year=None, limit=6):
        q = re.sub(r"\s+", " ", title.strip()).lower()
        if not q:
            return []
        first = q[0] if q[0].isalnum() and q[0].isascii() else "x"
        r = self.http.get(f"https://v3.sg.media-imdb.com/suggestion/{first}/{quote(q)}.json") or {}
        out = []
        for d in r.get("d", []):
            if not str(d.get("id", "")).startswith("tt"):
                continue
            if d.get("qid") in ("videoGame", "podcastSeries", "podcastEpisode", "musicVideo"):
                continue
            f = Film(ids={"imdb": d["id"]}, title=d.get("l"), year=d.get("y"), source="imdb")
            for star in (d.get("s") or "").split(","):     # top-billed stars - usable even when pages are blocked
                if star.strip():
                    f.add_person(star.strip(), "cast")
            out.append(f)
        return out[:limit]

    def fetch(self, tt):
        if self.pages_blocked:
            return None
        html = self.http.get(f"https://www.imdb.com/title/{tt}/", kind="text", browser=True,
                             validate=lambda h: bool(h) and "ld+json" in h)
        if not html:
            if not self.http.offline:
                self.pages_blocked = True
                log(ui.detail("(IMDb title pages are not accessible to scripts - using IMDb search and ids only)"), 1)
            return None
        f = Film(ids={"imdb": tt}, source="imdb")
        soup = BeautifulSoup(html, "html.parser")
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(s.string or "")
            except ValueError:
                continue
            f.add_title(ld.get("name"))
            f.add_title(ld.get("alternateName"))
            if (d := ld.get("datePublished")) and d[:4].isdigit():
                f.year = int(d[:4])
            if m := re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", ld.get("duration") or ""):
                if m.group(1) or m.group(2):
                    f.runtime = int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
            for key, role in (("director", "director"), ("actor", "cast"), ("creator", "writer")):
                for p in ld.get(key) or []:
                    if isinstance(p, dict) and p.get("@type") == "Person":
                        f.add_person(p.get("name"), role)
            break
        if (m := re.search(r"<title>(.+?) \((?:[^)]*?)(\d{4})", html)) and not f.year:
            f.year = int(m.group(2))
        self._fullcredits(tt, f)
        return f

    def _fullcredits(self, tt, f):
        html = self.http.get(f"https://www.imdb.com/title/{tt}/fullcredits/", kind="text", browser=True)
        if not html:
            return
        # assign every /name/nm.. link to the nearest preceding section anchor (works for old and new layouts)
        marks = [(m.start(), m.group(1)) for m in re.finditer(r'id="([a-z_]+)"', html) if m.group(1) in SECTION_ROLE]
        for m in re.finditer(r'href="/name/nm\d+/?[^"]*"[^>]*>\s*([^<]{3,60}?)\s*</a>', html):
            name = BeautifulSoup(m.group(1), "html.parser").get_text().strip()
            if not name or name.lower() in ("see more", "more"):
                continue
            role = "unknown"
            for pos, sec in marks:
                if pos > m.start():
                    break
                role = SECTION_ROLE[sec]
            f.add_person(name, role if role != "unknown" else "crew")

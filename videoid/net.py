"""HTTP client with per-host politeness delay, retries and a persistent SQLite cache."""
import json
import time
from urllib.parse import urlencode, urlparse

import requests

from . import __version__
from .log import log

TOOL_UA = f"VideoID/{__version__} (local media identification tool) python-requests"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")

MIN_INTERVAL = {
    "query.wikidata.org": 1.0,
    "www.wikidata.org": 0.3,
    "www.csfd.cz": 1.5,
    "www.imdb.com": 1.5,
    "api.themoviedb.org": 0.05,
}
SECRET_PARAMS = {"api_key", "apikey"}


class Http:
    def __init__(self, db, cache_days=30, offline=False):
        self.db = db
        self.max_age = cache_days * 86400
        self.offline = offline
        self.s = requests.Session()
        self._last = {}

    def _wait(self, host):
        gap = MIN_INTERVAL.get(host, 0.25)
        dt = time.time() - self._last.get(host, 0)
        if dt < gap:
            time.sleep(gap - dt)
        self._last[host] = time.time()

    def get(self, url, params=None, headers=None, kind="json", browser=False, cache=True, validate=None):
        """Returns parsed JSON / text, or None on failure. 404 is cached as a negative result.
        validate(decoded) -> bool rejects bot-check / error pages (never cached)."""
        params = params or {}
        pub = sorted((k, str(v)) for k, v in params.items() if k not in SECRET_PARAMS)
        key = url + ("?" + urlencode(pub) if pub else "")
        if cache:
            hit = self.db.cache_get(key, self.max_age)
            if hit:
                status, body = hit
                if status != 200:
                    return None
                val = self._decode(body, kind)
                if validate is None or validate(val):
                    return val
        if self.offline:
            return None
        h = {"User-Agent": BROWSER_UA if browser else TOOL_UA}
        if browser:
            h["Accept-Language"] = "cs,en;q=0.8"
            h["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        if headers:
            h.update(headers)
        host = urlparse(url).netloc
        for attempt in range(4):
            self._wait(host)
            try:
                r = self.s.get(url, params=params, headers=h, timeout=30)
            except requests.RequestException as e:
                log(f"  http error {host}: {e}", 2)
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 0) or 0) or 3 * (attempt + 1)
                log(f"  {host} -> {r.status_code}, retrying in {wait:.0f}s", 2)
                time.sleep(min(wait, 60))
                continue
            if r.status_code == 200:
                body = r.content
                val = self._decode(body, kind)
                if validate is not None and not validate(val):
                    log(f"  {host}: unusable response (bot check?) for {url}", 2)
                    return None
                if cache:
                    self.db.cache_put(key, 200, body)
                return val
            if r.status_code == 404 and cache:
                self.db.cache_put(key, 404, b"")
            log(f"  {host} -> HTTP {r.status_code} for {url}", 2)
            return None
        return None

    def post_json(self, url, payload, timeout=120):
        try:
            r = self.s.post(url, json=payload, timeout=timeout, headers={"User-Agent": TOOL_UA})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            log(f"  POST {url} failed: {e}", 1)
            return None

    @staticmethod
    def _decode(body, kind):
        if kind == "json":
            try:
                return json.loads(body)
            except ValueError:
                return None
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        return body

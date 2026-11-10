"""CSFD.cz (Czech-Slovak film database) via node-csfd-api, run in a persistent portable-Node bridge.

node-csfd-api completes CSFD's browser verification itself and the verification cookie is kept in
runtime/csfd/verification_cookie.txt (valid ~1 week). Every response is cached in the SQLite HTTP cache
and requests are spaced >= 1 s apart to keep the load on CSFD minimal.
"""
import atexit
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from ..config import ROOT
from ..film import Film
from ..log import log

NODE = ROOT / "runtime" / "node" / "node.exe"
CSFD_DIR = ROOT / "runtime" / "csfd"
BRIDGE = Path(__file__).resolve().parent.parent / "csfd_bridge.cjs"
MIN_INTERVAL = 1.0

CREATOR_ROLE = {
    "directors": "director", "writers": "writer", "basedOn": "writer", "cinematography": "dop",
    "music": "composer", "actors": "cast", "producers": "producer", "filmEditing": "editor",
    "costumeDesign": "costume", "productionDesign": "production_designer", "sound": "sound",
}


def available():
    node = NODE if NODE.exists() else None
    return bool(node and (CSFD_DIR / "node_modules" / "node-csfd-api").exists())


class _Bridge:
    def __init__(self):
        env = dict(os.environ, CSFD_DIR=str(CSFD_DIR))
        self.p = subprocess.Popen([str(NODE), str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, env=env, text=True, encoding="utf-8",
                                  creationflags=0x08000000 if os.name == "nt" else 0)
        self.lock = threading.Lock()
        self.n = 0
        ready = self.p.stdout.readline()
        if '"ready"' not in ready:
            raise RuntimeError("csfd bridge failed to start")
        atexit.register(self.close)

    def call(self, cmd, arg, timeout=90):
        with self.lock:
            self.n += 1
            self.p.stdin.write(json.dumps({"id": self.n, "cmd": cmd, "arg": arg}) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("csfd bridge died")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass


class CSFD:
    name = "csfd"

    def __init__(self, http):
        self.http = http
        self.db = http.db
        self._bridge = None
        self._last = 0.0
        self.blocked = not available()
        if self.blocked:
            log("  CSFD disabled: portable Node / node-csfd-api missing (run setup.cmd)", 1)

    # ------------------------------------------------------------ transport
    def _call(self, cmd, arg):
        key = f"csfd-api:{cmd}:{arg}"
        hit = self.db.cache_get(key, self.http.max_age)
        if hit:
            status, body = hit
            return json.loads(body) if status == 200 else None
        if self.blocked or self.http.offline:
            return None
        if self._bridge is None:
            try:
                self._bridge = _Bridge()
            except Exception as e:
                log(f"  CSFD bridge unavailable: {e}", 1)
                self.blocked = True
                return None
        wait = MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        try:
            r = self._bridge.call(cmd, arg)
        except Exception as e:
            log(f"  CSFD bridge error: {e}", 1)
            self._bridge = None
            return None
        if r.get("ok"):
            self.db.cache_put(key, 200, json.dumps(r["data"], ensure_ascii=False).encode("utf-8"))
            return r["data"]
        reason = r.get("reason")
        log(f"  CSFD {cmd} {arg}: {reason or r.get('error')}", 2)
        if reason == "not-found":
            self.db.cache_put(key, 404, b"")
        elif reason == "blocked":
            log("  CSFD blocked the request - CSFD skipped for this session", 1)
            self.blocked = True
        return None

    # ------------------------------------------------------------ API
    def search(self, title, year=None, limit=6):
        data = self._call("search", title) or {}
        out = []
        for m in (data.get("movies") or [])[:limit]:
            f = Film(ids={"csfd": m["id"]}, title=m.get("title"), year=m.get("year") or None, source="csfd")
            for role, key in (("director", "directors"), ("cast", "actors")):
                for c in (m.get("creators") or {}).get(key, []) or []:
                    f.add_person(c.get("name"), role)
            out.append(f)
        return out

    def fetch(self, csfd_id):
        m = self._call("movie", int(csfd_id))
        if not m:
            return None
        f = Film(ids={"csfd": m.get("id") or csfd_id}, title=m.get("title"), year=m.get("year") or None,
                 source="csfd")
        for t in m.get("titlesOther") or []:
            f.add_title(t.get("title"))
        d = m.get("duration")
        if isinstance(d, (int, float)) and d:
            f.runtime = int(d)
        elif isinstance(d, str):
            h = re.search(r"(\d+)\s*h", d)
            mm = re.search(r"(\d+)\s*min", d)
            if h or mm:
                f.runtime = int(h.group(1) if h else 0) * 60 + int(mm.group(1) if mm else 0)
        for key, role in CREATOR_ROLE.items():
            for c in (m.get("creators") or {}).get(key) or []:
                f.add_person(c.get("name"), role)
        return f

    def films_by_people(self, names, min_hits=2, max_people=10):
        """Films shared by several people read from the credits (CSFD person search -> filmography)."""
        counts = Counter()
        for n in names[:max_people]:
            data = self._call("search", n) or {}
            creators = data.get("creators") or []
            seen = set()
            for c in creators[:2]:
                person = self._call("creator", c["id"]) or {}
                for film in person.get("films") or []:
                    seen.add(film["id"])
            counts.update(seen)
        return [(fid, n) for fid, n in counts.most_common(30) if n >= min_hits]

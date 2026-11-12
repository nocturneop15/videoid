"""SQLite store: analysed files, unique OCR strings, people, films, matches and an HTTP cache."""
import json
import sqlite3
import time
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    size INTEGER, mtime REAL, oshash TEXT,
    duration REAL, width INTEGER, height INTEGER,
    local_json TEXT,
    ocr_done INTEGER DEFAULT 0, ocr_at TEXT, ocr_meta_json TEXT,
    matched_at TEXT, film_key TEXT, title TEXT, year INTEGER,
    verdict TEXT, confidence REAL, ids_json TEXT
);
CREATE INDEX IF NOT EXISTS files_hash ON files(oshash, size);
CREATE TABLE IF NOT EXISTS ocr_lines(
    file_id INTEGER, segment TEXT, text TEXT, t REAL, conf REAL, height REAL, seen INTEGER,
    PRIMARY KEY(file_id, segment, text)
);
CREATE TABLE IF NOT EXISTS ocr_frames(
    file_id INTEGER, segment TEXT, t REAL, boxes_json TEXT, n INTEGER DEFAULT 1,
    PRIMARY KEY(file_id, segment, t)
);
CREATE TABLE IF NOT EXISTS people(
    file_id INTEGER, name TEXT, role TEXT, source TEXT, seen INTEGER,
    PRIMARY KEY(file_id, name, role, source)
);
CREATE TABLE IF NOT EXISTS films(
    key TEXT PRIMARY KEY, ids_json TEXT, title TEXT, year INTEGER, runtime INTEGER,
    json TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS film_ids(
    source TEXT, ext_id TEXT, film_key TEXT, PRIMARY KEY(source, ext_id)
);
CREATE TABLE IF NOT EXISTS matches(
    file_id INTEGER, rank INTEGER, film_key TEXT, score REAL, confidence REAL, verdict TEXT,
    evidence_json TEXT, PRIMARY KEY(file_id, rank)
);
CREATE TABLE IF NOT EXISTS http_cache(
    key TEXT PRIMARY KEY, status INTEGER, body BLOB, fetched_at REAL
);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


class DB:
    def __init__(self, path):
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA)
        self._migrate()
        self.con.commit()

    def _migrate(self):
        """Additive schema upgrades for databases created by older versions."""
        cols = {r["name"] for r in self.con.execute("PRAGMA table_info(ocr_frames)")}
        if "n" not in cols:
            self.con.execute("ALTER TABLE ocr_frames ADD COLUMN n INTEGER DEFAULT 1")
        self.con.execute("PRAGMA user_version = 2")

    # ---------- http cache ----------
    def cache_get(self, key, max_age):
        r = self.con.execute("SELECT status, body, fetched_at FROM http_cache WHERE key=?", (key,)).fetchone()
        if r and time.time() - r["fetched_at"] < max_age:
            return r["status"], r["body"]
        return None

    def cache_put(self, key, status, body):
        self.con.execute("INSERT OR REPLACE INTO http_cache VALUES(?,?,?,?)", (key, status, body, time.time()))
        self.con.commit()

    # ---------- files ----------
    def file_lookup(self, path, size, oshash):
        r = self.con.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        if r:
            return r
        if oshash:   # same file moved/renamed (old path gone) - copies get their own record
            from pathlib import Path
            rows = self.con.execute("SELECT * FROM files WHERE oshash=? AND size=?", (oshash, size)).fetchall()
            r = next((x for x in rows if not Path(x["path"]).exists()), None)
            if r:
                self.con.execute("UPDATE files SET path=? WHERE id=?", (path, r["id"]))
                self.con.commit()
                return self.con.execute("SELECT * FROM files WHERE id=?", (r["id"],)).fetchone()
        return None

    def file_upsert(self, path, **fields):
        r = self.con.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if r is None:
            cur = self.con.execute("INSERT INTO files(path) VALUES(?)", (path,))
            fid = cur.lastrowid
        else:
            fid = r["id"]
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self.con.execute(f"UPDATE files SET {cols} WHERE id=?", (*fields.values(), fid))
        self.con.commit()
        return fid

    def file_move(self, fid, path):
        """The file was renamed: its record follows (a stale record of a file gone from that path is dropped)."""
        old = self.con.execute("SELECT id FROM files WHERE path=? AND id<>?", (path, fid)).fetchone()
        if old:
            for t in ("ocr_frames", "ocr_lines", "people", "matches"):
                self.con.execute(f"DELETE FROM {t} WHERE file_id=?", (old["id"],))
            self.con.execute("DELETE FROM files WHERE id=?", (old["id"],))
        self.con.execute("UPDATE files SET path=? WHERE id=?", (path, fid))
        self.con.commit()

    def file_get(self, fid):
        return self.con.execute("SELECT * FROM files WHERE id=?", (fid,)).fetchone()

    def files(self):
        return self.con.execute("SELECT * FROM files ORDER BY path").fetchall()

    # ---------- OCR + people ----------
    def save_frames(self, fid, frames):
        """Raw OCR boxes per frame - lets the credits parser be re-run without OCR (--reparse)."""
        self.con.execute("DELETE FROM ocr_frames WHERE file_id=?", (fid,))
        self.con.executemany("INSERT OR REPLACE INTO ocr_frames(file_id, segment, t, boxes_json, n) VALUES(?,?,?,?,?)", [
            (fid, fr[0], fr[1], json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
                                            for b in fr[2]], ensure_ascii=False), fr[3] if len(fr) > 3 else 1)
            for fr in frames])
        self.con.commit()

    def frames(self, fid):
        return [(r["segment"], r["t"], json.loads(r["boxes_json"]), r["n"] or 1) for r in self.con.execute(
            "SELECT * FROM ocr_frames WHERE file_id=? ORDER BY segment DESC, t", (fid,))]

    def save_ocr(self, fid, lines, people, meta):
        self.con.execute("DELETE FROM ocr_lines WHERE file_id=?", (fid,))
        self.con.execute("DELETE FROM people WHERE file_id=?", (fid,))
        self.con.executemany(
            "INSERT OR REPLACE INTO ocr_lines VALUES(?,?,?,?,?,?,?)",
            [(fid, l["segment"], l["text"], l["t"], l["conf"], l["height"], l["seen"]) for l in lines])
        self.save_people(fid, people)
        self.con.execute("UPDATE files SET ocr_done=1, ocr_at=?, ocr_meta_json=? WHERE id=?",
                         (now(), json.dumps(meta, ensure_ascii=False), fid))
        self.con.commit()

    def save_people(self, fid, people, source=None):
        if source:
            self.con.execute("DELETE FROM people WHERE file_id=? AND source=?", (fid, source))
        self.con.executemany(
            "INSERT OR REPLACE INTO people VALUES(?,?,?,?,?)",
            [(fid, p["name"], p["role"], p["source"], p.get("seen", 1)) for p in people])
        self.con.commit()

    def ocr_lines(self, fid):
        return [dict(r) for r in self.con.execute(
            "SELECT segment, text, t, conf, height, seen FROM ocr_lines WHERE file_id=? ORDER BY segment DESC, t",
            (fid,))]

    def people(self, fid):
        return [dict(r) for r in self.con.execute(
            "SELECT name, role, source, seen FROM people WHERE file_id=?", (fid,))]

    # ---------- films ----------
    def save_film(self, film):
        key = film.key()
        self.con.execute("INSERT OR REPLACE INTO films VALUES(?,?,?,?,?,?,?)", (
            key, json.dumps(film.ids), film.title, film.year, film.runtime,
            json.dumps(film.to_dict(), ensure_ascii=False), now()))
        for src, ext in film.ids.items():
            self.con.execute("INSERT OR REPLACE INTO film_ids VALUES(?,?,?)", (src, str(ext), key))
        self.con.commit()
        return key

    def film_json(self, key):
        r = self.con.execute("SELECT json FROM films WHERE key=?", (key,)).fetchone()
        return json.loads(r["json"]) if r else None

    def save_matches(self, fid, ranked):
        self.con.execute("DELETE FROM matches WHERE file_id=?", (fid,))
        for i, m in enumerate(ranked):
            self.con.execute("INSERT INTO matches VALUES(?,?,?,?,?,?,?)", (
                fid, i, m["film_key"], m["score"], m["confidence"], m["verdict"],
                json.dumps(m["evidence"], ensure_ascii=False)))
        self.con.commit()

    def choose_match(self, fid, rank):
        """The user picked candidate `rank` (scan --interactive): it becomes the CERTAIN winner, the rest move
        down. rank None = none of them is right: the file becomes UNKNOWN, the candidates are kept."""
        ms = self.matches(fid)
        if rank is None:
            for m in ms:
                m["evidence"] = json.loads(m["evidence_json"])
            if ms:
                ms[0]["evidence"]["rejected_by_user"] = True        # don't ask again on the next run
                self.save_matches(fid, ms)
            self.file_upsert_id(fid, film_key=None, title=None, year=None, verdict="UNKNOWN", confidence=0,
                                ids_json="{}")
            return None
        pick = ms.pop(rank)
        ev = json.loads(pick["evidence_json"])
        ev["chosen_by_user"] = True
        ev["warnings"] = []
        pick.update(verdict="CERTAIN", confidence=1.0, evidence=ev)
        for m in ms:
            m["evidence"] = json.loads(m["evidence_json"])
        self.save_matches(fid, [pick] + ms)
        film = self.film_json(pick["film_key"]) or {}
        self.file_upsert_id(fid, film_key=pick["film_key"], title=film.get("title"), year=film.get("year"),
                            verdict="CERTAIN", confidence=1.0, ids_json=json.dumps(film.get("ids") or {}))
        return pick

    def file_upsert_id(self, fid, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        self.con.execute(f"UPDATE files SET {cols} WHERE id=?", (*fields.values(), fid))
        self.con.commit()

    def matches(self, fid):
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM matches WHERE file_id=? ORDER BY rank", (fid,))]

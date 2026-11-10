# VideoID database schema

VideoID keeps everything it learns in one SQLite file, `videoid.db`, next to the tool.
You can move it with `videoid config db_path <path>`.
It runs in WAL mode, so it can be read with any SQLite client, even while a scan is running.

Schema version: `PRAGMA user_version = 2`. Upgrades only add things (`ALTER TABLE ... ADD COLUMN`), so older databases keep working.

```
files 1───* ocr_frames      raw OCR boxes per frame (re-parsable)
  │   1───* ocr_lines       unique OCR strings
  │   1───* people          people read from the credits
  │   1───* matches ───*──1 films ──1───* film_ids
  │                          (film_key)       (any external id -> film)
http_cache                  standalone: cached web / API responses
```

All timestamps are local ISO-8601 strings (`2026-11-10T07:12:44`), except `http_cache.fetched_at`, which is a Unix epoch.

---

## `files`: one row per analysed video file

| column | type | meaning |
|---|---|---|
| `id` | INTEGER PK | file id (used by `videoid show <id>`) |
| `path` | TEXT UNIQUE | absolute path (UNC paths kept as given) |
| `size` | INTEGER | bytes |
| `mtime` | REAL | modification time (epoch) |
| `oshash` | TEXT | OpenSubtitles hash: size plus 64-bit sums of the first and last 64 KiB. A moved or renamed file is matched back to its record; a copy at a new path gets its own record. |
| `duration` | REAL | seconds (from ffmpeg) |
| `width`, `height` | INTEGER | video resolution |
| `local_json` | TEXT (JSON) | evidence found without OCR, see below |
| `ocr_done` | INTEGER | 1 when OCR and credits parsing are stored |
| `ocr_at` | TEXT | when OCR ran |
| `ocr_meta_json` | TEXT (JSON) | `{"title_candidates": [...], "copyright_years": [...], "stats": {"frames", "ocr_frames", "seconds"}, "llm": "model"}` |
| `matched_at` | TEXT | when database matching last ran |
| `film_key` | TEXT | best match → `films.key` (NULL when nothing was found) |
| `title`, `year` | TEXT, INTEGER | best match, copied here for quick listing |
| `verdict` | TEXT | `CERTAIN`, `PROBABLE`, `POSSIBLE`, `UNLIKELY` or `UNKNOWN` |
| `confidence` | REAL | 0–0.999 |
| `ids_json` | TEXT (JSON) | best match ids `{"imdb": "tt…", "tmdb": "…", "csfd": "…", "wikidata": "Q…"}` |

Index: `files_hash (oshash, size)`.

`local_json`:
```json
{"titles": [["Kolja", "filename"], ["Kolja", "nfo"]], "years": [[1996, "filename"]],
 "ids": {"imdb": "tt0116790"}, "people": [{"name": "Jan Svěrák", "role": "director", "source": "nfo"}],
 "type": "movie", "season": null, "episode": null, "nfo_files": ["X:\\…\\movie.nfo"]}
```
The second item of each title/year pair is its source: `filename`, `folder`, `nfo` or `embedded` (container title tag).

## `ocr_frames`: raw OCR result for every sampled frame that contained text

| column | type | meaning |
|---|---|---|
| `file_id` | INTEGER | → `files.id` |
| `segment` | TEXT | `open` (opening titles) or `end` (end credits) |
| `t` | REAL | position in the video, seconds |
| `boxes_json` | TEXT (JSON) | `[{"text", "conf", "x", "y", "w", "h", "lang"}]`. Coordinates are relative (0–1) to the frame; `lang` is the recognition model that won (`latin`, `ch`, `devanagari`, …). |
| `n` | INTEGER | how many sampled frames this result covers. A static card is OCR'd once, and identical frames after it only increase `n`. |

PK `(file_id, segment, t)`. `videoid scan --reparse` re-runs the credits parser from this table without running OCR again.

## `ocr_lines`: unique strings read from the credits

| column | type | meaning |
|---|---|---|
| `file_id` | INTEGER | → `files.id` |
| `segment` | TEXT | `open` / `end` |
| `text` | TEXT | best reading. Near-identical OCR variants (fuzzy ratio ≥ 92) are merged into one row. |
| `t` | REAL | first time it was seen |
| `conf` | REAL | best OCR confidence |
| `height` | REAL | text height relative to the frame (large = title) |
| `seen` | INTEGER | number of frames it appeared in |

PK `(file_id, segment, text)`.

## `people`: people extracted from the credits

| column | type | meaning |
|---|---|---|
| `file_id` | INTEGER | → `files.id` |
| `name` | TEXT | as read (may be upper case or in a non-Latin script) |
| `role` | TEXT | `director`, `producer`, `exec_producer`, `composer`, `writer`, `dop`, `editor`, `production_designer`, `costume`, `casting`, `sound`, `makeup`, `vfx`, `dramaturg`, `first_ad`, `production_manager`, `music_supervisor`, `cast`, `crew` or `unknown` |
| `source` | TEXT | `ocr` (rule-based parser) or `llm` (local LLM) |
| `seen` | INTEGER | evidence weight (frames on screen) |

PK `(file_id, name, role, source)`. NFO people are not stored here; they live in `files.local_json`.

## `films`: film records merged from all databases

| column | type | meaning |
|---|---|---|
| `key` | TEXT PK | canonical key, the first available of `imdb:tt…`, `wikidata:Q…`, `tmdb:…`, `csfd:…`, else `title:<norm>:<year>` |
| `ids_json` | TEXT (JSON) | every known external id |
| `title` | TEXT | display title (the accented spelling is preferred) |
| `year` | INTEGER | release / production year |
| `runtime` | INTEGER | minutes |
| `json` | TEXT (JSON) | full record, see below |
| `updated_at` | TEXT | last refresh |

`json`:
```json
{"ids": {"imdb": "tt0060802", "wikidata": "Q738526", "tmdb": "789", "csfd": "6664"},
 "title": "Ostře sledované vlaky", "titles": ["Ostře sledované vlaky", "Closely Watched Trains", "…"],
 "year": 1966, "runtime": 92,
 "credits": {"director": ["Jiří Menzel"], "composer": ["Jiří Šust"], "cast": ["Václav Neckář", "…"]},
 "aliases": {"Zhang Yimou": ["张艺谋"]},
 "sources": ["csfd", "imdb", "tmdb", "wikidata"]}
```
- `credits` keeps cast in billing order.
- `aliases` holds the same person's name in other scripts or spellings (Wikidata labels in the OCR languages, TMDB `original_name`). They are used for matching but never counted twice.

## `film_ids`: lookup of any external id to the film

| column | type | meaning |
|---|---|---|
| `source` | TEXT | `imdb`, `tmdb`, `csfd`, `wikidata` |
| `ext_id` | TEXT | id in that database |
| `film_key` | TEXT | → `films.key` |

PK `(source, ext_id)`.

## `matches`: ranked candidates for every file (top 5)

| column | type | meaning |
|---|---|---|
| `file_id` | INTEGER | → `files.id` |
| `rank` | INTEGER | 0 = best |
| `film_key` | TEXT | → `films.key` |
| `score` | REAL | evidence score (sum of the weights below) |
| `confidence` | REAL | 0–0.999 |
| `verdict` | TEXT | as in `files.verdict` |
| `evidence_json` | TEXT (JSON) | why this film was chosen, see below |

PK `(file_id, rank)`.

`evidence_json`:
```json
{"people": [{"ocr": "JIRI SLAVICEK", "db": "Jiří Slavíček", "role": "director", "sim": 100, "src": "ocr"}],
 "title": {"sim": 100, "pair": ["KONEC STRAŠIDEL", "Konec strašidel", "ocr"]},
 "year": {"film": 1952, "diff": 0, "src": "ocr"},
 "runtime": {"film": 80, "file": 79, "diff": 1},
 "ids": ["imdb"],
 "warnings": ["year in file name/NFO 1996 differs from film year 1966"]}
```

Score weights:
- **Matched people, by role:** director 10, composer 6, cinematographer and writer 4, editor and producer 3, top-10 cast 3, other cast 1.5, others 1–2. A further ×1.25 when the role read from the credits agrees with the database.
- **Title** ≥ 92 % → +12, ≥ 82 % → +6.
- **Year:** exact → +6, ±1 → +3, a contradiction with the filename or NFO → −6.
- **Runtime:** within ±3 min → +4.
- **NFO id:** equal → +40, conflicting → −15.

Verdict:
- **CERTAIN:** strong evidence plus a clear lead over the runner-up (runner-up below 60 % of the score, or 15+ points behind). Strong evidence means any of:
  - the director and 2+ more people;
  - 5+ people;
  - 2 key crew roles and 3+ people;
  - an NFO id plus corroboration.
- **PROBABLE:** score ≥ 18 with a 6+ point lead.
- **POSSIBLE:** anything else above zero.

## `http_cache`: web and API responses

| column | type | meaning |
|---|---|---|
| `key` | TEXT PK | URL plus sorted query string (API keys removed), or `csfd-api:<cmd>:<arg>` for the ČSFD bridge |
| `status` | INTEGER | 200 (body stored) or 404 (negative result) |
| `body` | BLOB | raw response (JSON / HTML) |
| `fetched_at` | REAL | epoch; entries older than `cache_days` (default 30) are fetched again |

Bot-check / challenge pages are never cached. `videoid scan --offline` answers only from this table.

---

## Useful queries

```sql
-- library overview
SELECT verdict, COUNT(*) FROM files GROUP BY verdict;

-- everything identified, with links
SELECT path, title, year, json_extract(ids_json,'$.imdb') imdb, json_extract(ids_json,'$.csfd') csfd
FROM files WHERE verdict IN ('CERTAIN','PROBABLE') ORDER BY title;

-- files that need a look
SELECT id, path, title, verdict, confidence FROM files WHERE verdict NOT IN ('CERTAIN') ORDER BY confidence;

-- which people confirmed a match
SELECT j.value ->> 'db' AS person, j.value ->> 'role' AS role
FROM matches m, json_each(m.evidence_json, '$.people') j WHERE m.file_id = 1 AND m.rank = 0;

-- all films in the library by a director
SELECT f.path, fi.title, fi.year FROM files f JOIN films fi ON fi.key = f.film_key
WHERE EXISTS (SELECT 1 FROM json_each(fi.json, '$.credits.director') d WHERE d.value = 'Jiří Menzel');

-- duplicates (same film in several files)
SELECT film_key, COUNT(*) n, group_concat(path, ' | ') FROM files
WHERE film_key IS NOT NULL GROUP BY film_key HAVING n > 1;

-- what OCR read in a file's end credits
SELECT t, text, seen FROM ocr_lines WHERE file_id = 1 AND segment = 'end' ORDER BY t;

-- clear the web cache (forces fresh lookups)
DELETE FROM http_cache;
```

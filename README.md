# VideoID

**New here? Read the [user manual](MANUAL.md)**: installing, identifying films, Kodi, renaming, settings, troubleshooting.

Portable tool that identifies movie files. It combines several sources of evidence:

1. **File and folder name** (parsed with guessit: title, year, release junk removed)
2. **NFO files** in the same folder (Kodi XML or scene NFO): IMDb, TMDB and ČSFD links, title, year and people
3. **Container metadata** (embedded title tag) and **runtime**
4. **OCR of the opening titles and end credits**. It reads the first 6 and last 6 minutes and samples one frame every 2 s (`--fps 1` for every second).
   Black frames and repeated static frames are skipped. Only unique strings are kept.
5. **Credits parsing**. Role headers in ~10 languages (EN/CZ/SK/DE/FR/IT/ES/PL/HU), e.g. "Directed by", "Režie",
   "Hudba", "Kamera", "Hrají", become director, producer, composer, writer, DoP, editor,
   production designer, costumes, casting, sound, cast and other crew.
   Optionally a **local LLM** (Ollama / LM Studio, on the GPU) reads the unique strings as well.
6. **Database matching**:
   - **Wikidata** (no key). It links IMDb, TMDB and ČSFD IDs together and has credits.
   - **TMDB** (free key, recommended: full cast and crew).
   - **IMDb** search, plus the top-billed stars from the search results.
   - **ČSFD**: full credits, search and people's filmographies, through
     [node-csfd-api](https://github.com/bartholomej/node-csfd-api). It runs on a bundled portable Node.js and
     handles ČSFD's browser verification itself.

   Candidates come from title searches *and* from a reverse lookup: films that share several of
   the people read from the credits. Every candidate is scored on matched people (weighted by role),
   title, year, runtime and NFO IDs.

Result per file: `CERTAIN` / `PROBABLE` / `POSSIBLE`, with links to IMDb, TMDB, ČSFD and Wikidata.
Each result also lists which people matched, plus warnings when the file name or NFO contradicts the content.
`CERTAIN` means, for example, the director plus at least 2 more people, or 5+ people, matched, with a clear margin over the runner-up.

Everything is kept in `videoid.db` (SQLite): files (with OpenSubtitles hash, so renamed or moved files are recognised),
raw OCR boxes per frame, unique OCR strings, detected people, films with full credits, ranked matches and an HTTP cache.
**[DATABASE.md](DATABASE.md)** describes every table, column and JSON format, with example queries.

## OCR languages

Text detection runs once per frame. Every enabled recognition model (PP-OCRv5) then reads the detected boxes,
and the best-scoring reading wins. Each extra language adds roughly 30 % to OCR time.

| `--lang` / `ocr.languages` | script / languages |
|---|---|
| `latin` (default) | Czech, Slovak, German, French, Italian, Spanish, Polish, Hungarian, … (with diacritics) |
| `en` | English only (slightly better for English-only credits; cannot read accents, so keep `latin` alongside it) |
| `zh`, `ja` | Chinese (simplified + traditional) and Japanese (one model) |
| `ko` | Korean |
| `hi`, `mr`, `ne` | Devanagari: Hindi, Marathi, Nepali |
| `ta`, `te` | Tamil, Telugu |
| `th` | Thai |
| `ar`, `fa`, `ur` | Arabic script |
| `ru`, `uk`, `be` / `cyrillic` | East Slavic / other Cyrillic (Bulgarian, Serbian, Macedonian) |
| `el` | Greek |

```
videoid.cmd scan D:\Movies\Bollywood --lang latin,hi
videoid.cmd config ocr.languages "[\"latin\", \"ja\", \"ko\"]"
```
The credits parser knows the role headers in these languages (निर्देशक, 导演, 監督, 감독, режиссёр, இயக்கம், …).
Names in scripts without capital letters are found by shape and position.
Wikidata names are also fetched in the enabled scripts, so 张艺谋 on screen matches Zhang Yimou.
When two models (or two frames) read the same name and differ only in accents, the accented reading is kept.

### OCR quality vs speed

Frames are deinterlaced (`scan.deinterlace`: `auto` = frames flagged interlaced, `always`, `off`), then scaled
to 1920 px wide with lanczos (`scan.min_width` / `max_width` / `scale_flags`), so SD credit text gets enough pixels.
Text is found by the PP-OCRv5 **server** detector, which catches small and thin credit text much better.
It is about 4× slower than the mobile one. To trade accuracy back for speed:
```
videoid.cmd config ocr.det_model mobile
videoid.cmd config scan.max_width 1600
```
Each file reports its total time and OCR time.

### Parallel OCR (faster GPUs)

`ocr.workers` runs several OCR engines in parallel (`auto` = 2 on a GPU, 1 on CPU) while frames are decoded
in the background. Results are applied in frame order, so the output matches a single worker.
On DirectML, the workers' GPU calls are serialized by default (`ocr.gpu_lock: auto`), because concurrent
DirectML calls hung or crashed on some drivers. Their CPU work still overlaps, which is where a fast GPU
otherwise sits idle. CUDA runs them concurrently. To find the best setting for a GPU (it also checks correctness):
```
runtime\python\python.exe tests\test_parallel_ocr.py --video D:\Movies\film.mkv --workers 1,2,3,4
videoid.cmd config ocr.workers 3
videoid.cmd config ocr.gpu_lock false     # only if the test passes with it; a driver crash ends the process
```

## API keys

On first start, `scan` and `doctor` ask for any missing API keys. Optional keys can be skipped with Enter and are not asked again.
`videoid.cmd keys` enters or changes them later. The only key used today is TMDB's, which is optional but recommended.

## GPU

| Stage | Engine |
|---|---|
| Video decoding | ffmpeg `-hwaccel d3d11va` (falls back to CPU) |
| Text detection + recognition | PaddleOCR PP-OCRv5 (Latin model, reads diacritics) via RapidOCR on ONNX Runtime **DirectML**, which runs on any DX12 GPU (AMD / Intel / NVIDIA) |
| Credits reading (optional) | any local LLM served by Ollama / LM Studio (Vulkan/ROCm/CUDA) |

## Install (Windows x64, portable)

One line in PowerShell:
```powershell
irm https://raw.githubusercontent.com/nocturneop15/videoid/main/install.ps1 | iex
```
It asks for the installation folder (Enter = `%LOCALAPPDATA%\VideoID`; for an unattended install set `$env:VIDEOID_DIR` first),
downloads the code, builds the runtime and adds `videoid` to your user PATH. Running it again updates the code and keeps your database and config.

Or clone / download the repository and just run `videoid.cmd`. On first start it downloads everything it needs into `.\runtime`:
embedded Python 3.13, portable Node.js LTS, node-csfd-api, the Python packages and the Latin OCR models.
That takes about 560 MB on disk and 3 minutes. OCR models for other languages (~8 MB each) download the first time you use them.
`setup.ps1 -Force` rebuilds the runtime.

```
videoid doctor     # checks ffmpeg, GPU OCR, ČSFD, LLM, database
```

The whole folder can be copied to another Windows x64 PC. Nothing is installed system-wide; delete the folder to uninstall.

Optional but recommended (free key from https://www.themoviedb.org/settings/api):
```
videoid.cmd config tmdb_api_key YOUR_KEY_OR_READ_TOKEN
```

Optional local LLM: run Ollama (`ollama pull qwen2.5:7b`) or LM Studio. It is auto-detected. You can also set it explicitly:
```
videoid.cmd config llm.url http://localhost:11434/v1
videoid.cmd config llm.model qwen2.5:7b
```

## Usage

```
videoid.cmd scan D:\Movies                    # recursive; already identified files are skipped
videoid.cmd D:\Movies --kodi                  # "scan" is the default command
videoid.cmd scan D:\Movies\film.mkv --force   # redo OCR
videoid.cmd scan D:\Movies --rematch          # reuse stored OCR, query databases again
videoid.cmd scan D:\Movies --no-ocr           # file name + NFO only (fast)
videoid.cmd scan X --end 15 --open 10 --fps 1 # scan more of the film, every second
videoid.cmd scan X --json result.json
videoid.cmd scan D:\Movies --rename           # rename identified movies to "Title (Year)"
videoid.cmd scan D:\Movies -i                 # --interactive: ask which film it is when not CERTAIN (or type its name)

videoid.cmd list [--verdict POSSIBLE]
videoid.cmd show film.mkv --people --ocr      # stored result, people read, unique OCR strings
videoid.cmd export library.csv | library.json
videoid.cmd config [key [value]]              # e.g. config scan.end_minutes 15
```

### Log file per movie

Each scan appends its output for a movie to **`<movie>.log`** next to it, for example `Film (2001).log`.
The log holds what was read and searched, the result, and any rename or Kodi files written. Files skipped
as already identified only get an entry when something is renamed or written. To turn it off, use
`--no-log` or `config scan.log_file false`.

### Movies with text subtitles: no OCR needed

Text subtitles inside the file (SRT / ASS / mov_text / WebVTT) or next to it are read first, in seconds.
A candidate found by its title is accepted, as CERTAIN, without OCR when all of these hold:
- at least `scan.subtitle_characters` (3) of its characters (from TMDB and Wikidata) are named in the dialogue, at least twice each;
- its known length fits the file;
- no other film with the same characters scores nearly as high.

Picture subtitles (PGS, VobSub) are skipped. `--no-subs` turns this off.

### Movies with an NFO: quick check

If a movie has an NFO that VideoID didn't write, it is checked first, not fully scanned. VideoID reads
only the end credits, one frame every 3 seconds, and identifies the film **from the credits alone**,
ignoring the NFO and file name. The NFO is accepted only when that lands on the NFO's film with at
least 3 people matched. The log then says `NFO verified`. Otherwise it says `NFO NOT verified (...)`
and the normal full scan runs.

The check ignores the NFO on purpose: films sharing actors would otherwise confirm each other. The check
takes roughly a quarter of the full scan's OCR time.

| option / config | default | |
|---|---|---|
| `scan.nfo_check_every` | `3` | seconds between checked frames; `0` always does the full scan |
| `scan.nfo_check_people` | `3` | people the credits must match |
| `--no-quick` | off | full scan even when an NFO is present |

### Renaming

`scan --rename` renames each identified movie to `Title (Year)`. The files belonging to it move along:
the same name plus `.nfo`, `.srt`, `.cs.srt`, `.log`, and `-poster.jpg` / `-fanart.jpg` etc. Other
videos and `movie.nfo` stay as they are. The title is the one Kodi gets, in the `kodi.prefer` order,
so put `csfd` first for Czech titles. Nothing is ever overwritten: if the target name exists, the file
is left as it is. The old name is recorded in the movie's log.

| config | default | |
|---|---|---|
| `rename.min_verdict` | `CERTAIN` | `PROBABLE` renames more, with more risk |
| `rename.pattern` | `{title} ({year})` | also `{originaltitle}`, e.g. `{title} ({year}) - {originaltitle}` |

## Kodi metadata

```
videoid.cmd kodi D:\Movies                 # every identified file below the folder (alias: videoid nfo)
videoid.cmd scan D:\Movies --kodi          # identify and write in one go
videoid.cmd kodi film.mkv --dry-run        # print the NFO only
```

For every file matched `CERTAIN` (including films chosen with `scan --interactive`), this writes next to the video:
- **`<video>.nfo`** in [Kodi's movie NFO format](https://kodi.wiki/view/NFO_files/Movies). It holds the title, original title, plot, outline, tagline,
  runtime, ratings (TMDB, ČSFD), genres, countries, studios, tags, premiere date, year, directors, writers, and
  actors with character names and photos. It also gets `uniqueid`s for IMDb, TMDB, ČSFD and Wikidata.
- **`<video>-poster.jpg`** and **`<video>-fanart.jpg`**.

Kodi picks these up with "Local information only", or with any scraper (which uses local NFOs first).

| option / config | default | |
|---|---|---|
| `--min-verdict` / `kodi.min_verdict` | `CERTAIN` | `PROBABLE` writes more, with more risk; `POSSIBLE` for everything |
| `--prefer` / `kodi.prefer` | `tmdb,csfd` | which database's title, plot, genres and artwork come first; `csfd,tmdb` gives Czech metadata |
| `--no-artwork` / `kodi.artwork` | on | skip poster and fanart downloads |
| `--overwrite` | off | replace a `<video>.nfo` or artwork that VideoID didn't write. The old NFO is kept as `.nfo.bak`. |
| `--refresh` | off | fetch plot, genres and artwork again from TMDB / ČSFD |

- **Your files are safe:** NFOs that VideoID didn't write are never touched without `--overwrite`.
- **No self-confirmation:** the NFOs VideoID writes carry a `created by VideoID` comment, and `scan` ignores them as evidence, so a written NFO can't confirm its own match.
- **What's skipped:** TV episodes.
- **Language:** without a TMDB key, the descriptive metadata comes from ČSFD and is in Czech.

## Limits

- **IMDb** title pages answer scripts with an AWS WAF challenge, so IMDb is used for search and top-billed stars only.
- **ČSFD** requests are cached for 30 days and spaced at least 1 s apart. The verification cookie lives in
  `runtime\csfd\verification_cookie.txt`, valid for about a week.
- Without a TMDB key, credits come from Wikidata, ČSFD and IMDb's top-billed stars.
- Built for movies. TV episodes are only partly supported (series matched by title and credits, not per episode).

## Testing

`tests\evaluate.py` runs `videoid scan` on a scrambled test set and scores the results against the original names (see the script header).
```
runtime\python\python.exe tests\evaluate.py --mapping X:\mapping.txt --root X:\test --limit 20 -- --no-nfo
runtime\python\python.exe tests\evaluate.py --mapping X:\mapping.txt --root X:\test --start 20 --limit 20 -- --reparse --no-nfo
```

# VideoID

Portable tool that identifies movie files. It combines several sources of evidence:

1. **File and folder name** (parsed with guessit: title, year, release junk removed)
2. **NFO files** in the same folder (Kodi XML or scene NFO): IMDb, TMDB and ČSFD links, title, year and people
3. **Container metadata** (embedded title tag) and **runtime**
4. **OCR of the opening titles and end credits**. It reads the first 6 and last 12 minutes and samples 1 frame/s.
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
| `en` | English only (slightly better for English-only credits) |
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
It downloads the code to `%LOCALAPPDATA%\VideoID` (set `$env:VIDEOID_DIR` first to choose another folder),
builds the runtime and adds `videoid` to your user PATH. Running it again updates the code and keeps your database and config.

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
videoid.cmd scan D:\Movies\film.mkv --force   # redo OCR
videoid.cmd scan D:\Movies --rematch          # reuse stored OCR, query databases again
videoid.cmd scan D:\Movies --no-ocr           # file name + NFO only (fast)
videoid.cmd scan X --end 15 --open 10 --fps 2 # scan more of the film
videoid.cmd scan X --json result.json

videoid.cmd list [--verdict POSSIBLE]
videoid.cmd show film.mkv --people --ocr      # stored result, people read, unique OCR strings
videoid.cmd export library.csv | library.json
videoid.cmd config [key [value]]              # e.g. config scan.end_minutes 15
```

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

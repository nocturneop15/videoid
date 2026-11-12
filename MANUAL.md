# VideoID user manual

VideoID finds out which film a video file really is, even when the file is called `movie_final2.mkv`.
It reads the opening titles and end credits with OCR (text recognition on your graphics card) and looks
up the names it reads in film databases (TMDB, ČSFD, IMDb, Wikidata). It also uses the file name and
any NFO files. Afterwards it can write [Kodi](https://kodi.tv) metadata, with plot, cast and posters,
and rename the files to `Title (Year)`.

It never changes your video files themselves. Renaming happens only when you ask for it.

**Contents**
1. [Installing](#1-installing)
2. [First start](#2-first-start)
3. [Identifying films](#3-identifying-films)
4. [Reading the result](#4-reading-the-result)
5. [Kodi metadata](#5-kodi-metadata)
6. [Renaming files](#6-renaming-files)
7. [Films that already have an NFO](#7-films-that-already-have-an-nfo)
8. [The log file next to each film](#8-the-log-file-next-to-each-film)
9. [Looking at past results](#9-looking-at-past-results)
10. [Settings](#10-settings)
11. [Speed](#11-speed)
12. [Troubleshooting](#12-troubleshooting)
13. [Updating and uninstalling](#13-updating-and-uninstalling)

---

## 1. Installing

You need **Windows 10 or 11 (64-bit)** and about **600 MB** of disk space. A graphics card that runs
DirectX 12 makes it much faster: any NVIDIA, AMD or Intel card from the last ~8 years. Without one it
still works, just slowly.

Open **PowerShell**: press Start, type `powershell`, press Enter. Paste this line and press Enter:

```powershell
irm https://raw.githubusercontent.com/nocturneop15/videoid/main/install.ps1 | iex
```

1. It asks where to install. Press **Enter** for the default (`%LOCALAPPDATA%\VideoID`) or type a folder,
   e.g. `D:\Tools\VideoID`.
2. It downloads VideoID and everything it needs, including Python, ffmpeg and the OCR models. This takes
   a few minutes. Nothing is installed system-wide; everything stays in that folder.
3. It adds `videoid` to your PATH. **Close and reopen PowerShell** afterwards so the command is found.

Check that everything works:

```
videoid doctor
```

It shows one line per component. The important one is `OCR:`. It should start with `GPU (DirectML)`
and end with `read: ['Directed by Test Person']`, which means text recognition works on your graphics
card. `local LLM: none` and `TMDB key: not set` are fine; both are optional.

---

## 2. First start

The first time you run `videoid scan` or `videoid doctor`, it asks for a **TMDB API key**.

- **It's optional, but recommended.** With it, VideoID knows the full cast and crew of almost every film,
  which makes identification much more reliable. It also gives English plots and posters for Kodi.
- **It's free:** create an account at <https://www.themoviedb.org>, then open
  <https://www.themoviedb.org/settings/api> and request an API key ("Developer", personal use).
  Paste either the *API Key* or the *API Read Access Token*.
- To skip it, press **Enter**. You won't be asked again. Without it, VideoID uses Wikidata, ČSFD and IMDb.

To enter or change the key later:

```
videoid keys
```

---

## 3. Identifying films

Point VideoID at a folder. It goes through all subfolders:

```
videoid D:\Movies
```

That is short for `videoid scan D:\Movies`. You can also give a single file, or several paths:

```
videoid "D:\Movies\Some film.mkv"
videoid D:\Movies E:\More
```

Put paths with spaces in quotes. In PowerShell, single quotes work too: `'X:\Dokumentarni filmy'`.

**What happens for each file:**
1. The file name, folder name and any NFO file are read.
2. If the film has text subtitles, the characters' names in the dialogue are checked first. That can
   identify it in seconds, without reading the screen (see [Subtitles](#subtitles-the-fastest-way)).
3. If an NFO is there, a quick check runs next (see [section 7](#7-films-that-already-have-an-nfo)).
4. Otherwise, the first 6 and last 12 minutes are scanned, one frame per second, and all text is read.
5. The names found (director, actors, crew) are looked up in the film databases.
6. The best match is printed and saved.

Expect roughly **3-6 minutes per film** on a mid-range graphics card. On a laptop with integrated
graphics it can take 10-20 minutes. Each film shows its own time at the end.

Films that were already identified are skipped on the next run, so you can stop with **Ctrl+C** and
continue later. Files under 50 MB and files with "sample" in the name are skipped.

### Useful options

| command | what it does |
|---|---|
| `videoid D:\Movies --kodi` | also write Kodi metadata (section 5) |
| `videoid D:\Movies --rename` | also rename identified films (section 6) |
| `videoid D:\Movies --kodi --rename` | both |
| `videoid D:\Movies --interactive` | ask you which film it is when VideoID isn't sure (see [section 4](#choosing-the-film-yourself---interactive)) |
| `videoid film.mkv --force` | scan an already identified film again from scratch |
| `videoid D:\Movies --rematch` | reuse the stored text, look the films up in the databases again (fast) |
| `videoid D:\Movies --no-ocr` | use only file names and NFOs (very fast, less reliable) |
| `videoid film.mkv --end 20` | scan the last 20 minutes instead of 12 (long end credits) |
| `videoid film.mkv --open 10` | scan the first 10 minutes instead of 6 |
| `videoid D:\Movies --lang latin,ja` | also read Japanese (see [Languages](#languages)) |
| `videoid D:\Movies --no-quick` | always scan fully, even when an NFO exists |
| `videoid D:\Movies --no-subs` | don't try the subtitles first |
| `videoid D:\Movies --no-log` | don't write the log file next to each film |
| `videoid D:\Movies --cpu` | don't use the graphics card |
| `videoid D:\Movies --no-recursive` | only this folder, not subfolders |
| `videoid D:\Movies --json results.json` | also save all results to a JSON file |

Add `-v` for more detail on screen, or `-q` for less.

### Languages

By default VideoID reads Latin script with all diacritics (Czech, Slovak, German, French, Polish, …),
which also covers English. For films with credits in other scripts, add the language:

| `--lang` | script |
|---|---|
| `latin` (default) | Czech, Slovak, German, French, Spanish, Polish, Hungarian, English, … |
| `zh`, `ja` | Chinese, Japanese |
| `ko` | Korean |
| `ru`, `uk`, `be`, `cyrillic` | Russian, Ukrainian, Belarusian; other Cyrillic (Bulgarian, Serbian, …) |
| `el` | Greek |
| `ar`, `fa`, `ur` | Arabic script |
| `hi`, `mr`, `ne` | Hindi, Marathi, Nepali |
| `ta`, `te`, `th` | Tamil, Telugu, Thai |

Always keep `latin` in the list, e.g. `--lang latin,ru`. Each extra language adds about 30 % time.
To make it permanent: `videoid config ocr.languages "[\"latin\", \"ru\"]"`.

---

## 4. Reading the result

```
>> D:\Movies\knofl.1997.cz.mkv
 * [CERTAIN   99.9%  score  177.4]  Buttoners (1997)  102 min
      also known as: Knoflíkáři / Die Knöpfler
      imdb      https://www.imdb.com/title/tt0128292/
      csfd      https://www.csfd.cz/film/10080/
      matched 27 people:
        cast                 Pavel Lagner
        cast                 Svetlana Svobodová   (read: Světlana Svobodová)
        ...
      evidence: NFO id imdb,tmdb; title 100% via folder; year diff 0; runtime diff 1 min
   [POSSIBLE  24.1%  score   13.1]  Nadměrné maličkosti (?)  27 min
```

Each result starts with a coloured label: **IDENTIFIED** (green, CERTAIN), **NOT SURE** (yellow,
PROBABLE or POSSIBLE) or **NOT IDENTIFIED** (red). While a film is being checked, lines starting with `>`
say what VideoID is doing right now (reading the end credits, searching the databases, …), with a progress
bar and the time left while it reads the screen. Questions from `--interactive` stand out on a yellow bar.
A run over several files ends with a summary: how many were identified, not sure, not identified or failed.

Colours work in Windows Terminal and the standard Windows 10/11 console. They're left out automatically
when the output goes to a file, and the log files next to the films never contain them. To turn colours
off, set the environment variable `NO_COLOR=1`.

The verdict in brackets says how sure VideoID is:

| verdict | meaning | what to do |
|---|---|---|
| **CERTAIN** | many people from the credits match (e.g. the director and 2 more, or 5+ people), clearly more than any other film | nothing, it's right |
| **PROBABLE** | a good match, but with less evidence | usually right; confirm it with `--interactive` to get Kodi files |
| **POSSIBLE** | weak evidence, e.g. only the title fits | check it yourself |
| **UNKNOWN** | nothing found | see [Troubleshooting](#12-troubleshooting) |

- The starred line is the winner; the lines below are runner-ups.
- *matched N people* lists who was found. `(read: …)` shows how the name looked on screen when it differs
  from the database.
- *evidence* summarizes the rest: NFO ids, how well the title fits, the difference in year and length.
- **WARNING** lines appear when the file name or NFO contradicts what the credits say. Take them seriously.

After a folder, the total time is shown.

### Choosing the film yourself: `--interactive`

```
videoid D:\Movies --interactive        # or -i
```

When a film is not CERTAIN, VideoID stops and asks (with a single candidate, it asks whether it's that film):

```
   Not sure, file 52 min. Which film is it?
    1) 2019: Cíl Měsíc (2009)  [POSSIBLE]  52 min, 0 people matched, also 2019, destination lune
         https://www.csfd.cz/film/584914/
    2) 2019, destination lune (2009)  [POSSIBLE]  1 person matched
         https://www.imdb.com/title/tt1468821/
   1-5 = this film, 0 = none of these, Enter = leave as it is:
```

- **A number:** that film becomes the result, as CERTAIN, so `--kodi` and `--rename` use it.
  The result then says *chosen by you*.
- **`0`:** none of them is right. The file is marked UNKNOWN and you won't be asked about it again.
- **Enter:** leave it as it is. You'll be asked again on the next `--interactive` run.

The links let you check a candidate in the browser before answering. Films already identified are
asked about too, so `videoid D:\Movies -i` is a quick way to go through the uncertain ones from earlier
scans. `--force` or `--rematch` replaces your choice with a new automatic result.

---

## 5. Kodi metadata

```
videoid D:\Movies --kodi          # identify and write in one go
videoid kodi D:\Movies            # write for films already identified
```

For each film that is CERTAIN, or that you chose with `--interactive`, this writes next to the video:

- `Film name.nfo`: title, original title, plot, genres, countries, studios, ratings (TMDB, ČSFD),
  director, writers, actors with roles and photos, and the database ids.
- `Film name-poster.jpg` and `Film name-fanart.jpg`.

In Kodi, add the folder as a **Movies** source. Either scraper works: "Local information only" or any
scraper, since Kodi reads local NFOs first.

**If Kodi already knew the films before**, it won't notice the new NFOs. Remove the source and add it
again, or select a film → *Information* → *Refresh*.

**Czech metadata:** to get the plot, genres and title from ČSFD instead of TMDB:

```
videoid config kodi.prefer "[\"csfd\", \"tmdb\"]"
```

Options for `videoid kodi`:

| option | |
|---|---|
| `--min-verdict PROBABLE` | also write for PROBABLE films (`POSSIBLE` writes for everything) |
| `--prefer csfd,tmdb` | source order for this run |
| `--no-artwork` | NFO only, no poster or fanart |
| `--overwrite` | replace an NFO that VideoID didn't write (the old one is kept as `.nfo.bak`) |
| `--refresh` | fetch plot, genres and artwork again |
| `--dry-run` | print the NFO, write nothing |

NFO files that VideoID didn't write are never touched without `--overwrite`.

---

## 6. Renaming files

```
videoid D:\Movies --rename
```

Each **CERTAIN** film is renamed to `Title (Year)`, e.g. `knofl.1997.cz.mkv` → `Knoflíkáři (1997).mkv`.
Files that belong to it are renamed along with it, if they have the same name:
- subtitles (`.srt`, `.cs.srt`, …)
- the NFO
- the log
- artwork (`-poster.jpg`, `-fanart.jpg`, …)

**Safety:**
- **Nothing is overwritten.** If a file with the new name already exists, the film keeps its old name and
  the result says so.
- **Old names are kept.** The log file next to the film records the old name, and VideoID's database
  follows the file to its new name.
- **Unrelated files stay.** Other videos in the folder and a general `movie.nfo` are not touched.
- **Only films are renamed.** TV episodes are left alone.

**Which title:** the same one Kodi shows, taken in the `kodi.prefer` order (see section 5). With ČSFD
first, Czech films get their Czech title; with TMDB first, the English one.

**Settings:**

```
videoid config rename.min_verdict PROBABLE                       # also rename PROBABLE films
videoid config rename.pattern "{title} ({year}) - {originaltitle}"
```

The pattern can use `{title}`, `{originaltitle}` and `{year}`. Characters Windows doesn't allow in file
names are removed, and `:` becomes ` - `.

---

### Subtitles: the fastest way

Many films carry subtitles: as text tracks inside the file (SRT, ASS, MP4 text) or as a `.srt` file with
the same name next to it. VideoID reads them first, which takes seconds instead of minutes:

1. It collects the names spoken in the dialogue. Czech and Slovak endings are understood, so "Rentone" counts as Renton.
2. It looks up the film by its file name, folder name and NFO, as usual.
3. It compares the dialogue with the candidates' **character names** (from TMDB, and from Wikidata for
   well-known films).

The film is accepted, as CERTAIN, only when all of these hold:
- the title in the file name matches;
- at least 3 of the film's characters are named in the dialogue, each at least twice;
- the film's known length fits the file;
- no other film with the same characters (a sequel or remake) is nearly as likely.

Then the log says `Subtitles confirm …` and no OCR is needed. Otherwise it says
`Subtitles not conclusive (…)` and the normal checks run.

Good to know:
- **Picture subtitles don't count:** subtitles stored as pictures (PGS on Blu-ray rips, VobSub on DVD rips) would need OCR themselves, so they're skipped.
- **A TMDB key helps a lot here:** TMDB knows the characters of almost every film, Wikidata only of well-known ones.
- **The file name must point to the film:** subtitles don't help with files like `movie_final2.mkv`. Those still need the credits.
- **Settings:** `--no-subs` skips this step; `videoid config scan.subtitle_characters 5` requires more characters, and `0` turns it off.

## 7. Films that already have an NFO

Many films come with an NFO file saying what they are, but NFOs are sometimes wrong (copied from another
release, or for another film). VideoID checks them quickly instead of trusting them blindly:

1. It reads **only the end credits, one frame every 3 seconds**. That's about a quarter of the normal work.
2. It identifies the film **from the credits alone**, ignoring the NFO and file name.
3. If that is the film the NFO names, with at least 3 people matched, the NFO is confirmed. The log says
   `NFO verified: the end credits alone identify …` and the film is done.
4. If not, the log says `NFO NOT verified (…)` with the reason, and the normal full scan runs.

Step 2 matters because films by the same director often share actors: counting the NFO's own claim would
let a wrong NFO confirm itself.

Only NFOs that VideoID didn't write are checked this way. NFOs written by `--kodi` carry a
`created by VideoID` note and are never used as evidence.

| | |
|---|---|
| `--no-quick` | always do the full scan |
| `videoid config scan.nfo_check_every 5` | one frame every 5 s (faster, less to go on); `0` turns the quick check off |
| `videoid config scan.nfo_check_people 5` | require more matching people |

---

## 8. The log file next to each film

Each time a film is scanned, VideoID adds an entry to **`Film name.log`** next to it. The entry holds:
- the date and time
- what was found in the file name and NFO
- the quick NFO check and its result
- how much text was read, and which databases were searched
- the final result with matched people
- a rename or Kodi files written, if any

New entries are added at the end, so the file keeps the history of the film. Films skipped as already
identified only get an entry when they are renamed or get new Kodi files.

To turn it off: `--no-log` for one run, or `videoid config scan.log_file false` for good.

---

## 9. Looking at past results

Everything is stored in `videoid.db` in the installation folder.

```
videoid list                          # all files with their verdict
videoid list --verdict POSSIBLE       # only the uncertain ones - worth checking by hand
videoid show "D:\Movies\film.mkv"     # full result for one file (a part of the name is enough)
videoid show film.mkv --people        # people read from the credits
videoid show film.mkv --ocr           # every text string read from the screen
videoid export library.csv            # everything to a spreadsheet (or library.json)
```

VideoID recognizes files by their content, so moving or renaming files yourself doesn't lose the results.

---

## 10. Settings

Settings live in `config.json` in the installation folder. Change them with `videoid config`:

```
videoid config                          # show all settings
videoid config scan                     # show one group
videoid config scan.end_minutes 15      # change one
```

The ones you are most likely to want:

| setting | default | |
|---|---|---|
| `scan.open_minutes` | `6` | minutes of the beginning to scan (`0` = skip the opening) |
| `scan.end_minutes` | `12` | minutes of the end to scan |
| `scan.min_size_mb` | `50` | smaller files are skipped |
| `scan.log_file` | `true` | log file next to each film |
| `scan.subtitle_characters` | `3` | characters the subtitles must name to accept a film; `0` = off |
| `scan.nfo_check_every` | `3` | quick NFO check: seconds between frames; `0` = off |
| `ocr.languages` | `["latin"]` | scripts to read (section 3) |
| `kodi.prefer` | `["tmdb", "csfd"]` | which database's title, plot and artwork come first |
| `kodi.min_verdict` | `CERTAIN` | write Kodi files from this verdict up |
| `rename.min_verdict` | `CERTAIN` | rename from this verdict up |
| `rename.pattern` | `{title} ({year})` | new file name |
| `tmdb_language` | `en-US` | language of TMDB plots and titles, e.g. `cs-CZ` |

### Optional: a local AI model

If you run [Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai), VideoID finds it automatically
and lets the model read the credits as well. This helps with unusual credit layouts. In Ollama:
`ollama pull qwen2.5:7b`. Use `--no-llm` to skip it for one run, or
`videoid config llm.enabled false` to turn it off.

---

## 11. Speed

Most of the time goes into reading text from the frames. To make it faster:

- **Use the graphics card.** In `videoid doctor`, the `OCR:` line should start with `GPU`. If it says
  `CPU`, the graphics card isn't being used; update its driver.
- **Scan less.** If the credits are only at the end: `videoid config scan.open_minutes 3`.
- **Faster, less accurate text detection:** `videoid config ocr.det_model mobile` is about 4× faster,
  but misses more small text.
- **Fast graphics cards:** VideoID uses 2 parallel workers on a graphics card. On a powerful card, try 3:
  `videoid config ocr.workers 3`. Compare the per-film times.

The computer stays usable during a scan, though on laptops with integrated graphics everything may feel
slower until it finishes.

---

## 12. Troubleshooting

**"not found: …" for a folder that exists.**
Check the quotes around the path. Very old VideoID versions had trouble with a path ending in `\`
inside quotes in Windows PowerShell; update (section 13) or leave the final `\` off.

**A film comes out UNKNOWN or POSSIBLE.**
- Some films have very short credits or credits in a separate video. Try scanning more:
  `videoid film.mkv --force --end 20 --open 10`.
- Credits in another script need the language: `--lang latin,ja` etc. (section 3).
- Without a TMDB key, less-known films may lack credits to compare with. Add the key: `videoid keys`.
- `videoid show film.mkv --ocr` shows what was actually read. If it's garbled, the picture may be too
  small or blurred to read.

**The wrong film was chosen.**
Look at *matched people* in `videoid show film.mkv`. Films from the same director share many actors. Try
`--force --end 20` to read more of the credits. A wrong NFO in the folder can also mislead VideoID; look
for WARNING lines and for `NFO NOT verified` in the film's log.

**Kodi doesn't show the plot or posters.**
Kodi ignores NFOs added after it first scanned the film. Remove the source and add it again, or use
*Refresh* on the film.

**ČSFD doesn't answer.**
ČSFD sometimes asks for browser verification. VideoID handles it, and the result stays valid for about a
week. If it keeps failing, wait a while; results already fetched are cached for 30 days.

**Something else is broken.**
Run `videoid doctor`. To start from a clean runtime, run `setup.ps1 -Force` in the installation folder.
Your database and settings are kept.

---

## 13. Updating and uninstalling

**Update:** run the install line from section 1 again and give the same folder. It updates the program
and keeps your database, settings and API key.

**Uninstall:** delete the installation folder. Optionally remove it from your PATH too: Start → "Edit
environment variables for your account" → *Path*. Nothing else was installed.
The `.nfo`, artwork and `.log` files next to your films stay; they're ordinary files you can delete.

"""Configuration: config.json next to the tool, overridable by environment variables."""
import copy
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("VIDEOID_CONFIG", ROOT / "config.json"))

DEFAULTS = {
    "db_path": "videoid.db",
    "cache_days": 30,
    "tmdb_api_key": "",          # v3 API key or v4 read access token (free at themoviedb.org)
    "tmdb_language": "en-US",
    # languages of what Kodi gets (title, plot, genres, posters), most preferred first; each text is taken in
    # the first of them that has it. No default: you enter them on the first run (videoid languages changes
    # them); until then tmdb_language is used. Czech/Slovak first also puts ČSFD first, unless kodi.prefer is
    # set yourself.
    "scrape_language": [],
    "sources": {"wikidata": True, "tmdb": True, "imdb": True, "csfd": True},
    "scan": {
        "open_minutes": 6.0,     # how much of the beginning to scan for opening titles
        "end_minutes": 6.0,      # how much of the end to scan for end credits
        "fps_open": 0.5,         # frames read per second: 0.5 = one every 2 s (credits stay on screen longer)
        "fps_end": 0.5,
        "min_width": 1920,       # small videos are upscaled (lanczos) for better OCR - SD credit text is tiny
        "max_width": 1920,
        "deinterlace": "auto",   # auto (frames flagged interlaced) | always | off
        "scale_flags": "lanczos",
        "hwaccel": "auto",       # auto | d3d11va | dxva2 | none
        "min_size_mb": 50,       # skip smaller files (samples, extras)
        "log_file": True,        # append each run's output to <movie>.log next to the movie (scan --no-log)
        # a foreign NFO next to the movie is first only checked: end credits, one frame every N seconds;
        # it is accepted when the credits confirm the NFO's film with at least nfo_check_people names,
        # otherwise the normal full scan runs (scan --no-quick always does the full scan)
        # text subtitles (in the file or next to it) are checked first: a candidate found by its title is
        # accepted without OCR when this many of its characters are named in the dialogue (0 = off)
        "subtitle_characters": 3,
        "nfo_check_every": 3,    # 0 = always full scan
        "nfo_check_people": 3,
        "extensions": ["mkv", "mp4", "avi", "m4v", "mov", "wmv", "mpg", "mpeg", "ts",
                       "m2ts", "mts", "vob", "webm", "flv", "ogm", "divx", "3gp", "rmvb"],
    },
    # languages: OCR recognition models to run. Detection runs once, each extra language adds ~30 % time.
    # latin (cs, de, fr, ...), en, ch (Chinese + Japanese), ko, hi (Devanagari: Hindi/Marathi/Nepali), ta, te,
    # th, ar (Arabic/Persian/Urdu), ru (East Slavic), cyrillic (bg/sr/mk), el
    # det_model: server (accurate on small/thin credit text) | mobile (faster)
    # Diacritics (č ř ů ä ß ...) need the latin model - en and ch can only output plain ASCII Latin.
    # workers: parallel OCR engines - "auto" = 2 on a GPU, 1 on CPU; raise on fast GPUs (try 3-4)
    # gpu_lock: serialize GPU calls of the workers - "auto" = on for DirectML (concurrent DirectML calls hung /
    #   crashed on some drivers), off for CUDA; CPU work of the workers overlaps either way
    # rec_buckets: pad text-recognition inputs to a few fixed sizes - "auto" = on for DirectML, which
    #   re-prepares the model for every new input size (text-heavy credits were ~6x slower on an RTX 3070)
    # rec_device: where text recognition runs - "auto" times GPU vs CPU once at start and picks the faster
    #   (on an RTX 3070 via DirectML it was much slower on the GPU), or "gpu" / "cpu"
    "ocr": {"gpu": True, "min_confidence": 0.6, "languages": ["latin"], "det_model": "server",
            "workers": "auto", "gpu_lock": "auto", "rec_buckets": "auto", "rec_device": "auto"},
    # Kodi metadata (videoid kodi / scan --kodi): <video>.nfo + <video>-poster.jpg / -fanart.jpg
    #   min_verdict: write only for matches this sure and better - CERTAIN (default; includes films chosen with
    #   scan --interactive) / PROBABLE / POSSIBLE
    #   prefer: which database's plot, genres, title and artwork to use first (e.g. ["csfd", "tmdb"] for Czech)
    "kodi": {"min_verdict": "CERTAIN", "artwork": True, "prefer": ["tmdb", "csfd"]},
    # scan --rename: the movie and its same-named subtitles / NFO / artwork / log get the pattern's name;
    #   {title} is the title Kodi gets (kodi.prefer order - put csfd first for Czech titles), also
    #   {originaltitle} and {year}; only matches at least min_verdict sure are renamed
    "rename": {"min_verdict": "CERTAIN", "pattern": "{title} ({year})"},
    "skipped_keys": [],          # optional API keys the user chose not to enter (not asked again)
    "llm": {
        # Optional local LLM (Ollama / LM Studio / any OpenAI-compatible server) to read credits.
        # enabled: "auto" = use if a local server answers, true/false to force.
        "enabled": "auto",
        "url": "",               # e.g. http://localhost:11434/v1  (empty = autodetect Ollama, LM Studio)
        "model": "",             # empty = first model the server lists
        "timeout": 180,
    },
}


def _deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


TMDB_REGION = {"cs": "cs-CZ", "sk": "sk-SK", "en": "en-US", "de": "de-DE", "fr": "fr-FR", "it": "it-IT",
               "es": "es-ES", "pl": "pl-PL", "hu": "hu-HU", "pt": "pt-PT", "ru": "ru-RU", "uk": "uk-UA",
               "ja": "ja-JP", "zh": "zh-CN", "ko": "ko-KR", "nl": "nl-NL", "sv": "sv-SE", "da": "da-DK",
               "fi": "fi-FI", "no": "no-NO", "nb": "nb-NO"}


_LANG_RE = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$")


def parse_langs(value):
    """'cs, sk, en' / ['cs', 'sk', 'en'] / 'cs sk en' -> ['cs', 'sk', 'en'] (None if a code is not valid)."""
    items = value if isinstance(value, list) else re.split(r"[\s,;]+", str(value or ""))
    out = []
    for it in items:
        it = str(it).strip().lower()
        if not it:
            continue
        if not _LANG_RE.match(it):
            return None
        if it not in out:
            out.append(it)
    return out


def scrape_langs(cfg):
    """Preferred metadata languages, most preferred first, as two-letter codes."""
    langs = parse_langs(cfg.get("scrape_language")) or []
    langs = list(dict.fromkeys(lang[:2] for lang in langs))
    return langs or [str(cfg.get("tmdb_language") or "en")[:2].lower()]


def scrape_lang(cfg):
    return scrape_langs(cfg)[0]


def tmdb_lang(cfg):
    """TMDB request language: the first preferred one with its region (cs -> cs-CZ)."""
    full = parse_langs(cfg.get("scrape_language")) or []
    if not full:
        return cfg.get("tmdb_language") or "en-US"
    first = full[0]
    return f"{first[:2]}-{first[3:].upper()}" if "-" in first else TMDB_REGION.get(first, first)


def ask_languages(cfg, force=False):
    """First run: the user's order of metadata languages (saved to config.json, not asked again)."""
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        return cfg
    chosen = parse_langs(cfg.get("scrape_language")) or []
    if chosen and not force:
        return cfg
    print("\nMetadata languages: titles, plots, genres and posters for Kodi are taken in the first of these\n"
          "  languages that has them. Enter two-letter codes, most preferred first, e.g. cs, sk, en\n"
          "  (cs Czech, sk Slovak, en English, de German, pl Polish, fr French, ...)")
    keep = f" [Enter = keep {', '.join(chosen)}]" if chosen else ""
    while True:
        try:
            val = input(f"  languages{keep}: ").strip()
        except EOFError:
            return cfg
        if not val and chosen:
            return cfg
        langs = parse_langs(val)
        if langs:
            break
        print("  enter at least one two-letter language code, most preferred first, e.g. cs, sk, en")
    set_value("scrape_language", json.dumps(langs))
    cfg["scrape_language"] = langs
    print("  saved: " + ", ".join(langs) + " (change it later with 'videoid languages')")
    return cfg


def apply_scrape_language(cfg, prefer_given=False):
    """Czech / Slovak: ČSFD has the local titles and plots, so it comes first - unless the user chose kodi.prefer."""
    user_prefer = "prefer" in (_user_part().get("kodi") or {})
    if scrape_lang(cfg) in ("cs", "sk") and not user_prefer and not prefer_given:
        p = list(cfg["kodi"].get("prefer") or ["tmdb", "csfd"])
        cfg["kodi"]["prefer"] = ["csfd"] + [s for s in p if s != "csfd"]
    return cfg


def load():
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        _deep_update(cfg, json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    key = os.environ.get("VIDEOID_TMDB_KEY") or os.environ.get("TMDB_API_KEY")
    if key:
        cfg["tmdb_api_key"] = key
    return cfg


def _user_part():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def set_value(dotted, raw):
    """Persist a single dotted key (e.g. scan.end_minutes) into config.json."""
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    data = _user_part()
    node = data
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return value


# ---------------------------------------------------------------------------------------------
# API keys - asked for on start-up (interactive terminals only). Non-essential keys can be skipped.
# ---------------------------------------------------------------------------------------------
API_KEYS = [
    {
        "key": "tmdb_api_key",
        "name": "TMDB (The Movie Database) API key or read access token",
        "essential": False,
        "url": "https://www.themoviedb.org/settings/api",
        "why": "full cast & crew for almost every film, native-script names, finding films by people. "
               "Strongly recommended - without it credits come only from Wikidata, CSFD and IMDb top stars.",
        "check": "tmdb",
    },
]


def _check_key(spec, value):
    if spec.get("check") == "tmdb":
        import requests
        try:
            if len(value) > 40:
                r = requests.get("https://api.themoviedb.org/3/configuration",
                                 headers={"Authorization": f"Bearer {value}"}, timeout=15)
            else:
                r = requests.get("https://api.themoviedb.org/3/configuration", params={"api_key": value}, timeout=15)
            return r.status_code == 200
        except Exception:
            return None      # cannot verify (offline) - accept
    return True


def ask_keys(cfg, force=False):
    """Prompt for missing API keys. Essential keys must be given; others can be skipped (Enter)."""
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        return cfg
    skipped = set(cfg.get("skipped_keys") or [])
    for spec in API_KEYS:
        k = spec["key"]
        if cfg.get(k) or (k in skipped and not force):
            continue
        tag = "REQUIRED" if spec["essential"] else "optional - press Enter to skip"
        print(f"\n{spec['name']} ({tag})\n  why: {spec['why']}\n  get it free: {spec['url']}")
        while True:
            try:
                val = input("  key: ").strip()
            except EOFError:
                val = ""
            if not val:
                if spec["essential"]:
                    print("  this key is required")
                    continue
                skipped.add(k)
                set_value("skipped_keys", json.dumps(sorted(skipped)))
                print("  skipped (run 'videoid keys' to add it later)")
                break
            ok = _check_key(spec, val)
            if ok is False:
                print("  the key was rejected by the service - try again or press Enter to skip")
                continue
            set_value(k, json.dumps(val))
            skipped.discard(k)
            set_value("skipped_keys", json.dumps(sorted(skipped)))
            cfg[k] = val
            print("  saved" + (" (could not verify online)" if ok is None else " and verified"))
            break
    return cfg


def db_path(cfg):
    p = Path(cfg["db_path"])
    return p if p.is_absolute() else ROOT / p

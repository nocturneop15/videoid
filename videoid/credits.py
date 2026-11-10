"""Turn OCR'd credit frames into (name, role) pairs, unique strings, title and year candidates."""
import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz, process

from .log import log
from .textutil import CASED, compact, fold, norm, roman_to_int, script_of, split_camel

# ---------------------------------------------------------------------------
# Role vocabulary (accent-free, lowercase). Longest phrase wins, so specific crew
# titles must be listed to stop e.g. "music supervisor" being read as composer.
# ---------------------------------------------------------------------------
ROLE_PHRASES = {
    "director": [
        "directed by", "director", "a film by", "film by", "un film de", "ein film von", "un film di",
        "una pelicula de", "regie", "rezie", "rezia", "rezija", "rezyseria", "realisation", "realise par",
        "regia", "regia di", "dirigida por", "direccion", "rendezte", "scenar a rezie", "napsal a reziroval",
        "written and directed by", "film rezie",
    ],
    "producer": [
        "produced by", "producer", "producers", "producent", "producenti", "producentka", "produzent",
        "produzenten", "producteur", "producteurs", "prodotto da", "produttore", "productor", "producido por",
        "koproducent", "co producer", "coproducer", "co produced by", "producer by",
    ],
    "exec_producer": ["executive producer", "executive producers", "vykonny producent", "vykonna producentka",
                      "ausfuhrender produzent", "producteur executif"],
    "composer": [
        "music by", "music", "original music", "original music by", "original score", "original score by",
        "score by", "music composed by", "composer", "music composer", "hudba", "hudbu slozil", "hudba slozil",
        "hudbu zkomponoval", "autor hudby", "musik", "musique", "musique originale", "musica", "musica de",
        "zene", "muzyka", "music composed and conducted by", "original music composed by",
    ],
    "writer": [
        "written by", "screenplay", "screenplay by", "story by", "story", "writer", "writers", "screen story by",
        "scenar", "scenare", "namet", "literarni predloha", "podle romanu", "podle knihy", "based on the novel by",
        "based on the book by", "based on", "novel by", "drehbuch", "buch", "scenario", "scenario de",
        "sceneggiatura", "soggetto", "guion", "teleplay by", "adaptation", "dialogue", "dialogues",
        "scenaristka", "scenarista", "autor",
    ],
    "dop": [
        "director of photography", "cinematography", "cinematography by", "cinematographer", "photography",
        "photographed by", "kamera", "kameraman", "hlavni kameraman", "chef operator", "bildgestaltung",
        "fotografia", "direttore della fotografia", "directeur de la photographie", "image", "director de fotografia",
        "zdjecia",
    ],
    "editor": ["edited by", "editor", "editors", "film editor", "film editing by", "editing", "strih", "schnitt",
               "montage", "montaz", "montaggio", "montaje", "edited", "vagas"],
    "production_designer": ["production designer", "production design", "production design by", "architekt",
                            "architekt filmu", "filmovy architekt", "scenografie", "scenograf", "szenenbild",
                            "decors", "scenografia", "art director", "art direction", "vytvarnik"],
    "costume": ["costume designer", "costume design", "costume design by", "costumes", "costumes by", "kostymy",
                "kostymni vytvarnik", "kostymni vytvarnice", "kostumbild", "costumi", "vestuario"],
    "casting": ["casting", "casting by", "casting director", "obsazeni role", "besetzung"],
    "sound": ["sound", "sound by", "sound designer", "sound design", "sound mixer", "re recording mixer",
              "supervising sound editor", "zvuk", "mistr zvuku", "ton", "son", "sonido", "suono", "zvukovy design"],
    "makeup": ["make up", "makeup", "makeup artist", "make up artist", "hair", "masky", "maskerka", "maskenbild"],
    "vfx": ["visual effects", "visual effects supervisor", "special effects", "vfx supervisor", "vizualni efekty",
            "triky", "specialni efekty"],
    "dramaturg": ["dramaturg", "dramaturgie", "dramaturgyne"],
    "first_ad": ["assistant director", "first assistant director", "1st assistant director", "asistent rezie",
                 "pomocny rezie", "pomocna rezie"],
    "production_manager": ["unit production manager", "production manager", "vedouci produkce", "line producer",
                           "produkce"],
    "music_supervisor": ["music supervisor", "music supervisors", "music editor", "dirigent", "conductor",
                         "orchestrated by", "orchestrations", "orchestration", "music performed by"],
    "cast": [
        "starring", "cast", "also starring", "co starring", "featuring", "cast in order of appearance",
        "in order of appearance", "hraji", "hrali", "v hlavnich rolich", "v dalsich rolich", "v hlavni roli",
        "dale hraji", "a dale", "obsazeni", "ucinkuji", "hrajou", "in den hauptrollen", "mit", "avec", "con",
        "with", "and", "cast members", "darsteller", "interpretes", "interpreti", "w rolach", "szereplok",
    ],
    "crew": ["gaffer", "key grip", "best boy", "stunt coordinator", "script supervisor", "location manager",
             "camera operator", "steadicam operator", "focus puller", "colorist", "digital colorist",
             "choreographer", "choreography", "storyboard artist", "set decorator", "property master",
             "transportation captain", "stunts", "kaskaderi", "osvetleni", "asistent kamery", "stunt double",
             "unit publicist", "catering", "craft service", "special thanks", "thanks", "dekujeme"],
}
EXACT_ONLY = {"mit", "avec", "con", "with", "and", "son", "ton", "image", "story", "autor", "buch", "hair",
              "music", "musica", "editing", "edited", "produkce", "cast", "sound"}
ROLE_PRIORITY = ["director", "composer", "dop", "writer", "producer", "exec_producer", "editor", "cast",
                 "production_designer", "costume", "casting", "sound", "dramaturg", "makeup", "vfx",
                 "music_supervisor", "production_manager", "first_ad", "crew", "unknown"]
ROLES = [r for r in ROLE_PRIORITY if r != "unknown"]

# older Czech/Slovak credit wording
for _role, _phrases in {
    "production_designer": ["stavby", "stavby a dekorace", "navrhy dekoraci", "vyprava", "filmova architektura"],
    "costume": ["navrhy kostymu", "kostymy navrhl", "kostymy navrhla"],
    "makeup": ["navrhy masek"],
    "first_ad": ["asistenti rezisera", "asistent rezisera", "asistentka rezie", "asistenti rezie", "rezijni spoluprace",
                 "spoluprace na rezii", "spoluprace rezie"],
    "production_manager": ["vedouci vyroby", "vedouci vyroby filmu", "vyroba"],
    "sound": ["zvukovy mistr", "zvukovy technik", "zvukova spoluprace"],
    "music_supervisor": ["ridi", "orchestr ridi", "hudbu ridi", "diriguje", "filmovy symfonicky orchestr ridi"],
    "cast": ["v hlavnich ulohach hraji", "v hlavnich ulohach", "v hlavnich rolich hraji", "v detskych ulohach hraji",
             "v dalsich ulohach hraji", "v dalsich ulohach", "v ulohach", "ulohach hraji", "dale hraji", "hraje",
             "hraji a zpivaji", "a dale hraji"],
    "crew": ["asistent vyroby", "asistenti vyroby", "skript", "skriptka", "poradce", "odborny poradce", "fotografie",
             "texty pisni", "texty", "spoluprace", "pomocny kameraman", "druhy kameraman", "dekorace"],
}.items():
    ROLE_PHRASES[_role] = ROLE_PHRASES.get(_role, []) + _phrases

# Hindi, Chinese, Japanese, Korean, Russian, Thai, Arabic, Tamil, Telugu, Greek credit headers
for _role, _phrases in {
    "director": ["निर्देशक", "निर्देशन", "निर्देशित", "导演", "導演", "監督", "감독", "연출", "режиссёр", "режиссер",
                 "режиссёр-постановщик", "режиссер-постановщик", "постановка", "ผู้กำกับ", "กำกับการแสดง",
                 "إخراج", "المخرج", "இயக்கம்", "இயக்குனர்", "దర్శకత్వం", "దర్శకుడు", "σκηνοθεσία"],
    "producer": ["निर्माता", "निर्माण", "制片人", "製片人", "制片", "出品人", "プロデューサー", "製作", "제작",
                 "프로듀서", "продюсер", "продюсеры", "ผู้อำนวยการสร้าง", "إنتاج", "المنتج", "தயாரிப்பு",
                 "தயாரிப்பாளர்", "నిర్మాత", "παραγωγή"],
    "exec_producer": ["监制", "監製", "製作総指揮", "총괄 프로듀서", "исполнительный продюсер"],
    "composer": ["संगीत", "संगीतकार", "音乐", "音樂", "作曲", "音楽", "음악", "композитор", "музыка",
                 "ดนตรีประกอบ", "موسيقى", "இசை", "సంగీతం", "μουσική"],
    "writer": ["पटकथा", "कहानी", "लेखक", "संवाद", "编剧", "編劇", "脚本", "原作", "각본", "극본", "원작",
               "сценарий", "сценарист", "บทภาพยนตร์", "سيناريو", "تأليف", "கதை", "திரைக்கதை", "வசனம்",
               "కథ", "మాటలు", "σενάριο"],
    "dop": ["छायांकन", "छायाकार", "摄影", "攝影", "摄影指导", "撮影", "촬영", "оператор", "оператор-постановщик",
            "กำกับภาพ", "تصوير", "مدير التصوير", "ஒளிப்பதிவு", "ఛాయాగ్రహణం", "φωτογραφία"],
    "editor": ["संपादन", "संपादक", "剪辑", "剪輯", "編集", "편집", "монтаж", "ลำดับภาพ", "مونتاج",
               "படத்தொகுப்பு", "కూర్పు", "μοντάζ"],
    "production_designer": ["कला निर्देशक", "美术", "美術", "미술", "художник-постановщик", "கலை"],
    "costume": ["वेशभूषा", "服装", "服裝", "衣装", "의상", "художник по костюмам", "костюмы"],
    "sound": ["ध्वनि", "录音", "錄音", "録音", "음향", "녹음", "звукорежиссёр", "звукорежиссер", "звук"],
    "cast": ["कलाकार", "अभिनीत", "अभिनय", "मुख्य भूमिका", "主演", "演员", "演員", "领衔主演", "出演", "キャスト",
             "출연", "주연", "в ролях", "в главных ролях", "актёры", "роли исполняли", "นักแสดง", "นำแสดงโดย",
             "بطولة", "تمثيل", "நடிப்பு", "நடிகர்கள்", "నటీనటులు", "తారాగణం", "παίζουν", "πρωταγωνιστούν"],
}.items():
    ROLE_PHRASES[_role] = ROLE_PHRASES.get(_role, []) + _phrases


def _phrase_key(ph):
    return re.sub(r"[\W_]+", " ", fold(ph)).strip()


_kw = []
for role, phrases in ROLE_PHRASES.items():
    for ph in phrases:
        _kw.append((_phrase_key(ph), role))
_kw.sort(key=lambda x: -len(x[0]))
_ROLE_OF = {ph: r for ph, r in _kw}
_ROLE_RE = re.compile(
    r"^[\W_]*(" + "|".join(re.escape(ph).replace(r"\ ", r"[\W_]+") for ph, _ in _kw) + r")(?![a-z0-9])")

PARTICLES = {"van", "von", "de", "da", "del", "della", "di", "du", "la", "le", "der", "den", "ten", "ter", "bin",
             "al", "el", "dos", "das", "y", "e", "zu", "st", "mc", "mac", "o", "d"}
# words that make a string NOT a personal name
STOP = set("""
the end copyright all rights reserved pictures picture films film studios studio productions production
entertainment presents present presented company inc ltd llc gmbh sro spol music records record song songs
performed courtesy dolby digital stereo surround color colour technicolor eastmancolor association by of for in
on at to from special thanks made filmed location produced directed written based novel starring distributed
limited corporation corp group media television tv channel country republic city state kodak panavision arri
camera cameras lenses equipment rental post sound visual effects laboratory lab titles title czech ceska ceske
cesky ceskoslovensky ceskoslovenska slovak slovenska barrandov fond kinematografie televize ct rozhlas verze
dabing titulky character himself herself themselves narrator voice voices uncredited original score soundtrack
available album licensed under agreement worldwide international national foundation institute fund council
program programme european union creative europe agency bank department ministerstvo ministry kultury
culture copyrights warning unauthorized motion association mpaa approved certificate number www com net org
cz http https dvd blu ray release group subtitles subtitle encoded ripped presents a an kino cinema hotel
restaurant police officer doctor nurse soldier man woman boy girl guard driver waiter reporter old young
little mr mrs miss dr sir lady lord king queen prince princess captain sergeant lieutenant general colonel
private agent detective professor teacher student mother father son daughter brother sister uncle aunt
grandmother grandfather wife husband friend priest nun bartender waitress clerk manager secretary
laureat laureatka statni stalinovy stalinovy cena ceny narodni umelec umelkyne zasluzily zasluzila zasl nar
ing arch mudr judr phdr rndr prof doc akad mal mgr
rezie rezii rezisera strih hudba kamera stavby masky masek skript zvuk triky scenar namet autor produkce vyroby
vyroba vedouci asistent asistenti asistentka poradce spoluprace rezijni rezini navrhy kostymu kostymy texty pisni
povidka povidky orchestr symfonicky filmovy filmovo sbor detsky detske hraji hraje uloha ulohach role rolich
vypravi dekorace hlavnich dalsich detskych
pro bez pod nad pred mezi jako ktery kdyz ale nebo jen uz prvni druha druhy treti dale ze na ve do od za po se
uvadi uvede vyrobilo vyrobil studio studia statni ustredni pujcovna filmu filmy
""".split())

GIVEN = set()
_gn = Path(__file__).with_name("data") / "given_names.txt"
if _gn.exists():
    for _line in _gn.read_text(encoding="utf-8").splitlines():
        if not _line.startswith("#"):
            GIVEN.update(_line.split())
_ROLE_PHRASE_LIST = [ph for ph, _ in _kw if len(ph) >= 5]

# words that are never part of a person's name in non-Latin credits (companies, generic credit words)
STOP_NONLATIN = {
    "公司", "有限公司", "出品", "联合出品", "制作", "製作委員会", "株式会社", "委員会", "电影", "電影", "映画", "協力",
    "영화", "제작사", "주식회사", "प्रस्तुत", "प्रस्तुति", "फिल्म्स", "प्रोडक्शन", "प्राइवेट", "लिमिटेड",
    "киностудия", "фильм", "студия", "ภาพยนตร์", "บริษัท",
}


def is_given(tok):
    return norm(tok) in GIVEN


def match_role(text):
    """Return (role, remainder) if the text starts with a role phrase, else (None, text).
    Whole-line headers are also matched fuzzily, because OCR mangles them ('HRAIL:' = 'Hrají:')."""
    f = fold(text)
    m = _ROLE_RE.match(f)
    if not m:
        n = norm(text)
        if 4 <= len(n) <= 40 and len(n.split()) <= 5:
            header = text.rstrip().endswith(":")
            hit = process.extractOne(n, _ROLE_PHRASE_LIST, scorer=fuzz.ratio, score_cutoff=78 if header else 88)
            if hit:
                return _ROLE_OF[hit[0]], ""
        return None, text
    phrase = re.sub(r"[\W_]+", " ", m.group(1)).strip()
    rest = text[m.end():].strip(" :.-–—|·•,")
    if phrase in EXACT_ONLY and rest and not re.match(r"^[\W_]*" + re.escape(phrase) + r"\s*:", f):
        # ambiguous short word: only a role if what follows is clearly a person ("with Morgan Freeman")
        if not split_names(rest):
            return None, text
    return _ROLE_OF.get(phrase), rest


def clean_name(text):
    t = split_camel(text.strip(" .,:;·•|-–—*_\"'()[]"))
    t = re.sub(r"\s+", " ", t)
    return t


def _looks_like_uncased_name(t, script):
    """Scripts without capital letters (CJK, Hangul, Indic, Thai, Arabic): shape rules per script."""
    if re.search(r"[\d@#/\\=+<>{}$%&:：]", t) or any(w in t for w in STOP_NONLATIN):
        return False
    if match_role(t)[0]:
        return False
    letters = [c for c in t if c.isalpha()]
    if script == "cjk":       # 2-4 hanzi/kanji, or a katakana name with ・ / spaces (max ~12)
        return 2 <= len(letters) <= 12 and len(t.split()) <= 3
    if script == "hangul":
        return 2 <= len(letters) <= 5 and len(t.split()) <= 2
    words = t.split()
    return 1 <= len(words) <= 4 and all(len(w) >= 2 for w in words) and 3 <= len(letters) <= 40


def looks_like_name(text):
    t = clean_name(text)
    script = script_of(t)
    if script not in CASED:
        return _looks_like_uncased_name(t, script)
    if not (4 <= len(t) <= 48) or re.search(r"[\d@#/\\=+<>{}$%&]", t):
        return False
    toks = t.split(" ")
    if not (2 <= len(toks) <= 5):
        return False
    caps = 0
    for tok in toks:
        core = tok.strip(".'’-")
        if not core:
            return False
        if core.lower() in PARTICLES:
            continue
        if not core[0].isupper() or not re.fullmatch(r"[^\W\d_]+(?:[.'’\-][^\W\d_]*)*\.?", core):
            return False
        if _is_stop(core):
            return False
        caps += 1
    return caps >= 2


def _is_stop(tok):
    n = norm(tok)
    return n in STOP and n not in GIVEN


def _segment(toks):
    """Cut a token run into names, starting a new name at each given name.
    'KAREL EFFA ANTONÍN HOLZINGER' -> [Karel Effa, Antonín Holzinger]; 'UČITEL VÁCLAV ŠVORC' -> [Václav Švorc]."""
    # stop words (titles, role words, decorations) break names apart
    groups, cur = [], []
    for t in toks:
        if _is_stop(t.strip(".")):
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if any(is_given(t) for t in g):
            start = next(i for i, t in enumerate(g) if is_given(t))
            name = []
            for t in g[start:]:
                if is_given(t) and len(name) >= 2:
                    out.append((name, True))
                    name = []
                name.append(t)
            if name:
                out.append((name, True))
        else:
            out.append((g, False))
    names = []
    for toks_, verified in out:
        s = " ".join(toks_)
        if 2 <= len(toks_) <= 4 and looks_like_name(s):
            names.append((s, verified))
    return names


def extract_names(text):
    """All personal names in a text line -> [(name, verified_by_given_name)]."""
    text = clean_name(text)
    if script_of(text) not in CASED:
        # no capitals, no given-name list: separators only (、，／ and spaces between CJK names)
        parts = [p for p in re.split(r"\s*[,;/|•、，／]\s*", text) if p]
        if script_of(text) in ("cjk", "hangul") and len(parts) == 1 and len(text.split()) >= 2 \
                and all(2 <= len(w) <= 4 for w in text.split()):
            parts = text.split()           # "张艺谋 巩俐 葛优" - CJK names separated by spaces
        return [(p, False) for p in parts if looks_like_name(p)]
    parts = re.split(r"\s*(?:[,;/|•·]| and | a | & | und | et | y | e )\s*", text)
    out = []
    for p in parts:
        p = re.sub(r"\s+(?:as|jako|als|v roli)\s+.+$", "", p)          # "Actor as Character"
        # hyphen joins two people ("JAROSLAV SENÍK-FRANTIŠEK ČERNÝ") when a full "Given Surname" follows;
        # otherwise it is a double surname ("Richard Francis-Bruce")
        pieces = re.split(r"\s*-\s*", p)
        chunks = [pieces[0]]
        for pc in pieces[1:]:
            toks = pc.split()
            if len(toks) >= 2 and is_given(toks[0]) and len(chunks[-1].split()) >= 2:
                chunks.append(pc)
            else:
                chunks[-1] += "-" + pc
        for chunk in chunks:
            toks = [t for t in re.split(r"\s+", chunk.strip(" -")) if t]
            out += _segment(toks)
    return out


def split_names(text):
    return [n for n, _ in extract_names(text)]


def is_verified_name(name):
    toks = name.split()
    return any(is_given(t) for t in toks[:2])


# ---------------------------------------------------------------------------
class CreditCollector:
    """Accumulates OCR boxes frame by frame; keeps only unique strings."""

    def __init__(self):
        self.lines = {}          # compact key -> dict(text, conf, t, segment, height, seen)
        self.people = {}         # (compact name, role) -> dict
        self.title_cands = defaultdict(float)
        self.years = defaultdict(int)
        self._role = None
        self._segment = None
        self._last_text_t = -1e9

    # -------- unique strings --------
    def _add_line(self, segment, t, box):
        key = compact(box["text"])
        if len(key) < 2:
            return None
        if key not in self.lines and len(key) >= 6:
            # OCR jitter on scrolling credits: merge near-identical variants
            hit = process.extractOne(key, self.lines.keys(), scorer=fuzz.ratio, score_cutoff=92)
            if hit:
                key = hit[0]
        cur = self.lines.get(key)
        if cur is None:
            self.lines[key] = {"text": box["text"], "conf": box["conf"], "t": round(t, 1), "segment": segment,
                               "height": round(box["h"], 4), "seen": 1}
            return True
        cur["seen"] += 1
        if box["conf"] > cur["conf"]:
            cur["text"], cur["conf"] = box["text"], box["conf"]
        return False

    def _add_person(self, name, role, conf, verified=False):
        name = clean_name(name)
        key = (compact(name), role)
        p = self.people.get(key)
        w = getattr(self, "_weight", 1)
        if p:
            p["seen"] += w
            if conf > p["conf"]:
                p["name"], p["conf"] = name, conf
        else:
            self.people[key] = {"name": name, "role": role, "source": "ocr", "seen": w, "conf": conf,
                                "verified": verified}

    # -------- frame layout parsing --------
    @staticmethod
    def _rows(boxes):
        boxes = sorted(boxes, key=lambda b: b["y"] + b["h"] / 2)
        rows = []
        for b in boxes:
            yc = b["y"] + b["h"] / 2
            if rows:
                r = rows[-1]
                ryc = sum(x["y"] + x["h"] / 2 for x in r) / len(r)
                if abs(yc - ryc) < 0.6 * max(b["h"], max(x["h"] for x in r)):
                    r.append(b)
                    continue
            rows.append([b])
        return [sorted(r, key=lambda b: b["x"]) for r in rows]

    def feed(self, segment, t, boxes, weight=1):
        """boxes: list of dict(text, conf, x, y, w, h) with coordinates relative to frame size.
        weight: how many sampled frames this OCR result stood for (static cards are OCR'd once)."""
        self._weight = weight
        if segment != self._segment:
            self._segment, self._role = segment, None
        # a header only carries over to following frames while text stays on screen continuously
        # (scrolling credits); after a gap (new title card) the old role must not leak onto new names
        if t - self._last_text_t > 3.5:
            self._role = None
        self._last_text_t = t
        new = False
        for b in boxes:
            new |= bool(self._add_line(segment, t, b))
        self._years(boxes)
        rows = self._rows(boxes)
        if segment == "open":
            self._titles(rows, t)
        for row in rows:
            texts = [b["text"].strip() for b in row]
            conf = min(b["conf"] for b in row)
            whole = " ".join(texts)
            role, rest = match_role(whole)
            if role:
                for n, v in (extract_names(rest) if rest else []):
                    self._add_person(n, role, conf, v)
                self._role = role
                continue
            if len(row) >= 2:
                # column layout: "Music ........ Hans Zimmer"
                lrole, lrest = match_role(texts[0])
                if lrole and not lrest:
                    for n, v in extract_names(" ".join(texts[1:])):
                        self._add_person(n, lrole, conf, v)
                    continue
                # "Character ..... Actor" or one word per box ("ŘEDITEL" "BOHUŠ" "HRADIL"): read the whole
                # row (given-name segmentation separates the people) and each cell, keep unique names
                names, seen = [], set()
                for nv in extract_names(whole) + [nv for x in texts for nv in extract_names(x)]:
                    if compact(nv[0]) not in seen:
                        seen.add(compact(nv[0]))
                        names.append(nv)
            else:
                names = extract_names(whole)
            for n, v in names:
                self._add_person(n, self._role or "unknown", conf, v)
        return new

    def _years(self, boxes):
        for b in boxes:
            txt = b["text"]
            f = fold(txt)
            if not re.search(r"(©|\(c\)|copyright|\bmcm|\bmm)", f, re.I) and "©" not in txt:
                continue
            for y in re.findall(r"\b(19[0-9]\d|20[0-4]\d)\b", txt):
                self.years[int(y)] += 1
            for rn in re.findall(r"\b(M{1,2}[CDLXVI]{1,10})\b", txt.upper()):
                v = roman_to_int(rn)
                if v and 1900 <= v <= 2049:
                    self.years[v] += 1

    def _titles(self, rows, t=0.0):
        """Large text in the opening = title candidates. Consecutive large rows are joined
        ('HAŠKOVY POVÍDKY' + 'ZE STARÉHO MOCNÁŘSTVÍ'). Score = biggest size seen (not summed - an actor
        card on screen for 20 s must not beat a title shown for 2 s), with a bonus for the first 2 minutes."""
        if not rows:
            return
        early = 1.5 if t < 120 else 1.0
        big = []
        for row in rows:
            h = max(b["h"] for b in row)
            txt = " ".join(b["text"].strip() for b in row)
            conf = min(b["conf"] for b in row)
            if h < 0.045 or len(txt) < 2 or len(txt) > 70 or conf < 0.65 or match_role(txt)[0]:
                big.append(None)
                continue
            words = [w for w in norm(txt).split() if w]
            if not words or all(w in STOP for w in words):
                big.append(None)
                continue
            # "OSTRE SLEDOVANÉ VLAKY" is shaped like a name - only very large text overrides that;
            # a lone first name in big letters ("FRANTIŠEK") is part of an actor card, not a title
            names = extract_names(txt)
            if ((len(words) == 1 and words[0] in GIVEN) or (h < 0.09 and names)
                    or any(v for _, v in names)):          # contains a real person's name -> an actor card
                big.append(None)
                continue
            big.append((txt, h))
        run = []
        for item in big + [None]:
            if item and (not run or abs(item[1] - run[-1][1]) < 0.35 * run[-1][1]):
                run.append(item)
                continue
            if run:
                for txt, h in run:
                    self.title_cands[txt] = max(self.title_cands[txt], h * early)
                if len(run) > 1:
                    joined = " ".join(x for x, _ in run)
                    self.title_cands[joined] = max(self.title_cands[joined], max(h for _, h in run) * early * 1.2)
            run = [item] if item else []

    # -------- results --------
    def unique_lines(self):
        return sorted(self.lines.values(), key=lambda l: (l["segment"] != "open", l["t"]))

    def result_people(self, min_conf=0.0):
        # names not starting with a known given name must be seen in 2+ frames (filters OCR garbage)
        out = [p for p in self.people.values()
               if p["conf"] >= min_conf and (p.get("verified") or p["seen"] >= 2
                                             or (script_of(p["name"]) not in CASED and p["conf"] >= 0.85))]
        # fold OCR variants ("BOHUMIL ZAHORSKYUS" -> "BOHUMIL ZÁHORSKÝ") into the most frequent spelling
        out.sort(key=lambda p: -p["seen"])
        kept, keys = [], []
        for p in out:
            k = compact(p["name"])
            hit = process.extractOne(k, keys, scorer=fuzz.ratio, score_cutoff=88)
            if hit:
                q = kept[hit[2]]
                q["seen"] += p["seen"]
                if q["role"] == "unknown" and p["role"] != "unknown":
                    q["role"] = p["role"]
                continue
            kept.append(p)
            keys.append(k)
        return kept

    def meta(self):
        titles = sorted(self.title_cands.items(), key=lambda x: -x[1])[:5]
        years = sorted(self.years.items(), key=lambda x: -x[1])[:3]
        return {"title_candidates": [t for t, _ in titles], "copyright_years": [y for y, _ in years]}


# ---------------------------------------------------------------------------
# Optional: local LLM (runs on the GPU via Ollama / LM Studio) reads the unique strings.
# ---------------------------------------------------------------------------
LLM_PROMPT = """You read OCR output of a film's opening titles and end credits.
Lines are unique strings in screen order and may contain OCR errors.
Extract every PERSON credited and their role. Allowed roles:
director, producer, exec_producer, composer, writer, dop, editor, production_designer, costume,
casting, sound, makeup, vfx, dramaturg, cast, crew.
Rules: only real people (no companies, no character names), fix obvious OCR spacing, keep diacritics,
do not invent names. Also give the film title if it is clearly shown, and the copyright year if shown.
Answer with JSON only: {"title": str|null, "year": int|null, "people": [{"name": str, "role": str}]}"""


def detect_llm(http, llm_cfg):
    if llm_cfg.get("enabled") is False:
        return None
    candidates = [llm_cfg["url"]] if llm_cfg.get("url") else ["http://localhost:11434/v1", "http://localhost:1234/v1"]
    for base in candidates:
        base = base.rstrip("/")
        try:
            r = http.s.get(base + "/models", timeout=3)
            models = [m["id"] for m in r.json().get("data", [])]
        except Exception:
            continue
        model = llm_cfg.get("model") or next((m for m in models if "embed" not in m.lower()), None)
        if model:
            return {"url": base, "model": model, "timeout": llm_cfg.get("timeout", 180)}
    if llm_cfg.get("enabled") is True:
        log("  LLM enabled but no local OpenAI-compatible server found", 0)
    return None


def llm_extract(http, llm, lines):
    people, title, year = [], None, None
    chunk = 180
    for i in range(0, len(lines), chunk):
        part = "\n".join(f"[{l['segment']}] {l['text']}" for l in lines[i:i + chunk])
        resp = http.post_json(llm["url"] + "/chat/completions", {
            "model": llm["model"], "temperature": 0,
            "messages": [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": part}],
        }, timeout=llm["timeout"])
        if not resp:
            continue
        try:
            txt = resp["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            data = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        except (KeyError, IndexError, ValueError):
            log("  LLM returned unparsable output", 2)
            continue
        title = title or data.get("title")
        year = year or data.get("year")
        for p in data.get("people") or []:
            n, r = (p.get("name") or "").strip(), (p.get("role") or "unknown").strip().lower()
            if n and 2 <= len(n.split()) <= 5:
                people.append({"name": n, "role": r if r in ROLES else "crew", "source": "llm", "seen": 1})
    return people, title, year

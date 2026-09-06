#!/usr/bin/env python3
"""
LexiPopup Dictionary Pack Generator v2
=======================================
Pack      Target Size   Target Words    Contents
Minimal   ~15 MB        ~10,000         Top WordNet words + Hindi (OMW + fallback)
Standard  ~80 MB        ~146,000        All WordNet (single+multi-word) + Hindi
Full      ~200 MB       ~700K–1M+       Wiktionary English + WordNet + Hindi WordNet (OMW)

Data sources (all free / open):
  WordNet 3.1         — Princeton University         (free for all use)
  CMU Pronouncing     — Carnegie Mellon              (Public Domain)
  Wiktionary English  — kaikki.org pre-parsed JSONL  (CC BY-SA 3.0)
  Hindi / OMW 1.4     — Open Multilingual WordNet    (CC BY 3.0)

Requirements:
  pip install nltk tqdm requests
  python -c "import nltk; [nltk.download(x) for x in ['wordnet','omw-1.4','cmudict','words']]"

Run:
  python scripts/generate_dicts.py
Output written to: output/
"""

import sqlite3, gzip, json, shutil, hashlib, os, sys, time, re
import urllib.request

# ── Optional dependencies ───────────────────────────────────────────────────

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", unit="", **kw):
            self.it = iterable; self.total = total; self.desc = desc; self.n = 0
        def __iter__(self):
            for i, item in enumerate(self.it or []):
                self.n = i + 1
                if self.n % 10000 == 0:
                    print(f"  {self.desc}: {self.n:,}/{self.total or '?'}", flush=True)
                yield item
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def set_postfix_str(self, s): pass

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import words as nltk_words
    from nltk.corpus import cmudict
except ImportError:
    print("ERROR: nltk not installed.  pip install nltk tqdm requests")
    sys.exit(1)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── NLTK data ───────────────────────────────────────────────────────────────

def ensure_nltk():
    for pkg in ['wordnet', 'omw-1.4', 'cmudict', 'words']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            print(f"  Downloading NLTK '{pkg}'…")
            nltk.download(pkg, quiet=True)

ensure_nltk()

# ── CMU pronunciation dict ───────────────────────────────────────────────────

def load_cmu():
    try:
        return {w: phones for w, phones in cmudict.entries()}
    except Exception:
        return {}

CMU_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ",
    "B":"b","CH":"tʃ","D":"d","DH":"ð","EH":"ɛ","ER":"ɜr","EY":"eɪ",
    "F":"f","G":"g","HH":"h","IH":"ɪ","IY":"iː","JH":"dʒ","K":"k",
    "L":"l","M":"m","N":"n","NG":"ŋ","OW":"oʊ","OY":"ɔɪ","P":"p",
    "R":"r","S":"s","SH":"ʃ","T":"t","TH":"θ","UH":"ʊ","UW":"uː",
    "V":"v","W":"w","Y":"j","Z":"z","ZH":"ʒ",
}

def ipa_from_cmu(word, cmu):
    phones = cmu.get(word.lower())
    if not phones:
        return ""
    ipa = "".join(CMU_IPA.get(p.rstrip("012"), p.lower()) for p in phones)
    return f"/{ipa}/"

# ── Difficulty / frequency ───────────────────────────────────────────────────

def difficulty(freq_rank, total, word_len):
    pct = freq_rank / max(total, 1)
    if pct < 0.05 and word_len <= 6: return 1
    elif pct < 0.25: return 2
    elif pct < 0.6:  return 3
    else:            return 4

def freq_score(rank, total):
    return max(1, int(100 - (rank / max(total, 1)) * 99))

# ── Hindi: OMW lookup + curated fallback dict ────────────────────────────────

HINDI_FALLBACK = {
    "the":"यह (Yah)","be":"होना (Hona)","to":"को (Ko)","of":"का (Ka)",
    "and":"और (Aur)","a":"एक (Ek)","in":"में (Mein)","have":"होना (Hona)",
    "for":"के लिए (Ke liye)","not":"नहीं (Nahi)","on":"पर (Par)",
    "good":"अच्छा (Achha)","time":"समय (Samay)","year":"साल (Saal)",
    "water":"पानी (Pani)","fire":"आग (Aag)","air":"हवा (Hawa)",
    "food":"खाना (Khana)","love":"प्यार (Pyaar)","peace":"शांति (Shanti)",
    "happy":"खुश (Khush)","sad":"दुखी (Dukhi)","beautiful":"सुंदर (Sundar)",
    "strong":"मजबूत (Majboot)","fast":"तेज (Tej)","big":"बड़ा (Bada)",
    "small":"छोटा (Chhota)","old":"पुराना (Purana)","new":"नया (Naya)",
    "sun":"सूरज (Suraj)","moon":"चाँद (Chand)","star":"तारा (Tara)",
    "earth":"पृथ्वी (Prithvi)","sky":"आसमान (Aasmaan)","tree":"पेड़ (Ped)",
    "river":"नदी (Nadi)","mountain":"पहाड़ (Pahad)","ocean":"समुद्र (Samudra)",
    "dog":"कुत्ता (Kutta)","cat":"बिल्ली (Billi)","bird":"पक्षी (Pakshi)",
    "fish":"मछली (Machhli)","flower":"फूल (Phool)","fruit":"फल (Phal)",
    "book":"किताब (Kitab)","house":"घर (Ghar)","school":"स्कूल (School)",
    "money":"पैसा (Paisa)","friend":"दोस्त (Dost)","family":"परिवार (Parivar)",
    "mother":"माँ (Maa)","father":"पिता (Pita)","brother":"भाई (Bhai)",
    "sister":"बहन (Behen)","son":"बेटा (Beta)","daughter":"बेटी (Beti)",
    "city":"शहर (Shahar)","country":"देश (Desh)","road":"सड़क (Sadak)",
    "door":"दरवाज़ा (Darwaza)","room":"कमरा (Kamra)","table":"मेज़ (Mez)",
    "chair":"कुर्सी (Kursi)","bed":"बिस्तर (Bistar)","pen":"कलम (Kalam)",
    "word":"शब्द (Shabd)","language":"भाषा (Bhasha)","music":"संगीत (Sangeet)",
    "war":"युद्ध (Yuddh)","law":"कानून (Kanoon)","truth":"सच्चाई (Sacchai)",
    "dream":"सपना (Sapna)","hope":"आशा (Asha)","fear":"डर (Dar)",
    "joy":"खुशी (Khushi)","pain":"दर्द (Dard)","death":"मृत्यु (Mrityu)",
    "life":"जीवन (Jivan)","health":"स्वास्थ्य (Swasthya)","mind":"मन (Man)",
    "heart":"दिल (Dil)","body":"शरीर (Shareer)","eye":"आँख (Aankh)",
    "hand":"हाथ (Haath)","morning":"सुबह (Subah)","night":"रात (Raat)",
    "day":"दिन (Din)","week":"हफ़्ता (Hafta)","month":"महीना (Mahina)",
    "knowledge":"ज्ञान (Gyan)","education":"शिक्षा (Shiksha)",
    "science":"विज्ञान (Vigyan)","freedom":"स्वतंत्रता (Swatantrata)",
    "wisdom":"बुद्धि (Buddhi)","courage":"साहस (Sahas)","patience":"धैर्य (Dhairya)",
    "serendipity":"सुखद संयोग (Sukhad sanyog)","ephemeral":"क्षणिक (Kshanik)",
    "eloquent":"वाकपटु (Vakpatu)","resilient":"लचीला (Lacheela)",
    "ambiguous":"अस्पष्ट (Aspasht)","pragmatic":"व्यावहारिक (Vyavaharik)",
    "meticulous":"सावधान (Savdhan)","ubiquitous":"सर्वव्यापी (Sarvavyapi)",
    "tenacious":"दृढ़ (Dridh)","candid":"स्पष्टवादी (Spashtvadi)",
    "clandestine":"गुप्त (Gupt)","anomaly":"विसंगति (Visangati)",
    "altruistic":"परोपकारी (Paropkari)","diligent":"परिश्रमी (Parishraami)",
    "enigmatic":"रहस्यमय (Rahasyamay)","benevolent":"उदार (Udaar)",
    "audacious":"साहसी (Sahasi)","stoic":"उदासीन (Udaasin)",
    "verbose":"वाचाल (Vachal)","succinct":"संक्षिप्त (Sankshipt)",
    "plethora":"प्रचुरता (Prachurata)","profound":"गहरा (Gahra)",
    "procrastination":"टालमटोल (Taalmatol)","equanimity":"समभाव (Samabhav)",
}

# OMW Hindi lemma cache
_OMW_HINDI_CACHE: dict = {}

def get_hindi_omw(word: str) -> str:
    """Return Hindi Devanagari from Open Multilingual WordNet (OMW 1.4)."""
    if word in _OMW_HINDI_CACHE:
        return _OMW_HINDI_CACHE[word]
    result = ""
    try:
        for ss in wn.synsets(word):
            hin = ss.lemma_names('hin')
            if hin:
                result = hin[0].replace('_', ' ')
                break
    except Exception:
        pass
    _OMW_HINDI_CACHE[word] = result
    return result

def get_hindi(word: str, include_hindi: bool) -> str:
    if not include_hindi:
        return ""
    # 1. curated fallback (has transliteration)
    h = HINDI_FALLBACK.get(word.lower(), "")
    if h:
        return h
    # 2. OMW (Devanagari only — no transliteration available)
    h = get_hindi_omw(word.lower())
    return h

# ── POS helpers ──────────────────────────────────────────────────────────────

WN_POS = {wn.NOUN:"noun", wn.VERB:"verb", wn.ADJ:"adjective",
          wn.ADJ_SAT:"adjective", wn.ADV:"adverb"}

def pos_label(ss):
    return WN_POS.get(ss.pos(), "other")

# ── WordNet collector (all single-word AND hyphenated/compound) ──────────────

def collect_wordnet(cmu: dict) -> dict:
    """
    Returns word_map: word → entry dict.
    Includes all WordNet lemmas (single-word + hyphenated, ~155K).
    """
    print("\n[WordNet] Building word index…")
    word_map: dict = {}
    all_ss = list(wn.all_synsets())

    for ss in tqdm(all_ss, desc="WordNet synsets", total=len(all_ss)):
        for lemma in ss.lemmas():
            raw = lemma.name()
            # Normalise: replace underscores with spaces, keep hyphens
            word = raw.replace('_', ' ').strip().lower()
            if not word or len(word) > 45:
                continue
            # Allow letters, spaces, hyphens — no digits/symbols
            if not re.fullmatch(r"[a-z][a-z '\-]*", word):
                continue

            syns = list({l.name().replace('_',' ')
                         for l in ss.lemmas() if l.name() != raw})[:8]
            ants = list({ant.name().replace('_',' ')
                         for l in ss.lemmas()
                         for ant in l.antonyms()})[:6]
            examples = ss.examples()[:2]
            score = len(examples)*3 + len(ants)*2 + len(ss.definition())

            if word not in word_map or score > word_map[word]['_score']:
                word_map[word] = {
                    'word':            word,
                    'pos':             pos_label(ss),
                    'pronunciation':   ipa_from_cmu(word, cmu),
                    'short_meaning':   ss.definition()[:120],
                    'detailed_meaning':ss.definition(),
                    'example_sentence':examples[0] if examples else "",
                    'synonyms':        syns,
                    'antonyms':         ants,
                    'etymology':       "",
                    'source':          'wordnet',
                    '_score':          score,
                }

    print(f"  WordNet: {len(word_map):,} entries")
    return word_map


def rank_wordnet(word_map: dict) -> list:
    """
    Sort by commonness (NLTK words corpus membership + length).
    Returns list of words ordered most-common first.
    """
    try:
        common = set(w.lower() for w in nltk_words.words())
    except Exception:
        common = set()

    ordered = sorted(word_map.keys(),
                     key=lambda w: (w not in common, len(w), w))
    total = len(ordered)
    for rank, w in enumerate(ordered):
        word_map[w]['frequency_rating'] = freq_score(rank, total)
        word_map[w]['difficulty_level'] = difficulty(rank, total, len(w))
    return ordered


# ── Wiktionary via kaikki.org ─────────────────────────────────────────────────

KAIKKI_URL = ("https://kaikki.org/dictionary/English/"
              "kaikki.org-dictionary-English.jsonl")

KAIKKI_URL_FALLBACK = ("https://kaikki.org/dictionary/English/"
                       "kaikki.org-dictionary-English-all.jsonl")

def _stream_download(url: str, dest: str):
    """Download url to dest with progress. Uses requests if available."""
    print(f"  Downloading {url}")
    print(f"  → {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    t0 = time.time()

    if HAS_REQUESTS:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            done = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        mb  = done / (1 << 20)
                        elapsed = time.time() - t0
                        speed   = mb / max(elapsed, 0.1)
                        print(f"\r  {pct}%  {mb:.0f}/{total/(1<<20):.0f} MB"
                              f"  {speed:.1f} MB/s", end="", flush=True)
    else:
        urllib.request.urlretrieve(url, dest)

    size_mb = os.path.getsize(dest) / (1 << 20)
    elapsed = time.time() - t0
    print(f"\n  Done: {size_mb:.1f} MB in {elapsed:.0f}s")


def _download_wiktionary(dest: str):
    """
    Try primary URL, then fallback URL.
    Raises SystemExit with a clear message if both fail — the Full pack
    is not optional; the workflow must fail so the user re-runs it.
    """
    last_error = None
    for url in [KAIKKI_URL, KAIKKI_URL_FALLBACK]:
        try:
            _stream_download(url, dest)
            return  # success
        except Exception as e:
            last_error = e
            print(f"\n  ⚠️  {url} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)

    print("\n" + "="*65)
    print("❌  FATAL: Wiktionary download failed from all known URLs.")
    print()
    print("   Primary URL tried:")
    print(f"     {KAIKKI_URL}")
    print("   Fallback URL tried:")
    print(f"     {KAIKKI_URL_FALLBACK}")
    print()
    print("   The Full pack (~700K+ words) requires this file.")
    print("   The workflow will now FAIL so you can re-run it once")
    print("   kaikki.org is reachable again.")
    print()
    print("   To check the current URL, visit:")
    print("     https://kaikki.org/dictionary/English/index.html")
    print("="*65)
    sys.exit(1)


def parse_wiktionary(path: str, wordnet_map: dict, cmu: dict) -> dict:
    """
    Stream-parse the kaikki.org JSONL file.
    Returns wikt_map: word → entry dict.
    Words already in wordnet_map are enriched (IPA, etymology, examples added);
    new words are added fresh.
    
    FIXED: No f.tell() call to avoid OSError. Manual byte tracking instead.
    """
    print(f"\n[Wiktionary] Parsing {path}…")
    wikt_map: dict = {}
    skipped = 0
    t0 = time.time()
    file_size = os.path.getsize(path)
    bytes_read = 0

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for lineno, line in enumerate(f, 1):
            bytes_read += len(line.encode('utf-8'))
            
            if lineno % 200_000 == 0:
                mb_read = bytes_read / (1 << 20)
                elapsed = time.time() - t0
                print(f"  {lineno:,} lines | {mb_read:.0f}/{file_size/(1<<20):.0f} MB"
                      f" | {len(wikt_map):,} entries | {elapsed:.0f}s", flush=True)

            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only English words
            if obj.get('lang_code') != 'en':
                skipped += 1
                continue

            word = obj.get('word', '').strip().lower()
            if not word or len(word) > 45:
                continue
            if not re.fullmatch(r"[a-z][a-z '\-]*", word):
                continue

            pos = obj.get('pos', '')
            pos_norm = {
                'noun':'noun','verb':'verb','adj':'adjective',
                'adv':'adverb','name':'noun','phrase':'phrase',
                'prep':'preposition','conj':'conjunction',
                'intj':'interjection','pron':'pronoun','num':'numeral',
                'det':'determiner','article':'article','abbrev':'abbreviation',
                'character':'other','symbol':'other','punct':'other',
            }.get(pos, pos or 'other')

            # Senses → definitions
            senses = obj.get('senses', [])
            definitions = []
            examples    = []
            for s in senses:
                glosses = s.get('glosses', [])
                if glosses:
                    # Skip senses that are just "alternative form of X" etc.
                    g = glosses[-1]
                    if not re.match(r'^(alternative|obsolete|archaic|dated)\b', g, re.I):
                        definitions.append(g)
                for ex in s.get('examples', []):
                    t = ex.get('text', '').strip()
                    if t and len(t) < 300:
                        examples.append(t)

            if not definitions:
                continue

            short_def    = definitions[0][:120]
            detailed_def = " | ".join(definitions[:5])

            # IPA pronunciation
            ipa = ""
            for sound in obj.get('sounds', []):
                ipa_raw = sound.get('ipa', '')
                if ipa_raw:
                    ipa = ipa_raw if ipa_raw.startswith('/') else f"/{ipa_raw}/"
                    break
            if not ipa:
                ipa = ipa_from_cmu(word, cmu)

            # Etymology
            etym = obj.get('etymology_text', '')[:300]

            # Synonyms / antonyms
            syns = [s['word'] for s in obj.get('synonyms', [])
                    if isinstance(s, dict) and s.get('word')][:8]
            ants = [s['word'] for s in obj.get('antonyms', [])
                    if isinstance(s, dict) and s.get('word')][:6]

            entry = {
                'word':             word,
                'pos':              pos_norm,
                'pronunciation':    ipa,
                'short_meaning':    short_def,
                'detailed_meaning': detailed_def,
                'example_sentence': examples[0] if examples else "",
                'synonyms':         syns,
                'antonyms':         ants,
                'etymology':        etym,
                'source':           'wiktionary',
                'frequency_rating': 50,
                'difficulty_level': 2,
                '_score':           len(definitions)*3 + len(etym) + len(examples)*2,
            }

            # Keep the best-scoring entry per word
            if word not in wikt_map or entry['_score'] > wikt_map[word]['_score']:
                wikt_map[word] = entry

    elapsed = time.time() - t0
    print(f"  Wiktionary: {len(wikt_map):,} English entries parsed in {elapsed:.0f}s"
          f" (skipped {skipped:,} non-English lines)")
    return wikt_map


def merge_maps(wordnet_map: dict, wikt_map: dict,
               wn_ordered: list) -> dict:
    """
    Merge WordNet + Wiktionary.
    Wiktionary is richer (more words, IPA, etymology);
    WordNet is authoritative for synonyms/antonyms.
    Strategy:
      - Start with Wiktionary as base for words it has
      - Fill missing fields (synonyms, antonyms) from WordNet
      - Add WordNet-only words
      - Re-rank frequency for merged set
    """
    print("\n[Merge] Combining WordNet + Wiktionary…")

    # Build wn freq rank map
    wn_rank = {w: i for i, w in enumerate(wn_ordered)}
    wn_total = len(wn_ordered)

    merged: dict = {}

    # 1. Wiktionary base
    for word, we in wikt_map.items():
        entry = dict(we)
        # Enrich with WordNet synonyms/antonyms if available
        if word in wordnet_map:
            wne = wordnet_map[word]
            if not entry['synonyms']:
                entry['synonyms'] = wne.get('synonyms', [])
            if not entry['antonyms']:
                entry['antonyms'] = wne.get('antonyms', [])
            if not entry['pronunciation']:
                entry['pronunciation'] = wne.get('pronunciation', '')
            if not entry['etymology']:
                entry['etymology'] = wne.get('etymology', '')
            # Use WordNet IPA if Wiktionary didn't have one
            rank = wn_rank.get(word, wn_total)
            entry['frequency_rating'] = freq_score(rank, wn_total)
            entry['difficulty_level'] = difficulty(rank, wn_total, len(word))
        merged[word] = entry

    # 2. WordNet-only words (not in Wiktionary)
    wn_only = 0
    for word in wn_ordered:
        if word not in merged:
            merged[word] = dict(wordnet_map[word])
            wn_only += 1

    print(f"  Wiktionary: {len(wikt_map):,} | WordNet-only additions: {wn_only:,}")
    print(f"  Merged total: {len(merged):,}")
    return merged


# ── SQLite schema + builder ──────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS dictionary_cache (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    word                TEXT    UNIQUE NOT NULL,
    pronunciation       TEXT    DEFAULT '',
    part_of_speech      TEXT    DEFAULT '',
    short_meaning       TEXT    DEFAULT '',
    detailed_meaning    TEXT    DEFAULT '',
    hindi_meaning       TEXT    DEFAULT '',
    hindi_pronunciation TEXT    DEFAULT '',
    example_sentence    TEXT    DEFAULT '',
    synonyms            TEXT    DEFAULT '[]',
    antonyms            TEXT    DEFAULT '[]',
    etymology           TEXT    DEFAULT '',
    difficulty_level    INTEGER DEFAULT 2,
    frequency_rating    INTEGER DEFAULT 50,
    source              TEXT    DEFAULT '',
    created_at          INTEGER DEFAULT 0,
    last_accessed       INTEGER DEFAULT 0,
    access_count        INTEGER DEFAULT 0,
    is_favorite         INTEGER DEFAULT 0,
    user_note           TEXT    DEFAULT '',
    last_reviewed       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_word ON dictionary_cache(word);
CREATE INDEX IF NOT EXISTS idx_freq ON dictionary_cache(frequency_rating);
CREATE INDEX IF NOT EXISTS idx_diff ON dictionary_cache(difficulty_level);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO dictionary_cache
  (word, pronunciation, part_of_speech, short_meaning, detailed_meaning,
   hindi_meaning, hindi_pronunciation, example_sentence,
   synonyms, antonyms, etymology, difficulty_level, frequency_rating, source)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

def build_db(path: str, words_iter, source_tag: str,
             cmu: dict, include_hindi: bool, total_hint: int = 0):
    """Write words_iter (iterable of entry dicts) to SQLite at path."""
    print(f"  Writing {path}…")
    conn = sqlite3.connect(path)
    conn.executescript(CREATE_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    batch = []
    count = 0
    t0 = time.time()

    for w in words_iter:
        word  = w['word']
        hindi = get_hindi(word, include_hindi)
        pron  = w.get('pronunciation') or ipa_from_cmu(word, cmu)

        batch.append((
            word,
            pron,
            w.get('pos', ''),
            w.get('short_meaning', ''),
            w.get('detailed_meaning', ''),
            hindi,
            "",   # hindi_pronunciation (transliteration embedded in hindi_meaning)
            w.get('example_sentence', ''),
            json.dumps(w.get('synonyms', [])),
            json.dumps(w.get('antonyms', [])),
            w.get('etymology', ''),
            w.get('difficulty_level', 2),
            w.get('frequency_rating', 50),
            w.get('source', source_tag),
        ))
        count += 1

        if len(batch) >= 1000:
            conn.executemany(INSERT_SQL, batch)
            conn.commit()
            batch.clear()

        if count % 100_000 == 0:
            elapsed = time.time() - t0
            hint = f"/{total_hint:,}" if total_hint else ""
            print(f"    {count:,}{hint} rows  ({elapsed:.0f}s)", flush=True)

    if batch:
        conn.executemany(INSERT_SQL, batch)
        conn.commit()

    final = conn.execute("SELECT COUNT(*) FROM dictionary_cache").fetchone()[0]
    conn.close()
    elapsed = time.time() - t0
    print(f"  Done: {final:,} words in {path}  ({elapsed:.0f}s)")
    return final


def gzip_db(db_path: str, gz_path: str):
    print(f"  Compressing → {gz_path}…")
    with open(db_path, 'rb') as fi, gzip.open(gz_path, 'wb', compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)
    size_mb = os.path.getsize(gz_path) / (1 << 20)
    print(f"  Compressed: {size_mb:.1f} MB")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("output", exist_ok=True)

    print("=" * 65)
    print("LexiPopup Dictionary Pack Generator v2")
    print("=" * 65)

    # ── Step 1: Load CMU ─────────────────────────────────────────────────────
    print("\n[CMU] Loading pronunciation dictionary…")
    cmu = load_cmu()
    print(f"  {len(cmu):,} CMU entries loaded")

    # ── Step 2: WordNet ──────────────────────────────────────────────────────
    wordnet_map = collect_wordnet(cmu)
    wn_ordered  = rank_wordnet(wordnet_map)
    print(f"  WordNet ranked: {len(wn_ordered):,} words")

    # ── Step 3: Download + parse Wiktionary (required for Full pack) ─────────────
    wikt_raw = "output/kaikki-en.jsonl"
    # Accept old cached filename from previous runs too
    if not os.path.exists(wikt_raw) and os.path.exists("output/kaikki-en.json"):
        wikt_raw = "output/kaikki-en.json"

    if os.path.exists(wikt_raw):
        size_mb = os.path.getsize(wikt_raw) / (1 << 20)
        print(f"\n[Wiktionary] Using cached {wikt_raw} ({size_mb:.0f} MB)")
    else:
        wikt_raw = "output/kaikki-en.jsonl"
        print("\n[Wiktionary] Downloading (~2.7 GB — required for Full pack)…")
        _download_wiktionary(wikt_raw)   # exits with code 1 if both URLs fail

    checksums = {}

    # ── MINIMAL pack (~10,000 words) ──────────────────────────────────────────
    print("\n" + "="*65)
    print("=== MINIMAL pack (~10,000 most common words + Hindi) ===")
    minimal_words = [wordnet_map[w] for w in wn_ordered[:10_000]]
    db = "output/dict_minimal_v1.db"
    gz = "output/dict_minimal_v1.db.gz"
    build_db(db, minimal_words, "minimal", cmu,
             include_hindi=True, total_hint=10_000)
    gzip_db(db, gz)
    checksums["MINIMAL"] = sha256(gz)
    os.remove(db)
    print(f"  SHA-256: {checksums['MINIMAL']}")

    # ── STANDARD pack (~155,000 WordNet words + Hindi) ────────────────────────
    print("\n" + "="*65)
    print("=== STANDARD pack (~155,000 WordNet words + Hindi via OMW) ===")
    standard_words = [wordnet_map[w] for w in wn_ordered]
    db = "output/dict_standard_v1.db"
    gz = "output/dict_standard_v1.db.gz"
    build_db(db, standard_words, "standard", cmu,
             include_hindi=True, total_hint=len(standard_words))
    gzip_db(db, gz)
    checksums["STANDARD"] = sha256(gz)
    os.remove(db)
    print(f"  SHA-256: {checksums['STANDARD']}")

    # ── FULL pack (Wiktionary + WordNet + Hindi OMW) ──────────────────────────
    # NOTE: If we reach here, wikt_raw is guaranteed to exist (download succeeded
    # or was cached). _download_wiktionary() calls sys.exit(1) on failure so
    # the workflow always fails rather than silently skipping this pack.
    print("\n" + "="*65)
    print("=== FULL pack (Wiktionary + WordNet + Hindi WordNet) ===")
    wikt_map   = parse_wiktionary(wikt_raw, wordnet_map, cmu)
    merged_map = merge_maps(wordnet_map, wikt_map, wn_ordered)
    print(f"  Total entries to write: {len(merged_map):,}")
    # Sort full pack by frequency descending so most-common words are inserted first
    full_words = sorted(merged_map.values(),
                        key=lambda e: -e.get('frequency_rating', 50))
    db = "output/dict_full_v1.db"
    gz = "output/dict_full_v1.db.gz"
    build_db(db, full_words, "full", cmu,
             include_hindi=True, total_hint=len(full_words))
    gzip_db(db, gz)
    checksums["FULL"] = sha256(gz)
    os.remove(db)
    print(f"  SHA-256: {checksums['FULL']}")

    # Clean up large download cache to free runner disk space
    if os.path.exists(wikt_raw):
        print(f"\n[Cleanup] Removing {wikt_raw}…")
        os.remove(wikt_raw)

    # ── Checksums file ────────────────────────────────────────────────────────
    lines = [
        "# LexiPopup Dictionary Pack SHA-256 Checksums",
        "# Auto-generated — do not edit manually",
        "",
        f"MINIMAL  = {checksums['MINIMAL']}",
        f"STANDARD = {checksums['STANDARD']}",
        f"FULL     = {checksums['FULL']}",
    ]
    with open("output/checksums.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("ALL DONE! All 3 packs built successfully. Files in output/:")
    skip_files = {"kaikki-en.json", "kaikki-en.jsonl"}
    for fname in sorted(os.listdir("output")):
        if fname in skip_files:
            continue
        sz = os.path.getsize(f"output/{fname}") / (1 << 20)
        print(f"  {fname:<40} {sz:>7.1f} MB")
    print()
    print("Word counts per pack:")
    print(f"  MINIMAL   10,000 words  (top WordNet by frequency)")
    print(f"  STANDARD  ~{len(wn_ordered):,} words  (all WordNet)")
    print(f"  FULL      ~{len(merged_map):,} words  (Wiktionary + WordNet merged)")
    print()
    print("Checksums:")
    for k, v in checksums.items():
        print(f"  {k:<10} {v}")
    print()
    print("These checksums have been written to output/checksums.txt")
    print("The standalone workflow publishes them with the dictionary release.")
    print("="*65)


if __name__ == "__main__":
    main()
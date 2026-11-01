# ─────────────────────────────────────────────────────────────────────────────
# nêhiyawêwin orthography: SRO, syllabics, and the checks that keep a sentence
# from being half English.
#
# The conversion tables are the ones behind itwêwina (GiellaLT lang-crk,
# Latn-to-Cans.xfscript and Cans-to-Latn.xfscript, ALTLab / University of
# Alberta with the Maskwacîs Education Schools Commission and First Nations
# University). They are data, not invention, and every rule below is tested
# against the canonical example atim -> ᐊᑎᒼ.
# ─────────────────────────────────────────────────────────────────────────────

SRO_VOWELS = ("a", "e", "i", "o")
LONG_VOWELS = {"a": "â", "e": "ê", "i": "î", "o": "ô"}

_CONSONANT_SERIES = {
    "c": {"a": "ᒐ", "i": "ᒋ", "o": "ᒍ", "â": "ᒑ", "ê": "ᒉ", "î": "ᒌ", "ô": "ᒎ"},
    "k": {"a": "ᑲ", "i": "ᑭ", "o": "ᑯ", "â": "ᑳ", "ê": "ᑫ", "î": "ᑮ", "ô": "ᑰ"},
    "m": {"a": "ᒪ", "i": "ᒥ", "o": "ᒧ", "â": "ᒫ", "ê": "ᒣ", "î": "ᒦ", "ô": "ᒨ"},
    "n": {"a": "ᓇ", "i": "ᓂ", "o": "ᓄ", "â": "ᓈ", "ê": "ᓀ", "î": "ᓃ", "ô": "ᓅ"},
    "p": {"a": "ᐸ", "i": "ᐱ", "o": "ᐳ", "â": "ᐹ", "ê": "ᐯ", "î": "ᐲ", "ô": "ᐴ"},
    "s": {"a": "ᓴ", "i": "ᓯ", "o": "ᓱ", "â": "ᓵ", "ê": "ᓭ", "î": "ᓰ", "ô": "ᓲ"},
    "t": {"a": "ᑕ", "i": "ᑎ", "o": "ᑐ", "â": "ᑖ", "ê": "ᑌ", "î": "ᑏ", "ô": "ᑑ"},
    "y": {"a": "ᔭ", "i": "ᔨ", "o": "ᔪ", "â": "ᔮ", "ê": "ᔦ", "î": "ᔩ", "ô": "ᔫ"},
    "w": {"a": "ᐘ", "i": "ᐏ", "o": "ᐓ", "â": "ᐚ", "ê": "ᐍ", "î": "ᐑ", "ô": "ᐕ"},
}
_W_SERIES = {
    "c": {"a": "ᒝ", "i": "ᒕ", "o": "ᒙ", "â": "ᒟ", "ê": "ᒓ", "î": "ᒗ", "ô": "ᒛ"},
    "k": {"a": "ᑿ", "i": "ᑷ", "o": "ᑻ", "â": "ᒁ", "ê": "ᑵ", "î": "ᑹ", "ô": "ᑽ"},
    "m": {"a": "ᒷ", "i": "ᒯ", "o": "ᒳ", "â": "ᒹ", "ê": "ᒭ", "î": "ᒱ", "ô": "ᒵ"},
    "n": {"a": "ᓌ", "i": "ᣇ", "o": "ᣋ", "â": "ᓎ", "ê": "ᓊ", "î": "ᣉ", "ô": "ᣍ"},
    "p": {"a": "ᑅ", "i": "ᐽ", "o": "ᑁ", "â": "ᑇ", "ê": "ᐻ", "î": "ᐿ", "ô": "ᑃ"},
    "s": {"a": "ᔁ", "i": "ᓹ", "o": "ᓽ", "â": "ᔃ", "ê": "ᓷ", "î": "ᓻ", "ô": "ᓿ"},
    "t": {"a": "ᑢ", "i": "ᑚ", "o": "ᑞ", "â": "ᑤ", "ê": "ᑘ", "î": "ᑜ", "ô": "ᑠ"},
    "y": {"a": "ᔺ", "i": "ᔲ", "o": "ᔶ", "â": "ᔼ", "ê": "ᔰ", "î": "ᔴ", "ô": "ᔸ"},
}
_VOWEL_GLYPHS = {"a": "ᐊ", "i": "ᐃ", "o": "ᐅ", "â": "ᐋ", "ê": "ᐁ", "î": "ᐄ", "ô": "ᐆ"}
_TH_SERIES = {"a": "ᖬ", "e": "ᖧ", "i": "ᖨ", "o": "ᖪ"}
SYLLABICS_FINALS = {
    "c": "ᐨ", "h": "ᐦ", "k": "ᐠ", "m": "ᒼ", "n": "ᐣ", "p": "ᑊ",
    "s": "ᐢ", "t": "ᐟ", "w": "ᐤ", "y": "ᕀ", "l": "ᓬ", "r": "ᕒ",
}
_FINALS_TO_SRO = {v: k for k, v in SYLLABICS_FINALS.items()}
_HK_GLYPH = "ᕽ"
_FULL_STOP = "᙮"
_DOTTED_W = "ᐧ"

_SRO_TO_CANS: dict[str, str] = {}


def _add(sro: str, cans: str) -> None:
    _SRO_TO_CANS[sro] = cans


for _c, _row in _W_SERIES.items():
    for _v, _g in _row.items():
        _add(_c + "w" + _v, _g)
for _c, _row in _CONSONANT_SERIES.items():
    for _v, _g in _row.items():
        _add(_c + _v, _g)
for _c, _row in _TH_SERIES.items():
    _add("th" + _c, _row)
for _v, _g in _VOWEL_GLYPHS.items():
    _add(_v, _g)
_add("hk", _HK_GLYPH)
for _c, _g in SYLLABICS_FINALS.items():
    _add(_c, _g)
_add(".", _FULL_STOP)
_add("e", _VOWEL_GLYPHS["ê"])

_CANS_TO_SRO: dict[str, str] = {}
for _sro, _cans in _SRO_TO_CANS.items():
    if len(_cans) == 1 and _sro != "e":
        _CANS_TO_SRO.setdefault(_cans, _sro)

_COMPOSED_W: dict[str, str] = {}
for _c, _row in _CONSONANT_SERIES.items():
    for _v, _g in _row.items():
        _COMPOSED_W[_g] = _c + "w" + _v
for _v, _g in _VOWEL_GLYPHS.items():
    _COMPOSED_W[_g] = "w" + _v

_SYLLABICS_RANGES = ((0x1400, 0x167F), (0x18B0, 0x18FF))

CREE_LETTERS = set("ptkcsmnhwy" "lr" "aeio" "âêîô")


def macron_to_circumflex(text: str) -> str:
    return text.translate(str.maketrans({
        "ā": "â", "ē": "ê", "ī": "î", "ō": "ô",
        "Ā": "Â", "Ē": "Ê", "Ī": "Î", "Ō": "Ô",
    }))


def is_syllabics(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _SYLLABICS_RANGES)


def looks_like_syllabics(text: str) -> bool:
    return any(is_syllabics(ch) for ch in str(text))


def normalize_sro(text: str) -> str:
    """NFC, macrons to circumflexes, lowercase, collapsed whitespace.

    Long-vowel marks are preserved: removing them would turn different words
    into the same string (nêhiyaw / nehiyaw), which is exactly the damage a
    lazy normaliser does to a language that marks length.
    """
    text = unicodedata.normalize("NFC", str(text))
    text = macron_to_circumflex(text)
    text = text.replace("|", " ").replace("\u2010", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def to_syllabics(sro: str) -> str:
    text = normalize_sro(sro).replace("e", "ê")
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        for size in (3, 2, 1):
            chunk = text[i:i + size]
            if len(chunk) == size and chunk in _SRO_TO_CANS:
                out.append(_SRO_TO_CANS[chunk])
                i += size
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def to_sro(syllabics: str, long_vowels: str = "circumflex") -> str:
    text = unicodedata.normalize("NFC", str(syllabics))
    out: list[str] = []
    last_glyph: str | None = None
    for ch in text:
        if ch == _DOTTED_W:
            if last_glyph is not None:
                out[-1] = _COMPOSED_W.get(last_glyph, out[-1] + "w")
                last_glyph = None
            else:
                out.append("w")
            continue
        last_glyph = None
        if ch in _CANS_TO_SRO:
            out.append(_CANS_TO_SRO[ch])
            last_glyph = ch
        elif ch in _FINALS_TO_SRO:
            out.append(_FINALS_TO_SRO[ch])
        elif ch == _HK_GLYPH:
            out.append("hk")
        elif ch == _FULL_STOP:
            out.append(".")
        else:
            out.append(ch)
    text = "".join(out)
    if long_vowels == "macron":
        text = text.translate(str.maketrans({
            "â": "ā", "ê": "ē", "î": "ī", "ô": "ō",
        }))
    return text


# Words that carry almost no meaning on their own but identify a language.
_EN_STOP = {
    "the", "and", "of", "to", "in", "is", "was", "were", "that", "this",
    "for", "with", "he", "she", "they", "his", "her", "their", "you", "not",
    "but", "from", "have", "has", "had", "will", "would", "there", "what",
    "when", "who", "how", "all", "one", "two", "on", "at", "by", "as", "it",
}
_CR_STOP = {
    "êkwa", "mîna", "kî", "ê", "kâ", "ta", "ni", "ki", "wa", "ôma", "ôki",
    "êsa", "kêkway", "awiyak", "niya", "kiya", "êkosi", "êkotê", "itwêw",
    "itêw", "ayâw", "nipâw", "mîcisow", "pîhtokêw", "nama", "namôya",
}


def lang_guess(text: str) -> str:
    """'crk', 'eng' or 'mixed' from the words that actually appear.

    Character ratios alone are not enough: 'the' is made only of letters Cree
    uses. Function words and long-vowel marks are the reliable signal.
    """
    t = normalize_sro(text)
    if looks_like_syllabics(t):
        return "crk"
    words = re.findall(r"[a-zâêîô']+", t)
    if len(words) < 2:
        return "crk" if re.search(r"[âêîô]", t) else "eng"
    en = sum(w in _EN_STOP for w in words) / len(words)
    cr = sum(w in _CR_STOP for w in words) / len(words)
    diacritic = sum(bool(re.search(r"[âêîô]", w)) for w in words) / len(words)
    if cr > en and (cr + diacritic) > 0.03:
        return "crk"
    if en > cr and en > 0.04:
        return "eng"
    # Unknown but plausible Cree: short words, few Latin-only consonants.
    odd = sum(bool(re.search(r"[bdfgjqvxz]", w)) for w in words) / len(words)
    return "crk" if odd < 0.05 and diacritic > 0 else "mixed"


def cree_word_ratio(text: str) -> float:
    words = re.findall(r"[^\W\d_]+", normalize_sro(text), flags=re.UNICODE)
    if not words:
        return 0.0
    good = 0
    for w in words:
        if looks_like_syllabics(w):
            good += 1
        elif any(ch in CREE_LETTERS or ch.isalpha() and ch in "âêîô" for ch in w):
            if not re.search(r"[bdfgjqvxz]", w):
                good += 1
    return good / len(words)


def syllabify(sro: str) -> list[str]:
    word = normalize_sro(sro)
    chunks: list[str] = []
    i = 0
    vowels = set("aeiouâêîô")
    while i < len(word):
        j = i
        while j < len(word) and word[j] not in vowels:
            j += 1
        if j >= len(word):
            chunks.append(word[i:])
            break
        j += 1
        if j < len(word) and word[j] in vowels:
            j += 1
        if j < len(word) and word[j] not in vowels and word[j] not in "w":
            j += 1
        chunks.append(word[i:j])
        i = j
    return [c for c in chunks if c]


# The canonical checks from the converter's own test cases.
assert to_syllabics("atim") == "ᐊᑎᒼ", to_syllabics("atim")
assert to_sro("ᐊᑎᒼ") == "atim", to_sro("ᐊᑎᒼ")
assert to_syllabics("tânisi") == "ᑖᓂᓯ", to_syllabics("tânisi")
assert to_sro(to_syllabics("nêhiyawêwin")) == "nêhiyawêwin"

print("orthography ready:")
print("  atim      ->", to_syllabics("atim"))
print("  tânisi    ->", to_syllabics("tânisi"))
print("  nêhiyawêwin ->", to_syllabics("nêhiyawêwin"))

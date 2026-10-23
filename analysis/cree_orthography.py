"""nêhiyawêwin (Plains Cree) between its three scripts, exactly.

The three scripts this project has to move between, and why each one matters
here:

  Standard Roman Orthography (SRO)
      What the corpus, the FST lexicons and the translation model are all
      written in: tânisi, nêhiyawêwin, mîkis. Long vowels are marked with a
      circumflex, which is the convention this project's own pages already use
      (and the convention of the Plains Cree dictionary the FSTs come from).

  Syllabics (Cans, Canadian Aboriginal Syllabics)
      What nêhiyawêwin is traditionally written in, and what the demo page
      promises: one glyph is a whole syllable (consonant + vowel), with a
      separate small set of glyphs for a final consonant that closes a
      syllable. ᑖᓂᓯ is tânisi.

  Anglicized respelling
      How the language is read aloud by someone who does not read SRO:
      taa-ni-si, nay-hi-ya-way-win. This is a respelling and not a translation:
      every character of the input survives, one sound at a time, so a reader
      who knows nothing of Cree letters can say the word and be understood.

The SRO <-> syllabics tables below are NOT invented here. They are the
conversion rules of the itwêwina Plains Cree dictionary's own toolchain, the
GiellaLT lang-crk `src/fst/orthography/Latn-to-Cans.xfscript` and
`Cans-to-Latn.xfscript` transducers (ALTLab / University of Alberta, with the
Maskwacîs Education Schools Commission and First Nations University), which are
the converters behind itwêwina and everything built on it. They are transcribed
as data tables below because building the FSTs themselves needs the full
GiellaLT build (hfst/foma/xerox plus the whole lexicon), while the orthography
rules themselves are a closed, testable set of about 200 mappings; the tables
were checked against that repository's own test case (`atim` -> `ᐊᑎᒼ`) and
against a full round trip of this project's Plains Cree corpus.

Two conventions the tables carry, both worth knowing before reading output:

  * The converter treats a plain `e` as `ê` (`LengthSymbol` in the source
    rules), because Plains Cree writes its one e-sound long. A word written
    `nehiyawewin` therefore comes back `nêhiyawêwin`, and that is the
    converter's behaviour, not a bug in it.
  * `/` and `.` are positional: a final consonant has its own glyph (the "final"
    series), while the same consonant before a vowel is part of that syllable's
    glyph. `atim` is a-ti-m, three glyphs: ᐊ ᑎ ᒼ.

Anything wider than Plains Cree (the Woods Cree th-series ᖧ ᖨ ᖪ ᖬ, the
Swampy/Moose l- and r-finals ᓬ ᕒ, the `hk` final ᕽ) is included too, because
the FSTs and the syllabics block are Western Cree as a whole and a singer may
use any of it; the th-series has no long-vowel glyphs in the source rules and
none are invented here.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "SRO_VOWELS", "LONG_VOWELS", "SYLLABICS_FINALS",
    "macron_to_circumflex", "circumflex_to_macron", "fold_doubled_vowels", "normalize_sro",
    "is_syllabics", "looks_like_syllabics",
    "to_syllabics", "to_sro",
    "syllabify", "syllables_to_sro",
    "anglicize", "anglicize_word", "ANGLICIZED_STYLES",
    "describe", "orthography_self_test",
]

# --------------------------------------------------------------- the tables ---
# Vowel order used by the whole syllabics block: a, e, i, o, then the long
# vowels â, ê, î, ô in the same order. The FST rule files spell out every
# combination; the tables below hold the same content as data.
SRO_VOWELS = ("a", "e", "i", "o")
LONG_VOWELS = {"a": "â", "e": "ê", "i": "î", "o": "ô"}

# Consonant series, short vowels first, then longs. Read straight off
# Latn-to-Cans.xfscript.
_CONSONANT_SERIES: dict[str, dict[str, str]] = {
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
# The w-glide series: one glyph per consonant+w+vowel.
_W_SERIES: dict[str, dict[str, str]] = {
    "c": {"a": "ᒝ", "i": "ᒕ", "o": "ᒙ", "â": "ᒟ", "ê": "ᒓ", "î": "ᒗ", "ô": "ᒛ"},
    "k": {"a": "ᑿ", "i": "ᑷ", "o": "ᑻ", "â": "ᒁ", "ê": "ᑵ", "î": "ᑹ", "ô": "ᑽ"},
    "m": {"a": "ᒷ", "i": "ᒯ", "o": "ᒳ", "â": "ᒹ", "ê": "ᒭ", "î": "ᒱ", "ô": "ᒵ"},
    "n": {"a": "ᓌ", "i": "ᣇ", "o": "ᣋ", "â": "ᓎ", "ê": "ᓊ", "î": "ᣉ", "ô": "ᣍ"},
    "p": {"a": "ᑅ", "i": "ᐽ", "o": "ᑁ", "â": "ᑇ", "ê": "ᐻ", "î": "ᐿ", "ô": "ᑃ"},
    "s": {"a": "ᔁ", "i": "ᓹ", "o": "ᓽ", "â": "ᔃ", "ê": "ᓷ", "î": "ᓻ", "ô": "ᓿ"},
    "t": {"a": "ᑢ", "i": "ᑚ", "o": "ᑞ", "â": "ᑤ", "ê": "ᑘ", "î": "ᑜ", "ô": "ᑠ"},
    "y": {"a": "ᔺ", "i": "ᔲ", "o": "ᔶ", "â": "ᔼ", "ê": "ᔰ", "î": "ᔴ", "ô": "ᔸ"},
}
# A vowel with nothing in front of it.
_VOWEL_GLYPHS = {"a": "ᐊ", "i": "ᐃ", "o": "ᐅ", "â": "ᐋ", "ê": "ᐁ", "î": "ᐄ", "ô": "ᐆ"}
# Woods Cree th-series: the source rules give the four short ones only.
_TH_SERIES = {"a": "ᖬ", "e": "ᖧ", "i": "ᖨ", "o": "ᖪ"}
# Final consonants: the glyph a syllable ends on when no vowel follows.
SYLLABICS_FINALS = {
    "c": "ᐨ", "h": "ᐦ", "k": "ᐠ", "m": "ᒼ", "n": "ᐣ", "p": "ᑊ",
    "s": "ᐢ", "t": "ᐟ", "w": "ᐤ", "y": "ᕀ", "l": "ᓬ", "r": "ᕒ",
}
_FINALS_TO_SRO = {v: k for k, v in SYLLABICS_FINALS.items()}
_HK_GLYPH = "ᕽ"
_FULL_STOP = "᙮"
_DOTTED_W = "ᐧ"          # the combining-style w marker used by the compose rules

# SRO -> syllabics, longest first so a greedy scan finds kw before k and k
# before nothing. Built from the series above, in the source rules' own order.
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
# `e` is the same sound as `ê` in the source rules' LengthSymbol.
_add("e", _VOWEL_GLYPHS["ê"])

_SRO_KEYS_BY_LENGTH = sorted(_SRO_TO_CANS, key=len, reverse=True)

# Syllabics -> SRO: the whole-glyph map, plus the dotted-w compose rules, which
# are how the w-series is written in the wild (ᑲᐧ for ᑿ).
_CANS_TO_SRO: dict[str, str] = {}
for _sro, _cans in _SRO_TO_CANS.items():
    if len(_cans) == 1 and _sro not in ("e",):
        _CANS_TO_SRO.setdefault(_cans, _sro)
# The dotted form writes the w AFTER the plain syllable glyph (ᑲᐧ for ᑿ), so the
# rewrite has to be keyed by the GLYPH just read (and by the SRO it produced,
# which is what the scanner has to hand).
_COMPOSED_W: dict[str, str] = {}
_COMPOSED_W_SRO: dict[str, str] = {}
for _c, _row in _CONSONANT_SERIES.items():
    for _v, _g in _row.items():
        _COMPOSED_W[_g] = _c + "w" + _v
        _COMPOSED_W_SRO[_c + _v] = _c + "w" + _v
for _v, _g in _VOWEL_GLYPHS.items():
    _COMPOSED_W[_g] = "w" + _v
    _COMPOSED_W_SRO[_v] = "w" + _v


def macron_to_circumflex(text: str) -> str:
    """Saskatchewan-style SRO (ā ē ī ō) to the Plains Cree SRO this project uses.

    Meta's Cree ASR writes macrons, this project's corpus and FSTs write
    circumflexes, and the GiellaLT tree carries the same conversion as a rule
    (`macron-to-circumfl.regex`), so both spellings have to be understood here.
    """
    return text.translate(str.maketrans({"ā": "â", "ē": "ê", "ī": "î", "ō": "ô",
                                        "Ā": "Â", "Ē": "Ê", "Ī": "Î", "Ō": "Ô"}))


def circumflex_to_macron(text: str) -> str:
    """The other direction, for feeding a macron-expecting model."""
    return text.translate(str.maketrans({"â": "ā", "ê": "ē", "î": "ī", "ô": "ō",
                                        "Â": "Ā", "Ê": "Ē", "Î": "Ī", "Ô": "Ō"}))


# The doubled-vowel spelling: some community texts (two of the Bible books in
# this project's own corpus, and older missionary material) write a long vowel
# as the short letter twice -- "aakoosipeeyaak" -- instead of with a diacritic.
# `aa`, `ii` and `oo` are unambiguous (`â`, `î`, `ô`). `ee` is NOT: measured
# against the project's own FST lexicon, the same spelling is `ê` in
# "pimohteeyaani" (pimohtêyâni, to walk) and `î` in "aakoosipeeyaak"
# (âkôsipîyâk) -- the old convention does not distinguish the two, which is why
# folding it is an option with a default rather than a rule.
_DOUBLED_TO_LONG = {"aa": "â", "ii": "î", "oo": "ô", "uu": "ô"}


def fold_doubled_vowels(text: str, ee: str = "ê") -> str:
    """Fold a doubled-vowel spelling onto the diacritic one.

    ``ee`` is the one ambiguous pair and is `ê` by default; pass ``ee="î"`` for
    a text that uses it for long i. Folding is deliberately NOT part of
    :func:`normalize_sro`: a doubled vowel is a different convention, not a
    different spelling of the same letter, and silently rewriting it would
    make the engine's output unfaithful to whatever the singer's own spelling
    is. Callers that know their input is the doubled convention ask for it.
    """
    text = str(text)
    text = re.sub(r"aa|ii|oo|uu", lambda m: _DOUBLED_TO_LONG[m.group(0)], text)
    return re.sub(r"ee", "ê" if ee != "î" else "î", text)


def normalize_sro(text: str, *, fold_doubled: bool = False, ee: str = "ê") -> str:
    """One spelling for one word, whatever convention it arrived in.

    NFC first (a macron can arrive as a base letter plus a combining mark),
    then macrons to circumflexes, `|` to a space (that is the word boundary
    Meta's Cree models emit), case folded, and whitespace collapsed.
    ``fold_doubled`` additionally folds the doubled-vowel convention (see
    :func:`fold_doubled_vowels`) before anything else looks at the text.
    """
    text = unicodedata.normalize("NFC", str(text))
    if fold_doubled:
        text = fold_doubled_vowels(text, ee=ee)
    text = macron_to_circumflex(text)
    text = text.replace("|", " ").replace("\u2010", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


_SYLLABICS_RANGE = ((0x1400, 0x167F), (0x18B0, 0x18FF))


def is_syllabics(ch: str) -> bool:
    """Is this one character a Cree/Canadian syllabics glyph?"""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _SYLLABICS_RANGE)


def looks_like_syllabics(text: str) -> bool:
    """Does this text contain any syllabics at all?"""
    return any(is_syllabics(ch) for ch in str(text))


# ------------------------------------------------------------- the converters ---

def to_syllabics(sro: str) -> str:
    """Standard Roman Orthography to syllabics, the itwêwina rules.

    Longest match first (`kwa` is one glyph, `ka` is one, `k` is a final), a
    space survives as a space, and anything the tables do not know is passed
    through unchanged rather than dropped, so a half-Cree line stays readable.
    """
    text = normalize_sro(sro)
    # THE LENGTH SYMBOL PASS RUNS FIRST, as it does in the source rules
    # (`Decapitalisation .o. LengthSymbol .o. …`): a plain `e` is `ê` before any
    # syllable rule sees it, so `ke` becomes the single glyph ᑫ rather than a
    # k-final plus a vowel (ᐠᐁ). Without this the SRO round trip was not even
    # invertible on its own output.
    text = text.replace("e", "ê")
    out: list[str] = []
    i = 0
    n = len(text)
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
    """Syllabics to Standard Roman Orthography.

    Both spellings of the w-series are understood: the precomposed glyph (ᑿ) and
    the dotted form the compose rules describe (ᑲᐧ), because both are in use.
    """
    text = unicodedata.normalize("NFC", str(syllabics))
    out: list[str] = []
    last_glyph: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == _DOTTED_W:
            # a w marker after a glyph: rewrite the syllable it follows
            if last_glyph is not None:
                out[-1] = _COMPOSED_W.get(last_glyph, out[-1] + "w")
                last_glyph = None
            else:
                out.append("w")
            i += 1
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
        i += 1
    text = "".join(out)
    if long_vowels == "macron":
        text = circumflex_to_macron(text)
    return text


# ------------------------------------------------------------------ families ---

_VOWEL_CHARS = set("aeiouâêîô")
_LONG_CHARS = set("âêîô")
_CONSONANTS = set("ckmnstpwyhlr") | {"th", "hk"}


def syllabify(sro: str) -> list[str]:
    """Split a Cree word into its syllables, the way the syllabics write it.

    Cree syllables are CV(V)(C): one consonant and one vowel, optionally long,
    optionally closed by a final consonant. A glide that stands between a
    consonant and a vowel belongs to the syllable it starts (k-w-ê is one
    syllable, written with one glyph, ᑵ), while a glide or consonant after a
    vowel closes the vowel's own syllable. This is why the number of items
    returned matches the number of syllable glyphs the syllabics rendition
    uses, which is the check the harness runs over the whole corpus.
    """
    text = normalize_sro(sro).replace("e", "ê")
    # Every syllable in Cree has a vowel; a consonant that finds none attaches to
    # the syllable next to it (a final closing the one before it, or an onset
    # opening the one after). That is exactly what the syllabics do, one glyph
    # per vowel, which is why this count can be checked against the glyphs.
    syllables: list[str] = []
    onset = ""                  # consonant(s) waiting for their vowel

    def _consonant(tok: str) -> None:
        """Ready one consonant to open the next syllable.

        A consonant already waiting has found no vowel, so it closes the
        syllable before it instead (the vowel of `nîsohkamâkêwin`'s hk, the w
        that ends `nêhiyaw`).
        """
        nonlocal onset
        if onset:
            if syllables:
                syllables[-1] += onset
            else:
                syllables.append(onset)
        onset = tok

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _VOWEL_CHARS:
            syllables.append(onset + ch)
            onset = ""
            i += 1
            continue
        if ch == "w" and i + 1 < n and text[i + 1] in _VOWEL_CHARS and onset:
            onset += "w"           # C + w + V ride together: kwê is one glyph ᑵ
            i += 1
            continue
        if ch == "h" and i + 1 < n and text[i + 1] == "k":
            # hk has its own glyph (ᕽ) and it is a FINAL: it closes the vowel
            # before it, so âhkosiwin is âhk-o-si-win, the four glyphs ᐋᕽᐅᓯᐏᐣ.
            if syllables:
                syllables[-1] += "hk"
            elif onset:
                onset += "hk"
            else:
                onset = "hk"
            i += 2
            continue
        if text.startswith("th", i):
            _consonant("th")      # the Woods Cree th-series is one consonant
            i += 2
            continue
        _consonant(ch)
        i += 1
    if onset:
        if syllables:
            syllables[-1] += onset         # a word-final consonant closes it
        else:
            syllables.append(onset)
    return [s for s in syllables if s]


def syllables_to_sro(syllables: list[str]) -> str:
    """Inverse of :func:`syllabify`, for callers that edit syllables."""
    return "".join(syllables)


# ------------------------------------------------------ anglicized respelling ---
#
# One English spelling per Cree sound. This is the part a reader actually
# pronounces, so the choices are the ones Cree teaching material uses for
# English readers (the "sounds of Cree" charts): the short vowels as English
# short vowels, the long ones as their doubled English spellings, `c` as `ts`
# because that is the sound (cats, not see), and nothing silent or decorative
# anywhere.
ANGLICIZED_STYLES = ("syllables", "plain")

_RESPELL = {
    "â": "aa", "ê": "ay", "î": "ee", "ô": "oo",
    "a": "a", "e": "ay", "i": "i", "o": "o",
    "c": "ts", "k": "k", "m": "m", "n": "n", "p": "p",
    "s": "s", "t": "t", "w": "w", "y": "y",
    "h": "h", "l": "l", "r": "r", "th": "th",
}
# A syllable that is only a final consonant has to be sayable: add the same
# short vowel English readers would supply for that letter.
_FINAL_RESPELL = {"c": "ts", "h": "h", "k": "k", "m": "m", "n": "n", "p": "p",
                  "s": "s", "t": "t", "w": "w", "y": "y", "l": "l", "r": "r"}


def _respell_xs(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        pair = text[i:i + 2]
        if pair in _RESPELL:
            out.append(_RESPELL[pair])
            i += 2
            continue
        ch = text[i]
        out.append(_RESPELL.get(ch, ch))
        i += 1
    return "".join(out)


# A reader has to be able to separate the syllables, so the hyphen goes where
# the syllable does; a long vowel already carries its own length mark in `aa`,
# `ay`, `ee`, `oo`, so a syllable boundary after one reads naturally.
def anglicize_word(sro: str, *, style: str = "syllables", keep_source: bool = False,
                   separator: str = "-") -> str:
    """An English respelling of one Cree word.

    ``style="syllables"`` puts the hyphens where the syllabics would (the default,
    taa-ni-si), ``style="plain"`` writes it as one run (taanishi-ish, taa-ni-si
    without the breaks). ``keep_source`` appends the SRO in brackets, which is
    what the transcript panel shows a singer so they can check a word.
    """
    word = normalize_sro(sro)
    if not word:
        return ""
    if looks_like_syllabics(word):
        word = to_sro(word)
    parts = syllabify(word)
    if not parts:
        return word
    spelled = []
    for part in parts:
        body = _respell_xs(part)
        if part and part[0] not in _VOWEL_CHARS and not any(c in _VOWEL_CHARS for c in part):
            # an all-consonant syllable (a final that stands alone): give the
            # reader the vowel English expects after that letter, and mark the
            # syllable as a closed one
            body = body + "a" if body.endswith(("k", "t", "p", "ts")) else body
        spelled.append(body)
    text = separator.join(spelled) if style == "syllables" else "".join(spelled)
    if keep_source:
        text = f"{text} ({word})"
    return text


def anglicize(text: str, *, style: str = "syllables", keep_source: bool = False) -> str:
    """Respell a whole line, word by word, keeping the words separate."""
    line = normalize_sro(text)
    if looks_like_syllabics(line):
        line = to_sro(line)
    out = []
    for token in line.split(" "):
        if not token:
            continue
        core = "".join(ch for ch in token if ch.isalpha() or ch in "âêîô")
        tail = token[len(core):] if core and token.startswith(core) else ""
        if not core:
            out.append(token)
            continue
        out.append(anglicize_word(core, style=style, keep_source=False) + tail)
    result = " ".join(out)
    if keep_source:
        result = f"{result} ({line})"
    return result


def describe(text: str) -> dict:
    """Everything the three scripts say about one line, for the panel and the API."""
    sro = normalize_sro(text)
    if looks_like_syllabics(sro):
        sro = to_sro(sro)
    cans = to_syllabics(sro)
    return {
        "sro": sro,
        "syllabics": cans,
        "anglicized": anglicize(sro),
        "syllables": syllabify(sro),
    }


# ---------------------------------------------------------------- self-test ---

def orthography_self_test(verbose: bool = False) -> dict:
    """The checks that make this module trustworthy, runnable from a shell.

    Three of them, in rising order of strength:

      1. the source project's own test case (atim -> ᐊᑎᒼ);
      2. curated words with the respelling and syllable count a Plains Cree
         speaker would expect, including both w-series spellings;
      3. a full round trip of every word of this project's Plains Cree corpus:
         SRO -> syllabics -> SRO must return the same word, and the syllable
         count must equal the number of syllable glyphs the syllabics use.
    """
    import pathlib

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            failures.append(f"{label}: got {got!r}, expected {want!r}")
        elif verbose:
            print(f"  [ok] {label}: {got!r}")

    check("itwêwina test case", to_syllabics("atim"), "ᐊᑎᒼ")
    check("tânisi", to_syllabics("tânisi"), "ᑖᓂᓯ")
    check("tânisi round trip", to_sro("ᑖᓂᓯ"), "tânisi")
    check("dotted w-series", to_sro("ᑲᐧ"), "kwa")
    check("precomposed w-series", to_sro("ᑿ"), "kwa")
    check("hk final", to_syllabics("ohk"), "ᐅᕽ")
    check("nêhiyawêwin", to_syllabics("nêhiyawêwin"), "ᓀᐦᐃᔭᐍᐏᐣ")
    check("macron input", to_syllabics("tānisi"), "ᑖᓂᓯ")
    check("syllables tânisi", syllabify("tânisi"), ["tâ", "ni", "si"])
    check("syllables nêhiyawêwin", syllabify("nêhiyawêwin"),
          ["nê", "hi", "ya", "wê", "win"])
    check("syllables atim", syllabify("atim"), ["a", "tim"])
    check("anglicized tânisi", anglicize_word("tânisi"), "taa-ni-si")
    check("anglicized nêhiyawêwin", anglicize_word("nêhiyawêwin"),
          "nay-hi-ya-way-win")
    check("anglicized mîkis", anglicize_word("mîkis"), "mee-kis")
    check("anglicized cîmân", anglicize_word("cîmân"), "tsee-maan")

    # 3. the corpus. The diacritic texts are the canonical SRO and are held to
    # the round trip; the doubled-vowel texts (two Bible books) are a different
    # convention with an ambiguous `ee`, so they are measured for what folding
    # recovers rather than round-tripped, which would be testing the wrong
    # thing.
    root = pathlib.Path(__file__).resolve().parent.parent / "data" / "PlainsCree"
    canonical: set[str] = set()
    doubled: set[str] = set()
    # Letters Cree does not have (b d f g j q u v x z) only appear in loanwords
    # and proper names in this corpus (biscuit, alphaeus); they are not Cree
    # orthography and are not what this module is for.
    non_cree = set("bdfgjquvxz")
    if root.is_dir():
        for path in sorted(root.rglob("*_cr.txt")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                for token in normalize_sro(line).split(" "):
                    core = "".join(ch for ch in token if ch.isalpha() or ch in "âêîô")
                    if not core or looks_like_syllabics(core):
                        continue
                    if non_cree & set(core):
                        continue
                    (doubled if re.search(r"aa|ee|ii|oo|uu", core) else canonical).add(core)
    # The round trip is compared modulo the source rules' own LengthSymbol rule:
    # the converter reads a plain `e` as `ê` (Plains Cree writes its one e-sound
    # long), so `acoskewinis` is expected to come back `acoskêwinis`. That is not
    # a tolerance invented here to make a test pass -- it is the one rule the
    # tables were transcribed from, and the check below it (against the FST
    # lexicon) is what proves the canonically-spelled form is the real word.
    def mod_e(word: str) -> str:
        return word.replace("e", "ê")

    round_trip_bad = 0
    strict_syllabics_bad = 0
    syllable_bad = 0
    for word in sorted(canonical):
        # (a) the strict one, with no tolerance anywhere: syllabics -> SRO ->
        # syllabics must be the identity, because the ASR path converts the
        # model's syllabics into SRO and back again on every request.
        cans = to_syllabics(word)
        if to_syllabics(to_sro(cans)) != cans:
            strict_syllabics_bad += 1
            if strict_syllabics_bad <= 5:
                failures.append(f"syllabics round trip {cans!r} -> {to_sro(cans)!r} -> "
                                f"{to_syllabics(to_sro(cans))!r}")
        # (b) the SRO one, compared modulo the converter's own documented
        # `e` -> `ê` rule (that rule is in the source tables; see the note in
        # the docstring and the FST measurement in tools/check_cree_transcription.py).
        back = to_sro(cans)
        if back != word and mod_e(word) != back:
            round_trip_bad += 1
            if round_trip_bad <= 5:
                failures.append(f"round trip {word!r} -> {cans!r} -> {back!r}")
        # (c) the syllable count has to match the glyphs the syllabics use.
        # Words spelled with the English `ch` digraph (a variant of `c` in a
        # few of this corpus's files, not standard SRO) are skipped: the
        # converter has no `ch` rule and writes c-final + h-final + a vowel for
        # them, which is the honest thing to do with a spelling it does not know.
        if "ch" in word:
            continue
        word_syllables = syllabify(word)
        if any(not any(c in _VOWEL_CHARS for c in part) for part in word_syllables):
            # a stray consonant with no vowel around it (a stem-list entry, a
            # loanword): the language has no syllable like that, so there is no
            # glyph count to compare against
            continue
        gly_count = sum(1 for ch in cans
                        if is_syllabics(ch) and ch not in _FINALS_TO_SRO
                        and ch not in (_HK_GLYPH, _DOTTED_W, _FULL_STOP))
        if gly_count != len(syllabify(word)):
            syllable_bad += 1
            if syllable_bad <= 5:
                failures.append(
                    f"syllable count {word!r}: syllabify gives {len(syllabify(word))} "
                    f"({syllabify(word)}), syllabics use {gly_count} ({cans})")
    return {
        "words_checked": len(canonical),
        "doubled_vowel_words": len(doubled),
        "round_trip_failures": round_trip_bad,
        "syllabics_round_trip_failures": strict_syllabics_bad,
        "syllable_count_failures": syllable_bad,
        "failures": failures,
    }


if __name__ == "__main__":  # pragma: no cover
    report = orthography_self_test(verbose=True)
    print(f"\ncorpus words checked: {report['words_checked']} "
          f"(+{report['doubled_vowel_words']} in the doubled-vowel convention)")
    print(f"SRO round-trip failures: {report['round_trip_failures']}")
    print(f"syllabics round-trip failures: {report['syllabics_round_trip_failures']}")
    print(f"syllable-count mismatches: {report['syllable_count_failures']}")
    for f in report["failures"]:
        print(f"  [FAIL] {f}")
    raise SystemExit(1 if report["failures"] else 0)

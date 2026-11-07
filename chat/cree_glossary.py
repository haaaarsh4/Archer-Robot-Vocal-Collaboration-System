"""What a single nêhiyawêwin word means, from the project's own dictionaries.

Asking a 7B model what a Cree word means is asking it to guess. It answers with
the same confidence either way, so a wrong gloss ("tânisi means how many") is
indistinguishable from a right one by the reader. These are the two glossed word
lists already in the repository, and a lookup in them is the only thing here
that is allowed to state a meaning.

  * ``data/processed/watkins_dictionary_pairs.jsonl`` - Watkins' 1865 dictionary
    as word/English pairs, the widest single-word source in the project.
  * ``data/processed/parallel_cr_en.jsonl`` - the aligned sentence pairs; only
    the entries whose Cree side is a single word are usable as a gloss, and the
    ones carrying frequency digits are dropped because they are corpus noise
    (``"tânisi 31 tânispîhk 2"`` is not a gloss).

A word that is not in either list gets no gloss. The caller says so plainly
rather than letting the model fill the gap: an assistant speaking about a living
artist's language does not get to invent a meaning.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

__all__ = ["gloss_of", "asked_word"]

_ROOT = Path(__file__).resolve().parents[1]

_SOURCES = (
    (_ROOT / "data/processed/watkins_dictionary_pairs.jsonl", "Watkins 1865"),
    (_ROOT / "data/processed/parallel_cr_en.jsonl", None),
)

_WORD_CHARS = re.compile(r"^[a-zâêîôûāēīōū'’\u1400-\u167F\u18B0-\u18FF-]+$")
_DIGIT = re.compile(r"\d")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
# Watkins puts a qualifier in front of the headword and inside the gloss:
# "(black) muskwa" is the word muskwa, glossed "black bear".
_QUALIFIER = re.compile(r"^\s*\(([^)]*)\)\s*")

# A gloss is a word or a short phrase. Anything longer is a sentence from the
# parallel corpus that happened to have a one-word Cree side.
_MAX_GLOSS_WORDS = 6
_MAX_GLOSS_CHARS = 48

_index: dict[str, Counter] | None = None
_where: dict[str, str] = {}
_folded: dict[str, str | None] | None = None


def _norm(word: str) -> str:
    return unicodedata.normalize("NFC", str(word)).strip().strip("\"'“”‘’.,!?;:").lower()


def _fold(word: str) -> str:
    """The word without its long-vowel marks, for a lookup that misses.

    People type tânisi without the circumflex. The marks are orthography, not a
    different word, so a second index keyed without them finds it. Where two
    dictionary headwords fold onto the same spelling the mapping is dropped
    rather than guessed at.
    """
    text = unicodedata.normalize("NFD", _norm(word))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).replace("’", "'")


def _folded_index() -> dict[str, str | None]:
    global _folded
    if _folded is None:
        folded: dict[str, str | None] = {}
        for key in _load():
            f = _fold(key)
            folded[f] = key if folded.get(f, key) == key else None
        _folded = folded
    return _folded


def _usable(cree: str, english: str) -> tuple[str, str] | None:
    c, e = cree.strip(), english.strip()
    qualifier = _QUALIFIER.match(c)
    if qualifier:
        c = c[qualifier.end():].strip()
        e = f"{qualifier.group(1).strip()} {e}".strip()
    if not c or not e or _DIGIT.search(c) or _DIGIT.search(e):
        return None
    if len(c.split()) != 1 or not _WORD_CHARS.match(c.lower()):
        return None
    if len(e.split()) > _MAX_GLOSS_WORDS or len(e) > _MAX_GLOSS_CHARS:
        return None
    if e.lower() == c.lower():
        return None
    # the parallel corpus keeps the spacing of its source ("hello , how are you ?")
    return _norm(c), _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(e.split()))


def _load() -> dict[str, Counter]:
    global _index, _where
    if _index is not None:
        return _index
    index: dict[str, Counter] = {}
    for path, label in _SOURCES:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            pair = _usable(str(row.get("cree", "")), str(row.get("english", "")))
            if not pair:
                continue
            word, english = pair
            index.setdefault(word, Counter())[english] += 1
            if label and word not in _where:
                _where[word] = label
    _index = index
    return index


def gloss_of(word: str) -> tuple[str, str | None] | None:
    """The most common English gloss for one word, and where it came from.

    Returns None when the word is not in either list. Nothing is guessed and
    nothing is generated: an empty answer is the honest one.
    """
    key = _norm(word)
    if not key:
        return None
    entry = _load().get(key)
    if not entry:
        canonical = _folded_index().get(_fold(key))
        entry = _load().get(canonical) if canonical else None
        key = canonical or key
    if not entry:
        return None
    return entry.most_common(1)[0][0], _where.get(key)


# ------------------------------------------------------------- the question
# The shapes the panel actually receives. "What does maskwa mean?" is a gloss
# question; "explain Horse" is not, and neither is a request to translate.
_WHAT_MEANS = re.compile(
    r"\bwhat\s+(?:does|do)\s+(?:the\s+)?(?:cree\s+)?(?:word\s+)?"
    r"(?P<word>[^\s,?]+(?:\s+[^\s,?]+)?)\s+mean\b",
    re.IGNORECASE,
)
_MEANING_OF = re.compile(
    r"\bmeaning\s+of\s+(?:the\s+)?(?:cree\s+)?(?:word\s+)?(?P<word>[^\s,?]+)",
    re.IGNORECASE,
)
_EXPLAIN_WORD = re.compile(
    r"\bexplain\s+(?:the\s+)?(?:cree\s+)?word\s+(?P<word>[^\s,?]+)",
    re.IGNORECASE,
)


def asked_word(message: str) -> str | None:
    """The word a message is asking the meaning of, if it is asking about one."""
    text = unicodedata.normalize("NFC", str(message)).strip()
    for pattern in (_WHAT_MEANS, _MEANING_OF, _EXPLAIN_WORD):
        match = pattern.search(text)
        if not match:
            continue
        word = match.group("word").strip().strip("\"'“”‘’.,!?;:")
        # "what does this mean" and "what does that word mean" carry no word
        if not word or word.lower() in {"this", "that", "it", "word", "one"}:
            return None
        return word
    return None

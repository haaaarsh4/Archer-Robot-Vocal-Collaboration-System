"""Routing and prompting for the project's trained Cree model.

The chat has two models behind it: a general local model for conversation about
the project, and the nêhiyawêwin model trained by ``Cree_LLM_Training.ipynb``.
This module decides which one a question belongs to and supplies the system
prompt the Cree model was trained with.

The decision is deliberately conservative. A question is routed to the Cree
model when it is written in Cree (syllabics, or Cree words with long-vowel
marks), or when it explicitly asks for Cree. An English question about Cree
music stays on the general model, because the Cree model is a language tool, not
a better conversationalist.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "CREE_SYSTEM_PROMPT", "CREE_WORDS", "script_of", "looks_like_cree",
    "wants_cree", "choose_route", "format_cree_prompt",
]

# The exact prompt the model was instruction-tuned with. Changing it at
# inference time moves the model off the distribution it was trained on.
CREE_SYSTEM_PROMPT = (
    "You are the nêhiyawêwin (Plains Cree) language guide inside Archer's system. "
    "You answer in the script the user asks for: Standard Roman Orthography (SRO) or syllabics. "
    "You translate faithfully and keep names, numbers and negation intact. "
    "You never invent Cree words, meanings, song details or ceremony. "
    "When a question is about Archer's songs, use only the notes given to you. "
    "If the notes do not contain the answer, say plainly that the notes do not say. "
    "You are a language tool, not a performer, and you never speak as Archer."
)

_SYLLABICS = re.compile(r"[\u1400-\u167F\u18B0-\u18FF]")
# A request for Cree output, not a passing mention of the language. "Tell me
# about Cree music" belongs on the general model; "how do you say river in
# Cree" and "the Cree word for river" belong here.
_CREE_REQUEST = re.compile(
    r"\b(?:in|into|to)\s+(?:plains\s+)?cree\b"
    r"|\bcree\s+(?:word|translation|version|spelling|name)\b"
    r"|\b(?:translate|say|write|spell|pronounce)\b[^?.!]{0,60}\bcree\b"
    r"|\bhow\s+do\s+you\s+say\b[^?.!]{0,60}\bcree\b"
    r"|\b(?:in|into|to)\s+syllabics?\b"
    r"|\b(?:write|convert|show|transcribe)\b[^?.!]{0,60}\bsyllabics?\b",
    re.IGNORECASE,
)
_TO_ENGLISH = re.compile(
    r"\b(?:in|into|to)\s+english\b|\btranslate\b[^?.!]{0,60}\benglish\b",
    re.IGNORECASE,
)
_TO_SYLLABICS = re.compile(r"\bsyllabics?\b|\bcans\b", re.IGNORECASE)
_TO_SRO = re.compile(r"\bsro\b|\bstandard\s+roman\b|\broman\s+orthography\b", re.IGNORECASE)
# A request that should not be answered from song notes even when retrieval
# finds a superficially similar passage.
_TRANSLATION_REQUEST = re.compile(
    r"\btranslate\b|\bhow\s+do\s+you\s+say\b|\b(?:in|into|to)\s+english\b|"
    r"\b(?:in|into|to)\s+(?:plains\s+)?cree\b|\bcree\s+word\b|\bsyllabics?\b|\bsro\b",
    re.IGNORECASE,
)

# Function words and everyday vocabulary that identify a Cree sentence. The
# list is intentionally short: it is a signal, not a dictionary, and every word
# here is unambiguous enough not to fire on English.
CREE_WORDS = frozenset({
    "êkwa", "mîna", "kî", "kâ", "ê", "ta", "ni", "ki", "ôma", "ôki", "êsa",
    "kêkway", "awiyak", "niya", "kiya", "êkosi", "êkotê", "itwêw", "itêw",
    "tânisi", "tân'si", "nêhiyawêwin", "nehiyawewin", "kîkway", "nama",
    "namôya", "mîcisow", "nipâw", "pîhtokêw", "maskwa", "atim", "iskwew",
    "napew", "awâsis", "sîpiy", "askiy", "kîsikâw", "tipiskâw", "mîkis",
    "nîpisîy", "pêyak", "nîso", "nisto", "nêwo", "niyânan", "nikotwâsik",
    "têpakohp", "ênânêw", "kêkâc", "mistahi", "mâka", "kayâs", "piko",
})


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def script_of(text: str) -> str | None:
    """'Cans' when the text is written in syllabics, 'Latn' for SRO, None if neither."""
    text = _nfc(text)
    if _SYLLABICS.search(text):
        return "Cans"
    words = re.findall(r"[a-zâêîô']+", text.lower())
    if any(ch in "âêîô" for ch in text.lower()) or CREE_WORDS.intersection(words):
        return "Latn"
    return None


def looks_like_cree(text: str) -> bool:
    """Is this text itself Cree, rather than English about Cree?"""
    text = _nfc(text)
    if _SYLLABICS.search(text):
        return True
    words = re.findall(r"[a-zâêîô']+", text.lower())
    if not words:
        return False
    hits = sum(word in CREE_WORDS for word in words)
    marked = sum(bool(re.search(r"[âêîô]", word)) for word in words)
    # One marked word is a strong signal; otherwise ask for two function words,
    # so a loanword or a name does not drag an English question into Cree.
    return marked > 0 or hits >= 2


def wants_cree(text: str) -> bool:
    """Cree text, or an explicit request to translate into / write in Cree."""
    return looks_like_cree(text) or bool(_CREE_REQUEST.search(_nfc(text)))


def choose_route(message: str, requested: str = "auto", cree_model: str = "") -> str:
    """Return 'cree' or 'english' for one chat request.

    ``requested`` is the client's explicit choice. ``auto`` uses the text.
    An empty ``cree_model`` means the trained model is not installed, and the
    request stays on the general model rather than failing.
    """
    if requested == "english":
        return "english"
    if not cree_model:
        return "english"
    if requested == "cree":
        return "cree"
    return "cree" if wants_cree(message) else "english"


def format_cree_prompt(message: str, notes: list[tuple[str, str]] | None = None) -> str:
    """Wrap a chat message in the exact shape the notebook trained on.

    This is not cosmetic. The instruction data in ``08_sft_data.py`` gives every
    example a control prefix, and notes arrive inline as ``Notes on <song>:``
    followed by ``Question:``. At inference the app must reproduce that shape;
    a generic "context" header is a different distribution and the model treats
    it as prose to continue rather than notes to read.

    ``notes`` is a list of ``(song title, section text)`` pairs, already
    retrieved and relevance-filtered by the caller.
    """
    message = str(message).strip()
    explicit_translation = bool(_TRANSLATION_REQUEST.search(message))
    if notes and not explicit_translation:
        blocks = "\n\n".join(
            f"Notes on {title}:\n{text.strip()}" for title, text in notes if text.strip()
        )
        if blocks:
            return f"<|song|><|note|>\n{blocks}\n\nQuestion: {message}"
    if _TO_SYLLABICS.search(message) or _TO_SRO.search(message):
        # Direction matters: <|cans|><|sro|> converts syllabics to SRO, and
        # <|sro|><|cans|> is the other way. English to syllabics is a
        # translation, not a script conversion.
        script = script_of(message)
        if script == "Cans":
            return f"<|cans|><|sro|>\n{message}"
        if script == "Latn":
            return f"<|sro|><|cans|>\n{message}"
        return f"<|translate|><|english|><|sro|>\n{message}"
    if _TO_ENGLISH.search(message) or looks_like_cree(message):
        return f"<|translate|><|cree|><|english|>\n{message}"
    if wants_cree(message):
        return f"<|translate|><|english|><|sro|>\n{message}"
    return f"<|song|>\n{message}"

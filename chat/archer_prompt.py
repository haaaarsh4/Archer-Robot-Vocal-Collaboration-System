"""The one prompt the Archer model is trained and served with.

There is a single model. It answers conversation about the project, reads
retrieved song notes, translates in both directions and converts between
Standard Roman Orthography and syllabics. What changes between those jobs is
not the model, it is the task prefix and whether notes were retrieved.

The prefixes are part of the model's instruction training, so they are
reproduced here exactly: ``<|song|>`` for a question with no notes,
``<|song|><|note|>`` for a question answered from notes, and
``<|translate|>``/``<|sro|>``/``<|cans|>`` for the language tasks. A generic
"context" header is a different distribution and the model would treat it as
prose to continue rather than notes to read.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ARCHER_SYSTEM_PROMPT", "CREE_WORDS", "script_of", "looks_like_cree",
    "wants_cree", "task_of", "format_prompt",
]

# The exact prompt the model is instruction-tuned with. The trainer's SFT cell
# carries the same text; tests/test_chat_model.py fails if the two drift.
ARCHER_SYSTEM_PROMPT = (
    "You are Archer's assistant, running entirely on the user's own machine. "
    "You do four jobs: you talk about Archer's work and this system, you answer "
    "questions about Archer's songs from the notes you are given, you translate "
    "between English and nêhiyawêwin in Standard Roman Orthography and syllabics, "
    "and you explain and convert Cree words. "
    "When notes are given, they are the only evidence for anything about a song: "
    "answer from them, name the song, and if they do not contain the answer say "
    "plainly that the notes do not say. "
    "When no notes are given, answer about this project and about Plains Cree from "
    "what you know, and say when you are unsure instead of inventing a word, a "
    "meaning, a name, a number or a ceremony. "
    "Translate faithfully: keep names, numbers, negation and tense intact, and "
    "never leave an English word inside a Cree sentence. "
    "You never speak as Archer and you never claim his voice is yours."
)

_SYLLABICS = re.compile(r"[\u1400-\u167F\u18B0-\u18FF]")

# A request for Cree output, not a passing mention of the language.
_TO_CREE = re.compile(
    r"\b(?:in|into|to)\s+(?:plains\s+)?cree\b"
    r"|\bcree\s+(?:word|translation|version|spelling|name)\b"
    r"|\b(?:translate|say|write|spell|pronounce)\b[^?.!]{0,60}\bcree\b"
    r"|\bhow\s+do\s+you\s+say\b[^?.!]{0,60}\bcree\b",
    re.IGNORECASE,
)
_TO_ENGLISH = re.compile(
    r"\b(?:in|into|to)\s+english\b|\btranslate\b[^?.!]{0,60}\benglish\b",
    re.IGNORECASE,
)
_TO_SYLLABICS = re.compile(r"\bsyllabics?\b|\bcans\b", re.IGNORECASE)
_TO_SRO = re.compile(r"\bsro\b|\bstandard\s+roman\b|\broman\s+orthography\b", re.IGNORECASE)

# "What does maskwa mean?" and "Explain the Cree word maskwa" are questions
# about a word, not requests to translate one. "What is the Cree word for
# river?" is the other way round.
_MEANING_QUESTION = re.compile(
    r"\bwhat\s+does\b[^?.!]{0,80}\bmean\b"
    r"|\bmeaning\s+of\b"
    r"|\bexplain\b[^?.!]{0,40}\b(?:word|meaning|mean)\b",
    re.IGNORECASE,
)

# A question in English that happens to contain a Cree word is a question, not
# a translation request. "What does nêhiyawêwin mean?" gets an explanation;
# a bare "tânisi" gets translated.
_ENGLISH_QUESTION = re.compile(
    r"\b(?:what|hows|how|why|who|when|where|which|whose|explain|describe|tell|"
    r"mean|means|meaning|about|difference)\b",
    re.IGNORECASE,
)

# Function words and everyday vocabulary that identify a Cree sentence. A
# signal, not a dictionary; every word is unambiguous enough not to fire on
# English.
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
    """'Cans' for syllabics, 'Latn' for SRO, None when it is not Cree."""
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
    return marked > 0 or hits >= 2


def wants_cree(text: str) -> bool:
    """Cree text, or an explicit request to write or translate into Cree."""
    return looks_like_cree(text) or bool(_TO_CREE.search(_nfc(text)))


def _is_script_conversion(text: str) -> bool:
    """Cree text plus a request to change its script."""
    return bool(_TO_SYLLABICS.search(text) or _TO_SRO.search(text)) and script_of(text) is not None


def _is_translation(text: str) -> bool:
    if _TO_ENGLISH.search(text):
        return True
    if _MEANING_QUESTION.search(text):
        return False
    if _TO_CREE.search(text) or _is_script_conversion(text):
        return True
    # A bare Cree sentence, with no English question around it.
    return looks_like_cree(text) and not _ENGLISH_QUESTION.search(text)


def task_of(message: str, has_notes: bool = False) -> str:
    """Return 'orthography', 'translate', 'song_notes' or 'chat'.

    This is a prompt-shape decision, not a model choice: every branch is
    answered by the same model.
    """
    text = _nfc(str(message)).strip()
    if _is_script_conversion(text):
        return "orthography"
    if _is_translation(text):
        return "translate"
    # A question about a word is answered from what the model knows about that
    # word, not from whichever song note happened to match its spelling. The
    # gloss examples are trained on the <|song|> path, so the app uses it too.
    if _MEANING_QUESTION.search(text):
        return "chat"
    if has_notes:
        return "song_notes"
    return "chat"


def format_prompt(message: str, notes: list[tuple[str, str]] | None = None) -> str:
    """Wrap one user message in the exact shape the model was trained on.

    ``notes`` is a list of ``(song title, passage)`` pairs, already retrieved
    and relevance-filtered by the caller. Notes are dropped for a translation
    or script request, because song prose is not evidence about a word.
    """
    message = str(message).strip()
    task = task_of(message, bool(notes))

    if task == "orthography":
        if script_of(message) == "Cans":
            return f"<|cans|><|sro|>\nWrite this syllabics sentence in SRO:\n{message}"
        return f"<|sro|><|cans|>\nWrite this in Cree syllabics:\n{message}"

    if task == "translate":
        if _TO_ENGLISH.search(message) or (
            looks_like_cree(message) and not _TO_CREE.search(message)
        ):
            # Syllabics input carries its own marker in training, so the model
            # knows which script it is being asked to read.
            marker = "<|cans|>\n" if script_of(message) == "Cans" else ""
            return f"<|translate|><|cree|><|english|>\n{marker}Translate into English:\n{message}"
        if _TO_SYLLABICS.search(message):
            return f"<|translate|><|english|><|cans|>\nWrite this in Cree syllabics:\n{message}"
        return f"<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\n{message}"

    if task == "song_notes" and notes:
        blocks = "\n\n".join(
            f"Notes on {title}:\n{text.strip()}" for title, text in notes if text.strip()
        )
        if blocks:
            return f"<|song|><|note|>\n{blocks}\n\nQuestion: {message}"

    return f"<|song|>\n{message}"

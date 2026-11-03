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
from dataclasses import dataclass

__all__ = [
    "ARCHER_SYSTEM_PROMPT", "CREE_WORDS", "script_of", "looks_like_cree",
    "wants_cree", "task_of", "format_prompt",
]

# The exact prompt the model is instruction-tuned with. The trainer's SFT cell
# carries the same text; tests/test_chat_model.py fails if the two drift.
ARCHER_SYSTEM_PROMPT = (
    "You are Archer's assistant, running entirely on the user's own machine. "
    "You are warm and plain spoken. If someone greets you or asks how you are, "
    "answer in one or two short sentences in the first person, and ask what they "
    "need. Do not describe the project or list your jobs unless they ask. "
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
    r"\b(?:in|into|to)\s+(?:english|eng)\b|\btranslate\b[^?.!]{0,60}\b(?:english|eng)\b",
    re.IGNORECASE,
)

# The words that mark a request as a language exercise rather than a question
# about a song. "Explain Horse in Cree" is a song question with a language
# preference; "what is the Cree word for horse" is a translation.
_STRONG_VOCAB = re.compile(
    r"\b(?:cree\s+)?word\s+for\b"
    r"|\bhow\s+(?:do|would|can)\s+(?:you|i)\s+say\b"
    r"|\btranslate\b",
    re.IGNORECASE,
)

# A message that refers back to the last answer rather than carrying its own
# text. These are the follow ups the panel actually receives.
_DEICTIC = re.compile(
    r"\b(?:this|that|it|those|these|the\s+answer|the\s+last\s+answer|the\s+above)\b",
    re.IGNORECASE,
)

# A question word plus a retrieved title is a song question. The title match is
# what keeps "what does nêhiyawêwin mean" out of the song path even when a note
# happens to mention the word.
_SONG_INTENT = re.compile(
    r"\b(?:what|who|when|where|why|how|explain|describe|tell|is|are|was|were|does|did)\b",
    re.IGNORECASE,
)

_LEAD_PLEASE = re.compile(r"^\s*(?:pls|plz|please)\b[\s,:-]*", re.IGNORECASE)
_ENG_SHORT = re.compile(r"\b(to|into|in)\s+eng\b", re.IGNORECASE)
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


def _normalise_request(text: str) -> str:
    """Normalise the wrapper, never the text being translated.

    The panel receives shorthand: PLS, PLZ, "to ENG". Those are part of the
    request, not part of the sentence to translate, and the extractor cannot
    find the payload while they are in the way.
    """
    text = _nfc(text).strip()
    text = _LEAD_PLEASE.sub("please ", text)
    text = _ENG_SHORT.sub(lambda m: f"{m.group(1)} English", text)
    return text


def _last_assistant(history: list[tuple[str, str]] | None) -> str | None:
    for role, content in reversed(history or []):
        if role == "assistant" and str(content).strip():
            return str(content).strip()
    return None


def _refers_back(text: str) -> bool:
    """Is this a language task whose text is the previous answer?"""
    if not _DEICTIC.search(text):
        return False
    return any(pattern.search(text) for pattern in (
        _TO_CREE, _TO_ENGLISH, _TO_SYLLABICS, _TO_SRO,
    ))


def _mentions_note(text: str, notes: list[tuple[str, str]]) -> bool:
    low = _nfc(text).lower()
    for title, _passage in notes:
        title = str(title).strip().lower()
        if len(title) < 2:
            continue
        if re.search(rf"(?<!\w){re.escape(title)}(?!\w)", low):
            return True
    return False


def _asked_for_cree(text: str) -> bool:
    return bool(_TO_CREE.search(text)) and not bool(_MEANING_QUESTION.search(text))


def _english_question(text: str) -> str:
    """The question without the output-language wrapper.

    The first pass answers in English, so the prompt must not still say "answer
    in Cree". The Cree is produced from that answer, sentence by sentence, in
    the second pass.
    """
    q = _nfc(text).strip()
    q = re.sub(
        r"^\s*(?:please\s+)?(?:answer|reply|respond|explain|tell\s+me)\s+in\s+"
        r"(?:plains\s+)?cree\s*[:\-]\s*",
        "", q, flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+in\s+(?:plains\s+)?cree\s*([?.!]*)\s*$", r"\1", q,
               flags=re.IGNORECASE)
    return q.strip()


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


# Double quotes only. An apostrophe is not a quotation mark here, and treating
# it as one would cut "it's John's hat" down to "s John".
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,})[\"“”]")
_COLON = re.compile(
    r"^\s*(?:please\s+)?(?:translate|write|convert|render|say)\b[^:\n]*[:\n]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SAY = re.compile(
    r"^\s*(?:how\s+(?:do|would|can)\s+(?:you|i)\s+say)\s+(.+?)"
    r"(?:\s+in\s+(?:plains\s+)?cree)?\s*[?.!]*$",
    re.IGNORECASE,
)
_WORD_FOR = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+)?(?:the\s+)?(?:cree\s+)?(?:word|translation)\s+for\s+(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_IN_CREE = re.compile(
    r"^\s*what\s+is\s+(.+?)\s+in\s+(?:plains\s+)?cree\s*[?.!]*$",
    re.IGNORECASE | re.DOTALL,
)
_INTO = re.compile(
    r"^\s*(?:please\s+)?(?:translate|say|write|render|convert|give\s+me)\s+(.+?)\s+"
    r"(?:in|into|to)\s+(?:plains\s+)?(?:"
    r"cree\s+(?:syllabics?|cans)|"
    r"cree|english|syllabics?|cans|sro|standard\s+roman\s+orthography"
    r")\s*[?.!]*$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_body(body: str) -> str:
    body = body.strip().strip("\"“”'")
    if body.lower().startswith("the word "):
        body = body[len("the word "):].strip()
    return body


def _payload(message: str, history: list[tuple[str, str]] | None = None) -> str:
    """The text to translate, with the request around it removed.

    The model was trained on ``Translate into Plains Cree (SRO):`` followed by
    the sentence to translate, not by the question that asked for it. Passing
    "How do you say river in Cree?" as that sentence makes the model translate
    the question, and it answers something that is not "river". This unwraps
    the common request shapes so the training prompt receives the phrase.

    A follow up such as "can you answer that in Cree" carries no text of its
    own. ``history`` supplies the last assistant answer, which is what "that"
    means in a conversation. Without it the model is asked to translate the
    request, which is exactly the failure this function exists to prevent.
    """
    text = _normalise_request(message)
    quoted = _QUOTED.search(text)
    if quoted and quoted.group(1).strip():
        return quoted.group(1).strip()

    # An explicit colon wins over a pronoun: "translate this: <text>" names
    # its own payload even though it also says "this".
    match = _COLON.match(text)
    if match:
        body = _clean_body(match.group(1))
        if body and body.lower() not in {"this", "that", "it"}:
            return body

    if _refers_back(text):
        last = _last_assistant(history)
        if last:
            return last

    for pattern in (_SAY, _WORD_FOR, _IN_CREE, _INTO):
        match = pattern.match(text)
        if match:
            body = _clean_body(match.group(1))
            if body and body.lower() not in {"this", "that", "it"}:
                return body
    return text


def _is_translation(text: str) -> bool:
    if _TO_ENGLISH.search(text):
        return True
    if _MEANING_QUESTION.search(text):
        return False
    if _TO_CREE.search(text) or _is_script_conversion(text):
        return True
    # A bare Cree sentence, with no English question around it.
    return looks_like_cree(text) and not _ENGLISH_QUESTION.search(text)


def task_of(message: str, has_notes: bool = False,
            notes: list[tuple[str, str]] | None = None) -> str:
    """Return 'orthography', 'translate', 'song_notes' or 'chat'.

    This is a prompt-shape decision, not a model choice: every branch is
    answered by the same model.

    The order matters. "Explain Horse in Cree" contains the word Cree, but it
    is a question about a retrieved note, not a request to translate the
    sentence that asked it. A strong vocabulary marker comes first; then a
    retrieved title plus a question word; only then the language heuristics.
    """
    text = _normalise_request(message)
    if _is_script_conversion(text):
        return "orthography"
    if _STRONG_VOCAB.search(text):
        return "translate"
    if has_notes and notes and _mentions_note(text, notes) and _SONG_INTENT.search(text):
        return "song_notes"
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


def translation_prompt(text: str, target: str) -> str:
    """The exact translation header for one sentence or phrase."""
    text = str(text).strip()
    if target == "en":
        marker = "<|cans|>\n" if script_of(text) == "Cans" else ""
        return f"<|translate|><|cree|><|english|>\n{marker}Translate into English:\n{text}"
    if target == "cans":
        return f"<|translate|><|english|><|cans|>\nWrite this in Cree syllabics:\n{text}"
    return f"<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\n{text}"


@dataclass(frozen=True)
class PromptPlan:
    """One prompt, and what the caller needs to know about it."""
    task: str
    prompt: str
    payload: str
    answer_language: str       # "en" | "cr"
    used_history: bool         # the payload came from the previous answer


def route(message: str, notes: list[tuple[str, str]] | None = None,
          history: list[tuple[str, str]] | None = None) -> PromptPlan:
    """Resolve a message, its notes and its referent into one prompt plan.

    ``history`` is a list of ``(role, content)`` pairs, oldest first. It is
    used for one thing only: when a language request says "that" or "this",
    the previous assistant answer is the text to translate. The model still
    gets one prompt, not a pile of instructions.
    """
    notes = list(notes or [])
    message = str(message).strip()
    text = _normalise_request(message)
    task = task_of(message, bool(notes), notes)
    body = _payload(message, history)
    last = _last_assistant(history)
    used_history = bool(last and body == last and body != text)
    # A script request can refer back to the last answer ("convert that to
    # syllabics"). The request itself is not Cree, so task_of sees a chat; the
    # resolved body is the Cree text, and that is what decides the task.
    if (
        task == "chat"
        and _refers_back(text)
        and (_TO_SYLLABICS.search(text) or _TO_SRO.search(text))
        and script_of(body) is not None
    ):
        task = "orthography"
    answer_language = "cr" if task == "song_notes" and _asked_for_cree(text) else "en"

    if task == "orthography":
        if script_of(body) == "Cans":
            prompt = f"<|cans|><|sro|>\nWrite this syllabics sentence in SRO:\n{body}"
        else:
            prompt = f"<|sro|><|cans|>\nWrite this in Cree syllabics:\n{body}"
        return PromptPlan(task, prompt, body, "cr", used_history)

    if task == "translate":
        if _TO_ENGLISH.search(text) or (
            looks_like_cree(body) and not _TO_CREE.search(text)
        ):
            target = "en"
        elif _TO_SYLLABICS.search(text):
            target = "cans"
        else:
            target = "cr"
        return PromptPlan(
            task, translation_prompt(body, target), body,
            "cr" if target in {"cr", "cans"} else "en", used_history,
        )

    if task == "song_notes" and notes:
        blocks = "\n\n".join(
            f"Notes on {title}:\n{text_.strip()}" for title, text_ in notes if text_.strip()
        )
        if blocks:
            # The first pass is always English. A request for Cree is translated
            # from that grounded answer, sentence by sentence, rather than
            # asking the model to think and translate at the same time.
            question = _english_question(message) if answer_language == "cr" else message
            if answer_language == "cr":
                question = f"{question}\nAnswer in English."
            prompt = f"<|song|><|note|>\n{blocks}\n\nQuestion: {question}"
            return PromptPlan(task, prompt, body, answer_language, used_history)

    return PromptPlan("chat", f"<|song|>\n{message}", body, "en", used_history)


def format_prompt(message: str, notes: list[tuple[str, str]] | None = None,
                  history: list[tuple[str, str]] | None = None) -> str:
    """Wrap one user message in the exact shape the model was trained on.

    ``notes`` is a list of ``(song title, passage)`` pairs, already retrieved
    and relevance-filtered by the caller. Notes are dropped for a translation
    or script request, because song prose is not evidence about a word.

    The task is decided from the full message, because that is where the request
    lives. The text placed after the training header is the payload, so a
    question like "How do you say river in Cree?" sends ``river`` and not the
    question. ``history`` supplies the text when the request says "that".
    """
    return route(message, notes, history).prompt

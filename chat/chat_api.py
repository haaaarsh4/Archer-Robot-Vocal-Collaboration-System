from __future__ import annotations

import base64
import io
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from loguru import logger

from chat.rag_index import SongNotesIndex
from chat.llm_client import LocalLLMClient, LLMConfig, ChatTurn
from chat.archer_prompt import (
    ARCHER_SYSTEM_PROMPT,
    looks_like_cree,
    route,
    script_of,
    translation_prompt,
    wants_cree,
)
from chat.cree_glossary import asked_word, gloss_of
from chat.language_id import language_of
from chat.tts_engine import LocalTTS
from chat.memory import Memory
from chat import learning
from chat.web_search import WebSearch, query_for as _web_query, sources as _web_sources

router = APIRouter(prefix="/api/chat", tags=["chat"])

# A retrieved passage is only evidence when it is genuinely about the question.
# One floor for the chat and for the retrieval demo, so the panel cannot show a
# passage the model was not given.
MIN_NOTE_SCORE = 0.30

# One store and one search cache for the process. Both fail soft, so a locked
# database or an offline machine cannot take the chat down with it.
_MEMORY = Memory()
_WEB = WebSearch()

# Translation was trained on one sentence at a time. A paragraph is split before
# it reaches the model; handing it four sentences at once is what makes it copy
# the English back or answer a question instead of translating it.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CREE_TRANSLATION_MAX_SENTENCES = 2

# A Cree answer that is really copied English, or a stuck loop, is caught by
# the trained language identifier and the repetition counts below. There is no
# list of English words here any more: a list only catches the words somebody
# thought of, and English inside a Cree sentence is exactly what the identifier
# is trained to see.


def _plain_text(text: str) -> str:
    """Strip the markdown the model copies out of the notes."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    return text


# The assistant speaks; it does not write typography. An em dash where a comma
# belongs is the mark of machine-written English, and it is the first thing a
# reader notices in a two-sentence answer. The system prompt asks for ordinary
# punctuation, and this is the part that does not depend on the model obeying:
# every reply goes through here before it is sent.
#
# An em dash always becomes a comma. An en dash only does when it stands between
# spaces, because "1868–1870" is a range and a range is not an aside.
_DASH_ASIDE = re.compile(r"\s*\u2014\s*")
_SPACED_DASH = re.compile(r"\s+[\u2013-]\s+")
_DOUBLE_COMMA = re.compile(r",\s*,")
_COMMA_BEFORE_STOP = re.compile(r",\s+([.!?])")
# Only a comma at the very start, and only so a leading em dash that became
# ", " is dropped. It must not touch whitespace on its own: the streaming path
# runs this per token, and every token of a sentence arrives with the space in
# front of it.
_LEAD_COMMA = re.compile(r"^\s*,\s*")
# A token that is only a dash or only a space is not a word yet: the streaming
# path holds it one token so it is flattened with the words on either side.
_DASH_ONLY = re.compile(r"[\s\u2013\u2014-]*")


def _plain_dashes(text: str) -> str:
    """No em dashes, no spaced en dashes: a comma instead."""
    text = _DASH_ASIDE.sub(", ", str(text))
    text = _SPACED_DASH.sub(", ", text)
    text = _DOUBLE_COMMA.sub(",", text)
    text = _COMMA_BEFORE_STOP.sub(r"\1", text)
    text = _LEAD_COMMA.sub(lambda m: m.group(0).lstrip(" \t,"), text)
    return text


def _gloss_reply(message: str) -> str | None:
    """What one Cree word means, from the project's dictionaries, or nothing.

    A 7B model asked what a Cree word means will answer confidently whether or
    not it knows: that is how "tânisi" came back as "how many". A gloss is a
    fact, so it is read from Watkins' dictionary and the aligned pairs, and when
    neither carries the word the answer says so instead of guessing. Words that
    are not Cree are left to the model, which can talk about English words.
    """
    word = asked_word(message)
    # `script_of`, not `looks_like_cree`: one Cree word on its own is one signal,
    # and looks_like_cree wants two, so it would send "what does maskwa mean"
    # to the model and straight back to a guess. A word with a gloss is answered
    # whether or not it is recognised as Cree, because the dictionary knows it.
    hit = gloss_of(word or "")
    if not hit and (not word or script_of(word) is None):
        return None
    if not hit:
        return (
            f'I do not have a gloss for "{word}" in the dictionaries on this '
            "machine, so I will not guess at one."
        )
    meaning, source = hit
    said = f'{word} means "{meaning}".'
    return f"{said} ({source})" if source else said


def _bare_word_reply(message: str) -> str | None:
    """One Cree word sent on its own, answered as a word rather than translated.

    A single word is not a sentence to translate, and the translation path made
    a mess of one: "Tânisi" came back as "Tanis for". Any word the dictionaries
    carry is answered as its meaning; anything longer is a question for the
    model.
    """
    text = " ".join(str(message).split()).strip(" \t.?,!\u2026")
    if not text or len(text.split()) > 1:
        return None
    # The gloss decides, not the script guess: "maskisin" has no long vowel and
    # is not in the function-word list, so script_of calls it nothing and the
    # word went to the model, which answered "maskisina". A word the dictionary
    # carries is answered from the dictionary.
    hit = gloss_of(text)
    if not hit:
        if script_of(text) is None:
            return None
        return (
            f'I do not have a gloss for "{text}" in the dictionaries on this '
            "machine, so I will not guess at one."
        )
    meaning, source = hit
    said = f'{text} means "{meaning}".'
    return f"{said} ({source})" if source else said


def _quick_reply(message: str) -> str | None:
    """The only answers that skip the model: what a word means.

    A greeting, a thanks, "who are you" and every other conversational message
    is the model's own answer, written from the prompt. A table of canned
    sentences is not a conversation, and it only ever covers the exact cases
    somebody wrote down. What stays here is a fact the dictionaries hold, and a
    fact does not need a model to state it.
    """
    return _gloss_reply(message) or _bare_word_reply(message)


def _reliable_cree(text: str) -> bool:
    """True when a reply reads as Cree rather than copied English or a loop."""
    verdict = language_of(text)
    # Cree passes. A mixed reply passes only when almost all of it is Cree and
    # at most one word reads as English, which leaves room for a name and
    # rejects a clause of copied English. The reading is the identifier's, not a
    # list of English words.
    english_words = sum(1 for _w, label, _c in verdict.words if label == "en")
    if not (
        (verdict.lang == "cr" and verdict.cree_share >= 0.5)
        or (verdict.lang == "mixed" and verdict.cree_share >= 0.85 and english_words <= 1)
    ):
        return False
    words = re.findall(r"[A-Za-zâêîôûāēīōū'’‑-]+", text.lower())
    if len(words) >= 8 and len(set(words)) / len(words) < 0.4:
        return False
    counted = [word for word in words if len(word) >= 3]
    if counted:
        repeated = max(Counter(counted).values())
        if repeated >= 4 and repeated / len(counted) >= 0.2:
            return False
    return True

_rag_index: Optional[SongNotesIndex] = None
_llm_client: Optional[LocalLLMClient] = None
_tts: Optional[LocalTTS] = None
_svs_pipeline = None  # chat.svs_pipeline.ScratchToSungVoice, built with the app's neural_timbre instance


def _get_rag() -> SongNotesIndex:
    global _rag_index
    if _rag_index is None:
        _rag_index = SongNotesIndex(notes_dir="song_notes")
        n = _rag_index.build_or_load()
        logger.info(f"Song notes RAG index ready: {n} chunks indexed from song_notes/")
    return _rag_index


def _get_llm() -> LocalLLMClient:
    global _llm_client
    if _llm_client is None:
        try:
            from config.config_loader import get_config
            cfg = get_config()
            ccfg = cfg.get("chat", {})
            llm_cfg = LLMConfig(
                backend=ccfg.get("backend", "ollama"),
                ollama_url=ccfg.get("ollama_url", "http://127.0.0.1:11434"),
                model=str(ccfg.get("model", "archer:14b") or "archer:14b"),
                gguf_path=ccfg.get("gguf_path"),
                temperature=float(ccfg.get("temperature", 0.3)),
                max_tokens=int(ccfg.get("max_tokens", 260)),
                keep_alive=str(ccfg.get("keep_alive", "1h") or "1h"),
                num_thread=int(ccfg.get("num_thread", 0) or 0),
                min_p=float(ccfg.get("min_p", 0.05)),
                top_p=float(ccfg.get("top_p", 1.0)),
                top_k=int(ccfg.get("top_k", 0) or 0),
                dry_multiplier=float(ccfg.get("dry_multiplier", 0.8)),
                dry_base=float(ccfg.get("dry_base", 1.75)),
                dry_allowed_length=int(ccfg.get("dry_allowed_length", 2)),
                repeat_penalty=ccfg.get("repeat_penalty"),
                repeat_last_n=ccfg.get("repeat_last_n"),
            )
        except Exception:
            llm_cfg = LLMConfig()
        _llm_client = LocalLLMClient(llm_cfg)
    return _llm_client


def _get_tts() -> LocalTTS:
    global _tts
    if _tts is None:
        try:
            from config.config_loader import get_config
            voice_path = get_config().get("tts", {}).get("voice_path")
        except Exception:
            voice_path = None
        _tts = LocalTTS(voice_path=voice_path)
    return _tts


def _plain_for_translation(text: str) -> str:
    """Remove the formatting characters that are not part of the sentence."""
    text = re.sub(r"[*_`#]+", "", str(text))
    text = re.sub(r"^\s*[-\u2022]\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(_plain_for_translation(text)) if p.strip()]
    return parts or [_plain_for_translation(text)]


def _translation_target(plan) -> str:
    if plan.prompt.startswith("<|translate|><|english|><|cans|>"):
        return "cans"
    if plan.prompt.startswith("<|translate|><|english|>"):
        return "cr"
    return "en"


def _translate_sentences(llm: LocalLLMClient, text: str, target: str) -> str:
    """Translate one sentence at a time, the shape the model was trained on."""
    pieces: list[str] = []
    for sentence in _split_sentences(text):
        piece = llm.chat(
            translation_prompt(sentence, target), [], system_prompt=ARCHER_SYSTEM_PROMPT,
        ).strip()
        if piece:
            pieces.append(piece)
    return " ".join(pieces).strip()


def _answer_in_cree(llm: LocalLLMClient, english: str) -> str:
    return _translate_sentences(llm, english, "cr")


def _get_svs(neural_timbre):
    global _svs_pipeline
    if _svs_pipeline is None:
        from chat.svs_pipeline import ScratchToSungVoice, SVSConfig
        try:
            from config.config_loader import get_config
            scfg = get_config().get("svs", {})
            svs_cfg = SVSConfig(
                diffsinger_url=scfg.get("diffsinger_url", "http://127.0.0.1:8802"),
                diffsinger_timeout_s=float(scfg.get("diffsinger_timeout_s", 600.0)),
                default_voice_index=int(scfg.get("default_voice_index", 0)),
            )
        except Exception:
            svs_cfg = SVSConfig()
        _svs_pipeline = ScratchToSungVoice(neural_timbre, svs_cfg)
    return _svs_pipeline


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    mode: str = "text"          # "text" | "audio"
    session_id: str = "local"   # the browser's stable id, for memory


class PromptRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    session_id: str = "local"
    turn_id: int | None = None
    rating: str                 # "up" | "down"
    correction: str = ""
    note: str = ""


class RememberRequest(BaseModel):
    text: str
    session_id: str = "local"


@router.get("/health")
def chat_health():
    try:
        rag = _get_rag()
        rag_ok, rag_chunks = True, rag.num_chunks
    except Exception as e:
        rag_ok, rag_chunks = False, 0
        logger.error(f"RAG index unavailable: {e}")

    tts = _get_tts()
    llm = _get_llm()
    return {
        "rag_ready": rag_ok,
        "rag_chunks_indexed": rag_chunks,
        "tts_ready": tts.available,
        "backend": llm.cfg.backend,
        "model": llm.cfg.model if llm.cfg.backend == "ollama" else llm.cfg.gguf_path,
        "model_installed": llm.installed(),
        "note_floor": MIN_NOTE_SCORE,
        "memory": _MEMORY.stats(),
        "web_search": str(
            ((_chat_settings().get("web_search") or {}).get("mode", "auto"))
        ),
    }


@router.post("/feedback")
def chat_feedback(req: FeedbackRequest):
    """A vote, and the answer the visitor says was right.

    The correction is stored against the turn it corrects and exported at once
    as a DPO pair and a regression case, so the same mistake is both remembered
    immediately and unable to disappear from the next evaluation.
    """
    if req.rating not in {"up", "down"}:
        return JSONResponse(
            {"error": "rating must be 'up' or 'down'"}, status_code=400,
        )
    correction = req.correction
    if correction and learning.looks_like_correction(correction):
        correction = learning.corrected_answer(correction)
    feedback_id = _MEMORY.record_feedback(
        req.session_id, req.turn_id, req.rating, correction, req.note,
    )
    exported: dict = {}
    if req.rating == "down" and correction.strip():
        try:
            exported = learning.export(_MEMORY)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[learning] export failed: {exc}")
    return {
        "ok": bool(feedback_id),
        "feedback_id": feedback_id,
        "exported": exported,
        "stats": _MEMORY.stats(),
    }


@router.get("/memory")
def chat_memory():
    """What the assistant remembers, in counts. Nothing is sent off the machine."""
    return _MEMORY.stats()


@router.delete("/memory")
def chat_memory_clear():
    _MEMORY.clear()
    return {"ok": True, "stats": _MEMORY.stats()}


@router.post("/remember")
def chat_remember(req: RememberRequest):
    fact_id = _MEMORY.remember(req.text)
    return {"ok": bool(fact_id), "fact_id": fact_id, "stats": _MEMORY.stats()}


@router.post("/prompt")
def chat_prompt(req: PromptRequest):
    """The exact prompt the one model receives, for the site's demo.

    It runs the same retrieval and the same `format_prompt` as `chat()`, and
    stops before the model. The task label is the prompt shape, not a model
    choice: every branch is answered by the same model.
    """
    message = (req.message or "").strip()
    retrieved = _get_rag().query(message, top_k=4, min_score=MIN_NOTE_SCORE)
    notes = [(Path(r.chunk.source).stem, r.chunk.text) for r in retrieved if r.chunk.text.strip()]
    plan = route(message, notes, [])
    verdict = language_of(message)
    used_notes = notes if plan.task == "song_notes" else []
    return {
        "message": message,
        "task": plan.task,
        "answer_language": plan.answer_language,
        "script": script_of(message),
        "looks_like_cree": looks_like_cree(message),
        "language": verdict.lang,
        "language_confidence": round(verdict.confidence, 3),
        "cree_share": round(verdict.cree_share, 3),
        "words": [
            {"word": word, "language": label, "confidence": round(conf, 3)}
            for word, label, conf in verdict.words
        ] if verdict.lang == "mixed" else [],
        "wants_cree": wants_cree(message),
        "notes": [
            {"source": r.chunk.source, "score": round(r.score, 4), "text": r.chunk.text}
            for r in retrieved
        ],
        "notes_used": [title for title, _ in used_notes],
        "prompt": plan.prompt,
        "system_prompt": ARCHER_SYSTEM_PROMPT,
    }


@router.post("/reindex")
def chat_reindex():
    rag = _get_rag()
    n = rag.build_or_load(force_rebuild=True)
    return {"chunks_indexed": n}


@router.post("/retrieve")
def chat_retrieve(req: ChatRequest):
    """The retrieval half of the chat, exposed on its own for the site's demo.

    It runs exactly the same embedding and cosine search as ``chat()`` and
    stops before the language model, so the evidence can be inspected without
    Ollama running and without a generated sentence in the way.
    """
    try:
        rag = _get_rag()
        retrieved = rag.query(req.message, top_k=4, min_score=MIN_NOTE_SCORE)
        return {
            "query": req.message,
            "indexed": rag.num_chunks,
            "floor": MIN_NOTE_SCORE,
            "chunks": [
                {"source": r.chunk.source, "score": round(r.score, 4),
                 "text": r.chunk.text}
                for r in retrieved
            ],
        }
    except Exception as e:  # noqa: BLE001 - surfaced to the demo panel
        logger.error(f"[chat retrieve error] {e}")
        return JSONResponse({"error": f"Retrieval failed: {e}"}, status_code=500)


def warm_chat_model() -> bool:
    """Build the notes index and load the chat model once, at startup.

    The model is the expensive half. Loading it here, rather than inside the
    first question somebody asks, is what stops the first reply of a session
    from being the slowest one; ``keep_alive`` then keeps it resident.
    """
    try:
        rag = _get_rag()
        logger.info(f"[chat] notes index ready: {rag.num_chunks} chunks")
    except Exception as e:  # noqa: BLE001 - warming is best effort
        logger.warning(f"[chat] notes index warm-up failed: {e}")
    try:
        return _get_llm().warm()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[chat] model warm-up failed: {e}")
        return False


def _chat_settings() -> dict:
    try:
        from config.config_loader import get_config
        return (get_config() or {}).get("chat", {}) or {}
    except Exception:  # noqa: BLE001 - defaults are safe
        return {}


def _memory_on() -> bool:
    return bool((_chat_settings().get("memory") or {}).get("enabled", True))


def _server_history(req: ChatRequest) -> list[tuple[str, str]]:
    """The conversation for the model, from the client or from memory.

    A browser sends its own history. A script, a curl call or a page refreshed
    mid-conversation does not, and without this the assistant forgets the
    question it is being asked to follow up on.
    """
    pairs = [(m.role, m.content) for m in req.history]
    if pairs or not _memory_on():
        return pairs
    out: list[tuple[str, str]] = []
    for turn in _MEMORY.recent(req.session_id, limit=6):
        out.append(("user", turn.user))
        out.append(("assistant", turn.assistant))
    return out


def _gather_context(req: ChatRequest, plan, notes) -> tuple[list, list, str | None, str | None]:
    """Memory, corrections and web results for one message.

    Returns (memory_hits, web_results, correction, remembered). Every branch is
    guarded: context is an enhancement, and must never turn a chat into a 500.
    """
    hits: list = []
    web: list = []
    correction = None
    remembered = None
    settings = _chat_settings()
    if _memory_on():
        try:
            last = _MEMORY.last_turn(req.session_id)
            if last and learning.looks_like_correction(req.message):
                # Store the corrected answer, not the complaint and never the
                # answer being rejected. Returning last.assistant here put the
                # wrong answer back into the next prompt as the "correction".
                fix = learning.corrected_answer(req.message)
                _MEMORY.record_feedback(
                    req.session_id, last.id, "down", fix,
                    "auto-detected correction; rejected: " + last.assistant[:200],
                )
                if fix:
                    # A correction is a DPO pair and a regression the moment it
                    # is made. Storing it without exporting it made the fast loop
                    # work and the next training run forget the mistake.
                    try:
                        learning.export(_MEMORY)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"[learning] export failed: {exc}")
                correction = fix or None
            fact = learning.remember_command(req.message)
            if fact:
                _MEMORY.remember(fact)
                remembered = fact
            recall_n = int((settings.get("memory") or {}).get("recall", 3) or 3)
            hits = _MEMORY.recall(req.message, limit=recall_n)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[memory] context failed: {exc}")

    if plan.task in {"chat", "song_notes"} and plan.answer_language == "en":
        try:
            web_cfg = settings.get("web_search") or {}
            query = _web_query(
                req.message, has_notes=bool(notes),
                mode=str(web_cfg.get("mode", "auto")),
            )
            if query:
                web = _WEB.search(
                    query,
                    max_results=int(web_cfg.get("max_results", 5) or 5),
                    timeout=float(web_cfg.get("timeout", 8) or 8),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[web] context failed: {exc}")
    return hits, web, correction, remembered


def _record_turn(req: ChatRequest, plan, reply: str) -> int:
    if not _memory_on():
        return 0
    try:
        return _MEMORY.record_turn(
            req.session_id, req.message, reply, prompt=plan.prompt,
            task=plan.task, answer_language=plan.answer_language,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[memory] turn not saved: {exc}")
        return 0


def _with_context(req: ChatRequest, plan, notes):
    """Route again with memory, a correction and web results in the prompt."""
    history_pairs = _server_history(req)
    hits, web, correction, remembered = _gather_context(req, plan, notes)
    if hits or web or correction:
        plan = route(
            req.message, notes, history_pairs,
            memory=[(hit.kind, hit.text) for hit in hits],
            web=[{"title": r.title, "url": r.url, "snippet": r.snippet} for r in web],
            correction=correction,
        )
    return plan, hits, web, remembered


def _cree_answer_pair(llm: LocalLLMClient, english: str) -> tuple[str, bool]:
    """The English answer, with the same answer in Cree above it.

    Used when the visitor wrote in Cree or asked for an answer in Cree. The model
    answers once in English and that answer is translated sentence by sentence,
    which is the shape it was trained on. If the translation does not read as
    Cree, the English answer is shown alone and says why, rather than printing a
    half-English sentence as if it were Cree.
    """
    cree = ""
    if len(_split_sentences(english)) <= _CREE_TRANSLATION_MAX_SENTENCES:
        try:
            cree = _answer_in_cree(llm, english)
        except Exception as exc:  # noqa: BLE001 - the English answer is still useful
            logger.warning(f"[chat] Cree translation failed: {exc}")
            cree = ""
    if cree and _reliable_cree(cree):
        return f"{cree}\n\nEnglish answer it was translated from:\n{english}", True
    return (
        f"{english}\n\n"
        "The Cree translation was not reliable enough to show, so this is "
        "the English answer."
    ), False


@router.post("")
def chat(req: ChatRequest):
    started = time.perf_counter()
    # A word's meaning is a fact, so it is read from the dictionaries instead of
    # being generated: no retrieval, no model, no wait. See _quick_reply.
    quick = _quick_reply(req.message)
    if quick:
        llm = _get_llm()
        turn_id = 0
        if _memory_on():
            turn_id = _MEMORY.record_turn(
                req.session_id, req.message, quick, task="chat",
            )
        return {
            "reply": quick,
            "sources": [],
            "task": "chat",
            "answer_language": "en",
            "machine_translation": False,
            "used_history": False,
            "language": "english",
            "notes_used": 0,
            "model": llm.cfg.model if llm.cfg.backend == "ollama" else "llama_cpp",
            "turn_id": turn_id,
            "memory_used": 0,
            "web_used": 0,
            "remembered": False,
        }
    try:
        rag = _get_rag()
        llm = _get_llm()

        retrieved = rag.query(req.message, top_k=4, min_score=MIN_NOTE_SCORE)
        notes = [(Path(r.chunk.source).stem, r.chunk.text) for r in retrieved if r.chunk.text.strip()]
        history_pairs = _server_history(req)
        plan = route(req.message, notes, history_pairs)
        plan, hits, web, remembered = _with_context(req, plan, notes)
        # Notes are part of the prompt only when this is a notes question. A
        # translation request is not padded with song prose, and a general
        # question keeps its own answer instead of being forced onto a song.
        used_notes = notes if plan.task == "song_notes" else []
        sources = sorted({r.chunk.source for r in retrieved}) if used_notes else []
        sources = list(dict.fromkeys(sources + _web_sources(web)))

        history = [ChatTurn(role=role, content=content) for role, content in history_pairs]
        # A translation or script prompt already carries the text to work on.
        # Replaying the previous turn as chat makes the model answer the turn
        # instead of translating the payload, so those calls get no history.
        chat_history = [] if plan.task in {"translate", "orthography"} else history
        if plan.task == "translate" and len(_split_sentences(plan.payload)) > 1:
            # The model was trained on one sentence per translation prompt. A
            # multi-sentence answer is split instead of handed over as a block.
            reply_text = _translate_sentences(
                llm, plan.payload, _translation_target(plan),
            )
        else:
            reply_text = llm.chat(plan.prompt, chat_history, system_prompt=ARCHER_SYSTEM_PROMPT)
        machine_translation = False

        if plan.task == "translate":
            target = _translation_target(plan)
            if target in {"cr", "cans"} and not _reliable_cree(reply_text):
                reply_text = (
                    "The model could not produce a reliable Cree translation for that "
                    "text."
                )
            elif target == "en" and looks_like_cree(reply_text):
                reply_text = "The model did not produce an English translation for that text."

        # A Cree-bearing message can make the model answer in Cree even when the
        # visitor wrote English and the router chose English. Ask once more with
        # the instruction spelled out; if it still answers in Cree, translate the
        # answer so the output language matches the request.
        if (plan.answer_language == "en" and plan.task in {"chat", "song_notes"}
                and _reliable_cree(reply_text)):
            retry = (
                f"{plan.prompt}\n\nAnswer in English. Use English words only. "
                "Do not answer in Cree."
            )
            second = llm.chat(retry, chat_history, system_prompt=ARCHER_SYSTEM_PROMPT)
            if not _reliable_cree(second):
                reply_text = second
            elif second.strip():
                translated = _translate_sentences(llm, second, "en")
                reply_text = (
                    translated if translated and not _reliable_cree(translated) else second
                )
            else:
                reply_text = second

        if plan.answer_language == "cr" and plan.task in {"song_notes", "chat"}:
            reply_text, machine_translation = _cree_answer_pair(llm, reply_text)

        reply_text = _plain_dashes(_plain_text(reply_text))
        used_model = llm.cfg.model if llm.cfg.backend == "ollama" else "llama_cpp"
        # Report what the reply is, not what was requested. Trusting the plan
        # labelled an unprompted Cree answer as English, which is how a visitor
        # ends up with Cree on screen and a response field saying otherwise.
        reply_is_cree = _reliable_cree(reply_text)

        turn_id = _record_turn(req, plan, reply_text)
        response: dict = {
            "reply": reply_text,
            "sources": sources,
            "task": plan.task,
            "answer_language": plan.answer_language,
            "machine_translation": machine_translation,
            "used_history": plan.used_history,
            "language": "cree" if reply_is_cree else "english",
            "notes_used": len(used_notes),
            "model": used_model,
            "turn_id": turn_id,
            "memory_used": len(hits),
            "web_used": len(web),
            "remembered": bool(remembered),
        }

        if req.mode == "audio":
            tts = _get_tts()
            if not tts.available:
                response["audio_error"] = (
                    "Text-to-speech voice not configured, set tts.voice_path in config.yaml. "
                    "Returning text only."
                )
            else:
                wav_bytes = tts.synthesize(reply_text)
                response["audio_base64"] = base64.b64encode(wav_bytes).decode("ascii")

        # The one number that says whether a reply was slow because of the
        # model, the retrieval or the machine. It goes to the log on every
        # answer, so "it takes ten minutes" can be read rather than guessed.
        logger.info(
            f"[chat] {plan.task} answered in {time.perf_counter() - started:.1f}s "
            f"({len(used_notes)} notes, {len(reply_text)} chars, "
            f"{len(chat_history)} history turns)"
        )
        return response

    except RuntimeError as e:
        # Backend not reachable, or the one model is not installed: a clear
        # message rather than a 500 stack trace.
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        logger.error(f"[chat error] {e}")
        return JSONResponse({"error": f"Chat failed: {e}"}, status_code=500)


def _sse(payload: dict) -> str:
    """One server-sent event: a `data:` line, then the blank line that ends it."""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _stream_events(req: ChatRequest):
    """The same answer POST /api/chat gives, written out as it is generated.

    Two tasks are single-pass and can be streamed token by token: a general
    answer and a note answer, both in English. Everything else -- a translation
    split into sentences, a Cree answer translated from an English one -- is
    built from more than one model call, so it is answered once and sent as one
    piece rather than pretending to arrive gradually.
    """
    started = time.perf_counter()

    quick = _quick_reply(req.message)
    if quick:
        turn_id = 0
        if _memory_on():
            turn_id = _MEMORY.record_turn(
                req.session_id, req.message, quick, task="chat",
            )
        yield _sse({"type": "token", "text": quick})
        yield _sse({"type": "done", "reply": quick, "sources": [],
                    "task": "chat", "notes_used": 0, "answer_language": "en",
                    "turn_id": turn_id, "memory_used": 0, "web_used": 0,
                    "remembered": False})
        return

    try:
        llm = _get_llm()
        rag = _get_rag()
        retrieved = rag.query(req.message, top_k=4, min_score=MIN_NOTE_SCORE)
        notes = [(Path(r.chunk.source).stem, r.chunk.text)
                 for r in retrieved if r.chunk.text.strip()]
        history_pairs = _server_history(req)
        plan = route(req.message, notes, history_pairs)
        plan, hits, web, remembered = _with_context(req, plan, notes)
        used_notes = notes if plan.task == "song_notes" else []
        sources = sorted({r.chunk.source for r in retrieved}) if used_notes else []
        sources = list(dict.fromkeys(sources + _web_sources(web)))

        # A message containing Cree goes through the one-shot path below, which
        # can retry in English. Streaming it would show the Cree answer before
        # any check could intervene.
        if (plan.task in {"chat", "song_notes"} and plan.answer_language == "en"
                and language_of(req.message).cree_share == 0):
            yield _sse({"type": "meta", "task": plan.task,
                        "sources": sources, "notes_used": len(used_notes)})
            history = [ChatTurn(role=m.role, content=m.content) for m in req.history]
            pieces: list[str] = []
            held = ""
            for piece in llm.stream_chat(plan.prompt, history):
                held += piece
                if len(held) <= 4 and _DASH_ONLY.fullmatch(held):
                    continue          # a dash or a space with no word around it
                clean = _plain_dashes(held)
                held = ""
                if clean:
                    pieces.append(clean)
                    yield _sse({"type": "token", "text": clean})
            tail = _plain_dashes(held)
            if tail:
                pieces.append(tail)
                yield _sse({"type": "token", "text": tail})
            reply = _plain_text("".join(pieces).strip())
            if plan.task == "song_notes" and not reply:
                reply = "The notes do not say."
            logger.info(
                f"[chat] streamed {plan.task} in {time.perf_counter() - started:.1f}s "
                f"({len(used_notes)} notes, {len(reply)} chars)"
            )
            turn_id = _record_turn(req, plan, reply)
            yield _sse({"type": "done", "reply": reply, "sources": sources,
                        "task": plan.task, "notes_used": len(used_notes),
                        "answer_language": "en", "turn_id": turn_id,
                        "memory_used": len(hits), "web_used": len(web),
                        "remembered": bool(remembered)})
            return

        result = chat(req)
        if isinstance(result, JSONResponse):
            detail = {}
            try:
                detail = json.loads(result.body.decode("utf-8"))
            except Exception:  # noqa: BLE001 - body is not ours to trust
                pass
            yield _sse({"type": "error", "error": detail.get("error", "Chat failed.")})
            return
        yield _sse({"type": "token", "text": result.get("reply", "")})
        yield _sse({"type": "done", **{k: result.get(k) for k in (
            "reply", "sources", "task", "notes_used", "answer_language",
            "machine_translation", "audio_base64", "audio_error", "turn_id",
            "memory_used", "web_used", "remembered")}})
    except RuntimeError as e:
        # backend not reachable, or the model is not installed: the same clear
        # message the one-shot endpoint returns
        yield _sse({"type": "error", "error": str(e)})
    except Exception as e:  # noqa: BLE001
        logger.error(f"[chat] stream failed: {e}")
        yield _sse({"type": "error", "error": f"Chat failed: {e}"})


@router.post("/stream")
def chat_stream(req: ChatRequest):
    """POST /api/chat, as server-sent events.

    The panel writes the tokens as they arrive, so an answer starts appearing in
    about a second instead of after the whole reply has been generated. The
    answer itself is identical: same route, same notes, same model.
    """
    return StreamingResponse(
        _stream_events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/sing")
async def sing_from_scratch_vocal(
    file: UploadFile = File(...),
    voice_index: int = Form(0),
):
    try:
        import server as _server  # the main app module, for its already-built neural_timbre instance
        neural_timbre = getattr(_server, "neural_timbre", None)
    except Exception:
        neural_timbre = None

    if neural_timbre is None:
        return JSONResponse(
            {"error": "Neural timbre stage not loaded on the server, check startup logs."},
            status_code=503,
        )

    svs = _get_svs(neural_timbre)

    raw = await file.read()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    try:
        converted = svs.hum_to_archer(audio, sr, voice_index=voice_index)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    out_buf = io.BytesIO()
    sf.write(out_buf, converted.astype(np.float32), sr, format="WAV")
    out_buf.seek(0)
    return Response(content=out_buf.read(), media_type="audio/wav")


class SingFromLyricsRequest(BaseModel):
    lyrics: str
    notes: list[dict]     # [{"pitch_midi": 60, "duration_s": 0.4, "lyric": "sun"}, ...]
    voice_index: int = 0


@router.post("/sing-from-lyrics")
def sing_from_lyrics(req: SingFromLyricsRequest):
    try:
        import server as _server
        neural_timbre = getattr(_server, "neural_timbre", None)
    except Exception:
        neural_timbre = None

    if neural_timbre is None:
        return JSONResponse(
            {"error": "Neural timbre stage not loaded on the server, check startup logs."},
            status_code=503,
        )

    svs = _get_svs(neural_timbre)
    try:
        converted = svs.lyrics_and_melody_to_archer(req.lyrics, req.notes, voice_index=req.voice_index)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    out_buf = io.BytesIO()
    sf.write(out_buf, converted.astype(np.float32), 44100, format="WAV")
    out_buf.seek(0)
    return Response(content=out_buf.read(), media_type="audio/wav")

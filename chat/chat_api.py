from __future__ import annotations

import base64
import io
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
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
from chat.tts_engine import LocalTTS

router = APIRouter(prefix="/api/chat", tags=["chat"])

# A retrieved passage is only evidence when it is genuinely about the question.
# One floor for the chat and for the retrieval demo, so the panel cannot show a
# passage the model was not given.
MIN_NOTE_SCORE = 0.30

# Translation was trained on one sentence at a time. A paragraph is split before
# it reaches the model; handing it four sentences at once is what makes it copy
# the English back or answer a question instead of translating it.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CREE_TRANSLATION_MAX_SENTENCES = 2

# A Cree answer that still carries English function words is the model copying
# its input, not translating it. The app hides that instead of printing it.
_ENGLISH_FUNCTION_WORDS = re.compile(
    r"\b(?:the|and|is|are|was|were|be|been|being|to|of|in|on|for|with|that|this|"
    r"it|as|at|by|from|we|you|they|he|she|his|her|their|our|not|but|or|if|then|"
    r"when|what|which|who|how|why|can|could|will|would|should|may|might|must|have|"
    r"has|had|do|does|did)\b",
    re.IGNORECASE,
)


def _plain_text(text: str) -> str:
    """Strip the markdown the model copies out of the notes."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    return text


def _reliable_cree(text: str) -> bool:
    """True when a reply reads as Cree rather than copied English or a loop."""
    if not looks_like_cree(text):
        return False
    if _ENGLISH_FUNCTION_WORDS.search(text):
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
                model=str(ccfg.get("model", "archer:7b") or "archer:7b"),
                gguf_path=ccfg.get("gguf_path"),
                temperature=float(ccfg.get("temperature", 0.3)),
                max_tokens=int(ccfg.get("max_tokens", 500)),
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


class PromptRequest(BaseModel):
    message: str


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
    }


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
    used_notes = notes if plan.task == "song_notes" else []
    return {
        "message": message,
        "task": plan.task,
        "answer_language": plan.answer_language,
        "script": script_of(message),
        "looks_like_cree": looks_like_cree(message),
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


@router.post("")
def chat(req: ChatRequest):
    try:
        rag = _get_rag()
        llm = _get_llm()

        retrieved = rag.query(req.message, top_k=4, min_score=MIN_NOTE_SCORE)
        notes = [(Path(r.chunk.source).stem, r.chunk.text) for r in retrieved if r.chunk.text.strip()]
        history_pairs = [(m.role, m.content) for m in req.history]
        plan = route(req.message, notes, history_pairs)
        # Notes are part of the prompt only when this is a notes question. A
        # translation request is not padded with song prose, and a general
        # question keeps its own answer instead of being forced onto a song.
        used_notes = notes if plan.task == "song_notes" else []
        sources = sorted({r.chunk.source for r in retrieved}) if used_notes else []

        history = [ChatTurn(role=m.role, content=m.content) for m in req.history]
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

        if plan.task == "song_notes" and plan.answer_language == "cr":
            english = reply_text
            cree = ""
            if len(_split_sentences(english)) <= _CREE_TRANSLATION_MAX_SENTENCES:
                try:
                    cree = _answer_in_cree(llm, english)
                except Exception as exc:  # noqa: BLE001 - the English answer is still useful
                    logger.warning(f"[chat] Cree translation failed: {exc}")
                    cree = ""
            if cree and _reliable_cree(cree):
                reply_text = (
                    f"{cree}\n\n"
                    f"English answer it was translated from:\n{english}"
                )
                machine_translation = True
            else:
                reply_text = (
                    f"{english}\n\n"
                    "The Cree translation was not reliable enough to show, so this is "
                    "the English answer."
                )

        reply_text = _plain_text(reply_text)
        used_model = llm.cfg.model if llm.cfg.backend == "ollama" else "llama_cpp"
        reply_is_cree = plan.answer_language == "cr" and _reliable_cree(reply_text)

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

        return response

    except RuntimeError as e:
        # Backend not reachable, or the one model is not installed: a clear
        # message rather than a 500 stack trace.
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        logger.error(f"[chat error] {e}")
        return JSONResponse({"error": f"Chat failed: {e}"}, status_code=500)


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

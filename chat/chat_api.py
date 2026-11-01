from __future__ import annotations

import base64
import io
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
from chat.cree_prompt import (
    CREE_SYSTEM_PROMPT,
    choose_route,
    format_cree_prompt,
    looks_like_cree,
    script_of,
    wants_cree,
)
from chat.tts_engine import LocalTTS

router = APIRouter(prefix="/api/chat", tags=["chat"])

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
                ollama_model=ccfg.get("ollama_model", "llama3.1:8b-instruct-q4_K_M"),
                gguf_path=ccfg.get("gguf_path"),
                cree_model=str(ccfg.get("cree_model", "") or ""),
                cree_gguf_path=ccfg.get("cree_gguf_path"),
                cree_temperature=float(ccfg.get("cree_temperature", 0.2)),
                cree_max_tokens=int(ccfg.get("cree_max_tokens", 400)),
                temperature=float(ccfg.get("temperature", 0.4)),
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
    language: str = "auto"      # "auto" | "cree" | "english"


class RouteRequest(BaseModel):
    message: str
    language: str = "auto"      # "auto" | "cree" | "english"


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
        "cree_model": llm.cfg.cree_model or None,
        "cree_model_configured": bool(llm.cfg.cree_model),
    }


@router.post("/route")
def chat_route(req: RouteRequest):
    """The routing decision on its own, for the site's demo.

    This is the same `choose_route` call the chat makes, with the same config,
    so the panel cannot drift from the behaviour it claims to explain. It
    reports the text signal and the final route separately: on a machine where
    the Cree model is not configured, Cree text still is Cree text, it just
    stays on the general model.
    """
    llm = _get_llm()
    message = (req.message or "").strip()
    configured = bool(llm.cfg.cree_model)
    cree_question = wants_cree(message)
    return {
        "message": message,
        "requested": req.language,
        "configured": configured,
        "model": llm.cfg.cree_model or None,
        "script": script_of(message),
        "looks_like_cree": looks_like_cree(message),
        "wants_cree": cree_question,
        "route": choose_route(message, req.language, llm.cfg.cree_model),
        # The prompt is only meaningful when the text itself asks for Cree. On
        # the English route the app sends the general SYSTEM prompt instead.
        "cree_prompt": format_cree_prompt(message) if cree_question else None,
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
        retrieved = rag.query(req.message, top_k=4)
        return {
            "query": req.message,
            "indexed": rag.num_chunks,
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

        retrieved = rag.query(req.message, top_k=4)
        context_chunks = [r.chunk.text for r in retrieved]
        sources = sorted({r.chunk.source for r in retrieved})

        history = [ChatTurn(role=m.role, content=m.content) for m in req.history]

        # A Cree question goes to the model that was trained for it, with the
        # prompt it was trained with. Everything else stays on the general
        # model. An empty cree_model sends Cree text to the general model; a
        # configured tag that is missing from Ollama raises, and is reported
        # as a 503 rather than answered by the wrong model.
        route = choose_route(req.message, req.language, llm.cfg.cree_model)
        if route == "cree":
            # The notebook trains on "Notes on <song>:" followed by
            # "Question:", so the app sends that, not a generic context block.
            # Retrieval still runs for sources; only passages that clear a
            # relevance floor are put in front of the model, because a Cree
            # translation request should not be padded with song prose.
            cree_notes = [
                (Path(r.chunk.source).stem, r.chunk.text)
                for r in retrieved if r.score >= 0.30
            ]
            reply_text = llm.chat(
                format_cree_prompt(req.message, cree_notes), history, [],
                system_prompt=CREE_SYSTEM_PROMPT,
                model=llm.cfg.cree_model,
                temperature=llm.cfg.cree_temperature,
                max_tokens=llm.cfg.cree_max_tokens,
            )
            used_model = llm.cfg.cree_model
            if not cree_notes:
                sources = []
        else:
            reply_text = llm.chat(req.message, history, context_chunks)
            used_model = llm.cfg.ollama_model if llm.cfg.backend == "ollama" else "llama_cpp"

        response: dict = {
            "reply": reply_text,
            "sources": sources,
            "language": route,
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
        # backend not reachable, etc, a clear message rather than a 500 stack trace
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

import asyncio
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional
import threading
import time
import queue as _queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def _prewarm_rmvpe():
    from config.config_loader import get_config
    cfg = get_config()
    if not cfg.get("pitch", {}).get("rmvpe", {}).get("prewarm", True):
        return

    def _load():
        try:
            from analysis.pitch_detector import PitchDetector
            pd = PitchDetector()
            active = pd.set_method("rmvpe")
            if active == "rmvpe":
                print("[startup] RMVPE pre-warmed and ready.")
            else:
                print("[startup] RMVPE pre-warm skipped (see error above -- "
                      "check pitch.rmvpe.model_path in config.yaml).")
        except Exception as e:
            print(f"[startup] RMVPE pre-warm failed: {e}")

    threading.Thread(target=_load, daemon=True).start()

broadcast_queue: _queue.Queue = _queue.Queue(maxsize=64)
pipeline_running = False
pipeline_stop_event = threading.Event()

AUDIO_AVAILABLE = False
try:
    import pyaudio as _pyaudio
    _p = _pyaudio.PyAudio()
    _p.terminate()
    AUDIO_AVAILABLE = True
except Exception:
    pass

translator = None
translator_load_error = None
try:
    from translation.translate_transformer import NeuralTranslator
    translator = NeuralTranslator()
    print("Translation model loaded successfully.")
except Exception as e:
    translator_load_error = str(e)
    print(f"Translation model not loaded: {translator_load_error}")
    print("Check that data/models/transformer_mt.pt, spm.model, and "
          "spm_config.json exist relative to your project root.")


sentiment_analyzer = None            # VADER instance, if loaded
sentiment_load_error = None
roberta_sentiment = None             # transformers pipeline, if loaded
roberta_load_error = None

SENTIMENT_MODEL_DIR = "data/models/sentiment-roberta"

sentiment_analyzer = None            # VADER instance, if loaded
sentiment_load_error = None
roberta_sentiment = None             # transformers pipeline, if loaded
roberta_load_error = None

if not os.path.isdir(SENTIMENT_MODEL_DIR):
    roberta_load_error = (
        f"{SENTIMENT_MODEL_DIR} not found. Run 'python download_sentiment_model.py' once "
        "first (needs internet access to huggingface.co, one-time only) to populate it. "
        "server.py itself never downloads this model or makes any network call for it."
    )
    print(f"RoBERTa sentiment model not loaded: {roberta_load_error}")
    print("Falling back to VADER in the meantime.")
else:
    try:
        from transformers import pipeline as _hf_pipeline
        roberta_sentiment = _hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_DIR,
            tokenizer=SENTIMENT_MODEL_DIR,
            top_k=None,          # return all 3 class probabilities, not just the top label
            local_files_only=True,  # hard requirement: never touch the network from here
        )
        print(f"Sentiment analyzer (RoBERTa) loaded from local files at {SENTIMENT_MODEL_DIR}.")
    except Exception as e:
        roberta_load_error = str(e)
        print(f"RoBERTa sentiment model failed to load from {SENTIMENT_MODEL_DIR}: {roberta_load_error}")
        print("Falling back to VADER. If the directory looks right but this still fails, "
              "try re-running download_sentiment_model.py -- it may be an incomplete download.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print("Sentiment analyzer (VADER, fallback) loaded successfully.")
except Exception as e:
    sentiment_load_error = str(e)
    print(f"VADER fallback not loaded either: {sentiment_load_error}")
    print("Run: pip install vaderSentiment")

_INTENSIFIERS = {
    "very", "extremely", "so", "really", "deeply", "utterly", "completely",
    "totally", "absolutely", "incredibly", "always", "never", "forever",
}


def _roberta_valence(text: str):
    """Runs the real model. Returns (valence, emotional_charge) where
    valence is P(positive) - P(negative) in [-1, 1] -- a continuous score
    from the model's own class probabilities, not just its single loudest
    label -- and emotional_charge = 1 - P(neutral), i.e. how far the model
    is from calling this sentence neutral at all, in [0, 1]. Returns None
    if the model isn't loaded."""
    if roberta_sentiment is None:
        return None
    scores = {row["label"].lower(): row["score"] for row in roberta_sentiment(text)[0]}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral", 0.0)
    valence = max(-1.0, min(1.0, pos - neg))
    emotional_charge = max(0.0, min(1.0, 1.0 - neu))
    return valence, emotional_charge


def score_sentiment(text: str, vocal: dict = None) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided"}

    roberta_result = _roberta_valence(text)
    if roberta_result is not None:
        valence, model_charge = roberta_result
        engine = "roberta"
    elif sentiment_analyzer is not None:
        scores = sentiment_analyzer.polarity_scores(text)
        valence = max(-1.0, min(1.0, scores["compound"]))  # already -1..1
        model_charge = scores["pos"] + scores["neg"]  # 0..1, VADER's rough equivalent
        engine = "vader-fallback"
    else:
        return {"error": f"No sentiment backend loaded. RoBERTa: {roberta_load_error}. VADER: {sentiment_load_error}"}

    words = text.split()
    intensifier_hits = sum(1 for w in words if w.strip(".,!?").lower() in _INTENSIFIERS)
    exclaim = text.count("!")
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    length_factor = 1.0 if len(words) <= 6 else max(0.4, 6 / len(words))

    text_arousal_raw = (
        0.55 * model_charge
        + 0.12 * min(intensifier_hits, 3)
        + 0.12 * min(exclaim, 2)
        + 0.10 * min(caps_words, 2)
    ) * length_factor
    text_arousal = max(0.0, min(1.0, text_arousal_raw))

    audio_arousal = None
    if vocal:
        pitch_component = max(0.0, min(1.0, (vocal.get("pitch_range_semitones") or 0.0) / 24.0))
        loudness_component = max(0.0, min(1.0, vocal.get("rms_mean") or 0.0))
        dynamics_component = max(0.0, min(1.0, (vocal.get("rms_variance") or 0.0) * 8.0))
        audio_arousal = round(
            0.40 * loudness_component + 0.40 * pitch_component + 0.20 * dynamics_component, 3
        )

    arousal = round((text_arousal + audio_arousal) / 2.0, 3) if audio_arousal is not None else round(text_arousal, 3)

    if valence > 0.15:
        emotion = "joy"
        desc = "Positive tone detected in the translation."
        interval = "+4 (major 3rd)" if arousal > 0.5 else "+7 (5th)"
        osc = "triangle"
    elif valence < -0.15:
        emotion = "grief"
        desc = "Negative/sorrowful tone detected in the translation."
        interval = "+3 (minor 3rd)"
        osc = "sine"
    else:
        emotion = "neutral"
        desc = "No strong positive or negative tone detected."
        interval = "+7 (5th)"
        osc = "sine"

    vibrato = "3.5 Hz" if valence < -0.15 else "off"

    led_by_emotion = {
        "joy":     {"color": "#059669", "border": "#059669", "bg": "rgba(5,150,105,0.12)",  "pupil": "18px"},
        "grief":   {"color": "#818cf8", "border": "#818cf8", "bg": "rgba(129,140,248,0.10)", "pupil": "12px"},
        "neutral": {"color": "var(--muted)", "border": "var(--border)", "bg": "var(--bg3)",  "pupil": "14px"},
    }
    led = led_by_emotion[emotion]

    return {
        "text": text,
        "engine": engine,
        "valence": round(valence, 3),
        "arousal": arousal,
        "text_arousal": round(text_arousal, 3),
        "audio_arousal": audio_arousal,
        "emotion": emotion,
        "description": desc,
        "interval": interval,
        "osc": osc,
        "vibrato": vibrato,
        "led": led,
        "vader_raw": scores if engine == "vader-fallback" else None,  # only populated when VADER actually ran
    }


WHISPER_MODEL_DIR = "data/models/whisper"

MIC_WHISPER_MODEL_SIZE = "small.en"
MIC_WHISPER_MODEL_DIR = "data/models/faster-whisper-small.en"
TRACK_WHISPER_MODEL_SIZE = "large-v3-turbo"
TRACK_WHISPER_MODEL_DIR = "data/models/faster-whisper-large-v3-turbo"


def _load_faster_whisper_backend(model_size: str, model_dir: str, legacy_checkpoint: str):
    if os.path.isdir(model_dir):
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                  download_root=model_dir, local_files_only=True)
            print(f"faster-whisper ({model_size}, int8, CPU) loaded from local files at {model_dir}.")
            return model, "faster", None
        except Exception as e:
            print(f"faster-whisper ({model_size}) failed to load ({e}); trying legacy openai-whisper.")

    if os.path.isfile(legacy_checkpoint):
        try:
            import whisper as _whisper
            model = _whisper.load_model(legacy_checkpoint)
            print(f"[legacy backend] openai-whisper loaded from local files at {legacy_checkpoint}.")
            return model, "openai", None
        except Exception as e:
            return None, None, str(e)

    error = (
        f"Neither faster-whisper ({model_dir}) nor the legacy checkpoint ({legacy_checkpoint}) "
        f"were found. Run 'python download_faster_whisper_model.py' once (needs internet, one-time "
        f"only) or 'python download_whisper_model.py' for the older backend."
    )
    print(f"Whisper model not loaded ({model_size}): {error}")
    return None, None, error


_legacy_whisper_checkpoint = os.path.join(WHISPER_MODEL_DIR, "base.en.pt")

whisper_model_mic, whisper_backend_mic, whisper_load_error_mic = _load_faster_whisper_backend(
    MIC_WHISPER_MODEL_SIZE, MIC_WHISPER_MODEL_DIR, _legacy_whisper_checkpoint
)
whisper_model_track, whisper_backend_track, whisper_load_error_track = _load_faster_whisper_backend(
    TRACK_WHISPER_MODEL_SIZE, TRACK_WHISPER_MODEL_DIR, _legacy_whisper_checkpoint
)
if whisper_model_track is None and whisper_model_mic is not None:
    print("Track-upload model not available; falling back to the mic model for /api/transcribe/annotated too "
          "(less accurate than large-v3-turbo would be -- run download_faster_whisper_model.py to fix this).")
    whisper_model_track, whisper_backend_track = whisper_model_mic, whisper_backend_mic

whisper_model = whisper_model_mic or whisper_model_track
whisper_backend = whisper_backend_mic or whisper_backend_track
whisper_load_error = whisper_load_error_mic or whisper_load_error_track


VOSK_MODEL_DIR = "data/models/vosk-model-en-us-0.22"
vosk_model = None
vosk_load_error = None
if os.path.isdir(VOSK_MODEL_DIR):
    try:
        from vosk import Model as VoskModel, SetLogLevel as _vosk_set_log_level
        _vosk_set_log_level(-1)  # Vosk/Kaldi logs straight to stderr by default and is very chatty; -1 silences it
        vosk_model = VoskModel(VOSK_MODEL_DIR)
        print(f"Vosk streaming model loaded from local files at {VOSK_MODEL_DIR} (instant local live captions enabled).")
    except Exception as e:
        vosk_load_error = str(e)
        print(f"Vosk failed to load: {vosk_load_error}")
else:
    vosk_load_error = (
        f"{VOSK_MODEL_DIR} not found. Run 'python download_vosk_model.py' once (needs internet, "
        "one-time only) to enable instant local live captions. Optional -- everything else in this "
        "file works without it."
    )
    print(f"Vosk not loaded: {vosk_load_error}")


harmony_engine = None
harmony_load_error = None
try:
    from synthesis.harmony_engine import HarmonyEngine
    harmony_engine = HarmonyEngine()
    print("Shared HarmonyEngine initialized (sovereignty + mode selection live here).")
except Exception as e:
    harmony_load_error = str(e)
    print(f"HarmonyEngine not loaded: {harmony_load_error}")


neural_timbre = None
neural_timbre_load_error = None
try:
    from synthesis.neural_timbre import NeuralTimbreConverter
    neural_timbre = NeuralTimbreConverter()
    if neural_timbre.enabled:
        print("NeuralTimbreConverter loaded and enabled.")
    else:
        print("NeuralTimbreConverter loaded but disabled/inactive "
              "(see synthesis.neural in config.yaml) — using pure DSP voice output.")
except Exception as e:
    neural_timbre_load_error = str(e)
    print(f"NeuralTimbreConverter not loaded: {neural_timbre_load_error}")
    print("This is non-fatal — the pipeline runs on pure DSP synthesis without it.")


try:
    from chat.chat_api import router as chat_router
    app.include_router(chat_router)
    print("Chat router mounted at /api/chat (backend readiness checked lazily on first call).")
except Exception as e:
    print(f"Chat router not mounted: {e}")
    print("This is non-fatal -- the rest of the app runs without the chatbot.")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/architecture")
@app.get("/harmony")
@app.get("/synthesis")
@app.get("/sentiment")
@app.get("/live-demo")
def spa_routes():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/devices")
def list_devices():
    if not AUDIO_AVAILABLE:
        return {"devices": [], "available": False,
                "message": "Running on a cloud server — local mic not available. Use the browser mic below."}
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({"index": i, "name": info["name"]})
        p.terminate()
        return {"devices": devices, "available": True}
    except Exception as e:
        return {"devices": [], "available": False, "message": str(e)}


class TranslateRequest(BaseModel):
    text: str


@app.post("/api/translate")
def translate(req: TranslateRequest):
    if translator is None:
        return JSONResponse({"error": f"Model not loaded: {translator_load_error}"}, status_code=503)

    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        translation = translator.translate(text)
        return {"input": text, "translation": translation}
    except Exception as e:
        return JSONResponse({"error": f"Translation failed: {e}"}, status_code=500)


@app.get("/api/translate/health")
def translate_health():
    return {"status": "ok" if translator else "model_not_loaded", "error": translator_load_error}


CREE_FST_PATH = "data/models/crk-descriptive-analyzer.hfstol"
cree_analyzer_fst = None
cree_fst_load_error = None
try:
    import hfst
    _istr = hfst.HfstInputStream(CREE_FST_PATH)
    cree_analyzer_fst = _istr.read()
    print(f"Cree morphological analyzer loaded ({CREE_FST_PATH}).")
except Exception as e:
    cree_fst_load_error = str(e)
    print(f"Cree morphological analyzer not loaded: {cree_fst_load_error}")
    print("Run: pip install hfst, and make sure "
          f"{CREE_FST_PATH} exists (download from "
          "github.com/UAlbertaALTLab/plains-cree-fsts/releases).")

_CREE_FLAG_RE = re.compile(r"@[^@]*@")

_CREE_ALPHABET = list("acehiklmnopstwyâêîô")


def _clean_analysis_tag(raw: str) -> str:
    return _CREE_FLAG_RE.sub("", raw)


def _extract_lemma_pos(cleaned: str):
    parts = cleaned.split("+")

    def is_marker(p):
        return p.startswith("PV/") or (p.isalpha() and p.isupper())

    lemma = next((p for p in parts if p and not is_marker(p)), parts[0])
    pos_match = re.search(r"\+(N|V|Ipc|Pron|Prop|Adv|Num|Interj)\b", cleaned)
    return lemma, (pos_match.group(1) if pos_match else None)


def _edits1(word: str) -> set:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in _CREE_ALPHABET]
    inserts = [L + c + R for L, R in splits for c in _CREE_ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def suggest_cree_word(word: str, deep: bool = False, max_suggestions: int = 5) -> list:
    if cree_analyzer_fst is None or not word:
        return []

    def _found(candidates):
        out = []
        for cand in candidates:
            results = cree_analyzer_fst.lookup(cand)
            if results:
                cleaned = _clean_analysis_tag(results[0][0])
                lemma, pos = _extract_lemma_pos(cleaned)
                out.append({"surface": cand, "lemma": lemma, "pos": pos})
        return out

    ed1 = _edits1(word)
    found = _found(ed1)

    if deep and len(found) < max_suggestions and len(word) <= 10:
        seen_surfaces = {f["surface"] for f in found}
        ed2 = set()
        for w in ed1:
            ed2 |= _edits1(w)
        ed2 -= ed1
        ed2 -= {word}
        found += [f for f in _found(ed2) if f["surface"] not in seen_surfaces]

    # Dedupe by lemma (many surface forms can share one lemma), preserve order
    seen_lemmas, deduped = set(), []
    for f in found:
        if f["lemma"] not in seen_lemmas:
            seen_lemmas.add(f["lemma"])
            deduped.append(f)
        if len(deduped) >= max_suggestions:
            break
    return deduped


class SentimentRequest(BaseModel):
    text: str


class CreeAnalyzeRequest(BaseModel):
    text: str
    deep_suggestions: bool = False


def analyze_cree_word(word: str, deep_suggestions: bool = False) -> dict:
    if cree_analyzer_fst is None:
        return {"word": word, "recognized": None, "error": cree_fst_load_error}

    results = cree_analyzer_fst.lookup(word)
    if not results:
        suggestions = suggest_cree_word(word, deep=deep_suggestions)
        return {"word": word, "recognized": False, "analyses": [], "suggestions": suggestions}

    analyses = []
    for raw, weight in results:
        cleaned = _clean_analysis_tag(raw)
        lemma, pos = _extract_lemma_pos(cleaned)
        is_variant = "Err/Orth" in cleaned  # non-normative spelling (e.g. macrons dropped)
        analyses.append({"lemma": lemma, "pos": pos, "tag": cleaned, "is_orthographic_variant": is_variant})

    return {"word": word, "recognized": True, "analyses": analyses}


@app.post("/api/cree/analyze")
def cree_analyze(req: CreeAnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        return {"words": []}
    tokens = re.findall(r"[^\s]+", text)
    words = []
    for tok in tokens:
        stripped = tok.strip(".,!?;:\"'()")
        if not stripped:
            continue
        words.append(analyze_cree_word(stripped, deep_suggestions=req.deep_suggestions))
    return {"words": words}


@app.get("/api/cree/health")
def cree_health():
    return {"status": "ok" if cree_analyzer_fst else "not_loaded", "error": cree_fst_load_error}


@app.post("/api/sentiment")
def sentiment(req: SentimentRequest):
    result = score_sentiment(req.text)
    if "error" in result:
        no_backend = roberta_sentiment is None and sentiment_analyzer is None
        return JSONResponse(result, status_code=503 if no_backend else 400)
    return result


@app.get("/api/sentiment/health")
def sentiment_health():
    return {
        "status": "ok" if (roberta_sentiment or sentiment_analyzer) else "not_loaded",
        "engine": "roberta" if roberta_sentiment else ("vader-fallback" if sentiment_analyzer else None),
        "roberta_error": roberta_load_error,
        "vader_error": sentiment_load_error,
    }


class VocalFeatures(BaseModel):
    pitch_range_semitones: float = 0.0
    rms_mean: float = 0.0
    rms_variance: float = 0.0


class AnalyzeRequest(BaseModel):
    text: str
    already_english: bool = False
    vocal: Optional[VocalFeatures] = None


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if req.already_english:
        english = text
        translation_note = None
    else:
        if translator is None:
            return JSONResponse({"error": f"Translation model not loaded: {translator_load_error}"}, status_code=503)
        try:
            english = translator.translate(text)
        except Exception as e:
            return JSONResponse({"error": f"Translation failed: {e}"}, status_code=500)
        if not english or not english.strip():
            return JSONResponse({
                "error": "empty_translation",
                "message": "The translation model returned an empty result for this phrase, "
                           "likely too far outside its small training vocabulary. Try one of "
                           "the example phrases.",
            }, status_code=200)
        translation_note = None

    vocal_dict = req.vocal.model_dump() if req.vocal else None
    result = score_sentiment(english, vocal=vocal_dict)
    if "error" in result:
        no_backend = roberta_sentiment is None and sentiment_analyzer is None
        return JSONResponse(result, status_code=503 if no_backend else 400)

    result["input"] = text
    result["translation"] = english
    result["was_translated"] = not req.already_english
    return result


@app.get("/protocol/status")
def protocol_status():
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    return harmony_engine.protocol.status()


@app.post("/protocol/toggle")
def protocol_toggle():
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    new_state = harmony_engine.protocol.toggle(source="manual_ui")
    return {"ok": True, "enabled": new_state}


@app.post("/protocol/enable")
def protocol_enable():
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.protocol.enable(source="manual_ui")
    return {"ok": True, "enabled": True}


@app.post("/protocol/disable")
def protocol_disable():
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.protocol.disable(source="manual_ui")
    return {"ok": True, "enabled": False}


class FusionModeRequest(BaseModel):
    enabled: bool


@app.post("/harmony/fusion-mode")
def set_fusion_mode(req: FusionModeRequest):
    """Explicit opt-in for triadic harmony (contemporary/fusion only — never the default)."""
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.set_fusion_mode(req.enabled)
    return {"ok": True, "fusion_mode": req.enabled}


class TextureRequest(BaseModel):
    texture: str  # "solo" | "duet" | "choir"


@app.post("/harmony/texture")
def set_texture(req: TextureRequest):
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.set_texture(req.texture)
    return {"ok": True, "texture": req.texture}


@app.post("/harmony/texture/clear")
def clear_texture():
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.clear_texture_override()
    return {"ok": True}


@app.get("/pipeline/status")
def pipeline_status():
    return {"running": pipeline_running, "audio_available": AUDIO_AVAILABLE}


@app.get("/neural/status")
def neural_status():
    if neural_timbre is None:
        return JSONResponse(
            {"loaded": False, "enabled": False, "error": neural_timbre_load_error},
            status_code=200,
        )
    status = {
        "loaded": True,
        "enabled": neural_timbre.enabled,
        "sidecar_url": neural_timbre.sidecar_url,
        "sidecar_reachable": neural_timbre._reachable,
        "voices_configured": neural_timbre.num_voices_configured,
    }
    if neural_timbre.enabled and neural_timbre._reachable:
        try:
            import requests as _requests
            resp = _requests.get(f"{neural_timbre.sidecar_url}/health", timeout=1.5)
            resp.raise_for_status()
            sidecar_health = resp.json()
            status["device_detail"] = {
                "torch_cuda_available": sidecar_health.get("torch_cuda_available"),
                "torch_version": sidecar_health.get("torch_version"),
                "voices": sidecar_health.get("voices", []),
            }
        except Exception:
            pass  # non-fatal -- the basic status above is still returned
    return status


class RenderNoteRequest(BaseModel):
    target_hz: float
    vocable: str = "aah"
    duration_s: float = 0.45
    mode: str = "unison_shadowing"
    texture: str = "solo"
    voice_index: int = 0
    transpose_semitones: float | None = None


def _build_render_decision(req: "RenderNoteRequest", cfg: dict):
    from synthesis.accompaniment_modes import AccompanimentMode

    try:
        mode = AccompanimentMode(req.mode)
    except ValueError:
        mode = AccompanimentMode(cfg["harmony"]["default_mode"])

    texture_cfg = cfg.get("synthesis", {}).get("texture", {}).get(req.texture, {})
    vocable = req.vocable if req.vocable in cfg["synthesis"]["vocable_set"] else "aah"

    return SimpleNamespace(
        action="sing",
        target_hz=float(req.target_hz),
        duration_s=max(0.05, float(req.duration_s)),
        mode=mode,
        vocable=vocable,
        vowel_color=0.5,
        brightness=0.5,
        nasality=0.0,
        num_voices=int(texture_cfg.get("num_voices", 1)),
        reverb_amount=float(texture_cfg.get("reverb_amount", 0.08)),
        detune_spread_cents=float(texture_cfg.get("detune_spread_cents", 10.0)),
        timing_jitter_ms=float(texture_cfg.get("timing_jitter_ms", 15.0)),
        formant_spread=float(texture_cfg.get("formant_spread", 0.1)),
    )


@app.post("/api/neural/render")
def render_neural_note(req: RenderNoteRequest):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready -- check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )

    try:
        from config.config_loader import get_config
        from synthesis.vocable_synthesizer import VocableSynthesizer

        cfg = get_config()
        decision = _build_render_decision(req, cfg)

        synth = VocableSynthesizer()
        scratch_audio = synth.synthesize(decision)

        sample_rate = cfg["audio"]["sample_rate"]
        final_audio = neural_timbre.convert(
            scratch_audio, sample_rate, decision.target_hz, voice_index=req.voice_index,
            transpose_semitones=req.transpose_semitones,
        )

        buf = io.BytesIO()
        sf.write(buf, final_audio.astype(np.float32), sample_rate, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as e:
        print(f"[neural render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


def _render_track_offline(raw_audio_bytes: bytes, pitch_method: str, texture: str, voice_index: int,
                           mode_override: str | None = None, instruments_enabled: bool = True,
                           transpose_semitones: float | None = None, apply_neural: bool = True) -> bytes:
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector
    from analysis.rhythm_analyzer import RhythmAnalyzer
    from analysis.phonetic_analysis import CreeTokenizer
    from synthesis.harmony_engine import HarmonyEngine
    from synthesis.vocable_synthesizer import VocableSynthesizer
    import librosa

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    percussive_spans: list = []
    if not instruments_enabled:
        try:
            window_s = min(2.0, max(0.5, len(audio) / sample_rate))
            percussive_spans = _detect_percussive_spans(audio, sample_rate, window_s=window_s, min_duration_s=0.6)
        except Exception as e:
            print(f"[neural render] percussive detection skipped: {e}")

    def _in_percussive_span(t: float) -> bool:
        return any(start <= t <= end for start, end in percussive_spans)

    preproc = Preprocessor()
    pitch = PitchDetector()
    pitch.set_method(pitch_method)
    rhythm = RhythmAnalyzer()
    cree = CreeTokenizer()
    harmony = HarmonyEngine()
    harmony.set_texture(texture)
    if mode_override:
        harmony.set_forced_mode(mode_override)
    synth = VocableSynthesizer()

    n_frames = len(audio) // frame_size
    frame_hop_s = frame_size / sample_rate
    robot = np.zeros(len(audio) + int(4.0 * sample_rate), dtype=np.float32)

    runs: list[dict] = []
    current_run: dict | None = None

    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        clean, is_voiced = preproc.process(frame)
        rhythm.push_frame(clean, is_voiced)

        archer_hz = None
        phoneme_profile = cree._neutral_profile
        if is_voiced:
            pitch_input = frame if pitch.method == "rmvpe" else clean
            hz, conf = pitch.detect(pitch_input)
            if hz and not (percussive_spans and _in_percussive_span(i * frame_size / sample_rate)):
                archer_hz = hz
            phoneme_profile = cree.analyze(clean)
        else:
            pitch.reset()

        phrase = rhythm.phrase_state
        if archer_hz and phrase in ("silence", "phrase_end"):
            phrase = "singing"

        decision = harmony.decide(
            archer_hz=archer_hz, phrase_state=phrase,
            tempo_bpm=rhythm.current_tempo, phoneme_profile=phoneme_profile,
        )

        if decision.action == "sing":
            current_run = {"start_frame": i, "decision": decision, "n_frames": 1}
            runs.append(current_run)
        elif decision.action == "sustain":
            if current_run is None:
                current_run = {"start_frame": i, "decision": decision, "n_frames": 1}
                runs.append(current_run)
            else:
                current_run["n_frames"] += 1
        else:
            current_run = None

    MAX_RUN_SECONDS = 12.0  # safety cap so one stuck drone/note can't blow up render time/memory
    for run in runs:
        decision = run["decision"]
        decision.duration_s = min(run["n_frames"] * frame_hop_s, MAX_RUN_SECONDS)
        scratch = synth.synthesize(decision)
        start_sample = run["start_frame"] * frame_size
        end_sample = start_sample + len(scratch)
        if end_sample > len(robot):
            robot = np.pad(robot, (0, end_sample - len(robot)))
        robot[start_sample:end_sample] += scratch

    peak = float(np.max(np.abs(robot))) if robot.size else 0.0
    if peak > 1.0:
        robot = robot / peak  # avoid clipping where overlapping/sustained notes summed above 0dBFS

    if not apply_neural:
        out_buf = io.BytesIO()
        sf.write(out_buf, robot.astype(np.float32), sample_rate, format="WAV")
        return out_buf.getvalue()

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(robot, sample_rate, voice_index=voice_index, timeout_s=timeout_s,
                                                transpose_semitones=transpose_semitones)
    if converted is None:
        raise RuntimeError(
            "Neural sidecar conversion failed or timed out — check that neural_env/rvc_server.py "
            "is running and see its terminal output / GET /neural/status."
        )

    out_buf = io.BytesIO()
    sf.write(out_buf, converted.astype(np.float32), sample_rate, format="WAV")
    return out_buf.getvalue()


@app.post("/api/dsp/render-track")
async def render_dsp_track(
    file: UploadFile = File(...),
    texture: str = Form("solo"),
    pitch_method: str = Form("yin"),
    mode: str = Form(""),
    instruments_enabled: bool = Form(True),
):
    try:
        raw = await file.read()
        wav_bytes = await asyncio.to_thread(
            _render_track_offline, raw, pitch_method, texture, 0, (mode or None),
            instruments_enabled, None, False,
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[dsp track render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


@app.post("/api/neural/render-track")
async def render_neural_track(
    file: UploadFile = File(...),
    texture: str = Form("solo"),
    pitch_method: str = Form("rmvpe"),
    voice_index: int = Form(0),
    mode: str = Form(""),
    instruments_enabled: bool = Form(True),
    transpose_semitones: float | None = Form(None),
):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready -- check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )

    try:
        raw = await file.read()
        wav_bytes = await asyncio.to_thread(
            _render_track_offline, raw, pitch_method, texture, voice_index, (mode or None), instruments_enabled,
            transpose_semitones
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[neural track render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


def _silence_instrumental_spans(audio: np.ndarray, sample_rate: int, log_prefix: str) -> np.ndarray:
    duration_s = len(audio) / sample_rate
    window_s = min(2.0, max(0.5, duration_s))
    min_duration_s = max(0.2, min(0.6, duration_s / 4))
    spans = _detect_percussive_spans(audio, sample_rate, window_s=window_s, min_duration_s=min_duration_s)
    if not spans:
        return audio

    vocal_segments = []
    if whisper_model_track is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, sample_rate, format="WAV")
                tmp_path = tmp.name
            try:
                raw_segments = _whisper_transcribe_raw_segments(
                    tmp_path, "en", whisper_model_track, whisper_backend_track
                )
                vocal_segments = [s for s in raw_segments if s["no_speech_prob"] < WHISPER_NO_SPEECH_VOCAL_PRESENT_THRESHOLD]
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"[{log_prefix}] vocal-presence check failed ({e}) -- falling back to "
                  "rhythmic-density-only silencing for this render.")
    else:
        print(f"[{log_prefix}] Whisper not loaded -- can't confirm vocal presence, falling back to "
              "rhythmic-density-only silencing (may over-silence chant-over-drum content).")

    def _has_vocal_overlap(start_s, end_s):
        return any(seg["start"] < end_s and seg["end"] > start_s for seg in vocal_segments)

    kept_spans = [(s, e) for s, e in spans if not _has_vocal_overlap(s, e)]
    skipped = len(spans) - len(kept_spans)
    if skipped:
        print(f"[{log_prefix}] kept {skipped} percussive span(s) that overlapped confident vocal "
              "segments -- not silencing real singing just because a drum is also present.")

    if not kept_spans:
        return audio

    fade_samples = int(0.02 * sample_rate)  # 20ms, avoids an audible click at the silencing boundary
    for start_s, end_s in kept_spans:
        start_i = max(0, int(start_s * sample_rate))
        end_i = min(len(audio), int(end_s * sample_rate))
        if end_i <= start_i:
            continue
        audio[start_i:end_i] = 0.0
        fo = min(fade_samples, start_i)
        if fo > 0:
            audio[start_i - fo:start_i] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        fi = min(fade_samples, len(audio) - end_i)
        if fi > 0:
            audio[end_i:end_i + fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    print(f"[{log_prefix}] instruments disabled: silenced {len(kept_spans)} confirmed-instrumental "
          f"span(s) totaling {sum(e - s for s, e in kept_spans):.1f}s before RVC conversion")
    return audio


def _convert_track_direct(raw_audio_bytes: bytes, voice_index: int, pad_seconds: float | None = None,
                           instruments_enabled: bool = True, transpose_semitones: float | None = None) -> bytes:
    from config.config_loader import get_config
    import librosa

    cfg = get_config()

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    CONVERSION_SAMPLE_RATE_FLOOR = 44100
    sample_rate = in_sr if in_sr >= CONVERSION_SAMPLE_RATE_FLOOR else CONVERSION_SAMPLE_RATE_FLOOR
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if not instruments_enabled:
        try:
            audio = _silence_instrumental_spans(audio, sample_rate, "convert-track direct")
        except Exception as e:
            print(f"[convert-track direct] percussive detection skipped: {e}")

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(
        audio, sample_rate, voice_index=voice_index, timeout_s=timeout_s, pad_seconds=pad_seconds,
        transpose_semitones=transpose_semitones,
    )
    if converted is None:
        raise RuntimeError(
            "Neural sidecar conversion failed or timed out — check that neural_env/rvc_server.py "
            "is running and see its terminal output / GET /neural/status."
        )

    out_buf = io.BytesIO()
    sf.write(out_buf, converted.astype(np.float32), sample_rate, format="WAV")
    return out_buf.getvalue()


@app.post("/api/neural/convert-track")
async def convert_neural_track(
    file: UploadFile = File(...),
    voice_index: int = Form(0),
    pad_seconds: float | None = Form(None),
    instruments_enabled: bool = Form(True),
    transpose_semitones: float | None = Form(None),
):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready -- check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )
    try:
        raw = await file.read()
        wav_bytes = await asyncio.to_thread(
            _convert_track_direct, raw, voice_index, pad_seconds, instruments_enabled, transpose_semitones
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[neural direct convert error] {e}")
        return JSONResponse({"error": f"Conversion failed: {e}"}, status_code=500)


def _decode_audio_via_ffmpeg(raw_audio_bytes: bytes, target_sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(raw_audio_bytes)
        tmp_in_path = tmp_in.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp_in_path,
             "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"],
            capture_output=True, check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to decode audio: {e.stderr.decode(errors='replace')[:300]}")
    finally:
        try:
            os.unlink(tmp_in_path)
        except OSError:
            pass


def _analyze_pitch_offline(raw_audio_bytes: bytes, pitch_method: str) -> dict:
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]
    frame_time_s = frame_size / sample_rate

    audio = _decode_audio_via_ffmpeg(raw_audio_bytes, sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    preproc = Preprocessor()
    pitch = PitchDetector()
    pitch.set_method(pitch_method)

    n_frames = len(audio) // frame_size
    hz_timeline: list = []
    conf_timeline: list = []

    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        clean, is_voiced = preproc.process(frame)

        hz = None
        conf = 0.0
        if is_voiced:
            pitch_input = frame if pitch.method == "rmvpe" else clean
            hz, conf = pitch.detect(pitch_input)
        else:
            pitch.reset()

        hz_timeline.append(round(hz, 2) if hz else None)
        conf_timeline.append(round(float(conf), 3) if hz else None)

    try:
        window_s = min(2.0, max(0.5, len(audio) / sample_rate))
        perc_spans = _detect_percussive_spans(audio, sample_rate, window_s=window_s, min_duration_s=0.6)
    except Exception as e:
        print(f"[pitch analyze] percussive detection skipped: {e}")
        perc_spans = []

    return {
        "frame_time_s": frame_time_s,
        "hz": hz_timeline,
        "confidence": conf_timeline,
        "percussive_spans": [{"start": s, "end": e} for s, e in perc_spans],
    }


@app.post("/api/pitch/analyze-track")
async def analyze_pitch_track(
    file: UploadFile = File(...),
    pitch_method: str = Form("rmvpe"),
):
    try:
        raw = await file.read()
        result = await asyncio.to_thread(_analyze_pitch_offline, raw, pitch_method)
        return result
    except Exception as e:
        print(f"[pitch analyze error] {e}")
        return JSONResponse({"error": f"Pitch analysis failed: {e}"}, status_code=500)


WHISPER_LOGPROB_THRESHOLD = -1.0
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_NO_SPEECH_THRESHOLD = 0.6

WHISPER_NO_SPEECH_VOCAL_PRESENT_THRESHOLD = 0.5


def _whisper_transcribe_raw_segments(tmp_path: str, language, model, backend) -> list:
    if backend == "faster":
        def _run(lang, want_words):
            segments_iter, _info = model.transcribe(
                tmp_path, language=lang, beam_size=5, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400), word_timestamps=want_words,
            )
            return list(segments_iter)

        try:
            segments = _run(language, True)
        except Exception as e:
            if language is None:
                print(f"[transcribe] auto-detect decode failed ({e}); retrying with language='en'")
                language = "en"
            try:
                segments = _run(language, True)
            except Exception as e2:
                print(f"[transcribe] word-timestamp alignment failed ({e2}); retrying without word timestamps")
                segments = _run(language, False)
        return [
            {"start": s.start, "end": s.end, "text": s.text.strip(),
             "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
             "compression_ratio": s.compression_ratio,
             "words": [{"word": w.word.strip(), "start": w.start, "end": w.end} for w in (s.words or [])]}
            for s in segments
        ]

    if backend == "openai":
        def _run(lang, want_words):
            return model.transcribe(tmp_path, language=lang, fp16=False, word_timestamps=want_words)

        try:
            result = _run(language, True)
        except Exception as e:
            if language is None:
                print(f"[transcribe] auto-detect decode failed ({e}); retrying with language='en'")
                language = "en"
            try:
                result = _run(language, True)
            except Exception as e2:
                print(f"[transcribe] word-timestamp alignment failed ({e2}); retrying without word timestamps")
                result = _run(language, False)
        return [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip(),
             "avg_logprob": s.get("avg_logprob", 0), "no_speech_prob": s.get("no_speech_prob", 0),
             "compression_ratio": s.get("compression_ratio", 0),
             "words": [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
                       for w in s.get("words", [])]}
            for s in result.get("segments", [])
        ]

    return []


def _classify_segments(raw_segments: list) -> list:
    annotated = []
    for s in raw_segments:
        likely_hallucinated = (
            s["no_speech_prob"] > WHISPER_NO_SPEECH_THRESHOLD
            or s["avg_logprob"] < WHISPER_LOGPROB_THRESHOLD
            or s["compression_ratio"] > WHISPER_COMPRESSION_RATIO_THRESHOLD
            or not s["text"]
        )
        seg = {
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "label": s["text"] if not likely_hallucinated else "[non-lexical vocals]",
            "confident": not likely_hallucinated,
        }
        if not likely_hallucinated and s.get("words"):
            seg["words"] = [{"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
                             for w in s["words"] if w["word"]]
        annotated.append(seg)
    return annotated


_PANNS_INSTRUMENT_CLASSES = {
    "Violin, fiddle", "Viola", "Cello", "Double bass",
    "Flute", "Clarinet", "Oboe", "Bassoon", "Saxophone",
    "Trumpet", "Trombone", "French horn", "Brass instrument",
    "Guitar", "Electric guitar", "Bass guitar", "Banjo", "Mandolin", "Ukulele",
    "Piano", "Electric piano", "Organ", "Harpsichord",
    "Drum", "Drum kit", "Bass drum", "Snare drum", "Hi-hat", "Cymbal",
    "Tambourine", "Rattle (instrument)", "Maraca", "Wood block",
    "Marimba, xylophone", "Glockenspiel", "Chime", "Bell",
    "Harp", "Accordion", "Bagpipes", "Didgeridoo", "Shofar",
    "Sitar", "Steel guitar, slide guitar",
}

panns_model = None
panns_labels = None
try:
    from panns_inference import AudioTagging, labels as _panns_labels_list
    panns_model = AudioTagging(checkpoint_path=None, device="cpu")
    panns_labels = _panns_labels_list
    print("PANNs audio-tagging model loaded (per-instrument detection enabled).")
except Exception as e:
    print(f"PANNs not available ({e}) -- instrument spans will use the generic rhythm-only "
          f"fallback ('[instrumental / percussion]' instead of a named instrument). Run "
          f"'pip install panns-inference torchlibrosa' and restart the server to enable real "
          f"per-instrument labels (violin, flute, drum, guitar, ...).")
    panns_model, panns_labels = None, None


def _detect_instrument_spans(raw_bytes: bytes, min_duration_s: float) -> list:
    model, labels = panns_model, panns_labels

    if model is None:
        fallback_sr = 22050
        fb_audio = _decode_audio_via_ffmpeg(raw_bytes, fallback_sr)
        duration = len(fb_audio) / fallback_sr
        window_s = min(2.0, max(0.5, duration))
        spans = _detect_percussive_spans(fb_audio, fallback_sr, window_s=window_s, min_duration_s=min_duration_s)
        return [{"start": s, "end": e, "label": "[instrumental / percussion]", "confident": False} for s, e in spans]

    panns_sr = 32000  # PANNs' expected input sample rate
    audio = _decode_audio_via_ffmpeg(raw_bytes, panns_sr)
    duration = len(audio) / panns_sr

    win_samples = int(2.0 * panns_sr)
    hop_samples = int(1.0 * panns_sr)
    raw_spans = []
    i = 0
    while i < len(audio):
        chunk = audio[i:i + win_samples]
        if len(chunk) < panns_sr * 0.5:  # too short a tail to classify meaningfully
            break
        clipwise_output, _ = model.inference(chunk[None, :])
        top_idx = np.argsort(clipwise_output[0])[::-1][:5]
        for idx in top_idx:
            label = labels[idx]
            score = float(clipwise_output[0][idx])
            if label in _PANNS_INSTRUMENT_CLASSES and score > 0.15:
                start_t = i / panns_sr
                end_t = min((i + win_samples) / panns_sr, duration)
                raw_spans.append({"start": round(start_t, 2), "end": round(end_t, 2),
                                   "label": f"[{label.lower()}]", "confident": False})
        i += hop_samples

    raw_spans.sort(key=lambda s: (s["label"], s["start"]))
    merged = []
    for s in raw_spans:
        if merged and merged[-1]["label"] == s["label"] and s["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= min_duration_s]


def _add_percussive_segments(annotated: list, raw_bytes: bytes, min_duration_s: float) -> list:
    try:
        annotated.extend(_detect_instrument_spans(raw_bytes, min_duration_s))
    except Exception as e:
        print(f"[transcribe] instrument detection skipped: {e}")
    annotated.sort(key=lambda seg: seg["start"])
    return annotated


def _transcribe_audio_blob(raw: bytes, language: str = "en") -> dict:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        if os.path.getsize(tmp_path) < 512:
            return {"text": "", "bytes_received": len(raw), "segment_count": 0, "no_speech_probs": [],
                    "segments": [], "has_lexical_speech": False}

        raw_segments = _whisper_transcribe_raw_segments(tmp_path, language, whisper_model_mic, whisper_backend_mic)
        annotated = _classify_segments(raw_segments)
        annotated = _add_percussive_segments(annotated, raw, min_duration_s=0.6)

        confident_text = " ".join(s["label"] for s in annotated if s["confident"] and s["label"])
        no_speech_probs = [round(s["no_speech_prob"], 3) for s in raw_segments]
        return {
            "text": confident_text,
            "bytes_received": len(raw),
            "segment_count": len(raw_segments),
            "no_speech_probs": no_speech_probs,
            "segments": annotated,
            "has_lexical_speech": bool(confident_text.strip()),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _detect_percussive_spans(audio: np.ndarray, sr: int, window_s: float = 2.0,
                              density_threshold: float = 1.5, min_duration_s: float = 2.0) -> list:
    import librosa
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    peak_idx = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=5)
    peak_times = times[peak_idx]

    if not len(peak_times) or not len(times):
        return []

    duration = float(times[-1])
    spans = []
    cur_start = None
    t = 0.0
    while t < duration:
        count = np.sum((peak_times >= t) & (peak_times < t + window_s))
        density = count / window_s
        is_pulsing = density >= density_threshold
        if is_pulsing and cur_start is None:
            cur_start = t
        elif not is_pulsing and cur_start is not None:
            if t - cur_start >= min_duration_s:
                spans.append((round(cur_start, 2), round(t, 2)))
            cur_start = None
        t += window_s
    if cur_start is not None and duration - cur_start >= min_duration_s:
        spans.append((round(cur_start, 2), round(duration, 2)))
    return spans


def _transcribe_track_annotated(raw: bytes, language: str = "en") -> dict:
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        raw_segments = _whisper_transcribe_raw_segments(tmp_path, language, whisper_model_track, whisper_backend_track)
        annotated = _classify_segments(raw_segments)
        annotated = _add_percussive_segments(annotated, raw, min_duration_s=2.0)

        confident_text = " ".join(s["label"] for s in annotated if s["confident"] and s["label"])
        return {
            "segments": annotated,
            "text": confident_text,
            "has_lexical_speech": bool(confident_text.strip()),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("en")):
    if whisper_model_mic is None:
        return JSONResponse({"error": f"Whisper (mic model) not loaded: {whisper_load_error_mic}"}, status_code=503)
    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "No audio data received"}, status_code=400)
        whisper_language = None if language == "cr" else "en"
        diag = await asyncio.to_thread(_transcribe_audio_blob, raw, whisper_language)
        if not diag["text"]:
            print(f"[transcribe] empty result -- {diag['bytes_received']} bytes received, "
                  f"{diag['segment_count']} segments, no_speech_probs={diag['no_speech_probs']}")
        return diag
    except Exception as e:
        print(f"[transcribe error] {e}")
        return JSONResponse({"error": f"Transcription failed: {e}"}, status_code=500)


@app.post("/api/transcribe/annotated")
async def transcribe_annotated(file: UploadFile = File(...), language: str = Form("en")):
    if whisper_model_track is None:
        return JSONResponse({"error": f"Whisper (track model) not loaded: {whisper_load_error_track}"}, status_code=503)
    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "No audio data received"}, status_code=400)
        result = await asyncio.to_thread(_transcribe_track_annotated, raw, language)
        return result
    except Exception as e:
        print(f"[transcribe annotated error] {e}")
        return JSONResponse({"error": f"Annotated transcription failed: {e}"}, status_code=500)


@app.get("/api/transcribe/health")
def transcribe_health():
    return {
        "mic_model": {"status": "ok" if whisper_model_mic else "not_loaded",
                       "backend": whisper_backend_mic, "error": whisper_load_error_mic},
        "track_model": {"status": "ok" if whisper_model_track else "not_loaded",
                         "backend": whisper_backend_track, "error": whisper_load_error_track},
    }


@app.post("/pipeline/start")
def pipeline_start(body: dict):
    global pipeline_running, pipeline_stop_event
    if not AUDIO_AVAILABLE:
        return JSONResponse({"ok": False, "error": "Local audio not available on this server. Use the browser mic."})
    if pipeline_running:
        return JSONResponse({"ok": False, "error": "already running"})
    input_device = body.get("input_device")
    if input_device is None:
        return JSONResponse({"ok": False, "error": "input_device required"})
    pipeline_stop_event.clear()
    t = threading.Thread(target=run_local_pipeline,
                         args=(pipeline_stop_event, int(input_device)), daemon=True)
    t.start()
    pipeline_running = True
    return {"ok": True}


@app.post("/pipeline/stop")
def pipeline_stop():
    global pipeline_running
    pipeline_stop_event.set()
    pipeline_running = False
    return {"ok": True}


@app.websocket("/ws")
async def ws_broadcast(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                msg = broadcast_queue.get_nowait()
                await ws.send_text(json.dumps(msg))
            except _queue.Empty:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/mic")
async def ws_mic(ws: WebSocket):
    await ws.accept()
    if harmony_engine is None:
        await ws.send_text(json.dumps({"type": "error", "message": f"HarmonyEngine not loaded: {harmony_load_error}"}))
        await ws.close()
        return
    try:
        from config.config_loader import get_config
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        from analysis.rhythm_analyzer import RhythmAnalyzer
        from analysis.phonetic_analysis import CreeTokenizer
        import librosa as _lib

        cfg     = get_config()
        preproc = Preprocessor()
        pitch   = PitchDetector()
        rhythm  = RhythmAnalyzer()
        cree    = CreeTokenizer()
        harmony = harmony_engine   # shared instance, sovereignty state persists across sessions
        start       = time.perf_counter()

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") == "set_pitch_method":
                    active = await asyncio.to_thread(pitch.set_method, ctrl.get("method", ""))
                    await ws.send_text(json.dumps({
                        "type": "pitch_method_changed",
                        "method": active,
                        "requested": ctrl.get("method"),
                    }))
                continue

            data = message.get("bytes")
            if not data:
                continue
            frame = np.frombuffer(data, dtype=np.float32).copy()
            if len(frame) == 0:
                continue

            clean_frame, is_voiced = preproc.process(frame)
            rhythm.push_frame(clean_frame, is_voiced)

            archer_hz       = None
            phoneme_profile = cree._neutral_profile

            if is_voiced:
                pitch_input = frame if pitch.method == "rmvpe" else clean_frame
                hz, conf = await asyncio.to_thread(pitch.detect, pitch_input)
                if hz:
                    archer_hz = hz
                phoneme_profile = cree.analyze(clean_frame)
            else:
                pitch.reset()

            phrase = rhythm.phrase_state
            if archer_hz and phrase in ("silence", "phrase_end"):
                phrase = "singing"

            elapsed_s = time.perf_counter() - start

            harmony.protocol.check_sound_cue(archer_hz, is_voiced, elapsed_s)

            decision = harmony.decide(
                archer_hz=archer_hz, phrase_state=phrase,
                tempo_bpm=rhythm.current_tempo, phoneme_profile=phoneme_profile,
            )

            msg = {
                "type":            "pitch" if archer_hz else "silence",
                "singer_hz":       round(archer_hz, 1) if archer_hz else None,
                "singer_note":     _lib.hz_to_note(archer_hz) if archer_hz else None,
                "robot_hz":        round(decision.target_hz, 1) if decision.target_hz else None,
                "robot_note":      _lib.hz_to_note(decision.target_hz) if decision.target_hz and decision.target_hz > 0 else None,
                "action":          decision.action,
                "mode":            decision.mode.value,
                "mode_note":       decision.mode_note,
                "texture":         decision.texture,
                "num_voices":      decision.num_voices,
                "protocol_enabled": harmony.protocol.enabled,
                "tempo_bpm":       round(rhythm.current_tempo, 1),
                "phrase_state":    phrase,
                "elapsed_s":       round(elapsed_s, 1),
                "pitch_method":    pitch.method,
            }
            await ws.send_text(json.dumps(msg))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/mic error] {e}")


@app.websocket("/ws/pitch")
async def ws_pitch(ws: WebSocket):
    await ws.accept()
    try:
        from config.config_loader import get_config
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        import librosa as _lib

        cfg     = get_config()
        preproc = Preprocessor()
        pitch   = PitchDetector()

        requested_sample_rate = ws.query_params.get("sample_rate")
        if requested_sample_rate:
            try:
                pitch.set_source_sample_rate(int(float(requested_sample_rate)))
            except ValueError:
                pass

        requested_method = ws.query_params.get("method")
        if requested_method:
            await asyncio.to_thread(pitch.set_method, requested_method)

        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") == "set_pitch_method":
                    active = await asyncio.to_thread(pitch.set_method, ctrl.get("method", ""))
                    await ws.send_text(json.dumps({
                        "type": "pitch_method_changed",
                        "method": active,
                        "requested": ctrl.get("method"),
                    }))
                continue

            data = message.get("bytes")
            if not data:
                continue
            frame = np.frombuffer(data, dtype=np.float32).copy()
            if len(frame) == 0:
                continue

            clean_frame, is_voiced = preproc.process(frame)

            if is_voiced:
                pitch_input = frame if pitch.method == "rmvpe" else clean_frame
                # Run detect() off the event loop. For RMVPE this is a
                # CNN+BiGRU inference call over up to a second of resampled
                # audio and can take longer than one frame's real-time
                # budget on CPU -- if it runs inline here it blocks *every*
                # open connection on this single-process event loop for
                # that whole time, which is how the backlog snowballs and
                # the server keeps reporting pitch long after playback and
                # the browser's onended have already fired. Backgrounding
                # it doesn't make the inference faster, but it stops one
                # slow call from stalling everything else while it runs.
                hz, conf = await asyncio.to_thread(pitch.detect, pitch_input)
                # NOTE: pitch.detect() already applies the *correct*
                # confidence gate internally -- rmvpe.confidence_threshold
                # (0.15) for RMVPE, pitch.confidence_threshold (0.3) for
                # YIN/YINFFT -- and already returns hz=None for anything
                # below that gate. Previously this re-checked `conf` here
                # against cfg["pitch"]["confidence_threshold"] (the YIN
                # value, 0.3) regardless of which method was active, which
                # silently discarded every RMVPE reading with confidence
                # between 0.15 and 0.3 -- a large fraction of real notes.
                # Trust hz; don't re-gate it with the wrong threshold.
                if hz:
                    await ws.send_text(json.dumps({
                        "type":       "pitch",
                        "hz":         round(hz, 1),
                        "note":       _lib.hz_to_note(hz),
                        "confidence": round(conf, 3),
                        "method":     pitch.method,
                    }))
                    continue
            else:
                pitch.reset()

            await ws.send_text(json.dumps({"type": "silence", "method": pitch.method}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/pitch error] {e}")


@app.websocket("/ws/live-transcribe")
async def ws_live_transcribe(ws: WebSocket):
    """
    Real-time, fully local streaming transcription via Vosk -- purely
    for instant visual feedback while the mic pipeline's normal VAD is
    still waiting for a pause. This does NOT replace /api/transcribe:
    the accurate, hallucination-filtered, sentiment-integrated
    transcript still comes from the Whisper-based pipeline once a
    phrase ends. This endpoint just fills the gap while you're still
    talking, at whatever accuracy a small streaming model can manage --
    noticeably rougher than Whisper, in exchange for latency in the
    low hundreds of milliseconds instead of "however long the phrase
    plus a full Whisper pass takes".

    Same wire convention as /ws/pitch: raw float32 PCM bytes at the
    client's native AudioContext sample rate (passed as the
    `sample_rate` query param), mono. Resampling to the 16kHz int16 PCM
    Vosk actually expects happens here, not on the client, so the
    frontend doesn't need its own resampling code beyond what it
    already does to capture raw samples.

    Vosk's recognizer calls (AcceptWaveform/Result/PartialResult) are
    blocking C++ calls via its Python bindings, not async-native --
    they're dispatched through asyncio.to_thread() inside the loop
    below rather than called directly. An earlier version called them
    directly on the event loop; since a chunk arrives every ~100-250ms
    continuously while the mic is running, that monopolized the entire
    single-threaded event loop on every single chunk, which starved
    every other in-flight coroutine on the same loop -- including the
    already-computed HTTP response for a /api/transcribe request
    waiting for its turn to actually get sent. The accurate transcript
    wasn't stuck computing in that case; it was fully done and just
    couldn't get scheduled to deliver its response until this loop
    stopped sending (i.e., until the mic was stopped) -- which is
    exactly the "everything appears at once, only after clicking Stop"
    bug this fixes.
    """
    await ws.accept()
    if vosk_model is None:
        await ws.send_text(json.dumps({"type": "error", "message": vosk_load_error or "Vosk not loaded"}))
        await ws.close()
        return

    import vosk as _vosk
    recognizer = _vosk.KaldiRecognizer(vosk_model, 16000)
    recognizer.SetWords(False)

    try:
        source_sr = int(float(ws.query_params.get("sample_rate", "48000")))
    except ValueError:
        source_sr = 48000

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if not data:
                continue

            frame = np.frombuffer(data, dtype=np.float32)
            if len(frame) == 0:
                continue

            # Downsample to Vosk's required 16kHz and convert to int16
            # PCM. Linear interpolation, same lightweight approach used
            # elsewhere in this file for browser-rate audio -- this
            # doesn't need to be broadcast-quality, just intelligible
            # enough for a small streaming acoustic model.
            if source_sr != 16000:
                ratio = source_sr / 16000
                out_len = max(1, int(len(frame) / ratio))
                src_idx = np.arange(out_len) * ratio
                idx_low = np.floor(src_idx).astype(np.int64)
                idx_high = np.minimum(idx_low + 1, len(frame) - 1)
                frac = src_idx - idx_low
                resampled = frame[idx_low] * (1 - frac) + frame[idx_high] * frac
            else:
                resampled = frame

            pcm16 = np.clip(resampled, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)

            def _vosk_process_chunk(pcm_bytes: bytes):
                # Runs in a worker thread. See the docstring above for
                # why this can't run directly on the event loop.
                if recognizer.AcceptWaveform(pcm_bytes):
                    return "final", json.loads(recognizer.Result()).get("text", "")
                return "partial", json.loads(recognizer.PartialResult()).get("partial", "")

            kind, text = await asyncio.to_thread(_vosk_process_chunk, pcm16.tobytes())
            if text:
                await ws.send_text(json.dumps({"type": kind, "text": text}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/live-transcribe error] {e}")


def _apply_neural_timbre(audio: np.ndarray, sample_rate: int, decision) -> np.ndarray:
    """
    Runs the neural voice-conversion stage on VocableSynthesizer's output,
    if it's loaded and enabled. Fails safe to the original DSP audio for
    any reason (disabled, not loaded, model missing, inference error).

    Scope note: VocableSynthesizer._render_ensemble() already mixes a
    choir's N DSP voice layers into one buffer before returning it, so
    this currently reskins the whole mixed ensemble through a single
    trained voice (voice_index=0) rather than giving each of the N layers
    its own distinct trained timbre. A true multi-timbre neural choir
    needs VocableSynthesizer to expose per-voice stems before mixing, so
    each stem can go through a different NeuralTimbreConverter voice_index
    before being summed — a real follow-up, not implemented here. This is
    the correct, honest v1: one real trained voice, reskinning the DSP
    ensemble's existing detune/timing/formant variation.
    """
    if neural_timbre is None or not neural_timbre.enabled:
        return audio
    return neural_timbre.convert(audio, sample_rate, decision.target_hz, voice_index=0)


def run_local_pipeline(stop_event, input_device: int):
    global pipeline_running
    try:
        from config.config_loader import get_config
        from core.audio_capture import AudioCapture
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        from analysis.rhythm_analyzer import RhythmAnalyzer
        from analysis.phonetic_analysis import CreeTokenizer
        from synthesis.vocable_synthesizer import VocableSynthesizer
        from output.timing_sync import TimingSync
        import librosa as _lib

        cfg          = get_config()
        cfg["audio"]["input_device"] = input_device
        capture      = AudioCapture()
        preprocessor = Preprocessor()
        pitch        = PitchDetector()
        rhythm       = RhythmAnalyzer()
        cree         = CreeTokenizer()
        harmony      = harmony_engine   # shared instance — same sovereignty state as /ws/mic
        synth        = VocableSynthesizer()
        timing       = TimingSync()

        timing.start()
        capture.start()
        currently_singing = False
        start = time.perf_counter()

        while not stop_event.is_set():
            try:
                frame = capture.queue.get(timeout=0.1)
            except Exception:
                continue

            clean, is_voiced = preprocessor.process(frame)
            rhythm.push_frame(clean, is_voiced)

            archer_hz = None
            if is_voiced:
                pitch_input = frame if pitch.method == "rmvpe" else clean
                # run_local_pipeline already runs in its own background
                # thread (not the asyncio event loop), so no to_thread
                # needed here -- just the same gating fix as /ws/pitch and
                # /ws/mic: pitch.detect() already applies the correct
                # method-specific confidence threshold internally.
                hz, conf = pitch.detect(pitch_input)
                if hz:
                    archer_hz = hz
                cree.analyze(clean)
            else:
                pitch.reset()

            timing.update_tempo(rhythm.current_tempo)
            phrase = rhythm.phrase_state
            if archer_hz and phrase in ("silence", "phrase_end"):
                phrase = "singing"

            harmony.protocol.check_sound_cue(archer_hz, is_voiced, time.perf_counter() - start)

            decision = harmony.decide(
                archer_hz=archer_hz, phrase_state=phrase,
                tempo_bpm=rhythm.current_tempo,
                phoneme_profile=cree._neutral_profile,
            )

            if decision.action == "sing":
                scratch_audio = synth.synthesize(decision)
                final_audio = _apply_neural_timbre(scratch_audio, cfg["audio"]["sample_rate"], decision)
                timing.schedule(final_audio, decision.action)
                currently_singing = True
            elif decision.action == "sustain":
                currently_singing = True
            else:
                if currently_singing:
                    timing.flush()
                currently_singing = False

            if archer_hz:
                msg = {
                    "type":         "pitch",
                    "singer_hz":    round(archer_hz, 1),
                    "singer_note":  _lib.hz_to_note(archer_hz),
                    "robot_hz":     round(decision.target_hz, 1) if decision.target_hz else None,
                    "robot_note":   _lib.hz_to_note(decision.target_hz) if decision.target_hz and decision.target_hz > 0 else None,
                    "action":       decision.action,
                    "mode":         decision.mode.value,
                    "mode_note":    decision.mode_note,
                    "texture":      decision.texture,
                    "num_voices":   decision.num_voices,
                    "protocol_enabled": harmony.protocol.enabled,
                    "tempo_bpm":    round(rhythm.current_tempo, 1),
                    "phrase_state": phrase,
                    "elapsed_s":    round(time.perf_counter() - start, 1),
                    "pitch_method": pitch.method,
                }
                try:
                    broadcast_queue.put_nowait(msg)
                except _queue.Full:
                    pass
    except Exception as e:
        print(f"[pipeline error] {e}")
    finally:
        pipeline_running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-port", default=8000, type=int)
    args = parser.parse_args()
    port = int(os.environ.get("PORT", args.port))
    print(f"\n  Archer-Robot server starting")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"  Local audio hardware: {'available' if AUDIO_AVAILABLE else 'not available (cloud mode)'}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
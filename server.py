import asyncio
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from typing import Optional
import threading
import time
import queue as _queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import uvicorn
from scipy.signal import butter, lfilter
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.middleware("http")
async def fresh_frontend(request: Request, call_next):
    """Never hand back a stale build.

    Every asset in the page is asked for under a `?v=` number, and the page
    itself is what carries those numbers, so a page the browser cached this
    morning hands back this morning's numbers and this morning's script, and a
    fix written at noon is still not on screen at six. No page of the site, and
    nothing under /static, is served without asking the server first. Both send
    a strong ETag, so asking again is one 304 and not a download.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static/") or \
            response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


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
                print("[startup] RMVPE pre-warm skipped (see error above, "
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
              "try re-running download_sentiment_model.py, it may be an incomplete download.")

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


def _whisper_compute_device() -> tuple:
    """Pick the fastest device actually available instead of hardcoding CPU. int8-on-CPU was a
    reasonable default when it might be the only option, but if this is ever deployed on a box
    with an NVIDIA GPU, float16-on-CUDA is a large multiple faster for the same large-v3-turbo
    weights, and this project already treats CUDA-if-available as the standard pattern
    elsewhere (see DEVICE in translate_transformer.py). Falls back to CPU/int8 quietly if
    torch isn't installed or CUDA init fails for any reason."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception as e:
        print(f"[transcribe] CUDA check failed, staying on CPU ({e})")
    return "cpu", "int8"


def _load_faster_whisper_backend(model_size: str, model_dir: str, legacy_checkpoint: str,
                                  num_workers: int = 1, max_cpu_fraction: float = 1.0):
    if os.path.isdir(model_dir):
        try:
            from faster_whisper import WhisperModel
            device, compute_type = _whisper_compute_device()
            # cpu_threads/num_workers only matter for CPU decoding. A single-worker model (the
            # default) already uses every available core for ONE call, so when
            # _transcribe_track_annotated fires off several forced-clip re-decodes at once via
            # Python threads (see _force_transcribe_clip), they don't get real parallelism
            # they just fight over the same cores. Measured directly: 3 concurrent forced
            # clips took 58.4s, close to what 3 sequential calls would cost, not the
            # significant overlap concurrency should have bought. num_workers>1 makes
            # ctranslate2 spin up genuinely separate internal worker pools that can run truly
            # side by side; splitting cpu_threads across them keeps total CPU usage roughly
            # bounded, instead of N calls each trying to grab every core. Only the track model
            # needs multiple workers, the primary pass is one call regardless, but the
            # forced-clip re-decodes are where several calls happen at once.
            #
            # max_cpu_fraction caps the TOTAL thread budget below 100% of the machine's cores.
            # Letting transcription's worker pool claim every core starves whatever else is
            # running concurrently on the same box, specifically the neural voice-conversion
            # pipeline, which this project runs on its own background thread at the same time a
            # track is being transcribed. Both are supposed to make progress together, not have
            # one starve the other of CPU; leaving real headroom is what makes "at the same
            # time" actually mean something instead of transcription silently eating all the
            # compute the render pass needed.
            extra_kwargs = {}
            if num_workers > 1:
                cpu_count = os.cpu_count() or 4
                budget = max(num_workers, int(cpu_count * max_cpu_fraction))
                extra_kwargs["num_workers"] = num_workers
                extra_kwargs["cpu_threads"] = max(1, budget // num_workers)
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                      download_root=model_dir, local_files_only=True, **extra_kwargs)
            except Exception as e:
                if device == "cuda":
                    print(f"faster-whisper ({model_size}) failed to load on CUDA ({e}); falling back to CPU/int8.")
                    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                          download_root=model_dir, local_files_only=True, **extra_kwargs)
                    device, compute_type = "cpu", "int8"
                else:
                    raise
            print(f"faster-whisper ({model_size}, {compute_type}, {device.upper()}, "
                  f"num_workers={num_workers}) loaded from local files at {model_dir}.")
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
    # num_workers matches the max concurrency _transcribe_track_annotated actually uses for
    # forced-clip re-decodes (see the ThreadPoolExecutor(max_workers=min(4, len(merged_chunks)))
    # there), no point provisioning more worker pools than will ever be used concurrently.
    #
    # max_cpu_fraction=0.6 was tried here to leave headroom for the neural voice-conversion
    # pipeline to run alongside transcription. Reverted: measured directly, it took the primary
    # Whisper pass from 19.2s to 62.0s on the exact same reference track (a 3x regression) with
    # no confirmed benefit to synthesis speed to show for it, the synthesis chunk in that same
    # run still took 94 seconds, i.e. still very slow regardless. A machine that's genuinely CPU-
    # constrained enough for both heavy workloads to contend this badly doesn't have this much
    # spare capacity to redistribute either way; deliberately handicapping transcription's own
    # budget just cost it speed without visibly buying the other side anything.
    TRACK_WHISPER_MODEL_SIZE, TRACK_WHISPER_MODEL_DIR, _legacy_whisper_checkpoint,
    num_workers=min(4, os.cpu_count() or 1)
)
if whisper_model_track is None and whisper_model_mic is not None:
    print("Track-upload model not available; falling back to the mic model for /api/transcribe/annotated too "
          "(less accurate than large-v3-turbo would be, run download_faster_whisper_model.py to fix this).")
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
        "one-time only) to enable instant local live captions. Optional, everything else in this "
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

synthesizer = None
synthesizer_load_error = None
try:
    from synthesis.vocable_synthesizer import VocableSynthesizer
    synthesizer = VocableSynthesizer()
    if harmony_engine is not None:
        harmony_engine.set_synthesizer(synthesizer)
        print("Shared VocableSynthesizer initialized and wired to HarmonyEngine.")
    else:
        print("VocableSynthesizer created but HarmonyEngine not loaded yet.")
except Exception as e:
    synthesizer_load_error = str(e)
    print(f"VocableSynthesizer not loaded: {synthesizer_load_error}")

neural_timbre = None
neural_timbre_load_error = None
try:
    from synthesis.neural_timbre import NeuralTimbreConverter
    neural_timbre = NeuralTimbreConverter()
    if neural_timbre.enabled:
        print("NeuralTimbreConverter loaded and enabled.")
    else:
        print("NeuralTimbreConverter loaded but disabled/inactive "
              "(see synthesis.neural in config.yaml), using pure DSP voice output.")
except Exception as e:
    neural_timbre_load_error = str(e)
    print(f"NeuralTimbreConverter not loaded: {neural_timbre_load_error}")
    print("This is non-fatal, the pipeline runs on pure DSP synthesis without it.")


try:
    from chat.chat_api import router as chat_router
    app.include_router(chat_router)
    print("Chat router mounted at /api/chat (backend readiness checked lazily on first call).")
except Exception as e:
    print(f"Chat router not mounted: {e}")
    print("This is non-fatal, the rest of the app runs without the chatbot.")


# ---------------------------------------------------------------------------
# SingSong: a separate generative accompaniment system.
#
# This is deliberately lazy.  Importing torch and constructing three
# transformers at `python server.py` startup would add seconds to every run of
# the app even when nobody opens the SingSong page, and on a machine with no
# checkpoints there is nothing to construct anyway.  The engine is created on
# the first status/render request; status never loads weights, render does.
# ---------------------------------------------------------------------------
_sing_song_engine = None
_sing_song_engine_error = None


def _get_sing_song_engine():
    global _sing_song_engine, _sing_song_engine_error
    if _sing_song_engine is not None:
        return _sing_song_engine
    try:
        from sing_song import SingSongEngine
        from sing_song.config import load_sing_song_config

        cfg = load_sing_song_config()
        _sing_song_engine = SingSongEngine(cfg)
        _sing_song_engine_error = None
        return _sing_song_engine
    except Exception as e:  # noqa: BLE001 - surfaced by /singsong/status
        _sing_song_engine_error = str(e)
        raise


def _wav_bytes(audio, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _load_take(raw: bytes, sample_rate: int):
    import librosa

    take, _ = librosa.load(io.BytesIO(raw), sr=sample_rate, mono=True)
    return np.asarray(take, dtype=np.float32)


#: The band the built-in engine plays under a SingSong take.  Round dance is
#: the fullest of the six bands this project is for -- hand drum, rattles,
#: boom-chuck guitar and bass -- which is what "generate a band" should hand
#: back before the trained cascade is installed.
SINGSONG_BUILTIN_BAND = "round_dance"


#: One upload is rendered once into both WAVs the page uses -- the band it
#: plays under the take and the mix it saves -- because the expensive half is
#: the model pass, not the encoding.
_singsong_render_cache: dict[str, dict] = {}
_SINGSONG_RENDER_CACHE_LIMIT = 4


def _singsong_render_both(raw: bytes, band: str) -> dict[str, object]:
    """Render the band and the mix for one upload, with whichever engine can.

    The trained semantic -> coarse -> fine cascade answers whenever its four
    artifacts exist.  A fresh clone has no weights, so the same function falls
    back to the symbolic band engine (the one behind MySong) rather than
    refusing: the output is a complete band, not a placeholder, and the only
    thing that changes when the weights arrive is which model chose the notes.
    Both engines place the band against the take's own level, so the panel's
    voice is on top either way.
    """
    from mysong.accompaniment import (
        band_placement_gain_db,
        level_band_stem,
        mix_with_vocal,
    )

    engine = None
    try:
        candidate = _get_sing_song_engine()
        # A constructed engine is not a loaded one: with no weights `load()`
        # raises, and the variable must stay None so the fallback below runs.
        candidate.load()
        engine = candidate
    except Exception as e:  # noqa: BLE001 - the built-in band engine takes over
        print(f"SingSong: trained cascade unavailable ({e}); using the built-in band engine")

    if engine is None:
        from mysong.pipeline import create_song

        model = _get_mysong_model()
        with tempfile.TemporaryDirectory(prefix="singsong-") as tmp:
            path = Path(tmp) / "take.wav"
            path.write_bytes(raw)
            project = create_song(
                str(path),
                model=model,
                arrangement_style=band or SINGSONG_BUILTIN_BAND,
                sample_rate=44100,
                render=True,
            )
        sr = project.sample_rate
        headers = _mysong_readouts(project)
        headers.update({
            "X-SingSong-Engine": "built-in",
            "X-SingSong-Device": "symbolic band engine",
            "X-SingSong-Duration": f"{len(project.mix) / max(sr, 1):.1f}",
            "X-SingSong-Windows": "1",
            "X-SingSong-Band-Gain-Db": f"{float(project.band_placement_db):.2f}",
        })
        return {
            "mix": _wav_bytes(project.mix, sr),
            "band": _wav_bytes(project.band_stem, sr),
            "headers": headers,
        }

    instrumental_wav, result = engine.render_bytes(raw, return_mix=False)
    instrumental, sample_rate = sf.read(
        io.BytesIO(instrumental_wav), dtype="float32", always_2d=True
    )
    if instrumental.shape[1] > 1:
        instrumental = instrumental.mean(axis=1)
    take = _load_take(raw, sample_rate)
    return {
        "mix": _wav_bytes(
            mix_with_vocal(instrumental, take, sample_rate=sample_rate), sample_rate
        ),
        "band": _wav_bytes(level_band_stem(instrumental), sample_rate),
        "headers": {
            "X-SingSong-Engine": "neural",
            "X-SingSong-Device": str(result.get("device", "")),
            "X-SingSong-Duration": str(result.get("duration_s", "")),
            "X-SingSong-Windows": str(len(result.get("windows", []))),
            "X-SingSong-Band-Gain-Db": (
                f"{float(band_placement_gain_db(take, sample_rate=sample_rate)):.2f}"
            ),
        },
    }


@app.get("/singsong/status")
def singsong_status():
    try:
        engine = _get_sing_song_engine()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({
            "enabled": False, "ready": False, "loaded": False,
            "error": str(e),
            # The panel is never dead: the built-in band engine writes a band
            # under the take with no trained weights at all.
            "band_available": True, "engine": "built-in",
            "band": SINGSONG_BUILTIN_BAND,
        }, status_code=200)
    status = engine.status()
    if _sing_song_engine_error:
        status["error"] = _sing_song_engine_error
    status["band_available"] = True
    status["engine"] = "neural" if status.get("ready") else "built-in"
    status["band"] = SINGSONG_BUILTIN_BAND
    return status


@app.get("/singsong")
def singsong_page():
    # The standalone page was a demo shell; the working controls now live in
    # the Live Demo page with every other part of the instrument.  Keep the old
    # URL as a redirect so bookmarks and links from the docs do not 404.
    return RedirectResponse(url="/live-demo#singsong", status_code=307)


@app.post("/api/singsong/render")
async def singsong_render(
    file: UploadFile = File(...),
    return_mix: bool = Form(True),
    band: str = Form(SINGSONG_BUILTIN_BAND),
):
    """Generate an instrumental for an uploaded vocal and return a WAV.

    The response is the mix (vocal + instrumental) by default; pass
    ``return_mix=false`` for the isolated generated instrumental.  Two engines
    answer this route: the trained semantic -> coarse -> fine cascade whenever
    its four artifacts exist, and the built-in symbolic band engine until then,
    so the panel always renders something real.  ``X-SingSong-Engine`` says
    which one ran.
    """
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)

    cache_key = "|".join([hashlib.sha256(raw).hexdigest(), band or SINGSONG_BUILTIN_BAND])
    entry = _singsong_render_cache.get(cache_key)
    if entry is None:
        try:
            entry = await asyncio.to_thread(_singsong_render_both, raw, band)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": f"SingSong render failed: {e}"}, status_code=500)
        if len(_singsong_render_cache) >= _SINGSONG_RENDER_CACHE_LIMIT:
            _singsong_render_cache.pop(next(iter(_singsong_render_cache)))
        _singsong_render_cache[cache_key] = entry

    headers = dict(entry["headers"])
    headers["X-SingSong-Mode"] = "mix" if return_mix else "instrumental"
    return Response(
        content=entry["mix"] if return_mix else entry["band"],
        media_type="audio/wav",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# MySong: symbolic chord accompaniment from a sung or hummed take.
#
# Like SingSong this is lazy: the model is tiny, but loading it pulls in
# PyTorch for the optional BiGRU tagger, and a machine that never opens the
# Live Demo page should not pay for that at process start.  Inference itself
# is CPU-only by design.
# ---------------------------------------------------------------------------
_mysong_model = None
_mysong_model_path = None
_mysong_model_error = None
_mysong_lock = threading.Lock()


def _get_mysong_model():
    global _mysong_model, _mysong_model_path, _mysong_model_error
    if _mysong_model is not None:
        return _mysong_model
    if _mysong_model_error is not None:
        raise RuntimeError(_mysong_model_error)
    try:
        from mysong.model import MySongModel

        root = Path(__file__).parent
        candidates = [
            root / "data" / "mysong" / "model" / "mysong.npz",
            root / "data" / "mysong" / "model" / "mysong_large.npz",
        ]
        model_path = next((path for path in candidates if path.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                "no MySong model found; train one with `python -m mysong.cli train` "
                "or place mysong.npz in data/mysong/model/"
            )
        _mysong_model = MySongModel.load(model_path)
        _mysong_model_path = model_path
        return _mysong_model
    except Exception as e:  # noqa: BLE001 - surfaced by /api/mysong/status
        _mysong_model_error = str(e)
        raise


#: One render is enough for both WAVs the page plays (the mix it downloads and
#: the band it plays under the untouched take), so the last few renders are
#: kept here.  Keyed by the upload's digest and every parameter that shapes the
#: result, so a cache hit is always the same audio.
_mysong_render_cache: "dict[str, dict]" = {}
_MYSONG_RENDER_CACHE_LIMIT = 4


def _run_mysong_render(path: Path, style: str, key: str, return_mix: bool,
                       band_gain_db: float = 0.0, vocal_gain_db: float = 0.0):
    from mysong.pipeline import create_song

    model = _get_mysong_model()
    with _mysong_lock:
        return create_song(
            str(path),
            model=model,
            key=None if (key or "auto").strip().lower() == "auto" else key,
            arrangement_style=style or "hand_drum",
            mix_vocal=return_mix,
            sample_rate=44100,
            render=True,
            band_gain_db=float(band_gain_db),
            vocal_gain_db=float(vocal_gain_db),
        )


def _mysong_readouts(project) -> dict[str, str]:
    """The track strip's readouts, from a finished MySong render.

    Shared with the SingSong panel's built-in band, which runs the same engine,
    so both strips fill from the same numbers under the same header names.
    """
    notes = list(getattr(project.analysis, "notes", []) or [])
    pitches = sorted(int(n.pitch) for n in notes if getattr(n, "pitch", None) is not None)
    if pitches:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        low, high = pitches[0], pitches[-1]
        # ASCII only: header values cannot carry an en dash.
        pitch_readout = f"{names[low % 12]}{low // 12 - 1} - {names[high % 12]}{high // 12 - 1}"
    else:
        pitch_readout = "no voiced notes"
    return {
        "X-MySong-Key": project.harmony.key.name(),
        "X-MySong-Pitch-Range": pitch_readout,
        "X-MySong-Key-Confidence": f"{float(project.harmony.key_confidence):.2f}",
        "X-MySong-Tempo-Confidence": f"{float(project.analysis.tempo.confidence):.2f}",
        "X-MySong-Tempo": f"{project.analysis.tempo.bpm:.1f}",
        "X-MySong-Segments": str(len(project.harmony.segments)),
        "X-MySong-Summary": project.harmony.summary()[:900],
        "X-MySong-Style": project.arrangement.style.name,
        "X-MySong-Duration": f"{len(project.mix) / max(project.sample_rate, 1):.1f}",
    }


@app.get("/api/mysong/status")
def mysong_status():
    try:
        model = _get_mysong_model()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({
            "enabled": False,
            "ready": False,
            "error": str(e),
        }, status_code=200)
    tagger_meta = model.meta.config.get("tagger", {}) if model.meta else {}
    from mysong.accompaniment import ArrangementStyle

    groups = {
        "bands": [name for name in ArrangementStyle.INDIGENOUS],
        "other": [name for name in ArrangementStyle.presets()
                  if name not in ArrangementStyle.INDIGENOUS],
    }
    return {
        "enabled": True,
        "ready": True,
        "model": str(_mysong_model_path or ""),
        "songs": model.meta.songs,
        "transitions": model.meta.transitions,
        "tagger": model.tagger is not None,
        "tagger_extra_scores": tagger_meta.get("extra_scores", 0),
        "key_classifier": model.key_classifier is not None,
        # The bands for this project's own music come first; the rest are the
        # other bands the engine knows.
        "styles": groups["bands"] + groups["other"],
        "styles_grouped": groups,
    }


@app.post("/api/mysong/render")
async def mysong_render(
    file: UploadFile = File(...),
    style: str = Form("hand_drum"),
    key: str = Form("auto"),
    return_mix: bool = Form(True),
    band_gain_db: float = Form(0.0),
    vocal_gain_db: float = Form(0.0),
):
    """Harmonize an uploaded take and return a WAV accompaniment.

    ``return_mix=true`` returns the original vocal plus the generated band;
    ``false`` returns the isolated band.  The response carries the detected
    key, tempo, chord count and a compact chord summary in headers so the
    page can describe the result without a second inference pass.
    """
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    suffix = Path(file.filename or "take.wav").suffix.lower()
    if suffix not in {".wav", ".wave", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}:
        suffix = ".wav"

    def encode(audio, rate):
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    cache_key = "|".join([
        hashlib.sha256(raw).hexdigest(), str(style), str(key),
        f"{float(band_gain_db):.3f}", f"{float(vocal_gain_db):.3f}",
    ])
    cached = _mysong_render_cache.get(cache_key)
    project = None          # the project, when this request did the render
    fresh = False
    try:
        if cached is None:
            with tempfile.TemporaryDirectory(prefix="mysong-") as tmp:
                input_path = Path(tmp) / f"take{suffix}"
                input_path.write_bytes(raw)
                project = await asyncio.to_thread(
                    _run_mysong_render, input_path, style, key, True,
                    band_gain_db, vocal_gain_db,
                )
                band_stem = project.band_stem
                cached = {
                    "mix": encode(project.mix, project.sample_rate),
                    # The stem the page plays under the take: written at the
                    # app's one reference level, with the placement the page
                    # applies on top of it, so the band is always the same
                    # distance under this particular take.
                    "band": encode(band_stem, project.sample_rate) if band_stem is not None else None,
                    "placement_db": f"{float(project.band_placement_db):.2f}",
                }
                if len(_mysong_render_cache) >= _MYSONG_RENDER_CACHE_LIMIT:
                    _mysong_render_cache.pop(next(iter(_mysong_render_cache)))
                _mysong_render_cache[cache_key] = cached
            fresh = True
        if not return_mix and cached.get("band") is None:
            return JSONResponse(
                {"error": "this render has no band stem; request the mix first"},
                status_code=409,
            )
        wav_bytes = cached["mix"] if return_mix else cached["band"]
    except ValueError as e:
        return JSONResponse({"error": f"MySong could not read that take: {e}"}, status_code=400)
    except Exception as e:  # noqa: BLE001
        # audioread's NoBackendError and libsndfile's format error both stringify
        # to an empty message.  That is a bad upload, not a server failure, and
        # the page should say so rather than printing the word "failed" alone.
        if type(e).__name__ in {"NoBackendError", "LibsndfileError"} or not str(e).strip():
            return JSONResponse({
                "error": "MySong could not decode that file. Upload a WAV, FLAC, MP3, OGG or M4A that ffmpeg can read.",
            }, status_code=400)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"MySong render failed: {e}"}, status_code=500)

    if not fresh:
        # A cache hit: the strip's readouts come from the cached render, so the
        # page can fill them without a second model pass.
        summary = cached.get("summary", "")
        warnings = list(cached.get("warnings", []))
        pitch_readout = cached.get("pitch", "")
        key_name = cached.get("key", "")
        key_conf = cached.get("key_confidence", "")
        tempo = cached.get("tempo", "")
        tempo_conf = cached.get("tempo_confidence", "")
        segments = cached.get("segments", "")
        style_name = cached.get("style", style)
        duration = cached.get("duration", "")
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-MySong-Key": key_name,
                "X-MySong-Pitch-Range": pitch_readout,
                "X-MySong-Key-Confidence": key_conf,
                "X-MySong-Tempo-Confidence": tempo_conf,
                "X-MySong-Tempo": tempo,
                "X-MySong-Segments": str(segments),
                "X-MySong-Summary": summary,
                "X-MySong-Style": style_name,
                "X-MySong-Mode": "mix" if return_mix else "band",
                "X-MySong-Duration": str(duration),
                "X-MySong-Warnings": str(len(warnings)),
                "X-MySong-Band-Gain-Db": cached.get("placement_db", "0.00"),
            },
        )

    summary = project.harmony.summary()
    warnings = list(project.warnings)
    # The readouts the page's track strip shows next to the generated audio.
    readouts = _mysong_readouts(project)
    # Everything the strip needs is remembered with the audio, so the band-only
    # request (a cache hit) returns the same readouts.
    cached.update({
        "summary": summary,
        "warnings": warnings,
        "pitch": readouts["X-MySong-Pitch-Range"],
        "key": readouts["X-MySong-Key"],
        "key_confidence": readouts["X-MySong-Key-Confidence"],
        "tempo": readouts["X-MySong-Tempo"],
        "tempo_confidence": readouts["X-MySong-Tempo-Confidence"],
        "segments": readouts["X-MySong-Segments"],
        "style": readouts["X-MySong-Style"],
        "duration": readouts["X-MySong-Duration"],
    })
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            **readouts,
            "X-MySong-Mode": "mix" if return_mix else "band",
            "X-MySong-Warnings": str(len(warnings)),
            "X-MySong-Band-Gain-Db": f"{float(project.band_placement_db):.2f}",
        },
    )


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/architecture")
@app.get("/harmony")
@app.get("/synthesis")
@app.get("/sentiment")
@app.get("/live-demo")
@app.get("/chat")
@app.get("/singsong-guide")
@app.get("/mysong-guide")
@app.get("/measurements")
@app.get("/about")
def spa_routes():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/devices")
def list_devices():
    if not AUDIO_AVAILABLE:
        return {"devices": [], "available": False,
                "message": "Running on a cloud server, local mic not available. Use the browser mic below."}
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


class OrthographyRequest(BaseModel):
    text: str


@app.post("/api/cree/orthography")
def cree_orthography(req: OrthographyRequest):
    """SRO, syllabics and the anglicised respelling for one line.

    The converter is the same module the corpus round-trip test uses; this
    route exists so the site can show the three scripts side by side without
    duplicating the syllabary in JavaScript.
    """
    from analysis.cree_orthography import describe

    text = (req.text or "").strip()
    if not text:
        return {"sro": "", "syllabics": "", "anglicized": "", "syllables": []}
    return describe(text)


# ------------------------------------------------------- nêhiyawêwin speech --
#
# Transcription of the language Archer actually sings in. Whisper cannot do it:
# `language="cr"` is not a Plains Cree decoder, it is the English decoder with a
# language tag (the long note above `_transcribe_audio_blob` is the measurement
# that established that), and what it returns for sung Cree is English words.
# So nêhiyawêwin is transcribed by its own stack -- Meta's MMS-1b-all with both
# crk adapters, a lexicon and character model built from this project's own Cree
# texts, and a CTC beam search that cannot spell an English word -- and this is
# the endpoint that exposes it. Loaded lazily and guarded, because a server that
# cannot find the 1B weights must still start and still serve the rest of the app.

cree_transcriber = None
cree_transcriber_error = None
_cree_transcriber_lock = threading.Lock()


def _get_cree_transcriber():
    """The shared Cree transcriber, or None with the reason in the module global."""
    global cree_transcriber, cree_transcriber_error
    with _cree_transcriber_lock:
        if cree_transcriber is not None or cree_transcriber_error is not None:
            return cree_transcriber
        try:
            from analysis.cree_asr import get_transcriber

            transcriber = get_transcriber()
            if not transcriber.available:
                cree_transcriber_error = (transcriber._asr.load_error
                                          or "MMS crk models unavailable")
                print(f"[cree asr] not available: {cree_transcriber_error}")
                return None
            transcriber.text_model          # build the lexicon/character model now
            cree_transcriber = transcriber
            print(f"[cree asr] ready: {json.dumps(transcriber.health()['text_model'])[:200]}")
        except Exception as exc:  # noqa: BLE001 - reported through the endpoint
            cree_transcriber_error = f"{type(exc).__name__}: {exc}"
            print(f"[cree asr] load failed: {cree_transcriber_error}")
        return cree_transcriber


def _decode_upload_to_16k(raw: bytes) -> np.ndarray:
    """Any uploaded container (webm, mp4, wav, mp3) to float32 mono at 16 kHz."""
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        import librosa

        audio, _sr = librosa.load(path, sr=16000, mono=True)
        return np.asarray(audio, dtype=np.float32)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/api/cree/transcribe")
async def cree_transcribe(file: UploadFile = File(...), isolate: str = Form("")):
    """Transcribe a take of nêhiyawêwin singing or speech, in every script.

    Answers with the same keys whichever script is wanted, because the answer is
    one reading of the audio: `sro` (Standard Roman Orthography, the way a Cree
    reader writes it), `syllabics` (the way a Cree reader in most communities
    reads it) and `anglicized` (the way a non-speaker can sound it out), plus the
    per-segment detail behind them.
    """
    transcriber = _get_cree_transcriber()
    if transcriber is None:
        return JSONResponse(
            {"error": f"Cree transcriber not available: {cree_transcriber_error}"},
            status_code=503)
    try:
        raw = await file.read()
        if len(raw) < 512:
            return JSONResponse({"error": "No audio data received"}, status_code=400)
        audio = await asyncio.to_thread(_decode_upload_to_16k, raw)
        if audio.size == 0:
            return JSONResponse({"error": "Could not decode the audio"}, status_code=400)
        result = await asyncio.to_thread(
            transcriber.transcribe, audio, 16000, isolate=(isolate or None))
        payload = result.to_dict()
        payload["bytes_received"] = len(raw)
        payload["engine"] = "mms-1b-all-crk"
        return payload
    except Exception as exc:  # noqa: BLE001 - the endpoint must answer
        print(f"[cree transcribe error] {type(exc).__name__}: {exc}")
        return JSONResponse({"error": f"Cree transcription failed: {exc}"}, status_code=500)


@app.get("/api/cree/transcribe/health")
def cree_transcribe_health():
    if cree_transcriber is None:
        loaded = _get_cree_transcriber()
        if loaded is None:
            return {"status": "not_loaded", "error": cree_transcriber_error}
    return {"status": "ok", **cree_transcriber.health()}


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

@app.post("/harmony/octave-shift")
def set_octave_shift(req: dict):
    """Set the octave shift offset in real-time. Updates live during playback."""
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    semitones = float(req.get("semitones", 0.0))
    semitones = max(-24, min(24, semitones))
    harmony_engine.set_octave_shift(semitones)
    return {"ok": True, "semitones": semitones}

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
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    harmony_engine.set_fusion_mode(req.enabled)
    return {"ok": True, "fusion_mode": req.enabled}


class TextureRequest(BaseModel):
    texture: str  # "solo" | "duet" | "choir"


class VocableOverrideRequest(BaseModel):
    vocable: str | None = None  # a specific loaded word, or None/"" for "mix" (normal rotation)


@app.post("/harmony/vocable-override")
def set_vocable_override(req: VocableOverrideRequest):
    """
    Locks the shared live HarmonyEngine (used by /ws/mic and
    run_local_pipeline) to ONE specific recorded word and forces plain
    unison shadowing, the backend half of the frontend's "pick one
    word instead of Mix" DSP picker. Sending vocable=null/"" clears the
    lock and restores normal accompaniment-mode/rotation behavior.
    """
    if harmony_engine is None:
        return JSONResponse({"error": f"HarmonyEngine not loaded: {harmony_load_error}"}, status_code=503)
    vocable = (req.vocable or "").strip().lower()
    if vocable and vocable != "mix":
        harmony_engine.set_forced_mode("unison_shadowing")
        harmony_engine.set_vocable_override(vocable)
    else:
        harmony_engine.set_vocable_override(None)
        harmony_engine.set_forced_mode(None)
    return {"ok": True, "vocable": vocable or "mix"}


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
    # Ask the sidecar rather than repeating the verdict formed when this process
    # started. `python server.py` brings the sidecar up alongside it, so that
    # first probe can land while the sidecar is still loading its voices -- and
    # a one-shot probe then leaves the engine switched off and the badge reading
    # "neural sidecar not reachable" for as long as the process lives, however
    # healthy the sidecar's own log is. Probing here means opening the page is
    # what brings the engine back.
    neural_timbre.refresh_sidecar_state()
    status = {
        "loaded": True,
        "enabled": neural_timbre.enabled,
        "sidecar_url": neural_timbre.sidecar_url,
        "sidecar_reachable": neural_timbre._reachable,
        "voices_configured": neural_timbre.num_voices_configured,
    }
    if not neural_timbre._reachable and neural_timbre._probe_error:
        status["sidecar_error"] = neural_timbre._probe_error
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
            pass  # non-fatal, the basic status above is still returned
    return status


SAMPLES_NEURAL_DIR = Path(__file__).parent / "synthesis" / "samples" / "neural"

# The takes as they were ORIGINALLY recorded, what build_vocable_bank.py
# converts FROM. See _source_take_path for why the single-vocable render
# prefers these.
SOURCE_TAKES_DIR = Path(__file__).parent / "synthesis" / "source_takes"


@app.get("/api/vocables/manifest")
def vocables_manifest():
    """
    Exposes the pre-rendered neural vocable takes, the real recorded (and
    voice-converted) "cold"/"moon"/"need"/etc. samples that DSP mode's own
    Python pipeline already sings from, to the browser, so the live mic
    panel's DSP voice can play the ACTUAL recordings (pitch-shifted a
    little in real time) instead of a synthetic sawtooth-and-formants
    oscillator that has never had anything to do with what was recorded.

    Returns available=False (not an error) if the bank hasn't been built
    yet, see build_vocable_bank.py, so the browser can fall back to
    a placeholder voice instead of breaking.
    """
    global synthesizer
    if synthesizer is None or getattr(synthesizer, "_neural_bank", None) is None:
        return {"available": False, "vocables": []}

    bank = synthesizer._neural_bank
    vocables = []
    for vocable, bases in bank._bases.items():
        registers = []
        for base in bases:
            # Same onset-based segmentation the offline single-vocable
            # renderer and the hardware pipeline's _SyllableCycler both
            # use (see _boundaries_for_syllables), computed here, once,
            # server-side, so the browser doesn't need its own separate
            # onset-detection pass over the recording and, more
            # importantly, so the live voice's idea of "where syllable #2
            # starts" is identical to the offline render's. Falls back to
            # treating the whole take as one span if boundary detection
            # itself errors out, so a manifest request never 500s over
            # this.
            try:
                raw = np.asarray(base.audio, dtype=np.float32)
                boundary_samples = _boundaries_for_syllables(
                    raw, bank.sample_rate, min_syllable_s=_dsp_config_for(vocable)["min_syllable_s"]
                )
                syllable_bounds_s = [round(b / bank.sample_rate, 4) for b in boundary_samples]
            except Exception as e:
                print(f"[vocables manifest] syllable boundary detection failed for {base.name}: {e}")
                syllable_bounds_s = [0.0, round(len(base.audio) / bank.sample_rate, 4)]
            registers.append({
                "name": base.name,
                "f0_hz": round(float(base.f0_hz), 2),
                "brightness": round(float(base.brightness), 3),
                "duration_s": round(len(base.audio) / bank.sample_rate, 3),
                "url": f"/api/vocables/sample/{base.name}.wav",
                "syllable_bounds_s": syllable_bounds_s,
            })
        vocables.append({"vocable": vocable, "registers": registers})

    return {"available": True, "sample_rate": bank.sample_rate, "vocables": vocables}


@app.get("/api/vocables/sample/{filename}")
def vocables_sample(filename: str):
    # Guard against path traversal, only a bare filename, no separators
    # or "..", and it must resolve to something actually inside the
    # neural samples directory, never anything reachable by walking out
    # of it.
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)

    candidate = (SAMPLES_NEURAL_DIR / filename).resolve()
    try:
        candidate.relative_to(SAMPLES_NEURAL_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "invalid filename"}, status_code=400)

    if not candidate.exists() or candidate.suffix.lower() != ".wav":
        return JSONResponse({"error": "not found"}, status_code=404)

    return FileResponse(str(candidate), media_type="audio/wav")


class NoteSegment(BaseModel):
    target_hz: float
    duration_s: float


class RenderNoteRequest(BaseModel):
    target_hz: float = 261.63
    vocable: str = "aah"
    duration_s: float = 0.45
    mode: str = "unison_shadowing"
    texture: str = "solo"
    voice_index: int = 0
    transpose_semitones: float | None = None
    # A whole ANSWER PHRASE instead of one note, the shape the live call and
    # response path needs (see synthesis/call_response.py for the composer that
    # decides these notes). The notes keep their own pitches, rhythm and gaps,
    # and the phrase goes through the model in ONE pass: the conversion stage
    # keeps whatever pitch curve it is handed, so a phrase synthesized at its
    # real notes comes back as that same phrase in the trained voice, while a
    # note-at-a-time conversion could only ever arrive long after the note it
    # was meant to be (about 3s per call on the CPU sidecar this project runs
    # on, against answer notes 0.2-0.5s long).
    notes: list[NoteSegment] | None = None


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


def _is_hum_mode(mode: str | None) -> bool:
    """Is this accompaniment mode the closed-mouth hum? One place for the
    question, so every path that has to treat a hum differently asks it the same
    way and none of them compares against a hand-written string."""
    from synthesis.accompaniment_modes import AccompanimentMode

    return (mode or "").strip().lower() == AccompanimentMode.HUM.value


@app.post("/api/neural/render")
def render_neural_note(req: RenderNoteRequest):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready, check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )

    try:
        from config.config_loader import get_config
        from synthesis.vocable_synthesizer import VocableSynthesizer

        cfg = get_config()
        decision = _build_render_decision(req, cfg)

        global synthesizer
        if synthesizer is None:
            return JSONResponse({"error": "Synthesizer not loaded"}, status_code=503)
        sample_rate = cfg["audio"]["sample_rate"]

        notes = [n for n in (req.notes or []) if n.duration_s > 0.02 and n.target_hz > 20.0]
        if len(notes) >= 2:
            scratch_audio = _synthesize_answer_phrase(notes, req, cfg, sample_rate)
            lead_hz = float(notes[0].target_hz)
            pad_seconds = 0.25   # same context padding the live chunk conversions use
        else:
            scratch_audio = synthesizer.synthesize(decision)
            lead_hz = float(decision.target_hz)
            pad_seconds = None

        # A HUM IS NOT CONVERTED, and this is the one mode whose audio never
        # reaches the trained voice. Measured on the takes this endpoint is asked
        # for -- one hummed note a pitch, exactly the body the page's own live
        # bridge posts -- a converter trained on singing answers a closed-mouth
        # hum with a SUNG VOWEL: on a 123 Hz hum the take came back +12 to +18 dB
        # above the murmur's own partials between 500 Hz and 1.1 kHz and +28 to
        # +40 dB above 1.2 kHz, with a broadband tail out to 4 kHz the murmur
        # does not have at all (2.96 per cent of its harmonic energy above 700 Hz
        # against the murmur's 0.08), and on a 220 Hz hum it came back a further
        # 1.9 semitones sharp. That is what the panel's Humming mode sounded
        # like: not a hum in the robot's timbre but a vowel in it. Twelve, sixteen
        # and twenty-two dB dips cannot remove a formant the model put there, and
        # no filter can put a pitch back, so the murmur the DSP path renders IS
        # the hum, in both engines (see synthesis/hum_voice.py).
        if _is_hum_mode(req.mode):
            final_audio = scratch_audio
        else:
            final_audio = neural_timbre.convert(
                scratch_audio, sample_rate, lead_hz, voice_index=req.voice_index,
                pad_seconds=pad_seconds,
                transpose_semitones=req.transpose_semitones,
            )
        # Whether the trained voice actually did this one. The conversion stage
        # hands back the untouched DSP audio when the sidecar is busy (it runs
        # one conversion at a time), which is the right way to degrade, but the
        # page has to be able to say which of the two it is holding instead of
        # labelling every take as neural.
        converted = final_audio is not scratch_audio

        buf = io.BytesIO()
        sf.write(buf, final_audio.astype(np.float32), sample_rate, format="WAV")
        buf.seek(0)
        return Response(
            content=buf.read(), media_type="audio/wav",
            headers={"X-Neural-Applied": "1" if converted else "0"},
        )

    except Exception as e:
        print(f"[neural render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


def _synthesize_answer_phrase(notes: "list[NoteSegment]", req: "RenderNoteRequest",
                              cfg: dict, sample_rate: int) -> np.ndarray:
    """One answer phrase, in one scratch buffer.

    Each note is synthesized by the DSP voice at its own pitch for its own
    duration and laid down end to end, so the rhythm the composer decided is the
    rhythm that gets converted, and the model sees the whole phrase as one
    utterance rather than a stack of unrelated syllables. A short gap of real
    silence between notes is what makes a stepwise answer read as separate notes
    instead of one sliding tone; the tail of each take rings on over it, which is
    what a sung phrase does anyway.
    """
    gap_s = 0.035
    pieces: list[np.ndarray] = []
    for i, note in enumerate(notes):
        sub = RenderNoteRequest(
            target_hz=float(note.target_hz),
            vocable=req.vocable,
            duration_s=float(note.duration_s),
            mode=req.mode,
            texture=req.texture,
            voice_index=req.voice_index,
            transpose_semitones=req.transpose_semitones,
        )
        pieces.append(np.asarray(synthesizer.synthesize(_build_render_decision(sub, cfg)),
                                 dtype=np.float32))
        if i < len(notes) - 1:
            pieces.append(np.zeros(max(1, int(gap_s * sample_rate)), dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


# What the delay/echo effect is allowed to peak at, as a fraction of full
# scale. It sits just under where every other render's peak ceiling does, so
# switching the mode to Delayed response inside the panel is a change of
# character and never a change of level.
ECHO_PEAK_CEILING = 0.95

# What a hum's render is brought to, so the ROBOT VOLUME slider means one thing
# across the whole mode row: the average energy every other mode's render has
# already (measured over the panel's own endpoint: 0.14-0.24 RMS for the
# singing modes, 0.100 for the hum). See the note in _render_track_offline.
HUM_RMS_TARGET = 0.20


def apply_echo_effect(audio: np.ndarray, sample_rate: int, delay_ms: float, feedback: float,
                       max_repeats: int = 10, damping_hz: float = 3200.0) -> np.ndarray:
    feedback = float(np.clip(feedback, 0.0, 0.88))
    if feedback <= 0.001 or delay_ms <= 0 or len(audio) < 64:
        return audio

    delay_samples = int(delay_ms * sample_rate / 1000)
    if delay_samples < 1:
        return audio

    n_audible = int(np.ceil(np.log(10 ** (-50 / 20)) / np.log(feedback)))
    n_taps = max(1, min(max_repeats, n_audible))

    out = np.zeros(len(audio) + delay_samples * n_taps, dtype=np.float64)
    out[:len(audio)] = audio

    tap_signal = audio.astype(np.float64)
    nyquist = sample_rate / 2.0
    for k in range(1, n_taps + 1):
        gain = feedback ** k
        corner_hz = max(500.0, damping_hz * (0.8 ** (k - 1)))
        b, a = butter(2, min(0.99, corner_hz / nyquist), btype="low")
        tap_signal = lfilter(b, a, tap_signal)  # cumulative, each tap is filtered again on top of the last
        start = delay_samples * k
        end = start + len(tap_signal)
        out[start:end] += tap_signal * gain

    # A DELAY IS NOT A LOUDNESS CONTROL. Every tap is summed ON TOP of the dry
    # signal, so a tap landing where the dry part is loud pushes the peak past
    # full scale. This used to answer that by dividing the whole effect by its
    # own peak, but only WHEN it exceeded 1.0, which landed the result on
    # exactly full scale: measured through the panel's own endpoint, two of the
    # three textures came back at peak 1.000 with a sample sitting on the
    # clipping rail, i.e. the one mode whose character is an echo also came
    # back the loudest and the only one that clipped. The echo is brought back
    # to the level it was added to instead (never above its own dry signal),
    # and to the same ceiling every other render respects if that dry signal
    # was already close to it. The effect moves the sound, it doesn't raise it.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    dry_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    target_peak = min(dry_peak, ECHO_PEAK_CEILING) if dry_peak > 0 else ECHO_PEAK_CEILING
    if peak > target_peak > 0:
        out = out * (target_peak / peak)

    return out.astype(np.float32)


def _render_hum_ensemble(audio: np.ndarray, sample_rate: int, hum_cfg, voices: int,
                          detune_spread_cents: float, timing_jitter_ms: float,
                          seed: int = 4242) -> np.ndarray:
    """THE HUM, SUNG BY A SECTION INSTEAD OF ONE MOUTH.

    The panel asks for Solo, Duet or Choir in every engine and every mode, and
    Duet and Choir mean "several independent voices with a little space between
    them", which for the singing modes comes out of
    VocableSynthesizer._render_ensemble. Humming had no such path: it rendered
    one closed-mouth voice no matter which texture was picked, so measured
    through the panel's own endpoint the three textures came back as three
    bit-identical files and the texture row did nothing at all in this mode.

    The section is built the same way the singing one is, out of things that are
    real about several people humming together: each voice is detuned by its own
    small amount, its own hum starts a few milliseconds off the others', and the
    whole section is brought back to the loudness ONE hum had, so switching to
    Choir widens the mode without also turning it up.

    The melody is tracked once and shared; only the synthesis is per voice, which
    is what keeps a seven-voice choir affordable here (the pitch tracking is the
    expensive half, see HumConfig).
    """
    from synthesis.hum_voice import HumConfig, track_melody, synthesize_hum

    cfg = hum_cfg if isinstance(hum_cfg, HumConfig) else HumConfig.from_config(hum_cfg)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    melody = track_melody(audio, sample_rate, cfg)
    if melody.f0_hz.size == 0:
        return np.zeros_like(audio, dtype=np.float32)
    voiced_share = float(np.mean(melody.voiced))
    voiced_s = float(np.sum(melody.voiced) * melody.hop_s)
    if voiced_share < cfg.min_voiced_share or voiced_s < cfg.min_voiced_s:
        print(f"[hum] nothing to hum in this track: {voiced_share * 100:.1f}% of its frames are "
              f"sung ({voiced_s:.2f}s), under the {cfg.min_voiced_share * 100:.0f}% / "
              f"{cfg.min_voiced_s:.1f}s floor")
        return np.zeros_like(audio, dtype=np.float32)

    frames = max(1, int(math.ceil(audio.size / (melody.hop_s * sample_rate))))
    idx = np.clip(np.arange(frames), 0, melody.f0_hz.size - 1)
    f0 = melody.f0_hz[idx]
    rms = melody.rms[idx]
    voiced = melody.voiced[idx].astype(np.float64)

    voices = max(1, int(voices))
    if voices <= 1:
        return synthesize_hum(f0, sample_rate, cfg, rms_curve=rms, hop_s=melody.hop_s, voiced=voiced)

    offsets = np.linspace(-float(detune_spread_cents), float(detune_spread_cents), voices)
    offsets[0] = 0.0
    rng = np.random.default_rng(seed)
    jitter = np.zeros(voices, dtype=int)
    max_jitter = int(max(0.0, timing_jitter_ms) * sample_rate / 1000.0)
    if max_jitter > 0:
        jitter[1:] = rng.integers(-max_jitter, max_jitter + 1, voices - 1)

    solo = None
    mix = None
    for i in range(voices):
        cents = float(offsets[i])
        curve = f0 * float(2.0 ** (cents / 1200.0))
        voice_audio = np.asarray(
            synthesize_hum(curve, sample_rate, cfg, rms_curve=rms, hop_s=melody.hop_s,
                           voiced=voiced, rng=np.random.default_rng(seed + 17 * i)),
            dtype=np.float64,
        )
        if i == 0:
            solo = voice_audio.copy()
        if jitter[i]:
            voice_audio = np.roll(voice_audio, int(jitter[i]))
        gain = 1.0 / (1.0 + 0.12 * abs(cents) / max(float(detune_spread_cents), 1.0))
        mix = voice_audio * gain if mix is None else mix + voice_audio * gain

    solo_rms = float(np.sqrt(np.mean(solo ** 2))) if solo is not None and solo.size else 0.0
    mix_rms = float(np.sqrt(np.mean(mix ** 2))) if mix is not None and mix.size else 0.0
    if solo_rms > 0 and mix_rms > 0:
        # A SECTION IS NOT LOUDER THAN ONE SINGER, it is wider. The mix is
        # scaled back to the LOUDNESS the solo hum would have had, so the
        # texture changes the sound and never the fader. Loudness, not peak:
        # several independent voices summing have a lower crest factor than
        # one, so matching peaks would have made Choir measurably quieter than
        # Solo (measured: peak-matched copies came out 0.31 / 0.22 / 0.20 RMS
        # for solo / duet / choir, i.e. picking a wider texture turned the part
        # down by 4 dB).
        mix = mix * (solo_rms / mix_rms)
    print(f"[hum] {voices} closed-mouth voice(s), detuned up to "
          f"{float(detune_spread_cents):.0f} cents with up to {max_jitter / sample_rate * 1000:.0f} ms "
          f"of onset spread, brought back to the single voice's own level.")
    return np.asarray(mix, dtype=np.float32)


def _trim_excess_silence_at_edges(segment: np.ndarray, sample_rate: int,
                                   threshold_ratio: float = 0.08, max_dead_s: float = 0.015) -> np.ndarray:
    """
    Some syllables carry a longer-than-necessary near-silent stretch at
    their very start or end, a soft breath before the vowel really
    starts, or a natural decay tail. That's a real, unedited feature of
    the recording, not a defect, fine, even desirable, the one time a
    syllable is used at a genuine phrase start. But left in, that dead
    air becomes audible as a dropout-like gap every other time this
    syllable gets re-triggered mid-performance, where continuous sound
    is expected. This trims any such near-silent stretch down to a
    small margin, without touching a single sample of the part that's
    actually sounding, it only shortens genuinely near-zero material.
    """
    if len(segment) == 0:
        return segment
    peak = np.max(np.abs(segment))
    if peak <= 1e-9:
        return segment
    above = np.abs(segment) > threshold_ratio * peak
    if not np.any(above):
        return segment
    first = int(np.argmax(above))
    last = len(segment) - 1 - int(np.argmax(above[::-1]))
    max_dead = int(max_dead_s * sample_rate)
    trim_start = max(0, first - max_dead)
    trim_end = min(len(segment), last + max_dead + 1)
    return segment[trim_start:trim_end]


# Per-vocable tuning for syllable segmentation (see
# _segment_into_syllables' docstring for what this controls). Different
# recordings genuinely need different treatment here: a short, single-
# hit vocable ("hey", "oh") is basically already one syllable, and the
# default threshold leaves it alone either way. But a longer, fluidly
# SUNG multi-syllable take ("Lalanana") has enough internal amplitude/
# spectral wobble across each of its own vowels that the default 0.12s
# minimum lets onset detection carve a single sustained vowel into two
# or three spurious extra pieces. Splicing between fragments of what
# was really ONE syllable, at a spurious mid-vowel cut point that isn't
# a real syllable boundary at all, is exactly what read as "a weird
# breaking sound in the middle of the lalala", not a crossfade bug
# (the crossfade itself is fine, see _apply_internal_crossfades), but a
# segmentation bug feeding it a bad seam to blend in the first place.
#
# 0.42s for "lalanana" was picked by actually analyzing the uploaded
# take (see its amplitude envelope): it collapses the 13 onset-detector
# hits the 0.12s default finds down to the ~6 that line up with the
# recording's own real syllable swells, INCLUDING the one genuine
# internal break it has (a brief dip around the 2.1s mark), so that
# real break is kept as a boundary while the spurious ones inside each
# sustained vowel are merged away. If a future vocable turns out to have
# the same problem, add its own entry here rather than changing the
# shared default, that default still needs to stay short enough for
# genuinely short, fast vocables where 0.12s is already correct.
#
# xfade_ms is the OTHER half of "why does one lalala finishing and the
# next starting still sound sudden/broken, not human", every splice
# (syllable-to-syllable within a held note, note-to-note across the
# song, and the word wrapping back around to its own first syllable
# once it's used the whole thing) gets blended over this many ms (see
# _apply_internal_crossfades). This used to be a flat ~7ms everywhere
# just long enough to avoid an actual click, but nowhere near how long a
# real singer's voice actually takes to glide from one syllable into the
# next (natural co-articulation between vowels typically runs 30-80ms).
# 7ms reads as an abrupt swap, not a glide, which is exactly "too
# sudden, too immediate, doesn't sound human." Raised to 35ms by
# default, long enough to read as a real glide, short enough to still
# keep two genuinely different, fast consecutive syllables distinct
# rather than smearing them together. A vocable with unusually short,
# staccato syllables might want this shorter; add its own entry here if
# so, same as min_syllable_s.
# min_syllable_hold_s controls how fast THIS vocable is allowed to
# re-attack a fresh syllable at all (see the BUG FIX comment where
# min_hold_n is used, in the note-scheduling loop), how "sped up" or
# "chopped" a fast run of notes sounds is governed by this, not by
# min_syllable_s (which only controls where the recording gets cut into
# syllables in the first place). Defaults to the same value as
# min_syllable_s when not given explicitly, since a vocable whose own
# syllables are naturally that long shouldn't be asked to re-attack
# faster than that anyway.
DEFAULT_DSP_CONFIG = {"min_syllable_s": 0.12, "xfade_ms": 35, "min_syllable_hold_s": 0.12}
VOCABLE_DSP_CONFIG = {
    "lalanana": {"min_syllable_s": 0.42, "xfade_ms": 35, "min_syllable_hold_s": 0.42},
    "lala": {"min_syllable_s": 0.42, "xfade_ms": 35, "min_syllable_hold_s": 0.42},
}


def _dsp_config_for(vocable: str) -> dict:
    return VOCABLE_DSP_CONFIG.get((vocable or "").strip().lower(), DEFAULT_DSP_CONFIG)


def _apply_internal_crossfades(buf: np.ndarray, splice_offsets: list, overlap: int,
                                prev_tail: np.ndarray | None = None) -> np.ndarray:
    """
    ROOT-CAUSE FIX for "the DSP/locked-vocable voice sounds broken, with
    a lot of stops and cracks" and for "when the word finishes once and
    starts again, that part sounds broken too", both reports turned
    out to be the exact same underlying mechanism, just triggered at
    different rates.

    Every syllable _segment_into_syllables produces already has its own
    head and tail ramped down toward silence (see _declick_loop_seam)
    that's correct, and necessary, for looping any ONE syllable by
    itself without a click. But both consumers of that syllable list
    (_render_track_single_vocable_offline, building one continuous note
    out of possibly several syllables back to back and stringing notes
    together across a phrase; and _SyllableCycler, doing the same thing
    live) were placing these already-independently-declicked pieces
    directly next to each other with nothing else done at the seam.
    Two declicked edges touching is CLICK-free (no discontinuity), but
    it is NOT loudness-free: syllable A's tail ramps down toward zero,
    then syllable B's head ramps back up from zero, right next to it
    so the ear hears a brief, real dip toward silence at every single
    place two pieces meet. One such dip, on its own, is short enough
    (a few ms) to be inaudible. But a whole word is built from many of
    these, every syllable-to-syllable move within one held note, every
    note-to-note move across a sung phrase, AND the exact same thing
    again when a whole word finishes and wraps back around to its own
    first syllable to start over, so over the length of an actual
    performance they land at the RATE the song changes notes (several
    times a second isn't unusual for real singing). That reads as the
    voice repeatedly stopping and stuttering instead of flowing, and
    it's the same mechanism whether the two pieces on either side of a
    given seam happen to be two different syllables of the same word,
    the same syllable repeating on itself, or the last syllable of a
    word handing off to its own first syllable to start the word over.

    The fix is an actual audio crossfade at each seam, blending what's
    already sitting on both sides of it with equal-power (constant
    combined loudness through the blend, no dip) sin/cos curves, not
    a fresh fade to silence and back. This overwrites `overlap` samples
    starting at each offset in `splice_offsets` with that blend; it
    never inserts or removes samples, so a caller that already sized
    `buf` to land on a specific timeline (a note's own duration, a
    phrase's own boundaries) doesn't need to account for any change in
    length, only the content right at each seam changes.

    `prev_tail` optionally supplies the real audio that came immediately
    before `buf` even started (e.g. the previous streamed chunk's last
    few samples in the live path), for blending a seam that falls
    exactly on `buf`'s own first sample (offset 0), there's nothing
    inside `buf` itself to blend against there otherwise.
    """
    n = len(buf)
    if n == 0 or overlap <= 1:
        return buf
    for offset in splice_offsets:
        if offset <= 0:
            if prev_tail is None or len(prev_tail) < 2:
                continue
            ov = min(overlap, len(prev_tail), n)
            if ov <= 1:
                continue
            t = np.linspace(0, np.pi / 2, ov, dtype=np.float32)
            tail = prev_tail[-ov:].astype(np.float32)
            head = buf[:ov].astype(np.float32)
            buf[:ov] = tail * np.cos(t) + head * np.sin(t)
            continue
        ov = min(overlap, offset, n - offset)
        if ov <= 1:
            continue
        t = np.linspace(0, np.pi / 2, ov, dtype=np.float32)
        tail = buf[offset - ov:offset].copy()
        head = buf[offset:offset + ov].copy()
        buf[offset:offset + ov] = tail * np.cos(t) + head * np.sin(t)
    return buf


def _boundaries_for_syllables(raw_audio: np.ndarray, sample_rate: int, min_syllable_s: float = 0.12) -> list:
    """Just the sample-index cut points _segment_into_syllables would use,
    without slicing or declicking, kept separate so the SAME boundaries
    can be reused against a pitch-shifted (duration-preserving) version
    of the same recording later. See the note in
    _render_track_single_vocable_offline's pitch-shift cache for why that
    matters: shifting a note's pitch should never also change which
    stretch of the recording counts as "syllable #3"."""
    import librosa
    if len(raw_audio) == 0:
        return [0, 0]
    onset_samples = librosa.onset.onset_detect(y=raw_audio, sr=sample_rate, units="samples", backtrack=True)
    onset_samples = sorted(set(int(s) for s in onset_samples if 0 <= s < len(raw_audio)))
    if not onset_samples or onset_samples[0] != 0:
        onset_samples = [0] + onset_samples

    min_gap = int(min_syllable_s * sample_rate)
    merged = [onset_samples[0]]
    for s in onset_samples[1:]:
        if s - merged[-1] >= min_gap:
            merged.append(s)

    boundaries = merged + [len(raw_audio)]
    while len(boundaries) > 2 and (boundaries[-1] - boundaries[-2]) < min_gap:
        boundaries.pop(-2)
    return boundaries


def _slice_syllables_at_boundaries(audio: np.ndarray, boundaries: list, sample_rate: int) -> list:
    """Cuts `audio` at pre-computed `boundaries` (see
    _boundaries_for_syllables) and declicks each piece, the actual
    slicing/declicking half of what _segment_into_syllables used to do
    in one step, now reusable against any duration-preserving variant of
    the same original recording (e.g. a pitch-shifted copy) with
    identical cut points."""
    chunks = [
        audio[boundaries[i]:boundaries[i + 1]].copy()
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] > 200  # drop slivers too short to be a real syllable (< ~5ms)
    ]
    syllables = [
        _declick_loop_seam(_trim_excess_silence_at_edges(chunk, sample_rate), fade_samples=64)
        for chunk in chunks
    ]
    return syllables if syllables else [audio.copy()]


def _segment_into_syllables(raw_audio: np.ndarray, sample_rate: int, min_syllable_s: float = 0.12) -> list:
    """
    Splits ONE recorded take (e.g. "Ohoho" or "Lalanana") into its own
    natural syllables via onset detection, so each syllable can be
    re-triggered on its own in time with the song, see
    _render_track_single_vocable_offline's docstring for why this exists
    and what it fixes. Each returned syllable has had any excess dead
    air trimmed from its edges (see _trim_excess_silence_at_edges) and
    _declick_loop_seam applied, so looping any ONE of them individually
    (to sustain through a longer note) is click-free on its own.

    Onsets closer together than min_syllable_s are merged, a real
    syllable has some minimum natural duration; onset detection finding
    two "hits" 20ms apart is noise, not two syllables. This also merges
    a too-short FINAL segment into its predecessor: the gap-merge above
    only guards the space between consecutive onsets, not the tail
    segment's own length against the recording's actual end, so a
    recording whose last real onset happens to fall close to its own
    end would otherwise keep a short, fragile trailing sliver as its own
    "syllable." Falls back to treating the whole recording as a single
    unit if it isn't clearly multi-syllabic or onset detection finds
    nothing usable.
    """
    if len(raw_audio) == 0:
        return [raw_audio]
    boundaries = _boundaries_for_syllables(raw_audio, sample_rate, min_syllable_s)
    return _slice_syllables_at_boundaries(raw_audio, boundaries, sample_rate)


def _source_take_path(vocable: str):
    """The ORIGINAL recording of a vocable, if it is still on disk.

    synthesis/source_takes/ holds the takes exactly as they were recorded
    the same performances the neural bank is built FROM (build_vocable_bank.py
    converts each of these into synthesis/samples/neural/). Singing from the
    source take is what makes this render carry the voice that is actually on
    the input recording, instead of the bank's converted voice; the converted
    copy stays as the fallback for anything that only exists there.

    Matched on letters and digits only, so "Welcome to Reality", "welcome to
    reality" and "Welcome To Reality.wav" are the same take.
    """
    root = SOURCE_TAKES_DIR
    if not root.is_dir():
        return None
    want = "".join(ch for ch in str(vocable).strip().lower() if ch.isalnum())
    if not want:
        return None
    for p in sorted(root.glob("*.wav")):
        have = "".join(ch for ch in p.stem.lower() if ch.isalnum())
        if have == want:
            return p
    return None


def _trim_take_silence(audio: np.ndarray, sr: int, floor_db: float = -30.0,
                       margin_s: float = 0.02) -> np.ndarray:
    """A recorded take with the silent room cut off its head and tail.

    Measured on this project's own takes (synthesis/source_takes/): "Cold" is
    1.24 s long and its last third sits more than 40 dB under its own peak, so
    the word itself is about 0.85 s and the rest is room tone.

    That silence is harmless while the take is read once, it simply never
    sounds, but these renderers read a take CONTINUOUSLY and start it over
    whenever it runs out (world_voice.sing_along loops the take's frames, and
    the tile fallback below repeats the take), so a silent tail comes back
    around once per take length and reads as the robot dropping out in the
    middle of the word. Measured on a rendered "Cold" (a 1.24 s take) sung
    over a held note: silence from 1.02 s to 1.46 s and again from 2.26 s to
    2.62 s, once per take length exactly, while the input was still singing at
    full level through both. Worse than the tail itself, that tail analysed as
    unvoiced, and world_voice.sing_along silences unvoiced frames outright, so
    the drop was digital silence rather than a quiet room.

    `floor_db` is measured against the take's own loudest frame and 30 dB is
    this project's own convention for "sounding" (see
    melody_voice.LOUDNESS_FLOOR_RATIO, 0.05, used by the loudness normalizer):
    what is cut is inaudible under any mix. `margin_s` is kept either side so
    nothing is shaved off the mouth.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame = max(1, int(0.01 * sr))
    n_frames = len(x) // frame
    if n_frames < 4:
        return x
    fr = np.sqrt(np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    peak = float(fr.max())
    if peak <= 0:
        return x
    loud = fr > peak * (10 ** (floor_db / 20.0))
    if not loud.any():
        return x
    margin = max(1, int(round(margin_s * sr / frame)))
    first = max(0, int(np.argmax(loud)) - margin)
    last = min(n_frames, n_frames - int(np.argmax(loud[::-1])) + margin)
    if last <= first:
        return x
    return np.ascontiguousarray(x[first * frame: min(len(x), last * frame)], dtype=np.float32)


def _duck_accompaniment_under_voice(bed, voice, sample_rate: int, max_db: float = 4.0):
    """
    Pull the accompaniment down a few dB while the take is actually singing.

    This is what a mix engineer does to a lead vocal, and it is the difference
    between the words being audible and the words being buried: in the dense
    part of "Suzanne Kite Fm 120" the separated accompaniment and the sung
    take measured within a couple of dB of each other, and a lead that close to
    the bed loses its consonants, which is exactly the complaint ("you cannot
    even hear the words lalala").

    Two details keep it from sounding processed:

      - it LOOKS AHEAD. The gain curve is smoothed with a window centred on the
        activity, so the bed is already down when a note starts rather than
        ducking after it has begun, which is what a slow compressor does and
        what makes naive ducking audible as pumping.
      - it is bounded and shallow: 4 dB at most, only while the voice is
        sounding, and never applied where the voice is silent, an instrumental
        intro or a gap between phrases keeps the accompaniment at full level, so
        the track still leads the mix wherever the singer is not.

    Returns the bed, scaled. Never touches the voice, so the level the browser's
    ROBOT slider is calibrated against stays exactly as normalize_loudness set
    it.
    """
    bed = np.asarray(bed, dtype=np.float64)
    voice = np.asarray(voice, dtype=np.float64)
    n = min(len(bed), len(voice))
    if n == 0 or max_db <= 0.0:
        return bed

    def _moving_mean(x, win):
        win = max(1, int(win))
        if win < 2:
            return x
        pad = win // 2
        padded = np.pad(x, (pad, win - 1 - pad), mode="edge")
        cs = np.concatenate(([0.0], np.cumsum(padded)))
        return (cs[win:] - cs[:-win]) / win

    # Short-window loudness of the voice: 25 ms is long enough to ignore the
    # gap between two glottal pulses and short enough to still follow a
    # syllable, so a consonant inside a word ducks the bed just as a vowel does.
    win = max(2, int(0.025 * sample_rate))
    energy = _moving_mean(np.square(voice[:n]), win)**0.5
    ref = float(np.percentile(energy, 95.0)) if energy.size else 0.0
    if ref <= 1e-6:
        return bed

    # Soft activity ramp rather than a gate: a hard threshold would switch the
    # duck on and off mid-phrase and be heard as a click.
    activity = np.clip((energy - 0.20 * ref) / (0.35 * ref), 0.0, 1.0)
    gain_db = -float(max_db) * activity
    gain_db = _moving_mean(gain_db, int(0.150 * sample_rate))
    gain = np.power(10.0, gain_db / 20.0)

    out = bed.copy()
    out[:n] *= gain
    return out


class NothingToSingError(RuntimeError):
    """The track had no melody to sing, so the render refuses rather than lie.

    Raised by both DSP renderers when the part they produced comes out empty,
    see the guard inside _render_track_single_vocable_offline for what used to
    be sent instead, and /api/dsp/render-track for how the sentence reaches the
    page.
    """


NOTHING_SUNG_PEAK = 1.0e-4


def _apply_vocal_vibration_amount(audio: np.ndarray, sample_rate: int,
                                  vibrato_amount: float | None) -> np.ndarray:
    """The panel's Vocal fold vibration setting, applied to a rendered part.

    One door for every renderer (the DSP whole-track render, the locked-word
    render, the Neural full render and the live converted chunks) so that the
    setting means the same thing everywhere, which is the only way a slider can
    be honest about "all modes, both engines". `None` means the request did not
    say, and the shared default is used (see synthesis.vibrato); an explicit 0
    is a straight tone and is left exactly as it was rendered.
    """
    from synthesis.vibrato import (DEFAULT_VIBRATO_AMOUNT, VocalVibrationConfig,
                                   apply_vocal_vibration)
    amount = DEFAULT_VIBRATO_AMOUNT if vibrato_amount is None else float(vibrato_amount)
    if amount <= 0.0:
        return np.asarray(audio, dtype=np.float32)
    try:
        cfg = VocalVibrationConfig.from_config()
    except Exception as e:
        print(f"[vibrato] using the module's own defaults ({e})")
        cfg = None
    return apply_vocal_vibration(audio, sample_rate, amount, cfg)


def _render_track_single_vocable_offline(raw_audio_bytes: bytes, pitch_method: str, vocable: str,
                                         instruments_enabled: bool = True,
                                         vibrato_amount: float | None = None) -> bytes:
    """
    The render path for "one locked word instead of Mix".

    This is a singer, not a sampler. Every sample that plays back is still
    the recording itself, nothing is re-synthesized from an oscillator and
    the take is never time-stretched, but it is now placed on the SONG's
    own melody and rhythm instead of repeating at whatever pitch it happened
    to be recorded at:

      - WHAT IS SUNG: the performance's own pitch curve is tracked
        (synthesis.melody_voice.track_notes) with detector octave errors
        corrected first, so a new note begins where the SUNG PITCH actually
        moves, which is what a listener hears as the melody. The previous
        version used a generic onset detector as its note grid, which fires
        several times inside a single held vowel (vibrato reads as onsets)
        and misses a note change with a soft attack, so its rhythm was only
        loosely related to the singing.
      - AT WHAT PITCH, AND IN WHOSE VOICE: the take is analysed once with
        the WORLD vocoder (synthesis/world_voice.py), which separates a sung
        performance into its pitch curve, its spectral envelope (the vocal
        tract: formants, vowel colours, nasality) and its aperiodicity (the
        breath and the consonants). Only the pitch is rewritten, with the
        melody's own pitch frame by frame, so the vocal tract that sings the
        result is the recording's own at every pitch: no resampling and no
        phase vocoder anywhere in this path. The whole part is placed in the
        octave closest to the recording's own register (octave_placement
        chosen once for the part, so a melody that genuinely drops an octave
        still drops an octave here).
      - HOW IT MOVES: the melody inside a phrase is applied to each frame of
        the take as a PITCH, never as a re-read, so the word keeps the
        timing it was recorded at instead of being stretched and squeezed as
        the tune moves, which is what made the words hard to hear. The take's
        own fast pitch movement (its vibrato and micro-movement) is kept,
        scaled and capped, on top of the melody, and the frames it sang
        unvoiced stay unvoiced, so the consonants are the recording's own
        rather than a continuous vowel.
      - HOW LOUD: the output volume continuously follows the singer's own
        dynamics (apply_amplitude_curve_follow), taken from the ISOLATED
        VOCAL's own level, see the fix comment where amplitude_curve is
        filled in below for why the previous version of this silently did
        nothing at all.

    Wherever the track isn't singing there is still silence, gated by
    amplitude VAD plus a genuine singing-pitch check, both run against a
    Demucs-separated vocal (see the source-separation comment below for why
    amplitude and pitch alone are not enough on a full mixed track).

    `pitch_method` (yin / yinfft / rmvpe) is the caller's choice of detector,
    and it is what runs whenever the neural one cannot: this path tracks the
    melody with RMVPE when its checkpoint is present, because on real recorded
    voice the local detector locks onto harmonics of the sung notes (measured
    on this project's own reference track: a held note near 200 Hz reported at
    ~405 Hz for 1.8 s, so a whole phrase came out an octave high), and offline
    it can afford the slower, better one. When RMVPE is missing it degrades to
    the requested method with no failure, and `fold_octave_errors` in
    melody_voice.py still guards against octave flips in whatever curve it
    gets.
    """
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector
    from synthesis.vocable_synthesizer import apply_amplitude_curve_follow
    from synthesis.melody_voice import (
        track_notes, stabilize_note_octaves, octave_placement, take_f0,
        FormantPreservingShifter, normalize_loudness,
        flatten_syllable_pitch,
    )
    from synthesis import world_voice
    import bisect
    import librosa
    import math

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]
    frame_hop_s = frame_size / sample_rate
    synth_cfg = cfg.get("synthesis", {})
    sv_cfg = synth_cfg.get("single_vocable", {}) or {}

    global synthesizer
    if synthesizer is None or getattr(synthesizer, "_neural_bank", None) is None:
        raise RuntimeError("Neural vocable bank not loaded, build it with build_vocable_bank.py first")
    bases = synthesizer._neural_bank._bases.get(vocable.strip().lower())
    # WHICH RECORDING IS ACTUALLY SUNG. The neural bank is built by sending
    # each source take through the RVC voice conversion, so its copy is the
    # CONVERTED voice, a different person's timbre wearing this performance.
    # The user's own complaint about this mode ("it does not sound like the
    # singer who sang the lalala at all") is exactly that: the part was being
    # sung by the bank's voice, not by the voice on the recording. The source
    # take is the performance AS RECORDED, so the default is now to sing from
    # it (synthesis.single_vocable.voice_source: source_take) and to keep the
    # converted copy only as a fallback.
    voice_source = str(sv_cfg.get("voice_source", "source_take")).strip().lower()
    raw_audio = None
    source_path = _source_take_path(vocable) if voice_source == "source_take" else None
    if source_path is not None:
        try:
            take_audio, take_sr = sf.read(str(source_path), dtype="float32", always_2d=False)
            if take_audio.ndim > 1:
                take_audio = take_audio.mean(axis=1)
            if take_sr != sample_rate:
                take_audio = librosa.resample(take_audio, orig_sr=take_sr, target_sr=sample_rate)
            raw_audio = np.ascontiguousarray(take_audio, dtype=np.float32)
            print(f"[dsp/render-track] '{vocable}': singing from the ORIGINAL recorded take "
                  f"({source_path.name}), not the converted bank voice.")
        except Exception as e:
            print(f"[dsp/render-track] could not read the original take "
                  f"{source_path.name} ({e}), falling back to the bank's copy.")
    if raw_audio is None:
        if not bases:
            raise RuntimeError(f"no recorded take loaded for vocable '{vocable}'")
        raw_audio = np.asarray(bases[0].audio, dtype=np.float32)
    # THE TAKE LOOPS, SO ITS SILENT ROOM MUST NOT (see _trim_take_silence):
    # without this the take's own tail plays back once per loop, which is what
    # the dropouts measured every 1.24 s inside a held note were (exact
    # silence for 0.44 s per loop of a 1.24 s take).
    untrimmed_s = len(raw_audio) / float(sample_rate)
    raw_audio = _trim_take_silence(raw_audio, sample_rate)
    if len(raw_audio) / float(sample_rate) < untrimmed_s - 0.01:
        print(f"[dsp/render-track] '{vocable}': trim {untrimmed_s:.2f}s -> "
              f"{len(raw_audio) / float(sample_rate):.2f}s of sounding take, so the "
              "loop wraps at the end of the word and not in the room after it.")
    dsp_cfg = _dsp_config_for(vocable)
    syllables = _segment_into_syllables(raw_audio, sample_rate, min_syllable_s=dsp_cfg["min_syllable_s"])
    # Each syllable's own pitch, measured once: the shift a note needs is
    # relative to the pitch of the syllable actually being read, not to some
    # average for the whole take, a two-syllable word is often sung on two
    # different notes.
    # Flatten each syllable's own pitch movement BEFORE anything is shifted
    # (see melody_voice.flatten_syllable_pitch): measured on these takes, one
    # syllable sweeps through +/-200 cents or more, so a single constant shift
    # would leave the note out of tune across most of its length, and a long
    # note that loops the syllable would replay that sweep once per loop. After
    # flattening, a syllable holds ONE pitch, so the shift lands the note
    # exactly and the musical movement is added on top of it deliberately.
    # `take_f0` is no longer what a syllable's pitch is taken from: the median
    # of its own measured pitch curve is.
    register_hz = take_f0(raw_audio, sample_rate)
    # What each syllable measures as, once: the register the take sits in
    # (used to place the part) and the log line below. The audio those
    # measurements shaped is no longer what gets sung, that is the take as
    # one continuous performance, flattened as a unit further down.
    syllable_f0 = []
    for s in syllables:
        _, measured = flatten_syllable_pitch(s, sample_rate, register_hz=register_hz)
        syllable_f0.append(measured)
    described = ", ".join("unpitched" if f <= 0 else f"{f:.0f} Hz" for f in syllable_f0)
    print(f"[dsp/render-track] '{vocable}': {len(syllables)} syllable(s), {described}")
    # THE TAKE AS ONE CONTINUOUS PERFORMANCE. This is the material every
    # phrase is sung from, and it is the fix for "you cannot even hear the
    # words": the renderer used to re-trigger the word's NEXT SYLLABLE at
    # every note (see the phrase loop below), so on a track whose notes are
    # shorter than a syllable, about half a second on this project's own
    # take, against notes of 0.2 to 0.4 s on the reference track, every note
    # played the first fraction of a DIFFERENT syllable, each with its own
    # attack ramp. No listener ever heard a complete word, every vowel was cut
    # mid-mouth, and the result read as robotic fragments instead of singing.
    # Reading through the take in one pass instead means the whole word is
    # heard at its own recorded pace, as many times as the phrase lasts, and
    # the only thing that moves from note to note is the pitch it is placed
    # at.
    #
    # The slow-movement removal is the same one each syllable gets (see
    # melody_voice.flatten_syllable_pitch): the take's own long sweeps are
    # what a constant shift cannot follow, while its fast life (vibrato, the
    # micro-movement inside a syllable) is kept, because that is the part
    # that sounds like a person.
    word_take, word_hz = flatten_syllable_pitch(raw_audio, sample_rate, register_hz=register_hz)
    if word_hz <= 0:
        word_hz = register_hz
    print(f"[dsp/render-track] '{vocable}': the take as one performance, "
          f"{word_hz:.0f} Hz over {len(word_take) / sample_rate:.2f}s")

    max_shift = float(sv_cfg.get("max_shift_semitones", 12.0))
    # How far a note can sit from the take's own pitch before it stops
    # sounding like the same singer. Used to choose each phrase's octave (see
    # the phrase loop): a phrase is placed where its notes sit closest to the
    # take's own register, so as much of it as possible is sung nearly
    # unshifted and sounds most like the recording.
    COMFORT_SEMITONES = 7.0
    # How far, in cents, the melody's own movement inside one phrase is
    # allowed to pull the take (see the phrase loop): a phrase's notes sit
    # within a few semitones of each other in practice, and this is a bound on
    # the extreme, so the take can never be dragged somewhere unrecognisable.
    CURVE_CAP_CENTS = 700.0
    # The demo page's own default for its TRACK VOLUME slider (see
    # frontend/index.html: demoSourceVolume). The accompaniment channel is sent
    # at UNITY, baking this default in here would apply the page's own slider
    # twice, so the constant is used for one thing only: modelling the mix the
    # page will actually play at its default positions, so the shared trim below
    # can be fitted to the loudness a user hears without touching a slider.
    DEFAULT_DEMO_TRACK_VOLUME = 0.78
    # How far the accompaniment may be pulled down while the take is singing.
    # 4 dB is a normal lead-vocal duck and it is the whole of the fix for "you
    # cannot even hear the words": at the default sliders the two were within
    # ~2.6-3 dB of each other in the dense last phrase, which is close enough for
    # the bed to mask the consonants that carry the word.
    DUCK_MAX_DB = 4.0
    # One shifter for the whole render: memoizes each shifted copy by
    # (take, quarter-semitone), so the handful of pitches a melody actually
    # uses are computed once each instead of once per note.
    shifter = FormantPreservingShifter(sample_rate, max_shift_semitones=max_shift)
    # ONE ANALYSIS OF THE TAKE, shared by every phrase below: its own pitch
    # curve, its vocal tract and its breathiness (see
    # synthesis/world_voice.py). This replaces the chain that used to be here
    #, a bucket phase vocoder for a constant shift, plus a variable-rate
    # re-read for the melody's movement, for two structural reasons: the
    # re-read changed the SPEED of the word as it changed its pitch (so the
    # syllables were stretched and squeezed as the tune moved, which is what
    # made the word hard to hear), and the phase vocoder rebuilt the sound
    # from its own phase guesswork (which is what "robotic" actually is).
    performance = world_voice.analyze_performance(raw_audio, sample_rate, register_hz=register_hz)
    if performance is not None:
        perf_hz = performance["f0"][performance["voiced"]]
        if perf_hz.size >= 4:
            # the take's own register, measured on the same frames the part is
            # sung from rather than by the local detector
            word_hz = float(np.median(perf_hz))
        print(f"[dsp/render-track] '{vocable}': singing with WORLD "
              f"({world_voice.describe(performance)}); the take's own vocal tract and "
              "breathiness are kept and only its pitch is rewritten.")
    else:
        print("[dsp/render-track] WORLD unavailable, falling back to the formant-preserving "
              "shifter: the phrase's own register can still be sung, but the melody's "
              "movement inside it cannot be followed that way.")

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    preproc = Preprocessor()
    pitch = PitchDetector()
    requested_method = str(pitch_method or "yin").strip().lower()
    try:
        effective_method = pitch.set_method(requested_method)
    except Exception:
        effective_method = pitch.set_method("yin")

    # WHICH DETECTOR TRACKS THE MELODY. The offline render gets to look at the
    # whole song before it decides anything, so it should use the best pitch
    # information it can get, and on real recorded voice that is the neural
    # detector. Measured on this project's own reference track, the local one
    # locks onto a harmonic of a sung note rather than its fundamental, a
    # held note near 200 Hz was reported at ~405 Hz for 1.8 s, because that
    # female voice's fundamental is genuinely weak in the spectrum and its 2nd
    # harmonic is the strongest partial. A whole phrase came out an octave
    # above the singer that way, which is exactly the "part sounds like
    # something separate from the track" failure. RMVPE tracks the same frames
    # at ~203 Hz with no octave flip. So: use RMVPE whenever the model is
    # available, and fall back to whatever was asked for when it is not
    # (pitch.set_method already degrades to yin if the checkpoint is missing).
    melody_method = effective_method
    if effective_method != "rmvpe":
        try:
            if pitch.set_method("rmvpe") == "rmvpe":
                melody_method = "rmvpe"
                print(f"[dsp/render-track] tracking the melody with RMVPE (requested "
                      f"{effective_method}): the local detector locked onto harmonics of the "
                      "sung notes on real voice in testing, and this path has the whole song "
                      "to analyse, so it can afford the better one.")
        except Exception as e:
            print(f"[dsp/render-track] RMVPE unavailable ({e}), tracking the melody with "
                  f"{effective_method}.")

    # BUG FIX, ROUND 2: requiring a genuine pitch detection on top of
    # amplitude VAD still isn't enough on its own, it only catches an
    # instrumental backing that's atonal or unpitched (drums, noise). Plenty
    # of real backing is NOT that: a piano, a guitar, a synth pad holding a
    # chord all have a perfectly clear, confident, singing-range pitch of
    # their own. Pitch detection cannot tell "an instrument is playing a
    # note" apart from "a person is singing a note", those are genuinely
    # the same kind of signal to a pitch detector. That is NOT a threshold
    # to tune; it needs a different kind of information entirely: what's
    # actually a human voice versus what's an instrument, which is exactly
    # what source separation is for. This reuses _separate_vocal, the same
    # Demucs model _convert_track_direct's "instruments off" mode already
    # relies on, to isolate the vocal BEFORE running any VAD on it, so a
    # genuine instrumental-only pause reads as what it actually is
    # (near-silent in the isolated vocal), no matter how loud the full mix
    # still is right on top of it. Falls back to _silence_instrumental_spans
    # (a lighter harmonic-ratio/spectral-flatness heuristic, same one used
    # elsewhere in this file) on very short clips where loading the Demucs
    # model isn't worth it, and whenever Demucs itself fails for any reason.
    #
    # As well as the gate, this separated vocal is what the MELODY is tracked
    # from, a bass line or a synth pad has a perfectly good pitch of its
    # own and would otherwise be tracked as the melody in the gaps where the
    # singer is quiet.
    #
    # Nothing here is ever mixed into what is actually heard: the output is
    # built entirely from the robot's own recorded take.
    DEMUCS_MIN_DURATION_S = 6.0
    duration_s = len(audio) / sample_rate
    # The accompaniment comes out of the SAME separation as the vocal (see
    # _separate_stems): the vocal is what the melody/rhythm/silence detection
    # analyses, and the accompaniment is what the finished singer is heard
    # over. Both halves are needed, and one Demucs run produces both, so the
    # second half costs nothing.
    accompaniment = None
    if duration_s >= DEMUCS_MIN_DURATION_S:
        try:
            vocal_for_analysis, accompaniment = _separate_stems(audio, sample_rate)
            print(f"[dsp/render-track] separated vocal + accompaniment via Demucs "
                  f"({duration_s:.1f}s clip) for melody/rhythm/silence detection and "
                  "for what the singer is finally heard over.")
        except Exception as e:
            print(f"[dsp/render-track] Demucs separation failed ({e}), falling back to "
                  "frame-level instrumental muting for VAD on this render.")
            vocal_for_analysis = _silence_instrumental_spans(audio, sample_rate, "dsp/render-track")
    else:
        vocal_for_analysis = _silence_instrumental_spans(audio, sample_rate, "dsp/render-track")

    # BUG FIX: the gate below used to be `is_voiced_flags`, i.e. Preprocessor's
    # amplitude VAD alone, "loud enough to not be silence." That's a fine
    # proxy for "is anyone singing" on a solo a cappella take, but this
    # function renders a whole UPLOADED TRACK, which very often still has
    # drums, bass, or other instruments playing right through a genuine vocal
    # pause. Amplitude-only VAD reads straight through that as "still
    # voiced," so the syllable cycle just kept running underneath the
    # instrumental, "the singer pauses for a few seconds and the robot
    # keeps spamming la-la-la right through it." Requiring a genuine,
    # confident, singing-range pitch detection ON TOP of the amplitude gate,
    # run against the vocal-isolated signal above, fixes it: a real vocal
    # pause under instrumental backing now reads as NOT singing, even though
    # the track is still loud.
    n_frames = len(audio) // frame_size
    is_singing_flags = np.zeros(n_frames, dtype=bool)
    hz_curve = np.zeros(n_frames, dtype=np.float64)
    amplitude_curve = np.zeros(n_frames, dtype=np.float64)

    # BUG FIX: amplitude_curve used to be filled from the RMS of `clean`, the
    # PREPROCESSOR'S OWN OUTPUT. Preprocessor.process() normalizes every
    # voiced frame to a fixed target RMS before returning it, so that "RMS"
    # was the same number on every single voiced frame (measured on a real
    # render: exactly 0.1000 for all of them), which means
    # apply_amplitude_curve_follow divided a constant curve by its own median
    # and applied a flat 1.0 gain everywhere. The feature was silently doing
    # nothing at all, no matter what the singer did. It has to be measured on
    # the RAW isolated-vocal frame, which is what actually carries the
    # performance's dynamics.
    for i in range(n_frames):
        frame = vocal_for_analysis[i * frame_size:(i + 1) * frame_size]
        clean, is_voiced = preproc.process(frame)
        if is_voiced:
            pitch_input = frame if pitch.method == "rmvpe" else clean
            hz, conf = pitch.detect(pitch_input)
            if hz:
                is_singing_flags[i] = True
                hz_curve[i] = float(hz)
                amplitude_curve[i] = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        else:
            pitch.reset()

    # Contiguous voiced stretches = sung phrases, but a brief gap (a
    # breath, a consonant, a word boundary, even a short instrumental beat)
    # is NOT the phrase ending; only a real, sustained stop of a few seconds
    # is. Treating every single non-voiced frame as a hard phrase-end was the
    # actual cause of audible stuttering: real singing has plenty of gaps
    # well under a second between words even within one continuous phrase,
    # and each one was silencing and then restarting the syllable cycle from
    # scratch. Bridging them (up to GAP_TOLERANCE_S) keeps the syllable
    # sequence running straight through, the amplitude-follow below still
    # naturally dips the volume during the gap itself (it's genuinely quiet
    # there), it just doesn't hard-mute and restart for anything short of an
    # actual, multi-second stop in the singing.
    GAP_TOLERANCE_S = 2.5
    gap_tolerance_frames = max(1, int(GAP_TOLERANCE_S * sample_rate / frame_size))

    # BUG FIX: "the synthesis starts before the singer actually does, then
    # finishes its syllable only after the singer's already resumed", a
    # single-frame false-positive reading used to be enough, on its own, to
    # open a whole new phrase immediately. Requiring a short run of
    # CONSECUTIVE genuinely-singing frames before actually opening a new
    # phrase, not applied to a phrase that's already running, filters a
    # one-frame blip out without adding any perceptible lag to a real onset.
    MIN_PHRASE_START_S = 0.05
    min_phrase_start_frames = max(1, int(round(MIN_PHRASE_START_S * sample_rate / frame_size)))
    confirmed = np.zeros(n_frames, dtype=bool)
    run = 0
    for i, v in enumerate(is_singing_flags):
        run = run + 1 if v else 0
        confirmed[i] = run >= min_phrase_start_frames

    phrase_spans = []
    start = None
    silence_run = 0
    for i, v in enumerate(is_singing_flags):
        if v:
            if start is None and confirmed[i]:
                # Back-date the phrase's start to where the confirmed run
                # itself began, not to this frame, the debounce only delays
                # the DECISION to open a phrase, it shouldn't clip the audio
                # of a run that really was genuine singing the whole time.
                start = (i - min_phrase_start_frames + 1) * frame_size
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run > gap_tolerance_frames:
                phrase_spans.append((start, (i - silence_run + 1) * frame_size))
                start = None
                silence_run = 0
    if start is not None:
        phrase_spans.append((start, n_frames * frame_size))

    # ==== the melody ======================================================
    # Notes come from the singing itself, octave-corrected and de-jittered
    # (see synthesis/melody_voice.py). Everything below is placed on this
    # grid rather than on a generic onset detector's.
    # The audio goes into the tracker as well: when a reading insists on
    # sitting a whole octave above the register the melody is tracking, the
    # one ambiguity that decides whether the robot sings a whole phrase an
    # octave high, the tracker asks the spectrum of that frame for a second
    # opinion (see melody_voice._octave_evidence).
    corrected_hz, notes = track_notes(
        hz_curve, is_singing_flags, frame_hop_s,
        audio=vocal_for_analysis, sr=sample_rate, hop_samples=frame_size,
    )
    # Any note the tracker left a whole octave away from the melody around it
    # is put back before anything is placed on the grid (see
    # stabilize_note_octaves): measured on this project's reference track, the
    # same phrase came out at 348 Hz in one render and at 86.9 Hz in another, a
    # difference of two octaves, and a note two octaves off its neighbours is
    # exactly what a listener hears as "a second, unrelated part".
    moved_octaves = stabilize_note_octaves(notes)
    if moved_octaves:
        print(f"[dsp/render-track] {moved_octaves} note(s) sat a whole octave away "
              "from the melody around them and were put back on their neighbours' octave.")
    note_starts_s = [n.start_s for n in notes]
    note_ends_s = [n.end_s for n in notes]
    note_hz = [n.f0_hz for n in notes]
    median_note_hz = float(np.median(note_hz)) if note_hz else 0.0
    measured_f0 = [f for f in syllable_f0 if f > 0]
    # The take's own register, as the part is placed against it: the WORLD
    # analysis' own median when it ran (measured from the same frames the part
    # is sung from), otherwise the per-syllable measurements.
    base_hz = word_hz if word_hz > 0 else (float(np.median(measured_f0)) if measured_f0 else register_hz)
    octave_k = octave_placement(median_note_hz, base_hz) if (median_note_hz > 0 and base_hz > 0) else 0
    print(f"[dsp/render-track] '{vocable}': {len(notes)} notes tracked "
          f"(median {median_note_hz:.0f} Hz), vocable register {base_hz:.0f} Hz, "
          f"octave placement {octave_k:+d}")

    def melody_hz_at(t: float) -> float:
        """The sung pitch at time t, or 0.0 if no note covers it."""
        if not note_hz:
            return 0.0
        i = bisect.bisect_right(note_starts_s, t) - 1
        if i < 0:
            return note_hz[0]
        if t <= note_ends_s[i] + frame_hop_s:
            return note_hz[i]
        # a gap between tracked notes: hold the last pitch rather than
        # dropping to "no pitch" mid-phrase (the gate, not the tracker,
        # decides where the part is actually silent)
        return note_hz[i]

    # There is deliberately no note grid here any more. The grid used to be
    # built (each sung note plus the track's own onsets as re-articulations)
    # and then used to slice the part note by note; the phrase loop below
    # never slices, and melody_hz_at already answers "what is being sung at
    # time t" from the tracked notes, which is all the loop needs.
    XFADE_N = max(24, int(dsp_cfg["xfade_ms"] / 1000 * sample_rate))

    robot = np.zeros(len(audio) + int(1.0 * sample_rate), dtype=np.float32)
    content_end = 0

    for (phrase_start, phrase_end) in phrase_spans:
        phrase_start_s = phrase_start / sample_rate
        phrase_end_s = phrase_end / sample_rate

        # WHICH OCTAVE THE TAKE SINGS THIS PHRASE IN, chosen once for the
        # phrase and never per note: a per-note choice turns a smooth line
        # into a staircase of register jumps, and a register jump is audible
        # as the part changing singer in the middle of a line. The question
        # here is not "which octave avoids the shift cap" but "which octave
        # keeps this phrase closest to the take's own register", because an
        # octave of separation is a different voice to a listener's ear even
        # when every note in it is in tune. So the cost counts the notes that
        # would sit outside COMFORT_SEMITONES of the take's own pitch first,
        # then how far outside they land, then prefers the smallest departure
        # from the part's own placement.
        phrase_notes = [melody_hz_at(hz_time) for hz_time in
                        (n.start_s for n in notes if phrase_start_s < n.start_s < phrase_end_s)]
        phrase_notes = [h for h in phrase_notes if h > 0]
        phrase_octave = octave_k
        if phrase_notes and word_hz > 0:
            best_cost = None
            for m in range(-3, 4):
                offs = [abs(12.0 * float(np.log2(h / word_hz)) + 12.0 * (octave_k + m))
                        for h in phrase_notes]
                over = np.array([o - COMFORT_SEMITONES for o in offs if o > COMFORT_SEMITONES])
                cost = (int(over.size),
                        float(np.sum(over ** 2)) if over.size else 0.0,
                        0 if m == 0 else 1,
                        abs(m))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    phrase_octave = octave_k + m
            if phrase_octave != octave_k:
                print(f"[dsp/render-track] phrase at {phrase_start_s:.1f}s: singing "
                      f"{phrase_octave - octave_k:+d} octave from the part's placement "
                      f"to keep it near the take's own register ({len(phrase_notes)} notes).")

        # ==== the phrase, sung through in one pass =========================
        # WHAT PLAYS. The take is read from its own start, continuously, for as
        # long as the phrase lasts: the whole word, at its own recorded pace,
        # starting over when it runs out (with a short re-articulation at that
        # point). There is no note-by-note slicing, no per-note re-trigger and
        # no time-warping anywhere in this path, and all three of those are why
        # the word is finally audible: notes on a real track are usually
        # SHORTER than a syllable of the take (0.2-0.4 s against about half a
        # second), so a per-note fill could only ever play the first fraction
        # of each syllable, and the melody-following step that replaced it
        # re-read the audio at a varying rate, which stretched and squeezed the
        # word's own syllables as the tune moved.
        #
        # HOW ITS PITCH FOLLOWS THE MELODY. The phrase's own register goes to
        # one pitch, `base_hz`, the take's register moved by this phrase's
        # octave placement, and then the melody's movement inside the phrase
        # is applied to each frame of the take as a PITCH, not as a re-read
        # (see synthesis/world_voice.py). Every frame of the take lands on
        # exactly one frame of the output, so the word's timing stays the
        # singer's, at any pitch, and the vocal tract stays the singer's too.
        phrase_len = phrase_end - phrase_start
        if phrase_len <= 0:
            continue
        ref_hz = word_hz
        if phrase_notes and word_hz > 0:
            ref_hz = float(np.exp2(np.mean(np.log2(np.asarray(phrase_notes, dtype=np.float64)))))
        base_hz = ref_hz * float(2.0 ** phrase_octave) if (ref_hz > 0 and word_hz > 0) else 0.0

        material = None
        if performance is not None and base_hz > 0:
            # The melody's own movement inside the phrase, one value per
            # synthesis frame: where the tune is at that instant, measured
            # against the phrase's own register. Bounded, so a single wild
            # tracking reading can never drag a whole phrase somewhere
            # unrecognisable.
            frame_len = max(1, int(round(sample_rate * world_voice.FRAME_PERIOD_MS / 1000.0)))
            n_frames = max(1, int(math.ceil(phrase_len / frame_len)))
            frame_times = (phrase_start + np.arange(n_frames, dtype=np.float64) * frame_len) / sample_rate
            frame_hz = np.asarray([melody_hz_at(t) for t in frame_times], dtype=np.float64)
            safe_hz = np.where(frame_hz > 0, frame_hz, ref_hz if ref_hz > 0 else base_hz)
            curve = 1200.0 * np.log2(np.maximum(safe_hz, 1e-6) / max(ref_hz, 1e-6))
            capped = int(np.sum(np.abs(curve) > CURVE_CAP_CENTS))
            if capped:
                print(f"[dsp/render-track] phrase at {phrase_start_s:.1f}s: {capped} of "
                      f"{curve.size} frames sit more than {CURVE_CAP_CENTS / 100:.1f} "
                      "semitones from the phrase's own register and are held at the cap.")
                curve = np.clip(curve, -CURVE_CAP_CENTS, CURVE_CAP_CENTS)
            material = world_voice.sing_along(performance, sample_rate, phrase_len, base_hz, curve)
            if not np.any(material):
                material = None
        if material is None:
            # WORLD unavailable, or a phrase it could not sing: the older path,
            # one constant formant-preserved shift with the take tiled through
            # the phrase. It still puts the phrase's own register onto the take,
            # it just cannot follow the melody's movement inside it.
            shift_st = 0.0
            if word_hz > 0 and base_hz > 0:
                shift_st = 12.0 * float(np.log2(base_hz / word_hz))
                shift_st = float(np.clip(shift_st, -max_shift, max_shift))
            base = shifter.shift(0, word_take, shift_st)
            reps = int(np.ceil(phrase_len / max(len(base), 1)))
            material = np.tile(base, reps)[:phrase_len].astype(np.float32).copy()
            seam_offsets = [k * len(base) for k in range(1, reps) if k * len(base) < phrase_len]
            if seam_offsets:
                # the word starting over is a real seam between different audio,
                # so it gets a real equal-power blend rather than a bare cut
                material = _apply_internal_crossfades(material, seam_offsets, XFADE_N)
        end_sample = phrase_start + phrase_len
        if end_sample > len(robot):
            robot = np.pad(robot, (0, end_sample - len(robot)))
        robot[phrase_start:end_sample] = material
        content_end = max(content_end, end_sample)

        # A natural attack at this PHRASE's true start and a short release at
        # its true end, the phrase boundaries are where the performance
        # itself starts and stops, so these are the two places a fade belongs.
        ATTACK_S = 0.05
        RELEASE_S = 0.05
        attack_n = min(int(ATTACK_S * sample_rate), (phrase_end - phrase_start) // 2)
        release_n = min(int(RELEASE_S * sample_rate), (phrase_end - phrase_start) // 2)
        if attack_n > 1:
            robot[phrase_start:phrase_start + attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
        if release_n > 1:
            robot[phrase_end - release_n:phrase_end] *= np.linspace(1.0, 0.0, release_n, dtype=np.float32)

    robot = robot[:max(len(audio), content_end)]

    # A ~0.4 s smoothing of the dynamics curve was tried here (on the theory
    # that following the raw per-frame RMS follows the mouth's consonants
    # rather than the performance's loudness) and removed again: measured
    # against the isolated vocal's own envelope, it made the match WORSE at
    # every scale (0.46 -> 0.40 at 250 ms, 0.57 -> 0.48 at 500 ms), because
    # the syllable-level movement it removed was part of the thing being
    # followed. The raw curve stays.
    if np.any(amplitude_curve > 0):
        # The performance's own dynamics, followed gently. Following the raw
        # per-frame curve matched the track's loudness best at every scale
        # when it was measured (0.46 at 250ms, 0.57 at 500ms), so the shape of
        # the curve is never smoothed; what is limited is only how far one dip
        # may go. min_gain=0.6 with a 110ms release: a consonant or a breath
        # inside a word is a dip in the recording, and it has to stay a dip
        # rather than reading as the part being cut off, which is what the old
        # 0.08 floor and 45ms release did.
        robot = apply_amplitude_curve_follow(
            robot, sample_rate, amplitude_curve=amplitude_curve, frame_hop_s=frame_hop_s,
            min_gain=0.6, max_gain=1.8, release_ms=110.0,
        )

    # One predictable level for every vocable and every track: the takes
    # themselves differ by several dB from each other ("Moon" is 5 dB hotter
    # than "You Know You're Going to Feel It"), so without this the ROBOT
    # VOLUME slider would mean a different thing depending on which word was
    # picked. Measured over the part's own sung frames, including the
    # silence in the average would make a sparse track come out much louder
    # than a dense one, and measuring over the input track's gate flags
    # instead (as this used to) counted the moments where the gate was open
    # but this render placed nothing, which pulled the measured level down
    # and then had to be undone by the peak ceiling (see normalize_loudness).
    robot = normalize_loudness(robot, sample_rate, target_rms=float(sv_cfg.get("target_rms", 0.12)),
                               peak_ceiling=0.95)

    # NOTHING SUNG IS AN ERROR, NEVER THE INSTRUMENTS ALONE.
    #
    # If the melody pass above found nothing to sing, an upload whose separator
    # could isolate no vocal, or a clip too short to track, the take comes out
    # empty, and this used to send it anyway: with "Sing along with
    # instruments" On that was a two-channel render whose SECOND channel was
    # digital silence, so what the page played was channel 0, the track's own
    # accompaniment with its lead vocal taken out. "Only instrument noise and
    # none of the words" is the one answer this panel must never give, and with
    # the toggle Off it was a silent file of the right length instead. Both are
    # refused here, and the page prints the reason under the panel rather than
    # playing anything.
    take_peak = float(np.max(np.abs(robot))) if robot.size else 0.0
    if take_peak < NOTHING_SUNG_PEAK:
        raise NothingToSingError(
            "nothing sung was found in this track, so there is no part to sing "
            "along with. The word is sung along the track's own melody, and no "
            "melody was detected in the vocal that was separated out of it."
        )

    # WHAT IS ACTUALLY SENT BACK: TWO CHANNELS.
    #
    # Channel 0 is the track with its own original singer removed (the
    # accompaniment from the separation above). Channel 1 is the robot
    # singing the take along the melody. The browser plays them together,
    # under its two existing volume sliders, and never plays the ORIGINAL
    # track over the top, which is the fix for the oldest and loudest
    # complaint about this mode: "you can hear the original and then the
    # synthesis", "three different audios", "you cannot even hear the words
    # lalala". All three had one cause, and it was not the synthesis at all:
    # the page was playing the uploaded track, original lead vocal included,
    # at the same time as the robot. Two lead voices on one melody, one of
    # them the take's own recording, which is what the ear locks onto, and
    # whatever the robot sang was masked, doubled, or dragged out of tune by
    # the original sitting on top of it. Removing the original vocal from
    # what plays leaves the one thing this feature is supposed to be: the
    # track's instruments, and a singer who sounds like the person who
    # recorded the take.
    #
    # Channel 0 leaves here at unity, the page applies its own TRACK slider to
    # it, so baking that slider in here would apply it twice, and the two
    # channels are trimmed TOGETHER, by one shared gain, only if the pair summed
    # at the demo's own default slider positions (TRACK 78%, ROBOT 100%) would
    # clip. One shared gain rather than two separate ones is what keeps the
    # balance between the track and the voice exactly as designed, and it is
    # only ever a fraction of a dB: the voice's own level is already one
    # predictable number for every vocable and every track (normalize_loudness
    # above), so what the trim is really fitting is the swept peak of the mix.
    # "SING ALONG WITH INSTRUMENTS: OFF" MEANS THE TAKE ALONE, HERE TOO.
    # This path used to ignore the toggle completely, so Off came back as the
    # same two channels as On: the band, plus the take on top of it. On a
    # drum-led track that is the one thing the setting exists to stop, the
    # accompaniment playing with the words either buried under it or, in the
    # stretches where this render places nothing, playing on its own with no
    # words at all. The separation itself still runs when the toggle is Off,
    # because its vocal half is what the melody, the rhythm and the singing
    # gate are all read from; it is only the accompaniment half that stops
    # being sent back. What is sent is then the take on its own, one channel,
    # and the page plays the uploaded file itself underneath it at TRACK
    # VOLUME, so the track is in the mix in both positions of the toggle and
    # that slider is live in both. Off means the take arrives with no
    # accompaniment of its own, not that the file that was uploaded goes
    # missing from what is heard.
    # The movement of real folds on the sung take, applied once, here, after the
    # take has been placed, levelled and normalised and before anything is
    # packaged (see _apply_vocal_vibration_amount).
    robot = _apply_vocal_vibration_amount(robot, sample_rate, vibrato_amount)

    if accompaniment is None or not instruments_enabled:
        out = np.asarray(robot, dtype=np.float32)
        if accompaniment is None:
            print("[dsp/render-track] no separated accompaniment for this render, sending the "
                  "voice alone, to be played over the upload.")
        else:
            print("[dsp/render-track] instruments off: sending the sung take alone, one channel, "
                  "the track's own accompaniment is not in this render; the page plays the "
                  "upload under it at TRACK VOLUME.")
    elif (accompaniment.shape[-1] if accompaniment.ndim == 2 else len(accompaniment)) >= len(audio):
        # The accompaniment is kept in the model's two channels (see
        # _separate_stems); this render carries it as ONE channel beside the
        # sung take, so it is averaged here.
        bed = np.asarray(_mono_downmix(accompaniment[..., :len(audio)] if accompaniment.ndim == 2
                                       else accompaniment[:len(audio)]), dtype=np.float64)
        robot_ch = np.asarray(robot, dtype=np.float64)
        if len(bed) < len(robot_ch):
            bed = np.pad(bed, (0, len(robot_ch) - len(bed)))
        bed = bed[:len(robot_ch)]
        # Lead-vocal ducking: the accompaniment steps back while the take
        # sings. Applied BEFORE the peak is measured, so the trim below fits
        # what is really sent. See _duck_accompaniment_under_voice.
        bed = _duck_accompaniment_under_voice(bed, robot_ch, sample_rate, max_db=DUCK_MAX_DB)
        default_peak = (float(np.max(np.abs(DEFAULT_DEMO_TRACK_VOLUME * bed + robot_ch)))
                        if robot_ch.size else 0.0)
        mix_trim = 1.0
        if default_peak > 0.95 and default_peak > 0:
            mix_trim = 0.95 / default_peak
            bed *= mix_trim
            robot_ch *= mix_trim
            default_peak *= mix_trim
        out = np.stack([bed.astype(np.float32), robot_ch.astype(np.float32)], axis=1)
        print(f"[dsp/render-track] sending the accompaniment and the sung take as two "
              f"channels (accompaniment ducked by up to {DUCK_MAX_DB:.1f} dB under the voice, "
              f"mix trimmed {mix_trim:.3f}, default-slider mix peaks at "
              f"{default_peak:.3f}), the original lead vocal is not in what the page plays.")
    buf = io.BytesIO()
    sf.write(buf, out, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()

# THE SAME HUM IS NOT RENDERED TWICE.
#
# A hum render is the slowest thing in the panel by a wide margin -- the melody
# is tracked over the whole excerpt with pyin, then the murmur is synthesized
# sample by sample -- and the panel asks for the same one over and over: seven
# live chunks on a fifty second track, and then all seven again the moment
# Play is pressed a second time, because Stop clears the page's own hold on
# them. Measured on this project's own reference track, a live hum chunk costs
# 20-40 s of backend time on a loaded machine, so a second listen was another
# two minutes of a machine that had already rendered exactly that audio. The
# render is deterministic (the hum voice seeds its own rng per note, see
# _render_hum_ensemble), so the answer for a given input and settings is
# always the same bytes, and it is held here against the content hash of the
# material plus the settings that shape it. Bounded, oldest out first: this is
# a replay cache, not a library.
_RENDER_MEMO_LOCK = threading.Lock()
_RENDER_MEMO: "OrderedDict[tuple, bytes]" = OrderedDict()
_RENDER_MEMO_MAX = 8


def _hum_render_key(raw_audio_bytes: bytes, texture: str, voice_index: int,
                    instruments_enabled: bool, transpose_semitones,
                    vibrato_amount) -> tuple:
    import hashlib
    return ("hum", hashlib.sha1(raw_audio_bytes).hexdigest(), str(texture),
            int(voice_index or 0), bool(instruments_enabled),
            None if transpose_semitones is None else round(float(transpose_semitones), 3),
            None if vibrato_amount is None else round(float(vibrato_amount), 3))


def _memoized_hum_render(key: tuple, render):
    """Run `render` unless these exact bytes and settings were rendered already."""
    with _RENDER_MEMO_LOCK:
        hit = _RENDER_MEMO.get(key)
        if hit is not None:
            _RENDER_MEMO.move_to_end(key)
            return hit, True
    out = render()
    with _RENDER_MEMO_LOCK:
        _RENDER_MEMO[key] = out
        _RENDER_MEMO.move_to_end(key)
        while len(_RENDER_MEMO) > _RENDER_MEMO_MAX:
            _RENDER_MEMO.popitem(last=False)
    return out, False


def _render_track_offline(raw_audio_bytes: bytes, pitch_method: str, texture: str, voice_index: int,
                           mode_override: str | None = None, instruments_enabled: bool = True,
                           transpose_semitones: float | None = None, apply_neural: bool = True,
                           delay_ms: float | None = None, echo_amount: float | None = None,
                           contour_baseline_db: float | None = None,
                           contour_max_bright_db: float | None = None,
                           contour_max_dark_db: float | None = None,
                           vocable: str | None = None,
                           vibrato_amount: float | None = None) -> bytes:
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector
    from analysis.rhythm_analyzer import RhythmAnalyzer
    from analysis.phonetic_analysis import CreeTokenizer
    from synthesis.harmony_engine import HarmonyEngine
    from synthesis.vocable_synthesizer import (
        VocableSynthesizer, apply_pitch_curve_follow, apply_amplitude_curve_follow,
    )
    import librosa

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]

    mcfg = cfg.get("modes", {})
    effective_delay_ms = float(delay_ms) if delay_ms is not None else float(mcfg.get("delay_echo_ms", 350.0))
    effective_echo_feedback = (float(echo_amount) / 100.0) if echo_amount is not None \
        else float(mcfg.get("delay_echo_feedback", 0.45))

    # See the frame loop's own comment for why the robot asks a stricter
    # question of the pitch detector than the live pipeline does.
    robot_pitch_confidence_min = float((cfg.get("pitch", {}) or {}).get(
        "robot_confidence_threshold", 0.45))
    weak_pitch_frames = 0

    synth_cfg_for_contour = cfg.get("synthesis", {})
    _default_baseline_shift = float(synth_cfg_for_contour.get("contour_baseline_brightness_shift", 0.18))
    _default_brightness_amount = float(synth_cfg_for_contour.get("contour_brightness_amount", 0.45))
    effective_contour_baseline_db = float(contour_baseline_db) if contour_baseline_db is not None \
        else _default_baseline_shift * 10.0
    effective_contour_max_bright_db = float(contour_max_bright_db) if contour_max_bright_db is not None \
        else effective_contour_baseline_db + _default_brightness_amount * 10.0
    effective_contour_max_dark_db = float(contour_max_dark_db) if contour_max_dark_db is not None \
        else max(0.0, _default_brightness_amount * 10.0 - effective_contour_baseline_db)

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

    # NOTE: a locked single vocable ("one word, not Mix") is no longer
    # handled in this function at all, see
    # _render_track_single_vocable_offline and its own docstring for why
    # this note-by-note path can't be made to work for a multi-second
    # recorded word, and render_dsp_track (the /api/dsp/render-track
    # handler) for where that path is chosen instead. This function only
    # ever runs in normal "Mix" accompaniment mode now.
    # call_and_response is composed as a whole, not decided frame by frame:
    # its answers are phrases, placed in the pauses the singer leaves, which is
    # a question about the track's timeline rather than about this frame. So the
    # usual per-frame note machinery below is skipped for it, and what this loop
    # collects instead is the raw timeline (time, sung pitch, loudness) that
    # synthesis/call_response.py analyses and composes against.
    call_response_mode = False
    humming_mode = False
    if mode_override:
        harmony.set_forced_mode(mode_override)
        try:
            from synthesis.accompaniment_modes import AccompanimentMode as _AccMode
            _forced = _AccMode(mode_override)
            call_response_mode = _forced == _AccMode.CALL_RESPONSE
            humming_mode = _forced == _AccMode.HUM
        except ValueError:
            call_response_mode = False
    cr_times: list[float] = []
    cr_pitch: list[float] = []
    cr_rms: list[float] = []

    harmony.set_contour_brightness_db(
        effective_contour_baseline_db, effective_contour_max_bright_db, effective_contour_max_dark_db
    )

    global synthesizer
    if synthesizer is None:
        raise RuntimeError("Synthesizer not loaded - check server startup logs")
    synth = synthesizer
    harmony.set_synthesizer(synth)

    n_frames = len(audio) // frame_size
    frame_hop_s = frame_size / sample_rate
    robot = np.zeros(len(audio) + int(4.0 * sample_rate), dtype=np.float32)

    # HUMMING COMPOSES NOTHING FRAME BY FRAME. A hum is one continuous line with
    # breath and movement in it, so it is rendered as a whole by the hum voice
    # (synthesis/hum_voice.py) rather than decided note by note through the
    # accompaniment engine, and the per-frame machinery below -- the note runs,
    # the vocable choice, the pitch decisions -- does not run for it at all. The
    # loop is skipped rather than the work being duplicated: nothing in it
    # touches the hum path.
    if humming_mode:
        n_frames = 0

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
            # A HESITATION IS NOT A MELODY. The detector's own confidence gate is
            # deliberately low (0.3) because it is shared with the live pipeline,
            # where a fraction of a second's lag matters more than a wobble. On a
            # track, a reading that weak is usually breath, a consonant, or a
            # voice buried in the mix -- and taking it moves the robot to a note
            # the singer never sang, which is exactly what a listener hears as
            # the synthesis falling apart on unclear singing while it sounds fine
            # on a clearly sung line. Below the threshold the frame is treated as
            # "no new pitch": the engine holds the note it is on (see
            # HarmonyEngine.decide's own hold-through-a-gap behaviour), so an
            # unclear stretch keeps singing the note the phrase was on instead of
            # chasing noise.
            if (hz and conf >= robot_pitch_confidence_min
                    and not (percussive_spans and _in_percussive_span(i * frame_size / sample_rate))):
                archer_hz = hz
            elif hz and conf < robot_pitch_confidence_min:
                weak_pitch_frames += 1
            phoneme_profile = cree.analyze(clean)
        else:
            pitch.reset()

        # RMS of this frame's own (gated/cleaned) input, one reading per frame,
        # same cadence as the pitch curve above, computed before the decision
        # because call and response wants the timeline and the accompaniment
        # modes want it in the run they build.
        frame_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))) if is_voiced else 0.0

        if call_response_mode:
            cr_times.append(i * frame_hop_s)
            cr_pitch.append(archer_hz if archer_hz else float("nan"))
            cr_rms.append(frame_rms)
            continue

        phrase = rhythm.phrase_state
        if archer_hz and phrase in ("silence", "phrase_end"):
            phrase = "singing"

        decision = harmony.decide(
            archer_hz=archer_hz, phrase_state=phrase,
            tempo_bpm=rhythm.current_tempo, phoneme_profile=phoneme_profile,
        )

        # (frame_rms is measured above, before the decision: see its comment.)

        if decision.action == "sing":
            current_run = {"start_frame": i, "decision": decision, "n_frames": 1,
                            "pitch_curve": [decision.target_hz],
                            "amplitude_curve": [frame_rms]}
            runs.append(current_run)
        elif decision.action == "sustain":
            if current_run is None:
                current_run = {"start_frame": i, "decision": decision, "n_frames": 1,
                                "sustained": True,
                                "pitch_curve": [decision.target_hz],
                                "amplitude_curve": [frame_rms]}
                runs.append(current_run)
            else:
                current_run["sustained"] = True
                current_run["n_frames"] += 1
                if decision.target_hz is not None and decision.target_hz > 0:
                    current_run["pitch_curve"].append(decision.target_hz)
                current_run["amplitude_curve"].append(frame_rms)
        else:
            current_run = None

    SAME_WORD_CROSSFADE_S = 0.03
    WORD_CHANGE_CROSSFADE_S = 0.09
    GAP_FADE_IN_S = 0.008  # just long enough to kill a true zero-to-full-amplitude click after real silence
    MAX_RUN_SECONDS = 12.0  # safety cap so one stuck drone/note can't blow up render time/memory

    content_end = 0  # furthest sample index any run actually wrote audio into
    prev_vocable: str | None = None
    join_samples = int(SAME_WORD_CROSSFADE_S * sample_rate)
    # A sustained note, a drone most of all, is ONE run for as long as it
    # is held, which on a real track means the whole song. MAX_RUN_SECONDS is
    # there to bound the size of a single synthesizer call, so it has to bound
    # THE CALL, not the note: applying it to the note is what made a forced
    # drone render its first 12 seconds and then fall completely silent for
    # the remaining 23 of a 35 second track, no drone under two thirds of
    # the song, which reads as the mode being broken rather than as a drone at
    # all. A long run is therefore continued in chunks, each one bounded by
    # the cap, stitched with the same same-word crossfade every other join
    # here uses. The synthesizer already distinguishes the two cases for us:
    # legato=True is exactly "this chunk is a mid-note continuation, not a
    # real onset and not a real ending" (see VocableSynthesizer._apply_envelope,
    # which keeps its full attack and release for chunk 0 only), so the drone
    # stays one continuous tone instead of pulsing once per chunk.
    for run in runs:
        decision = run["decision"]
        run_seconds = run["n_frames"] * frame_hop_s
        # A sustained run may be as long as the input itself and no longer, so
        # bounding it there is still a bound: it just is not a bound that is
        # smaller than the track.
        cap_seconds = (len(audio) / sample_rate + 2.0) if run.get("sustained") else MAX_RUN_SECONDS
        n_chunks = max(1, int(np.ceil(run_seconds / cap_seconds)))
        frames_per_chunk = max(1, int(round(run_seconds / n_chunks / frame_hop_s)))
        written = 0  # samples of this run already placed, by ACTUAL rendered length
        for chunk_i in range(n_chunks):
            first = chunk_i * frames_per_chunk
            last = run["n_frames"] if chunk_i == n_chunks - 1 else (chunk_i + 1) * frames_per_chunk
            if last <= first:
                continue
            chunk = dataclasses.replace(decision, duration_s=(last - first) * frame_hop_s)
            scratch = synth.synthesize(chunk, blocking=True, apply_crossfade=False,
                                       legato=(chunk_i > 0))

            pitch_all, amp_all = run["pitch_curve"], run["amplitude_curve"]
            if len(pitch_all) == run["n_frames"] and len(amp_all) == run["n_frames"]:
                pitch_part, amp_part = pitch_all[first:last], amp_all[first:last]
            else:
                # A frame with no target pitch appends nothing, so these curves
                # do not always hold one value per frame. Slicing them per chunk
                # would then slide them out of time against the audio, so they
                # are applied to the first chunk only, which is everything a
                # single-chunk run ever did.
                pitch_part, amp_part = (pitch_all, amp_all) if chunk_i == 0 else ([], [])

            pitch_curve = np.asarray(pitch_part, dtype=np.float64)
            if pitch_curve.size >= 2 and chunk.target_hz:
                scratch = apply_pitch_curve_follow(
                    scratch, sample_rate, base_hz=chunk.target_hz,
                    pitch_curve_hz=pitch_curve, frame_hop_s=frame_hop_s,
                )

            amplitude_curve = np.asarray(amp_part, dtype=np.float64)
            if amplitude_curve.size >= 2:
                scratch = apply_amplitude_curve_follow(
                    scratch, sample_rate, amplitude_curve=amplitude_curve, frame_hop_s=frame_hop_s,
                )
            scratch = np.asarray(scratch, dtype=np.float32).copy()

            # Continuing a run overlaps the previous chunk by one crossfade so
            # the join is a blend rather than a butt splice; the run's first
            # chunk starts exactly at the run's own frame position.
            start_sample = run["start_frame"] * frame_size + written - (join_samples if chunk_i > 0 else 0)
            end_sample = start_sample + len(scratch)
            if end_sample > len(robot):
                robot = np.pad(robot, (0, end_sample - len(robot)))

            vocable_changed = prev_vocable is not None and getattr(chunk, "vocable", None) != prev_vocable
            overlap = content_end - start_sample

            if overlap > 0:
                cf = int((WORD_CHANGE_CROSSFADE_S if vocable_changed else SAME_WORD_CROSSFADE_S) * sample_rate)
                cf = max(0, min(cf, overlap, len(scratch)))
                if cf > 0:
                    if vocable_changed:
                        t = np.linspace(0, np.pi / 2, cf)
                        fade_in, fade_out = np.sin(t), np.cos(t)
                    else:
                        fade_in = np.linspace(0, 1, cf)
                        fade_out = np.linspace(1, 0, cf)
                    existing = robot[start_sample:start_sample + cf]
                    robot[start_sample:start_sample + cf] = existing * fade_out + scratch[:cf] * fade_in
                    robot[start_sample + cf:end_sample] = scratch[cf:]
                else:
                    robot[start_sample:end_sample] = scratch
            else:
                gap_fade = min(int(GAP_FADE_IN_S * sample_rate), len(scratch))
                if gap_fade > 0 and start_sample > 0:
                    scratch[:gap_fade] *= np.linspace(0.0, 1.0, gap_fade)
                robot[start_sample:end_sample] = scratch

            content_end = max(content_end, end_sample)
            prev_vocable = getattr(chunk, "vocable", None)
            written += len(scratch)

    robot = robot[:max(len(audio), content_end)]

    hum_part_is_final = False
    if humming_mode:
        # THE HUM ITSELF, for the whole track, and the SAME voice in both
        # engines: the closed-mouth murmur, rendered straight by
        # _render_hum_ensemble. It used to be handed to the trained voice in the
        # Neural engine -- render the hum, convert it, then try to put its closed
        # mouth back -- and measured on this project's own material that is not
        # something a singing converter can do: it answers a hum with a vowel, a
        # formant or two louder between 500 Hz and 1.6 kHz, a broadband tail out
        # to 4 kHz no murmur has, and sometimes a re-sung pitch a semitone and a
        # half off the note it was handed. So the murmur IS the hum here too, and
        # the part this mode renders is finished audio, not material for the
        # converter (see hum_part_is_final below).
        from synthesis.hum_voice import HumConfig
        from synthesis.accompaniment_modes import (AccompanimentMode as _HumMode,
                                                  DEFAULT_TEXTURE as _HUM_DEFAULT_TEXTURE)
        hum_cfg = HumConfig.from_config(cfg)
        # WHICH TEXTURE THE HUM IS SUNG IN. Read from the same place the
        # answering phrases read theirs, so a hum in Duet really is two mouths
        # and a hum in Choir really is a section (see _render_hum_ensemble).
        hum_texture = harmony._texture_override or _HUM_DEFAULT_TEXTURE.get(_HumMode.HUM)
        hum_layer = harmony.texture_params.get(hum_texture)
        hum_voices = int(getattr(hum_layer, "num_voices", 1))
        hum_detune = float(getattr(hum_layer, "detune_spread_cents", 0.0))
        hum_jitter = float(getattr(hum_layer, "timing_jitter_ms", 0.0))
        robot = _render_hum_ensemble(audio, sample_rate, hum_cfg, hum_voices,
                                     hum_detune, hum_jitter)
        hum_part_is_final = True

        # ONE LEVEL FOR EVERY MODE. The hum renders through its own voice, not
        # through the note-based part above, so it never got that part's level
        # treatment and came back where its own renderer left it: measured
        # through the panel's own endpoint, a hum sat at 0.100 RMS against
        # 0.14-0.24 for every other mode, and peaked at 0.276 against 0.83-0.89,
        # i.e. the ROBOT VOLUME slider meant about 6 dB less in this one mode,
        # and the hum read as "quieter and further away" than the singing.
        # Matching RMS rather than peak is deliberate: a hum is continuous where
        # a sung line breathes, so peak-matching a drone makes it the loudest
        # thing in the panel. This puts the same average energy into the mix as
        # every other mode does, and the peak ceiling below keeps the headroom
        # honest. Only ever upward, so a hum that is already there is left alone.
        hum_rms = float(np.sqrt(np.mean(np.square(robot, dtype=np.float64)))) if robot.size else 0.0
        if 0.0 < hum_rms < HUM_RMS_TARGET:
            gain = HUM_RMS_TARGET / hum_rms
            peak_now = float(np.max(np.abs(robot)))
            if peak_now * gain > 0.95:
                gain = 0.95 / max(peak_now, 1e-9)
            robot = robot * gain
            print(f"[hum] brought to the level every other mode renders at: RMS "
                  f"{hum_rms:.3f} -> {np.sqrt(np.mean(np.square(robot, dtype=np.float64))):.3f} "
                  f"({20 * math.log10(gain):+.1f} dB), peak now {float(np.max(np.abs(robot))):.2f}")

    if call_response_mode:
        # The whole mode, composed against the track instead of guessed at per
        # frame (see synthesis/call_response.py and its own long docstring for
        # the musical rules, all of them read out of the practice rather than
        # invented here). What came back is the robot's part: the answering
        # phrases, placed in the pauses the singer actually left, and nothing at
        # all anywhere else. There is no accompaniment under the call, because
        # this mode's whole identity is that it does not sing over the singing.
        from synthesis.accompaniment_modes import AccompanimentMode, DEFAULT_TEXTURE, VoiceTexture
        from synthesis.call_response import CallResponseCfg, render_call_response

        texture_enum = harmony._texture_override or DEFAULT_TEXTURE.get(
            AccompanimentMode.CALL_RESPONSE, VoiceTexture.SOLO)
        layer = harmony.texture_params.get(texture_enum)
        cr_cfg = CallResponseCfg.from_config(cfg)
        part, answers, cr_track = render_call_response(
            np.asarray(cr_times), np.asarray(cr_pitch), np.asarray(cr_rms),
            frame_hop_s, sample_rate, total_samples=len(audio),
            cfg=cr_cfg, synthesizer=synth, tempo_bpm=rhythm.current_tempo,
            voices=layer.num_voices, detune_spread_cents=layer.detune_spread_cents,
            timing_jitter_ms=layer.timing_jitter_ms, reverb_amount=layer.reverb_amount,
        )
        robot = part
        if not answers:
            # Two different refusals, because they are two different things to be
            # told, and neither of them is "the render failed".
            if cr_track.calls:
                raise NothingToSingError(
                    "Archer never paused long enough for the robot to answer. Call and "
                    "response needs a breath between phrases, and this track is sung "
                    "straight through without one."
                )
            raise NothingToSingError(
                "nothing sung was found in this track, so there is no phrase for the "
                "robot to answer. No melodic material was detected in it."
            )

    # THE MOVEMENT IN EVERY HELD NOTE, for every mode, in both engines.
    #
    # Applied here, deliberately: after the whole part exists (so it does not
    # care which mode built any of it, or how the notes were joined), and BEFORE
    # the neural pass below (so a converted part is handed a voice that already
    # has the folds' movement in it, and the model carries the movement rather
    # than having it pasted on afterwards). Zero means zero: a straight tone is
    # left exactly as rendered.
    robot = _apply_vocal_vibration_amount(robot, sample_rate, vibrato_amount)
    if vibrato_amount and float(vibrato_amount) > 0:
        print(f"[render-track] vocal fold vibration at {float(vibrato_amount):.0f}% on the "
              f"whole part, every mode")
    if weak_pitch_frames:
        pct = 100.0 * weak_pitch_frames / max(1, n_frames)
        print(f"[render-track] held the current note on {weak_pitch_frames} frame(s) "
              f"({pct:.1f}% of the track) whose pitch reading was under "
              f"{robot_pitch_confidence_min:.2f} confidence, instead of following it.")

    peak = float(np.max(np.abs(robot))) if robot.size else 0.0
    if peak > 1.0:
        robot = robot / peak  # avoid clipping where overlapping/sustained notes summed above 0dBFS

    # The same refusal the single-word render makes, for the same reason: an
    # empty part is not a render. Mix builds its part from the accompaniment
    # engine over the whole track, so this fires when that engine placed
    # nothing at all, and it fires BEFORE the neural pass below, so a track with
    # no sung material never spends a model conversion on silence either.
    if peak < NOTHING_SUNG_PEAK:
        raise NothingToSingError(
            "nothing sung was found in this track, so there is no part for the "
            "robot to sing. No melodic material was detected in it."
        )

    if mode_override == "delayed_response":
        robot = apply_echo_effect(robot, sample_rate, effective_delay_ms, effective_echo_feedback)

    if not apply_neural or hum_part_is_final:
        out_buf = io.BytesIO()
        sf.write(out_buf, robot.astype(np.float32), sample_rate, format="WAV")
        return out_buf.getvalue()

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(robot, sample_rate, voice_index=voice_index, timeout_s=timeout_s,
                                                transpose_semitones=transpose_semitones)
    if converted is None:
        raise RuntimeError(
            "Neural sidecar conversion failed or timed out, check that neural_env/rvc_server.py "
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
    delay_ms: float | None = Form(None),
    echo_amount: float | None = Form(None),
    contour_baseline_db: float | None = Form(None),
    contour_max_bright_db: float | None = Form(None),
    contour_max_dark_db: float | None = Form(None),
    vocable: str = Form("mix"),
    vibrato_amount: float | None = Form(None),
):
    try:
        raw = await file.read()
        single_vocable = (vocable or "").strip().lower()
        # THE MODE WINS OVER THE LOCKED WORD. A hum has no word in it, so
        # mode=solo_humming with a vocable set is not a locked-word render with a
        # different label -- it used to take this first branch and hand back that
        # one recorded word sung along the melody, every part of it the wrong
        # voice. Humming is the whole-track path below (see _render_track_offline
        # and synthesis/hum_voice.py), so the mode is asked here, once, before the
        # vocable decides anything. The page asks the same question the same way
        # (demoSingleVocable in frontend/index.html), so the panel and the server
        # cannot disagree about which renderer a Humming render is.
        if single_vocable and single_vocable != "mix" and not _is_hum_mode(mode):
            # One locked word: that one recording, sung along the track's
            # own melody and rhythm, see
            # _render_track_single_vocable_offline's own docstring. (It used
            # to just loop the word at whatever pitch it was recorded at;
            # this is the renderer that replaced that.)
            wav_bytes = await asyncio.to_thread(
                _render_track_single_vocable_offline, raw, pitch_method, single_vocable,
                instruments_enabled, vibrato_amount,
            )
        else:
            # "Mix" (or any mode of its own, humming included): the accompaniment
            # engine, rendered offline for the whole
            # track, one pass, the same shape as the single-word path above
            # and the same shape the live engine feeds note by note. It used to
            # write silence here "by explicit request", which made the panel's
            # own Play button produce nothing at all on the default vocable:
            # the uploaded track played and the robot's part was an empty file.
            #
            # What comes back is the robot's part alone, one channel, which is
            # what this signal asks the page for: a one-channel render means
            # "this is the voice only", so the page plays the upload underneath
            # it and puts ROBOT VOLUME on the render and TRACK VOLUME on the
            # file. The two-channel stem shape belongs to the single-word path
            # and to the neural one, where the render carries the instruments
            # with it. apply_neural is False because this is the DSP engine:
            # its output is the real synthesis, never the trained voice.
            def _render_dsp_part():
                return _render_track_offline(
                    raw,
                    pitch_method,
                    texture,
                    0,                      # voice_index: unused without the neural pass
                    mode or None,           # mode_override
                    instruments_enabled,
                    None,                   # transpose_semitones: the DSP path is never re-pitched
                    False,                  # apply_neural
                    delay_ms,
                    echo_amount,
                    contour_baseline_db,
                    contour_max_bright_db,
                    contour_max_dark_db,
                    None,                   # vocable: Mix is the engine's own choice per phrase
                    vibrato_amount,
                )

            # A hum is the one render here that costs a whole melody tracking pass
            # (see _RENDER_MEMO), so the DSP engine holds its hums too: play the
            # same track in the same mode twice and the second one is instant.
            if _is_hum_mode(mode):
                wav_bytes, from_memo = await asyncio.to_thread(
                    _memoized_hum_render,
                    _hum_render_key(raw, texture, 0, instruments_enabled, None, vibrato_amount),
                    _render_dsp_part,
                )
                if from_memo:
                    print("[dsp hum] this track's hum is already rendered, playing it back "
                          "rather than rendering it again")
            else:
                wav_bytes = await asyncio.to_thread(_render_dsp_part)
        return Response(content=wav_bytes, media_type="audio/wav")
    except NothingToSingError as e:
        # A refusal rather than a failure, and it is returned with the sentence
        # itself as the error, so the panel prints it under the settings instead
        # of playing a track with nothing of the robot in it.
        print(f"[dsp render-track] refused: {e}")
        return JSONResponse({"error": str(e)}, status_code=422)
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
    delay_ms: float | None = Form(None),
    echo_amount: float | None = Form(None),
    contour_baseline_db: float | None = Form(None),
    contour_max_bright_db: float | None = Form(None),
    contour_max_dark_db: float | None = Form(None),
    vibrato_amount: float | None = Form(None),
):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready, check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )

    try:
        raw = await file.read()

        def _render_neural_part():
            return _render_track_offline(
                raw, pitch_method, texture, voice_index,
                mode_override=(mode or None), instruments_enabled=instruments_enabled,
                transpose_semitones=transpose_semitones,
                delay_ms=delay_ms, echo_amount=echo_amount,
                contour_baseline_db=contour_baseline_db,
                contour_max_bright_db=contour_max_bright_db,
                contour_max_dark_db=contour_max_dark_db,
                vibrato_amount=vibrato_amount,
            )

        # A hum here is the murmur itself (see _render_track_offline's hum block),
        # so it is deterministic and worth holding on to: see _RENDER_MEMO.
        if _is_hum_mode(mode):
            wav_bytes, from_memo = await asyncio.to_thread(
                _memoized_hum_render,
                _hum_render_key(raw, texture, voice_index, instruments_enabled,
                                transpose_semitones, vibrato_amount),
                _render_neural_part,
            )
            if from_memo:
                print("[neural hum] this track's hum is already rendered, playing it back "
                      "rather than rendering it again")
        else:
            wav_bytes = await asyncio.to_thread(_render_neural_part)
        return Response(content=wav_bytes, media_type="audio/wav")
    except NothingToSingError as e:
        # The same refusal the DSP endpoint makes, in the same shape, because it
        # is a refusal and not a failure: under call and response it is how the
        # panel is told this track has no pause to answer into, or nothing sung
        # in it at all.
        print(f"[neural render-track] refused: {e}")
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        print(f"[neural track render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


_demucs_model = None
_demucs_lock = threading.Lock()


def _get_demucs_model():
    global _demucs_model
    if _demucs_model is not None:
        return _demucs_model
    with _demucs_lock:
        if _demucs_model is None:
            from demucs.pretrained import get_model
            print("[demucs] loading htdemucs source-separation model (first call only, "
                  "downloads weights if not already cached)...")
            _demucs_model = get_model("htdemucs")
            _demucs_model.eval()
            print("[demucs] htdemucs ready.")
    return _demucs_model


def _separate_vocal(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The isolated vocal. Thin wrapper over _separate_stems (see it for why
    the accompaniment is available too) for every caller that only needs the
    vocal."""
    vocal, _bed = _separate_stems(audio, sample_rate)
    return vocal


def _separate_stems(audio: np.ndarray, sample_rate: int):
    """Split a track into (isolated vocal, accompaniment) in ONE model run.

    Demucs is run once and both halves are kept, because the same separation
    answers two different questions on this project: which frames a person is
    actually singing (the vocal, what the offline single-vocable render
    analyses), and what the track sounds like WITHOUT that person (the
    accompaniment, what the singer is finally heard over; see the render's
    own comment for why playing the original track underneath the robot is
    what made the result sound like two or three unrelated things at once).

    The accompaniment is the model's own estimate of the other stems summed
    (drums + bass + other), not the input minus the vocal: subtraction leaves
    the separation's own error in the mix as a vocal-ish smear, while the
    model's other stems are what it actually believes the instruments are.

    The TWO HALVES HAVE DIFFERENT SHAPES ON PURPOSE, because they are used for
    two different things. The vocal is mono: every consumer of it is either a
    voice model that resamples to 16 kHz and works in mono, or a measure of
    where the singing is. The accompaniment keeps the model's own two channels:
    it is the half of the track that is ever PLAYED (see
    /api/neural/backing-bed/track), and collapsing a stereo mix to mono to play
    it under the robot is heard immediately as a narrower, cheaper backing --
    the width comes back with the channels, at the cost of the second channel's
    worth of memory, which is why the cache below is bounded.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = _get_demucs_model()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, n) mono
    if wav.shape[0] == 1:
        wav = wav.expand(2, -1)  # demucs expects stereo-shaped input; duplicate mono to both channels
    wav = convert_audio(wav, sample_rate, model.samplerate, model.audio_channels)

    # SHIFTS AND OVERLAP, and they are not defaults by accident. Demucs averages
    # its estimate over a random time shift (shifts) and over overlapping
    # windows (overlap); the library's defaults are shifts=0, overlap=0.25,
    # i.e. the cheapest, least-averaged estimate it can make. Measured here on
    # the panel's own demo track, shifts=1/overlap=0.5 costs 1.5x the default
    # (0.53 -> 0.78x real time, still a background job that finishes long
    # before a person has finished listening to the first verse) and it is the
    # difference the two halves of this function are FOR: the vocal stem, which
    # is what the trained voice is fed in every live chunk, comes back with the
    # percussion and the instruments further down, and the accompaniment comes
    # back cleaner. shifts=2/overlap=0.75 measured 3.0x real time (12 minutes on
    # a four-minute song) and its stems were within 0.1 dB of these, so it buys
    # nothing for the wait. Both halves are used on every render, the vocal as
    # the model's input and the bed as the track the page plays, so the cost is
    # paid once per upload and not once per chunk.
    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0), device="cpu", progress=False,
                              shifts=1, overlap=0.5)[0]

    source_names = model.sources  # typically ["drums", "bass", "other", "vocals"]
    vocals_idx = source_names.index("vocals")
    vocal_stem = sources[vocals_idx].mean(dim=0).cpu().numpy()
    bed_stem = sum(sources[i] for i in range(len(source_names)) if i != vocals_idx).cpu().numpy()
    # (channels, n). One channel in, one channel out of the consumers that only
    # want the bed as a single channel; the cache keeps both channels.
    if bed_stem.ndim == 2 and bed_stem.shape[0] == 1:
        bed_stem = bed_stem[0]

    if model.samplerate != sample_rate:
        import librosa
        vocal_stem = np.asarray(librosa.resample(vocal_stem, orig_sr=model.samplerate,
                                                target_sr=sample_rate, axis=-1), dtype=np.float32)
        bed_stem = np.asarray(librosa.resample(bed_stem, orig_sr=model.samplerate,
                                               target_sr=sample_rate, axis=-1), dtype=np.float32)

    n = len(audio)

    def _fit(stem):
        stem = np.asarray(stem, dtype=np.float32)
        if stem.ndim == 2:
            if stem.shape[-1] == n:
                return stem
            if stem.shape[-1] > n:
                return stem[:, :n]
            return np.pad(stem, ((0, 0), (0, n - stem.shape[-1])))
        if len(stem) == n:
            return stem
        if len(stem) > n:
            return stem[:n]
        return np.pad(stem, (0, n - len(stem)))

    return _fit(vocal_stem), _fit(bed_stem)


def _mono_downmix(stem: np.ndarray) -> np.ndarray:
    """One channel from a stem that may carry two. Used wherever a separated
    accompaniment is a CHANNEL of a render rather than the track itself."""
    stem = np.asarray(stem, dtype=np.float32)
    return stem.mean(axis=0) if stem.ndim == 2 else stem


def _decode_mono_for_conversion(raw_audio_bytes: bytes, floor: int = 44100):
    """Decode an upload to the mono float32 array every conversion path works in.

    One definition, used by both the chunk/full conversion and the whole-track
    backing-bed separation, so the two always agree about sample rate and
    channel layout, the bed is sliced by OFFSET into this timeline, and an
    off-by-anything here would misalign the band against the voice.
    """
    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sample_rate = in_sr if in_sr >= floor else floor
    if in_sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


# ===========================================================================
# THE TRACK'S OWN BACKING TRACK, SEPARATED ONCE.
#
# Live mode cannot afford Demucs per chunk, a call costs tens of seconds and a
# chunk has to come back in about ten, but it does not have to: the separation
# is a property of the UPLOADED FILE, not of the chunk. So it is done once per
# upload, in the background, and a live chunk then takes the slice of it that
# lines up with the chunk's own offset in the track. Same separation the full
# render uses, so Live and Full now play the same backing track.
#
# Why it is worth the machinery, measured on this project's reference recording
# over the SAME 1-5 s window with the leakage metric (20*log10 of the isolated
# lead vocal projected onto the bed, see the note where the live bed is built
# for what those numbers mean): the per-chunk harmonic/percussive bed sits
# 17.0 dB under the recording's own vocal, this separation 70.6 dB under it,
# i.e. gone. Being the model's own "everything that is not the singer", it also
# carries the melodic instruments that a percussive-only reconstruction throws
# away, which is the difference between "sing along with instruments" playing
# a real backing track and it playing a hi-hat.
_BACKING_BED_LOCK = threading.Lock()
_BACKING_BED_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_BACKING_BED_MAX_TRACKS = 2


def _backing_bed_key(raw_audio_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha1(raw_audio_bytes).hexdigest()[:16]


def _backing_bed_status(track_key: str | None) -> str:
    """'running' | 'ready' | 'failed', or 'unknown'/'none' for a key nothing
    is cached under. The page only uses this to say what bed it is hearing."""
    if not track_key:
        return "none"
    with _BACKING_BED_LOCK:
        entry = _BACKING_BED_CACHE.get(track_key)
    return entry["status"] if entry else "unknown"


def _backing_stem_slice(track_key: str | None, stem: str, offset_s: float | None, sample_rate: int,
                        length: int):
    """One separated stem of the upload, covering a live chunk, or None.

    None covers every case where there is nothing usable to hand back, no key,
    a separation that is still running or that failed, an offset past the end of
    the track, and each of those simply leaves the chunk on its own built bed.

    "audio" is the accompaniment and "vocal" is the other half of the same
    separation, the isolated lead vocal.
    """
    if not track_key or offset_s is None or length <= 0:
        return None
    with _BACKING_BED_LOCK:
        entry = _BACKING_BED_CACHE.get(track_key)
        if entry is None or entry["status"] != "ready":
            return None
        bed = entry.get(stem)
        bed_sr = entry["sample_rate"]
    if bed is None or bed_sr <= 0:
        return None
    start = int(round(max(0.0, float(offset_s)) * bed_sr))
    total = bed.shape[-1] if bed.ndim == 2 else len(bed)
    if start >= total:
        return None
    stop = min(total, start + int(round(length * bed_sr / float(sample_rate))))
    # The accompaniment is kept in the model's own stereo; a live chunk's bed is
    # a CHANNEL of a response, and every consumer of that channel is mono, so a
    # slice of it is averaged here, once, at the point of use.
    slice_ = np.asarray(bed[..., start:stop] if bed.ndim == 2 else bed[start:stop], dtype=np.float32)
    if slice_.shape[-1] < 2:
        return None
    if bed_sr != sample_rate:
        import librosa
        slice_ = np.asarray(librosa.resample(slice_, orig_sr=bed_sr, target_sr=sample_rate, axis=-1),
                            dtype=np.float32)
    return np.ascontiguousarray(_mono_downmix(slice_), dtype=np.float32)


def _backing_bed_slice(track_key: str | None, offset_s: float | None, sample_rate: int,
                       length: int):
    """The separated accompaniment covering a live chunk, or None."""
    return _backing_stem_slice(track_key, "audio", offset_s, sample_rate, length)


def _backing_vocal_slice(track_key: str | None, offset_s: float | None, sample_rate: int,
                         length: int):
    """The separated LEAD VOCAL covering a live chunk, or None.

    This is what a live chunk hands the voice model once the separation is
    ready: a voice converter cannot decline to convert a drum hit, and the raw
    chunk on a drum-led track is mostly drums, which is what the model then
    re-sings. See the chunk's own note in _convert_track_direct, and
    _silence_instrumental_spans, which is close to a no-op on a real mix.
    """
    return _backing_stem_slice(track_key, "vocal", offset_s, sample_rate, length)


def _start_backing_bed_separation(raw_audio_bytes: bytes, track_key: str,
                                  min_duration_s: float = 6.0) -> bool:
    """Separate this upload's accompaniment in the background (idempotent).

    Returns True when a separation was actually started here, False when there
    was nothing to do (already cached, or a clip too short for the model, in
    which case Live mode keeps using each chunk's own bed).
    """
    with _BACKING_BED_LOCK:
        entry = _BACKING_BED_CACHE.get(track_key)
        if entry is not None:
            _BACKING_BED_CACHE.move_to_end(track_key)
            return False

    try:
        audio, sample_rate = _decode_mono_for_conversion(raw_audio_bytes)
    except Exception as e:
        print(f"[backing bed] could not decode the upload for separation ({e})")
        return False
    if len(audio) / float(sample_rate) < min_duration_s:
        return False

    mix_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    with _BACKING_BED_LOCK:
        _BACKING_BED_CACHE[track_key] = {
            "status": "running", "audio": None, "vocal": None, "sample_rate": 0,
            "mix_rms": mix_rms,
        }
        while len(_BACKING_BED_CACHE) > _BACKING_BED_MAX_TRACKS:
            _BACKING_BED_CACHE.popitem(last=False)

    def _work():
        t0 = time.time()
        try:
            vocal, bed = _separate_stems(audio, sample_rate)
            with _BACKING_BED_LOCK:
                _BACKING_BED_CACHE[track_key] = {
                    "status": "ready", "audio": np.asarray(bed, dtype=np.float32),
                    # The other half of the same separation, kept because a
                    # live chunk needs it: it is the only voice-only material
                    # this side of the sidecar has, and one 47 s stem is 8 MB.
                    "vocal": np.asarray(vocal, dtype=np.float32),
                    "sample_rate": sample_rate,
                    # The upload's own loudness, kept so the accompaniment can
                    # be handed back at the level of the file it came from (see
                    # the track endpoint): a page that swaps one for the other
                    # must not also swap the volume.
                    "mix_rms": mix_rms,
                }
            print(f"[backing bed] separated the upload's accompaniment in {time.time() - t0:.1f}s "
                  f"({len(audio) / sample_rate:.1f}s of audio), Live mode with instruments On now "
                  "plays the track's real instruments, with the original lead vocal not in it.")
        except Exception as e:
            with _BACKING_BED_LOCK:
                _BACKING_BED_CACHE[track_key] = {
                    "status": "failed", "audio": None, "vocal": None, "sample_rate": 0,
                }
            print(f"[backing bed] separation failed ({e}), Live mode keeps using each chunk's "
                  "own bed instead.")

    threading.Thread(target=_work, name="backing-bed", daemon=True).start()
    print(f"[backing bed] separating the upload's accompaniment in the background "
          f"({len(audio) / sample_rate:.1f}s of audio)...")
    return True


def _silence_instrumental_spans(audio: np.ndarray, sample_rate: int, log_prefix: str,
                                 frame_s: float = 0.10, min_span_s: float = 0.3,
                                 harmonic_ratio_threshold: float = 0.35,
                                 flatness_threshold: float = 0.35) -> np.ndarray:
    n = len(audio)
    if n < int(frame_s * sample_rate):
        return audio

    import librosa
    harmonic, percussive = librosa.effects.hpss(audio)

    hop = max(1, int(frame_s * sample_rate))
    n_frames = max(1, n // hop)

    stft_mag = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)) + 1e-10
    flatness_full = librosa.feature.spectral_flatness(S=stft_mag)[0]
    flatness_times = librosa.frames_to_time(np.arange(len(flatness_full)), sr=sample_rate, hop_length=256)

    is_instrumental = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start_i = i * hop
        end_i = min(n, start_i + hop)
        he = float(np.sum(harmonic[start_i:end_i].astype(np.float64) ** 2))
        pe = float(np.sum(percussive[start_i:end_i].astype(np.float64) ** 2))
        total = he + pe
        if total <= 1e-9:
            continue
        ratio = he / total

        t0, t1 = start_i / sample_rate, end_i / sample_rate
        mask = (flatness_times >= t0) & (flatness_times < t1)
        flat = float(np.mean(flatness_full[mask])) if np.any(mask) else 1.0

        is_instrumental[i] = not (ratio >= harmonic_ratio_threshold and flat <= flatness_threshold)

    spans = []
    cur_start = None
    for i in range(n_frames):
        if is_instrumental[i] and cur_start is None:
            cur_start = i * hop / sample_rate
        elif not is_instrumental[i] and cur_start is not None:
            end_t = i * hop / sample_rate
            if end_t - cur_start >= min_span_s:
                spans.append((cur_start, end_t))
            cur_start = None
    if cur_start is not None:
        end_t = n / sample_rate
        if end_t - cur_start >= min_span_s:
            spans.append((cur_start, end_t))

    if not spans:
        return audio

    audio = audio.copy()
    fade_samples = int(0.02 * sample_rate)
    for start_s, end_s in spans:
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

    print(f"[{log_prefix}] (fallback) silenced {len(spans)} instrumental span(s) totaling "
          f"{sum(e - s for s, e in spans):.1f}s (frame-level HPSS, {n / sample_rate:.1f}s clip)")
    return audio


def _convert_track_direct(raw_audio_bytes: bytes, voice_index: int, pad_seconds: float | None = None,
                           instruments_enabled: bool = True, transpose_semitones: float | None = None,
                           track_key: str | None = None, track_offset_s: float | None = None,
                           vibrato_amount: float | None = None):
    """Convert an upload, or one live chunk of it, through the trained voice.

    Returns (wav_bytes, bed_kind). `bed_kind` says what, if anything, came back
    alongside the voice and is what the endpoint reports to the page in
    X-Backing-Bed: "demucs" (the full render's separated accompaniment),
    "separation" (the same separation, sliced for a live chunk), "drums" (the
    per-chunk fallback bed, used while a separation is still running) or
    "none" (voice alone).
    """
    from config.config_loader import get_config
    import librosa

    cfg = get_config()

    audio, sample_rate = _decode_mono_for_conversion(raw_audio_bytes)
    bed_kind = "none"

    DEMUCS_MIN_DURATION_S = 6.0
    # How far the drums are held down under the converted voice in a live
    # chunk's bed. The percussive layer is what keeps the pulse going while the
    # robot sings, but it is also what the voice competes with for space, so it
    # comes back below unity rather than at full level.
    LIVE_BED_DRUM_SUPPORT = 0.75
    # How patient the harmonic/percussive split is with a live chunk's bed.
    # At the default margin of 1.0 the percussive layer keeps some of the
    # original lead vocal's own energy, a snare is a broadband transient and
    # so is a sung consonant, and measured against this project's test
    # recording that ghost singer sat only 10.8 dB under the recording's real
    # level in the bed. Tightening the split drops it to about 17 dB, which is
    # the difference between "there is another singer faintly in the bed" and
    # "there is not", as far as a per-chunk split can get. The separated
    # whole-track bed above reaches 70.6 dB on the same window, so this one is
    # the FALLBACK, for the minute before a new track's separation is ready and
    # for clips too short to separate at all.
    LIVE_BED_HPSS_MARGIN = 2.0
    # How far the accompaniment steps back while the robot is singing, on the
    # two beds that arrive at full band level (the separated ones). 4 dB is the
    # same shallow lead-vocal duck the single-vocable render has always used
    # see _duck_accompaniment_under_voice for why that is what makes the words
    # audible. The per-chunk fallback bed below is already a deliberately thin
    # drums-only reconstruction sitting ~9 dB under the voice, so it is left
    # alone: ducking that as well would pull the only rhythmic anchor out of the
    # mix exactly where it is needed.
    BED_DUCK_DB = 4.0
    duration_s = len(audio) / sample_rate
    is_live_chunk = pad_seconds is not None

    audio_for_rvc = audio
    keep_bed = None

    # WHAT THE VOICE MODEL IS HANDED, AND WHAT THE TOGGLE ACTUALLY MEANS.
    #
    # "Sing along with instruments" used to mean "hand the raw mix to the
    # trained voice, drums and all", and that is the one thing a
    # voice-conversion model cannot do: it has no concept of a drum hit, so it
    # reads a snare transient as a violent pitch and timbre jump and re-sings
    # it, the "singing drum machine" glitch that the section above is about,
    # and the reason this mode sounded broken and unclear with the toggle ON
    # while OFF sounded merely normal. Neither setting was right: ON wrecked
    # the voice on every percussive frame, and OFF threw the whole arrangement
    # away.
    #
    # So the model is now ALWAYS given voice-only material (drums removed one
    # way or another, below), and the toggle decides what comes back alongside
    # it:
    #   ON  -> the converted voice AND the track's own instruments, as two
    #          channels. The page plays those two and never plays the uploaded
    #          original over the top, so it is a real sing-along with the band
    #          with one voice in it rather than two.
    #   OFF -> the converted voice alone. The robot performs solo, exactly as
    #          this toggle has always been documented to mean.
    if is_live_chunk:
        # A live chunk is 1-4 seconds and has to come back in real time, so
        # Demucs (tens of seconds per call) is not available here.
        #
        # WHAT THE MODEL HEARS IS NOW THE SAME IN BOTH TOGGLE POSITIONS, and
        # that is the fix for "with instruments On this sounds broken and
        # unclear". On used to hand the trained voice the chunk's whole
        # harmonic content, the lead vocal PLUS every melodic instrument at
        # once, and a voice converter cannot decline to convert something:
        # it re-sings guitars, keys and the old lead vocal as one chaotic
        # voice, which is exactly the broken, unclear result. Suppressing the
        # percussion alone was never enough; the melodic instruments are just
        # as unvoice-like as the drums are. The model is now always handed
        # voice-focused material (the same treatment "Off" always got, which
        # is the setting that sounded normal), and the toggle only decides
        # what comes back alongside it, see keep_bed below.
        # WHAT THE MODEL IS FED, ONCE THE SEPARATION EXISTS.
        #
        # The sentence above says the model is always handed voice-focused
        # material, and for the first version of this the only tool a live
        # chunk had was the frame-level muting below, which on a real mix is
        # very close to a no-op: measured on this project's own test chunk it
        # silenced 0% of it, because a snare and a sung consonant are the same
        # kind of broadband transient and the harmonic/percussive ratio of a
        # full mix stays above the threshold on drums too. So the model was
        # being handed the chunk as it came in, drums and all, and a voice
        # converter re-sings a drum hit as a drum hit. With "Sing along with
        # instruments" On that was survivable, the drums are in the bed
        # anyway, but with it OFF there is no bed, so the drum re-singing was
        # the ENTIRE output: a live chunk that came back as drum noise and no
        # voice, which is the exact complaint it has to answer.
        #
        # The separation above already isolates the lead vocal, and it is a
        # property of the file rather than of the chunk (see
        # /api/neural/backing-bed), so once it is ready the model is fed the
        # matching slice of it and hears nothing but the thing it is there to
        # re-sing. Until then, and for a clip too short to separate, the muting
        # below is still the fallback, which is why it is still here.
        separated_vocal = _backing_vocal_slice(track_key, track_offset_s, sample_rate, len(audio))
        if separated_vocal is not None and len(separated_vocal) > 1:
            audio_for_rvc = separated_vocal
            print(f"[convert-track direct] live chunk: the model is fed the upload's own "
                  f"separated lead vocal ({len(separated_vocal) / sample_rate:.1f}s slice at "
                  f"{float(track_offset_s or 0.0):.1f}s), with no drums and no instruments in "
                  "it, so what comes back is a re-sung voice rather than re-sung percussion.")
        else:
            try:
                audio_for_rvc = _silence_instrumental_spans(audio, sample_rate, "convert-track direct")
            except Exception as e:
                print(f"[convert-track direct] instrumental-span muting on a live chunk failed ({e}) "
                      ", sending the chunk as-is")
                audio_for_rvc = audio

        if instruments_enabled:
            # THE BED FOR THIS CHUNK, IN TWO TIERS.
            #
            # 1. The upload's own separated accompaniment, when it exists: the
            #    page names the track (track_key) and where this chunk sits in
            #    it (track_offset_s), and what comes back is the slice of the
            #    separation that lines up with it. That is the same
            #    accompaniment the full render plays, so it is the track's real
            #    instruments, melodic ones included, and the original lead
            #    vocal is 58 dB down in it. It is background work started when
            #    the file was picked (see _start_backing_bed_separation), so it
            #    costs this request nothing.
            #
            # 2. While that separation is still running (or for a track whose
            #    clip is too short to separate), the per-chunk bed below. It is
            #    built so the ORIGINAL LEAD VOCAL IS NOT IN IT either:
            #    `audio - audio_for_rvc` is precisely the material that was NOT
            #    being sung on (the gaps between phrases), so it carries
            #    instruments and no original voice; the percussive layer is
            #    added back across the whole chunk so the drums keep the groove
            #    going while the robot sings. What the page plays is then
            #    band-in-the-gaps + drums-under-the-voice + one converted voice,
            #    instead of the uploaded recording's own lead vocal singing
            #    along with the robot.
            #
            # Tier 2's rebuild deliberately does NOT require that the
            # span-muting above actually changed anything. A short chunk is very
            # often one unbroken phrase with no instrumental gap in it at all,
            # and the first version of this skipped the bed entirely in that
            # case, which is the common case, so the toggle did nothing. There
            # the residual term is simply zero and the drums alone carry the
            # chunk, which is still the right answer: a pulsing bed with one
            # clean converted voice over it, not the recording's own singer.
            separated_bed = _backing_bed_slice(track_key, track_offset_s, sample_rate, len(audio))
            if separated_bed is not None:
                keep_bed = separated_bed
                bed_kind = "separation"
                print(f"[convert-track direct] live chunk: bed taken from the upload's own separated "
                      f"accompaniment ({len(separated_bed) / sample_rate:.1f}s slice at "
                      f"{float(track_offset_s or 0.0):.1f}s), the track's real instruments, with "
                      "the original lead vocal not in them.")
            else:
                try:
                    _harmonic, percussive = librosa.effects.hpss(audio, margin=LIVE_BED_HPSS_MARGIN)
                    residual = audio - audio_for_rvc
                    keep_bed = np.ascontiguousarray(
                        residual + percussive * LIVE_BED_DRUM_SUPPORT, dtype=np.float32)
                    bed_kind = "drums"
                    bed_energy = float(np.sum(np.abs(keep_bed)) / max(1e-9, float(np.sum(np.abs(audio)))) * 100.0)
                    gap_energy = float(np.sum(np.abs(residual)) / max(1e-9, float(np.sum(np.abs(audio)))) * 100.0)
                    print(f"[convert-track direct] live chunk: bed rebuilt ({bed_energy:.0f}% of the chunk's "
                          f"energy; {gap_energy:.0f}% of it the un-sung material, the rest drums held "
                          f"under the voice), no original lead vocal in it. (Separated accompaniment "
                          f"for this chunk: {_backing_bed_status(track_key)}, so this one gets the "
                          "drums-only bed instead.)")
                except Exception as e:
                    print(f"[convert-track direct] live bed rebuild failed ({e}), "
                          "the chunk comes back voice-only.")
                    keep_bed = None
    elif duration_s >= DEMUCS_MIN_DURATION_S:
        try:
            vocal_stem, bed = _separate_stems(audio, sample_rate)
            audio_for_rvc = vocal_stem
            if instruments_enabled:
                # One channel in the response (see _separate_stems on why the
                # cached bed keeps its two).
                keep_bed = _mono_downmix(bed)
                bed_kind = "demucs"
                print(f"[convert-track direct] separated vocal from instrumental via Demucs "
                      f"({duration_s:.1f}s clip), RVC converts the isolated vocal only, and the "
                      "instruments come back with it as a second channel, so the page plays a "
                      "standalone sing-along instead of the uploaded original.")
            else:
                print(f"[convert-track direct] separated vocal from instrumental via Demucs "
                      f"({duration_s:.1f}s clip), RVC converts the isolated vocal only, and "
                      "the instrumental is discarded entirely: \"instruments off\" means the "
                      "robot performs solo, with no drum/instrument sound in the output at all.")
        except Exception as e:
            print(f"[convert-track direct] Demucs separation failed ({e}), falling back "
                  "to frame-level instrumental muting for this render.")
            audio_for_rvc = _silence_instrumental_spans(audio, sample_rate, "convert-track direct")
    else:
        audio_for_rvc = _silence_instrumental_spans(audio, sample_rate, "convert-track direct")

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(
        audio_for_rvc, sample_rate, voice_index=voice_index, timeout_s=timeout_s, pad_seconds=pad_seconds,
        transpose_semitones=transpose_semitones,
    )
    if converted is None:
        raise RuntimeError(
            "Neural sidecar conversion failed or timed out, check that neural_env/rvc_server.py "
            "is running and see its terminal output / GET /neural/status."
        )

    final = np.asarray(converted, dtype=np.float32)
    if final.ndim > 1:
        final = final.mean(axis=1)

    # THE ROBOT'S FOLDS VIBRATE IN THE LIVE ENGINE TOO. The panel's Vocal fold
    # vibration setting is a property of the robot's voice, not of the offline
    # renderer, so a converted chunk gets the same movement the full renders get
    # -- applied to the voice BEFORE the bed is ducked under it, so the duck
    # still fits what is really sent. A chunk is short, so the vibrato's own
    # "arrives after the attack" timing is what carries it across the chunk
    # boundary without a seam (see synthesis/vibrato.py).
    final = _apply_vocal_vibration_amount(final, sample_rate, vibrato_amount)

    if keep_bed is not None:
        # Two channels: the track's own instruments at unity, and the converted
        # voice. One shared trim across both, so the balance between them is
        # exactly as designed (see the two-channel note in
        # _render_track_single_vocable_offline for why it is shared).
        n = min(len(final), len(keep_bed))
        bed = np.asarray(keep_bed[:n], dtype=np.float32)
        voc = final[:n]
        # A separated accompaniment arrives at FULL band level, i.e. within a
        # couple of dB of the voice it is about to be played under, and a lead
        # that close to the bed loses its consonants. So on the real backing
        # tracks, and only there, see BED_DUCK_DB, the bed steps back while
        # the robot is singing, with the same look-ahead curve the
        # single-vocable render uses. Applied before the peak is measured, so
        # the shared trim below fits what is actually sent.
        if bed_kind in ("demucs", "separation"):
            bed = np.asarray(
                _duck_accompaniment_under_voice(bed, voc, sample_rate, max_db=BED_DUCK_DB),
                dtype=np.float32,
            )
        peak = float(np.max(np.abs(bed + voc))) if n else 0.0
        trim = (0.95 / peak) if peak > 0.95 else 1.0
        final = np.stack([bed * trim, voc * trim], axis=1)
        duck_note = (f", ducked {BED_DUCK_DB:.1f} dB under the voice where it sings"
                     if bed_kind in ("demucs", "separation") else "")
        print(f"[convert-track direct] sending two channels (instruments + converted voice, "
              f"bed '{bed_kind}'{duck_note}, trim {trim:.3f}, peaks at {peak * trim:.3f}): the "
              "uploaded original is not played over this, so only one voice is heard.")

    out_buf = io.BytesIO()
    sf.write(out_buf, final.astype(np.float32), sample_rate, format="WAV")
    return out_buf.getvalue(), bed_kind


@app.post("/api/neural/convert-track")
async def convert_neural_track(
    file: UploadFile = File(...),
    voice_index: int = Form(0),
    pad_seconds: float | None = Form(None),
    instruments_enabled: bool = Form(True),
    transpose_semitones: float | None = Form(None),
    track_key: str | None = Form(None),
    track_offset_s: float | None = Form(None),
    mode: str = Form(""),
    vibrato_amount: float | None = Form(None),
):
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready, check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )
    if _is_hum_mode(mode):
        # HUMMING IS NOT A RE-VOICING. This route's job is to re-sing the words
        # that are in the upload in the trained voice, and a hum has no words in
        # it to re-sing: what the robot hums is the MELODY, rendered by the hum
        # voice itself and then converted, which is the whole-track hum path in
        # _render_track_offline. What comes back is the same thing this route
        # always returns -- the robot's voice, one channel, which the page plays
        # over the uploaded file -- so the panel needs no special case for it.
        try:
            raw = await file.read()
            # "yin" is only what the project's own detector is asked for before
            # the hum path is taken: the hum tracks its own melody (see
            # synthesis/hum_voice.py's own HumConfig.pitch_method), and this
            # argument has to name one of the detectors PitchDetector actually
            # has, which "pyin" is not.
            hum_key = _hum_render_key(raw, "solo", voice_index, False,
                                      transpose_semitones, vibrato_amount)
            wav_bytes, from_memo = await asyncio.to_thread(
                _memoized_hum_render, hum_key,
                lambda: _render_track_offline(
                    raw, "yin", "solo", voice_index,
                    "solo_humming", False, transpose_semitones, True,
                    None, None, None, None, None, None, vibrato_amount,
                ),
            )
            if from_memo:
                print("[neural hum] this chunk's hum is already rendered, playing it back "
                      "rather than rendering it again")
            return Response(content=wav_bytes, media_type="audio/wav",
                            headers={"X-Backing-Bed": "none",
                                     "X-Render-Replayed": "1" if from_memo else "0"})
        except NothingToSingError as e:
            # The same refusal, in the same shape, the DSP and neural render
            # routes make: a track with nothing melodic in it has nothing to hum.
            return JSONResponse({"error": str(e)}, status_code=422)
        except Exception as e:
            print(f"[neural hum error] {e}")
            return JSONResponse({"error": f"Humming failed: {e}"}, status_code=500)
    try:
        raw = await file.read()
        wav_bytes, bed_kind = await asyncio.to_thread(
            _convert_track_direct, raw, voice_index, pad_seconds, instruments_enabled,
            transpose_semitones, track_key, track_offset_s, vibrato_amount,
        )
        # X-Backing-Bed tells the page which bed it is actually hearing, so Live
        # mode can say so and so the browser side can be checked without
        # listening to it: "demucs"/"separation" are real backing tracks,
        # "drums" is the fallback while a separation is still running.
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"X-Backing-Bed": bed_kind})
    except Exception as e:
        print(f"[neural direct convert error] {e}")
        return JSONResponse({"error": f"Conversion failed: {e}"}, status_code=500)


@app.post("/api/neural/backing-bed")
async def neural_backing_bed(file: UploadFile = File(...)):
    """Start (once per upload) the whole-track backing-bed separation.

    Returns immediately with the track_key the caller should send back with its
    live chunks; the separation itself runs in a background thread. Called as
    soon as a file is picked, so that by the time Live mode asks for a chunk
    there is usually a real backing track to slice. Idempotent: asking again for
    the same file is free.
    """
    try:
        raw = await file.read()
        key = _backing_bed_key(raw)
        started = await asyncio.to_thread(_start_backing_bed_separation, raw, key)
        status = _backing_bed_status(key)
        print(f"[backing bed] track {key}: {'started' if started else 'already known'} ({status})")
        return JSONResponse({"track_key": key, "status": status, "started": started})
    except Exception as e:
        print(f"[backing bed error] {e}")
        return JSONResponse({"error": f"Could not start the backing-track separation: {e}"},
                            status_code=500)


@app.get("/api/neural/backing-bed/status")
async def neural_backing_bed_status(track_key: str = ""):
    """'running' | 'ready' | 'failed' | 'unknown' | 'none', see
    _backing_bed_status. Polled only to word the Live Demo's own status line."""
    return JSONResponse({"track_key": track_key, "status": _backing_bed_status(track_key or None)})


@app.get("/api/neural/backing-bed/track")
async def neural_backing_bed_track(track_key: str = ""):
    """The upload's separated accompaniment, WHOLE, as the track the page plays.

    WHY THIS EXISTS. The page used to play the uploaded file itself under the
    robot in every engine and every mode, on the reading that a person who
    drops a track in is owed the sound of that track. In the Neural engine that
    reading is wrong, and it is the one thing about this panel that people
    heard immediately: Neural does not add a part beside the singer, it RE-SINGS
    the singer -- the same words, the same melody, the same timing, from the
    same notes -- so what played was the recording's own lead and the model's
    re-sung copy of it at once, two voices on one line, a few milliseconds
    apart. That is a chorus effect, not a duet: the two leads sum into comb
    filtering across the whole vocal range, and the words inside it are harder
    to pick out than either voice alone. What the robot is supposed to be heard
    WITH is the track's instruments, and the file that has exactly those and no
    singer in it is this one: the same Demucs separation every live chunk is
    already converted from (57-70 dB of original-lead rejection, see below),
    here handed over whole so the page can play it as the track.

    The two halves are DUCKED by the page the same way the two-channel renders
    are: TRACK VOLUME is this endpoint's output, ROBOT VOLUME is the robot's.
    Nothing about the modes changes -- call and response and humming still play
    the uploaded file, because in those two the robot is not singing the
    singer's own line and the singer is the one being answered.

    Returns 409 with the separation's status while it is not ready (or failed),
    and the WAV at the upload's own length and loudness once it is.
    """
    if not track_key:
        return JSONResponse({"error": "track_key is required"}, status_code=400)
    with _BACKING_BED_LOCK:
        entry = _BACKING_BED_CACHE.get(track_key)
        if entry is None:
            return JSONResponse({"status": "unknown"}, status_code=409)
        if entry["status"] != "ready":
            return JSONResponse({"status": entry["status"]}, status_code=409)
        bed = entry.get("audio")
        bed_sr = int(entry.get("sample_rate") or 0)
        mix_rms = float(entry.get("mix_rms") or 0.0)
    if bed is None or bed_sr <= 0:
        return JSONResponse({"status": "unknown"}, status_code=409)

    audio = np.asarray(bed, dtype=np.float32)
    # WEIGHTED TO THE FILE IT CAME FROM, so the two are interchangeable under
    # TRACK VOLUME. Removing a lead vocal genuinely removes the loudest thing in
    # most mixes: on this project's synthetic song (voice over a drum bed, so
    # the band is drums alone) the raw separation measures 15.5 dB under the mix
    # it came from, and a page that swaps one for the other must not also swap
    # the volume. So the band is amplified to the mix's own RMS, capped at +-12 dB
    # so a stem that is wildly off (a nearly silent mix, a broken separation)
    # cannot be amplified into the panel.
    bed_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    if mix_rms > 1e-6 and bed_rms > 1e-6:
        gain = min(4.0, max(0.25, mix_rms / bed_rms))
        audio = (audio * gain).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = (audio * (0.99 / peak)).astype(np.float32)

    out_buf = io.BytesIO()
    sf.write(out_buf, audio.T if audio.ndim == 2 else audio, bed_sr,
             format="WAV", subtype="PCM_16")
    return Response(content=out_buf.getvalue(), media_type="audio/wav",
                    headers={"X-Separation-Channels": str(audio.shape[0] if audio.ndim == 2 else 1),
                             "X-Sample-Rate": str(bed_sr)})


def _probe_channels(path: str) -> int:
    """How many channels this file really has, from its header, without decoding it.

    soundfile reads the header of everything it can open (wav, flac, ogg, mp3);
    ffprobe is the fallback for containers it cannot (m4a/aac and friends), and
    the final answer for anything neither can open is 1, which keeps the worst
    case to a quiet decode rather than a loud, clipped one.
    """
    try:
        import soundfile as _sf
        return int(_sf.info(path).channels)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=channels", "-of", "csv=p=0", path],
            capture_output=True, check=True, timeout=20,
        ).stdout.decode(errors="replace").strip().splitlines()[0].split(",")[0]
        return int(float(out))
    except Exception:
        return 1


def _decode_audio_via_ffmpeg(raw_audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode any upload to mono float32 at target_sr, as a TRUE average of the channels.

    ffmpeg's own `-ac 1` is not an average. For stereo it is an energy-preserving
    downmix, (L+R)/sqrt(2), which on a file whose channels are correlated (the
    usual case: a mono voice panned centre, or any normal stereo master) is
    +3.0 dB over the real (L+R)/2. Two things follow from that, and both were
    happening here: every analysis stage fed this function heard the upload a
    third of a power-doubling louder than the conversion stages did, and a track
    mastered near full scale came back CLIPPED, because the sum of two channels
    that each peak at 0.95 has nowhere to go in float. So a stereo source is
    decoded to its own two channels and averaged HERE, where it is exact.

    THE CHANNEL COUNT HAS TO BE ASKED FOR, not forced, and that is not a detail:
    ffmpeg's `-ac 2` on a MONO source is an energy-preserving upmix, one channel
    of x/sqrt(2) on each side, so averaging a forced-stereo decode of a mono file
    loses 3.0 dB -- exactly the error this function was fixed to remove, just
    pointing the other way. A mono upload is therefore decoded with `-ac 1`,
    which is that same channel unchanged, and a stereo upload with `-ac 2`,
    which is those two channels unchanged. Anything with more than two channels
    (5.1, ambisonic) goes through ffmpeg's own stereo downmix and then the same
    average, since there is no honest two-channel answer to average there.
    """
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(raw_audio_bytes)
        tmp_in_path = tmp_in.name
    try:
        channels = _probe_channels(tmp_in_path)
        ac = "1" if channels == 1 else "2"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp_in_path,
             "-f", "f32le", "-ac", ac, "-ar", str(target_sr), "-"],
            capture_output=True, check=True,
        )
        interleaved = np.frombuffer(proc.stdout, dtype=np.float32)
        if ac == "2" and interleaved.size >= 2:
            frames = interleaved.size // 2
            stereo = interleaved[:frames * 2].reshape(-1, 2)
            return np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
        return interleaved.copy()
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


def _detect_vocal_pitch_spans(raw_audio_bytes: bytes, min_duration_s: float = 0.3,
                               max_gap_s: float = 0.5) -> list:
    """Scan for stretches with a real, trackable fundamental frequency, not Whisper's
    no_speech_prob. A stretch of audio with a detected pitch in it is, by definition, not
    silence, regardless of whether Whisper's speech-likelihood model recognizes it as "speech".
    This is the ground truth used to force a transcription attempt on anything Whisper's own
    silence gate would otherwise have thrown away.

    This used to run this project's own frame-by-frame RMVPE pitch tracker, the same one that
    drives the live accompaniment pitch tracking, calling it once per individual audio frame
    (roughly 1,500-3,000+ separate model inferences for a single short track). That tool is
    built for real-time, one-frame-at-a-time tracking; reusing it for an offline full-file scan
    meant paying that per-frame model-call overhead thousands of times over just to answer a
    yes/no "is a pitch present here" question, and was very likely the dominant cost in the
    whole transcription pipeline, worse than the actual Whisper decode. librosa.pyin answers
    the same question with one vectorized call across the entire signal instead of one Python
    loop iteration + neural inference per frame. This is a genuinely different algorithm (a
    classical pitch estimator, not the same neural model), so exact span boundaries can shift by
    a fraction of a second versus the old RMVPE-based version, the merge/gap-bridging step
    downstream already smooths over that, but it answers the same question and is
    dramatically cheaper to compute. NOTE: this adds `librosa` as a new dependency if it isn't
    already installed in this project's environment, add it to requirements.txt.

    Short gaps between pitched frames (a breath, a consonant, the dip between two notes in an
    "ah-ah-ah" style vocable run) are bridged rather than treated as separate spans, since
    those are one continuous vocal phrase, not silence.
    """
    import librosa

    sample_rate = 16000
    audio = _decode_audio_via_ffmpeg(raw_audio_bytes, sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    # hop_length=512 (32ms resolution at 16kHz) rather than librosa's finer defaults: this only
    # needs to answer "is a pitch present, roughly when" to feed a 0.3s-minimum/0.5s-gap-bridge
    # span detector and a several-second merge step downstream, so paying for finer time
    # resolution than that buys nothing. Measured directly against this project's reference
    # track: hop=256 took ~12s, hop=512 ~2s, for effectively the same voiced-frame ratio.
    # hop=1024 was tried and rejected, it throws an internal librosa error on short tracks
    # (not enough frames for its transition-matrix sizing), so it isn't safe to use here.
    hop_length = 512
    frame_time_s = hop_length / sample_rate
    _f0, voiced_flags, _voiced_prob = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sample_rate, hop_length=hop_length)

    spans = []
    cur_start = None
    last_voiced_end = 0.0
    for i, voiced in enumerate(voiced_flags):
        t = i * frame_time_s
        if voiced:
            if cur_start is None:
                cur_start = t
            last_voiced_end = t + frame_time_s
        elif cur_start is not None and (t - last_voiced_end) > max_gap_s:
            spans.append((cur_start, last_voiced_end))
            cur_start = None
    if cur_start is not None:
        spans.append((cur_start, last_voiced_end))

    return [(round(s, 2), round(e, 2)) for s, e in spans if e - s >= min_duration_s]




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

# Whisper's decoder has a language model baked into its weights, and that LM actively prefers
# common, fluent English word sequences over unusual ones, it isn't a decoding-parameter bug
# that "well, hooray!" outscores "wha-he-oh", that's the model doing exactly what it was
# trained to do: pick the highest-probability real English continuation. Nothing in
# beam_size/temperature/best_of turns that off, because it's not a search-strategy setting,
# it's what the probabilities themselves say. A real fix for that is a phone-level recognizer
# with no word/language model at all (tested here with Allosaurus, a universal IPA phone
# recognizer), but on this specific audio the drums bleed straight through as spurious
# consonant clusters, since nothing has separated the voice from the percussion first; that
# needs a source-separation stage (e.g. Demucs) in front of it to be trustworthy, which is a
# meaningfully bigger addition (new heavy model dependency, its own latency and failure modes)
# and hasn't been verified against this project's actual deployment. Until that's built and
# tested, this initial_prompt is the mitigation: it primes Whisper's decoder with the hyphenated
# vocable *style* we want, nudging it away from snapping to grammatical interjections, without
# fully eliminating the underlying bias (that's a model-weights-level bias, not something any
# prompt can turn off completely).
#
# This was shortened at one point to a single short phrase, on a theory that the original
# 8-phrase comma-separated list was priming the decoder to "keep producing more list items,"
# after a run showed pathological repetition ("la-way-ya-hoo, la-way-ya-hoo, ..."). That theory
# was never confirmed, and the shortened version created a worse, directly-observed failure
# instead: on uncertain audio, Whisper started echoing the short prompt itself back nearly
# verbatim ("wha-he-oh hoo-wee wha-he-oh hoo-wee-ya", "oh-wee-ya-oh") rather than transcribing
# anything from the actual clip. The long-list version, by contrast, has a direct positive
# result on this project's own reference track ("way-hey-oh, hoo-wee", "Weyayoo, weyayoo,
# weyayoo, ah-hey-ya, weyayoo"), confirmed working, not a guess. Reverted to it. The repeat-
# collapse regex fix (see _REPEAT_RUN_RE) is the actual defense against degenerate repetition
# now, and it doesn't require gutting the prompt to work.
WHISPER_VOCABLE_PROMPT = ("wha-he-oh, hoo-wee-ya, na-ha-way, hey-ya-hoo, la-way-na, "
                           "ah-hey-ya, way-hoo-la, ya-na-hey")


def _whisper_transcribe_raw_segments(tmp_path: str, language, model, backend,
                                      use_vad: bool = True,
                                      no_speech_threshold: "float | None" = WHISPER_NO_SPEECH_THRESHOLD,
                                      beam_size: int = 5, best_of: int = 5,
                                      temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                      initial_prompt: "str | None" = None) -> list:
    """Run Whisper decode.

    use_vad controls whether Silero VAD pre-filters audio before Whisper ever
    sees it. VAD is tuned to detect *talking*, pauses, breath, silence
    between sentences. Sustained singing, chant, and wordless vocables (long
    held notes, "ah-ah-ah" type vocalizing) do not always look like "speech"
    to it, so VAD can silently drop whole stretches of a song before Whisper
    gets a chance to attempt a decode at all. That produces the symptom of
    "the first spoken line transcribes fine, then nothing for the rest of
    the track", it's not that Whisper heard the rest and rejected it, VAD
    never handed it over. Track/song uploads should therefore run with
    use_vad=False so every second of audio gets a decode attempt; the
    hallucination-confidence check downstream is what separates real speech
    from noise, not VAD. Live mic input can keep VAD, since that path is
    about chunking a real-time stream on pauses, not full-song coverage.

    beam_size/best_of/temperature are exposed as overridable knobs, though the
    forced/uncertain-material pass now uses the same full-quality config as the primary
    pass by default (see _force_transcribe_clip for why a cheaper greedy decode was tried
    here and reverted).

    initial_prompt primes the decoder's context with an example of the target
    output style, which is the mitigation used for Whisper's tendency to
    "autocorrect" unfamiliar vocalizing into fluent, real English words
    see WHISPER_VOCABLE_PROMPT.
    """
    if backend == "faster":
        def _run(lang, want_words, vad):
            kwargs = dict(
                language=lang, beam_size=beam_size, best_of=best_of, word_timestamps=want_words,
                condition_on_previous_text=False,  # prevents repetition-loop hallucination cascades on music
                temperature=temperature,  # retries at higher temp instead of giving up silently
                compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                log_prob_threshold=WHISPER_LOGPROB_THRESHOLD,
                no_speech_threshold=no_speech_threshold,
                initial_prompt=initial_prompt,
            )
            if vad:
                kwargs["vad_filter"] = True
                kwargs["vad_parameters"] = dict(min_silence_duration_ms=400)
            else:
                kwargs["vad_filter"] = False
            segments_iter, _info = model.transcribe(tmp_path, **kwargs)
            return list(segments_iter)

        try:
            segments = _run(language, True, use_vad)
        except Exception as e:
            if language is None:
                print(f"[transcribe] auto-detect decode failed ({e}); retrying with language='en'")
                language = "en"
            try:
                segments = _run(language, True, use_vad)
            except Exception as e2:
                print(f"[transcribe] word-timestamp alignment failed ({e2}); retrying without word timestamps")
                segments = _run(language, False, use_vad)
        return [
            {"start": s.start, "end": s.end, "text": s.text.strip(),
             "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
             "compression_ratio": s.compression_ratio,
             "words": [{"word": w.word.strip(), "start": w.start, "end": w.end} for w in (s.words or [])]}
            for s in segments
        ]

    if backend == "openai":
        def _run(lang, want_words):
            return model.transcribe(
                tmp_path, language=lang, fp16=False, word_timestamps=want_words,
                condition_on_previous_text=False,
                temperature=temperature,
                compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
                no_speech_threshold=no_speech_threshold,
                beam_size=beam_size, best_of=best_of,
                initial_prompt=initial_prompt,
            )

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



# Whisper meta-tags it sometimes emits for non-speech audio ("[Music]", "(singing)", music
# notes, etc.), these aren't a transcription of anything and should never appear in the
# anglicized text.
_WHISPER_META_TAG_RE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|[\u266a\u266b\u2669-\u266c]")
# Collapses pathological repeat loops (the classic Whisper-on-music failure mode: the same
# token or short phrase repeated dozens of times) down to a natural handful of repeats instead
# of either the full garbage run or discarding the segment outright.
#
# A "token" here includes internal hyphens ("la-way-ya-hoo" is ONE token, not four words joined
# by a non-word character), the original version used bare \w+, which doesn't treat "-" as
# part of a word, so it could never actually recognize a hyphenated vocable phrase as the
# repeating unit in the first place. Separators between repeats also now accept commas, not
# just whitespace: confirmed directly against a real pathological decode
# ("la-way-ya-hoo, la-way-ya-hoo, la-way-ya-hoo, ...", comma-separated) that the old
# whitespace-only separator let this exact failure mode straight through into the transcript.
_HYPHEN_TOKEN = r"[A-Za-z]+(?:-[A-Za-z]+)*"
_REPEAT_RUN_RE = re.compile(
    rf"\b({_HYPHEN_TOKEN}(?:\s+{_HYPHEN_TOKEN}){{0,3}})\b(?:[\s,]+\1\b){{2,}}", re.IGNORECASE)


def _anglicize_cleanup(text: str, max_repeats: int = 3) -> str:
    """Turn a raw, not-fully-confident Whisper decode into a readable anglicized line.

    This is deliberately NOT translation and never touches meaning, it only cleans up the
    artifacts of forcing an English decoder to render sounds (including Cree words and
    wordless vocables) it doesn't have real words for: bracketed meta-tags, music-note
    glyphs, and the repeated-token loops Whisper falls into when it isn't sure what it's
    hearing. What's left is Whisper's best phonetic-English spelling of the sound, which is
    exactly what an anglicized rendering is supposed to be.
    """
    if not text:
        return ""
    cleaned = _WHISPER_META_TAG_RE.sub(" ", text)

    def _collapse(m: "re.Match") -> str:
        phrase = m.group(1)
        return ", ".join([phrase] * max_repeats)

    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _REPEAT_RUN_RE.sub(_collapse, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-\u2026")
    return cleaned


def _merge_nearby_gaps(gaps: list, merge_gap_s: float = 2.5, max_chunk_s: float = 8.0) -> list:
    """Bundle nearby uncovered pitch gaps into a handful of phrase-length spans instead of
    dozens of individual pulses. Each merged span becomes exactly one forced-decode call
    below, which is what keeps the call count low without falling back to a single pass over
    the *entire* track, dumping a whole song into one decode is what buried short vocal
    bursts inside a mostly-silent/percussive window and made Whisper skip them outright
    (that was the previous version's regression). max_chunk_s puts a hard ceiling on how much
    gets merged into one clip for the same reason: even if the pitch detector finds long
    near-continuous stretches, capping each forced-decode clip at a few seconds keeps enough
    of a short-context advantage that Whisper can't just gloss over a burst the way it does
    inside a much longer window."""
    if not gaps:
        return []
    ordered = sorted(gaps, key=lambda g: g["start"])
    merged = [dict(ordered[0])]
    for g in ordered[1:]:
        candidate_end = max(merged[-1]["end"], g["end"])
        fits_gap = g["start"] - merged[-1]["end"] <= merge_gap_s
        fits_cap = candidate_end - merged[-1]["start"] <= max_chunk_s
        if fits_gap and fits_cap:
            merged[-1]["end"] = candidate_end
        else:
            merged.append(dict(g))

    # THE CEILING IS ENFORCED, NOT JUST WISHED FOR. Merging could only ever
    # respect max_chunk_s on the pieces it was joining; a single gap longer than
    # the cap went through whole, because there was nothing to merge it with and
    # nothing split it. That is not a small hole in the intent above -- it is the
    # case the intent exists for. The pitch scan reads a whole drum-and-bass
    # intro as one continuous pitched stretch, and the clip handed to Whisper was
    # therefore 28 seconds long, decoded in one call, and came back as a single
    # confident-sounding English line nobody sang. A stretch longer than the cap
    # is cut into equal pieces of at most the cap here, so every decode still
    # gets the short context the docstring above is about.
    split = []
    for m in merged:
        span_s = m["end"] - m["start"]
        if span_s <= max_chunk_s:
            split.append(m)
            continue
        pieces = int(np.ceil(span_s / max_chunk_s))
        step = span_s / pieces
        for k in range(pieces):
            split.append({"start": round(m["start"] + k * step, 2),
                          "end": round(m["start"] + (k + 1) * step, 2)})
    return split


def _extract_audio_clip_to_tempfile(raw_audio_bytes: bytes, start_s: float, end_s: float,
                                     pad_s: float = 0.15) -> str:
    """Trim raw_audio_bytes down to [start_s - pad_s, end_s + pad_s] and write it out as a
    16kHz mono wav. The small padding gives Whisper a sliver of surrounding audio so a burst
    right at the clip boundary doesn't get its onset/tail clipped mid-word."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
        tmp_in.write(raw_audio_bytes)
        tmp_in_path = tmp_in.name
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        clip_start = max(0.0, start_s - pad_s)
        duration = max(0.05, (end_s + pad_s) - clip_start)
        # -ss placed AFTER -i (output/decode seeking) instead of before it: input seeking on a
        # compressed format like mp3 is only approximately frame-accurate, and a clip that's
        # off by even a fraction of a second can hand Whisper the tail of one phrase and the
        # head of the next instead of the actual phrase the pitch detector flagged, which
        # would look exactly like "transcribing the wrong thing" even if the decode itself is
        # working correctly. Output seeking costs a bit more (ffmpeg decodes from the start of
        # the file to find the exact point) but guarantees the clip boundaries are the real
        # ones, and these clips are short enough that the extra cost is negligible.
        # Mono, at a TRUE average, for the same reason _decode_audio_via_ffmpeg
        # does it here: ffmpeg's `-ac 1` is (L+R)/sqrt(2) on stereo, 3 dB hot
        # and able to clip, and this clip is handed to Whisper, which is being
        # asked which WORDS are in it. A mono source keeps the plain `-ac 1`
        # path, where there is nothing to downmix and no gain to get wrong.
        try:
            channels = sf.info(tmp_in_path).channels
        except Exception:
            channels = 1
        downmix = (["-af", "pan=mono|c0=0.5*c0+0.5*c1"] if channels == 2 else ["-ac", "1"])
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_in_path, "-ss", f"{clip_start:.3f}",
             "-t", f"{duration:.3f}", *downmix, "-ar", "16000", out_path],
            capture_output=True, check=True,
        )
        return out_path
    except subprocess.CalledProcessError as e:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg clip extraction failed: {e.stderr.decode(errors='replace')[:300]}")
    finally:
        try:
            os.unlink(tmp_in_path)
        except OSError:
            pass


def _force_transcribe_clip(raw_audio_bytes: bytes, start_s: float, end_s: float,
                            language, model, backend) -> str:
    """Force a transcription attempt on a merged phrase-length clip the pitch detector has
    confirmed has a real singing voice somewhere in it, with Whisper's silence gate off so it
    hands back its best phonetic-English guess instead of being allowed to call it silence.

    Uses the SAME beam_size=5/best_of=5/full 6-step temperature ladder as the primary pass,
    not a cheaper decode. Two different cheap-decode shortcuts were tried here and both made
    output actively wrong rather than just slower to get:
    - beam_size=1 (greedy) commits early to a single continuation and can predict an
      end-of-segment token well before the audio actually ends, with no alternative hypotheses
      to fall back on, a 6-second clip collapsed to "Hey", a 10-second clip to "named".
    - Shortening the temperature ladder to (0.0, 0.4, 0.8) measurably broke the transcription
      itself, not just speed, confirmed the moment it shipped. The full ladder isn't just
      overhead on hard content; skipping steps like 0.2/0.6 changed which decode attempt ends
      up accepted, and the result was worse, not merely slower to arrive at.
    Both are reverted. The remaining lever for the cost this pass adds is genuine
    parallelism, see num_workers on the model loader, not cutting what each individual
    decode call does.

    initial_prompt=WHISPER_VOCABLE_PROMPT is a separate, still-active mitigation for a
    different problem: Whisper's decoder has a language model baked into its weights that
    actively prefers common, fluent English word sequences over unusual phonetic ones
    "well, hooray!" scores higher to it than "wha-he-oh" simply because "well, hooray" is a
    common English phrase and "wha-he-oh" isn't. That's not a decoding-parameter bug; it's the
    model doing what it was trained to do. Priming its context with an example of the
    hyphenated-vocable output style nudges it away from "autocorrecting" to a grammatical
    interjection, without fully eliminating the bias, that's baked into the model weights,
    not something any prompt turns off completely. A phone-level recognizer with no
    word/language model at all would fix this at the root (tested with Allosaurus against
    this project's reference track), but on audio with drums under the vocals it picks up
    percussion transients as spurious consonant clusters unless the voice is cleanly
    separated from the percussion first, that needs a source-separation stage (e.g. Demucs)
    in front of it, which is a meaningfully larger addition that hasn't been built or
    verified against this project's deployment yet."""
    clip_path = _extract_audio_clip_to_tempfile(raw_audio_bytes, start_s, end_s)
    _t0 = time.perf_counter()
    try:
        raw_segs = _whisper_transcribe_raw_segments(
            clip_path, language, model, backend, use_vad=False, no_speech_threshold=None,
            initial_prompt=WHISPER_VOCABLE_PROMPT)
    finally:
        try:
            os.unlink(clip_path)
        except OSError:
            pass
    print(f"[transcribe] timing: forced clip {start_s:.1f}-{end_s:.1f}s took "
          f"{time.perf_counter() - _t0:.1f}s wall-clock")
    joined = " ".join(s["text"] for s in raw_segs if s["text"]).strip()
    return _anglicize_cleanup(joined, max_repeats=2)


# WHAT WHISPER SAYS WHEN IT IS LISTENING TO AN INSTRUMENT.
#
# Whisper is a speech recognizer; it has no idea what music is, so on an
# instrumental passage it does what a language model does when it is handed
# something it cannot decode: it produces the most likely-looking caption, and
# on music that is very often the same handful of phrases every time. This
# project's own reference track came back with a line reading "Música" across a
# drum break -- a word nobody sang, in a language nobody sang it in, sitting in
# the transcript where an instrument label belongs.
#
# The captions come in TWO FAMILIES, and the difference between them is whether
# a real recording could ever contain the words as somebody's actual line:
#
#   * _WHISPER_MEDIUM_CAPTIONS -- the caption for the MEDIUM itself. "Música"
#     across a drum break, "Instrumental", "Subtitles by ...", "Amara.org",
#     "Please subscribe". Nobody sings these, so they are dropped wherever they
#     turn up, whether or not Whisper was confident about them: a confident
#     "Music Music Thank you" stretched over a sixteen-second drum intro is still
#     a caption, and it is the one that reached this project's own panel.
#   * _WHISPER_SOUND_CAPTIONS -- the caption for a SOUND, which a real recording
#     genuinely can contain somebody saying or singing ("bye bye", "okay",
#     "applause", "thank you"). These are dropped only where Whisper is NOT
#     confident, which is the tier already being shown as a lower-confidence
#     read, and never out of a line Whisper is sure it heard.
#
# Matching is against the WHOLE segment, and it tolerates the padding Whisper
# actually emits -- the doubled word in "Music Music", the "Thank you" that
# trails it -- through _WHISPER_CAPTION_GLUE: a segment counts as a caption when
# every word in it is either a caption or glue AND at least one word is really a
# caption. That is what keeps the failure direction safe: "music in my heart"
# keeps its "in" and its "heart", and "Honour song for the people" keeps
# "honour" and "people", so neither is ever mistaken for a caption.
_WHISPER_MEDIUM_CAPTIONS = frozenset({
    # What Whisper calls music. The first four are the forms its own training
    # data produces most, "Música" being the one this project actually hit.
    "musica", "music", "musique", "musik", "muzyka", "melodia", "melody",
    "tune", "instrumental", "instrumental music", "instrumental version",
    "cancion", "cancion instrumental", "song", "songs",
    # Credits and calls to subscribe: captions of the UPLOAD, not of the audio.
    "sous titres realises par", "sous titres realises", "sous titres",
    "subtitles by", "subtitles", "subtitled by",
    "amara org", "www amara org", "youtube", "youtube com", "www youtube com",
    "thanks for watching", "thank you for watching",
    "please subscribe", "like and subscribe", "subscribe",
    "subscribe to my channel",
})

# Captions for a SOUND rather than for the music. Kept separate because each of
# these is a thing a person can really say or sing, so only a read Whisper is
# already unsure of is allowed to lose it.
_WHISPER_SOUND_CAPTIONS = frozenset({
    "thank you", "thanks", "thank you very much",
    "merci", "merci beaucoup", "gracias", "muchas gracias", "danke", "dankeschon",
    "bye", "bye bye", "okay", "ok", "applause", "laughter", "silence",
})

# CREDIT MARKERS: the phrases that are only ever an upload's own subtitle credit,
# whatever else the line is padded with. Checked as a SUBSTRING of the normalized
# text, because the real thing wraps the marker in words this project cannot
# enumerate -- this project's own reference track produced "Sous-titrage Societe
# Radio-Canada" across a fourteen-second instrumental intro, which a word-by-word
# test cannot see ("Radio-Canada" is not a caption word, so the segment read as
# somebody's line). Every marker here is at least two words or a hyphenated
# compound in the original, so none of them is a word a lyric would contain by
# accident.
_WHISPER_CREDIT_MARKERS = (
    "titrage", "sous titres", "sous titre",
    "subtitles", "subtitled", "subtitle by",
    "amara org", "www amara", "youtube com", "www youtube",
)

# The padding a caption arrives wrapped in. "Music Music Thank you" is the same
# caption as "Music", and so is "Thank you for watching".
_WHISPER_CAPTION_GLUE = frozenset({
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "with", "this",
    "that", "is", "it", "so", "now", "very", "much", "here", "there",
    "you", "your", "my", "thank", "thanks", "please", "like", "watch",
    "watching", "subscribe", "www", "com", "org", "sous", "titres", "titre",
    "realise", "realises", "realises", "par", "by", "subtitled",
})

# Kept as the union of both families for anything that just needs the question
# "is this one of the captions" without caring which family it came from.
_WHISPER_MUSIC_ARTIFACTS = _WHISPER_MEDIUM_CAPTIONS | _WHISPER_SOUND_CAPTIONS


def _is_whisper_music_artifact(text: str, *, confident: bool = False) -> bool:
    """Is this whole segment one of Whisper's known non-speech artifacts?

    A WHOLE-SEGMENT test on normalized text: accents folded away, punctuation
    and the music symbols stripped, case and spacing flattened. A segment with
    any other real word in it fails the test and is kept as sung material, which
    is the failure direction that matters -- dropping a real line to tidy a
    transcript would be much worse than leaving one caption in it.

    `confident=True` is for segments that passed Whisper's own reliability
    checks. Those only lose the MEDIUM family (the music and credit captions,
    which nobody ever sings), never the SOUND family, and never the glue-padded
    match: a line Whisper is sure it heard is left alone unless it is
    recognisably the recognizer's own caption for the music.
    """
    import unicodedata
    raw = (text or "").strip()
    if not raw:
        return False
    if not any(ch.isalnum() for ch in raw):
        # Nothing but symbols: a music note, an ellipsis, a bar. There is no
        # word in this and never was.
        return True
    folded = unicodedata.normalize("NFKD", raw.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in folded)
    cleaned = " ".join(folded.split())
    if not cleaned:
        return False
    if cleaned in _WHISPER_MEDIUM_CAPTIONS:
        return True
    if cleaned in _WHISPER_SOUND_CAPTIONS:
        return not confident
    # A credit line, wherever it sits in the sentence and whatever else it is
    # wrapped in. This is the medium's own family, so it goes in both tiers.
    if any(marker in cleaned for marker in _WHISPER_CREDIT_MARKERS):
        return True
    words = cleaned.split()
    if len(words) > 8:
        # Long enough to be a sentence, so it is somebody's line whatever words
        # happen to be in it.
        return False
    # The doubled word Whisper pads captions with: "Music Music Thank you".
    deduped: list[str] = []
    for w in words:
        if deduped and deduped[-1] == w:
            continue
        deduped.append(w)
    joined = " ".join(deduped)
    if joined in _WHISPER_MEDIUM_CAPTIONS:
        return True
    if joined in _WHISPER_SOUND_CAPTIONS:
        return not confident
    if confident:
        # A line Whisper is confident about is only ever dropped for being a
        # caption word for word, never for being padded with glue.
        return False
    has_caption = False
    for w in words:
        if w in _WHISPER_MEDIUM_CAPTIONS or w in _WHISPER_SOUND_CAPTIONS:
            has_caption = True
        elif w not in _WHISPER_CAPTION_GLUE:
            # A word that is neither a caption nor padding: this is real text.
            return False
    return has_caption


def _uniform_vocal_word(label: str) -> "str | None":
    """The single word this label is nothing but repetitions of, or None.

    "Way" -> "way", and so is "Way Way" and "Way, way" -- which is what lets a
    run of three same-word rows merge in one pass instead of the first merge
    immediately looking like a two-word line and stopping the run.
    """
    words = re.sub(r"[^\w'\u2019-]+", " ", (label or "").lower()).split()
    if not words:
        return None
    return words[0] if all(w == words[0] for w in words) else None


def _collapse_repeated_vocal_rows(annotated: list, max_merged_s: float = 6.0,
                                   max_gap_s: float = 0.6) -> list:
    """One word sung over and over is ONE utterance, not seven rows.

    On wordless material the recognizer latches onto a single English word for
    each short window it is handed and emits it again and again a second or two
    apart -- this project's own reference track came back with six separate
    "Way" rows between 26 and 34 seconds. Every one of them is a real decode,
    and read one per line they make the log look broken and make the playback
    highlight jump from row to row for no musical reason.

    Rows are only merged when the decision is unambiguous: both rows are vocal
    (not an instrument label), both are the SAME single word, they are nearly
    touching, and the result is still short enough to be one phrase. The label
    is then run through the same repeat collapsing the anglicized path already
    uses (three rows reading "Way" come out as one row reading "Way, Way"), which
    is what that helper exists for.
    Word-level timings are dropped on a merged row: they describe the words the
    row no longer has, and the panel's own even-spacing fallback is the honest
    highlight for a collapsed vocable line.
    """
    out: list = []
    for seg in annotated:
        prev = out[-1] if out else None
        word = _uniform_vocal_word(seg.get("label", ""))
        same_word = (prev is not None
                     and word is not None
                     and word == _uniform_vocal_word(prev.get("label", ""))
                     and prev.get("type") == seg.get("type")
                     and prev.get("type") in ("vocal", "speech"))
        if (same_word
                and seg["start"] - prev["end"] <= max_gap_s
                and seg["end"] - prev["start"] <= max_merged_s):
            prev["label"] = _anglicize_cleanup(prev["label"] + " " + seg["label"], max_repeats=2)
            prev["end"] = max(prev["end"], seg["end"])
            prev["confident"] = bool(prev.get("confident")) and bool(seg.get("confident"))
            prev.pop("words", None)
        else:
            out.append(seg)
    return out


def _classify_segments(raw_segments: list) -> list:
    """Three-tier classification instead of a blunt confident/discarded split.

    Tier "speech", passes all of Whisper's own reliability checks: shown as normal,
                         high-confidence English text (this is where genuine spoken English,
                         e.g. the emcee's introduction, ends up).
    Tier "vocal", fails the strict checks but Whisper still heard something
                         voice-like (no_speech_prob isn't near-certain silence/noise) and
                         produced text: shown as an anglicized, lower-confidence line after
                         cleanup. This is where sung Cree and wordless vocables land, instead
                         of being thrown away as a generic "[non-lexical vocals]" tag.
    Tier "silent", no_speech_prob is high AND there's no usable text left after
                         cleanup: genuinely nothing lexical here. Left unlabeled; the
                         instrument/percussion pass fills these gaps in afterward.
    """
    annotated = []
    for s in raw_segments:
        passes_strict_check = (
            s["no_speech_prob"] <= WHISPER_NO_SPEECH_THRESHOLD
            and s["avg_logprob"] >= WHISPER_LOGPROB_THRESHOLD
            and s["compression_ratio"] <= WHISPER_COMPRESSION_RATIO_THRESHOLD
            and s["text"]
        )
        # A SEGMENT MADE ONLY OF SYMBOLS CAN NEVER BE A LYRIC, whatever
        # Whisper's own confidence says about it. The caption artifacts that
        # matter most on an instrumental (a line of music notes, an ellipsis)
        # carry no letters at all, and one of them arriving through the strict
        # checks was the one way a music caption could still reach the panel as
        # if it were text. It is demoted here rather than deleted, so the
        # artifact filter below (and then the instrument pass) takes it.
        if passes_strict_check and not any(ch.isalnum() for ch in s["text"]):
            print(f"[transcribe] a music-symbol segment ({s['text']!r}) passed the strict "
                  f"checks at {s['start']:.1f}-{s['end']:.1f}s; treating it as the "
                  f"non-linguistic material it is.")
            passes_strict_check = False
        # ...AND NEITHER CAN THE RECOGNIZER'S OWN CAPTION FOR THE MUSIC, however
        # sure of it Whisper is. A confident "Music" or "Instrumental" over a
        # drum break is a segment this pass would otherwise show as a lyric,
        # which is the one way a caption of the medium itself could still reach
        # the panel. Only the music/credit family is tested here (see
        # _is_whisper_music_artifact's `confident` argument), so a line Whisper
        # is sure it heard is never dropped for being a single ordinary word.
        if passes_strict_check and _is_whisper_music_artifact(s["text"], confident=True):
            print(f"[transcribe] dropped Whisper's own caption {s['text']!r} at "
                  f"{s['start']:.1f}-{s['end']:.1f}s even though it passed the strict "
                  f"checks; the gap gets the instrument label instead.")
            passes_strict_check = False
        if passes_strict_check:
            seg = {
                "start": round(s["start"], 2), "end": round(s["end"], 2),
                "label": s["text"], "confident": True, "type": "speech",
            }
            if s.get("words"):
                seg["words"] = [{"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
                                 for w in s["words"] if w["word"]]
            annotated.append(seg)
            continue

        # Not a clean pass, but was there anything voice-like here at all? A no_speech_prob
        # near 1.0 means Whisper itself thinks this stretch is silence or non-vocal noise, in
        # which case there's nothing to anglicize and it should fall through to the
        # instrumental pass instead of showing an empty or fabricated line.
        anglicized = _anglicize_cleanup(s["text"]) if s["no_speech_prob"] < 0.92 else ""
        if anglicized and _is_whisper_music_artifact(anglicized):
            # Not a lyric: the recognizer's own caption for music. Drop it here
            # so this stretch falls through to the instrument pass below, which
            # names what is actually playing in it.
            print(f"[transcribe] dropped Whisper's non-speech artifact "
                  f"{anglicized!r} at {s['start']:.1f}-{s['end']:.1f}s; the gap "
                  f"gets the instrument label instead.")
            anglicized = ""
        if anglicized:
            annotated.append({
                "start": round(s["start"], 2), "end": round(s["end"], 2),
                "label": anglicized, "confident": False, "type": "vocal",
            })
        # else: genuinely nothing lexical/vocal here, leave the gap for instrument detection.
    return annotated


# ONE NAME PER SOUND, WHICH IS NOT ONE NAME PER PANNS CLASS -- and that difference is the
# whole reason a straight 2 s window used to become two or three overprinted rows. PANNs'
# own list is far finer than anything a listener would say: "Wood block", "Drum", "Drum kit",
# "Snare drum" and "Hi-hat" are five names for one thing a person hears, and on a real
# recording they trade places from window to window, so a single drum pattern came back as a
# stack of "<drum playing>" / "<wood block playing>" rows covering each other's time. Every
# class a row may NAME is mapped here to the name a person would use for it, so consecutive
# windows that keep hearing the same family merge into the one row the ear actually hears.
# (The mapping is also the allow-list: a class that is not in it is not something this project
# is willing to name, which is what keeps a label honest instead of merely confident.)
_PANNS_INSTRUMENT_LABELS = {
    # Percussion: the kit, and everything else that is hit rather than pitched.
    "Drum": "drums playing", "Drum kit": "drums playing", "Bass drum": "drums playing",
    "Snare drum": "drums playing", "Hi-hat": "drums playing", "Cymbal": "drums playing",
    "Tambourine": "tambourine playing", "Rattle (instrument)": "rattle playing",
    "Maraca": "shaker playing", "Wood block": "wood block playing",
    "Marimba, xylophone": "mallet percussion playing",
    "Glockenspiel": "bells playing", "Chime": "bells playing", "Bell": "bells playing",
    # Low end, where a mistake between two candidates is most audible.
    "Double bass": "double bass playing", "Bass guitar": "bass guitar playing",
    # Strings.
    "Violin, fiddle": "violin playing", "Viola": "viola playing", "Cello": "cello playing",
    "Harp": "harp playing", "Banjo": "banjo playing", "Mandolin": "mandolin playing",
    "Ukulele": "ukulele playing", "Sitar": "sitar playing",
    "Steel guitar, slide guitar": "slide guitar playing",
    "Guitar": "guitar playing", "Electric guitar": "electric guitar playing",
    # Keys.
    "Piano": "piano playing", "Electric piano": "electric piano playing",
    "Organ": "organ playing", "Harpsichord": "harpsichord playing",
    # Winds and brass.
    "Flute": "flute playing", "Clarinet": "clarinet playing", "Oboe": "oboe playing",
    "Bassoon": "bassoon playing", "Saxophone": "saxophone playing",
    "Trumpet": "trumpet playing", "Trombone": "trombone playing",
    "French horn": "french horn playing", "Brass instrument": "brass playing",
    "Accordion": "accordion playing", "Bagpipes": "bagpipes playing",
    "Didgeridoo": "didgeridoo playing", "Shofar": "shofar playing",
}

# AudioSet classes that indicate wordless vocal sound (chant, humming, vocalizing) rather than
# an instrument. Used only for gaps Whisper produced no usable text for at all; it never feeds
# back into the "vocal" tier above, which already comes straight from Whisper's own
# (anglicized) text.
#
# THE LABEL NAMES THE VOICE, NOT A CATEGORY. This used to be one string, "<wordless vocals>",
# for all ten classes -- a singer who is humming and a choir holding a chant were written the
# same way, in the one phrasing that says nothing at all about what is being heard. Each
# class now names the sound it stands for ("<humming>", "<chanting>", "<choir singing>", ...),
# which is both more use to the person reading the log and strictly more honest: PANNs is being
# asked which of these it hears, and this is the answer it gave.
_PANNS_VOICE_LABELS = {
    "Singing": "singing",
    "Vocal music": "singing",
    "A capella": "a cappella singing",
    "Choir": "choir singing",
    "Male singing": "a man singing",
    "Female singing": "a woman singing",
    "Child singing": "a child singing",
    "Chant": "chanting",
    "Humming": "humming",
    "Yodeling": "yodeling",
}

# Everything PANNs is allowed to put a name on, either half. A class outside this set is
# dropped silently rather than shown (see _name_span_from_panns).
_PANNS_LABELS = {**_PANNS_INSTRUMENT_LABELS, **_PANNS_VOICE_LABELS}

# Below this, PANNs is not saying anything worth writing down. Deliberately where it is: a
# mislabelled instrument is worse than no label at all (this project's own rule for the
# fallback path too), and the scores on real music are low -- this project's reference track
# peaks around 0.3, and most windows never pass 0.2.
PANNS_MIN_SCORE = 0.15

# How much of a stretch the tagger has to account for before its name may stand in for a
# stretch Whisper heard no words in (see _what_is_sounding). Half is the same bar the
# vocal/instrument subtraction uses: a name that only explains a corner of the row is not
# the name of that row.
PANNS_SPAN_COVERAGE_MIN = 0.5

panns_model = None
panns_labels = None
try:
    from panns_inference import AudioTagging, labels as _panns_labels_list
    panns_model = AudioTagging(checkpoint_path=None, device="cpu")
    panns_labels = _panns_labels_list
    print("PANNs audio-tagging model loaded (per-instrument detection enabled).")
except Exception as e:
    print(f"PANNs not available ({e}), instrument spans will use the generic rhythm-only "
          f"fallback ('<instruments playing>' instead of a named instrument). Run "
          f"'pip install panns-inference torchlibrosa' and restart the server to enable real "
          f"per-instrument labels (violin, flute, drum, guitar, ...).")
    panns_model, panns_labels = None, None


def _name_instrument_span(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> str:
    """What is playing through one span, from what this project can measure
    itself.

    PANNs is the good answer here (see _detect_instrument_spans) and this is
    what runs when it is not installed, because a transcript that says
    "<instruments playing>" everywhere names nothing at all -- and "say the
    instrument that is playing" is the whole point of these rows. So this asks
    the two questions the signal can actually answer, and answers only those:

      * is this stretch hits or notes? Harmonic/percussive separation (the
        project's own librosa.effects.hpss, the same measure the light
        instrumental gate already uses) says whether what is there is mostly
        transients (a kit, a shaker, claps) or mostly sustained tone.
      * if it is tone, is the tone down low? A span whose energy sits under a
        couple of hundred hertz with little above it is a bass, not a fiddle.

    Anything it cannot tell apart stays the honest generic label. A
    mislabelled instrument would be worse than no label, which is why this
    never guesses between two instruments it cannot separate.
    """
    try:
        import librosa
        seg = np.asarray(audio, dtype=np.float32)[int(start_s * sr):int(end_s * sr)]
        if seg.size < int(0.2 * sr):
            return "instruments playing"
        harmonic, percussive = librosa.effects.hpss(seg)
        h_e = float(np.sum(harmonic.astype(np.float64) ** 2))
        p_e = float(np.sum(percussive.astype(np.float64) ** 2))
        if h_e + p_e <= 0:
            return "instruments playing"
        if p_e > 1.6 * h_e:
            return "percussion playing"
        # Tone: is its weight down in the bass? Compare the energy below and
        # above 250 Hz on the harmonic part.
        spec = np.abs(np.fft.rfft(harmonic.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(harmonic), d=1.0 / sr)
        total = float(spec.sum())
        if total <= 0:
            return "instruments playing"
        low = float(spec[freqs <= 250.0].sum()) / total
        if low >= 0.6:
            return "bass playing"
        return "instruments playing"
    except Exception as e:
        print(f"[transcribe] could not name the span at {start_s:.1f}-{end_s:.1f}s ({e}); "
              f"using the generic label.")
        return "instruments playing"


def _detect_instrument_spans(raw_bytes: bytes, min_duration_s: float) -> list:
    model, labels = panns_model, panns_labels

    if model is None:
        fallback_sr = 22050
        fb_audio = _decode_audio_via_ffmpeg(raw_bytes, fallback_sr)
        duration = len(fb_audio) / fallback_sr
        window_s = min(2.0, max(0.5, duration))
        spans = _detect_percussive_spans(fb_audio, fallback_sr, window_s=window_s, min_duration_s=min_duration_s)
        return [{"start": s, "end": e,
                 "label": f"<{_name_instrument_span(fb_audio, fallback_sr, s, e)}>",
                 "confident": False, "type": "instrumental"} for s, e in spans]

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
        # ONE WINDOW, ONE NAME -- THE STRONGEST ONE.
        #
        # This used to add a span for every class in the top 5 that passed the score
        # threshold, instruments and voices alike, which is how one window could hand back
        # both "<drum playing>" and "<wordless vocals>" over the same two seconds, and two
        # instrument classes over it as well. Those overlapped each other in the timeline
        # (nothing subtracts an instrument span from another instrument span) and the
        # transcript showed the same moment two or three times, in two or three different
        # names, while the strongest reading -- the one the model actually preferred -- was
        # buried under the weaker ones. A window is one moment, so it gets one name: the
        # highest scoring class that this project is willing to name. top_idx is already
        # sorted, so the first class in the allow-list with an acceptable score IS the best
        # one, and the loop stops there rather than continuing to weaker alternatives.
        for idx in top_idx:
            label = labels[idx]
            score = float(clipwise_output[0][idx])
            if score < PANNS_MIN_SCORE:
                break
            name = _PANNS_LABELS.get(label)
            if not name:
                continue
            start_t = i / panns_sr
            end_t = min((i + win_samples) / panns_sr, duration)
            raw_spans.append({"start": round(start_t, 2), "end": round(end_t, 2),
                              "label": f"<{name}>",
                              "confident": False, "type": "instrumental"})
            break
        i += hop_samples

    raw_spans.sort(key=lambda s: (s["label"], s["start"]))
    merged = []
    for s in raw_spans:
        if merged and merged[-1]["label"] == s["label"] and s["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= min_duration_s]


def _what_is_sounding(start_s: float, end_s: float, instrument_spans: list) -> tuple:
    """The honest name for a stretch Whisper heard no words in, and whether that
    name is an instrument's or a voice's.

    WHY THIS EXISTS AT ALL. The forced-decode path below hands a pitch-scanned
    clip to Whisper and, when the decode comes back empty, used to write
    "<wordless vocals>" over the whole clip no matter what was actually there.
    But an empty decode does not mean a voice: the pitch scan that nominated the
    clip is a fundamental-frequency detector, and a drum pattern, a bass line or
    a didgeridoo all present one to it just as a voice does, which is how a
    sixteen-second drum-and-bass intro came back to the panel as one long row
    reading "<wordless vocals>". The tagger has already answered the question
    that matters -- WHAT is playing -- in the same run (the instrument pass is
    started in parallel with the Whisper decode, see
    _transcribe_track_annotated), so this asks it instead of guessing.

    Returns (label, kind) where kind is "instrumental" when the name came from
    the tagger's own reading of the audio and "vocal" when nothing named that
    stretch and the fallback voice label has to stand.
    """
    length = max(1e-6, end_s - start_s)
    overlap: dict = {}
    for span in instrument_spans or []:
        shared = min(end_s, span["end"]) - max(start_s, span["start"])
        if shared <= 0:
            continue
        overlap[span["label"]] = overlap.get(span["label"], 0.0) + shared
    if overlap:
        label, covered = max(overlap.items(), key=lambda kv: kv[1])
        if covered / length >= PANNS_SPAN_COVERAGE_MIN:
            return label, "instrumental"
    return "<wordless vocals>", "vocal"


def _subtract_intervals(span: dict, occupied: list) -> list:
    """Cut the parts of `span` that overlap any interval already claimed by a vocal/speech
    segment, returning zero or more leftover pieces. This is what keeps the transcript from
    showing an "<instruments playing> 0:00-0:34" row sitting on top of a confident
    "Honour song for..." row at 0:00-0:02, the vocal segment always wins the overlap, and
    the instrumental row only covers what's actually left over."""
    pieces = [(span["start"], span["end"])]
    for occ_start, occ_end in occupied:
        next_pieces = []
        for s, e in pieces:
            if occ_end <= s or occ_start >= e:
                next_pieces.append((s, e))  # no overlap with this occupied interval
                continue
            if occ_start > s:
                next_pieces.append((s, occ_start))
            if occ_end < e:
                next_pieces.append((occ_end, e))
        pieces = next_pieces
    return [{**span, "start": round(s, 2), "end": round(e, 2)} for s, e in pieces]


def _add_percussive_segments(annotated: list, raw_bytes: bytes, min_duration_s: float,
                              instrument_spans: "list | None" = None) -> list:
    """Add instrument/percussion spans, but only into the gaps the vocal pass left uncovered,
    so the final transcript is a single ordered, non-overlapping timeline instead of two
    independently-generated layers stacked on top of each other.

    instrument_spans lets a caller pass in an already-computed result (e.g. from a background
    thread started at the same time as the Whisper decode) instead of paying for
    _detect_instrument_spans a second time, same detection, just not redone."""
    occupied = sorted((seg["start"], seg["end"]) for seg in annotated)
    if instrument_spans is None:
        try:
            instrument_spans = _detect_instrument_spans(raw_bytes, min_duration_s)
        except Exception as e:
            print(f"[transcribe] instrument detection skipped: {e}")
            instrument_spans = []

    # EARLIER MATERIAL WINS ITS OWN MOMENT, and what has already been claimed is
    # added to `occupied` as the loop goes. Without that, two spans that name
    # DIFFERENT things can still overlap in time -- a window that hears a drum
    # kit and the next one, a second later, that hears a wood block each cover
    # the moment they share -- and the panel showed that moment twice, in two
    # names, as if two things were playing where one is. A span now only ever
    # fills the time nothing has filled yet, so the timeline is one ordered,
    # non-overlapping sequence, which is what a reader assumes a transcript is.
    # Ordered by start rather than by label, because the transcript reads left to
    # right and the earlier reading is the one made with an unbroken view of the
    # music before it.
    for span in sorted(instrument_spans, key=lambda s: (s["start"], -s["end"])):
        for piece in _subtract_intervals(span, occupied):
            if piece["end"] - piece["start"] >= min_duration_s:
                annotated.append(piece)
                occupied.append((piece["start"], piece["end"]))
        occupied.sort()

    annotated.sort(key=lambda seg: seg["start"])

    # Adjacent pieces of the same label that ended up back-to-back after
    # subtraction (e.g. a drum span split around a short vocal line) read better merged into
    # one row than as two near-identical rows a fraction of a second apart. The test is on
    # type == "instrumental", which is what a row the forced-decode path named after an
    # instrument gets too (see _what_is_sounding): a long intro cut into several clips
    # (see _merge_nearby_gaps) is ONE stretch of a drum kit in the ear, and it has to read as
    # one row here, not one row per decode.
    merged = []
    for seg in annotated:
        if (merged and merged[-1].get("type") == "instrumental" == seg.get("type")
                and merged[-1]["label"] == seg["label"] and seg["start"] - merged[-1]["end"] <= 0.5):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def _transcribe_audio_blob(raw: bytes, language: str = "en") -> dict:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        if os.path.getsize(tmp_path) < 512:
            return {"text": "", "bytes_received": len(raw), "segment_count": 0, "no_speech_probs": [],
                    "segments": [], "has_lexical_speech": False}

        raw_segments = _whisper_transcribe_raw_segments(
            tmp_path, language, whisper_model_mic, whisper_backend_mic, use_vad=True)
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


def _span_is_sung(audio: np.ndarray, sr: int, start_s: float, end_s: float,
                  min_voiced_ratio: float = 0.4) -> bool:
    """Is a person singing through this stretch? Asked only about candidate
    percussive spans, and only as a veto on them (see _detect_percussive_spans).

    A pitch tracker is the one thing a drum break cannot fool for long: cymbals
    and snares come back unvoiced, while syllables come back voiced. Read at
    one hop length, over the span alone, so this costs a fraction of a second
    per candidate.
    """
    import librosa
    seg = np.asarray(audio, dtype=np.float32)[int(start_s * sr):int(end_s * sr)]
    if seg.size < int(0.25 * sr):
        return False
    try:
        _f0, voiced, _prob = librosa.pyin(seg, fmin=65.0, fmax=1200.0, sr=sr,
                                          frame_length=2048, hop_length=512)
    except Exception:
        return False
    if voiced is None or not np.any(voiced):
        return False
    return float(np.mean(np.asarray(voiced, dtype=bool))) >= min_voiced_ratio


def _detect_percussive_spans(audio: np.ndarray, sr: int, window_s: float = 2.0,
                              density_threshold: float = 1.5, min_duration_s: float = 2.0) -> list:
    """Stretches that are dense with onsets: where the robot does not sing,
    because the track is playing a percussion break rather than a voice.

    DENSE ONSETS ALONE ARE NOT PERCUSSION, and on a solo vocal take that is not
    a small point: syllables are onsets, an ordinary phrase runs well past 1.5
    of them a second, so the whole take reads as one percussive span, every
    pitched frame inside it is muted, and the accompaniment engine is handed a
    track it hears as having nothing sung in it -- which is how a 3.3 s solo
    take came back from Call and response as a 422 refusal ("no melodic
    material was detected") with the panel's own setting, while the same render
    with instruments on, which skips this detector entirely, worked. A span is
    therefore only percussion if nobody is SINGING through it: any candidate
    with a confidently voiced pitch track is dropped here, and only the ones
    that are genuinely unpitched -- a drum break, a cymbal wash -- survive.
    """
    import librosa
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    peak_idx = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=5)
    peak_times = times[peak_idx]

    if not len(peak_times) or not len(times):
        return []

    duration = float(times[-1])
    candidates = []
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
                candidates.append((round(cur_start, 2), round(t, 2)))
            cur_start = None
        t += window_s
    if cur_start is not None and duration - cur_start >= min_duration_s:
        candidates.append((round(cur_start, 2), round(duration, 2)))

    spans = []
    for start_s, end_s in candidates:
        if _span_is_sung(audio, sr, start_s, end_s):
            continue
        spans.append((start_s, end_s))
    return spans


def _transcribe_track_annotated(raw: bytes, language: str = "en") -> dict:
    # Real timing instead of guessing: every round of "make it faster" so far has been aimed at
    # a theory of where the time goes, and this logs the actual wall-clock breakdown of each
    # stage so the next bottleneck (if there is one) shows up in the logs instead of needing
    # another guess-and-check pass.
    _t_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        # The primary Whisper decode, the pitch-presence scan, and the PANNs instrument
        # classification pass are all independent of each other, none of them needs
        # another's output, they only need the raw audio, but they were running one after
        # another, which is pure wasted wall-clock time on top of however long each one
        # genuinely takes. This does not change a single threshold, decode parameter, or
        # classification rule; it computes exactly the same three results, just overlapping in
        # time instead of stacked. use_vad=False on the primary pass: this is a full song
        # upload, not a live mic stream, so we want a decode attempt across the *entire* file
        # rather than letting speech-tuned VAD silently drop the sung/chanted stretches before
        # Whisper ever sees them (see the docstring on _whisper_transcribe_raw_segments).
        _t_stage = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            future_pass1 = pool.submit(
                _whisper_transcribe_raw_segments, tmp_path, language,
                whisper_model_track, whisper_backend_track, use_vad=False)
            future_pitch = pool.submit(_detect_vocal_pitch_spans, raw)
            future_instruments = pool.submit(_detect_instrument_spans, raw, 2.0)

            raw_segments = future_pass1.result()
            _t_whisper_pass1 = time.perf_counter() - _t_stage
            try:
                pitch_spans = future_pitch.result()
            except Exception as e:
                print(f"[transcribe] pitch-based vocal detection skipped: {e}")
                pitch_spans = []
            _t_pitch = time.perf_counter() - _t_stage
            try:
                instrument_spans = future_instruments.result()
            except Exception as e:
                print(f"[transcribe] instrument detection skipped: {e}")
                instrument_spans = []
            _t_instruments = time.perf_counter() - _t_stage
        print(f"[transcribe] timing: STAGE 1/3 (whisper main pass + pitch scan + instrument "
              f"scan, overlapped) done in {time.perf_counter() - _t_stage:.1f}s, NOT the "
              f"total, forced-decode chunks and final assembly still to come "
              f"(whisper pass1 alone would be ~{_t_whisper_pass1:.1f}s, pitch scan "
              f"~{_t_pitch:.1f}s, instrument scan ~{_t_instruments:.1f}s, these three "
              f"overlapped, so the stage total above is roughly the largest of them, not "
              f"the sum)")

        # The three-tier classifier below is what separates real speech from anglicized
        # vocalizing from actual silence, not VAD. Even with VAD off, Whisper's own
        # no_speech_threshold check can still decide a whole stretch is silent and drop it
        # before it ever becomes a raw segment at all, which is exactly what was collapsing
        # every sung/chanted pulse in the track into one giant "[instrumental / percussion]"
        # block. The pitch detector is the ground truth here: if it finds a real fundamental
        # frequency in a stretch Whisper left uncovered, that stretch gets an anglicized line,
        # no exceptions.
        annotated = _classify_segments(raw_segments)

        occupied = sorted((seg["start"], seg["end"]) for seg in annotated)
        uncovered_gaps = [
            gap for p_start, p_end in pitch_spans
            for gap in _subtract_intervals({"start": p_start, "end": p_end}, occupied)
            if gap["end"] - gap["start"] >= 0.3
        ]

        # Nearby gaps are bundled into a handful of phrase-length clips (a few seconds each)
        # rather than either one decode per pitch pulse (correct, but ~20-30 model calls on a
        # short track, the original slowdown) or one decode over the entire file (fast, but
        # it buries each short burst inside a mostly-silent/percussive window and Whisper just
        # omits it, the regression that dropped every vocal line down to "[wordless
        # vocalizing]"). Each merged clip gets exactly one forced, full-quality decode call
        # (see _force_transcribe_clip for why it isn't a cheaper one) and becomes one
        # continuous segment, which also fills in the sub-2-second silences between pulses that
        # were falling through both the vocal and instrumental passes entirely. The clips are
        # independent of each other, different, non-overlapping audio, so they're also run
        # concurrently instead of one at a time.
        merged_chunks = _merge_nearby_gaps(uncovered_gaps, merge_gap_s=2.5)
        _t_forced = time.perf_counter()
        if merged_chunks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(merged_chunks))) as pool:
                chunk_futures = [
                    pool.submit(_force_transcribe_clip, raw, chunk["start"], chunk["end"],
                                language, whisper_model_track, whisper_backend_track)
                    for chunk in merged_chunks
                ]
                for chunk, future in zip(merged_chunks, chunk_futures):
                    text = future.result()
                    # THE FORCED PASS IS WHERE THE LAST HALLUCINATION HID. These
                    # chunks are stretches the main pass never covered and the
                    # pitch scan says are sung, so the decode is being asked
                    # about audio Whisper has no context for -- which is exactly
                    # when it answers with a stock artifact ("Bye-bye", "Thank
                    # you.", a music note) instead of admitting it heard
                    # nothing. The classifier filters the main pass; this path
                    # built its rows by hand and skipped that filter, so a
                    # hallucinated English line from here reached the panel as a
                    # real lyric. Same test, same failure direction as
                    # everywhere else: only a whole-segment artifact is
                    # replaced, and a label with any other content in it is
                    # kept exactly as decoded.
                    #
                    # AND THE FALLBACK IS THE ANGLE-BRACKETED NAME, not a
                    # square-bracketed one: every label this project invents is
                    # written <like this>, and a label still in square brackets
                    # is a Whisper tag the panel is supposed to be hiding. The
                    # chunk is vocal by construction (the pitch scan said so),
                    # so the honest name for it is the same one the
                    # anglicized-sung path uses.
                    label = text.strip() if text else ""
                    row_type = "vocal"
                    onset_reason = ""
                    if not label or _is_whisper_music_artifact(label):
                        # Say so, rather than only showing the result: this is the
                        # path a caption like "Music Music Thank you" takes when
                        # the main pass never covered the stretch, and a silent
                        # substitution here is impossible to tell from a decode
                        # that genuinely came back empty.
                        if label:
                            onset_reason = f"Whisper's own caption {label!r}"
                        else:
                            onset_reason = "an empty decode"
                        # AND THE NAME COMES FROM WHAT IS ACTUALLY PLAYING, not from
                        # the shape of the decode. See _what_is_sounding: an empty
                        # decode over a drum intro is not wordless singing, and the
                        # tagger already said which it is. When it names an
                        # instrument the row is typed as one, so that the instrument
                        # pass's own gap fill does not then write the same two
                        # seconds a second time, and so that neighbouring rows of the
                        # same name merge into one (see _add_percussive_segments).
                        label, row_type = _what_is_sounding(chunk["start"], chunk["end"],
                                                            instrument_spans)
                        print(f"[transcribe] forced decode of {chunk['start']:.1f}-"
                              f"{chunk['end']:.1f}s produced {onset_reason}; naming the "
                              f"stretch {label!r} instead.")
                    annotated.append({
                        "start": chunk["start"], "end": chunk["end"],
                        "label": label,
                        "confident": False, "type": row_type,
                    })
            annotated.sort(key=lambda seg: seg["start"])
        # One repeated word over and over is one utterance, not six rows (see
        # _collapse_repeated_vocal_rows). Runs before the instrument gap-fill so
        # the two passes agree on what the vocal timeline actually covers.
        annotated = _collapse_repeated_vocal_rows(annotated)
        print(f"[transcribe] timing: STAGE 2/3 ({len(merged_chunks)} forced-decode chunk(s)) "
              f"took {time.perf_counter() - _t_forced:.1f}s, NOT the total, instrument "
              f"gap-fill and final assembly still to come")

        annotated = _add_percussive_segments(annotated, raw, min_duration_s=2.0,
                                              instrument_spans=instrument_spans)

        print(f"[transcribe] timing: STAGE 3/3 DONE, TOTAL transcription time "
              f"{time.perf_counter() - _t_start:.1f}s (this is the real end-to-end number)")
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
            print(f"[transcribe] empty result, {diag['bytes_received']} bytes received, "
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
                hz, conf = await asyncio.to_thread(pitch.detect, pitch_input)
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
    if neural_timbre is None or not neural_timbre.enabled:
        return audio
    return neural_timbre.convert(audio, sample_rate, decision.target_hz, voice_index=0)


def _declick_loop_seam(audio: np.ndarray, fade_samples: int = 128) -> np.ndarray:
    """
    A click remover, not a transformation of the sound. Looping ANY
    recording that doesn't already end exactly where it began produces
    an audible digital click at the seam, a sharp, broadband
    discontinuity that isn't part of the recording, just an artifact of
    cutting it into a loop. This fades only the first/last few
    milliseconds down toward silence (NOT toward each other, this
    isn't a crossfade blend of head and tail content, which would mix
    the two together and is exactly the kind of transformation this was
    asked not to do) so the seam is a silent gap instead of a click.
    Every other sample of the actual recording is completely untouched;
    at 44.1kHz, 128 samples is under 3ms, shorter than the ear's
    integration time for pitch/timbre perception, so this doesn't
    change how the word sounds, only removes the click at the cut.
    """
    n = len(audio)
    fade_samples = min(fade_samples, n // 4)
    if fade_samples < 2:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


class _SyllableCycler:
    """
    Real-time counterpart to _render_track_single_vocable_offline /
    _segment_into_syllables, see that function's docstring for the
    full reasoning. Pitch is never touched here either: every sample
    played back is byte-for-byte from the original recording.

    The recording is split into its own natural syllables once (on
    load). Each time a NEW onset is detected in the live singing (via
    RhythmAnalyzer's own onset detector, passed in as `onset_id`, any
    value that changes exactly when a new onset fires; its caller passes
    rhythm.latest_onset), this advances to the next syllable in order
    and restarts it from its own beginning, landing on the singer's
    actual rhythm instead of an arbitrary loop position. Between onsets,
    the current syllable simply loops on itself (click-free) to sustain
    through a held note. A genuine new PHRASE (real silence, amplitude
    VAD, not a pitch-confidence blip) resets back to the first syllable,
    the same way a person starts the word over from its beginning each
    time they start singing again rather than resuming mid-word.

    Every phrase also gets a natural ATTACK when it starts, ramping up
    from silence over ATTACK_S instead of snapping to full volume
    instantly, since a real voice doesn't switch on like a light switch.

    On the way OUT, this does NOT fade out wherever the silence happened
    to begin, it finishes playing the CURRENT syllable to its own full
    natural length first (one whole "La", not half of one), and only
    THEN actually goes silent. That syllable's own edges are already
    declicked (see _segment_into_syllables), so letting it finish is
    already a smooth, natural close on its own; no separate fade needed.
    If singing resumes before that syllable finishes, nothing was ever
    cut short in the first place, it just continues.

    Volume (never pitch) continuously follows the input's own loudness,
    identical mechanism to before, this envelope is layered on top of
    that, not a replacement for it.
    """
    ATTACK_S = 0.05
    GAP_TOLERANCE_S = 2.5  # a real, sustained stop in the singing, not a breath, consonant, or word boundary; see next_chunk

    def __init__(self):
        self.vocable: str | None = None
        self.syllables: list | None = None
        self.syllable_idx = 0
        self.read_pos = 0
        self.was_voiced = False
        self._last_onset_id = None
        self._typical_rms: float | None = None
        self._gain = 1.0
        self._silence_s = 0.0    # elapsed real time since voicing was last seen, for gap-bridging
        self._phrase_env = 0.0   # 0..1 attack/release envelope for the current phrase
        # Real-time counterpart of the offline renderer's splice
        # crossfade (see _apply_internal_crossfades), the exact same
        # "two independently-declicked syllables placed next to each
        # other dip toward silence at the seam" issue applies here too,
        # it's just spread across live chunk boundaries instead of one
        # in-memory buffer. `_last_tail` remembers the real last few
        # samples actually returned so the NEXT call can blend against
        # them even when the seam falls exactly on a chunk boundary;
        # None means "nothing to blend against" (fresh word load, or
        # right after a genuine phrase restart from real silence, where
        # blending against old, unrelated material would be wrong).
        self._last_tail: np.ndarray | None = None
        self._xfade_n = 0
        self._in_release_tail = False  # True while playing out the current syllable after a genuine phrase end
        self._release_tail_s = 0.0     # elapsed time spent in that tail, so it can be capped, see next_chunk

    def _ensure_loaded(self, vocable: str):
        if vocable == self.vocable and self.syllables is not None:
            return
        global synthesizer
        bank = getattr(synthesizer, "_neural_bank", None) if synthesizer else None
        bases = bank._bases.get(vocable) if bank else None
        if not bases:
            self.syllables = None
            return
        raw_audio = np.asarray(bases[0].audio, dtype=np.float32)
        dsp_cfg = _dsp_config_for(vocable)
        self.syllables = _segment_into_syllables(raw_audio, bank.sample_rate, min_syllable_s=dsp_cfg["min_syllable_s"])
        self.vocable = vocable
        self.syllable_idx = 0
        self.read_pos = 0
        self._xfade_n = max(24, int(dsp_cfg["xfade_ms"] / 1000 * bank.sample_rate))
        self._last_tail = None  # a different word shares no waveform to blend against

    def _read_syllables(self, n_samples: int) -> np.ndarray:
        """Fills n_samples by reading forward through the syllable
        sequence, continuing into the next syllable once the current one
        is used up, shared by normal playback and the release tail
        below, since a release still needs real recorded material to
        fade out, not silence.

        FIX: every advance to a new syllable used to just concatenate
        its (already independently head/tail-declicked) audio directly
        onto whatever came before, click-free but with a real dip toward
        silence right at the seam, see _apply_internal_crossfades'
        docstring for the full reasoning (this is the live-path
        counterpart of the exact same fix in the offline single-vocable
        renderer). Every such seam inside this call is now blended
        instead of just concatenated; a seam that happens to fall
        exactly on THIS call's own first sample (the previous call ended
        precisely at a syllable boundary) is blended against
        `self._last_tail`, the real audio actually returned last time.
        """
        starts_on_fresh_syllable = (self.read_pos == 0)
        out = np.empty(n_samples, dtype=np.float32)
        filled = 0
        guard = 0  # safety: if every syllable somehow ended up empty, bail to silence instead of spinning forever
        splice_offsets = []
        while filled < n_samples:
            guard += 1
            if guard > len(self.syllables) + 4:
                out[filled:] = 0.0
                break
            syll = self.syllables[self.syllable_idx % len(self.syllables)]
            n = len(syll)
            if n == 0:
                self.syllable_idx += 1
                self.read_pos = 0
                continue
            take = min(n_samples - filled, n - self.read_pos)
            out[filled:filled + take] = syll[self.read_pos:self.read_pos + take]
            filled += take
            self.read_pos += take
            if self.read_pos >= n and filled < n_samples:
                # Sustaining through a held note past this syllable's own
                # material: continue into the word's next syllable
                # rather than looping this one again, see the offline
                # renderer's identical spot for the full reasoning. This
                # seam gets crossfaded below, same as every other.
                splice_offsets.append(filled)
                self.syllable_idx += 1
                self.read_pos = 0

        if self._xfade_n > 1:
            _apply_internal_crossfades(
                out, splice_offsets, self._xfade_n,
                prev_tail=self._last_tail if starts_on_fresh_syllable else None,
            )
            self._last_tail = out[-min(self._xfade_n, len(out)):].copy()
        return out

    def next_chunk(self, vocable: str, is_voiced: bool, onset_id, input_rms: float, n_samples: int, sample_rate: int):
        self._ensure_loaded(vocable)
        if not self.syllables:
            self._gain = 1.0
            self._phrase_env = 0.0
            self.was_voiced = False
            return None

        if is_voiced:
            self._silence_s = 0.0
            bridged_voiced = True
        else:
            self._silence_s += n_samples / float(sample_rate)
            # Bridge brief gaps instead of hard-stopping on every single
            # non-voiced frame, real singing has plenty of gaps well
            # under GAP_TOLERANCE_S between words even mid-phrase, and
            # treating each one as a full phrase-end was exactly what
            # caused audible stuttering (stop / restart / stop / restart
            # within a single sung phrase). The input_rms fed in for a
            # bridged gap is genuinely near-zero, so the volume-follow
            # below still naturally ducks it, it just doesn't hard-mute
            # and reset the syllable sequence for a gap this short.
            bridged_voiced = self.was_voiced and (self._silence_s <= self.GAP_TOLERANCE_S)

        if not bridged_voiced:
            if not self.was_voiced:
                # Already finished and fully stopped, true silence.
                self._gain = 1.0
                self._in_release_tail = False
                return None

            # A genuine phrase end: don't cut off wherever we happen to
            # be. Finish the CURRENT syllable a little further, full
            # gain, no fade, don't advance or loop it, rather than
            # snapping to silence mid-consonant.
            #
            # BUG FIX: this used to let the syllable run all the way to
            # its own natural end with NO cap at all. If a vocable's
            # segmentation ever produces one long syllable, or fails
            # to find real boundaries and falls back to treating the
            # WHOLE recording as one chunk (see _segment_into_syllables'
            # fallback), that "natural end" can be several seconds
            # away. That let this tail keep playing straight through
            # however long the real silence in the input actually was:
            # a genuine multi-second pause just got played over, which
            # is exactly "sounds like one big loop" / "plays through a
            # 5s gap." RELEASE_GRACE_S caps how long this tail is
            # allowed to run past the real phrase end, long enough to
            # avoid an abrupt mid-consonant cutoff (the original goal),
            # never long enough to swallow an actual pause.
            RELEASE_GRACE_S = 0.35
            if not self._in_release_tail:
                self._in_release_tail = True
                self._release_tail_s = 0.0
            self._release_tail_s += n_samples / float(sample_rate)

            syll = self.syllables[self.syllable_idx % len(self.syllables)]
            n = len(syll)
            if self.read_pos < n and self._release_tail_s <= RELEASE_GRACE_S:
                take = min(n_samples, n - self.read_pos)
                out = np.zeros(n_samples, dtype=np.float32)
                out[:take] = syll[self.read_pos:self.read_pos + take]
                self.read_pos += take
                return out * self._gain
            else:
                # Either the syllable finished naturally, or the grace
                # window ran out first, either way, actually stop now.
                self.was_voiced = False
                self._gain = 1.0
                self._phrase_env = 0.0
                self._in_release_tail = False
                return None

        if not self.was_voiced:
            # A genuine new phrase (a real, sustained stop of a few
            # seconds, per GAP_TOLERANCE_S, just ended): keep going
            # forward through the SAME continuous syllable sequence
            # instead of jumping back to the word's own first syllable
            # every time. Resetting to 0 here unconditionally was why
            # this sounded like it kept starting over from scratch
            # syllable_idx is a persistent, ever-advancing counter (see
            # _ensure_loaded, which resets it only when a genuinely
            # different WORD gets loaded), so a real gap just advances
            # it by one, exactly like an ordinary new-note onset would,
            # rather than snapping it back to the beginning.
            self.syllable_idx += 1
            self.read_pos = 0
            self._last_onset_id = onset_id
            # A real silence just ended, whatever's in _last_tail is
            # from before that gap, so it isn't something to blend this
            # phrase's own true attack against. This phrase's own
            # ATTACK_S ramp (below) is the right way to come in from
            # true silence.
            self._last_tail = None
        elif onset_id is not None and onset_id != self._last_onset_id:
            # A new onset within an ongoing phrase: advance to the next
            # syllable, restarted from its own beginning.
            self._last_onset_id = onset_id
            self.syllable_idx += 1
            self.read_pos = 0
        self.was_voiced = True

        # Advance the attack envelope toward 1 (a no-op once it's
        # already there, mid-phrase).
        attack_step = n_samples / (sample_rate * self.ATTACK_S)
        start_env, end_env = self._phrase_env, min(1.0, self._phrase_env + attack_step)
        env_ramp = np.linspace(start_env, end_env, n_samples, dtype=np.float32)
        self._phrase_env = end_env

        # Same loudness-following mechanism as before, dynamics only.
        input_rms = max(float(input_rms), 1e-6)
        if self._typical_rms is None:
            self._typical_rms = input_rms
        else:
            self._typical_rms = 0.98 * self._typical_rms + 0.02 * input_rms
        # Wider range than before, matches the offline renderer's
        # widened bounds so live and offline dynamics-following feel
        # consistent, and so swells/dips are actually pronounced rather
        # than clamped to a narrow band.
        target_gain = float(np.clip(input_rms / max(self._typical_rms, 1e-6), 0.1, 2.0))
        self._gain = 0.72 * self._gain + 0.28 * target_gain

        chunk = self._read_syllables(n_samples)
        return chunk * env_ramp * self._gain


def run_local_pipeline(stop_event, input_device: int):
    global pipeline_running
    try:
        from config.config_loader import get_config
        from core.audio_capture import AudioCapture
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        from analysis.rhythm_analyzer import RhythmAnalyzer
        from analysis.phonetic_analysis import CreeTokenizer
        from output.timing_sync import TimingSync
        import librosa as _lib

        cfg          = get_config()
        cfg["audio"]["input_device"] = input_device
        capture      = AudioCapture()
        preprocessor = Preprocessor()
        pitch        = PitchDetector()
        rhythm       = RhythmAnalyzer()
        cree         = CreeTokenizer()
        harmony      = harmony_engine   # shared instance, same sovereignty state as /ws/mic
        timing       = TimingSync()
        syllable_cycler = _SyllableCycler()

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

            # blocking=False (the default) is deliberate here, NOT an
            # oversight: this is the live, real-time path, and blocking=True
            # forces NeuralVocableBank.get_blocking(), a synchronous,
            # full-quality phase-vocoder render meant for offline batch
            # rendering only (see its own docstring). Calling it per note
            # in this loop was stalling the whole capture/decide loop on
            # every single onset. blocking=False uses the instant cached /
            # fast-resample path and upgrades the cache to full quality in
            # a background thread, which is what this loop actually needs.
            #
            # The live per-note RVC re-voicing pass (_apply_neural_timbre)
            # has also been removed from this loop. It isn't a quality
            # setting for real-time use, it's a genuinely slow, separate
            # network model pass (the architecture page says so directly:
            # "analyze, then play", not instant), and it was costing
            # ~750-800ms of synchronous stall PER NOTE (see your own log:
            # "Neural chunk convert: 784ms round-trip"), which is what was
            # actually causing the choppy, seemingly-random, note-behind-
            # reality sound, not a synthesis quality problem. The
            # samples in synthesis/samples/neural/ are already the
            # trained voice's timbre (build_vocable_bank.py converted them
            # through RVC once, offline), singing them live via the DSP
            # engine already carries that voice; this was a redundant
            # second live RVC pass on top, not something adding quality
            # worth an 800ms stall. If you want the extra live re-voicing
            # pass back, it needs to run asynchronously and swap into
            # `timing` when ready (the same instant-then-swapped pattern
            # index.html's own browser demo uses for its Neural mode),
            # never synchronously inline like this.
            if harmony._vocable_override:
                # A word is locked (see /harmony/vocable-override): play
                # the recorded .wav's own syllables, completely
                # unmodified in pitch, re-triggered in time with the
                # singer's real rhythm (rhythm.latest_onset, from
                # RhythmAnalyzer's own onset detector below) instead of
                # looping blindly on the recording's own clock. Silence
                # brief gap doesn't reset it, see GAP_TOLERANCE_S in
                # _SyllableCycler. Volume (not pitch) continuously
                # follows the input's own loudness, see its docstring.
                # Same fix as _render_track_single_vocable_offline's
                # is_singing_flags: gate on a genuine detected singing
                # pitch (archer_hz), not just raw amplitude VAD, so
                # instrumental sound bleeding into the mic during a real
                # vocal pause doesn't read as "still singing" and keep
                # the syllable cycle running through it.
                is_singing = bool(is_voiced and archer_hz)
                input_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))) if is_singing else 0.0
                # BUG FIX: this referenced `frame_size`, which is never
                # defined anywhere in this function, cfg["audio"]
                # ["frame_size"] was never pulled into a local variable
                # here (unlike the offline render functions, which do).
                # That's a NameError on the very first frame processed
                # with any vocable locked, which propagates all the way
                # up to this function's own outer try/except and kills
                # the whole capture/decide/sing loop, pipeline_running
                # goes False and the mic pipeline silently stops. This is
                # what a locked vocable "sounding very broken" on the
                # local hardware pipeline actually was: not a synthesis
                # quality problem, the pipeline was crashing outright.
                # len(clean) is also more correct than the config value
                # would have been anyway, it's the real length of the
                # frame just processed, guaranteed consistent with what
                # capture actually delivered, rather than assuming it
                # always matches cfg["audio"]["frame_size"].
                chunk = syllable_cycler.next_chunk(
                    harmony._vocable_override, is_singing, rhythm.latest_onset, input_rms, len(clean),
                    cfg["audio"]["sample_rate"],
                )
                if chunk is not None:
                    timing.schedule(chunk, "sing")
                    currently_singing = True
                elif currently_singing:
                    timing.flush()
                    currently_singing = False

            # else: "Mix" selected, deliberately does nothing (no robot
            # audio at all) for now. decision (above) is still computed
            # so the pitch-tracking telemetry message below keeps
            # working; it's just no longer used to drive any synthesis.

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
        import traceback
        print(f"[pipeline error] {e}")
        traceback.print_exc()
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

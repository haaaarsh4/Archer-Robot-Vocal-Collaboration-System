import asyncio
import concurrent.futures
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
from scipy.signal import butter, lfilter
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
            # Python threads (see _force_transcribe_clip), they don't get real parallelism --
            # they just fight over the same cores. Measured directly: 3 concurrent forced
            # clips took 58.4s, close to what 3 sequential calls would cost, not the
            # significant overlap concurrency should have bought. num_workers>1 makes
            # ctranslate2 spin up genuinely separate internal worker pools that can run truly
            # side by side; splitting cpu_threads across them keeps total CPU usage roughly
            # bounded, instead of N calls each trying to grab every core. Only the track model
            # needs multiple workers -- the primary pass is one call regardless, but the
            # forced-clip re-decodes are where several calls happen at once.
            #
            # max_cpu_fraction caps the TOTAL thread budget below 100% of the machine's cores.
            # Letting transcription's worker pool claim every core starves whatever else is
            # running concurrently on the same box -- specifically the neural voice-conversion
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
    # there) -- no point provisioning more worker pools than will ever be used concurrently.
    #
    # max_cpu_fraction=0.6 was tried here to leave headroom for the neural voice-conversion
    # pipeline to run alongside transcription. Reverted: measured directly, it took the primary
    # Whisper pass from 19.2s to 62.0s on the exact same reference track (a 3x regression) with
    # no confirmed benefit to synthesis speed to show for it -- the synthesis chunk in that same
    # run still took 94 seconds, i.e. still very slow regardless. A machine that's genuinely CPU-
    # constrained enough for both heavy workloads to contend this badly doesn't have this much
    # spare capacity to redistribute either way; deliberately handicapping transcription's own
    # budget just cost it speed without visibly buying the other side anything.
    TRACK_WHISPER_MODEL_SIZE, TRACK_WHISPER_MODEL_DIR, _legacy_whisper_checkpoint,
    num_workers=min(4, os.cpu_count() or 1)
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
    unison shadowing -- the backend half of the frontend's "pick one
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


SAMPLES_NEURAL_DIR = Path(__file__).parent / "synthesis" / "samples" / "neural"


@app.get("/api/vocables/manifest")
def vocables_manifest():
    """
    Exposes the pre-rendered neural vocable takes -- the real recorded (and
    voice-converted) "cold"/"moon"/"need"/etc. samples that DSP mode's own
    Python pipeline already sings from -- to the browser, so the live mic
    panel's DSP voice can play the ACTUAL recordings (pitch-shifted a
    little in real time) instead of a synthetic sawtooth-and-formants
    oscillator that has never had anything to do with what was recorded.

    Returns available=False (not an error) if the bank hasn't been built
    yet -- see build_vocable_bank.py -- so the browser can fall back to
    a placeholder voice instead of breaking.
    """
    global synthesizer
    if synthesizer is None or getattr(synthesizer, "_neural_bank", None) is None:
        return {"available": False, "vocables": []}

    bank = synthesizer._neural_bank
    vocables = []
    for vocable, bases in bank._bases.items():
        registers = [
            {
                "name": base.name,
                "f0_hz": round(float(base.f0_hz), 2),
                "brightness": round(float(base.brightness), 3),
                "duration_s": round(len(base.audio) / bank.sample_rate, 3),
                "url": f"/api/vocables/sample/{base.name}.wav",
            }
            for base in bases
        ]
        vocables.append({"vocable": vocable, "registers": registers})

    return {"available": True, "sample_rate": bank.sample_rate, "vocables": vocables}


@app.get("/api/vocables/sample/{filename}")
def vocables_sample(filename: str):
    # Guard against path traversal -- only a bare filename, no separators
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

        global synthesizer
        if synthesizer is None:
            return JSONResponse({"error": "Synthesizer not loaded"}, status_code=503)
        scratch_audio = synthesizer.synthesize(decision)

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
        tap_signal = lfilter(b, a, tap_signal)  # cumulative -- each tap is filtered again on top of the last
        start = delay_samples * k
        end = start + len(tap_signal)
        out[start:end] += tap_signal * gain

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak

    return out.astype(np.float32)


def _trim_excess_silence_at_edges(segment: np.ndarray, sample_rate: int,
                                   threshold_ratio: float = 0.08, max_dead_s: float = 0.015) -> np.ndarray:
    """
    Some syllables carry a longer-than-necessary near-silent stretch at
    their very start or end -- a soft breath before the vowel really
    starts, or a natural decay tail. That's a real, unedited feature of
    the recording, not a defect -- fine, even desirable, the one time a
    syllable is used at a genuine phrase start. But left in, that dead
    air becomes audible as a dropout-like gap every other time this
    syllable gets re-triggered mid-performance, where continuous sound
    is expected. This trims any such near-silent stretch down to a
    small margin, without touching a single sample of the part that's
    actually sounding -- it only shortens genuinely near-zero material.
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


def _segment_into_syllables(raw_audio: np.ndarray, sample_rate: int, min_syllable_s: float = 0.12) -> list:
    """
    Splits ONE recorded take (e.g. "Ohoho" or "Lalanana") into its own
    natural syllables via onset detection, so each syllable can be
    re-triggered on its own in time with the song -- see
    _render_track_single_vocable_offline's docstring for why this exists
    and what it fixes. Each returned syllable has had any excess dead
    air trimmed from its edges (see _trim_excess_silence_at_edges) and
    _declick_loop_seam applied, so looping any ONE of them individually
    (to sustain through a longer note) is click-free on its own.

    Onsets closer together than min_syllable_s are merged -- a real
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
    import librosa
    if len(raw_audio) == 0:
        return [raw_audio]

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

    chunks = [
        raw_audio[boundaries[i]:boundaries[i + 1]].copy()
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] > 200  # drop slivers too short to be a real syllable (< ~5ms)
    ]
    syllables = [
        _declick_loop_seam(_trim_excess_silence_at_edges(chunk, sample_rate), fade_samples=64)
        for chunk in chunks
    ]
    return syllables if syllables else [raw_audio.copy()]


def _render_track_single_vocable_offline(raw_audio_bytes: bytes, pitch_method: str, vocable: str) -> bytes:
    """
    The render path for "one locked word instead of Mix". Pitch is NEVER
    touched anywhere in this function, by explicit, repeated request --
    every sample that plays back is byte-for-byte from the original
    recording, never pitch-shifted or time-stretched.

    What this DOES do is make the recording rhythmically follow the
    song: the recording is split into its own natural syllables (see
    _segment_into_syllables -- "Ohoho" becomes roughly "Oh" / "ho" /
    "ho" / ... as its own onsets actually fall), and those syllables are
    re-triggered, in order, on the SONG's own real onsets -- its actual
    note attacks -- rather than the recording just looping blindly on
    its own internal clock with no relationship to the performance. A
    syllable loops on itself (click-free) to sustain through a longer
    note if the song holds one, and gets cut cleanly to the next
    syllable the instant the next note starts if the song moves faster
    than the syllable's own natural length. This is the same idea a
    producer chopping a vocal sample into an MPC/sampler and triggering
    the chops on the beat uses -- rhythmic connection to the performance
    without ever altering the recording's own pitch or timbre.

    Wherever the track isn't singing at all, there's silence (gated by
    the same voice-activity detection as before), and the OUTPUT VOLUME
    still continuously follows the input track's own loudness contour
    (apply_amplitude_curve_follow), so it swells and fades with the real
    performance on top of following its rhythm.
    """
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from synthesis.vocable_synthesizer import apply_amplitude_curve_follow
    import librosa

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]

    global synthesizer
    if synthesizer is None or getattr(synthesizer, "_neural_bank", None) is None:
        raise RuntimeError("Neural vocable bank not loaded -- build it with build_vocable_bank.py first")
    bases = synthesizer._neural_bank._bases.get(vocable.strip().lower())
    if not bases:
        raise RuntimeError(f"No recorded take loaded for vocable '{vocable}'")
    raw_audio = np.asarray(bases[0].audio, dtype=np.float32)
    syllables = _segment_into_syllables(raw_audio, sample_rate)

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    preproc = Preprocessor()
    n_frames = len(audio) // frame_size
    robot = np.zeros(len(audio), dtype=np.float32)
    amplitude_curve = np.zeros(n_frames, dtype=np.float64)

    is_voiced_flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        clean, is_voiced = preproc.process(audio[i * frame_size:(i + 1) * frame_size])
        is_voiced_flags[i] = is_voiced
        if is_voiced:
            amplitude_curve[i] = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))

    # Contiguous voiced stretches = sung phrases -- but a brief gap
    # (a breath, a consonant, a word boundary) is NOT the phrase ending;
    # only sustained silence beyond GAP_TOLERANCE_S is. Treating every
    # single non-voiced frame as a hard phrase-end was the actual cause
    # of audible stuttering: real singing has plenty of gaps well under
    # half a second between words even within one continuous phrase, and
    # each one was silencing and then restarting the syllable cycle from
    # scratch. Bridging them keeps the syllable sequence running straight
    # through -- the amplitude-follow below still naturally dips the
    # volume during the gap itself (it's genuinely quiet there), it just
    # doesn't hard-mute and restart.
    GAP_TOLERANCE_S = 0.25
    gap_tolerance_frames = max(1, int(GAP_TOLERANCE_S * sample_rate / frame_size))

    phrase_spans = []
    start = None
    silence_run = 0
    for i, v in enumerate(is_voiced_flags):
        if v:
            if start is None:
                start = i * frame_size
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run > gap_tolerance_frames:
                phrase_spans.append((start, (i - silence_run + 1) * frame_size))
                start = None
                silence_run = 0
            # else: within tolerance -- keep the phrase going, bridge the gap
    if start is not None:
        phrase_spans.append((start, n_frames * frame_size))

    # The song's own real onsets, whole-track, once -- this is the
    # rhythmic backbone every syllable gets triggered against. Onsets
    # closer together than ~90ms are merged; that's below any real sung
    # note's duration and would just be onset-detector noise chattering
    # the same syllable on and off.
    MIN_ONSET_GAP_S = 0.09
    raw_onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate, units="samples", backtrack=True)
    onsets = sorted(int(o) for o in raw_onsets)
    merged_onsets = []
    for o in onsets:
        if not merged_onsets or o - merged_onsets[-1] >= int(MIN_ONSET_GAP_S * sample_rate):
            merged_onsets.append(o)

    FADE_N = 96  # ~2ms at 44.1kHz -- declicks every note-segment boundary regardless of where a syllable got cut

    for (phrase_start, phrase_end) in phrase_spans:
        onsets_in_phrase = [phrase_start] + [o for o in merged_onsets if phrase_start < o < phrase_end]
        boundaries = sorted(set(onsets_in_phrase)) + [phrase_end]
        last_note_i = len(boundaries) - 2
        phrase_actual_end = phrase_end

        for note_i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[note_i], boundaries[note_i + 1]
            sub_idx = note_i
            syll = syllables[sub_idx % len(syllables)]
            n_syll = len(syll)
            if n_syll == 0:
                continue

            if note_i == last_note_i:
                # This is the note that was still sounding when the
                # singer actually stopped -- don't cut it off wherever
                # the detected phrase end happens to land mid-syllable.
                # Let THIS ONE syllable finish its own full natural
                # length first (capped only by the track's own end),
                # same as a real singer finishing the word they were on
                # rather than being cut off mid-consonant. If another
                # phrase starts before this tail finishes, that phrase's
                # own fill simply overwrites wherever they'd overlap --
                # an acceptable rare edge case, not a new artifact.
                seg_end = min(len(robot), seg_start + n_syll)
                phrase_actual_end = seg_end

            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue

            pos, filled = 0, 0
            guard = 0  # safety: bail to silence rather than spin forever if a syllable ever ends up empty
            while filled < seg_len:
                guard += 1
                if guard > len(syllables) + 4:
                    break
                take = min(seg_len - filled, n_syll - pos)
                robot[seg_start + filled:seg_start + filled + take] = syll[pos:pos + take]
                filled += take
                pos += take
                if pos >= n_syll and filled < seg_len:
                    # This syllable's own material is used up but the
                    # note is still held: continue into the WORD'S NEXT
                    # syllable rather than looping this same chunk again.
                    # Looping one short/quiet syllable many times to fill
                    # a long held note is exactly what caused audible
                    # crackling -- each repeat carries its own tiny
                    # declick fade (needed for safe self-looping), and a
                    # short syllable repeated 5-6 times packs that many
                    # fade transitions into under half a second. Moving
                    # forward through the word's OWN natural sequence
                    # instead means a long note is filled with more of
                    # the actual recording's real material, sounds like
                    # the word continuing rather than stuttering, and
                    # only wraps back to repeating from the start if the
                    # entire word gets used up.
                    sub_idx += 1
                    syll = syllables[sub_idx % len(syllables)]
                    n_syll = len(syll)
                    pos = 0

            # Declick this note segment's own tail -- whether it ended
            # naturally or was cut off mid-syllable by the next note.
            # For the LAST segment specifically this is now mostly a
            # no-op layered on top of that syllable's own already-
            # declicked natural tail (see _segment_into_syllables), not
            # a fresh cut -- it only does real work in the rare overlap
            # case above.
            fade_n = min(FADE_N, seg_len // 2)
            if fade_n > 1:
                ramp = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
                robot[seg_end - fade_n:seg_end] *= ramp

        # A natural attack at this PHRASE's true start, and a short
        # release at its (now syllable-complete) true end -- separate
        # from, and layered on top of, the tight ~2ms declicks above.
        # Those only exist to stop a digital click at each note-to-note
        # handoff and are deliberately short so the rhythm stays tight;
        # the phrase's own start still benefits from a real attack ramp
        # rather than snapping to full volume instantly. The release is
        # now short and mostly redundant with the syllable's own natural
        # tail (since we just let it finish completely above) -- kept
        # small only to smooth the rare overlap edge case, not to fade
        # out material that was still mid-word.
        ATTACK_S = 0.05
        RELEASE_S = 0.05
        attack_n = min(int(ATTACK_S * sample_rate), (phrase_actual_end - phrase_start) // 2)
        release_n = min(int(RELEASE_S * sample_rate), (phrase_actual_end - phrase_start) // 2)
        if attack_n > 1:
            robot[phrase_start:phrase_start + attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
        if release_n > 1:
            robot[phrase_actual_end - release_n:phrase_actual_end] *= np.linspace(1.0, 0.0, release_n, dtype=np.float32)

    if np.any(amplitude_curve > 0):
        # Wider gain range than the function's own default (0.15-1.6) --
        # explicitly requested to sound "very similar" to the track's
        # own loudness shape, so the swells and dips need to actually
        # read as such rather than being clamped to a narrow band.
        robot = apply_amplitude_curve_follow(
            robot, sample_rate, amplitude_curve=amplitude_curve, frame_hop_s=frame_size / sample_rate,
            min_gain=0.08, max_gain=2.2,
        )

    buf = io.BytesIO()
    sf.write(buf, robot, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _render_track_offline(raw_audio_bytes: bytes, pitch_method: str, texture: str, voice_index: int,
                           mode_override: str | None = None, instruments_enabled: bool = True,
                           transpose_semitones: float | None = None, apply_neural: bool = True,
                           delay_ms: float | None = None, echo_amount: float | None = None,
                           contour_baseline_db: float | None = None,
                           contour_max_bright_db: float | None = None,
                           contour_max_dark_db: float | None = None,
                           vocable: str | None = None) -> bytes:
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
    # handled in this function at all -- see
    # _render_track_single_vocable_offline and its own docstring for why
    # this note-by-note path can't be made to work for a multi-second
    # recorded word, and render_dsp_track (the /api/dsp/render-track
    # handler) for where that path is chosen instead. This function only
    # ever runs in normal "Mix" accompaniment mode now.
    if mode_override:
        harmony.set_forced_mode(mode_override)

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

        # RMS of this frame's own (gated/cleaned) input -- one reading per
        # frame, same cadence as the pitch curve above. This is what lets
        # the rendered note's OWN loudness follow the real performance's
        # dynamics afterward (see apply_amplitude_curve_follow) instead of
        # every note coming out at one fixed, static volume regardless of
        # whether the singer was belting or barely audible.
        frame_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))) if is_voiced else 0.0

        if decision.action == "sing":
            current_run = {"start_frame": i, "decision": decision, "n_frames": 1,
                            "pitch_curve": [decision.target_hz],
                            "amplitude_curve": [frame_rms]}
            runs.append(current_run)
        elif decision.action == "sustain":
            if current_run is None:
                current_run = {"start_frame": i, "decision": decision, "n_frames": 1,
                                "pitch_curve": [decision.target_hz],
                                "amplitude_curve": [frame_rms]}
                runs.append(current_run)
            else:
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
    for run in runs:
        decision = run["decision"]
        decision.duration_s = min(run["n_frames"] * frame_hop_s, MAX_RUN_SECONDS)
        scratch = synth.synthesize(decision, blocking=True, apply_crossfade=False)

        pitch_curve = np.asarray(run["pitch_curve"], dtype=np.float64)
        if pitch_curve.size >= 2 and decision.target_hz:
            scratch = apply_pitch_curve_follow(
                scratch, sample_rate, base_hz=decision.target_hz,
                pitch_curve_hz=pitch_curve, frame_hop_s=frame_hop_s,
            )

        amplitude_curve = np.asarray(run["amplitude_curve"], dtype=np.float64)
        if amplitude_curve.size >= 2:
            scratch = apply_amplitude_curve_follow(
                scratch, sample_rate, amplitude_curve=amplitude_curve, frame_hop_s=frame_hop_s,
            )
        scratch = np.asarray(scratch, dtype=np.float32).copy()

        start_sample = run["start_frame"] * frame_size
        end_sample = start_sample + len(scratch)
        if end_sample > len(robot):
            robot = np.pad(robot, (0, end_sample - len(robot)))

        vocable_changed = prev_vocable is not None and getattr(decision, "vocable", None) != prev_vocable
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
        prev_vocable = getattr(decision, "vocable", None)

    robot = robot[:max(len(audio), content_end)]

    peak = float(np.max(np.abs(robot))) if robot.size else 0.0
    if peak > 1.0:
        robot = robot / peak  # avoid clipping where overlapping/sustained notes summed above 0dBFS

    if mode_override == "delayed_response":
        robot = apply_echo_effect(robot, sample_rate, effective_delay_ms, effective_echo_feedback)

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
    delay_ms: float | None = Form(None),
    echo_amount: float | None = Form(None),
    contour_baseline_db: float | None = Form(None),
    contour_max_bright_db: float | None = Form(None),
    contour_max_dark_db: float | None = Form(None),
    vocable: str = Form("mix"),
):
    try:
        raw = await file.read()
        single_vocable = (vocable or "").strip().lower()
        if single_vocable and single_vocable != "mix":
            # One locked word: raw playback, looped, gated on voice
            # activity -- see _render_track_single_vocable_offline's own
            # docstring.
            wav_bytes = await asyncio.to_thread(
                _render_track_single_vocable_offline, raw, pitch_method, single_vocable,
            )
        else:
            # "Mix" deliberately does nothing for now, by explicit
            # request -- silence, matching the input track's own
            # duration/sample rate, rather than the old accompaniment-
            # mode engine (_render_track_offline, left untouched below
            # for anything that still calls it directly).
            info = sf.info(io.BytesIO(raw))
            silence = np.zeros(int(info.frames), dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, silence, info.samplerate, format="WAV", subtype="FLOAT")
            wav_bytes = buf.getvalue()
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
    delay_ms: float | None = Form(None),
    echo_amount: float | None = Form(None),
    contour_baseline_db: float | None = Form(None),
    contour_max_bright_db: float | None = Form(None),
    contour_max_dark_db: float | None = Form(None),
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
            _render_track_offline, raw, pitch_method, texture, voice_index,
            mode_override=(mode or None), instruments_enabled=instruments_enabled,
            transpose_semitones=transpose_semitones,
            delay_ms=delay_ms, echo_amount=echo_amount,
            contour_baseline_db=contour_baseline_db, contour_max_bright_db=contour_max_bright_db,
            contour_max_dark_db=contour_max_dark_db,
        )
        return Response(content=wav_bytes, media_type="audio/wav")
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
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    model = _get_demucs_model()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, n) mono
    if wav.shape[0] == 1:
        wav = wav.expand(2, -1)  # demucs expects stereo-shaped input; duplicate mono to both channels
    wav = convert_audio(wav, sample_rate, model.samplerate, model.audio_channels)

    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0), device="cpu", progress=False)[0]

    source_names = model.sources  # typically ["drums", "bass", "other", "vocals"]
    vocals_idx = source_names.index("vocals")
    vocal_stem = sources[vocals_idx].mean(dim=0).cpu().numpy()

    if model.samplerate != sample_rate:
        import librosa
        vocal_stem = librosa.resample(vocal_stem, orig_sr=model.samplerate, target_sr=sample_rate)

    vocal_stem = np.asarray(vocal_stem, dtype=np.float32)
    n = len(audio)
    if len(vocal_stem) == n:
        return vocal_stem
    if len(vocal_stem) > n:
        return vocal_stem[:n]
    return np.pad(vocal_stem, (0, n - len(vocal_stem)))


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

    DEMUCS_MIN_DURATION_S = 6.0
    duration_s = len(audio) / sample_rate

    audio_for_rvc = audio

    if not instruments_enabled:
        if duration_s >= DEMUCS_MIN_DURATION_S:
            try:
                vocal_stem = _separate_vocal(audio, sample_rate)
                audio_for_rvc = vocal_stem
                print(f"[convert-track direct] separated vocal from instrumental via Demucs "
                      f"({duration_s:.1f}s clip) -- RVC converts the isolated vocal only, and "
                      "the instrumental is discarded entirely: \"instruments off\" means the "
                      "robot performs solo, with no drum/instrument sound in the output at all.")
            except Exception as e:
                print(f"[convert-track direct] Demucs separation failed ({e}) -- falling back "
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
            "Neural sidecar conversion failed or timed out — check that neural_env/rvc_server.py "
            "is running and see its terminal output / GET /neural/status."
        )

    final = converted

    out_buf = io.BytesIO()
    sf.write(out_buf, final.astype(np.float32), sample_rate, format="WAV")
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


def _detect_vocal_pitch_spans(raw_audio_bytes: bytes, min_duration_s: float = 0.3,
                               max_gap_s: float = 0.5) -> list:
    """Scan for stretches with a real, trackable fundamental frequency -- not Whisper's
    no_speech_prob. A stretch of audio with a detected pitch in it is, by definition, not
    silence, regardless of whether Whisper's speech-likelihood model recognizes it as "speech".
    This is the ground truth used to force a transcription attempt on anything Whisper's own
    silence gate would otherwise have thrown away.

    This used to run this project's own frame-by-frame RMVPE pitch tracker -- the same one that
    drives the live accompaniment pitch tracking -- calling it once per individual audio frame
    (roughly 1,500-3,000+ separate model inferences for a single short track). That tool is
    built for real-time, one-frame-at-a-time tracking; reusing it for an offline full-file scan
    meant paying that per-frame model-call overhead thousands of times over just to answer a
    yes/no "is a pitch present here" question, and was very likely the dominant cost in the
    whole transcription pipeline -- worse than the actual Whisper decode. librosa.pyin answers
    the same question with one vectorized call across the entire signal instead of one Python
    loop iteration + neural inference per frame. This is a genuinely different algorithm (a
    classical pitch estimator, not the same neural model), so exact span boundaries can shift by
    a fraction of a second versus the old RMVPE-based version -- the merge/gap-bridging step
    downstream already smooths over that -- but it answers the same question and is
    dramatically cheaper to compute. NOTE: this adds `librosa` as a new dependency if it isn't
    already installed in this project's environment -- add it to requirements.txt.

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
    # hop=1024 was tried and rejected -- it throws an internal librosa error on short tracks
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
# common, fluent English word sequences over unusual ones -- it isn't a decoding-parameter bug
# that "well, hooray!" outscores "wha-he-oh", that's the model doing exactly what it was
# trained to do: pick the highest-probability real English continuation. Nothing in
# beam_size/temperature/best_of turns that off, because it's not a search-strategy setting,
# it's what the probabilities themselves say. A real fix for that is a phone-level recognizer
# with no word/language model at all (tested here with Allosaurus, a universal IPA phone
# recognizer) -- but on this specific audio the drums bleed straight through as spurious
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
# weyayoo, ah-hey-ya, weyayoo") -- confirmed working, not a guess. Reverted to it. The repeat-
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
    sees it. VAD is tuned to detect *talking* -- pauses, breath, silence
    between sentences. Sustained singing, chant, and wordless vocables (long
    held notes, "ah-ah-ah" type vocalizing) do not always look like "speech"
    to it, so VAD can silently drop whole stretches of a song before Whisper
    gets a chance to attempt a decode at all. That produces the symptom of
    "the first spoken line transcribes fine, then nothing for the rest of
    the track" -- it's not that Whisper heard the rest and rejected it, VAD
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
    "autocorrect" unfamiliar vocalizing into fluent, real English words --
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
# notes, etc.) -- these aren't a transcription of anything and should never appear in the
# anglicized text.
_WHISPER_META_TAG_RE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]|[\u266a\u266b\u2669-\u266c]")
# Collapses pathological repeat loops (the classic Whisper-on-music failure mode: the same
# token or short phrase repeated dozens of times) down to a natural handful of repeats instead
# of either the full garbage run or discarding the segment outright.
#
# A "token" here includes internal hyphens ("la-way-ya-hoo" is ONE token, not four words joined
# by a non-word character) -- the original version used bare \w+, which doesn't treat "-" as
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

    This is deliberately NOT translation and never touches meaning -- it only cleans up the
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
    the *entire* track -- dumping a whole song into one decode is what buried short vocal
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
    return merged


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
        # head of the next instead of the actual phrase the pitch detector flagged -- which
        # would look exactly like "transcribing the wrong thing" even if the decode itself is
        # working correctly. Output seeking costs a bit more (ffmpeg decodes from the start of
        # the file to find the exact point) but guarantees the clip boundaries are the real
        # ones, and these clips are short enough that the extra cost is negligible.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_in_path, "-ss", f"{clip_start:.3f}",
             "-t", f"{duration:.3f}", "-ac", "1", "-ar", "16000", out_path],
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
      to fall back on -- a 6-second clip collapsed to "Hey", a 10-second clip to "named".
    - Shortening the temperature ladder to (0.0, 0.4, 0.8) measurably broke the transcription
      itself, not just speed -- confirmed the moment it shipped. The full ladder isn't just
      overhead on hard content; skipping steps like 0.2/0.6 changed which decode attempt ends
      up accepted, and the result was worse, not merely slower to arrive at.
    Both are reverted. The remaining lever for the cost this pass adds is genuine
    parallelism -- see num_workers on the model loader -- not cutting what each individual
    decode call does.

    initial_prompt=WHISPER_VOCABLE_PROMPT is a separate, still-active mitigation for a
    different problem: Whisper's decoder has a language model baked into its weights that
    actively prefers common, fluent English word sequences over unusual phonetic ones --
    "well, hooray!" scores higher to it than "wha-he-oh" simply because "well, hooray" is a
    common English phrase and "wha-he-oh" isn't. That's not a decoding-parameter bug; it's the
    model doing what it was trained to do. Priming its context with an example of the
    hyphenated-vocable output style nudges it away from "autocorrecting" to a grammatical
    interjection, without fully eliminating the bias -- that's baked into the model weights,
    not something any prompt turns off completely. A phone-level recognizer with no
    word/language model at all would fix this at the root (tested with Allosaurus against
    this project's reference track), but on audio with drums under the vocals it picks up
    percussion transients as spurious consonant clusters unless the voice is cleanly
    separated from the percussion first -- that needs a source-separation stage (e.g. Demucs)
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


def _classify_segments(raw_segments: list) -> list:
    """Three-tier classification instead of a blunt confident/discarded split.

    Tier "speech"    -- passes all of Whisper's own reliability checks: shown as normal,
                         high-confidence English text (this is where genuine spoken English,
                         e.g. the emcee's introduction, ends up).
    Tier "vocal"      -- fails the strict checks but Whisper still heard something
                         voice-like (no_speech_prob isn't near-certain silence/noise) and
                         produced text: shown as an anglicized, lower-confidence line after
                         cleanup. This is where sung Cree and wordless vocables land, instead
                         of being thrown away as a generic "[non-lexical vocals]" tag.
    Tier "silent"     -- no_speech_prob is high AND there's no usable text left after
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

        # Not a clean pass -- but was there anything voice-like here at all? A no_speech_prob
        # near 1.0 means Whisper itself thinks this stretch is silence or non-vocal noise, in
        # which case there's nothing to anglicize and it should fall through to the
        # instrumental pass instead of showing an empty or fabricated line.
        anglicized = _anglicize_cleanup(s["text"]) if s["no_speech_prob"] < 0.92 else ""
        if anglicized:
            annotated.append({
                "start": round(s["start"], 2), "end": round(s["end"], 2),
                "label": anglicized, "confident": False, "type": "vocal",
            })
        # else: genuinely nothing lexical/vocal here -- leave the gap for instrument detection.
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

# AudioSet classes that indicate wordless vocal sound (chant, humming, vocalizing) rather than
# an instrument. Used only to pick a more honest label than the generic percussion fallback
# for gaps Whisper produced no usable text for at all -- it never feeds back into the "vocal"
# tier above, which already comes straight from Whisper's own (anglicized) text.
_PANNS_VOCAL_CLASSES = {
    "Singing", "Chant", "Humming", "Yodeling", "Vocal music", "A capella",
    "Choir", "Male singing", "Female singing", "Child singing",
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
        return [{"start": s, "end": e, "label": "[instrumental / percussion]", "confident": False,
                 "type": "instrumental"} for s, e in spans]

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
                                   "label": f"[{label.lower()}]", "confident": False, "type": "instrumental"})
            elif label in _PANNS_VOCAL_CLASSES and score > 0.15:
                # Whisper produced no usable text for this stretch at all, but this is
                # wordless vocalizing rather than an instrument -- label it honestly as that
                # instead of folding it into "[instrumental / percussion]".
                start_t = i / panns_sr
                end_t = min((i + win_samples) / panns_sr, duration)
                raw_spans.append({"start": round(start_t, 2), "end": round(end_t, 2),
                                   "label": "[wordless vocals]", "confident": False, "type": "instrumental"})
        i += hop_samples

    raw_spans.sort(key=lambda s: (s["label"], s["start"]))
    merged = []
    for s in raw_spans:
        if merged and merged[-1]["label"] == s["label"] and s["start"] <= merged[-1]["end"] + 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(dict(s))
    return [s for s in merged if s["end"] - s["start"] >= min_duration_s]


def _subtract_intervals(span: dict, occupied: list) -> list:
    """Cut the parts of `span` that overlap any interval already claimed by a vocal/speech
    segment, returning zero or more leftover pieces. This is what keeps the transcript from
    showing a "[instrumental / percussion] 0:00-0:34" row sitting on top of a confident
    "Honour song for..." row at 0:00-0:02 -- the vocal segment always wins the overlap, and
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
    _detect_instrument_spans a second time -- same detection, just not redone."""
    occupied = sorted((seg["start"], seg["end"]) for seg in annotated)
    if instrument_spans is None:
        try:
            instrument_spans = _detect_instrument_spans(raw_bytes, min_duration_s)
        except Exception as e:
            print(f"[transcribe] instrument detection skipped: {e}")
            instrument_spans = []

    for span in instrument_spans:
        for piece in _subtract_intervals(span, occupied):
            if piece["end"] - piece["start"] >= min_duration_s:
                annotated.append(piece)

    annotated.sort(key=lambda seg: seg["start"])

    # Adjacent pieces of the same instrumental label that ended up back-to-back after
    # subtraction (e.g. a drum span split around a short vocal line) read better merged into
    # one row than as two near-identical rows a fraction of a second apart.
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
        # classification pass are all independent of each other -- none of them needs
        # another's output, they only need the raw audio -- but they were running one after
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
              f"scan, overlapped) done in {time.perf_counter() - _t_stage:.1f}s -- NOT the "
              f"total, forced-decode chunks and final assembly still to come "
              f"(whisper pass1 alone would be ~{_t_whisper_pass1:.1f}s, pitch scan "
              f"~{_t_pitch:.1f}s, instrument scan ~{_t_instruments:.1f}s -- these three "
              f"overlapped, so the stage total above is roughly the largest of them, not "
              f"the sum)")

        # The three-tier classifier below is what separates real speech from anglicized
        # vocalizing from actual silence, not VAD. Even with VAD off, Whisper's own
        # no_speech_threshold check can still decide a whole stretch is silent and drop it
        # before it ever becomes a raw segment at all -- which is exactly what was collapsing
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
        # short track -- the original slowdown) or one decode over the entire file (fast, but
        # it buries each short burst inside a mostly-silent/percussive window and Whisper just
        # omits it -- the regression that dropped every vocal line down to "[wordless
        # vocalizing]"). Each merged clip gets exactly one forced, full-quality decode call
        # (see _force_transcribe_clip for why it isn't a cheaper one) and becomes one
        # continuous segment, which also fills in the sub-2-second silences between pulses that
        # were falling through both the vocal and instrumental passes entirely. The clips are
        # independent of each other -- different, non-overlapping audio -- so they're also run
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
                    annotated.append({
                        "start": chunk["start"], "end": chunk["end"],
                        "label": text if text else "[wordless vocalizing]",
                        "confident": False, "type": "vocal",
                    })
            annotated.sort(key=lambda seg: seg["start"])
        print(f"[transcribe] timing: STAGE 2/3 ({len(merged_chunks)} forced-decode chunk(s)) "
              f"took {time.perf_counter() - _t_forced:.1f}s -- NOT the total, instrument "
              f"gap-fill and final assembly still to come")

        annotated = _add_percussive_segments(annotated, raw, min_duration_s=2.0,
                                              instrument_spans=instrument_spans)

        print(f"[transcribe] timing: STAGE 3/3 DONE -- TOTAL transcription time "
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
    an audible digital click at the seam -- a sharp, broadband
    discontinuity that isn't part of the recording, just an artifact of
    cutting it into a loop. This fades only the first/last few
    milliseconds down toward silence (NOT toward each other -- this
    isn't a crossfade blend of head and tail content, which would mix
    the two together and is exactly the kind of transformation this was
    asked not to do) so the seam is a silent gap instead of a click.
    Every other sample of the actual recording is completely untouched;
    at 44.1kHz, 128 samples is under 3ms -- shorter than the ear's
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
    _segment_into_syllables -- see that function's docstring for the
    full reasoning. Pitch is never touched here either: every sample
    played back is byte-for-byte from the original recording.

    The recording is split into its own natural syllables once (on
    load). Each time a NEW onset is detected in the live singing (via
    RhythmAnalyzer's own onset detector, passed in as `onset_id` -- any
    value that changes exactly when a new onset fires; its caller passes
    rhythm.latest_onset), this advances to the next syllable in order
    and restarts it from its own beginning -- landing on the singer's
    actual rhythm instead of an arbitrary loop position. Between onsets,
    the current syllable simply loops on itself (click-free) to sustain
    through a held note. A genuine new PHRASE (real silence -- amplitude
    VAD, not a pitch-confidence blip) resets back to the first syllable,
    the same way a person starts the word over from its beginning each
    time they start singing again rather than resuming mid-word.

    Every phrase also gets a natural ATTACK when it starts -- ramping up
    from silence over ATTACK_S instead of snapping to full volume
    instantly, since a real voice doesn't switch on like a light switch.

    On the way OUT, this does NOT fade out wherever the silence happened
    to begin -- it finishes playing the CURRENT syllable to its own full
    natural length first (one whole "La", not half of one), and only
    THEN actually goes silent. That syllable's own edges are already
    declicked (see _segment_into_syllables), so letting it finish is
    already a smooth, natural close on its own; no separate fade needed.
    If singing resumes before that syllable finishes, nothing was ever
    cut short in the first place -- it just continues.

    Volume (never pitch) continuously follows the input's own loudness,
    identical mechanism to before -- this envelope is layered on top of
    that, not a replacement for it.
    """
    ATTACK_S = 0.05
    GAP_TOLERANCE_S = 0.25  # a breath, a consonant, a word boundary -- not a phrase ending; see next_chunk

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
        self.syllables = _segment_into_syllables(raw_audio, bank.sample_rate)
        self.vocable = vocable
        self.syllable_idx = 0
        self.read_pos = 0

    def _read_syllables(self, n_samples: int) -> np.ndarray:
        """Fills n_samples by reading forward through the syllable
        sequence, continuing into the next syllable once the current one
        is used up -- shared by normal playback and the release tail
        below, since a release still needs real recorded material to
        fade out, not silence."""
        out = np.empty(n_samples, dtype=np.float32)
        filled = 0
        guard = 0  # safety: if every syllable somehow ended up empty, bail to silence instead of spinning forever
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
                # rather than looping this one again -- see the offline
                # renderer's identical spot for the full reasoning.
                self.syllable_idx += 1
                self.read_pos = 0
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
            # non-voiced frame -- real singing has plenty of gaps well
            # under GAP_TOLERANCE_S between words even mid-phrase, and
            # treating each one as a full phrase-end was exactly what
            # caused audible stuttering (stop / restart / stop / restart
            # within a single sung phrase). The input_rms fed in for a
            # bridged gap is genuinely near-zero, so the volume-follow
            # below still naturally ducks it -- it just doesn't hard-mute
            # and reset the syllable sequence for a gap this short.
            bridged_voiced = self.was_voiced and (self._silence_s <= self.GAP_TOLERANCE_S)

        if not bridged_voiced:
            if not self.was_voiced:
                # Already finished and fully stopped -- true silence.
                self._gain = 1.0
                return None

            # A genuine phrase end: don't cut off wherever we happen to
            # be. Finish the CURRENT syllable's own natural length first
            # -- full gain, no fade, don't advance or loop it -- and
            # only actually go silent once it's genuinely done. If
            # singing resumes before that happens, next_chunk just falls
            # through to the normal path below on the next call; nothing
            # here needs undoing since nothing was cut short.
            syll = self.syllables[self.syllable_idx % len(self.syllables)]
            n = len(syll)
            if self.read_pos < n:
                take = min(n_samples, n - self.read_pos)
                out = np.zeros(n_samples, dtype=np.float32)
                out[:take] = syll[self.read_pos:self.read_pos + take]
                self.read_pos += take
                return out * self._gain
            else:
                # The syllable finished naturally exactly on/before this
                # call -- now actually stop.
                self.was_voiced = False
                self._gain = 1.0
                self._phrase_env = 0.0
                return None

        if not self.was_voiced:
            # A genuine new phrase (real amplitude silence, sustained
            # past the gap tolerance, just ended): start the word over
            # from its own first syllable. _phrase_env is already 0 here
            # (set when the previous phrase actually finished stopping),
            # so the attack ramp below starts from true silence.
            self.syllable_idx = 0
            self.read_pos = 0
            self._last_onset_id = onset_id
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

        # Same loudness-following mechanism as before -- dynamics only.
        input_rms = max(float(input_rms), 1e-6)
        if self._typical_rms is None:
            self._typical_rms = input_rms
        else:
            self._typical_rms = 0.98 * self._typical_rms + 0.02 * input_rms
        # Wider range than before -- matches the offline renderer's
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
        harmony      = harmony_engine   # shared instance — same sovereignty state as /ws/mic
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
            # forces NeuralVocableBank.get_blocking() -- a synchronous,
            # full-quality phase-vocoder render meant for offline batch
            # rendering only (see its own docstring). Calling it per note
            # in this loop was stalling the whole capture/decide loop on
            # every single onset. blocking=False uses the instant cached /
            # fast-resample path and upgrades the cache to full quality in
            # a background thread, which is what this loop actually needs.
            #
            # The live per-note RVC re-voicing pass (_apply_neural_timbre)
            # has also been removed from this loop. It isn't a quality
            # setting for real-time use -- it's a genuinely slow, separate
            # network model pass (the architecture page says so directly:
            # "analyze, then play", not instant), and it was costing
            # ~750-800ms of synchronous stall PER NOTE (see your own log:
            # "Neural chunk convert: 784ms round-trip"), which is what was
            # actually causing the choppy, seemingly-random, note-behind-
            # reality sound -- not a synthesis quality problem. The
            # samples in synthesis/samples/neural/ are already the
            # trained voice's timbre (build_vocable_bank.py converted them
            # through RVC once, offline) -- singing them live via the DSP
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
                # brief gap doesn't reset it -- see GAP_TOLERANCE_S in
                # _SyllableCycler. Volume (not pitch) continuously
                # follows the input's own loudness -- see its docstring.
                input_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2))) if is_voiced else 0.0
                chunk = syllable_cycler.next_chunk(
                    harmony._vocable_override, is_voiced, rhythm.latest_onset, input_rms, frame_size,
                    cfg["audio"]["sample_rate"],
                )
                if chunk is not None:
                    timing.schedule(chunk, "sing")
                    currently_singing = True
                elif currently_singing:
                    timing.flush()
                    currently_singing = False

            # else: "Mix" selected -- deliberately does nothing (no robot
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

import asyncio
import io
import json
import os
import re
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
    """
    Loads RMVPE into the process-wide model cache (see PitchDetector's
    _RMVPE_MODEL_CACHE) at server startup instead of on the first request
    that actually needs it. Without this, whichever click happens to be
    first -- toggling RMVPE, hitting Play, starting a mic -- eats the
    one-time load cost live, which on a short demo clip can mean the whole
    clip finishes before RMVPE ever reports a single note. Every load
    after this one hits the cache and is instant regardless of prewarm.

    Set pitch.rmvpe.prewarm: false in config.yaml to skip this (e.g. if
    you never use RMVPE and don't want the torch import / checkpoint read
    slowing down server startup).
    """
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

    # Runs in a background thread so it doesn't delay the server actually
    # starting to accept connections -- YIN and everything else works
    # immediately either way; RMVPE just becomes fast a few seconds sooner.
    threading.Thread(target=_load, daemon=True).start()

broadcast_queue: _queue.Queue = _queue.Queue(maxsize=64)
pipeline_running = False
pipeline_stop_event = threading.Event()

# Detect whether local audio hardware is available
AUDIO_AVAILABLE = False
try:
    import pyaudio as _pyaudio
    _p = _pyaudio.PyAudio()
    _p.terminate()
    AUDIO_AVAILABLE = True
except Exception:
    pass

# Load the trained Cree -> English translation model once at startup
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


# Sentiment scorer, two backends:
#
# PRIMARY: cardiffnlp/twitter-roberta-base-sentiment-latest -- a real
# trained neural network (125M-parameter RoBERTa), fine-tuned by the
# CardiffNLP research group on ~124 million tweets for 3-class sentiment
# (negative/neutral/positive), released on HuggingFace. This is a genuine,
# widely-used, dataset-trained classifier, not a lexicon lookup -- it reads
# a whole sentence's context (negation, sarcasm markers, word order),
# something a word-by-word dictionary like VADER structurally cannot do.
# Downloads its weights (~500MB) from huggingface.co on first run and
# caches them locally after that.
#
# FALLBACK: VADER (a lexicon + hand-coded-rules tool, not ML -- see below)
# for environments with no internet access to huggingface.co, e.g. an
# offline server or a sandboxed CI job. Which one actually ran is reported
# in every response as "engine", so this is never silently degraded.
sentiment_analyzer = None            # VADER instance, if loaded
sentiment_load_error = None
roberta_sentiment = None             # transformers pipeline, if loaded
roberta_load_error = None

# Sentiment scorer, two backends:
#
# PRIMARY: cardiffnlp/twitter-roberta-base-sentiment-latest -- a real
# trained neural network (125M-parameter RoBERTa), fine-tuned by the
# CardiffNLP research group on ~124 million tweets for 3-class sentiment
# (negative/neutral/positive). This is a genuine, widely-used, dataset-
# trained classifier, not a lexicon lookup -- it reads a whole sentence's
# context (negation, sarcasm markers, word order), something a word-by-word
# dictionary like VADER structurally cannot do.
#
# server.py NEVER downloads this model itself and makes no network calls
# for it, at startup or per-request. It only loads from SENTIMENT_MODEL_DIR
# on disk, with local_files_only=True enforced explicitly -- so if that
# directory is missing or incomplete, this fails immediately and falls
# back to VADER rather than silently attempting a network fetch. Run
# download_sentiment_model.py once, separately, before starting the
# server, to populate that directory.
#
# FALLBACK: VADER (a lexicon + hand-coded-rules tool, not ML -- see below)
# for whenever the local model directory isn't there. Which one actually
# ran is reported in every response as "engine", so this is never silently
# degraded without you knowing.
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
    """
    Real (not canned) valence + heuristic arousal for one English sentence,
    plus the same musical/LED mapping the frontend has always used for the
    hand-authored CREE_EXAMPLES. Kept here, not in the frontend, so the
    mapping only lives in one place once the model driving it is real.

    `vocal`, if given, is real signal-derived features from the actual
    recorded audio (see /api/analyze): pitch_range_semitones (from this
    project's own RMVPE/YIN pitch engine via /api/pitch/analyze-track) and
    rms_mean / rms_variance (loudness, from the raw waveform). These are
    blended into arousal only -- wider pitch range and louder, more
    dynamic singing reads as more aroused, which is well-supported by
    vocal-emotion research (Scherer et al.). Valence stays text-only: there
    is no comparably reliable way to get valence out of solo vocal audio
    with signal processing alone (no harmony/key reference to read
    major/minor from), so this deliberately doesn't pretend to.
    """
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

    # Arousal heuristic (v1, not itself a trained signal, regardless of
    # which valence engine ran above):
    #  - emotional charge: how far the active engine is from calling this
    #    neutral at all (RoBERTa's 1-P(neutral), or VADER's pos+neg)
    #  - intensifier words ("very", "so", "never"...)
    #  - exclamation marks / ALL-CAPS words as emphasis markers
    #  - shorter, punchier sentences read as more intense than long ones
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
        # pitch_range_semitones: 0-24 (two octaves) mapped to 0-1. Louder
        # (rms_mean, already 0-1) and more dynamically varied (rms_variance)
        # singing both read as more aroused too.
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


# Server-side speech-to-text via Whisper, loaded from local files only --
# same no-network-in-server.py pattern as the sentiment model above. This
# replaces the browser's built-in webkitSpeechRecognition entirely: that
# API only works in official Google Chrome/Edge (it silently streams audio
# to Google's servers using a private API key open-source Chromium/Brave/
# Firefox/Safari don't have), which makes it fundamentally unreliable
# across browsers -- not a bug fixable from this page's JavaScript.
# Transcribing on our own server instead means the browser's only job is
# "record audio, send it here" -- works identically everywhere.
WHISPER_MODEL_DIR = "data/models/whisper"
whisper_model = None
whisper_load_error = None

_whisper_checkpoint = os.path.join(WHISPER_MODEL_DIR, "base.en.pt")
if not os.path.isfile(_whisper_checkpoint):
    whisper_load_error = (
        f"{_whisper_checkpoint} not found. Run 'python download_whisper_model.py' once first "
        "(needs internet access, one-time only) to populate it. server.py itself never "
        "downloads this model or makes any network call for it."
    )
    print(f"Whisper transcription model not loaded: {whisper_load_error}")
else:
    try:
        import whisper as _whisper
        whisper_model = _whisper.load_model(_whisper_checkpoint)
        print(f"Whisper transcription model loaded from local files at {_whisper_checkpoint}.")
    except Exception as e:
        whisper_load_error = str(e)
        print(f"Whisper transcription model failed to load: {whisper_load_error}")


harmony_engine = None
harmony_load_error = None
try:
    from synthesis.harmony_engine import HarmonyEngine
    harmony_engine = HarmonyEngine()
    print("Shared HarmonyEngine initialized (sovereignty + mode selection live here).")
except Exception as e:
    harmony_load_error = str(e)
    print(f"HarmonyEngine not loaded: {harmony_load_error}")


# Optional neural voice-conversion stage — reskins VocableSynthesizer's DSP
# output into a trained real-voice timbre (see NEURAL_VOICE_ROADMAP.md).
# Disabled by default via synthesis.neural.enabled in config.yaml; if it's
# off, not installed, or has no trained model configured, the pipeline
# behaves exactly as it did before this stage existed — pure DSP output.
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


# Real Plains Cree word validation, via ALTLab's open-source morphological
# FST (github.com/giellalt/lang-crk, mirrored at
# github.com/UAlbertaALTLab/plains-cree-fsts). This is NOT speech
# recognition -- there is no live Cree ASR anywhere, see the note on the
# sentiment page -- it's a real morphological analyzer: given a typed word,
# it tells you whether it's a recognized nêhiyawêwin word form, its
# dictionary lemma, and its part of speech, using the same ~16,500-stem
# analyzer that powers ALTLab's own e-dictionary and spellchecker. It also
# recognizes ASCII-typed words (no macrons) as informal spellings of the
# correctly-accented form -- e.g. "ewapamat" analyzes as a non-normative
# ("Err/Orth") spelling of "êwâpamât".
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

# Alphabet used for spelling-suggestion edits: standard Cree SRO letters,
# both plain and macron-accented vowels, so a candidate like "tanisi" can
# still reach "tânisi" within one edit even though the accented form uses
# a different character than the one typed.
_CREE_ALPHABET = list("acehiklmnopstwyâêîô")


def _clean_analysis_tag(raw: str) -> str:
    return _CREE_FLAG_RE.sub("", raw)


def _extract_lemma_pos(cleaned: str):
    """Pull the dictionary lemma + POS out of a cleaned analysis string
    like 'PV/e+wâpamêw+V+TA+...' or 'IC+tasôw+V+AI+...'. Skips leading
    grammatical markers -- preverbs (PV/...) and bare tags like the
    Initial-Change marker 'IC' -- rather than just the first '+'-segment,
    since either can precede the actual stem."""
    parts = cleaned.split("+")

    def is_marker(p):
        return p.startswith("PV/") or (p.isalpha() and p.isupper())

    lemma = next((p for p in parts if p and not is_marker(p)), parts[0])
    pos_match = re.search(r"\+(N|V|Ipc|Pron|Prop|Adv|Num|Interj)\b", cleaned)
    return lemma, (pos_match.group(1) if pos_match else None)


def _edits1(word: str) -> set:
    """All strings one insertion/deletion/substitution/transposition away
    from `word`, restricted to Cree SRO letters. Standard spelling-
    correction technique (same idea as Peter Norvig's spell-checker),
    just checked against the real FST instead of a frequency dictionary."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in _CREE_ALPHABET]
    inserts = [L + c + R for L, R in splits for c in _CREE_ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def suggest_cree_word(word: str, deep: bool = False, max_suggestions: int = 5) -> list:
    """'Did you mean' suggestions for a word the FST doesn't recognize.
    Real, not fabricated: every suggestion returned is a string that the
    FST itself confirms is a valid nêhiyawêwin word form. edit-distance-1
    is fast enough to run on every keystroke (~10-30ms); edit-distance-2
    (deep=True) is ~1s, so it's reserved for one-off checks like pressing
    Translate on a single unrecognized word, not live-as-you-type."""
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
    """Looks up one word in the real FST. recognized=False just means this
    specific analyzer (16,500 stems, still a work in progress per ALTLab)
    doesn't have it -- not proof the word isn't valid Cree."""
    if cree_analyzer_fst is None:
        return {"word": word, "recognized": None, "error": cree_fst_load_error}

    results = cree_analyzer_fst.lookup(word)
    if not results:
        suggestions = suggest_cree_word(word, deep=deep_suggestions)
        return {"word": word, "recognized": False, "analyses": [], "suggestions": suggestions}

    analyses = []
    for raw, weight in results:
        cleaned = _clean_analysis_tag(raw)
        # Preverbs (PV/e, PV/ka, ...) and bare markers (IC, ...) can
        # precede the actual stem -- _extract_lemma_pos skips those.
        lemma, pos = _extract_lemma_pos(cleaned)
        is_variant = "Err/Orth" in cleaned  # non-normative spelling (e.g. macrons dropped)
        analyses.append({"lemma": lemma, "pos": pos, "tag": cleaned, "is_orthographic_variant": is_variant})

    return {"word": word, "recognized": True, "analyses": analyses}


@app.post("/api/cree/analyze")
def cree_analyze(req: CreeAnalyzeRequest):
    """Real-time Cree word validation -- is this a recognized nêhiyawêwin
    word, its lemma/part of speech, and if not, real 'did you mean'
    suggestions confirmed against the same FST (see suggest_cree_word).
    deep_suggestions=True also checks edit-distance-2, at ~1s cost -- use
    it for one-off checks (e.g. a Translate button), not on every
    keystroke."""
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
    """Scores English text directly. See score_sentiment() for how RoBERTa
    (primary) vs. VADER (fallback) get chosen, and what's real vs.
    heuristic in the arousal score either way."""
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
    """Real signal-derived features from the actual recorded audio, not
    estimates. pitch_range_semitones comes from this project's own
    RMVPE/YIN pitch engine via /api/pitch/analyze-track; rms_mean/
    rms_variance are plain loudness stats computed client-side from the
    same recording via the Web Audio API. See score_sentiment()'s
    docstring for how (and why only partially) these feed into the result."""
    pitch_range_semitones: float = 0.0
    rms_mean: float = 0.0
    rms_variance: float = 0.0


class AnalyzeRequest(BaseModel):
    text: str
    already_english: bool = False
    vocal: Optional[VocalFeatures] = None


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    One call for the sentiment demo: Cree text in -> translate -> sentiment,
    or English speech transcript in -> sentiment directly.
    already_english=True skips translation (e.g. text captured live via the
    browser's English speech recognition) and scores it directly.
    vocal, if provided, blends real pitch/loudness features from the sung
    audio into arousal -- see score_sentiment().
    """
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
    """Pin a voice texture regardless of accompaniment mode — e.g. a
    'concert mode' UI button that blooms whatever's playing into a choir."""
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
    return {
        "loaded": True,
        "enabled": neural_timbre.enabled,
        "sidecar_url": neural_timbre.sidecar_url,
        "sidecar_reachable": neural_timbre._reachable,
        "voices_configured": neural_timbre.num_voices_configured,
    }


class RenderNoteRequest(BaseModel):
    target_hz: float
    vocable: str = "aah"
    duration_s: float = 0.45
    mode: str = "unison_shadowing"
    texture: str = "solo"
    voice_index: int = 0


def _build_render_decision(req: "RenderNoteRequest", cfg: dict):
    """
    Builds a stand-in for HarmonyDecision, good enough to drive
    VocableSynthesizer.synthesize() for one browser-triggered note.

    This is NOT the real harmony_engine.decide() output. The browser's
    client-side accompaniment engine (index.html) doesn't run Cree
    phoneme analysis -- there's no MFCC extraction in JS -- so there is
    no real brightness/nasality/vowel_color to send up. This uses the
    same neutral profile server.py already falls back to for unvoiced
    frames elsewhere (cree._neutral_profile in ws_mic / run_local_pipeline).
    If your CreeTokenizer's actual neutral values aren't
    (brightness=0.5, vowel_color=0.5, nasality=0.0), change them here
    to match -- this is the one place in this endpoint that's a
    reasonable guess rather than read from your code.
    """
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
    """
    Renders ONE sung note through the real Python pipeline --
    VocableSynthesizer -> NeuralTimbreConverter -- and returns it as a
    WAV file.

    This is what actually gets your trained RVC voice into the browser
    Live Demo. The demo's RobotVoice/VoiceEnsemble classes in index.html
    are a from-scratch JS reimplementation of the synthesizer that runs
    entirely client-side and never calls into this backend or the neural
    sidecar at all -- that's why synthesis.neural.enabled alone changed
    nothing you could hear on that page. See the NeuralNoteBridge class
    in index.html for the frontend side of this.

    Costs one HTTP round trip + one VocableSynthesizer render + one RVC
    sidecar call per request -- expect roughly 50-500ms depending on
    whether synthesis.neural.device is actually hitting a free GPU (see
    max_latency_warn_s in config.yaml, which the sidecar call already
    logs against). Because of that latency, the frontend plays its
    instant local oscillator first and only swaps this audio in once it
    arrives -- it never goes silent to wait for this endpoint.
    """
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

        # A fresh VocableSynthesizer per request is deliberate, not an
        # oversight. VocableSynthesizer keeps _prev_audio state to
        # crossfade successive notes together, which assumes one
        # continuous sequential caller -- exactly true in
        # run_local_pipeline, not true here, where this endpoint can get
        # concurrent requests from multiple demo sessions or overlapping
        # notes. Sharing one instance would crossfade unrelated notes
        # into each other. The default sinusoidal engine does no file
        # I/O in __init__, so this costs milliseconds, not a model
        # reload -- only the wavetable engine would make this expensive,
        # and it isn't your configured engine.
        synth = VocableSynthesizer()
        scratch_audio = synth.synthesize(decision)

        sample_rate = cfg["audio"]["sample_rate"]
        final_audio = neural_timbre.convert(
            scratch_audio, sample_rate, decision.target_hz, voice_index=req.voice_index
        )

        buf = io.BytesIO()
        sf.write(buf, final_audio.astype(np.float32), sample_rate, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    except Exception as e:
        print(f"[neural render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


def _render_track_offline(raw_audio_bytes: bytes, pitch_method: str, texture: str, voice_index: int,
                           mode_override: str | None = None) -> bytes:
    """
    Runs entirely inside a worker thread (see the asyncio.to_thread call
    in render_neural_track below) -- this is real, potentially slow, CPU
    work: framewise pitch/rhythm/harmony analysis over the WHOLE track,
    DSP synthesis of every note, then one whole-track call to the neural
    sidecar. None of it belongs on the asyncio event loop.

    This is the offline counterpart to run_local_pipeline()/NeuralNoteBridge:
    same analysis -> harmony -> synthesis chain, but walked over a fixed
    in-memory buffer instead of a live mic queue, with no real-time
    budget and no per-note single-flight drop -- every note that the
    harmony engine decides to sing gets synthesized and included.
    """
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

    preproc = Preprocessor()
    pitch = PitchDetector()
    pitch.set_method(pitch_method)
    rhythm = RhythmAnalyzer()
    cree = CreeTokenizer()
    # A FRESH HarmonyEngine for this render -- deliberately not the
    # shared module-level harmony_engine. That instance carries live
    # sovereignty/mode state for whatever's happening on /ws/mic right
    # now; a batch render shouldn't read that state or mutate it out
    # from under a live session (same reasoning as the fresh
    # VocableSynthesizer per call in render_neural_note above).
    harmony = HarmonyEngine()
    harmony.set_texture(texture)
    if mode_override:
        harmony.set_forced_mode(mode_override)
    synth = VocableSynthesizer()

    n_frames = len(audio) // frame_size
    # Tail padding: a note starting near the very end of the track can
    # still be mid-sustain once the source audio runs out, so the output
    # buffer needs room past len(audio) or its tail gets clipped off.
    robot = np.zeros(len(audio) + int(4.0 * sample_rate), dtype=np.float32)

    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        clean, is_voiced = preproc.process(frame)
        rhythm.push_frame(clean, is_voiced)

        archer_hz = None
        phoneme_profile = cree._neutral_profile
        if is_voiced:
            pitch_input = frame if pitch.method == "rmvpe" else clean
            hz, conf = pitch.detect(pitch_input)
            if hz:
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
            scratch = synth.synthesize(decision)
            start_sample = i * frame_size
            end_sample = start_sample + len(scratch)
            if end_sample > len(robot):
                robot = np.pad(robot, (0, end_sample - len(robot)))
            robot[start_sample:end_sample] += scratch

    peak = float(np.max(np.abs(robot))) if robot.size else 0.0
    if peak > 1.0:
        robot = robot / peak  # avoid clipping where overlapping/sustained notes summed above 0dBFS

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(robot, sample_rate, voice_index=voice_index, timeout_s=timeout_s)
    if converted is None:
        raise RuntimeError(
            "Neural sidecar conversion failed or timed out — check that neural_env/rvc_server.py "
            "is running and see its terminal output / GET /neural/status."
        )

    out_buf = io.BytesIO()
    sf.write(out_buf, converted.astype(np.float32), sample_rate, format="WAV")
    return out_buf.getvalue()


@app.post("/api/neural/render-track")
async def render_neural_track(
    file: UploadFile = File(...),
    texture: str = Form("solo"),
    pitch_method: str = Form("rmvpe"),
    voice_index: int = Form(0),
    mode: str = Form(""),
):
    """
    Offline ("bounce") neural render for the Upload-MP3 demo panel's
    Neural voice-engine mode.

    Unlike /api/neural/render (real-time, one note at a time, DSP-first-
    then-swap -- see NeuralNoteBridge in index.html), this processes the
    WHOLE uploaded track through analysis -> harmony -> DSP synthesis
    first, then sends the ENTIRE resulting robot-voice track through the
    neural sidecar in a single call and waits for it, however long that
    takes. Nothing plays in the browser until this request resolves.

    This is the actual answer to real-time neural conversion never
    keeping up with a live singer: it isn't disabled here, it's just
    moved off the real-time path entirely, same idea as bouncing a
    track in a DAW instead of monitoring a plugin live.
    """
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
            _render_track_offline, raw, pitch_method, texture, voice_index, (mode or None)
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[neural track render error] {e}")
        return JSONResponse({"error": f"Render failed: {e}"}, status_code=500)


def _convert_track_direct(raw_audio_bytes: bytes, voice_index: int) -> bytes:
    """
    Sends the uploaded track's ACTUAL audio straight through the trained
    RVC voice model -- no harmony engine, no VocableSynthesizer, no
    generated aah/ooo/mmm/hey content in between. This is the same thing
    RVC WebUI's Model Inference tab does: HuBERT extracts whatever real
    phonetic content is actually in the file (real words, if the file has
    real singing/speech in it) and the generator resynthesizes it in the
    trained voice's timbre.

    Unlike _render_track_offline (which generates a NEW robot vocal part
    from scratch to accompany the track), this REPLACES the voice on the
    track you uploaded -- the output IS the track, re-voiced, not a
    second part layered on top of it.

    Note: if the uploaded file has instrumental backing mixed in with the
    vocals (i.e. it's a normal song, not an isolated vocal stem), that
    backing gets fed into HuBERT too. RVC-Project's own docs recommend an
    isolated vocal stem for best results -- this function doesn't do that
    separation for you, it converts exactly the audio you sent it. If
    results are muddy, running the file through a vocal separator (UVR5,
    Demucs, etc.) first and uploading just the vocal stem will help a lot.
    """
    from config.config_loader import get_config
    import librosa

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    timeout_s = float(cfg.get("synthesis", {}).get("neural", {}).get("offline_render_timeout_s", 900))
    converted = neural_timbre.convert_blocking(audio, sample_rate, voice_index=voice_index, timeout_s=timeout_s)
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
):
    """
    Direct RVC conversion of the uploaded file's own audio -- the 'give
    me the actual words back, just sung in the trained voice' mode.

    Compare to /api/neural/render-track: that endpoint runs analysis ->
    harmony engine -> VocableSynthesizer to generate a brand new wordless
    accompaniment part (aah/ooo/mmm/hey) and THEN sends that generated
    audio through RVC, so the output was never going to contain words no
    matter what voice model you point it at -- there were never words in
    what it was converting. This endpoint skips all of that and sends
    your uploaded audio's own content straight through RVC instead.
    """
    if neural_timbre is None or not neural_timbre.enabled or not neural_timbre._reachable:
        return JSONResponse(
            {"error": "Neural stage not ready -- check synthesis.neural.enabled in "
                      "config.yaml and that neural_env/rvc_server.py is running. "
                      "See GET /neural/status for details."},
            status_code=503,
        )
    try:
        raw = await file.read()
        wav_bytes = await asyncio.to_thread(_convert_track_direct, raw, voice_index)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[neural direct convert error] {e}")
        return JSONResponse({"error": f"Conversion failed: {e}"}, status_code=500)


def _analyze_pitch_offline(raw_audio_bytes: bytes, pitch_method: str) -> dict:
    """
    Runs in a worker thread (see asyncio.to_thread in analyze_pitch_track
    below). Analyzes an entire uploaded track's pitch ONCE, up front, and
    returns a fixed-hop timeline covering the whole file.

    This is the pitch-only counterpart to _render_track_offline above,
    and exists for exactly the same reason: RMVPE is a CNN+BiGRU that
    routinely takes longer than one audio frame's real-time budget on
    CPU. Streaming a pre-recorded upload's audio to /ws/pitch the same
    way a live mic has to would hit that same wall -- the backend falls
    behind, and what it reports drifts further from what's actually
    playing the longer the track runs. Since the whole file already
    exists up front, there's no reason to pretend it's a live stream:
    analyze it once, return a timeline, and let the frontend look pitch
    up by playback time instead.
    """
    from config.config_loader import get_config
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector
    import librosa

    cfg = get_config()
    sample_rate = cfg["audio"]["sample_rate"]
    frame_size = cfg["audio"]["frame_size"]
    frame_time_s = frame_size / sample_rate

    audio, in_sr = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if in_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sample_rate)
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

    return {"frame_time_s": frame_time_s, "hz": hz_timeline, "confidence": conf_timeline}


@app.post("/api/pitch/analyze-track")
async def analyze_pitch_track(
    file: UploadFile = File(...),
    pitch_method: str = Form("rmvpe"),
):
    """
    Offline whole-track pitch analysis for the Upload-MP3 demo panel.

    Used when the panel's pitch engine is set to RMVPE: instead of
    streaming the uploaded track's audio to /ws/pitch frame-by-frame
    (the same real-time RMVPE path a live mic has no choice but to use,
    and the same one whose CPU inference time exceeds the real-time
    frame budget), the whole file is analyzed once, here, and the
    resulting timeline is sent back for the frontend to look up by
    playback time. YIN doesn't need this at all -- it stays on its
    existing cheap real-time path -- this endpoint is RMVPE-specific.
    """
    try:
        raw = await file.read()
        result = await asyncio.to_thread(_analyze_pitch_offline, raw, pitch_method)
        return result
    except Exception as e:
        print(f"[pitch analyze error] {e}")
        return JSONResponse({"error": f"Pitch analysis failed: {e}"}, status_code=500)


def _transcribe_audio_blob(raw: bytes) -> dict:
    """Runs in a worker thread (see /api/transcribe) -- writes the
    uploaded audio to a temp file (Whisper/ffmpeg need a real file path,
    not in-memory bytes) and transcribes it. Cleans up the temp file
    either way. Returns diagnostics alongside the text -- specifically
    Whisper's own no_speech_prob per segment, its internal confidence that
    a stretch of audio contains no speech at all (0=confident there IS
    speech, 1=confident there ISN'T) -- so an empty result is debuggable
    (audio arrived but Whisper judged it silence/noise) rather than a
    black box (was anything even received?)."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        result = whisper_model.transcribe(tmp_path, language="en", fp16=False)
        segments = result.get("segments", [])
        no_speech_probs = [round(s.get("no_speech_prob", 0), 3) for s in segments]
        return {
            "text": result["text"].strip(),
            "bytes_received": len(raw),
            "segment_count": len(segments),
            "no_speech_probs": no_speech_probs,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Server-side English speech-to-text via Whisper, for the sentiment
    demo's mic input. See the WHISPER_MODEL_DIR loading block above for
    why this exists instead of the browser's built-in speech recognition:
    that only works in official Chrome/Edge, this works in every browser
    that can record audio at all.
    """
    if whisper_model is None:
        return JSONResponse({"error": f"Whisper not loaded: {whisper_load_error}"}, status_code=503)
    try:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "No audio data received"}, status_code=400)
        diag = await asyncio.to_thread(_transcribe_audio_blob, raw)
        if not diag["text"]:
            print(f"[transcribe] empty result -- {diag['bytes_received']} bytes received, "
                  f"{diag['segment_count']} segments, no_speech_probs={diag['no_speech_probs']}")
        return diag
    except Exception as e:
        print(f"[transcribe error] {e}")
        return JSONResponse({"error": f"Transcription failed: {e}"}, status_code=500)


@app.get("/api/transcribe/health")
def transcribe_health():
    return {"status": "ok" if whisper_model else "not_loaded", "error": whisper_load_error}


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
    """Browser streams float32 PCM -> Python runs aubio YIN -> sends back note JSON."""
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
            # A single frame of audio arrives as raw bytes; a mode switch
            # arrives as JSON text on the same socket. receive() (rather
            # than receive_bytes()) lets both share this one connection, so
            # the frontend can flip pitch algorithms without tearing down
            # and reconnecting the mic stream — which is what "switch mid-
            # song" actually requires, since a reconnect would drop audio
            # and reset every bit of rhythm/harmony state below.
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

            # Same preprocessing path the local pipeline uses: one silence
            # check, plus RMS normalization so quiet browser-mic input
            # actually reaches the pitch detector at a usable level instead
            # of being fed in raw. Previously this handler recomputed its
            # own silence check inline and skipped normalization entirely,
            # so the web demo and the local pipeline were quietly running
            # on two different signal paths.
            clean_frame, is_voiced = preproc.process(frame)
            # This call was missing entirely. Without it, rhythm.phrase_state
            # never leaves its starting value and rhythm.current_tempo never
            # leaves 0.0, which meant call_and_response and the tempo-based
            # switch to contour_following could never actually trigger on
            # this path, only in the local pipeline, which does call this.
            rhythm.push_frame(clean_frame, is_voiced)

            archer_hz       = None
            phoneme_profile = cree._neutral_profile

            if is_voiced:
                # RMVPE gets the raw frame, not clean_frame -- see
                # preprocessor.py's per-frame RMS renormalization, which
                # introduces a gain discontinuity at every frame boundary.
                # YIN handles that fine (hence clean_frame unchanged for
                # it); RMVPE's mel-spectrogram over a long continuous
                # window does not, and it has its own internal amplitude
                # handling already (confirmed in test_rmvpe_standalone.py,
                # which never renormalizes per-frame either).
                pitch_input = frame if pitch.method == "rmvpe" else clean_frame
                # Off the event loop for the same reason as /ws/pitch (see
                # comment there) -- inline RMVPE inference here would block
                # every other open connection this process is serving.
                hz, conf = await asyncio.to_thread(pitch.detect, pitch_input)
                # pitch.detect() already applies the correct method-specific
                # confidence gate and returns hz=None below it -- don't
                # re-check `conf` against the YIN-only cfg value here (see
                # the detailed comment in /ws/pitch for why that silently
                # dropped valid RMVPE readings).
                if hz:
                    archer_hz = hz
                phoneme_profile = cree.analyze(clean_frame)
            else:
                pitch.reset()

            phrase = rhythm.phrase_state
            if archer_hz and phrase in ("silence", "phrase_end"):
                phrase = "singing"

            elapsed_s = time.perf_counter() - start

            # Sovereignty: checked every frame, independent of translation/pitch
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
    """
    Lightweight pitch-only sibling of /ws/mic: no rhythm/harmony/Cree
    state, just "audio in -> {hz, note, confidence, method} out". This is
    what the frontend's live-demo pitch toggle (YIN vs RMVPE) talks to,
    since running actual RMVPE inference in the browser isn't practical —
    it's a trained U-Net + BiGRU, not a few lines of DSP like the YIN path
    the demos already run client-side in JS. Streaming the mic to this
    endpoint instead keeps the "toggle mid-song" requirement working
    without duplicating /ws/mic's full harmony pipeline for a demo page
    that only needs the detected note.

    Query param `method` ("yin" | "yinfft" | "rmvpe") sets the starting
    algorithm; the same {"type":"set_pitch_method","method":...} control
    message /ws/mic accepts works here too, for switching mid-stream.
    """
    await ws.accept()
    try:
        from config.config_loader import get_config
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        import librosa as _lib

        cfg     = get_config()
        preproc = Preprocessor()
        pitch   = PitchDetector()

        # The mic-capture panels force their AudioContext to 44100Hz to
        # match config.yaml, but the Upload-MP3 panel opens a plain
        # `new AudioContext()` with no rate specified -- the browser/OS
        # picks (commonly 48000). Trusting cfg's sample_rate here instead
        # of what the client actually sent silently mis-resamples every
        # downstream pitch calculation. Apply the client's real rate
        # before touching pitch.detect() at all.
        requested_sample_rate = ws.query_params.get("sample_rate")
        if requested_sample_rate:
            try:
                pitch.set_source_sample_rate(int(float(requested_sample_rate)))
            except ValueError:
                pass

        requested_method = ws.query_params.get("method")
        if requested_method:
            # set_method() synchronously loads the RMVPE checkpoint the
            # first time it's ever requested in this process (later calls
            # hit PitchDetector's process-wide model cache and return
            # near-instantly). Running that first, one-time load in a
            # thread keeps it from blocking every other open connection
            # on this single-process event loop while it happens.
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
                # See the identical comment in /ws/mic above: RMVPE gets
                # the raw frame, YIN keeps getting clean_frame.
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
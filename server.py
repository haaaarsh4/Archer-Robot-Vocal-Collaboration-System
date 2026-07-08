import asyncio
import json
import os
import sys
import threading
import time
import queue as _queue
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

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


# ------------------------------------------------------------------ #
# STATIC ROUTES
# ------------------------------------------------------------------ #

@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/architecture")
@app.get("/sentiment")
@app.get("/live-demo")
def spa_routes():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ------------------------------------------------------------------ #
# DEVICE LISTING (used by the frontend's device dropdown)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# TRANSLATION (Cree -> English)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# PIPELINE CONTROL (local mode only)
# ------------------------------------------------------------------ #

@app.get("/pipeline/status")
def pipeline_status():
    return {"running": pipeline_running, "audio_available": AUDIO_AVAILABLE}


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


# ------------------------------------------------------------------ #
# WEBSOCKET — broadcast from local pipeline to browser
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# WEBSOCKET — browser streams mic audio → server analyses → returns note JSON
# ------------------------------------------------------------------ #

@app.websocket("/ws/mic")
async def ws_mic(ws: WebSocket):
    """Browser streams float32 PCM -> Python runs aubio YIN -> sends back note JSON."""
    await ws.accept()
    try:
        from config.config_loader import get_config
        from analysis.pitch_detector import PitchDetector
        from analysis.rhythm_analyzer import RhythmAnalyzer
        from analysis.phonetic_analysis import CreeTokenizer
        from synthesis.harmony_engine import HarmonyEngine
        import librosa as _lib

        cfg     = get_config()
        pitch   = PitchDetector()
        rhythm  = RhythmAnalyzer()
        cree    = CreeTokenizer()
        harmony = HarmonyEngine()
        conf_thresh = cfg["pitch"]["confidence_threshold"]
        sil_db      = cfg["preprocessing"]["silence_threshold_db"]
        start       = time.perf_counter()

        from aubio import db_spl

        while True:
            data  = await ws.receive_bytes()
            frame = np.frombuffer(data, dtype=np.float32).copy()
            if len(frame) == 0:
                continue

            db        = float(db_spl(frame))
            is_voiced = np.isfinite(db) and db > sil_db
            rhythm.push_frame(frame, is_voiced)

            archer_hz       = None
            phoneme_profile = cree._neutral_profile

            if is_voiced:
                hz, conf = pitch.detect(frame)
                if hz and conf >= conf_thresh and pitch.min_freq < hz < pitch.max_freq:
                    archer_hz = hz
                phoneme_profile = cree.analyze(frame)
            else:
                pitch.reset()

            phrase = rhythm.phrase_state
            if archer_hz and phrase in ("silence", "phrase_end"):
                phrase = "singing"

            decision = harmony.decide(
                archer_hz=archer_hz, phrase_state=phrase,
                tempo_bpm=rhythm.current_tempo, phoneme_profile=phoneme_profile,
            )

            msg = {
                "type":         "pitch" if archer_hz else "silence",
                "singer_hz":    round(archer_hz, 1) if archer_hz else None,
                "singer_note":  _lib.hz_to_note(archer_hz) if archer_hz else None,
                "robot_hz":     round(decision.target_hz, 1) if decision.target_hz else None,
                "robot_note":   _lib.hz_to_note(decision.target_hz) if decision.target_hz and decision.target_hz > 0 else None,
                "action":       decision.action,
                "tempo_bpm":    round(rhythm.current_tempo, 1),
                "phrase_state": phrase,
                "elapsed_s":    round(time.perf_counter() - start, 1),
            }
            await ws.send_text(json.dumps(msg))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/mic error] {e}")


# ------------------------------------------------------------------ #
# LOCAL PIPELINE (only used when running on a machine with a mic)
# ------------------------------------------------------------------ #

def run_local_pipeline(stop_event, input_device: int):
    global pipeline_running
    try:
        from config.config_loader import get_config
        from core.audio_capture import AudioCapture
        from core.preprocessor import Preprocessor
        from analysis.pitch_detector import PitchDetector
        from analysis.rhythm_analyzer import RhythmAnalyzer
        from analysis.phonetic_analysis import CreeTokenizer
        from synthesis.harmony_engine import HarmonyEngine
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
        harmony      = HarmonyEngine()
        synth        = VocableSynthesizer()
        timing       = TimingSync()
        conf_thresh  = cfg["pitch"]["confidence_threshold"]

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
                hz, conf = pitch.detect(clean)
                if hz and conf >= conf_thresh and pitch.min_freq < hz < pitch.max_freq:
                    archer_hz = hz
                cree.analyze(clean)
            else:
                pitch.reset()

            timing.update_tempo(rhythm.current_tempo)
            phrase = rhythm.phrase_state
            if archer_hz and phrase in ("silence", "phrase_end"):
                phrase = "singing"

            decision = harmony.decide(
                archer_hz=archer_hz, phrase_state=phrase,
                tempo_bpm=rhythm.current_tempo,
                phoneme_profile=cree._neutral_profile,
            )

            if decision.action == "sing":
                timing.schedule(synth.synthesize(decision), decision.action)
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
                    "tempo_bpm":    round(rhythm.current_tempo, 1),
                    "phrase_state": phrase,
                    "elapsed_s":    round(time.perf_counter() - start, 1),
                }
                try:
                    broadcast_queue.put_nowait(msg)
                except _queue.Full:
                    pass
    except Exception as e:
        print(f"[pipeline error] {e}")
    finally:
        pipeline_running = False


# ------------------------------------------------------------------ #
# ENTRYPOINT
# ------------------------------------------------------------------ #

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

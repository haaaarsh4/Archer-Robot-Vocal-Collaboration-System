import argparse
import asyncio
import json
import os
import sys
import threading
import time
import queue as _queue
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

broadcast_queue: _queue.Queue = _queue.Queue(maxsize=64)

app = FastAPI()
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

pipeline_thread = None
pipeline_running = False
pipeline_stop_event = threading.Event()


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/devices")
def list_devices():
    import pyaudio
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            devices.append({"index": i, "name": info["name"],
                             "default_sr": int(info.get("defaultSampleRate", 44100))})
    p.terminate()
    return {"devices": devices}


@app.post("/pipeline/start")
def pipeline_start(body: dict):
    global pipeline_thread, pipeline_running, pipeline_stop_event
    if pipeline_running:
        return JSONResponse({"ok": False, "error": "already running"})
    device = body.get("input_device")
    if device is None:
        return JSONResponse({"ok": False, "error": "input_device required"})
    pipeline_stop_event.clear()
    pipeline_thread = threading.Thread(
        target=run_pipeline, args=(int(device), pipeline_stop_event), daemon=True
    )
    pipeline_thread.start()
    pipeline_running = True
    return {"ok": True}


@app.post("/pipeline/stop")
def pipeline_stop():
    global pipeline_running
    pipeline_stop_event.set()
    pipeline_running = False
    while not broadcast_queue.empty():
        try:
            broadcast_queue.get_nowait()
        except Exception:
            break
    return {"ok": True}


@app.get("/pipeline/status")
def pipeline_status():
    return {"running": pipeline_running}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
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


def run_pipeline(input_device: int, stop_event: threading.Event):
    from loguru import logger
    logger.remove()
    os.makedirs("logs", exist_ok=True)
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    from config.config_loader import get_config
    cfg = get_config()
    cfg["audio"]["input_device"] = input_device

    from core.audio_capture import AudioCapture
    from core.preprocessor import Preprocessor
    from analysis.pitch_detector import PitchDetector
    from analysis.rhythm_analyzer import RhythmAnalyzer
    from analysis.cree_tokenizer import CreeTokenizer
    from synthesis.harmony_engine import HarmonyEngine
    from synthesis.vocable_synthesizer import VocableSynthesizer
    from output.timing_sync import TimingSync

    capture      = AudioCapture()
    preprocessor = Preprocessor()
    pitch        = PitchDetector()
    rhythm       = RhythmAnalyzer()
    cree         = CreeTokenizer()
    harmony      = HarmonyEngine()
    synth        = VocableSynthesizer()
    timing       = TimingSync()

    confidence_threshold = cfg["pitch"]["confidence_threshold"]
    timing.start()
    capture.start()
    logger.info(f"Pipeline started on device {input_device}")

    frame_count = 0
    start_time  = time.perf_counter()
    currently_singing = False

    try:
        while not stop_event.is_set():
            try:
                frame = capture.queue.get(timeout=0.1)
            except Exception:
                continue

            frame_count += 1
            clean_frame, is_voiced = preprocessor.process(frame)
            rhythm.push_frame(clean_frame, is_voiced)

            archer_hz       = None
            phoneme_profile = cree._neutral_profile

            if is_voiced:
                archer_hz, confidence = pitch.detect(clean_frame)
                if (archer_hz is not None
                        and confidence >= confidence_threshold
                        and pitch.min_freq < archer_hz < pitch.max_freq):
                    pass
                else:
                    archer_hz = None
                phoneme_profile = cree.analyze(clean_frame)

            if not is_voiced:
                pitch.reset()

            timing.update_tempo(rhythm.current_tempo)

            phrase_state = rhythm.phrase_state
            if archer_hz is not None and phrase_state in ("silence", "phrase_end"):
                phrase_state = "singing"

            decision = harmony.decide(
                archer_hz       = archer_hz,
                phrase_state    = phrase_state,
                tempo_bpm       = rhythm.current_tempo,
                phoneme_profile = phoneme_profile,
            )

            if decision.action == "sing":
                audio = synth.synthesize(decision)
                timing.schedule(audio, decision.action)
                currently_singing = True
            elif decision.action == "sustain":
                currently_singing = True
            else:
                if currently_singing:
                    timing.flush()
                currently_singing = False

            import librosa as _librosa
            singer_note = _librosa.hz_to_note(archer_hz) if archer_hz else None
            robot_note  = _librosa.hz_to_note(decision.target_hz) if decision.target_hz and decision.target_hz > 0 else None

            msg = {
                "type":         "pitch" if archer_hz else "silence",
                "singer_hz":    round(archer_hz, 1) if archer_hz else None,
                "singer_note":  singer_note,
                "robot_hz":     round(decision.target_hz, 1) if decision.target_hz else None,
                "robot_note":   robot_note,
                "action":       decision.action,
                "tempo_bpm":    round(rhythm.current_tempo, 1),
                "phrase_state": phrase_state,
                "elapsed_s":    round(time.perf_counter() - start_time, 1),
                "frames":       frame_count,
            }
            try:
                broadcast_queue.put_nowait(msg)
            except _queue.Full:
                pass

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        capture.stop()
        timing.stop()
        logger.info("Pipeline stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="Archer-Robot web server")
    parser.add_argument("-port", default=8000, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n  Archer-Robot server starting")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"  Start the pipeline from the website\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
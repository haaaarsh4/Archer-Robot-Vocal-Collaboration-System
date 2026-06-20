import time
import threading
import queue
import numpy as np
from loguru import logger
from config.config_loader import get_config
import sounddevice as sd

# Receives (audio, decision) pairs from the synthesis pipeline and plays them at the right moment
class TimingSync:
    def __init__(self):
        cfg = get_config()
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.response_delay_ms = cfg["timing"]["response_delay_ms"]
        self.max_hold_ms = cfg["timing"]["max_hold_ms"]
        self.output_device = cfg["audio"]["output_device"]
        self.volume = cfg["output"]["volume"]

        self._playback_queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: threading.Thread | None = None
        self._running = False

        self._last_play_time: float = 0.0
        self._current_tempo_bpm: float = 0.0

    # Creates and starts the background playback thread:
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        logger.info("TimingSync playback thread started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("TimingSync stopped")

    # Receive the latest tempo from the rhythm analyzer
    def update_tempo(self, bpm: float):
        self._current_tempo_bpm = bpm

    # Calculates when to play and drops it in the queue
    def schedule(self, audio: np.ndarray, action: str):
        if action == "rest":
            return

        play_at = self._compute_play_time()

        try:
            self._playback_queue.put_nowait((audio, play_at))
        except queue.Full:
            logger.debug("Playback queue full — dropping stale audio chunk")

    # Compute the wall-clock time at which to play the next note
    def _compute_play_time(self) -> float:
        now = time.perf_counter()
        target = now + (self.response_delay_ms / 1000.0)

        if self._current_tempo_bpm > 0:
            beat_dur = 60.0 / self._current_tempo_bpm
            # How far into the current beat are we?
            if self._last_play_time > 0:
                elapsed_since_last = now - self._last_play_time
                phase = elapsed_since_last % beat_dur
                remaining_in_beat = beat_dur - phase
                # Snap forward to next beat if we're close enough
                if remaining_in_beat < beat_dur * 0.25:
                    target = now + remaining_in_beat
                elif remaining_in_beat > beat_dur * 0.75:
                    target = now + beat_dur - phase

        return target

    # Runs in its own thread. Waits until play_at, then outputs audio
    def _playback_loop(self):
        while self._running:
            try:
                audio, play_at = self._playback_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Wait until the scheduled time
            now = time.perf_counter()
            wait = play_at - now
            if wait > 0:
                time.sleep(wait)

            # Safety: skip if audio is stale (more than max_hold_ms old)
            staleness = time.perf_counter() - play_at
            if staleness > (self.max_hold_ms / 1000.0):
                logger.debug(f"Skipping stale audio ({staleness*1000:.0f}ms old)")
                continue

            try:
                sd.play(
                    audio,
                    samplerate=self.sample_rate,
                    device=self.output_device,
                    blocking=False,
                )
                self._last_play_time = time.perf_counter()
            except Exception as e:
                logger.error(f"Playback error: {e}")

    def flush(self):
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except Exception:
                break
        sd.stop()

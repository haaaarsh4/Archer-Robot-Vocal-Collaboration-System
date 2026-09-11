import time
import threading
import numpy as np
from loguru import logger
from config.config_loader import get_config
import sounddevice as sd


class TimingSync:
    """
    Receives (audio, decision) pairs from the synthesis pipeline and feeds
    them into ONE continuous sounddevice output stream.

    This used to fire a separate sd.play() call per chunk from a
    background thread. That doesn't work the way it looks like it does:
    sd.play() manages a single global "currently playing" buffer, so
    calling it again while a previous chunk is still sounding stops that
    chunk immediately and starts the new one from sample zero -- it does
    not queue or overlap. That meant every chunk handed to schedule() was
    a hard interruption of whatever was still playing, no matter how
    carefully VocableSynthesizer._crossfade() blended the sample arrays
    together upstream -- that blended tail never actually reached the
    speaker, because the chunk it was blended into had already been cut
    off. It also meant the beat-snap delay in the old _compute_play_time
    was applied to every chunk equally, including the sustain "refill"
    chunks used to keep a held note going (see server.py) -- so a note
    that was supposed to be one continuous tone could get an artificial
    gap stuffed into the middle of it while it waited to land on a beat.

    Now: a single OutputStream runs continuously once start() is called,
    pulling from a plain list of pending sample arrays (self._pending).
    schedule() just appends to that list -- a "sustain"/continuation
    chunk lands immediately after whatever's already queued, sample-
    accurate, no interruption, no re-triggered playback. Only a genuine
    new note ("sing", or resuming after the buffer actually ran dry) gets
    the beat-aware response delay, inserted as literal silence samples
    ahead of it in the same buffer -- so that delay is exact too, instead
    of being at the mercy of a playback thread's wake-up jitter.
    """

    def __init__(self):
        cfg = get_config()
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.channels = cfg["audio"].get("channels", 1)
        self.response_delay_ms = cfg["timing"]["response_delay_ms"]
        self.max_hold_ms = cfg["timing"]["max_hold_ms"]
        self.output_device = cfg["audio"]["output_device"]
        # NOTE: volume is already applied once, upstream, in
        # VocableSynthesizer.synthesize(). Not re-applied here -- the
        # original code also read this and never used it, so this isn't
        # a behavior change, just documenting it instead of leaving a
        # silently-dead config read.
        self.volume = cfg["output"]["volume"]

        self._buffer_lock = threading.Lock()
        self._pending: list[np.ndarray] = []      # sample arrays, in play order
        self._had_content_last_note: bool = False  # did the buffer have audio in it as of the last schedule() call?
        self._last_onset_perf: float = 0.0         # perf_counter time of the last genuine note attack, for beat-snap math

        self._current_tempo_bpm: float = 0.0
        self._stream: sd.OutputStream | None = None
        self._running = False

    def start(self):
        self._running = True
        self._pending = []
        self._had_content_last_note = False
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            device=self.output_device,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info("TimingSync output stream started")

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"TimingSync stream close error (non-fatal): {e}")
            self._stream = None
        logger.info("TimingSync stopped")

    def update_tempo(self, bpm: float):
        self._current_tempo_bpm = bpm

    def schedule(self, audio: np.ndarray, action: str):
        if action == "rest":
            return

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)

        max_samples = int(self.max_hold_ms / 1000.0 * self.sample_rate)

        with self._buffer_lock:
            backlog = sum(len(c) for c in self._pending)
            if backlog + len(audio) > max_samples:
                # Mirrors the old "queue full, drop stale chunk" behavior:
                # if we're this far behind real-time, queuing more audio
                # just makes the live performance feel laggier, not better.
                logger.debug("Playback backlog too large — dropping incoming chunk")
                return

            # A genuine new note gets the beat-aware response delay. A
            # "sustain" continuation only gets it too if the buffer had
            # actually run dry since the last chunk (a real gap already
            # happened, so treat it like a fresh attack) -- otherwise it
            # goes straight on the end of what's already queued, with
            # zero added delay, so a held note stays one unbroken tone.
            is_new_onset = (action == "sing") or not self._had_content_last_note
            if is_new_onset:
                delay_s = self._compute_onset_delay_s()
                if delay_s > 0:
                    self._pending.append(np.zeros(int(delay_s * self.sample_rate), dtype=np.float32))
                self._last_onset_perf = time.perf_counter() + delay_s

            self._pending.append(audio)
            self._had_content_last_note = True

    def _compute_onset_delay_s(self) -> float:
        base = self.response_delay_ms / 1000.0
        if self._current_tempo_bpm <= 0 or self._last_onset_perf <= 0:
            return base

        beat_dur = 60.0 / self._current_tempo_bpm
        elapsed = time.perf_counter() - self._last_onset_perf
        phase = elapsed % beat_dur
        remaining = beat_dur - phase

        if remaining < beat_dur * 0.25:
            return remaining
        if remaining > beat_dur * 0.75:
            return beat_dur - phase
        return base

    def _callback(self, outdata, frames, time_info, status):
        if status:
            logger.debug(f"TimingSync stream status: {status}")

        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._buffer_lock:
            while filled < frames and self._pending:
                chunk = self._pending[0]
                take = min(len(chunk), frames - filled)
                out[filled:filled + take] = chunk[:take]
                filled += take
                if take < len(chunk):
                    self._pending[0] = chunk[take:]
                else:
                    self._pending.pop(0)
            if not self._pending:
                self._had_content_last_note = False

        if self.channels == 1:
            outdata[:, 0] = out
        else:
            outdata[:] = np.tile(out.reshape(-1, 1), (1, self.channels))

    def flush(self):
        with self._buffer_lock:
            self._pending.clear()
            self._had_content_last_note = False
        # No sd.stop() needed -- the stream keeps running and the
        # callback above just emits silence once _pending is empty,
        # which is cheaper than tearing down and restarting a stream
        # every time a phrase ends.

import numpy as np
import collections
from loguru import logger
from config.config_loader import get_config
import librosa

class RhythmAnalyzer:
    PHRASE_STATES = ("singing", "phrase_end", "silence")

    # Loads settings from config and sets up all the state variables
    def __init__(self):
        cfg = get_config()
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.frame_size = cfg["audio"]["frame_size"]
        self.onset_threshold = cfg["rhythm"]["onset_threshold"]
        self.min_bpm = cfg["rhythm"]["min_bpm"]
        self.max_bpm = cfg["rhythm"]["max_bpm"]
        self.phrase_end_silence_ms = cfg["rhythm"]["phrase_end_silence_ms"]

        # Rolling buffer: keep 2 seconds of frames for onset/tempo analysis
        buffer_frames = int(2.0 * self.sample_rate / self.frame_size)
        self._buffer = collections.deque(maxlen=buffer_frames)

        # State tracking
        self.current_tempo: float = 0.0         # BPM
        self.latest_onset: float | None = None  # timestamp in seconds
        self.phrase_state: str = "silence"

        # Silence counter for phrase detection
        self._silent_frame_count: int = 0
        self._phrase_end_frames = int(
            (self.phrase_end_silence_ms / 1000.0) * self.sample_rate / self.frame_size
        )
        self._frame_time: float = self.frame_size / self.sample_rate
        self._elapsed: float = 0.0

        # Onset history for tempo estimation
        self._onset_times: collections.deque = collections.deque(maxlen=16)

    # Ingest a new audio frame and update all rhythm state.
    def push_frame(self, frame: np.ndarray, is_voiced: bool):
        self._elapsed += self._frame_time
        self._buffer.append(frame)

        # Update phrase state
        self._update_phrase_state(is_voiced)

        # Only run onset detection on voiced frames
        if is_voiced and len(self._buffer) >= 4:
            self._detect_onset_in_buffer()
            self._estimate_tempo()

    # tracks where Archer is in a musical phrase. 
    # It has three possible states and transitions between them based on silence
    def _update_phrase_state(self, is_voiced: bool):
        if is_voiced:
            self._silent_frame_count = 0
            self.phrase_state = "singing"
        else:
            self._silent_frame_count += 1
            if self._silent_frame_count >= self._phrase_end_frames:
                if self.phrase_state == "singing":
                    self.phrase_state = "phrase_end"
                    logger.debug(f"Phrase ended at t={self._elapsed:.2f}s")
                else:
                    self.phrase_state = "silence"

    # Looks through the 2 second rolling buffer for the most recent note attack using spectral flux
    def _detect_onset_in_buffer(self):
        try:
            audio = np.concatenate(list(self._buffer))
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.frame_size // 4,
                delta=self.onset_threshold,
                units="time",
            )

            if len(onset_frames) > 0:
                last_onset_in_buffer = float(onset_frames[-1])
                # Convert buffer-relative time to absolute session time
                buffer_duration = len(self._buffer) * self._frame_time
                abs_onset = self._elapsed - buffer_duration + last_onset_in_buffer

                # Only record if this is newer than the last onset we stored
                if not self._onset_times or abs_onset > self._onset_times[-1] + 0.05:
                    self._onset_times.append(abs_onset)
                    self.latest_onset = abs_onset
                    logger.debug(f"Onset detected at t={abs_onset:.3f}s")

        except Exception as e:
            logger.error(f"Onset detection error: {e}")

    # Estimate the current BPM from the rolling audio buffer
    def _estimate_tempo(self):
        if len(self._onset_times) < 3:
            return

        try:
            audio = np.concatenate(list(self._buffer))
            tempo, _ = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate,
                bpm=self.current_tempo if self.current_tempo > 0 else None,
            )
            # Constrain to reasonable vocal range
            tempo = float(np.clip(tempo, self.min_bpm, self.max_bpm))
            if tempo != self.current_tempo:
                self.current_tempo = tempo
                logger.debug(f"Tempo updated: {tempo:.1f} BPM")

        except Exception as e:
            logger.error(f"Tempo estimation error: {e}")

    # helper function to convert current BPM into seconds per beat
    @property
    def beat_duration_seconds(self) -> float:
        if self.current_tempo <= 0:
            return 0.5  
        return 60.0 / self.current_tempo

    @property
    def is_phrase_active(self) -> bool:
        return self.phrase_state == "singing"

    def reset(self):
        self._buffer.clear()
        self._onset_times.clear()
        self.current_tempo = 0.0
        self.latest_onset = None
        self.phrase_state = "silence"
        self._silent_frame_count = 0
        self._elapsed = 0.0
        logger.info("RhythmAnalyzer reset")

import numpy as np
import collections
from loguru import logger
from config.config_loader import get_config
import librosa


class RhythmAnalyzer:
    PHRASE_STATES = ("singing", "phrase_end", "silence")

    def __init__(self):
        cfg = get_config()
        self.sample_rate            = cfg["audio"]["sample_rate"]
        self.frame_size             = cfg["audio"]["frame_size"]
        self.onset_threshold        = cfg["rhythm"]["onset_threshold"]
        self.min_bpm                = cfg["rhythm"]["min_bpm"]
        self.max_bpm                = cfg["rhythm"]["max_bpm"]
        self.phrase_end_silence_ms  = cfg["rhythm"]["phrase_end_silence_ms"]

        buffer_frames = int(2.0 * self.sample_rate / self.frame_size)
        self._buffer: collections.deque = collections.deque(maxlen=buffer_frames)

        self.current_tempo: float       = 0.0
        self.latest_onset:  float|None  = None
        self.phrase_state:  str         = "silence"

        self._silent_frame_count: int = 0
        self._phrase_end_frames: int = int(
            (self.phrase_end_silence_ms / 1000.0) * self.sample_rate / self.frame_size
        )
        self._frame_time: float = self.frame_size / self.sample_rate
        self._elapsed:    float = 0.0

        self._onset_times: collections.deque = collections.deque(maxlen=16)

        self._frames_since_onset_check: int = 0
        self._onset_check_interval: int = buffer_frames // 2  

        self._frames_since_tempo_check: int = 0
        self._tempo_check_interval: int = buffer_frames   


    def push_frame(self, frame: np.ndarray, is_voiced: bool):
        self._elapsed += self._frame_time
        self._buffer.append(frame)

        self._update_phrase_state(is_voiced)

        if not is_voiced:
            return   

        self._frames_since_onset_check += 1
        self._frames_since_tempo_check += 1

        if (self._frames_since_onset_check >= self._onset_check_interval
                and len(self._buffer) >= self._onset_check_interval):
            self._detect_onset_in_buffer()
            self._frames_since_onset_check = 0

        if (self._frames_since_tempo_check >= self._tempo_check_interval
                and len(self._onset_times) >= 3):
            self._estimate_tempo()
            self._frames_since_tempo_check = 0


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

            if len(onset_frames) == 0:
                return

            buffer_duration = len(self._buffer) * self._frame_time
            buffer_start    = self._elapsed - buffer_duration

            for t_rel in onset_frames:
                abs_onset = buffer_start + float(t_rel)
                if not self._onset_times or abs_onset > self._onset_times[-1] + 0.03:
                    self._onset_times.append(abs_onset)
                    self.latest_onset = abs_onset
                    logger.debug(f"Onset detected at t={abs_onset:.3f}s")

        except Exception as e:
            logger.error(f"Onset detection error: {e}")

    def _estimate_tempo(self):
        try:
            audio = np.concatenate(list(self._buffer))
            tempo, _ = librosa.beat.beat_track(
                y=audio,
                sr=self.sample_rate,
                bpm=self.current_tempo if self.current_tempo > 0 else None,
            )
            tempo = float(np.atleast_1d(tempo)[0])
            tempo = float(np.clip(tempo, self.min_bpm, self.max_bpm))

            if abs(tempo - self.current_tempo) > 2.0:   
                self.current_tempo = tempo
                logger.debug(f"Tempo updated: {tempo:.1f} BPM")

        except Exception as e:
            logger.error(f"Tempo estimation error: {e}")


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
        self.current_tempo  = 0.0
        self.latest_onset   = None
        self.phrase_state   = "silence"
        self._silent_frame_count = 0
        self._elapsed = 0.0
        self._frames_since_onset_check = 0
        self._frames_since_tempo_check = 0
        logger.info("RhythmAnalyzer reset")

import collections
import numpy as np
from loguru import logger
from config.config_loader import get_config
import librosa


class PitchDetector:
    ACCUMULATE_FRAMES: int = 8

    def __init__(self):
        cfg = get_config()
        self.engine = "pyin"
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.viterbi = cfg["pitch"]["viterbi_smoothing"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.frame_size  = cfg["audio"]["frame_size"]

        self.min_freq = float(cfg["pitch"]["min_frequency"])
        self.max_freq = float(cfg["pitch"]["max_frequency"])

        self._acc_buffer: collections.deque = collections.deque(
            maxlen=self.ACCUMULATE_FRAMES
        )

        self._last_hz:   float | None = None
        self._last_conf: float = 0.0

        logger.info(
            f"Pitch engine: pYIN  "
            f"(accumulate={self.ACCUMULATE_FRAMES} frames ≈ "
            f"{self.ACCUMULATE_FRAMES * self.frame_size / self.sample_rate * 1000:.0f} ms, "
            f"fmin={self.min_freq:.0f} Hz, fmax={self.max_freq:.0f} Hz)"
        )


    def detect(self, frame: np.ndarray) -> tuple[float | None, float]:
        self._acc_buffer.append(frame)

        if len(self._acc_buffer) < self.ACCUMULATE_FRAMES:
            return self._last_hz, self._last_conf

        window = np.concatenate(list(self._acc_buffer)).astype(np.float32)

        hz, conf = self._detect_pyin(window)

        if hz is not None and conf >= self.confidence_threshold:
            self._last_hz   = hz
            self._last_conf = conf
        elif hz is None:
            self._last_hz   = None
            self._last_conf = 0.0

        return self._last_hz, self._last_conf

    def reset(self):
        self._acc_buffer.clear()
        self._last_hz   = None
        self._last_conf = 0.0

    def _detect_pyin(self, audio: np.ndarray) -> tuple[float | None, float]:
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.min_freq,
                fmax=self.max_freq,
                sr=self.sample_rate,
            )

            voiced = f0[voiced_flag]
            if len(voiced) == 0:
                return None, 0.0

            hz   = float(np.nanmedian(voiced))
            conf = float(np.nanmedian(voiced_probs[voiced_flag]))

            return hz, conf

        except Exception as e:
            logger.error(f"pYIN pitch detection error: {e}")
            try:
                hz_arr = librosa.yin(
                    audio,
                    fmin=self.min_freq,
                    fmax=self.max_freq,
                    sr=self.sample_rate,
                )
                hz = float(np.nanmedian(hz_arr))
                if hz > 0:
                    return hz, 0.6   
                return None, 0.0
            except Exception as e2:
                logger.error(f"YIN fallback also failed: {e2}")
                return None, 0.0


    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))
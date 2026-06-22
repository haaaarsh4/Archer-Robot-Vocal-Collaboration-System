import numpy as np
import aubio
from loguru import logger
from config.config_loader import get_config
import librosa

class PitchDetector:

    HOP_SIZE = 512
    BUF_SIZE = 2048

    def __init__(self):
        cfg = get_config()
        self.sample_rate          = cfg["audio"]["sample_rate"]
        self.frame_size           = cfg["audio"]["frame_size"]
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.min_freq             = float(cfg["pitch"]["min_frequency"])
        self.max_freq             = float(cfg["pitch"]["max_frequency"])

        self._pitch_o = aubio.pitch("yin", self.BUF_SIZE, self.HOP_SIZE, self.sample_rate)
        self._pitch_o.set_unit("Hz")
        self._pitch_o.set_silence(-40)
        self._pitch_o.set_tolerance(0.8)

        self._leftovers   = np.array([], dtype=np.float32)
        self._history     = []
        self._HISTORY_LEN = 3

        logger.info(f"Pitch engine: aubio YIN (hop={self.HOP_SIZE}, buf={self.BUF_SIZE}, sr={self.sample_rate}, fmin={self.min_freq:.0f} Hz, fmax={self.max_freq:.0f} Hz)")

    def detect(self, frame: np.ndarray) -> tuple:
        samples   = np.concatenate([self._leftovers, frame.astype(np.float32)])
        best_hz   = None
        best_conf = 0.0
        i = 0

        while i + self.HOP_SIZE <= len(samples):
            hop  = samples[i : i + self.HOP_SIZE]
            hz   = float(self._pitch_o(hop)[0])
            conf = float(self._pitch_o.get_confidence())
            i   += self.HOP_SIZE
            if self.min_freq <= hz <= self.max_freq and conf >= self.confidence_threshold:
                best_hz   = hz
                best_conf = conf

        self._leftovers = samples[i:]

        if best_hz is None:
            return None, 0.0

        self._history.append(best_hz)
        if len(self._history) > self._HISTORY_LEN:
            self._history.pop(0)

        return float(np.median(self._history)), best_conf

    def reset(self):
        self._history.clear()
        self._leftovers = np.array([], dtype=np.float32)

    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))
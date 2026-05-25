import numpy as np
from loguru import logger
from config.config_loader import get_config
import librosa


class PitchDetector:
    # Loads how the detector is configured from the config file
    def __init__(self):
        cfg = get_config()
        self.engine = "pyin"
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.viterbi = cfg["pitch"]["viterbi_smoothing"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.min_freq = cfg["pitch"]["min_frequency"]
        self.max_freq = cfg["pitch"]["max_frequency"]
        logger.info("Pitch engine: pYIN")

    # Uses pYIN (via librosa) to detect pitch
    def detect(self, frame):
        return self._detect_pyin(frame)

    def _detect_pyin(self, frame: np.ndarray) -> tuple[float | None, float]:
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                frame,
                fmin=self.min_freq,
                fmax=self.max_freq,
                sr=self.sample_rate,
            )
            if f0 is None or len(f0) == 0:
                return None, 0.0

            freq = float(f0[-1]) if not np.isnan(f0[-1]) else None
            conf = float(voiced_probs[-1]) if voiced_probs is not None else 0.0

            if freq is None or conf < self.confidence_threshold:
                return None, conf

            return freq, conf

        except Exception as e:
            logger.error(f"pYIN pitch detection error: {e}")
            return None, 0.0

    # Creating a function to convert a frequency to the nearest note name, e.g. 440.0 → 'A4'.
    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        note = librosa.hz_to_note(freq_hz)
        return note

    # Another helper function that converts a frequency to a MIDI note number
    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))
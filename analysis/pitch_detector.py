import numpy as np
from loguru import logger
from config.config_loader import get_config
import crepe as _crepe
import librosa

class PitchDetector:
    # Loads how the detector is conkfigured from the config file
    def __init__(self):
        cfg = get_config()
        self.engine = cfg["pitch"]["engine"]
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.viterbi = cfg["pitch"]["viterbi_smoothing"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.min_freq = cfg["pitch"]["min_frequency"]
        self.max_freq = cfg["pitch"]["max_frequency"]

        self._crepe = None
        self._init_engine()

    # Load the pitch detector (CREPE library)
    def _init_engine(self):
        if self.engine == "crepe":
            try:
                self._crepe = _crepe
                logger.info("Pitch engine: CREPE loaded")
            except ImportError:
                logger.warning("CREPE not available — falling back to pYIN")
                self.engine = "pyin"
        if self.engine == "pyin":
            # librosa pYIN imported on demand in detect()
            logger.info("Pitch engine: pYIN")

    # Router to detect what engine is loaded
    def detect(self, frame):
        if self.engine == "crepe":
            return self._detect_crepe(frame)
        else:
            return self._detect_pyin(frame)

    # Uses the CREPE neural network to detect pitch
    def _detect_crepe(self, frame: np.ndarray) -> tuple[float | None, float]:
        try:
            _, frequency, confidence, _ = self._crepe.predict(
                frame,
                self.sample_rate,
                viterbi=self.viterbi,
                verbose=0,
            )
            freq = float(frequency[-1])
            conf = float(confidence[-1])

            if conf < self.confidence_threshold:
                return None, conf
            if not (self.min_freq <= freq <= self.max_freq):
                return None, conf

            return freq, conf

        except Exception as e:
            logger.error(f"CREPE pitch detection error: {e}")
            return None, 0.0

    # Uses pYIN (via librosa) to detect pitch
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

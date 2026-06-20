import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from config.config_loader import get_config
import librosa

try:
    import joblib as _joblib
except ImportError:
    _joblib = None


@dataclass
class PhonemeProfile:
    vowel_color    : float = 0.5
    nasality       : float = 0.0
    brightness     : float = 0.5
    detected_class : str   = "neutral"
    confidence     : float = 0.0
    influence      : float = 0.0


class CreeTokenizer:
    CREE_PHONEME_PROFILES = {
        "long_a":  {"brightness": 0.7, "vowel_color": 0.3, "nasality": 0.0},
        "long_e":  {"brightness": 0.8, "vowel_color": 0.1, "nasality": 0.0},
        "long_i":  {"brightness": 0.9, "vowel_color": 0.0, "nasality": 0.0},
        "long_o":  {"brightness": 0.4, "vowel_color": 0.8, "nasality": 0.0},
        "short_a": {"brightness": 0.6, "vowel_color": 0.4, "nasality": 0.1},
        "short_i": {"brightness": 0.8, "vowel_color": 0.1, "nasality": 0.1},
        "short_o": {"brightness": 0.3, "vowel_color": 0.9, "nasality": 0.1},
        "nasal_n": {"brightness": 0.4, "vowel_color": 0.5, "nasality": 0.9},
        "nasal_m": {"brightness": 0.3, "vowel_color": 0.6, "nasality": 1.0},
    }

    def __init__(self):
        cfg = get_config()
        self.enabled     = cfg["cree_tokenizer"]["enabled"]
        self.model_path  = cfg["cree_tokenizer"]["model_path"]
        self.influence   = cfg["cree_tokenizer"]["phoneme_influence"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        self._model = None
        self._neutral_profile = PhonemeProfile(influence=0.0)

        if self.enabled:
            self._load_model()
        else:
            logger.info("Cree tokenizer disabled — running in neutral mode")


    def _load_model(self):
        if self.model_path is None:
            logger.warning(
                "Cree tokenizer enabled but no model_path set in config. "
                "Using heuristic feature matching until model is ready."
            )
            return

        if _joblib is None:
            logger.error("joblib is not installed — cannot load Cree model.")
            return

        try:
            self._model = _joblib.load(self.model_path)
            logger.info(f"Cree phoneme model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Cree model: {e}")
            self._model = None


    def analyze(self, frame: np.ndarray) -> PhonemeProfile:
        if not self.enabled:
            return self._neutral_profile

        if self._model is not None:
            return self._model_predict(frame)
        else:
            return self._heuristic_analyze(frame)


    def _model_predict(self, frame: np.ndarray) -> PhonemeProfile:
        try:
            mfccs = librosa.feature.mfcc(y=frame, sr=self.sample_rate, n_mfcc=13)
            feature_vector = np.mean(mfccs, axis=1).reshape(1, -1)

            probs      = self._model.predict_proba(feature_vector)[0]
            class_idx  = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            class_name = list(self.CREE_PHONEME_PROFILES.keys())[class_idx]

            profile_params = self.CREE_PHONEME_PROFILES[class_name]
            return PhonemeProfile(
                vowel_color    = profile_params["vowel_color"],
                nasality       = profile_params["nasality"],
                brightness     = profile_params["brightness"],
                detected_class = class_name,
                confidence     = confidence,
                influence      = self.influence * confidence,
            )
        except Exception as e:
            logger.error(f"Cree model prediction error: {e}")
            return self._neutral_profile

    def _heuristic_analyze(self, frame: np.ndarray) -> PhonemeProfile:
        try:
            mfccs = librosa.feature.mfcc(y=frame, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=frame, sr=self.sample_rate
            )

            brightness = float(np.clip(
                (np.mean(spectral_centroid) - 80) / (4000 - 80), 0.0, 1.0
            ))

            vowel_color = float(np.clip(
                (np.mean(mfccs[1]) + 50) / 100, 0.0, 1.0
            ))

            nasality = float(np.clip(
                np.mean(np.abs(mfccs[3:6])) / 30, 0.0, 1.0
            ))

            return PhonemeProfile(
                vowel_color    = vowel_color,
                nasality       = nasality,
                brightness     = brightness,
                detected_class = "heuristic",
                confidence     = 0.5,
                influence      = self.influence * 0.5,
            )

        except Exception as e:
            logger.error(f"Cree heuristic analysis error: {e}")
            return self._neutral_profile

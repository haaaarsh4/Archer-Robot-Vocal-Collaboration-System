import numpy as np
from loguru import logger
from config.config_loader import get_config


class Preprocessor:

    # Load the gate thresold and normalize target from the config file
    def __init__(self):
        cfg = get_config()
        self.gate_threshold = cfg["preprocessing"]["noise_gate_threshold"]
        self.normalize_target = cfg["preprocessing"]["normalize_target_rms"]

    # Take an audio frame and return a cleaned up version along with whether Archer is singing 
    def process(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        rms = self._rms(frame)

        if rms < self.gate_threshold:
            return np.zeros_like(frame), False

        if rms > 0:
            frame = frame * (self.normalize_target / rms)

        frame = np.clip(frame, -1.0, 1.0)

        return frame, True


    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame ** 2)))

    def update_threshold(self, new_threshold: float):
        self.gate_threshold = new_threshold
        logger.info(f"Noise gate threshold updated to {new_threshold:.4f}")

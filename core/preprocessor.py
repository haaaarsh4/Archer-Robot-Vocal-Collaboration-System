import numpy as np
import aubio
from loguru import logger
from config.config_loader import get_config

class Preprocessor:

    LOG_EVERY_N = 43

    def __init__(self):
        cfg = get_config()
        self.normalize_target = cfg["preprocessing"]["normalize_target_rms"]
        self.silence_db       = cfg["preprocessing"]["silence_threshold_db"]
        self._frame_count     = 0
        self._voiced_count    = 0
        logger.info(f"Preprocessor ready — silence threshold: {self.silence_db} dBFS")

    def process(self, frame: np.ndarray) -> tuple:
        samples = frame.astype(np.float32)
        self._frame_count += 1

        db        = float(aubio.db_spl(samples))
        is_voiced = np.isfinite(db) and db > self.silence_db

        if is_voiced:
            self._voiced_count += 1

        if self._frame_count % self.LOG_EVERY_N == 0:
            pct = 100 * self._voiced_count / self._frame_count
            logger.debug(f"[mic] {db:+6.1f} dBFS  threshold={self.silence_db} dBFS  → {'VOICED' if is_voiced else 'SILENT'}  (voiced so far: {pct:.0f}%)")

        if not is_voiced:
            return np.zeros_like(samples), False

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms > 1e-8:
            samples = samples * (self.normalize_target / rms)

        return np.clip(samples, -1.0, 1.0), True

    def update_threshold(self, db: float):
        self.silence_db = db
        logger.info(f"Silence threshold updated to {db} dBFS")
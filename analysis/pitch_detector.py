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
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.min_freq             = float(cfg["pitch"]["min_frequency"])
        self.max_freq             = float(cfg["pitch"]["max_frequency"])

        # "yin" is the classic time-domain method already in use here.
        # "yinfft" is a spectral variant aubio also ships, and tends to hold
        # up better on real-world, non-isolated audio (background noise,
        # room tone) at a small extra CPU cost. Configurable rather than
        # picked for you, since which one performs better depends on your
        # actual input signal, not something to assume from a synthetic test.
        pitch_method = cfg["pitch"].get("method", "yin")

        self._pitch_o = aubio.pitch(pitch_method, self.BUF_SIZE, self.HOP_SIZE, self.sample_rate)
        self._pitch_o.set_unit("Hz")
        # Was hardcoded to -40 here, independently of preprocessing.silence_threshold_db.
        # If that config value ever changed, this one silently wouldn't follow it,
        # and the two silence gates could disagree without any error or warning.
        self._pitch_o.set_silence(cfg["preprocessing"]["silence_threshold_db"])
        # 0.15 is aubio's own documented default for this parameter. The
        # previous 0.8 was far outside that. It didn't reproduce a failure
        # in clean synthetic testing here, but it's still being reset to a
        # value aubio's own docs consider reasonable rather than left as an
        # unexplained outlier.
        self._pitch_o.set_tolerance(cfg["pitch"].get("tolerance", 0.15))

        self._leftovers   = np.array([], dtype=np.float32)
        self._history     = []
        self._HISTORY_LEN = 3

        # How far a new pitch can drift from the last stable one, in cents,
        # before it's treated as a genuinely new note rather than smoothed
        # in with recent history. Without this, a real jump between two
        # different notes gets briefly averaged into a pitch that matches
        # neither, right at the moment it matters most.
        self._jump_tolerance_cents = cfg["pitch"].get("jump_tolerance_cents", 80.0)
        self._last_stable_hz: float | None = None

        logger.info(
            f"Pitch engine: aubio {pitch_method} (hop={self.HOP_SIZE}, buf={self.BUF_SIZE}, "
            f"sr={self.sample_rate}, fmin={self.min_freq:.0f} Hz, fmax={self.max_freq:.0f} Hz)"
        )

    def detect(self, frame: np.ndarray) -> tuple:
        samples = np.concatenate([self._leftovers, frame.astype(np.float32)])

        candidates: list[tuple[float, float]] = []  # (hz, confidence) for every in-range hop this call
        i = 0
        while i + self.HOP_SIZE <= len(samples):
            hop  = samples[i:i + self.HOP_SIZE]
            hz   = float(self._pitch_o(hop)[0])
            conf = float(self._pitch_o.get_confidence())
            i   += self.HOP_SIZE
            if self.min_freq <= hz <= self.max_freq and conf >= self.confidence_threshold:
                candidates.append((hz, conf))
        self._leftovers = samples[i:]

        if not candidates:
            return None, 0.0

        # Actually pick the highest-confidence candidate from this batch,
        # not just whichever one happened to be processed last.
        best_hz, best_conf = max(candidates, key=lambda c: c[1])

        smoothed_hz = self._smooth(best_hz)
        return smoothed_hz, best_conf

    def _smooth(self, hz: float) -> float:
        """
        Median-of-recent smoothing, but aware of real jumps. A new pitch
        more than _jump_tolerance_cents away from the last stable reading
        is treated as a new note: history resets to it immediately instead
        of blending it in, which is what caused a couple of frames of
        wrong, in-between output right at the start of every real pitch
        change. Small drift (vibrato, natural pitch wobble) still gets
        smoothed normally.
        """
        if self._last_stable_hz is not None and self._last_stable_hz > 0:
            cents_diff = abs(1200.0 * np.log2(hz / self._last_stable_hz))
            if cents_diff > self._jump_tolerance_cents:
                self._history = [hz]
                self._last_stable_hz = hz
                return hz

        self._history.append(hz)
        if len(self._history) > self._HISTORY_LEN:
            self._history.pop(0)

        result = float(np.median(self._history))
        self._last_stable_hz = result
        return result

    def reset(self):
        self._history.clear()
        self._leftovers = np.array([], dtype=np.float32)
        self._last_stable_hz = None

    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))

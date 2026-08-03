import numpy as np
import aubio
from loguru import logger
from config.config_loader import get_config
import librosa

try:
    import soxr  # noqa: F401 -- if present, librosa.resample can use the higher-quality soxr backend
    _HAS_SOXR = True
except ImportError:
    _HAS_SOXR = False


_RMVPE_MODEL_CACHE: dict = {}


class _RMVPEStream:
    TARGET_SR = 16000
    HOP_SAMPLES_16K = 160   # RMVPE's native 10ms hop at 16kHz
    MIN_CONTEXT_HOPS = 32   # DeepUnet's 2**5 time-downsampling needs >= this many hops of context

    def __init__(self, model, source_sample_rate: int, context_seconds: float = 1.0,
                 voicing_threshold: float = 0.03, infer_stride_hops: int = 1):
        self._model = model
        self._voicing_threshold = voicing_threshold
        self._context_seconds = context_seconds
        self._infer_stride_hops = max(1, int(infer_stride_hops))
        self._pushes_since_infer = 0
        self._raw_buffer = np.zeros(0, dtype=np.float32)
        self._last_hz: float | None = None
        self._last_conf: float = 0.0
        self.set_source_sample_rate(source_sample_rate)

    def set_source_sample_rate(self, source_sample_rate: int):
        self._source_sr = int(source_sample_rate)
        self._context_samples_raw = max(
            int(self._context_seconds * self._source_sr),
            int(self.MIN_CONTEXT_HOPS * self.HOP_SAMPLES_16K / self.TARGET_SR * self._source_sr),
        )
        self._min_raw_needed = int(self.HOP_SAMPLES_16K * 4 / self.TARGET_SR * self._source_sr)
        self.reset()

    def reset(self):
        self._raw_buffer = np.zeros(0, dtype=np.float32)
        self._pushes_since_infer = 0
        self._last_hz = None
        self._last_conf = 0.0

    def push(self, frame: np.ndarray) -> tuple:
        self._raw_buffer = np.concatenate([self._raw_buffer, frame.astype(np.float32)])
        if len(self._raw_buffer) > self._context_samples_raw:
            self._raw_buffer = self._raw_buffer[-self._context_samples_raw:]

        if len(self._raw_buffer) < self._min_raw_needed:
            return None, 0.0  # not enough audio yet for a meaningful window

        self._pushes_since_infer += 1
        if self._pushes_since_infer < self._infer_stride_hops and self._last_hz is not None:
            return self._last_hz, self._last_conf
        self._pushes_since_infer = 0

        if self._source_sr != self.TARGET_SR:
            audio_16k = librosa.resample(
                self._raw_buffer,
                orig_sr=self._source_sr, target_sr=self.TARGET_SR,
                res_type="soxr_hq" if _HAS_SOXR else "kaiser_best",
            )
        else:
            audio_16k = self._raw_buffer

        f0_hops, conf_hops = self._model.infer(audio_16k, voicing_threshold=self._voicing_threshold)
        if len(f0_hops) == 0:
            return None, 0.0

        hz = float(f0_hops[-1])
        conf = float(conf_hops[-1])
        self._last_hz = hz if hz > 0 else None
        self._last_conf = conf
        return self._last_hz, conf


class PitchDetector:

    HOP_SIZE = 512
    BUF_SIZE = 2048

    def __init__(self):
        cfg = get_config()
        self.sample_rate          = cfg["audio"]["sample_rate"]
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.min_freq             = float(cfg["pitch"]["min_frequency"])
        self.max_freq             = float(cfg["pitch"]["max_frequency"])

        self._pitch_method = cfg["pitch"].get("method", "yin")
        self._silence_threshold_db = cfg["preprocessing"]["silence_threshold_db"]
        self._tolerance = cfg["pitch"].get("tolerance", 0.15)

        self._pitch_o = self._build_pitch_object()

        self._rmvpe_cfg = cfg["pitch"].get("rmvpe", {}) or {}
        self._rmvpe_stream: _RMVPEStream | None = None
        if self._pitch_method == "rmvpe":
            self._load_rmvpe()

        self._leftovers   = np.array([], dtype=np.float32)
        self._history     = []
        self._HISTORY_LEN = 3

        self._jump_tolerance_cents = cfg["pitch"].get("jump_tolerance_cents", 80.0)
        self._last_stable_hz: float | None = None

        self._pending_jump_hz: float | None = None

        logger.info(
            f"Pitch engine: {self._pitch_method} "
            f"(hop={self.HOP_SIZE}, buf={self.BUF_SIZE}, sr={self.sample_rate}, "
            f"fmin={self.min_freq:.0f} Hz, fmax={self.max_freq:.0f} Hz)"
        )

    def _build_pitch_object(self):
        aubio_method = self._pitch_method if self._pitch_method in ("yin", "yinfft") else "yin"
        pitch_o = aubio.pitch(aubio_method, self.BUF_SIZE, self.HOP_SIZE, self.sample_rate)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(self._silence_threshold_db)
        pitch_o.set_tolerance(self._tolerance)
        return pitch_o

    def _load_rmvpe(self):
        if self._rmvpe_stream is not None:
            return
        from analysis.rmvpe_detector import RMVPEModel  # heavy import (torch) -- kept lazy

        model_path = self._rmvpe_cfg.get("model_path", "assets/rmvpe/rmvpe.pt")
        device = self._rmvpe_cfg.get("device", "cpu")
        is_half = bool(self._rmvpe_cfg.get("is_half", False))
        context_seconds = float(self._rmvpe_cfg.get("context_seconds", 1.0))
        voicing_threshold = float(self._rmvpe_cfg.get("voicing_threshold", 0.03))
        infer_stride_hops = int(self._rmvpe_cfg.get("infer_stride_hops", 1))
        self._rmvpe_confidence_threshold = float(self._rmvpe_cfg.get("confidence_threshold", 0.15))
        self._rmvpe_debug_counter = 0

        cache_key = (model_path, device, is_half)
        model = _RMVPE_MODEL_CACHE.get(cache_key)
        if model is not None:
            logger.info(f"RMVPE: reusing already-loaded model for '{model_path}' on {device} (warm)")
        else:
            try:
                model = RMVPEModel(model_path, device=device, is_half=is_half)
            except FileNotFoundError:
                logger.error(
                    f"RMVPE checkpoint not found at '{model_path}'. Download it from "
                    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt "
                    "and set pitch.rmvpe.model_path in config.yaml, or place it at the default "
                    "path above. Staying on YIN for this session."
                )
                self._pitch_method = "yin"
                return
            except Exception as e:
                logger.error(f"Failed to load RMVPE ({e}). Staying on YIN for this session.")
                self._pitch_method = "yin"
                return
            _RMVPE_MODEL_CACHE[cache_key] = model
            logger.info(f"RMVPE loaded (cold, first time) from '{model_path}' on {device} (half={is_half})")

        self._rmvpe_stream = _RMVPEStream(
            model=model, source_sample_rate=self.sample_rate,
            context_seconds=context_seconds, voicing_threshold=voicing_threshold,
            infer_stride_hops=infer_stride_hops,
        )

    def set_source_sample_rate(self, sample_rate: int):
        sample_rate = int(sample_rate)
        if sample_rate == self.sample_rate:
            return
        logger.info(f"Pitch detector source sample rate: {self.sample_rate} -> {sample_rate}")
        self.sample_rate = sample_rate
        self._pitch_o = self._build_pitch_object()
        if self._rmvpe_stream is not None:
            self._rmvpe_stream.set_source_sample_rate(sample_rate)
        self.reset()

    def set_method(self, method: str) -> str:
        method = (method or "").lower()
        if method not in ("yin", "yinfft", "rmvpe"):
            raise ValueError(f"Unknown pitch method: {method!r} (expected yin/yinfft/rmvpe)")
        if method == self._pitch_method:
            return self._pitch_method

        previous = self._pitch_method
        if method == "rmvpe":
            self._load_rmvpe()  # sets self._pitch_method itself; falls back to "yin" on failure
            if self._rmvpe_stream is not None:
                self._pitch_method = "rmvpe"
        else:
            self._pitch_method = method
            self._pitch_o = self._build_pitch_object()

        if self._pitch_method != previous:
            logger.info(f"Pitch method switched: {previous} -> {self._pitch_method}")
        self.reset()
        return self._pitch_method

    @property
    def method(self) -> str:
        return self._pitch_method

    def detect(self, frame: np.ndarray) -> tuple:
        if self._pitch_method == "rmvpe" and self._rmvpe_stream is not None:
            return self._detect_rmvpe(frame)
        return self._detect_aubio(frame)

    def _detect_aubio(self, frame: np.ndarray) -> tuple:
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

    def _detect_rmvpe(self, frame: np.ndarray) -> tuple:
        hz, conf = self._rmvpe_stream.push(frame)

        self._rmvpe_debug_counter += 1
        if self._rmvpe_debug_counter % 20 == 0:  # ~4x/sec at 512-sample hops @44.1kHz -- enough to see real numbers, not enough to flood the log
            logger.debug(
                f"[rmvpe raw] hz={hz if hz else 0:.1f} conf={conf:.3f} "
                f"(gate: conf>={self._rmvpe_confidence_threshold}, hz in [{self.min_freq:.0f},{self.max_freq:.0f}])"
            )

        if hz is None or conf < self._rmvpe_confidence_threshold or not (self.min_freq <= hz <= self.max_freq):
            return None, conf

        smoothed_hz = self._smooth(hz)
        return smoothed_hz, conf

    def _smooth(self, hz: float) -> float:
        if self._last_stable_hz is not None and self._last_stable_hz > 0:
            cents_diff = abs(1200.0 * np.log2(hz / self._last_stable_hz))
            if cents_diff > self._jump_tolerance_cents:
                if self._pending_jump_hz is not None:
                    confirm_diff = abs(1200.0 * np.log2(hz / self._pending_jump_hz))
                    if confirm_diff <= self._jump_tolerance_cents:
                        # Two consecutive frames agree on the new pitch — commit it.
                        self._history = [self._pending_jump_hz, hz]
                        self._last_stable_hz = hz
                        self._pending_jump_hz = None
                        return hz
                # Not confirmed (or contradicts a still-pending candidate).
                # Hold the old stable pitch and wait one more frame.
                self._pending_jump_hz = hz
                return self._last_stable_hz
            else:
                self._pending_jump_hz = None
        else:
            self._pending_jump_hz = None

        self._history.append(hz)
        if len(self._history) > self._HISTORY_LEN:
            self._history.pop(0)

        result = float(np.median(self._history))
        self._last_stable_hz = result
        return result

    def reset(self):
        rmvpe_dirty = self._rmvpe_stream is not None and len(self._rmvpe_stream._raw_buffer) > 0
        already_clean = (
            not self._history
            and self._last_stable_hz is None
            and self._pending_jump_hz is None
            and len(self._leftovers) == 0
            and not rmvpe_dirty
        )
        if already_clean:
            return

        self._history.clear()
        self._leftovers = np.array([], dtype=np.float32)
        self._last_stable_hz = None
        self._pending_jump_hz = None
        self._pitch_o = self._build_pitch_object()
        if self._rmvpe_stream is not None:
            self._rmvpe_stream.reset()

    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))

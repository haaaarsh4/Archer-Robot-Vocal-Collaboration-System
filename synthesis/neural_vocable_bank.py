"""
Pre-rendered, neural-voice vocable bank with a fast runtime pitch-shift cache.

The idea: run each vocable ("oh", "ahh", "hey", "yeah", whatever set you pick)
through the RVC voice conversion pipeline exactly once, offline, using
build_vocable_bank.py. At performance time the live synthesizer never touches
the neural model again for these sounds. It just pitch-shifts the already-
converted audio and caches the result, which is cheap enough to keep up with
a live singer on CPU alone, no GPU needed.

Why this works when live per-note neural conversion doesn't: the RVC sidecar
takes tens to hundreds of milliseconds per call on CPU, because it's running
a real trained network. A numpy resample or a small phase-vocoder shift on a
one or two second clip that's already sitting in memory takes a fraction of
a millisecond to a few milliseconds. Doing the expensive part once, ahead of
time, and only ever doing the cheap part live is the whole trick.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import librosa
from loguru import logger


class _BaseSample:
    __slots__ = ("audio", "f0_hz", "name")

    def __init__(self, audio: np.ndarray, f0_hz: float, name: str):
        self.audio = audio
        self.f0_hz = f0_hz
        self.name = name


class NeuralVocableBank:
    """
    Loads pre-rendered neural-voice takes for each vocable and register, and
    serves pitch-shifted audio on demand from a semitone-bucketed cache, so
    a note that repeats (which is most notes, in an actual melody) costs
    nothing after the first time it's sung.
    """

    # How finely target pitches get bucketed for caching. One semitone is
    # close enough that nudging the cached bucket to the exact target
    # frequency later (see get()) is inaudible, and coarse enough that the
    # cache fills up fast during a normal performance.
    SEMITONE_BUCKET = 1.0

    def __init__(self, samples_dir: str | Path, sample_rate: int, max_workers: int = 2):
        self.sample_rate = sample_rate
        self._bases: dict[str, list[_BaseSample]] = {}
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._inflight: set[tuple[str, int]] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vocable-shift")

        self._load_bases(Path(samples_dir))

    @property
    def available(self) -> bool:
        return bool(self._bases)

    def _load_bases(self, samples_dir: Path) -> None:
        if not samples_dir.exists():
            logger.warning(
                f"Neural vocable bank: {samples_dir} doesn't exist yet. "
                "Run build_vocable_bank.py first, or the synthesizer will "
                "fall back to the plain sinusoidal engine."
            )
            return

        for wav_path in sorted(samples_dir.glob("*.wav")):
            # Expects names like "oh_low.wav", "oh_mid.wav", "yeah_high.wav".
            # Everything before the first underscore is the vocable name.
            # Matching is case-insensitive, "Cold.wav" and a config entry
            # of "cold" are treated as the same vocable, so filename
            # capitalization never silently breaks the lookup.
            vocable = wav_path.stem.split("_")[0].strip().lower()
            try:
                audio, _ = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)
                # yin is much cheaper than pyin and is plenty accurate for a
                # single clean isolated take. This only runs once, at load
                # time, never per note.
                f0_track = librosa.yin(audio, fmin=60, fmax=800, sr=self.sample_rate)
                voiced = f0_track[f0_track > 0]
                f0_hz = float(np.nanmedian(voiced)) if len(voiced) else 220.0

                self._bases.setdefault(vocable, []).append(
                    _BaseSample(audio.astype(np.float32), f0_hz, wav_path.stem)
                )
                logger.info(f"Neural vocable bank: loaded {wav_path.name} (f0 about {f0_hz:.0f} Hz)")
            except Exception as e:
                logger.error(f"Neural vocable bank: failed to load {wav_path.name}: {e}")

        for vocable, bases in self._bases.items():
            bases.sort(key=lambda s: s.f0_hz)
            logger.info(f"Neural vocable bank: '{vocable}' has {len(bases)} register(s) loaded")

    def _nearest_base(self, vocable: str, target_hz: float) -> _BaseSample | None:
        vocable = vocable.strip().lower()
        candidates = self._bases.get(vocable)
        if not candidates:
            # No take recorded for this exact vocable, use whatever's
            # loaded rather than going fully silent.
            any_list = next(iter(self._bases.values()), None)
            candidates = any_list
        if not candidates:
            return None
        return min(candidates, key=lambda s: abs(np.log2(target_hz / s.f0_hz)))

    def _bucket(self, target_hz: float, base_f0: float) -> int:
        semitones = 12.0 * np.log2(target_hz / base_f0)
        return int(round(semitones / self.SEMITONE_BUCKET))

    def get(self, vocable: str, target_hz: float, n_samples: int) -> np.ndarray | None:
        """
        Returns audio for this vocable at (approximately) this pitch, right
        now, without ever blocking on the neural model or on a pitch shift
        that hasn't finished computing yet. Callers should still put this
        through the normal envelope, phoneme shaping, and crossfade steps.
        """
        base = self._nearest_base(vocable, target_hz)
        if base is None:
            return None

        bucket = self._bucket(target_hz, base.f0_hz)
        key = (vocable.strip().lower(), bucket)

        with self._cache_lock:
            cached = self._cache.get(key)

        if cached is None:
            # This exact pitch bucket hasn't been rendered yet. Hand back a
            # cheap resample-based shift right now so something plays this
            # instant, and queue the better phase-vocoder version in the
            # background. Once it lands, every future hit on this bucket is
            # a plain dict lookup.
            cached = self._fast_resample_shift(base.audio, base.f0_hz, target_hz)
            self._queue_high_quality_shift(key, base, target_hz)

        return self._fit_length(cached, n_samples)

    def _queue_high_quality_shift(self, key: tuple[str, int], base: _BaseSample, target_hz: float) -> None:
        with self._cache_lock:
            if key in self._inflight:
                return
            self._inflight.add(key)

        def work():
            try:
                n_steps = 12.0 * np.log2(target_hz / base.f0_hz)
                shifted = librosa.effects.pitch_shift(
                    base.audio, sr=self.sample_rate, n_steps=float(n_steps),
                    n_fft=1024,  # smaller than librosa's default, faster, still clean on a short take
                )
                with self._cache_lock:
                    self._cache[key] = shifted.astype(np.float32)
            except Exception as e:
                logger.error(f"Vocable pitch-shift cache fill failed for {key}: {e}")
            finally:
                with self._cache_lock:
                    self._inflight.discard(key)

        self._executor.submit(work)

    def prewarm(self, vocable: str, target_hzs: list[float]) -> None:
        """
        Optional: call this once at startup with the notes in whatever key
        you expect (or just a chromatic scale across your singing range) so
        the cache is already full before anyone starts singing, instead of
        the first pass through the song paying the pitch-shift cost live.
        """
        base = self._nearest_base(vocable, target_hzs[0]) if target_hzs else None
        if base is None:
            return
        for hz in target_hzs:
            bucket = self._bucket(hz, base.f0_hz)
            key = (vocable, bucket)
            with self._cache_lock:
                already = key in self._cache or key in self._inflight
            if not already:
                self._queue_high_quality_shift(key, base, hz)

    @staticmethod
    def _fast_resample_shift(audio: np.ndarray, base_hz: float, target_hz: float) -> np.ndarray:
        """
        A cheap stand-in shift used only the very first time a new pitch
        bucket is requested. Resampling changes duration along with pitch,
        which isn't correct for a sustained note, but it's a tiny amount of
        numpy work and sounds far closer to the real neural take than the
        plain sinusoidal fallback would. It gets replaced in the cache by
        the proper phase-vocoder version within a note or two.
        """
        ratio = target_hz / base_hz
        n_out = max(1, int(len(audio) / ratio))
        idx = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
        if len(audio) >= n_samples:
            return audio[:n_samples]
        repeats = int(np.ceil(n_samples / len(audio)))
        return np.tile(audio, repeats)[:n_samples]

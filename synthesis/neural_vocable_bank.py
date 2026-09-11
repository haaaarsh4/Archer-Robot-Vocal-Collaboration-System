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


# ============================================================
# Formant-preserving pitch shift
# ============================================================
# librosa.effects.pitch_shift moves EVERYTHING in the spectrum together --
# fundamental pitch and formants (the vocal-tract resonances that make a
# voice sound like an adult human voice) alike. That's fine for a small
# shift, but drone_support and octave_below both ask for a full octave
# (12 semitones) or more, and a real human throat doesn't work that way --
# your formants stay roughly fixed regardless of what note you're singing.
# Dragging them down right along with the pitch is exactly what produces
# the "monster voice" / "chipmunk in reverse" effect: technically the
# right pitch, but with a vocal-tract size that doesn't exist in nature.
#
# The fix is a standard cepstral technique: separate a sound into its
# spectral ENVELOPE (the slow-moving shape across frequency -- formants)
# and its EXCITATION (the fast-moving fine harmonic detail -- pitch).
# Pitch-shift as normal, then strip out whatever envelope the shift left
# behind and re-impose the ORIGINAL take's own envelope on top of the
# shifted excitation. The formants stay where a real voice would keep
# them; only the pitch actually moves.
def _cepstral_envelope(log_magnitude: np.ndarray, n_fft: int, n_coeffs: int) -> np.ndarray:
    """
    Smooths a single log-magnitude spectrum down to just its slow-moving
    envelope shape (formants) via real-cepstrum low-quefrency liftering,
    discarding the fast-moving harmonic fine structure (pitch).
    log_magnitude: shape (n_fft//2 + 1,)
    """
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft)
    lifter = np.zeros_like(cepstrum)
    lifter[:n_coeffs] = 1.0
    lifter[-(n_coeffs - 1):] = 1.0  # real cepstrum is symmetric; keep both ends
    return np.fft.rfft(cepstrum * lifter, n=n_fft).real


def _average_envelope(audio: np.ndarray, sr: int, n_fft: int = 1024,
                       hop_length: int = 256, n_coeffs: int = 30) -> np.ndarray:
    """
    One representative formant envelope for a whole (short, roughly
    steady-state vowel) take -- averaged across all its frames rather
    than tracked frame-by-frame, since the shifted output will generally
    have a different number of frames (a different duration) than the
    original once pitch-shifted, so there's no clean 1:1 frame
    correspondence to align against anyway. A single averaged envelope
    is standard practice for short, stationary vocable takes like these.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    log_mag = np.log(np.abs(stft) + 1e-8)
    avg_log_mag = np.mean(log_mag, axis=1)
    return _cepstral_envelope(avg_log_mag, n_fft, n_coeffs)


def _impose_envelope(audio: np.ndarray, target_envelope: np.ndarray, sr: int,
                      n_fft: int = 1024, hop_length: int = 256, n_coeffs: int = 30) -> np.ndarray:
    """
    Frame by frame: strips OUT whatever envelope `audio` currently has
    (flattening it back toward a neutral excitation) and imposes
    `target_envelope` in its place. Used to put the ORIGINAL take's own
    formants back onto its already pitch-shifted self.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft) + 1e-8
    phase = np.angle(stft)
    log_mag = np.log(magnitude)

    n_frames = log_mag.shape[1]
    corrected_log_mag = np.empty_like(log_mag)
    for i in range(n_frames):
        frame_envelope = _cepstral_envelope(log_mag[:, i], n_fft, n_coeffs)
        corrected_log_mag[:, i] = log_mag[:, i] - frame_envelope + target_envelope

    corrected_stft = np.exp(corrected_log_mag) * np.exp(1j * phase)
    return librosa.istft(corrected_stft, hop_length=hop_length, length=len(audio))


def _formant_preserving_pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Pitch-shifts `audio` by n_steps semitones while keeping its ORIGINAL
    formants in place, instead of dragging them along with the pitch.
    Falls back to a plain (uncorrected) shift for a small movement --
    correction has its own small risk of artifacts, and isn't worth
    that risk when the shift is modest enough that formant drag was
    never going to be very audible anyway -- and falls back the same
    way on any numerical trouble, so a rare bad correction degrades to
    "the old, already-shipped sound" rather than silence or garbage.
    """
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(n_steps), n_fft=1024)
    if abs(n_steps) < 2.0:
        return shifted.astype(np.float32)

    try:
        target_envelope = _average_envelope(audio, sr)
        corrected = _impose_envelope(shifted, target_envelope, sr)
        if not np.all(np.isfinite(corrected)):
            return shifted.astype(np.float32)

        # Envelope correction changes spectral SHAPE, which can shift
        # overall energy up or down as a side effect -- rescale so the
        # corrected version's loudness matches the plain shift's (RMS,
        # not peak -- preserves perceived loudness without being thrown
        # off by a single transient sample). This is a timbre fix, not a
        # volume control; it shouldn't hand back something noticeably
        # louder or quieter than what it's replacing.
        shifted_rms = float(np.sqrt(np.mean(shifted.astype(np.float64) ** 2)))
        corrected_rms = float(np.sqrt(np.mean(corrected.astype(np.float64) ** 2)))
        if corrected_rms > 1e-8 and shifted_rms > 1e-8:
            corrected = corrected * (shifted_rms / corrected_rms)

        peak = float(np.max(np.abs(corrected))) if corrected.size else 0.0
        if peak > 3.0 or peak == 0.0:
            return shifted.astype(np.float32)
        return corrected.astype(np.float32)
    except Exception as e:
        logger.warning(f"Formant correction failed, using plain pitch shift instead: {e}")
        return shifted.astype(np.float32)


def _spectral_brightness(audio: np.ndarray, sr: int) -> float:
    """
    A single number describing how bright/forward vs. warm/rounded a
    short vowel take actually sounds, measured from the real recording
    rather than guessed from its filename. Spectral centroid (the
    "center of mass" of the spectrum) is the standard, cheap way to do
    this: a bright vowel like "hey" or "ee" concentrates energy higher
    up and has a high centroid, a rounder one like "oh"/"ooo" sits
    lower. Returned in Hz (un-normalized) -- NeuralVocableBank._tag_
    brightness below is what turns a whole set of these into a
    comparable 0-1 scale across the vocables actually loaded.
    """
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return float(np.median(centroid))
    except Exception:
        return 1500.0  # a plausible mid-range fallback, never used for real ranking if this fails


class _BaseSample:
    __slots__ = ("audio", "f0_hz", "name", "brightness_hz", "brightness")

    def __init__(self, audio: np.ndarray, f0_hz: float, name: str, brightness_hz: float = 1500.0):
        self.audio = audio
        self.f0_hz = f0_hz
        self.name = name
        self.brightness_hz = brightness_hz
        # Normalized 0 (warmest loaded vocable) .. 1 (brightest loaded
        # vocable) -- filled in once every base is loaded, see
        # NeuralVocableBank._tag_relative_brightness. Defaults to a
        # neutral midpoint so nothing downstream crashes if it's ever
        # read before that pass runs.
        self.brightness = 0.5


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
        # Cache stores tuples: (audio, is_fast)
        # is_fast=True means it's a resample-based shift (changes duration)
        # is_fast=False means it's a phase-vocoder shift (preserves duration)
        self._cache: dict[tuple[str, int], tuple[np.ndarray, bool]] = {}
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
                brightness_hz = _spectral_brightness(audio, self.sample_rate)

                self._bases.setdefault(vocable, []).append(
                    _BaseSample(audio.astype(np.float32), f0_hz, wav_path.stem, brightness_hz)
                )
                logger.info(
                    f"Neural vocable bank: loaded {wav_path.name} "
                    f"(f0 about {f0_hz:.0f} Hz, brightness centroid about {brightness_hz:.0f} Hz)"
                )
            except Exception as e:
                logger.error(f"Neural vocable bank: failed to load {wav_path.name}: {e}")

        for vocable, bases in self._bases.items():
            bases.sort(key=lambda s: s.f0_hz)
            logger.info(f"Neural vocable bank: '{vocable}' has {len(bases)} register(s) loaded")

        self._tag_relative_brightness()

    def _tag_relative_brightness(self) -> None:
        """
        Turns each base sample's raw spectral-centroid Hz reading into a
        0-1 brightness score RELATIVE to every other vocable actually
        loaded in this bank, and averages that across a vocable's
        registers (low/mid/high takes of the same word can differ
        somewhat in brightness just from being sung at a different
        pitch, so one representative score per vocable is what the
        picker in harmony_engine.py actually consumes).

        Deliberately relative rather than fixed thresholds: "bright"
        and "warm" only mean something in comparison to the rest of
        the loaded set, and this way the picker keeps working sensibly
        whether the configured vocable_set is four wordless syllables
        or forty real words, without needing to hand-tune absolute Hz
        cutoffs for whatever set someone records.
        """
        if not self._bases:
            return
        all_centroids = [s.brightness_hz for bases in self._bases.values() for s in bases]
        lo, hi = min(all_centroids), max(all_centroids)
        span = hi - lo
        for bases in self._bases.values():
            for s in bases:
                s.brightness = 0.5 if span < 1e-6 else float(np.clip((s.brightness_hz - lo) / span, 0.0, 1.0))

    def vocable_brightness_map(self) -> dict[str, float]:
        """
        One representative 0-1 brightness score per loaded vocable
        (averaged across its registers), for harmony_engine.py's vocable
        picker to match against the melody's own register/trend instead
        of rotating through vocable_set blindly. Empty dict if nothing's
        loaded yet, which callers should treat as "no brightness data
        available, fall back to whatever simpler logic doesn't need it."
        """
        return {
            vocable: float(np.mean([s.brightness for s in bases]))
            for vocable, bases in self._bases.items()
            if bases
        }

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

        if cached is not None:
            # return the audio regardless of fast or not; fast is good enough for live
            return self._fit_length(cached[0], n_samples, self.sample_rate)

        # Not in cache: use fast resample and queue high-quality in background
        fast_audio = self._fast_resample_shift(base.audio, base.f0_hz, target_hz)
        with self._cache_lock:
            self._cache[key] = (fast_audio, True)   # mark as fast
        self._queue_high_quality_shift(key, base, target_hz)

        return self._fit_length(fast_audio, n_samples, self.sample_rate)

    def get_blocking(self, vocable: str, target_hz: float, n_samples: int) -> np.ndarray | None:
        """
        Same idea as get(), but for offline/batch rendering (see
        VocableSynthesizer's realtime=False mode), where there is no live
        performance to protect from stalling and therefore no reason to
        ever hand back the cheap resample-based placeholder from get().
        That placeholder ties pitch to duration (a resample-based octave
        shift plays back at half speed, doubling length), which is fine
        for a fraction of a second during a live performance before the
        real shift lands, but is completely wrong to leave baked into a
        finished render -- a batch job has no real-time budget to
        protect, so it should simply always compute and wait for the
        correct, duration-preserving phase-vocoder shift.

        Still fills and reuses the same cache as get(), but if the cache
        contains a *fast* entry (is_fast=True), we ignore it and compute
        the correct shift, then overwrite the cache with the high-quality
        result so future calls (even non-blocking) get the right audio.
        """
        base = self._nearest_base(vocable, target_hz)
        if base is None:
            return None

        bucket = self._bucket(target_hz, base.f0_hz)
        key = (vocable.strip().lower(), bucket)

        with self._cache_lock:
            cached = self._cache.get(key)

        if cached is not None and not cached[1]:
            # Good quality already available
            return self._fit_length(cached[0], n_samples, self.sample_rate)

        # Either missing or fast entry – compute high-quality synchronously
        try:
            n_steps = 12.0 * np.log2(target_hz / base.f0_hz)
            shifted = _formant_preserving_pitch_shift(base.audio, self.sample_rate, n_steps)
        except Exception as e:
            logger.error(f"Vocable pitch-shift failed for {key}: {e}")
            return None

        with self._cache_lock:
            self._cache[key] = (shifted, False)   # mark as high-quality
        return self._fit_length(shifted, n_samples, self.sample_rate)

    def _queue_high_quality_shift(self, key: tuple[str, int], base: _BaseSample, target_hz: float) -> None:
        with self._cache_lock:
            if key in self._inflight:
                return
            self._inflight.add(key)

        def work():
            try:
                n_steps = 12.0 * np.log2(target_hz / base.f0_hz)
                shifted = _formant_preserving_pitch_shift(base.audio, self.sample_rate, n_steps)
                with self._cache_lock:
                    # Overwrite any existing (fast) entry with the high-quality version
                    self._cache[key] = (shifted.astype(np.float32), False)
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

    def prewarm_range(self, fmin: float, fmax: float, step_semitones: int = 1) -> None:
        """
        Prewarm all vocables across the frequency range [fmin, fmax].
        Uses semitone steps to limit the number of cached pitches.
        """
        if not self.available:
            return
        # Generate frequencies in semitone steps
        hzs = []
        freq = fmin
        while freq <= fmax:
            hzs.append(freq)
            freq *= 2 ** (step_semitones / 12.0)
        for vocable in self._bases.keys():
            self.prewarm(vocable, hzs)

    @staticmethod
    def _fast_resample_shift(audio: np.ndarray, base_hz: float, target_hz: float) -> np.ndarray:
        """
        A cheap stand-in shift used only the very first time a new pitch
        bucket is requested. It gets replaced in the cache by the proper
        phase-vocoder version within a note or two.

        Shifting pitch by resampling also changes duration (a resample-
        based octave-down shift plays back at half speed) -- that's fine
        for the pitch, but wrong for timing, and audible as the whole
        note dragging or rushing for the brief window before the
        high-quality version lands. So this does the resample-based shift
        for pitch, same as before, then resamples the RESULT a second
        time back to the original sample count -- that restores the
        original duration on top of the already-shifted pitch. Two cheap
        numpy interpolations, still a fraction of a millisecond, and no
        longer ties note length to how far the note is being shifted.
        """
        ratio = target_hz / base_hz
        n_orig = len(audio)
        n_shifted = max(1, int(n_orig / ratio))
        idx = np.linspace(0, n_orig - 1, n_shifted)
        shifted = np.interp(idx, np.arange(n_orig), audio)

        idx_back = np.linspace(0, n_shifted - 1, n_orig)
        return np.interp(idx_back, np.arange(n_shifted), shifted).astype(np.float32)

    @staticmethod
    def _make_loop_safe(segment: np.ndarray, crossfade_len: int) -> np.ndarray:
        """
        Turns `segment` into one that can be repeated by plain
        concatenation without an audible seam. Standard sampler-loop
        technique: blend the segment's own TAIL into its HEAD (equal-
        power crossfade) to build a new, slightly shorter head that
        already contains a smooth transition from "end" back to
        "start" -- then the rest of the segment can just follow it
        untouched. Concatenating copies of the result back-to-back now
        flows smoothly at every repeat, because each copy's built-in
        blended head IS that transition.
        """
        n = len(segment)
        if crossfade_len <= 0 or crossfade_len * 2 >= n:
            return segment
        head = segment[:crossfade_len]
        tail = segment[n - crossfade_len:]
        t = np.linspace(0, np.pi / 2, crossfade_len)
        fade_in = np.sin(t)
        fade_out = np.cos(t)
        blended_head = tail * fade_out + head * fade_in
        body = segment[crossfade_len:n - crossfade_len]
        return np.concatenate([blended_head, body]).astype(np.float32)

    @staticmethod
    def _fit_length(audio: np.ndarray, n_samples: int, sample_rate: int = 44100) -> np.ndarray:
        """
        Extends or trims `audio` to exactly n_samples, WITHOUT ever
        fragmenting the recording's own internal structure.

        This used to split the take into attack / sustain / release and
        loop just the middle "sustain" slice to fill extra time -- exactly
        right for a single steady-state vowel (its middle really is one
        stable, loop-safe sound), but actively wrong for anything with
        real word structure. The vocables actually configured right now
        (vocable_set: Lalanana, ohoho; vocable_rare_set includes whole
        phrases like "welcome to reality") are not steady-state vowels,
        they're multi-syllable recordings with real onsets and
        consonants throughout. Looping an arbitrary slice out of the
        middle of "Lalanana" doesn't hold "Lalanana", it chops it at
        whatever syllable boundary the slice happened to land on and
        repeats THAT fragment -- a real recorded take of "La-la-na-na"
        could turn into a mechanically repeating "-lana-lana-lana-",
        which is a different, meaningless sound, not a held version of
        the word. That's not a subtle quality issue, it's the actual
        reason the sung output could stop being comprehensible as a word
        at all.

        Now the WHOLE recording is always kept intact, at every note
        length:
          - A note shorter than the take's natural duration speeds the
            ENTIRE recording up (phase-vocoder time-stretch, pitch
            unaffected), rather than chopping the front or back off --
            you hear the whole word delivered faster, not half a word.
          - A note longer than the take slows the ENTIRE recording down,
            up to MAX_STRETCH_RATIO. Real syllables just take longer to
            get through, the way a singer actually elongates a lyric
            over a long note.
          - Past MAX_STRETCH_RATIO, rather than stretching one utterance
            into unnaturally slow mush, the whole word is time-stretched
            once up to that ceiling and then repeated back-to-back --
            like a singer actually repeating a lyric across a very long
            held phrase -- with a real crossfade at the repeat boundary
            (_make_loop_safe, reused here at the level of a WHOLE
            repeated word rather than an arbitrary internal fragment).
        """
        n_orig = len(audio)
        if n_orig == 0:
            return np.zeros(n_samples, dtype=np.float32)
        if n_orig == n_samples:
            return audio.astype(np.float32)

        MAX_STRETCH_RATIO = 2.5   # how far one utterance is slowed down before repeating instead
        MIN_SQUEEZE_RATIO = 0.4   # how far it's sped up before we stop pushing it any faster

        def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
            try:
                return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
            except Exception as e:
                logger.warning(f"Vocable time-stretch failed ({e}); falling back to raw audio")
                return y

        target_stretch = n_samples / n_orig  # >1 means we need MORE audio (slow down); <1 means LESS (speed up)

        if target_stretch < 1.0:
            # Speeding the whole word up to fit a short note. librosa's
            # `rate` convention is inverted from ours (rate>1 shortens),
            # and floored at MIN_SQUEEZE_RATIO so a very short note
            # doesn't compress consonants into an unrecognizable blur --
            # past that floor we just accept the note ends slightly
            # early rather than mangling the word trying to hit it
            # exactly.
            stretch_rate = min(1.0 / target_stretch, 1.0 / MIN_SQUEEZE_RATIO)
            out = _stretch(audio, stretch_rate)
            if len(out) >= n_samples:
                return out[:n_samples].astype(np.float32)
            return np.pad(out, (0, n_samples - len(out))).astype(np.float32)

        if target_stretch <= MAX_STRETCH_RATIO:
            stretch_rate = 1.0 / target_stretch
            out = _stretch(audio, stretch_rate)
            if len(out) >= n_samples:
                return out[:n_samples].astype(np.float32)
            return np.pad(out, (0, n_samples - len(out))).astype(np.float32)

        # The note needs to hold for so much longer than the take's own
        # length that stretching it that far in one piece would turn it
        # to mush. Stretch once up to the ceiling, then repeat the
        # WHOLE (still fully intact) word, crossfading tail into head at
        # every repeat -- never a mid-word splice.
        stretched_once = _stretch(audio, 1.0 / MAX_STRETCH_RATIO)
        if len(stretched_once) == 0:
            stretched_once = audio

        crossfade_len = min(len(stretched_once) // 6, int(0.12 * sample_rate))
        loop_unit = (
            NeuralVocableBank._make_loop_safe(stretched_once, crossfade_len)
            if crossfade_len >= 8 else stretched_once
        )
        if len(loop_unit) == 0:
            loop_unit = stretched_once

        n_repeats = max(1, int(np.ceil(n_samples / len(loop_unit))))
        out = np.tile(loop_unit, n_repeats)[:n_samples]
        if len(out) < n_samples:
            out = np.pad(out, (0, n_samples - len(out)))
        return out[:n_samples] 
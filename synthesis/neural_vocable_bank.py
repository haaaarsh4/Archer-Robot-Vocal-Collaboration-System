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


# Formant-preserving pitch shift
# librosa.effects.pitch_shift moves EVERYTHING in the spectrum together
# fundamental pitch and formants (the vocal-tract resonances that make a
# voice sound like an adult human voice) alike. That's fine for a small
# shift, but drone_support and octave_below both ask for a full octave
# (12 semitones) or more, and a real human throat doesn't work that way
# your formants stay roughly fixed regardless of what note you're singing.
# Dragging them down right along with the pitch is exactly what produces
# the "monster voice" / "chipmunk in reverse" effect: technically the
# right pitch, but with a vocal-tract size that doesn't exist in nature.
#
# The fix is a standard cepstral technique: separate a sound into its
# spectral ENVELOPE (the slow-moving shape across frequency, formants)
# and its EXCITATION (the fast-moving fine harmonic detail, pitch).
# Pitch-shift as normal, then strip out whatever envelope the shift left
# behind and re-impose the ORIGINAL take's own envelope on top of the
# shifted excitation. The formants stay where a real voice would keep
# them; only the pitch actually moves.
#
# Analysis window used by the formant-corrected shift below. 2048 samples
# (~46ms at 44.1kHz) is large enough to resolve a low singer's harmonics
# cleanly, and the 1/4-window hop keeps the phase vocoder's own smearing
# down. Both were chosen by measuring, not by taste. On this project's own
# seven recorded takes, shifting -7/-5/-3/+3/+5/+7 semitones and comparing
# the result's formant envelope against the original at matched loudness
# (so the number describes timbre, not gain), the mean absolute envelope
# error was 12.3/10.6/9.5/5.3/4.5/5.2 dB with the old 1024-sample window
# and an averaged envelope, versus 3.5/2.6/2.6/3.5/3.3/4.1 dB with these
# parameters and a per-frame one, i.e. roughly a 2.5x reduction in how
# far the voice's own character moves, with the periodicity of the result
# unchanged (a phase vocoder's harmonics-to-noise behaviour is set by the
# shift amount, not by which envelope is painted back on afterward).
_SHIFT_N_FFT = 2048
_SHIFT_HOP = 512
# Cepstral coefficients kept when smoothing a log-spectrum down to its slow
# moving envelope. Lower = smoother (fewer, broader formants), and past a
# point too smooth to follow a real vowel's second formant; 20/24/32 all
# measured within 0.1 dB of each other, 16 was clearly worse, and the old
# value of 30 is only meaningful at the old window size.
_SHIFT_ENVELOPE_COEFFS = 24


def _per_frame_envelopes(log_magnitude: np.ndarray, n_fft: int, n_coeffs: int) -> np.ndarray:
    """
    Smooths a whole spectrogram's log-magnitudes down to just their slow-
    moving envelope shape (formants) via real-cepstrum low-quefrency
    liftering, discarding the fast-moving harmonic fine structure (pitch).

    This is the difference between the old correction and this one. The old
    one measured ONE averaged envelope for the whole take and painted it
    over the whole shifted take, which can only ever describe a steady-
    state vowel, any take where the vowel itself moves ("cold" gliding
    through its diphthong, any real word) had its formant movement flattened
    to the take's average as a side effect. Correcting frame by frame, 1:1
    with the original (pitch shifting preserves duration, so frame i of the
    shifted audio really does correspond to frame i of the original), keeps
    the formants moving exactly as they did in the recording.
    log_magnitude: shape (n_fft//2 + 1, n_frames)
    """
    cepstrum = np.fft.irfft(log_magnitude, n=n_fft, axis=0)
    lifter = np.zeros(n_fft)
    lifter[:n_coeffs] = 1.0
    lifter[-(n_coeffs - 1):] = 1.0
    return np.fft.rfft(cepstrum * lifter[:, None], n=n_fft, axis=0).real


def _formant_preserving_pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Pitch-shifts `audio` by n_steps semitones while keeping its ORIGINAL
    formants in place, instead of dragging them along with the pitch.
    Falls back to a plain (uncorrected) shift for a small movement
    correction has its own small risk of artifacts, and isn't worth
    that risk when the shift is modest enough that formant drag was
    never going to be very audible anyway, and falls back the same
    way on any numerical trouble, so a rare bad correction degrades to
    "the old, already-shipped sound" rather than silence or garbage.
    """
    audio = np.asarray(audio, dtype=np.float32)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(n_steps),
                                           n_fft=_SHIFT_N_FFT, hop_length=_SHIFT_HOP,
                                           res_type="soxr_hq")
    if abs(n_steps) < 1.0:
        return shifted.astype(np.float32)

    try:
        orig_stft = librosa.stft(audio, n_fft=_SHIFT_N_FFT, hop_length=_SHIFT_HOP)
        shifted_stft = librosa.stft(shifted, n_fft=_SHIFT_N_FFT, hop_length=_SHIFT_HOP)
        n_frames = min(orig_stft.shape[1], shifted_stft.shape[1])
        if n_frames < 2:
            return shifted.astype(np.float32)

        log_orig = np.log(np.abs(orig_stft[:, :n_frames]) + 1e-8)
        log_shift = np.log(np.abs(shifted_stft[:, :n_frames]) + 1e-8)
        corrected_log = (log_shift
                         - _per_frame_envelopes(log_shift, _SHIFT_N_FFT, _SHIFT_ENVELOPE_COEFFS)
                         + _per_frame_envelopes(log_orig, _SHIFT_N_FFT, _SHIFT_ENVELOPE_COEFFS))
        corrected_stft = np.exp(corrected_log) * np.exp(1j * np.angle(shifted_stft[:, :n_frames]))
        corrected = librosa.istft(corrected_stft, hop_length=_SHIFT_HOP, length=len(audio))
        if not np.all(np.isfinite(corrected)):
            return shifted.astype(np.float32)

        # Envelope correction changes spectral SHAPE, which can shift
        # overall energy up or down as a side effect, rescale so the
        # corrected version's loudness matches the plain shift's (RMS,
        # not peak, preserves perceived loudness without being thrown
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
    lower. Returned in Hz (un-normalized), NeuralVocableBank._tag_
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
        # vocable), filled in once every base is loaded, see
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

        relative rather than fixed thresholds: "bright"
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

    def get(self, vocable: str, target_hz: float, n_samples: int,
            hold: bool = False) -> np.ndarray | None:
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
            return self._fit_length(cached[0], n_samples, self.sample_rate, hold=hold, f0_hz=target_hz)

        # Not in cache: use fast resample and queue high-quality in background
        fast_audio = self._fast_resample_shift(base.audio, base.f0_hz, target_hz)
        with self._cache_lock:
            self._cache[key] = (fast_audio, True)   # mark as fast
        self._queue_high_quality_shift(key, base, target_hz)

        return self._fit_length(fast_audio, n_samples, self.sample_rate, hold=hold, f0_hz=target_hz)

    def get_blocking(self, vocable: str, target_hz: float, n_samples: int,
                     hold: bool = False) -> np.ndarray | None:
        """
        Same idea as get(), but for offline/batch rendering (see
        VocableSynthesizer's realtime=False mode), where there is no live
        performance to protect from stalling and therefore no reason to
        ever hand back the cheap resample-based placeholder from get().
        That placeholder ties pitch to duration (a resample-based octave
        shift plays back at half speed, doubling length), which is fine
        for a fraction of a second during a live performance before the
        real shift lands, but is completely wrong to leave baked into a
        finished render, a batch job has no real-time budget to
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
            return self._fit_length(cached[0], n_samples, self.sample_rate, hold=hold, f0_hz=target_hz)

        # Either missing or fast entry ,  compute high-quality synchronously
        try:
            n_steps = 12.0 * np.log2(target_hz / base.f0_hz)
            shifted = _formant_preserving_pitch_shift(base.audio, self.sample_rate, n_steps)
        except Exception as e:
            logger.error(f"Vocable pitch-shift failed for {key}: {e}")
            return None

        with self._cache_lock:
            self._cache[key] = (shifted, False)   # mark as high-quality
        return self._fit_length(shifted, n_samples, self.sample_rate, hold=hold, f0_hz=target_hz)

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
        based octave-down shift plays back at half speed), that's fine
        for the pitch, but wrong for timing, and audible as the whole
        note dragging or rushing for the brief window before the
        high-quality version lands. So this does the resample-based shift
        for pitch, same as before, then resamples the RESULT a second
        time back to the original sample count, that restores the
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
        "start", then the rest of the segment can just follow it
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
    def _phase_aligned_loop_length(audio: np.ndarray, approx_len: int, search: int) -> int:
        """
        Pick the loop length whose END actually continues the START.

        Quantizing the loop to a whole number of periods is the right idea
        and not enough on its own: the material being looped has already
        been through the formant-preserving phase-vocoder shift, so its real
        period is close to the requested one but not exactly it, and a slice
        that ends even a few samples out of phase leaves a discontinuity at
        every repeat. (Measured with the naive quantization: amplitude
        modulation at 2-5 Hz 6 dB deeper than the word-repeat it replaced,
        because the seam came round twice a second.) So the length is chosen
        by MATCHING the waveform instead: the head is compared against every
        candidate end point within a period and a half, and the one that
        lines up best is used. That makes the join carry on the same
        oscillation rather than cancel it.
        """
        n = len(audio)
        win = int(min(256, max(32, n // 8)))
        lo = max(win + 1, approx_len - search)
        hi = min(n - win - 1, approx_len + search)
        if hi <= lo:
            return max(win + 1, min(approx_len, n - 1))
        head = audio[:win].astype(np.float64)
        head = head - head.mean()
        head_norm = float(np.linalg.norm(head)) + 1e-12
        best_len, best_score = approx_len, -2.0
        for cand in range(lo, hi + 1):
            tail = audio[cand:cand + win].astype(np.float64)
            tail = tail - tail.mean()
            score = float(np.dot(head, tail)) / (head_norm * (float(np.linalg.norm(tail)) + 1e-12))
            if score > best_score:
                best_score, best_len = score, cand
        return best_len

    @staticmethod
    def _fit_length(audio: np.ndarray, n_samples: int, sample_rate: int = 44100,
                    hold: bool = False, f0_hz: float | None = None) -> np.ndarray:
        """
        Extends or trims `audio` to exactly n_samples, WITHOUT ever
        fragmenting the recording's own internal structure.

        This used to split the take into attack / sustain / release and
        loop just the middle "sustain" slice to fill extra time, exactly
        right for a single steady-state vowel (its middle really is one
        stable, loop-safe sound), but actively wrong for anything with
        real word structure. The vocables actually configured right now
        (vocable_set: Lalanana, ohoho; vocable_rare_set includes whole
        phrases like "welcome to reality") are not steady-state vowels,
        they're multi-syllable recordings with real onsets and
        consonants throughout. Looping an arbitrary slice out of the
        middle of "Lalanana" doesn't hold "Lalanana", it chops it at
        whatever syllable boundary the slice happened to land on and
        repeats THAT fragment, a real recorded take of "La-la-na-na"
        could turn into a mechanically repeating "-lana-lana-lana-",
        which is a different, meaningless sound, not a held version of
        the word. That's not a subtle quality issue, it's the actual
        reason the sung output could stop being comprehensible as a word
        at all.

        Now the WHOLE recording is always kept intact, at every note
        length:
          - A note shorter than the take's natural duration speeds the
            ENTIRE recording up (phase-vocoder time-stretch, pitch
            unaffected), rather than chopping the front or back off
            you hear the whole word delivered faster, not half a word.
          - A note longer than the take slows the ENTIRE recording down,
            up to MAX_STRETCH_RATIO. Real syllables just take longer to
            get through, the way a singer actually elongates a lyric
            over a long note.
          - Past MAX_STRETCH_RATIO, rather than stretching one utterance
            into unnaturally slow mush, the whole word is time-stretched
            once up to that ceiling and then repeated back-to-back
            like a singer actually repeating a lyric across a very long
            held phrase, with a real crossfade at the repeat boundary
            (_make_loop_safe, reused here at the level of a WHOLE
            repeated word rather than an arbitrary internal fragment).
        """
        n_orig = len(audio)
        if n_orig == 0:
            return np.zeros(n_samples, dtype=np.float32)
        if n_orig == n_samples:
            return audio.astype(np.float32)

        if hold:
            # A HELD note is not a repeated word, it is ONE sustained vowel
            # held for as long as the mode asks. Everything below this is
            # about keeping a WORD intact, and for a drone that
            # is exactly backwards: a drone_support note on a 35 second track
            # needs about 35 seconds of audio from a take under two seconds
            # long, so it took the whole-word-repeat branch and said the word
            # again every time the loop came round. Measured on the real
            # render: the level fell about 30 dB for roughly a second every
            # 4.5 seconds, in step with the repeat, which is what a listener
            # hears as the pad dropping out and starting again rather than as
            # a drone. Holding instead loops the take's own STEADY MIDDLE
            # one vowel, with no onset, no consonant and no ending inside it
            # so the sound never restarts.
            core = audio
            if n_orig > int(0.3 * sample_rate):
                core = audio[int(0.30 * n_orig):int(0.80 * n_orig)]
            if len(core) == 0:
                core = audio
            # The loop length is quantized to a WHOLE NUMBER OF PERIODS of the
            # pitch being held, which is the difference between a held tone and
            # a tremolo. A voiced sound is periodic at its own f0, so a slice
            # that is not a whole number of periods ends mid-cycle: played back
            # to back, the end of one copy and the start of the next are
            # out of phase, and where they disagree they partially cancel.
            # That cancellation comes round once per loop, which on the first
            # version of this (an arbitrary 0.45 s slice, ~2 loops a second)
            # measured as a deep level dip in 30 of 35 one-second windows
            # audibly worse than the word-repeat it replaced. Quantizing
            # removes the seam instead of fading over it.
            period = (sample_rate / float(f0_hz)) if (f0_hz and f0_hz > 0) else 0.0
            if period >= 2.0:
                if len(core) >= n_samples:
                    m = max(1, int(round(n_samples / period)))
                    seg = min(int(round(m * period)), len(core))
                    start = (len(core) - seg) // 2
                    return core[start:start + seg].astype(np.float32)
                m = max(1, int(len(core) / period))
                target_len = int(round(m * period))
                target_len = NeuralVocableBank._phase_aligned_loop_length(
                    core, target_len, int(round(period * 1.5))
                )
                unit = core[:target_len]
                if len(unit) < 8:
                    unit = core
            else:
                if len(core) >= n_samples:
                    start = (len(core) - n_samples) // 2
                    return core[start:start + n_samples].astype(np.float32)
                cf = min(len(core) // 6, int(0.12 * sample_rate))
                unit = NeuralVocableBank._make_loop_safe(core, cf) if cf >= 8 else core
                if len(unit) == 0:
                    unit = core
            reps = max(1, int(np.ceil(n_samples / max(1, len(unit)))))
            out = np.tile(unit, reps)[:n_samples]
            if len(out) < n_samples:
                out = np.pad(out, (0, n_samples - len(out)))
            return out.astype(np.float32)

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
            # doesn't compress consonants into an unrecognizable blur
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
        # every repeat, never a mid-word splice.
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
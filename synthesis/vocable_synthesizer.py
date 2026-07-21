import numpy as np
import os
from pathlib import Path
from loguru import logger
from config.config_loader import get_config
from synthesis.harmony_engine import HarmonyDecision
from synthesis.accompaniment_modes import AccompanimentMode
import librosa
from scipy.signal import lfilter, butter, iirnotch, iirpeak


# Generates a numpy audio array for the robot to sing based on the decision from HarmonyDecision
class VocableSynthesizer:
    # Vowel formant frequencies (F1, F2, F3 in Hz) for standard vocables, plus
    # a bandwidth per formant. These are approximate sung-voice formant
    # centers (roughly midway between typical male/female values) for each
    # vocable's "bright" and "dark" endpoint, blended by vowel_color.
    #
    # Real formants for sung vowels sit close together — the previous table
    # used gaps so wide (e.g. F1/F2 nearly 400-1000Hz apart with harmonics
    # weighted by raw Gaussian distance) that the resulting spectrum had no
    # continuous formant "resonance," just isolated spikes, which is a big
    # part of why the old engine sounded synthetic/alien rather than vocal.
    FORMANTS = {
        # (F1, F2, F3) at the bright end, then the dark end
        "aah": {"bright": (730, 1400, 2600), "dark": (680, 1100, 2450)},   # "ah" as in father
        "ooo": {"bright": (400, 900,  2300), "dark": (320, 700,  2200)},   # "oo" as in boot
        "mmm": {"bright": (280, 950,  2100), "dark": (250, 850,  2000)},   # closed/nasal hum
        "hey": {"bright": (600, 2100, 2700), "dark": (500, 1750, 2600)},   # "eh" as in bed
    }
    # Relative amplitude weight and bandwidth (Q) for each formant.
    FORMANT_WEIGHTS = (1.0, 0.55, 0.22)
    FORMANT_Q = (10.0, 12.0, 14.0)

    def __init__(self):
        cfg = get_config()
        self.engine = cfg["synthesis"]["engine"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.crossfade_ms = cfg["synthesis"]["crossfade_ms"]
        self.ddsp_model_path = cfg["synthesis"]["ddsp_model_path"]
        self.vocable_set = cfg["synthesis"]["vocable_set"]
        self.volume = cfg["output"]["volume"]
        self.timbral_detune_cents = cfg["synthesis"].get("timbral_detune_cents", 6)
        self.breathiness = cfg["synthesis"].get("breathiness", 0.045)
        self.vibrato_rate_hz = cfg["synthesis"].get("vibrato_rate_hz", 5.3)
        self.vibrato_depth_cents = cfg["synthesis"].get("vibrato_depth_cents", 12)

        self._ddsp_model = None
        self._wavetable_samples: dict[str, np.ndarray] = {}

        self._crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        self._prev_audio: np.ndarray | None = None  # for crossfading
        self._vibrato_phase: float = 0.0             # keeps vibrato continuous across notes

        self._init_engine()

    # Routes to the right loader based on which engine is configured
    def _init_engine(self):
        if self.engine == "ddsp":
            self._load_ddsp_model()
        elif self.engine == "wavetable":
            self._load_wavetable_samples()
        logger.info(f"Vocable synthesizer engine: {self.engine}")

    # Loads pre-recorded WAV files from synthesis/samples/
    def _load_wavetable_samples(self):
        try:
            samples_dir = Path(__file__).parent / "samples"
            if not samples_dir.exists():
                logger.warning(
                    f"No samples directory at {samples_dir} — "
                    "falling back to sinusoidal synthesis. "
                    "Add WAV files named aah.wav, ooo.wav, mmm.wav, hey.wav "
                    "to synthesis/samples/ for wavetable mode."
                )
                self.engine = "sinusoidal"
                return

            for vocable in self.vocable_set:
                path = samples_dir / f"{vocable}.wav"
                if path.exists():
                    audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
                    self._wavetable_samples[vocable] = audio
                    logger.info(f"Loaded sample: {vocable}.wav ({len(audio)} samples)")
                else:
                    logger.warning(f"Sample not found: {path}")

            if not self._wavetable_samples:
                logger.warning("No samples loaded — falling back to sinusoidal")
                self.engine = "sinusoidal"

        except Exception as e:
            logger.error(f"Wavetable load error: {e} — falling back to sinusoidal")
            self.engine = "sinusoidal"

    # Generate audio for the given HarmonyDecision
    def synthesize(self, decision):
        if decision.action == "rest" or decision.target_hz <= 0:
            n_samples = int(0.1 * self.sample_rate)
            return np.zeros(n_samples, dtype=np.float32)

        n_samples = int(decision.duration_s * self.sample_rate)
        if n_samples <= 0:
            return np.zeros(1024, dtype=np.float32)

        if self.engine == "wavetable":
            audio = self._synthesize_wavetable(decision, n_samples)
        else:
            audio = self._synthesize_sinusoidal(decision, n_samples)

        # Apply amplitude envelope (attack + release to avoid clicks)
        audio = self._apply_envelope(audio)

        # Apply Cree phoneme timbre shaping
        audio = self._apply_phoneme_shaping(audio, decision)

        # Volume
        audio = audio * self.volume

        # Crossfade with previous output for smooth transitions
        audio = self._crossfade(audio)

        self._prev_audio = audio
        return audio.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Source-filter vowel synthesis
    # ------------------------------------------------------------------ #
    #
    # This models a singing voice the way a real one works: a bright,
    # buzzy glottal source rich in harmonics (the vocal folds), shaped by a
    # handful of resonant formant filters (the vocal tract) that boost
    # specific frequency bands depending on vowel shape. The previous
    # version instead weighted each harmonic individually by its Gaussian
    # distance from F1/F2 — mathematically similar in spirit, but it
    # produced a sparse, comb-like spectrum with no continuous resonance,
    # which reads to the ear as synthetic/metallic rather than vocal.
    # Adding a touch of vibrato and breath noise closes the rest of the gap.
    def _synthesize_sinusoidal(self, decision, n_samples):
        t = np.arange(n_samples) / self.sample_rate

        f0 = decision.target_hz
        if decision.mode == AccompanimentMode.TIMBRAL:
            f0 *= 2 ** (self.timbral_detune_cents / 1200.0)

        # Vibrato: subtle pitch wobble, more present on longer held notes so
        # short call-and-response style hits stay clean and percussive.
        vibrato_amount = np.clip(decision.duration_s / 0.6, 0.0, 1.0)
        vib_rate = self.vibrato_rate_hz
        phase_inc = 2 * np.pi * vib_rate / self.sample_rate
        vib_phase = self._vibrato_phase + np.arange(n_samples) * phase_inc
        self._vibrato_phase = float((vib_phase[-1] + phase_inc) % (2 * np.pi)) if n_samples else self._vibrato_phase
        vibrato_cents = self.vibrato_depth_cents * vibrato_amount * np.sin(vib_phase)
        instantaneous_f0 = f0 * (2 ** (vibrato_cents / 1200.0))
        phase = 2 * np.pi * np.cumsum(instantaneous_f0) / self.sample_rate

        # Glottal-style harmonic source: natural voices roll off roughly
        # -12 dB/octave, so 1/k^1.6 reads as noticeably more "breath and
        # body" than a flat 1/k saw while still keeping plenty of brightness
        # for the formant filters to shape.
        max_harmonic = int(self.sample_rate / 2 / f0)
        source = np.zeros(n_samples, dtype=np.float64)
        for k in range(1, min(max_harmonic + 1, 40)):
            amp = 1.0 / (k ** 1.6)
            source += amp * np.sin(k * phase)

        peak = np.max(np.abs(source))
        if peak > 0:
            source /= peak

        # Formant filtering: three resonant bandpass filters, blended
        # between each vocable's bright/dark formant endpoints by vowel_color.
        vc = decision.vowel_color
        table = self.FORMANTS.get(decision.vocable, self.FORMANTS["aah"])
        bright, dark = table["bright"], table["dark"]
        formant_freqs = [b * (1 - vc) + d * vc for b, d in zip(bright, dark)]

        audio = np.zeros(n_samples, dtype=np.float64)
        nyquist = self.sample_rate / 2.0
        for freq, weight, q in zip(formant_freqs, self.FORMANT_WEIGHTS, self.FORMANT_Q):
            freq = float(np.clip(freq, 40.0, nyquist * 0.98))
            resonated = self._bandpass(source, freq, q)
            brightness_boost = 1.0 + decision.brightness * 0.4
            audio += resonated * weight * brightness_boost

        # A little of the raw (unfiltered) source underneath gives the tone
        # a fundamental "body" that pure formant resonances can lack.
        audio += source * 0.12

        # Breath noise: filtered through the same formants at low level so
        # it reads as air moving through a vocal tract, not static.
        if self.breathiness > 0:
            noise = np.random.default_rng().normal(0, 1, n_samples)
            breath = self._bandpass(noise, formant_freqs[0], 3.0) * 0.6
            breath += self._bandpass(noise, formant_freqs[1], 4.0) * 0.4
            audio += breath * self.breathiness

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak

        return audio.astype(np.float32)

    def _bandpass(self, signal: np.ndarray, center_hz: float, q: float) -> np.ndarray:
        nyquist = self.sample_rate / 2.0
        bandwidth = max(center_hz / q, 10.0)
        low = max((center_hz - bandwidth / 2) / nyquist, 1e-4)
        high = min((center_hz + bandwidth / 2) / nyquist, 0.999)
        if low >= high:
            return signal
        try:
            b, a = butter(2, [low, high], btype="band")
            return lfilter(b, a, signal)
        except Exception as e:
            logger.error(f"Formant filter error at {center_hz:.0f}Hz: {e}")
            return signal

    # Takes a pre-recorded human voice sample and shifts its pitch to match the target frequency
    def _synthesize_wavetable(self, decision, n_samples):
        try:
            sample = self._wavetable_samples.get(
                decision.vocable,
                next(iter(self._wavetable_samples.values()))
            )

            # Estimate original pitch of the sample
            f0_orig, _, _ = librosa.pyin(
                sample, fmin=60, fmax=800, sr=self.sample_rate
            )
            f0_orig_mean = float(np.nanmedian(f0_orig)) if f0_orig is not None else 220.0

            # Semitone shift needed
            n_steps = 12 * np.log2(decision.target_hz / f0_orig_mean)

            # Phase vocoder pitch shift
            shifted = librosa.effects.pitch_shift(
                sample, sr=self.sample_rate, n_steps=float(n_steps)
            )

            # Loop or trim to n_samples
            if len(shifted) < n_samples:
                repeats = int(np.ceil(n_samples / len(shifted)))
                shifted = np.tile(shifted, repeats)
            audio = shifted[:n_samples]

            return audio.astype(np.float32)

        except Exception as e:
            logger.error(f"Wavetable synthesis error: {e} — using sinusoidal fallback")
            return self._synthesize_sinusoidal(decision, n_samples)

    # Prevents clicks at note boundaries by fading in and out
    def _apply_envelope(self, audio):
        attack_samples = min(int(0.01 * self.sample_rate), len(audio) // 4)
        release_samples = min(int(0.03 * self.sample_rate), len(audio) // 4)

        envelope = np.ones(len(audio))
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

        return audio * envelope

    # Applies filters based on the Cree phoneme profile to shape the frequency content.
    # Uses standard, numerically-stable biquad designs (shelf + notch) instead
    # of hand-rolled coefficients, which previously risked instability/artifacts
    # at higher brightness/nasality values.
    def _apply_phoneme_shaping(self, audio, decision):
        try:
            # Spectral tilt based on brightness: a proper high-shelf boost
            # rather than a 2-tap differencer (which cuts everything below
            # its corner, not just adds highs, and was a large contributor
            # to the thin/harsh "alien" quality at high brightness values).
            if decision.brightness > 0.55:
                gain_db = (decision.brightness - 0.55) * 10.0  # up to ~4.5 dB
                audio = self._high_shelf(audio, corner_hz=3200.0, gain_db=gain_db)

            # Nasal anti-formant: notch around 1kHz when nasality is high,
            # via scipy's iirnotch (guaranteed-stable) instead of a manual
            # biquad with an unusual, unverified pole formula.
            if decision.nasality > 0.3:
                notch_freq = 1000.0
                q = 4.0
                depth = decision.nasality * 0.6
                b_notch, a_notch = iirnotch(notch_freq, q, fs=self.sample_rate)
                notched = lfilter(b_notch, a_notch, audio)
                audio = audio * (1 - depth) + notched * depth

                # Nasal formant boost around 250Hz gives the notch somewhere
                # to "put" the nasal quality instead of just sounding hollow.
                b_peak, a_peak = iirpeak(250.0, 2.0, fs=self.sample_rate)
                nasal_boost = lfilter(b_peak, a_peak, audio)
                audio = audio * (1 - depth * 0.3) + nasal_boost * (depth * 0.3)

            # Normalize after filtering
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio /= peak

            return audio

        except Exception as e:
            logger.error(f"Phoneme shaping error: {e}")
            return audio

    # RBJ-cookbook high-shelf biquad — boosts everything above corner_hz by
    # roughly gain_db, smoothly, without touching the low end.
    def _high_shelf(self, audio: np.ndarray, corner_hz: float, gain_db: float) -> np.ndarray:
        if gain_db <= 0:
            return audio
        try:
            A = 10 ** (gain_db / 40.0)
            w0 = 2 * np.pi * corner_hz / self.sample_rate
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.9 - 1) + 2)
            cos_w0 = np.cos(w0)
            sqrt_A = np.sqrt(A)

            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

            b = np.array([b0, b1, b2]) / a0
            a = np.array([a0, a1, a2]) / a0
            return lfilter(b, a, audio)
        except Exception as e:
            logger.error(f"High-shelf filter error: {e}")
            return audio

    # Blends the start of the new note with the tail of the previous one over
    def _crossfade(self, new_audio: np.ndarray) -> np.ndarray:
        if self._prev_audio is None or len(new_audio) < self._crossfade_samples:
            return new_audio

        cf = self._crossfade_samples
        fade_in = np.linspace(0, 1, cf)
        fade_out = np.linspace(1, 0, cf)

        prev_tail = self._prev_audio[-cf:] if len(self._prev_audio) >= cf else self._prev_audio

        new_audio = new_audio.copy()
        overlap_len = min(cf, len(prev_tail), len(new_audio))
        new_audio[:overlap_len] = (
            new_audio[:overlap_len] * fade_in[:overlap_len]
            + prev_tail[:overlap_len] * fade_out[:overlap_len]
        )
        return new_audio
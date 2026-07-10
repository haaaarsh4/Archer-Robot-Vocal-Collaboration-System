import numpy as np
import os
from pathlib import Path
from loguru import logger
from config.config_loader import get_config
from synthesis.harmony_engine import HarmonyDecision
from synthesis.accompaniment_modes import AccompanimentMode
import librosa
from scipy.signal import lfilter


# Generates a numpy audio array for the robot to sing based on the decision from HarmonyDecision 
class VocableSynthesizer:
    # Vowel formant frequencies (F1, F2 in Hz) for standard vocables.
    FORMANTS = {
        "aah": [(800,  1200), (1000, 1400)],  # (F1, F2) low-high endpoints
        "ooo": [(300,  700),  (400,  800)],
        "mmm": [(250,  900),  (300,  950)],
        "hey": [(600,  1800), (700,  1900)],
    }

    def __init__(self):
        cfg = get_config()
        self.engine = cfg["synthesis"]["engine"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.crossfade_ms = cfg["synthesis"]["crossfade_ms"]
        self.ddsp_model_path = cfg["synthesis"]["ddsp_model_path"]
        self.vocable_set = cfg["synthesis"]["vocable_set"]
        self.volume = cfg["output"]["volume"]
        self.timbral_detune_cents = cfg["synthesis"].get("timbral_detune_cents", 6)

        self._ddsp_model = None
        self._wavetable_samples: dict[str, np.ndarray] = {}

        self._crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        self._prev_audio: np.ndarray | None = None  # for crossfading

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

    def _synthesize_sinusoidal(self, decision, n_samples):
        t = np.linspace(0, decision.duration_s, n_samples, endpoint=False)
        audio = np.zeros(n_samples, dtype=np.float64)

        f0 = decision.target_hz

        if getattr(decision, "mode", None) == AccompanimentMode.TIMBRAL:
            f0 *= 2 ** (self.timbral_detune_cents / 1200.0)
        formant_params = self.FORMANTS.get(decision.vocable, self.FORMANTS["aah"])

        # Blend formant position based on vowel_color (0=bright, 1=dark)
        vc = decision.vowel_color
        f1 = formant_params[0][0] * (1 - vc) + formant_params[1][0] * vc
        f2 = formant_params[0][1] * (1 - vc) + formant_params[1][1] * vc

        # Add harmonics up to Nyquist, weighted by distance from formants
        max_harmonic = int(self.sample_rate / 2 / f0)
        for k in range(1, min(max_harmonic + 1, 32)):
            freq = f0 * k
            # Amplitude shaped by proximity to F1 and F2
            amp_f1 = np.exp(-((freq - f1) ** 2) / (2 * (150 ** 2)))
            amp_f2 = np.exp(-((freq - f2) ** 2) / (2 * (200 ** 2)))
            amp = (amp_f1 + amp_f2 * 0.6) / k  # roll off with harmonic number
            # Add slight brightness boost from phoneme profile
            amp *= (1.0 + decision.brightness * 0.5)
            audio += amp * np.sin(2 * np.pi * freq * t)

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio /= peak

        return audio.astype(np.float32)

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

    # Applies two filters based on the Cree phoneme profile to shape the frequency content
    def _apply_phoneme_shaping(self, audio, decision):
        try:
            # Spectral tilt based on brightness
            if decision.brightness > 0.6:
                # Boost highs: first-order high-shelf approximation
                b = np.array([1 + decision.brightness * 0.3, -0.5])
                a = np.array([1.0])
                audio = lfilter(b, a, audio)

            # Nasal anti-formant: notch around 1kHz when nasality is high
            if decision.nasality > 0.3:
                notch_freq = 1000.0
                Q = 5.0
                w0 = notch_freq / (self.sample_rate / 2)
                b_notch, a_notch = self._notch_filter(w0, Q)
                depth = decision.nasality * 0.6
                notched = lfilter(b_notch, a_notch, audio)
                audio = audio * (1 - depth) + notched * depth

            # Normalize after filtering
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio /= peak

            return audio

        except Exception as e:
            logger.error(f"Phoneme shaping error: {e}")
            return audio

    @staticmethod
    def _notch_filter(w0: float, Q: float) -> tuple:
        b0 = 1.0
        b1 = -2 * np.cos(np.pi * w0)
        b2 = 1.0
        a0 = 1.0
        a1 = -2 * np.cos(np.pi * w0) * (Q / (Q + 1))
        a2 = (Q - 1) / (Q + 1)
        return np.array([b0, b1, b2]), np.array([a0, a1, a2])

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
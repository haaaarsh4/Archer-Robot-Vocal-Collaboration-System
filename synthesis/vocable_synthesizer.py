import numpy as np
import os
from pathlib import Path
from loguru import logger
from config.config_loader import get_config
from synthesis.harmony_engine import HarmonyDecision
from synthesis.accompaniment_modes import AccompanimentMode
import librosa
from scipy.signal import lfilter, butter, iirnotch, iirpeak


class VocableSynthesizer:
    FORMANTS = {
        "aah": {"bright": (730, 1400, 2600, 3300), "dark": (680, 1100, 2450, 3200)},   # "ah" as in father
        "ooo": {"bright": (400, 900,  2300, 3000), "dark": (320, 700,  2200, 2900)},   # "oo" as in boot
        "mmm": {"bright": (280, 950,  2100, 2900), "dark": (250, 850,  2000, 2800)},   # closed/nasal hum
        "hey": {"bright": (600, 2100, 2700, 3300), "dark": (500, 1750, 2600, 3200)},   # "eh" as in bed
    }
    FORMANT_WEIGHTS    = (1.0, 0.6, 0.28, 0.14)
    FORMANT_BANDWIDTH_HZ = (80.0, 90.0, 130.0, 220.0)

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
        self._neural_bank = None   # set up in _init_engine() if synthesis.engine == "neural_wavetable"
        self._warned_unknown_vocables: set = set()

        self._crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        self._prev_audio: np.ndarray | None = None  # for crossfading

        self._max_voice_layers = 12
        self._vibrato_phases = [0.0] * self._max_voice_layers

        self._rng = np.random.default_rng()

        self._init_engine()

    # Routes to the right loader based on which engine is configured
    def _init_engine(self):
        if self.engine == "ddsp":
            self._load_ddsp_model()
        elif self.engine == "wavetable":
            self._load_wavetable_samples()
        elif self.engine == "neural_wavetable":
            self._load_neural_bank()
        logger.info(f"Vocable synthesizer engine: {self.engine}")

    # Loads pre-rendered, already RVC-converted vocable takes (see
    # synthesis/build_vocable_bank.py) and gets them ready for fast
    # runtime pitch shifting. No neural inference happens here or at
    # play time, the conversion already happened offline.
    def _load_neural_bank(self):
        try:
            from synthesis.neural_vocable_bank import NeuralVocableBank
        except ImportError as e:
            logger.error(f"neural_wavetable engine selected but neural_vocable_bank.py "
                          f"couldn't be imported ({e}) — falling back to sinusoidal.")
            self.engine = "sinusoidal"
            return

        samples_dir = Path(__file__).parent / "samples" / "neural"
        self._neural_bank = NeuralVocableBank(samples_dir, self.sample_rate)

        if not self._neural_bank.available:
            logger.warning(
                "neural_wavetable engine selected but no converted samples were "
                f"found at {samples_dir}. Run synthesis/build_vocable_bank.py first "
                "(see its docstring). Falling back to sinusoidal for now."
            )
            self.engine = "sinusoidal"
            self._neural_bank = None

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

        num_voices = max(1, int(getattr(decision, "num_voices", 1)))

        if num_voices <= 1:
            audio = self._render_voice(decision, n_samples, voice_index=0,
                                        f0_hz=decision.target_hz, formant_scale=1.0)
        else:
            audio = self._render_ensemble(decision, n_samples, num_voices)

        # Apply amplitude envelope (attack + release to avoid clicks)
        audio = self._apply_envelope(audio)

        # Apply Cree phoneme timbre shaping
        audio = self._apply_phoneme_shaping(audio, decision)

        reverb_amount = float(getattr(decision, "reverb_amount", 0.08))
        if reverb_amount > 0:
            audio = self._apply_reverb(audio, reverb_amount)

        # Volume
        audio = audio * self.volume

        audio = np.tanh(audio * 1.15) / np.tanh(1.15)

        # Crossfade with previous output for smooth transitions
        audio = self._crossfade(audio)

        self._prev_audio = audio
        return audio.astype(np.float32)

    def _render_ensemble(self, decision, n_samples: int, num_voices: int) -> np.ndarray:
        detune_spread = float(getattr(decision, "detune_spread_cents", 10.0))
        jitter_ms = float(getattr(decision, "timing_jitter_ms", 15.0))
        formant_spread = float(getattr(decision, "formant_spread", 0.1))
        max_jitter_samples = int(jitter_ms * self.sample_rate / 1000)

        if num_voices == 1:
            detune_offsets = [0.0]
        else:
            base = np.linspace(-detune_spread, detune_spread, num_voices)
            detune_offsets = base + self._rng.normal(0, detune_spread * 0.15, num_voices)
            detune_offsets[0] = 0.0  # keep one voice dead-center as the "lead"

        mix = np.zeros(n_samples, dtype=np.float64)
        for i in range(num_voices):
            cents = float(detune_offsets[i])
            f0 = decision.target_hz * (2 ** (cents / 1200.0))
            f_scale = 1.0 + (self._rng.uniform(-1, 1) * formant_spread if i > 0 else 0.0)

            voice = self._render_voice(decision, n_samples, voice_index=i,
                                        f0_hz=f0, formant_scale=f_scale)

            if max_jitter_samples > 0 and i > 0:
                shift = int(self._rng.integers(-max_jitter_samples, max_jitter_samples + 1))
                voice = self._shift_samples(voice, shift)

            gain = 1.0 / (1.0 + 0.12 * abs(cents) / max(detune_spread, 1.0))
            mix += voice * gain

        peak = np.max(np.abs(mix))
        if peak > 0:
            target_peak = min(0.98, 0.82 + 0.025 * num_voices)
            mix = mix / peak * target_peak

        return mix.astype(np.float64)

    def _shift_samples(self, audio: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return audio
        out = np.zeros_like(audio)
        if shift > 0:
            out[shift:] = audio[:len(audio) - shift]
        else:
            out[:shift] = audio[-shift:]
        return out

    def _render_voice(self, decision, n_samples: int, voice_index: int,
                       f0_hz: float, formant_scale: float) -> np.ndarray:
        if self.engine == "neural_wavetable" and self._neural_bank is not None:
            audio = self._neural_bank.get(decision.vocable, f0_hz, n_samples)
            if audio is not None:
                return audio
            # Bank has nothing loaded for this vocable at all (not just a
            # cache miss, get() already handles those). Fall through just
            # this once rather than going silent.
        if self.engine == "wavetable":
            return self._synthesize_wavetable(decision, n_samples)
        return self._synthesize_sinusoidal(decision, n_samples, voice_index=voice_index,
                                            f0_override=f0_hz, formant_scale=formant_scale)

    def _synthesize_sinusoidal(self, decision, n_samples, voice_index: int = 0,
                                f0_override: float | None = None, formant_scale: float = 1.0):
        t = np.arange(n_samples) / self.sample_rate

        f0 = f0_override if f0_override is not None else decision.target_hz
        if decision.mode == AccompanimentMode.TIMBRAL and f0_override is None:
            f0 *= 2 ** (self.timbral_detune_cents / 1200.0)

        is_humming = getattr(decision, "mode", None) == AccompanimentMode.HUM
        vibrato_ceiling = 0.5 if is_humming else 1.0

        slot = voice_index % self._max_voice_layers
        rate_offset = 0.0 if voice_index == 0 else (voice_index * 0.13) % 0.6 - 0.3
        vib_rate = max(0.5, self.vibrato_rate_hz + rate_offset)

        vibrato_amount = np.clip(decision.duration_s / 0.6, 0.0, 1.0) * vibrato_ceiling
        phase_inc = 2 * np.pi * vib_rate / self.sample_rate
        vib_phase = self._vibrato_phases[slot] + np.arange(n_samples) * phase_inc
        self._vibrato_phases[slot] = float((vib_phase[-1] + phase_inc) % (2 * np.pi)) if n_samples else self._vibrato_phases[slot]
        vibrato_cents = self.vibrato_depth_cents * vibrato_amount * np.sin(vib_phase)

        jitter = self._smoothed_noise(n_samples, std=0.0035, cutoff_hz=12.0)
        instantaneous_f0 = f0 * (2 ** (vibrato_cents / 1200.0)) * (1.0 + jitter)
        phase = 2 * np.pi * np.cumsum(instantaneous_f0) / self.sample_rate

        max_harmonic = int(self.sample_rate / 2 / f0)
        source = np.zeros(n_samples, dtype=np.float64)
        for k in range(1, min(max_harmonic + 1, 32)):
            amp = 1.0 / (k ** 1.75)
            source += amp * np.sin(k * phase)

        peak = np.max(np.abs(source))
        if peak > 0:
            source /= peak

        vocable = "mmm" if is_humming else decision.vocable
        vc = decision.vowel_color
        if vocable not in self.FORMANTS and vocable != "mmm" and vocable not in self._warned_unknown_vocables:
            # Only the original four vowels (aah/ooo/mmm/hey) have a real
            # formant profile. Any other vocable, e.g. anything from a
            # neural_wavetable bank that isn't currently loaded, silently
            # became "aah" here before with zero trace of why. Logging it
            # once per unique vocable (not per note -- this can otherwise
            # fire dozens of times a second) turns a confusing wrong sound
            # into an obvious, explainable one -- if you see this while
            # neural_wavetable is meant to be active, its bank likely
            # isn't actually loaded (check for "neural_wavetable engine
            # selected but no converted samples were found" in the log).
            logger.warning(f"Sinusoidal engine has no formant profile for vocable '{vocable}' "
                            "-- substituting 'aah'. This is expected only when the configured "
                            "engine has fallen back to sinusoidal.")
            self._warned_unknown_vocables.add(vocable)
        table = self.FORMANTS.get(vocable, self.FORMANTS["aah"])
        bright, dark = table["bright"], table["dark"]
        formant_freqs = [(b * (1 - vc) + d * vc) * formant_scale for b, d in zip(bright, dark)]

        audio = np.zeros(n_samples, dtype=np.float64)
        nyquist = self.sample_rate / 2.0
        brightness = decision.brightness * (0.5 if is_humming else 1.0)
        for freq, weight, bw in zip(formant_freqs, self.FORMANT_WEIGHTS, self.FORMANT_BANDWIDTH_HZ):
            freq = float(np.clip(freq, 40.0, nyquist * 0.98))
            resonated = self._bandpass(source, freq, bw)
            brightness_boost = 1.0 + brightness * 0.4
            audio += resonated * weight * brightness_boost

        audio += source * 0.12

        breathiness = self.breathiness * (0.2 if is_humming else 1.0)
        if breathiness > 0:
            noise = self._rng.normal(0, 1, n_samples)
            breath = self._bandpass(noise, formant_freqs[0], 220.0) * 0.6
            breath += self._bandpass(noise, formant_freqs[1], 260.0) * 0.4
            audio += breath * breathiness

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.92

        shimmer = self._smoothed_noise(n_samples, std=0.035, cutoff_hz=9.0)
        audio = audio * (1.0 + shimmer)

        return audio.astype(np.float32)

    def _smoothed_noise(self, n_samples: int, std: float, cutoff_hz: float) -> np.ndarray:
        if n_samples < 4:
            return np.zeros(n_samples)
        try:
            noise = self._rng.normal(0, 1, n_samples)
            nyquist = self.sample_rate / 2.0
            wn = min(cutoff_hz / nyquist, 0.99)
            b, a = butter(2, wn, btype="low")
            smoothed = lfilter(b, a, noise)
            rms = np.sqrt(np.mean(smoothed ** 2))
            if rms > 0:
                smoothed = smoothed / rms * std
            return smoothed
        except Exception:
            return np.zeros(n_samples)

    def _bandpass(self, signal: np.ndarray, center_hz: float, bandwidth_hz: float) -> np.ndarray:
        nyquist = self.sample_rate / 2.0
        bandwidth_hz = max(bandwidth_hz, 10.0)
        low = max((center_hz - bandwidth_hz / 2) / nyquist, 1e-4)
        high = min((center_hz + bandwidth_hz / 2) / nyquist, 0.999)
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

    def _apply_phoneme_shaping(self, audio, decision):
        try:
            if decision.brightness > 0.55:
                gain_db = (decision.brightness - 0.55) * 10.0  # up to ~4.5 dB
                audio = self._high_shelf(audio, corner_hz=3200.0, gain_db=gain_db)

            if decision.nasality > 0.3:
                notch_freq = 1000.0
                q = 4.0
                depth = decision.nasality * 0.6
                b_notch, a_notch = iirnotch(notch_freq, q, fs=self.sample_rate)
                notched = lfilter(b_notch, a_notch, audio)
                audio = audio * (1 - depth) + notched * depth

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

    _COMB_DELAYS_MS   = (29.7, 37.1, 41.4, 43.7)
    _COMB_FEEDBACK    = (0.805, 0.827, 0.783, 0.764)
    _ALLPASS_DELAYS_MS = (5.0, 1.7)
    _ALLPASS_FEEDBACK  = 0.7

    def _comb_filter(self, signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
        if delay_samples < 1 or delay_samples >= len(signal):
            return signal
        a = np.zeros(delay_samples + 1)
        a[0] = 1.0
        a[delay_samples] = -feedback
        return lfilter([1.0], a, signal)

    def _allpass_filter(self, signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
        if delay_samples < 1 or delay_samples >= len(signal):
            return signal
        b = np.zeros(delay_samples + 1)
        b[0] = -feedback
        b[delay_samples] = 1.0
        a = np.zeros(delay_samples + 1)
        a[0] = 1.0
        a[delay_samples] = -feedback
        return lfilter(b, a, signal)

    def _apply_reverb(self, audio: np.ndarray, wet_amount: float) -> np.ndarray:
        if wet_amount <= 0 or len(audio) < 64:
            return audio
        try:
            wet = np.zeros_like(audio, dtype=np.float64)
            for delay_ms, fb in zip(self._COMB_DELAYS_MS, self._COMB_FEEDBACK):
                d = int(delay_ms * self.sample_rate / 1000)
                wet += self._comb_filter(audio, d, fb)
            wet /= len(self._COMB_DELAYS_MS)

            for delay_ms in self._ALLPASS_DELAYS_MS:
                d = int(delay_ms * self.sample_rate / 1000)
                wet = self._allpass_filter(wet, d, self._ALLPASS_FEEDBACK)

            peak = np.max(np.abs(wet))
            if peak > 0:
                wet /= peak

            wet_amount = float(np.clip(wet_amount, 0.0, 1.0))
            return audio * (1 - wet_amount) + wet * wet_amount
        except Exception as e:
            logger.error(f"Reverb error: {e} — returning dry signal")
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
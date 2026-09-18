import numpy as np
import os
from pathlib import Path
from loguru import logger
from config.config_loader import get_config
from synthesis.harmony_engine import HarmonyDecision
from synthesis.accompaniment_modes import AccompanimentMode
import librosa
from scipy.signal import lfilter, butter, iirnotch, iirpeak


def apply_pitch_curve_follow(audio: np.ndarray, sample_rate: int, base_hz: float,
                              pitch_curve_hz: np.ndarray, frame_hop_s: float,
                              max_cents: float = 250.0) -> np.ndarray:
    """
    Nudges an already-rendered, single-pitch buffer to continuously
    track a real performance's own pitch curve across the length of one
    held note, instead of staying frozen at whatever pitch was detected
    the instant the note first triggered.

    Why this exists: a "run" of frames belonging to one sustained note
    only ever gets ONE synthesize() call, at the pitch measured on the
    run's very first frame (see _render_track_offline in server.py).
    Everything measured on every later frame of that same note -- a
    singer's natural vibrato, a slow bend into the next phrase, just the
    ordinary micro-wobble of a real human voice holding a note -- was
    being thrown away entirely. The note was accurate at the instant it
    started and then went perfectly, robotically flat for its whole
    duration, however long that was. That mismatch between "the exact
    pitch the moment it happened" and "a static pitch held afterward" is
    a large part of why a held note can sound like it doesn't actually
    track the real performance.

    The fix is a variable-rate resample: reading faster through the
    already-rendered buffer raises its momentary pitch, reading slower
    lowers it (same principle neural_vocable_bank.py's own
    _fast_resample_shift already uses for a full shift, just applied
    continuously here instead of once). `pitch_curve_hz` is one f0
    reading per source frame across the run; it gets interpolated up to
    per-sample resolution, turned into a per-sample instantaneous
    speed ratio against `base_hz` (the pitch the buffer was actually
    rendered at), integrated into a warped read position, and the
    buffer is re-read along that warped position.

    `max_cents` bounds how far this is allowed to pull -- it's meant for
    the natural wobble within a note (typically well under a semitone),
    not for correcting a genuinely wrong onset pitch, which is what note
    re-triggering already handles. A big, sustained deviation gets
    clamped rather than dragging the buffer arbitrarily far, since sign-
    ificant pitch movement should already have re-triggered a new note
    (see HarmonyEngine.NOTE_CHANGE_THRESHOLD_CENTS) rather than showing
    up here.
    """
    n = len(audio)
    if n == 0 or pitch_curve_hz.size == 0 or base_hz <= 0:
        return audio
    if pitch_curve_hz.size == 1:
        return audio  # nothing to follow -- the note never moved

    frame_times = (np.arange(len(pitch_curve_hz)) + 0.5) * frame_hop_s
    sample_times = np.arange(n) / float(sample_rate)
    curve = np.interp(sample_times, frame_times, pitch_curve_hz,
                       left=pitch_curve_hz[0], right=pitch_curve_hz[-1])

    cents_offset = 1200.0 * np.log2(np.clip(curve, 1e-6, None) / base_hz)
    cents_offset = np.clip(cents_offset, -max_cents, max_cents)
    ratio = 2.0 ** (cents_offset / 1200.0)

    warped_pos = np.cumsum(ratio)
    warped_pos -= warped_pos[0]
    warped_pos = np.clip(warped_pos, 0, n - 1)

    followed = np.interp(warped_pos, np.arange(n), audio)
    return followed.astype(np.float32)


def apply_amplitude_curve_follow(audio: np.ndarray, sample_rate: int,
                                  amplitude_curve: np.ndarray, frame_hop_s: float,
                                  min_gain: float = 0.15, max_gain: float = 1.6,
                                  attack_ms: float = 15.0, release_ms: float = 45.0) -> np.ndarray:
    """
    Continuously reshapes an already-rendered note's own loudness to track
    the real performance's dynamics across the note -- the amplitude
    counterpart of apply_pitch_curve_follow above.

    Why this exists: every note synth.synthesize() renders comes out at
    one fixed loudness (self.volume, plus whatever the envelope/limiter
    already do to it), set once and held for the note's whole duration.
    Everything the real singer's own volume did across that same stretch
    of time -- swelling into a held note, trailing off at the end of a
    phrase, the ordinary breath-driven rise and fall of an actual human
    voice -- was thrown away entirely. A robot part that sings at the
    same loudness whether the input was a belt or a whisper doesn't read
    as "following" a real performance nearly as much as one that actually
    swells and recedes with it.

    `amplitude_curve` is one RMS reading per source frame across the
    run (see _render_track_offline), same cadence as the pitch curve.
    It's normalized against its OWN median rather than some fixed
    absolute loudness, so a steadily-sung note ends up close to unity
    gain throughout and only genuine relative swells/dips in the real
    performance move the output up or down from there -- this is a
    dynamics shape, not a volume control. Clamped to [min_gain, max_gain]
    so a brief quiet moment inside an otherwise-sung note doesn't fall to
    total silence (that would read as a dropout, not as dynamics) and a
    loud moment doesn't blow past a sane ceiling.

    FIX (was a flat, symmetric moving-average smoother -- see below):
    smoothing is now an ASYMMETRIC attack/release envelope-follower (the
    same idea a compressor's detector uses), applied at FRAME rate
    instead of a box average applied at sample rate. That distinction is
    what was actually causing an audible "start stop start stop" pumping
    on real singing. Ordinary word structure -- a consonant, a breath,
    the closure between two syllables of the SAME word, e.g. every "l" in
    a run of "la-la-la" -- drives the raw RMS to near-zero for a few tens
    of milliseconds at a time, even in the middle of one continuous,
    intentionally-unbroken phrase. The old 30ms symmetric box average
    barely rounds off a dip that brief (a box filter's response time is
    on the same order as its own window, and syllables repeat faster than
    that), so the OUTPUT gain was still riding those same near-silent
    troughs almost as sharply as the raw RMS did, all the way down to
    whatever min_gain was configured -- audible as the robot cutting out
    and snapping back at roughly syllable rate, not as musical dynamics.
    A slower release (default 45ms here; a caller deliberately chasing
    syllable-by-syllable dynamics closely, e.g. a raw locked-vocable
    render, should pass something longer -- see
    _render_track_single_vocable_offline in server.py, which now also
    raises its own min_gain floor for the same reason) lets a brief
    consonant-driven dip get bridged instead of ever fully registering,
    while a fast attack (default 15ms) keeps the robot responsive to a
    genuine new swell instead of smearing it out.
    """
    n = len(audio)
    if n == 0 or amplitude_curve.size == 0:
        return audio
    if amplitude_curve.size == 1:
        return audio  # only one reading for the whole note -- nothing to follow

    voiced = amplitude_curve[amplitude_curve > 0]
    reference = float(np.median(voiced)) if voiced.size else 0.0
    if reference <= 1e-9:
        return audio  # the whole run measured as silence -- leave it alone rather than dividing by ~0

    frame_gain = np.clip(amplitude_curve / reference, min_gain, max_gain)

    # Envelope-follow at FRAME rate -- cheap (hundreds to a few thousand
    # elements even for a whole multi-minute track) rather than running a
    # stateful attack/release loop at per-sample resolution, which would
    # mean an unvectorizable Python loop over potentially millions of
    # samples. One analysis frame (frame_hop_s, typically ~10-25ms) is
    # short enough that following at frame rate instead of sample rate
    # changes nothing audible, and this is what actually keeps a
    # whole-track offline render fast.
    attack_coeff = float(np.exp(-frame_hop_s / max(1e-4, attack_ms / 1000.0)))
    release_coeff = float(np.exp(-frame_hop_s / max(1e-4, release_ms / 1000.0)))
    smoothed = np.empty_like(frame_gain)
    level = float(frame_gain[0])
    for i in range(len(frame_gain)):
        target = float(frame_gain[i])
        coeff = attack_coeff if target > level else release_coeff
        level = coeff * level + (1.0 - coeff) * target
        smoothed[i] = level
    frame_gain = smoothed

    frame_times = (np.arange(len(frame_gain)) + 0.5) * frame_hop_s
    sample_times = np.arange(n) / float(sample_rate)
    gain = np.interp(sample_times, frame_times, frame_gain,
                      left=frame_gain[0], right=frame_gain[-1])

    return (audio * gain).astype(np.float32)


class VocableSynthesizer:

    def __init__(self):
        cfg = get_config()
        self.engine = cfg["synthesis"]["engine"]
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.crossfade_ms = cfg["synthesis"]["crossfade_ms"]
        self.ddsp_model_path = cfg["synthesis"]["ddsp_model_path"]
        self.vocable_set = cfg["synthesis"]["vocable_set"]
        self.volume = cfg["output"]["volume"]

        self._ddsp_model = None
        self._wavetable_samples: dict[str, np.ndarray] = {}
        self._neural_bank = None
        self._warned_unknown_vocables: set = set()

        self._crossfade_samples = int(self.crossfade_ms * self.sample_rate / 1000)
        self._prev_audio: np.ndarray | None = None
        # Which vocable the last rendered chunk actually was -- lets
        # _crossfade tell "the pitch moved within the same word" apart
        # from "we just switched from singing one word to a completely
        # different one", which need very different blending (see
        # _crossfade for why).
        self._prev_vocable: str | None = None
        # A longer, gentler blend used only when the vocable identity
        # itself changes, on top of the ordinary crossfade_ms used for
        # same-word note-to-note transitions. Two unrelated recorded
        # words don't share any waveform structure to align, so a short
        # linear blend between them reads as a hard cut with a slight
        # dip rather than an actual transition -- a longer, equal-power
        # (constant-perceived-loudness) blend gives the ear enough time
        # to register it as one sound handing off to another instead of
        # one stopping and a different one starting.
        self._identity_crossfade_samples = max(
            self._crossfade_samples, int(cfg["synthesis"].get("vocable_change_crossfade_ms", 90) * self.sample_rate / 1000)
        )

        self._max_voice_layers = 12

        self._rng = np.random.default_rng()

        # Cache of the per-voice detune/timing-jitter/formant "fingerprint"
        # for whatever note is currently being sustained, so legato
        # top-ups of the SAME held note reuse it instead of redrawing
        # fresh randomness every refill -- see _render_ensemble.
        self._locked_ensemble_params: tuple | None = None
        self._locked_note_key: tuple | None = None

        # NEW: live octave shift in semitones
        self._octave_shift_semitones = 0.0

        self._init_engine()

    # NEW method
    def set_octave_shift(self, semitones: float):
        """Update the octave shift offset in real-time."""
        self._octave_shift_semitones = float(semitones)

    def get_vocable_brightness_map(self) -> dict[str, float] | None:
        """
        Exposes the neural vocable bank's own measured, audio-derived
        brightness score (0=warmest loaded vocable, 1=brightest) per
        vocable, so HarmonyEngine._choose_vocable can pick a word whose
        actual recorded tone matches the melody's register/trend instead
        of rotating through vocable_set with no regard for what any of
        the words actually sound like. Returns None (not an empty dict)
        when there's nothing to report -- wrong engine, bank not loaded,
        or bank failed to load -- so the caller can tell "no data" apart
        from "loaded, but somehow empty" and fall back cleanly either way.
        """
        if self.engine != "neural_wavetable" or self._neural_bank is None:
            return None
        try:
            brightness_map = self._neural_bank.vocable_brightness_map()
        except Exception as e:
            logger.warning(f"Could not read vocable brightness map: {e}")
            return None
        return brightness_map or None

    def _init_engine(self):
        if self.engine == "ddsp":
            self._load_ddsp_model()
        elif self.engine == "wavetable":
            self._load_wavetable_samples()
        elif self.engine == "neural_wavetable":
            self._load_neural_bank()
            # Prewarm the neural bank if it loaded successfully
            if self._neural_bank is not None and self._neural_bank.available:
                try:
                    cfg = get_config()
                    fmin = cfg["pitch"]["min_frequency"]
                    fmax = cfg["pitch"]["max_frequency"]
                    self._neural_bank.prewarm_range(fmin, fmax)
                    logger.info(f"Neural vocable bank prewarmed from {fmin}Hz to {fmax}Hz")
                except Exception as e:
                    logger.warning(f"Neural vocable bank prewarm failed: {e} — will warm up lazily on first notes")
        logger.info(f"Vocable synthesizer engine: {self.engine}")

    def _load_ddsp_model(self):
        try:
            from synthesis.ddsp_synthesizer import DDSPVocoder
            if os.path.exists(self.ddsp_model_path):
                self._ddsp_model = DDSPVocoder(self.ddsp_model_path, self.sample_rate)
                logger.info(f"DDSP model loaded from {self.ddsp_model_path}")
            else:
                logger.error(f"DDSP model not found at {self.ddsp_model_path}")
        except Exception as e:
            logger.error(f"DDSP model load error: {e}")

    def _load_neural_bank(self):
        try:
            from synthesis.neural_vocable_bank import NeuralVocableBank
        except ImportError as e:
            logger.error(
                f"neural_wavetable engine selected but neural_vocable_bank.py couldn't "
                f"be imported ({e}). Singing is DISABLED (silent) until this is fixed -- "
                "not substituting a different-sounding synth engine."
            )
            return

        samples_dir = Path(__file__).parent / "samples" / "neural"
        self._neural_bank = NeuralVocableBank(samples_dir, self.sample_rate)

        if not self._neural_bank.available:
            logger.error(
                "neural_wavetable engine selected but no converted samples were found "
                f"at {samples_dir}. Run synthesis/build_vocable_bank.py first (see its "
                "docstring). Singing is DISABLED (silent) until this is fixed -- not "
                "substituting a different-sounding synth engine."
            )
            self._neural_bank = None

    def _load_wavetable_samples(self):
        try:
            samples_dir = Path(__file__).parent / "samples"
            if not samples_dir.exists():
                logger.error(
                    f"No samples directory at {samples_dir}. Add WAV files named "
                    "aah.wav, ooo.wav, mmm.wav, hey.wav to synthesis/samples/ for "
                    "wavetable mode. Singing is DISABLED (silent) until this is fixed "
                    "-- not substituting a different-sounding synth engine."
                )
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
                logger.error(
                    "No wavetable samples loaded. Singing is DISABLED (silent) until "
                    "this is fixed -- not substituting a different-sounding synth engine."
                )

        except Exception as e:
            logger.error(
                f"Wavetable load error: {e}. Singing is DISABLED (silent) until this "
                "is fixed -- not substituting a different-sounding synth engine."
            )

    def synthesize(self, decision, blocking: bool = False, legato: bool = False, apply_crossfade: bool = True):
        """
        legato=True says "this chunk is a mid-note continuation, not a real
        note onset or a real note ending" -- see _apply_envelope for why
        that matters. Callers doing one-shot renders (offline bounce,
        single-note preview) should leave this False, which is also the
        default so existing call sites don't change behavior.

        apply_crossfade=True (the default) blends this call's start
        against whatever the LAST synthesize() call on this instance
        produced -- correct for the live, one-call-follows-the-next-in-
        real-time streaming path, where "last call" and "immediately
        before this, on the timeline" are the same thing. They are NOT
        the same thing for an offline batch render that places each
        note by its own absolute sample position (see
        _render_track_offline in server.py): two calls can be seconds
        apart on the real timeline while still being back-to-back calls
        into this shared, reused synthesizer instance. Blending against
        self._prev_audio there would smear in whatever unrelated word
        happened to render last, not whatever's actually adjacent on the
        timeline. Batch callers that do their own timeline-aware
        stitching should pass apply_crossfade=False and handle blending
        themselves once they know each note's real placement.
        """
        if decision.action == "rest" or decision.target_hz <= 0:
            n_samples = int(0.1 * self.sample_rate)
            return np.zeros(n_samples, dtype=np.float32)

        n_samples = int(decision.duration_s * self.sample_rate)
        if n_samples <= 0:
            return np.zeros(1024, dtype=np.float32)

        num_voices = max(1, int(getattr(decision, "num_voices", 1)))

        if num_voices <= 1:
            audio = self._render_voice(decision, n_samples, voice_index=0,
                                        f0_hz=decision.target_hz, formant_scale=1.0,
                                        blocking=blocking)          # <-- ADD blocking
        else:
            audio = self._render_ensemble(decision, n_samples, num_voices,
                                        blocking=blocking, legato=legato)

        audio = self._apply_envelope(audio, legato=legato)
        audio = self._apply_phoneme_shaping(audio, decision)

        reverb_amount = float(getattr(decision, "reverb_amount", 0.08))
        if reverb_amount > 0:
            audio = self._apply_reverb(audio, reverb_amount)

        audio = audio * self.volume
        audio = np.tanh(audio * 1.15) / np.tanh(1.15)

        if apply_crossfade:
            audio = self._crossfade(audio, vocable=getattr(decision, "vocable", None))
            self._prev_audio = audio
            self._prev_vocable = getattr(decision, "vocable", None)

        return audio.astype(np.float32)

    def _render_ensemble(self, decision, n_samples: int, num_voices: int,
                        blocking: bool = False, legato: bool = False) -> np.ndarray:
        detune_spread = float(getattr(decision, "detune_spread_cents", 10.0))
        jitter_ms = float(getattr(decision, "timing_jitter_ms", 15.0))
        formant_spread = float(getattr(decision, "formant_spread", 0.1))
        max_jitter_samples = int(jitter_ms * self.sample_rate / 1000)

        # Identifies "the note currently being held" -- pitch, vocable,
        # and the texture parameters that shape it. When a note is truly
        # just continuing (legato=True, same signature as last time),
        # reuse the exact same per-voice detune offsets, jitter shifts,
        # and formant scales as last call instead of redrawing them.
        #
        # Before this: a held multi-voice note (duet/choir texture --
        # drone_support's whole identity, for instance) got its ensemble
        # re-randomized on every single legato top-up (server.py refills
        # the queue roughly every 1-2s while sustaining). The pitch
        # wasn't moving, but which way each voice leaned in cents, how
        # far its onset was offset, and its formant "size" all silently
        # reshuffled underneath it -- an otherwise-static pad kept
        # subtly shimmering/warping every refill instead of holding one
        # stable chord. A genuine new note (legato=False, or the
        # signature actually changed) still draws fresh randomness, same
        # as before -- this only stops re-rolling a note that hasn't
        # actually changed.
        note_key = (round(float(decision.target_hz), 3), decision.vocable, num_voices,
                    round(detune_spread, 2), round(jitter_ms, 2), round(formant_spread, 3))

        if legato and self._locked_ensemble_params is not None and self._locked_note_key == note_key:
            detune_offsets, jitter_shifts, formant_scales = self._locked_ensemble_params
        else:
            if num_voices == 1:
                detune_offsets = np.array([0.0])
            else:
                base = np.linspace(-detune_spread, detune_spread, num_voices)
                detune_offsets = base + self._rng.normal(0, detune_spread * 0.15, num_voices)
                detune_offsets[0] = 0.0

            jitter_shifts = np.zeros(num_voices, dtype=int)
            if max_jitter_samples > 0:
                for i in range(1, num_voices):
                    jitter_shifts[i] = int(self._rng.integers(-max_jitter_samples, max_jitter_samples + 1))

            formant_scales = np.ones(num_voices)
            for i in range(1, num_voices):
                formant_scales[i] = 1.0 + self._rng.uniform(-1, 1) * formant_spread

            self._locked_ensemble_params = (detune_offsets, jitter_shifts, formant_scales)
            self._locked_note_key = note_key

        mix = np.zeros(n_samples, dtype=np.float64)
        for i in range(num_voices):
            cents = float(detune_offsets[i])
            f0 = decision.target_hz * (2 ** (cents / 1200.0))
            f_scale = float(formant_scales[i])

            voice = self._render_voice(decision, n_samples, voice_index=i,
                                        f0_hz=f0, formant_scale=f_scale,
                                        blocking=blocking)          # <-- PASS blocking

            shift = int(jitter_shifts[i])
            if shift != 0:
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
                    f0_hz: float, formant_scale: float,
                    blocking: bool = False) -> np.ndarray:   # <-- ADD blocking param
        shifted_f0 = f0_hz

        if self.engine == "neural_wavetable":
            if self._neural_bank is None:
                # ... error handling ...
                return np.zeros(n_samples, dtype=np.float32)

            # Use the correct retrieval method based on blocking flag
            if blocking:
                audio = self._neural_bank.get_blocking(decision.vocable, shifted_f0, n_samples)
            else:
                audio = self._neural_bank.get(decision.vocable, shifted_f0, n_samples)

            if audio is not None:
                return audio

            # ... warning if unknown vocable ...
            return np.zeros(n_samples, dtype=np.float32)

        if self.engine == "wavetable":
            if not self._wavetable_samples:
                if "__no_wavetable__" not in self._warned_unknown_vocables:
                    self._warned_unknown_vocables.add("__no_wavetable__")
                    logger.error(
                        "wavetable engine has no samples loaded -- singing is silent "
                        "until this is fixed (see the startup error above)."
                    )
                return np.zeros(n_samples, dtype=np.float32)
            return self._synthesize_wavetable(decision, n_samples, shifted_f0)

        if self.engine == "ddsp":
            if self._ddsp_model is None:
                if "__no_ddsp_model__" not in self._warned_unknown_vocables:
                    self._warned_unknown_vocables.add("__no_ddsp_model__")
                    logger.error(
                        "ddsp engine has no model loaded -- singing is silent until "
                        "this is fixed (see the startup error above)."
                    )
                return np.zeros(n_samples, dtype=np.float32)
            try:
                return self._ddsp_model.synthesize(decision, n_samples, shifted_f0)
            except Exception as e:
                if "__ddsp_error__" not in self._warned_unknown_vocables:
                    self._warned_unknown_vocables.add("__ddsp_error__")
                    logger.error(f"DDSP synthesis failed: {e} -- resting this note "
                                 "instead of substituting a different-sounding synth voice.")
                return np.zeros(n_samples, dtype=np.float32)

        logger.error(
            f"synthesis.engine={self.engine!r} isn't a recognized engine "
            "(expected neural_wavetable, wavetable, or ddsp) -- singing is silent."
        )
        return np.zeros(n_samples, dtype=np.float32)

    def _synthesize_wavetable(self, decision, n_samples, f0_hz):
        """
        Synthesize from wavetable samples using phase vocoder pitch shifting.
        The key fix: librosa.effects.pitch_shift() preserves duration while
        changing pitch.
        """
        try:
            sample = self._wavetable_samples.get(
                decision.vocable,
                next(iter(self._wavetable_samples.values()))
            )

            f0_orig, _, _ = librosa.pyin(
                sample, fmin=60, fmax=800, sr=self.sample_rate
            )
            f0_orig_mean = float(np.nanmedian(f0_orig)) if f0_orig is not None else 220.0

            n_steps = 12 * np.log2(f0_hz / f0_orig_mean)

            # Phase vocoder pitch shift (preserves duration!)
            shifted = librosa.effects.pitch_shift(
                sample, sr=self.sample_rate, n_steps=float(n_steps)
            )

            if len(shifted) < n_samples:
                repeats = int(np.ceil(n_samples / len(shifted)))
                shifted = np.tile(shifted, repeats)
            audio = shifted[:n_samples]

            return audio.astype(np.float32)

        except Exception as e:
            logger.error(f"Wavetable synthesis error: {e} -- resting this note")
            return np.zeros(n_samples, dtype=np.float32)

    def _apply_envelope(self, audio, legato: bool = False):
        """
        Attack/release ramp for a genuinely new note onset or a genuine
        note ending. Skipped (down to a hairline 2ms declick) for legato
        continuation chunks.

        Why this matters: every synthesize() call already gets stitched to
        the one before it by _crossfade(), which does its own 30ms
        fade-out-of-old / fade-in-of-new blend. That's the right place for
        smoothing a real transition. But when a note is just being
        *continued* -- same pitch, same vocable, singer still holding the
        note, another chunk queued up because the previous one is about to
        run out (see the sustain handling in server.py's live pipeline) --
        this envelope used to fade the audio down toward silence and back
        up on every single one of those chunks too. Stacked on top of
        _crossfade()'s own fade, that produced two multiplied fades
        instead of one: the tail fades out roughly quadratically instead
        of linearly (audibly faster/deeper), and the new chunk's rise to
        full volume gets smeared out to ~30ms instead of reaching volume
        by 10ms. The audible result is a soft "pulse" or "breath" every
        time a held note gets topped up -- which, over a multi-second
        sustained note refilled several times, sounds exactly like the
        voice periodically starting and stopping instead of holding one
        continuous tone.

        A true onset (a brand new "sing" action) still gets its full
        attack, and a true one-shot render (preview endpoint, offline
        bounce) still gets its full attack+release -- legato is only
        for the "there is more of this same note coming" case.
        """
        envelope = np.ones(len(audio))

        if legato:
            declick_samples = min(int(0.002 * self.sample_rate), len(audio) // 4)
            if declick_samples > 0:
                envelope[:declick_samples] = np.linspace(0, 1, declick_samples)
                envelope[-declick_samples:] = np.linspace(1, 0, declick_samples)
            return audio * envelope

        attack_samples = min(int(0.01 * self.sample_rate), len(audio) // 4)
        release_samples = min(int(0.03 * self.sample_rate), len(audio) // 4)

        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

        return audio * envelope

    def _apply_phoneme_shaping(self, audio, decision):
        try:
            if decision.brightness > 0.55:
                gain_db = (decision.brightness - 0.55) * 10.0
                audio = self._high_shelf(audio, corner_hz=3200.0, gain_db=gain_db)
            elif decision.brightness < 0.45:
                # Was boost-only -- brightness had no way to darken the
                # tone at all, only ever brighten it or do nothing. Fine
                # when nothing ever pushed brightness DOWN from neutral,
                # but contour_following now deliberately does exactly
                # that on a falling phrase (see HarmonyEngine.decide),
                # and with no cut available that half of its effect would
                # have been completely silent -- audibly identical to
                # sitting at neutral. Mirrors the boost side: same
                # corner, same 10x scale, just negative gain (a cut).
                gain_db = (decision.brightness - 0.45) * 10.0
                audio = self._high_shelf(audio, corner_hz=3200.0, gain_db=gain_db)

            # vowel_color has carried a real value through PhonemeProfile
            # -> HarmonyDecision this whole time (0=bright front vowel
            # like "ee", 1=dark back vowel like "oh/oo" -- see
            # CREE_PHONEME_PROFILES in phonetic_analysis.py) but nothing
            # here ever actually read it -- it reached this function and
            # was simply never used. brightness's shelf is a broad tilt
            # across everything above ~3.2kHz; this is different and
            # complementary -- a single resonant peak that actually SLIDES
            # in frequency with vowel_color, from a bright ~2400Hz "ee"-
            # ish formant down to a dark ~900Hz "oh/oo"-ish one, which
            # reads as the vowel itself shifting rather than a tone
            # control moving. Skipped near dead-center (0.45-0.55) so an
            # ordinary, unremarkable vowel doesn't get colored for no
            # reason -- this is for when something (contour_following's
            # continuous lean, or the Cree phoneme classifier itself)
            # has actually pushed it toward one end or the other.
            if abs(decision.vowel_color - 0.5) > 0.05:
                formant_hz = 2400.0 - decision.vowel_color * 1500.0  # 0 -> 2400Hz, 1 -> 900Hz
                depth = min(1.0, (abs(decision.vowel_color - 0.5) - 0.05) / 0.45) * 0.55
                b_formant, a_formant = iirpeak(formant_hz, 3.0, fs=self.sample_rate)
                formant_shaped = lfilter(b_formant, a_formant, audio)
                audio = audio * (1 - depth) + formant_shaped * depth

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

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio /= peak

            return audio

        except Exception as e:
            logger.error(f"Phoneme shaping error: {e}")
            return audio

    def _high_shelf(self, audio: np.ndarray, corner_hz: float, gain_db: float) -> np.ndarray:
        if gain_db == 0:
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

    def _crossfade(self, new_audio: np.ndarray, vocable: str | None = None) -> np.ndarray:
        if self._prev_audio is None:
            return new_audio

        identity_changed = (
            vocable is not None and self._prev_vocable is not None and vocable != self._prev_vocable
        )
        cf = self._identity_crossfade_samples if identity_changed else self._crossfade_samples

        if len(new_audio) < cf:
            cf = len(new_audio)
        if cf <= 0:
            return new_audio

        prev_tail = self._prev_audio[-cf:] if len(self._prev_audio) >= cf else self._prev_audio
        overlap_len = min(cf, len(prev_tail), len(new_audio))
        if overlap_len <= 0:
            return new_audio

        if identity_changed:
            # Equal-power (constant perceived loudness through the
            # blend) rather than the plain linear ramp used for a
            # same-word pitch move. A linear crossfade between two
            # DIFFERENT recorded words has an audible dip in the middle
            # (loudness bottoms out around 0.5+0.5 instead of staying
            # constant), which reads as a little gap between "one sound
            # stopping" and "another starting" -- exactly the abrupt
            # jump-cut this exists to avoid. The sin/cos pair keeps
            # combined energy roughly constant throughout the blend.
            t = np.linspace(0, np.pi / 2, overlap_len)
            fade_in = np.sin(t)
            fade_out = np.cos(t)
        else:
            fade_in = np.linspace(0, 1, overlap_len)
            fade_out = np.linspace(1, 0, overlap_len)

        new_audio = new_audio.copy()
        new_audio[:overlap_len] = (
            new_audio[:overlap_len] * fade_in
            + prev_tail[:overlap_len] * fade_out
        )
        return new_audio
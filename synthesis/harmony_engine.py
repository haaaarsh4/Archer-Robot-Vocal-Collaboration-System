import time
import random
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from config.config_loader import get_config
from analysis.phonetic_analysis import PhonemeProfile
from core.protocol_guard import ProtocolGuard
from synthesis.accompaniment_modes import (
    AccompanimentMode,
    ModeProposal,
    MusicalContext,
    ModeFunctions,
    AccompanimentModeSelector,
    VoiceTexture,
    TextureParams,
    DEFAULT_TEXTURE,
)
import librosa


@dataclass
class HarmonyDecision:
    target_hz  : float
    vocable    : str
    duration_s : float
    vowel_color: float
    nasality   : float
    brightness : float
    action     : str
    mode       : AccompanimentMode
    mode_note  : str = ""
    texture         : str   = "solo"
    num_voices      : int   = 1
    detune_spread_cents: float = 0.0
    timing_jitter_ms: float = 0.0
    formant_spread   : float = 0.0
    reverb_amount    : float = 0.08


class HarmonyEngine:
    INTERVALS = {
        "unison": 0,
        "third":  4,
        "fifth":  7,
        "octave": 12,
    }

    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    def __init__(self):
        cfg = get_config()
        harmony_cfg = cfg["harmony"]
        self.detune_cents = harmony_cfg["detune_cents"]
        self.scale_lock = harmony_cfg["scale_lock"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        self.protocol = ProtocolGuard(cfg)

        self.mode_selector = AccompanimentModeSelector(cfg)
        self.mode_functions = ModeFunctions(cfg)
        self.ctx = MusicalContext()
        self._fusion_third_semitones = harmony_cfg.get("fusion_third_semitones", 4)

        self.texture_params = TextureParams(cfg)
        self._texture_override: Optional[VoiceTexture] = None

        synth_cfg = cfg.get("synthesis", {})
        self._vocable_common = synth_cfg.get("vocable_set", ["aah", "ooo", "mmm", "hey"])
        # Dedicated close chorus-detune for timbral_thickening -- see the
        # fix in decide() below. Separate from any texture's own
        # detune_spread_cents, which is a wider, ensemble-oriented spread
        # meant for a duet/choir, not a single "thicker" voice.
        self._timbral_detune_cents = float(synth_cfg.get("timbral_detune_cents", 6.0))
        # Drone's own narrow detune -- a soft, slow beating between two
        # voices holding the same low note, not the wider chorus-y
        # spread the DUET texture uses for a moving melodic line.
        self._drone_detune_cents = float(synth_cfg.get("drone_detune_cents", 3.0))
        # How far contour_following's continuous timbral lean (see
        # ModeFunctions._contour_timbre_lean) is allowed to push
        # brightness/vowel_color at full deflection. Kept as separate
        # knobs since brightness and vowel_color aren't the same
        # perceptual axis -- brightness alone reads as "louder overtones",
        # while also nudging vowel_color toward the bright end (see decide()
        # below) is what actually makes it sound like the vowel itself is
        # shifting (toward "ee"/"ah" rising, toward "oh"/"oo" falling),
        # not just a tone-control sweep. Deliberately large (was 0.30) --
        # a subtle EQ nudge disappears against a real mix; this needs to
        # be unmistakable, not tasteful.
        #
        # Split into _up/_down pairs (rather than one symmetric amount)
        # so the frontend's Max brighten / Max darken sliders -- two
        # independent controls -- can actually produce an asymmetric
        # swing. Config only has one value each for backward
        # compatibility (a config-only setup with no frontend override
        # has no reason to want asymmetry), so both directions default
        # to the same config value; set_contour_brightness_db (below)
        # is what actually splits them apart per-request.
        _contour_brightness_default = float(synth_cfg.get("contour_brightness_amount", 0.45))
        _contour_vowel_color_default = float(synth_cfg.get("contour_vowel_color_amount", 0.45))
        self._contour_brightness_amount_up = _contour_brightness_default
        self._contour_brightness_amount_down = _contour_brightness_default
        self._contour_vowel_color_amount_up = _contour_vowel_color_default
        self._contour_vowel_color_amount_down = _contour_vowel_color_default
        # Applied UNCONDITIONALLY, regardless of lean -- see decide()'s
        # CONTOUR block. Without this, lean == 0 (a genuinely flat
        # melody) meant brightness/vowel_color were completely
        # untouched, i.e. still silently identical to unison right when
        # it mattered most. This guarantees contour_following's resting
        # tone is already distinguishable before any movement happens at
        # all -- an always-on identity, not one that only shows up when
        # the melody cooperates.
        self._contour_baseline_brightness_shift = float(synth_cfg.get("contour_baseline_brightness_shift", 0.18))
        self._default_vocable_name = next(
            (v for v in self._vocable_common if "oh" in v.lower()),
            (self._vocable_common[0] if self._vocable_common else "aah"),
        )
        # Dedicated vocable for drone_support -- a held grounding tone
        # wants one consistent, smooth, closed-mouth-ish sound (a hum or
        # "ohh"), not whatever's rotating through vocable_set for the
        # melodic line above it. Configurable via synthesis.drone_vocable;
        # falls back to the same "oh"-containing pick as the bootstrap
        # default, since that's the closest thing to a plain sustained
        # vowel already available in most vocable_set configs.
        self._drone_vocable_name = synth_cfg.get("drone_vocable", self._default_vocable_name)
        self._vocable_rare = synth_cfg.get("vocable_rare_set", [])
        self._vocable_rare_chance = float(synth_cfg.get("vocable_rare_chance", 0.0))
        self._locked_vocable: str | None = None

        # How many notes a single chosen vocable is allowed to carry on
        # for before a fresh pick is forced even without a rest in
        # between -- see the "word_overstayed" check in decide() below.
        # Purely a variety safety valve for an unusually long unbroken
        # phrase; ordinary phrase-to-phrase changes are driven by actual
        # rests, not this counter.
        self._vocable_max_notes_per_word = int(synth_cfg.get("vocable_max_notes_per_word", 8))
        self._notes_since_vocable_change: int = 0
        # The word actually picked last time, purely so _choose_vocable
        # can avoid handing back the exact same word twice in a row when
        # it has any other reasonable option -- see _pick_by_melodic_fit.
        self._last_used_vocable: str | None = None
        self._vocable_rng = np.random.default_rng()
        # When set (see set_vocable_override), _choose_vocable always
        # returns this exact word and every rotation/brightness-matching
        # priority below is skipped -- this is what backs the frontend's
        # "pick one word instead of Mix" picker for DSP mode.
        self._vocable_override: str | None = None

        self._pitch_history: collections.deque = collections.deque(maxlen=32)
        self._current_key_root: int = 0
        self._current_scale: list = self.MAJOR_SCALE

        self._key_infer_counter: int = 0
        self._key_update_interval: int = 43

        self._current_decision: HarmonyDecision | None = None
        self._frames_on_current_note: int = 0
        self._max_frames_per_note: int = 200

        self._pending_note_hz: float | None = None
        self._pending_note_frames: int = 0

        self._last_good_hz: float | None = None
        # FIX: this was `_gap_frames: int` counted against a
        # `_max_gap_frames = 8` cap -- at this project's normal frame
        # settings that's roughly 90ms, a fraction of an ordinary breath
        # or word-boundary pause in real singing. Once a dropout ran past
        # that, decide() (below) didn't hold the last real note -- it
        # snapped the target pitch to self._default_start_hz, a flat,
        # hardcoded 220Hz completely unrelated to the melody, for as long
        # as the gap continued. That's a real, audible wrong-note glitch
        # landing on every ordinary pause in the singing, not just long
        # ones -- likely a big part of why the output could sound like it
        # shares nothing with the input: it wasn't just briefly quiet at
        # each breath, it was briefly singing THE WRONG NOTE.
        #
        # Fixed by just holding the last real pitch for as long as
        # phrase_state keeps reporting "singing" -- no expiring counter,
        # no fallback pitch. "Start from exactly where it stopped" is
        # exactly what continuing to hold the last real note IS. Whether
        # a given pause is short enough to still count as "singing" (and
        # so gets bridged) or long enough to actually end the phrase is
        # phrase_state's call, not this engine's -- that's decided by
        # RhythmAnalyzer.push_frame's own hangover/silence timing
        # (analysis/rhythm_analyzer.py), which wasn't available to check
        # while making this change. If short pauses are STILL cutting the
        # music off after this fix, that file is almost certainly where
        # the remaining threshold lives, in the same family as this bug.
        # _default_start_hz below is now used ONLY for true bootstrap (no
        # real pitch has ever locked in this performance yet), never as a
        # mid-phrase gap filler.
        # Comfortable mid pitch (A3) used only for the bootstrap case
        # above -- no real pitch has ever locked yet. Not a musical
        # choice, just a reasonable starting point that gets corrected
        # the instant real pitch data arrives.
        self._default_start_hz: float = 220.0

        self._manual_override = False
        self._current_interval: str = "fifth"

        self._forced_mode: AccompanimentMode | None = None

        # NEW: reference to the synthesizer for live octave shift
        self._synthesizer = None

        self._t0 = time.monotonic()

        logger.info(
            f"HarmonyEngine ready — default_mode={self.mode_selector.default_mode.value}, "
            f"protocol_enabled={self.protocol.enabled}, fusion_mode={self.mode_selector.fusion_mode}"
        )

    # NEW: set the synthesizer reference
    def set_synthesizer(self, synthesizer):
        self._synthesizer = synthesizer

    # NEW: forward octave shift to the synthesizer
    def set_octave_shift(self, semitones: float):
        if self._synthesizer is not None:
            self._synthesizer.set_octave_shift(semitones)
        else:
            logger.warning("Cannot set octave shift: synthesizer not initialized")

    def decide(
        self,
        archer_hz: float | None,
        phrase_state: str,
        tempo_bpm: float,
        phoneme_profile: PhonemeProfile,
    ) -> HarmonyDecision:
        now_s = time.monotonic() - self._t0
        beat_duration_s = (60.0 / tempo_bpm) if tempo_bpm > 0 else 0.5
        phrase_just_ended = phrase_state == "phrase_end"

        if archer_hz:
            self._last_good_hz = archer_hz
            self._pitch_history.append(archer_hz)
            self._key_infer_counter += 1
            if self._key_infer_counter >= self._key_update_interval:
                self._key_infer_counter = 0
                self._infer_key()
        elif phrase_state == "singing" and self._last_good_hz is not None:
            # An ordinary pause inside an active phrase (a breath, a
            # consonant, a brief pitch-tracker dropout) -- hold the last
            # real pitch exactly as-is rather than touching it. No
            # counter, no expiry: as long as phrase_state keeps reporting
            # "singing", this keeps the note going from precisely where
            # it left off, for however long that takes.
            archer_hz = self._last_good_hz
        elif phrase_state == "singing":
            # True bootstrap only: the singer is clearly active but no
            # confident pitch has EVER locked yet, e.g. the very start of
            # a performance before the first successful detection. Never
            # leave dead air waiting for a perfect reading -- start on a
            # sensible default pitch now; once real pitch data arrives it
            # registers as a normal note change through the usual
            # onset-detection logic below, nothing special-cased.
            archer_hz = self._default_start_hz

        key_root_hz = self._key_root_hz()

        octave_shift = 0.0
        if self._synthesizer is not None:
            octave_shift = self._synthesizer._octave_shift_semitones

        self.ctx.update(
            archer_hz=archer_hz,
            now_s=now_s,
            beat_duration_s=beat_duration_s,
            key_root_hz=key_root_hz,
            phrase_just_ended=phrase_just_ended,
            octave_shift_semitones=octave_shift,  # <-- ADD THIS
        )

        is_protocol_sensitive = (
            phoneme_profile is not None
            and phoneme_profile.detected_class in self.protocol.sensitive_phoneme_classes
        )

        if self._manual_override:
            proposal = self._manual_proposal(archer_hz)
            mode = AccompanimentMode.UNISON
        elif self._forced_mode is not None:
            if not self.protocol.enabled or is_protocol_sensitive:
                mode = AccompanimentMode.SILENT
                proposal = self.mode_functions.silent(
                    "protocol is off or the current phoneme is protocol-sensitive"
                )
            else:
                mode = self._forced_mode
                proposal = self._invoke_mode(mode, self.ctx)
        else:
            mode = self.mode_selector.select(
                protocol_enabled=self.protocol.enabled,
                is_protocol_sensitive=is_protocol_sensitive,
                archer_hz=archer_hz,
                phrase_state=phrase_state,
                phrase_just_ended=phrase_just_ended,
                tempo_bpm=tempo_bpm,
                frames_on_current_note=self._frames_on_current_note,
            )
            proposal = self._invoke_mode(mode, self.ctx)

        target_hz = self._finalize_pitch(proposal, mode)

        # APPLY OCTAVE SHIFT HERE (at harmony engine level)
        # This ensures the shift is applied ONCE, not stacked with mode shifts
        if target_hz is not None and target_hz > 0 and self._synthesizer is not None:
            octave_shift = self._synthesizer._octave_shift_semitones
            if octave_shift != 0:
                # octave_shift is in SEMITONES (see /harmony/octave-shift in
                # server.py and set_octave_shift below) -- 12 semitones per
                # octave, not 1200 (that's cents). Dividing by 1200 made this
                # knob almost inaudible on its own -- a +/-24 semitone request
                # only moved pitch by a few cents.
                target_hz = target_hz * (2 ** (octave_shift / 12.0))

        duration = min(max(proposal.hold_beats, 0.25) * beat_duration_s, 4.0)

        action = self._resolve_action(proposal, target_hz)

        if action == "sing":
            if mode == AccompanimentMode.DRONE:
                # A grounding drone should always sound like one smooth,
                # steady "ohh" underneath the phrase -- not whatever
                # rhythmic syllable happens to be rotating through
                # vocable_set (Lalanana/ohoho by default), which is
                # tuned for a moving melodic line, not a note held for
                # many seconds at a time. Bypasses the normal rotation
                # entirely for this mode. Only relevant at an actual
                # (re)lock moment now (see ModeFunctions.drone()) since
                # a stable drone rarely re-triggers "sing" at all.
                self._locked_vocable = self._drone_vocable_name
                self._notes_since_vocable_change = 0
            elif self._last_good_hz is None:
                # Bootstrap note -- no real pitch detection has landed
                # yet, we're singing on the default pitch above. Open
                # with the requested default vocable rather than
                # whatever an empty pitch_history rotation would pick.
                self._locked_vocable = self._default_vocable_name
                self._notes_since_vocable_change = 0
            else:
                # A held musical phrase should carry ONE vocable through
                # it, the way a real backing vocalist rides one syllable
                # across a phrase instead of switching words on every
                # new note -- so a fresh word is only actually picked at
                # a genuine phrase START (self._locked_vocable is None,
                # meaning the last thing that happened before this note
                # was a rest) or once a single word has already carried
                # on for _vocable_max_notes_per_word notes without a
                # break. Every other "sing" action within an ongoing
                # phrase just keeps singing the word already locked in.
                # This is the actual fix for a vocable that has no
                # relationship to the moment it shows up in: before this,
                # _choose_vocable ran fresh on every single note, so an
                # unrelated word could cut in mid-phrase for no musical
                # reason at all.
                self._notes_since_vocable_change += 1
                is_new_phrase = self._locked_vocable is None
                word_overstayed = self._notes_since_vocable_change > self._vocable_max_notes_per_word
                if is_new_phrase or word_overstayed:
                    self._locked_vocable = self._choose_vocable(phoneme_profile)
                    self._notes_since_vocable_change = 0
        elif action == "rest":
            self._locked_vocable = None
            self._notes_since_vocable_change = 0
        vocable = self._locked_vocable or self._choose_vocable(phoneme_profile)

        influence = getattr(phoneme_profile, "influence", 0.0) or 0.0
        vowel_color = 0.5 * (1 - influence) + phoneme_profile.vowel_color * influence
        nasality    = phoneme_profile.nasality * influence
        brightness  = 0.5 * (1 - influence) + phoneme_profile.brightness * influence

        texture = self._texture_override or DEFAULT_TEXTURE.get(mode, VoiceTexture.SOLO)
        layer = self.texture_params.get(texture)

        num_voices = layer.num_voices
        detune_spread_cents = layer.detune_spread_cents
        timing_jitter_ms = layer.timing_jitter_ms
        formant_spread = layer.formant_spread
        reverb_amount = layer.reverb_amount

        if mode == AccompanimentMode.TIMBRAL:
            # Timbral thickening's whole identity is a second, closely
            # detuned copy of the SAME note -- with only one voice there's
            # nothing to detune against, so it was audibly indistinguishable
            # from unison_shadowing (see ModeFunctions.timbral(), which
            # returns the exact same pitch as unison on purpose -- this is
            # meant to thicken, not harmonize). Guarantee a doubled voice
            # here regardless of whatever Voice Texture the person has
            # picked -- Solo would otherwise silently erase this mode's
            # whole character -- and use the mode's own tight, deliberately
            # narrow detune (synthesis.timbral_detune_cents in config.yaml,
            # previously read from config but never actually applied
            # anywhere) instead of the chosen texture's generic, wider,
            # ensemble-oriented spread. Thickening should sound like one
            # warmer, fuller voice, not a small crowd -- so jitter and
            # reverb are capped even if a bigger texture (e.g. Choir) is
            # selected, and a touch of formant spread is guaranteed so the
            # second layer reads as a second voice, not just a pitch-
            # doubled copy of the same one.
            num_voices = max(2, num_voices)
            detune_spread_cents = self._timbral_detune_cents
            timing_jitter_ms = min(timing_jitter_ms, 6.0)
            formant_spread = max(formant_spread, 0.08)
            reverb_amount = max(reverb_amount, 0.12)

        if mode == AccompanimentMode.DRONE:
            # A grounding drone should be warm and rock-still, not a
            # moving ensemble -- guaranteed doubling here regardless of
            # whatever Voice Texture is selected (Solo would otherwise
            # leave it as one thin low note), but with the drone's OWN
            # tight, deliberately narrow detune (a soft, slow beating
            # like a tanpura or two singers holding the same low note --
            # not the wider chorus-y spread meant for a moving melodic
            # duet/choir). Jitter is capped very low: on a note that's
            # sustained for many seconds, even a small per-voice onset
            # offset reads as a slow drift rather than natural ensemble
            # looseness. Reverb gets a boost -- a drone should feel like
            # it's sitting in the room underneath everything else, not
            # close-mic'd like the lead melodic voice.
            num_voices = max(2, num_voices)
            detune_spread_cents = self._drone_detune_cents
            timing_jitter_ms = min(timing_jitter_ms, 4.0)
            formant_spread = min(formant_spread, 0.06)
            reverb_amount = max(reverb_amount, 0.18)

        if mode == AccompanimentMode.CONTOUR:
            # contour_following's discrete harmony step (see
            # ModeFunctions.contour) only engages once real melodic
            # movement crosses a threshold -- it has to work that way
            # under scale_lock, which rounds pitch to the nearest scale
            # degree and would erase anything subtler anyway. The result:
            # whenever a phrase holds fairly still, the mode's target
            # pitch is EXACTLY archer_hz -- identical output to
            # unison_shadowing, nothing else distinguishing it, for
            # however long the melody stays put (which for a short,
            # mostly-spoken vocal take can be most of it).
            #
            # Brightness/vowel_color aren't touched by scale_lock, so
            # they can track melodic direction continuously instead of
            # waiting for a threshold crossing -- _contour_timbre_lean
            # (-1 falling .. +1 rising) is updated every single call in
            # ModeFunctions.contour. But lean alone still hits EXACTLY
            # zero whenever the melody is genuinely flat -- and at
            # lean == 0 the line below is brightness + 0, i.e. still
            # silently identical to unison right when it matters most.
            # self._contour_baseline_brightness_shift fixes that: a
            # constant, unconditional nudge applied regardless of lean,
            # so this mode's resting/neutral tone is ALREADY a
            # noticeably brighter, more forward voice than unison's,
            # before any melodic movement is even considered. The lean
            # swings around that baseline, not around neutral.
            lean = self.mode_functions._contour_timbre_lean
            brightness_amount = self._contour_brightness_amount_up if lean >= 0 else self._contour_brightness_amount_down
            vowel_color_amount = self._contour_vowel_color_amount_up if lean >= 0 else self._contour_vowel_color_amount_down
            brightness = float(np.clip(
                brightness + self._contour_baseline_brightness_shift
                + lean * brightness_amount,
                0.0, 1.0,
            ))
            # Lower vowel_color reads as brighter ("ee"/"ah"-like) per
            # CREE_PHONEME_PROFILES in phonetic_analysis.py (long_i:
            # brightness 0.9/vowel_color 0.0 vs long_o: brightness
            # 0.4/vowel_color 0.8) -- so rising (lean > 0) should PULL
            # vowel_color down, hence the minus signs here (baseline
            # shift included, mirroring brightness's own baseline).
            vowel_color = float(np.clip(
                vowel_color - self._contour_baseline_brightness_shift
                - lean * vowel_color_amount,
                0.0, 1.0,
            ))

        decision = HarmonyDecision(
            target_hz=target_hz or 0.0,
            vocable=vocable,
            duration_s=duration if action != "rest" else 0.0,
            vowel_color=vowel_color,
            nasality=nasality,
            brightness=brightness,
            action=action,
            mode=mode,
            mode_note=proposal.note,
            texture=texture.value,
            num_voices=num_voices,
            detune_spread_cents=detune_spread_cents,
            timing_jitter_ms=timing_jitter_ms,
            formant_spread=formant_spread,
            reverb_amount=reverb_amount,
        )

        self._current_decision = decision
        return decision

    def _invoke_mode(self, mode: AccompanimentMode, ctx: MusicalContext) -> ModeProposal:
        fn = self.mode_functions
        if mode == AccompanimentMode.UNISON:
            return fn.unison(ctx)
        if mode == AccompanimentMode.OCTAVE:
            return fn.octave(ctx)
        if mode == AccompanimentMode.DELAYED:
            return fn.delayed(ctx)
        if mode == AccompanimentMode.TIMBRAL:
            return fn.timbral(ctx)
        if mode == AccompanimentMode.CONTOUR:
            return fn.contour(ctx)
        if mode == AccompanimentMode.DRONE:
            return fn.drone(ctx)
        if mode == AccompanimentMode.CALL_RESPONSE:
            return fn.call_response(ctx)
        if mode == AccompanimentMode.TRIADIC:
            return fn.triadic(ctx, self._fusion_third_semitones)
        if mode == AccompanimentMode.HUM:
            return fn.hum(ctx)
        return fn.silent("protocol is off or the current phoneme is protocol-sensitive")

    def _manual_proposal(self, archer_hz: Optional[float]) -> ModeProposal:
        if archer_hz is None:
            return ModeProposal(AccompanimentMode.UNISON, None, "rest",
                                 note="manual interval mode — waiting for a pitch")
        semitones = self.INTERVALS.get(self._current_interval, 7)
        target = archer_hz * (2 ** (semitones / 12.0))
        return ModeProposal(AccompanimentMode.UNISON, target, "sing",
                             note=f"manual interval override ({self._current_interval})")

    # Modes whose entire job is to reproduce a pitch Archer ACTUALLY sang
    # (right now, an octave away, a beat ago, or doubled) rather than to
    # introduce a new harmonic relationship on top of his melody. For all
    # of these, his own voice is already the ground truth -- there is no
    # "wrong note" for scale-lock to correct, only a real pitch it can
    # pull the robot away from. Snapping these to the nearest scale
    # degree was the actual reason unison_shadowing (and the others
    # below) never quite matched Archer's real tone: it silently rounded
    # his exact pitch to whatever scale degree was closest, every single
    # frame, even when he was singing perfectly in tune with himself just
    # not with the engine's own inferred scale. CALL_RESPONSE and TRIADIC
    # are deliberately left OUT of this set: they invent a genuinely new
    # pitch relationship (resolving toward an inferred tonic, or a fixed
    # chordal interval) that scale-lock is legitimately meant to keep
    # in-key. CONTOUR is handled specially just below, since it's a
    # mode that's sometimes a pure copy and sometimes a real interval,
    # depending on its own internal hysteresis state.
    _EXACT_COPY_MODES = frozenset({
        AccompanimentMode.UNISON,
        AccompanimentMode.OCTAVE,
        AccompanimentMode.DELAYED,
        AccompanimentMode.TIMBRAL,
        AccompanimentMode.DRONE,
    })

    def _finalize_pitch(self, proposal: ModeProposal, mode: AccompanimentMode) -> Optional[float]:
        target_hz = proposal.target_hz
        if target_hz is None or target_hz <= 0:
            return target_hz

        is_exact_copy = mode in self._EXACT_COPY_MODES
        # contour_following is only a "new interval" mode while it has
        # actually committed to a step away from unison (see
        # ModeFunctions.contour's hysteresis state machine). While its
        # state is "unison" its target IS archer_hz, unmodified -- exactly
        # like unison_shadowing at that moment -- so it deserves the same
        # exemption for as long as that's true, and only that long.
        if mode == AccompanimentMode.CONTOUR and self.mode_functions._contour_state == "unison":
            is_exact_copy = True

        if self.scale_lock and not is_exact_copy:
            target_hz = self._snap_to_scale(target_hz)

        if self.detune_cents and mode not in (AccompanimentMode.UNISON, AccompanimentMode.DRONE):
            target_hz *= 2 ** (self.detune_cents / 1200.0)

        return target_hz

    NOTE_CHANGE_THRESHOLD_CENTS = 70.0

    def _resolve_action(self, proposal: ModeProposal, target_hz: Optional[float]) -> str:
        if proposal.action == "rest" or target_hz is None or target_hz <= 0:
            self._frames_on_current_note = 0
            self._pending_note_hz = None
            self._pending_note_frames = 0
            return "rest"

        no_prior_note = (
            self._current_decision is None
            or self._current_decision.action == "rest"
            or self._current_decision.target_hz is None
            or self._current_decision.target_hz <= 0
        )

        if no_prior_note:
            self._frames_on_current_note = 0
            self._pending_note_hz = None
            self._pending_note_frames = 0
            return "sing"

        cents_diff = abs(1200.0 * np.log2(target_hz / self._current_decision.target_hz))
        pitch_moved = cents_diff > self.NOTE_CHANGE_THRESHOLD_CENTS

        if pitch_moved:
            if self._pending_note_hz is not None and \
                    abs(1200.0 * np.log2(target_hz / self._pending_note_hz)) <= self.NOTE_CHANGE_THRESHOLD_CENTS:
                self._pending_note_frames += 1
            else:
                self._pending_note_hz = target_hz
                self._pending_note_frames = 1
        else:
            self._pending_note_hz = None
            self._pending_note_frames = 0

        is_new_note = (
            self._pending_note_frames >= 2
            or self._frames_on_current_note >= self._max_frames_per_note
        )

        if is_new_note:
            self._frames_on_current_note = 0
            self._pending_note_hz = None
            self._pending_note_frames = 0
            return "sing"

        self._frames_on_current_note += 1
        return "sustain"

    def _snap_to_scale(self, freq_hz: float) -> float:
        try:
            midi = librosa.hz_to_midi(freq_hz)
            note_class = int(round(midi)) % 12

            distances = [abs(note_class - (self._current_key_root + d) % 12)
                         for d in self._current_scale]
            nearest_degree = self._current_scale[int(np.argmin(distances))]
            snapped_note_class = (self._current_key_root + nearest_degree) % 12

            octave = int(midi) // 12
            snapped_midi = octave * 12 + snapped_note_class
            return float(librosa.midi_to_hz(snapped_midi))

        except Exception as e:
            logger.error(f"Scale snap error: {e}")
            return freq_hz

    def _key_root_hz(self) -> Optional[float]:
        if not self._pitch_history:
            return None
        try:
            ref_midi = librosa.hz_to_midi(self._pitch_history[-1])
            octave = int(ref_midi) // 12
            root_midi = octave * 12 + self._current_key_root
            return float(librosa.midi_to_hz(root_midi))
        except Exception as e:
            logger.error(f"Key-root Hz conversion error: {e}")
            return None

    def _infer_key(self):
        if len(self._pitch_history) < 8:
            return
        try:
            midi_notes = [int(round(librosa.hz_to_midi(hz)))
                          for hz in self._pitch_history if hz > 0]
            if not midi_notes:
                return

            chroma = np.zeros(12)
            for n in midi_notes:
                chroma[n % 12] += 1
            chroma /= chroma.sum()

            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                                       2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                                       2.54,4.75,3.98,2.69,3.34,3.17])

            best_corr = -np.inf
            best_root = 0
            best_scale = self.MAJOR_SCALE

            for root in range(12):
                shifted_chroma = np.roll(chroma, -root)
                major_corr = float(np.corrcoef(shifted_chroma, major_profile)[0, 1])
                minor_corr = float(np.corrcoef(shifted_chroma, minor_profile)[0, 1])

                if major_corr > best_corr:
                    best_corr, best_root = major_corr, root
                    best_scale = self.MAJOR_SCALE
                if minor_corr > best_corr:
                    best_corr, best_root = minor_corr, root
                    best_scale = self.MINOR_SCALE

            if best_root != self._current_key_root or best_scale != self._current_scale:
                self._current_key_root = best_root
                self._current_scale = best_scale
                logger.debug(f"Key updated: root={best_root}, scale={best_scale}")

        except Exception as e:
            logger.error(f"Key inference error: {e}")

    def _choose_vocable(self, profile: PhonemeProfile) -> str:
        """
        Picks the next word/syllable to sing. Only actually called at a
        genuine phrase start or after a word has overstayed its welcome
        (see the caller in decide()) -- never on every note -- so
        whatever comes back here is going to carry across an entire
        musical phrase, and it needs to earn that.

        Priority order:
          0. A forced single-vocable override (see set_vocable_override)
             -- when the caller has locked this engine to one specific
             recorded word (the frontend's "one word, not Mix" picker),
             every single one of the priorities below is skipped
             entirely. This is deliberately checked before even the rare
             set, so a locked choice is never silently swapped out for a
             surprise word.
          1. An occasional deliberate "surprise" word from
             vocable_rare_set, same as before.
          2. If the loaded voice bank can tell us how each word actually
             SOUNDS (see NeuralVocableBank.vocable_brightness_map --
             measured from the real recordings, not guessed from
             filenames), pick whichever configured word's own natural
             tone best matches where the melody sits right now: a high
             or rising phrase leans toward a brighter-sounding word, a
             low or falling one toward a warmer, rounder one. This is
             the actual fix for words that used to show up with zero
             relationship to the moment -- "cold" cutting in mid-phrase
             for no reason was len(pitch_history) % N, a plain counter
             with no idea what any word sounds like or what the melody
             was doing.
          3. If no measured brightness data is available at all (a
             non-neural engine, or the bank hasn't loaded), fall back to
             the old vowel-color split when a real Cree phoneme reading
             is actually available, then to a plain rotation as the last
             resort -- but never repeating the word already playing.
        """
        if self._vocable_override:
            self._last_used_vocable = self._vocable_override
            return self._vocable_override

        if self._vocable_rare and self._vocable_rare_chance > 0 and random.random() < self._vocable_rare_chance:
            choice = random.choice(self._vocable_rare)
            self._last_used_vocable = choice
            return choice

        if not self._vocable_common:
            return "aah"
        if len(self._vocable_common) == 1:
            self._last_used_vocable = self._vocable_common[0]
            return self._vocable_common[0]

        brightness_map = self._synthesizer.get_vocable_brightness_map() if self._synthesizer else None
        # Only worth using if it actually covers something in the
        # configured word set -- a bank loaded with a totally different
        # vocable_set than what's configured now shouldn't silently
        # steer word choice.
        usable_brightness_map = None
        if brightness_map:
            usable_brightness_map = {v: b for v, b in brightness_map.items() if v in self._vocable_common}

        if usable_brightness_map:
            target = self._melodic_target_brightness(profile)
            return self._pick_by_melodic_fit(usable_brightness_map, target)

        if profile is not None and profile.influence >= 0.1:
            # Same four-way split as before, but only ever returns a
            # word that's actually configured -- this used to hand back
            # "hey"/"aah"/"ooo"/"mmm" unconditionally even if none of
            # them were in vocable_set at all.
            ordered = ["hey", "aah", "ooo", "mmm"]
            thresholds = [0.25, 0.5, 0.75]
            start = next((i for i, t in enumerate(thresholds) if profile.vowel_color < t), len(thresholds))
            rotated = ordered[start:] + ordered[:start]
            candidate = next((n for n in rotated if n in self._vocable_common), None)
            if candidate is not None:
                self._last_used_vocable = candidate
                return candidate

        idx = len(self._pitch_history) % len(self._vocable_common)
        candidate = self._vocable_common[idx]
        if candidate == self._last_used_vocable:
            idx = (idx + 1) % len(self._vocable_common)
            candidate = self._vocable_common[idx]
        self._last_used_vocable = candidate
        return candidate

    def _melodic_target_brightness(self, profile: PhonemeProfile) -> float:
        """
        0 (warm/rounded) .. 1 (bright/forward) -- what the melody
        itself is asking for right now, combining two signals:

          register: is the current note sitting high or low relative to
          what Archer's actually been singing so far (a running z-score
          over recent pitch history, not a fixed absolute range, so this
          adapts to whoever's actually singing rather than assuming
          everyone's range starts at the same place).

          trend: is the melody climbing or falling right now -- the same
          continuous lean contour_following already uses for its own
          brightness/vowel_color shaping (see ModeFunctions._contour_
          timbre_lean), reused here so a rising phrase pulls toward a
          brighter word for the same reason it already pulls toward a
          brighter tone.

        When a real Cree phoneme reading is actively influencing this
        decision (profile.influence > 0), its own vowel_color reading is
        blended in on top, proportional to how much it's actually
        influencing everything else -- a genuine linguistic signal about
        the vowel being sung shouldn't be thrown away just because
        there's also a brightness map available now.
        """
        target = 0.5
        archer_hz = self.ctx.archer_hz
        if archer_hz and len(self._pitch_history) >= 2:
            history = np.array([h for h in self._pitch_history if h and h > 0], dtype=np.float64)
            if len(history) >= 2:
                log_hist = np.log2(history)
                median_log = float(np.median(log_hist))
                spread = float(np.std(log_hist)) or 1e-6
                register_lean = float(np.clip((np.log2(archer_hz) - median_log) / (spread * 2.0), -1.0, 1.0))

                trend_semitones = self.ctx.pitch_trend_semitones(0.8) if hasattr(self.ctx, "pitch_trend_semitones") else 0.0
                trend_lean = float(np.clip(trend_semitones / 6.0, -1.0, 1.0))

                target = 0.5 + 0.3 * register_lean + 0.25 * trend_lean

        influence = (getattr(profile, "influence", 0.0) or 0.0) if profile is not None else 0.0
        if profile is not None and influence > 0:
            vowel_target = float(np.clip(1.0 - profile.vowel_color, 0.0, 1.0))
            target = target * (1.0 - influence) + vowel_target * influence

        return float(np.clip(target, 0.0, 1.0))

    def _pick_by_melodic_fit(self, brightness_map: dict[str, float], target: float) -> str:
        """
        Picks among the words whose measured brightness is close to
        `target`, weighted toward the closest matches but not
        deterministic -- a real singer doesn't hit the exact same
        syllable every single time the melody revisits a given register,
        and always picking the single closest match would just trade
        one totally-predictable mechanical pattern (the old rotation)
        for a different, tone-based one. The word already playing is
        excluded whenever there's any other reasonable option, which is
        what stops the same word from repeating phrase after phrase even
        when it genuinely is the best tonal fit.
        """
        names = list(brightness_map.keys())
        distances = {n: abs(brightness_map[n] - target) for n in names}
        best = min(distances.values())

        # Wide enough to usually keep a handful of tonally-reasonable
        # options in play, narrow enough that "close" still means
        # something on a normally-sized vocable set.
        tolerance = 0.18
        candidates = [n for n in names if distances[n] <= best + tolerance]

        if self._last_used_vocable in candidates and len(candidates) > 1:
            candidates = [n for n in candidates if n != self._last_used_vocable]

        if len(candidates) == 1:
            choice = candidates[0]
        else:
            weights = np.array([1.0 / (distances[n] + 0.05) for n in candidates])
            weights = weights / weights.sum()
            choice = str(self._vocable_rng.choice(candidates, p=weights))

        self._last_used_vocable = choice
        return choice

    def set_interval(self, interval: str):
        if interval in self.INTERVALS:
            self._current_interval = interval
            self._manual_override = True
            logger.info(f"Manual interval override engaged: {interval}")
        else:
            logger.warning(f"Unknown interval '{interval}'. Options: {list(self.INTERVALS)}")

    def clear_manual_override(self):
        self._manual_override = False
        logger.info("Manual interval override cleared")

    def set_fusion_mode(self, enabled: bool):
        self.mode_selector.fusion_mode = bool(enabled)
        logger.info(f"Fusion mode {'ENABLED' if enabled else 'DISABLED'}")

    def set_texture(self, texture_name: str):
        try:
            self._texture_override = VoiceTexture(texture_name)
            logger.info(f"Voice texture pinned to: {texture_name}")
        except ValueError:
            logger.warning(f"Unknown texture '{texture_name}'. Options: {[t.value for t in VoiceTexture]}")

    def clear_texture_override(self):
        self._texture_override = None
        logger.info("Voice texture back to per-mode defaults")

    def set_vocable_override(self, vocable: str | None):
        """
        Locks this engine to ONE specific recorded word, or clears the
        lock back to the normal rotation/brightness-matching behavior in
        _choose_vocable. This is what backs the frontend's "pick one
        word instead of Mix" DSP picker: every note this engine sings
        from here on uses exactly this recording (still pitch-shifted
        live to follow the melody), nothing else.

        Also immediately reassigns _locked_vocable so the change takes
        effect on the very next note, rather than waiting for the
        current phrase to end -- picking a word in the UI should be
        heard right away, not queued up behind whatever's already
        playing.
        """
        vocable = (vocable or "").strip().lower() or None
        self._vocable_override = vocable
        if vocable:
            self._locked_vocable = vocable
            logger.info(f"Vocable locked to single word: '{vocable}' (accompaniment mode selection bypassed)")
        else:
            logger.info("Vocable override cleared -- back to normal rotation")

    def set_contour_brightness_db(self, baseline_db: float, max_bright_db: float, max_dark_db: float):
        """
        Per-request override for contour_following's three frontend
        sliders (Contour baseline / Max brighten / Max darken -- see
        index.html's demoContourBaselineDb etc.), all specified in dB to
        match what's actually shown in the UI and how the Neural/JS
        engine already works natively.

        This engine's own brightness math runs in a 0-1 "brightness"
        unit space (see PhonemeProfile/HarmonyDecision), not dB -- the
        conversion below is a deliberately simple, approximately-linear
        mapping (dB / 10.0) that matches the *10.0 scale
        vocable_synthesizer.py's _apply_phoneme_shaping already uses to
        turn a brightness delta into a shelf gain. It is not an exact
        inverse of that formula (which has a small dead zone between
        0.45-0.55 that this doesn't try to reproduce), so the actual
        rendered dB at baseline may land a fraction of a dB off from
        what's requested -- close enough for tuning by ear, which is the
        only thing these sliders are for.

        max_bright_db is the ABSOLUTE ceiling a full-rising lean reaches
        (matching the UI showing it as e.g. "+10dB", not a delta), so the
        up-swing is (max_bright_db - baseline_db). max_dark_db arrives as
        a positive MAGNITUDE below baseline (the UI shows it as e.g.
        "-4dB"), so the down-swing is (baseline_db + max_dark_db) -- the
        distance from baseline down to that absolute floor. Both clamped
        to >= 0 so an unusual slider combination (baseline set past the
        brighten ceiling, say) can't invert a rising or falling lean into
        pushing brightness the wrong way.

        vowel_color has no separate frontend control -- it mirrors
        brightness's own amounts directly, moving in lockstep with it
        (see decide()'s CONTOUR block for how the two combine).
        """
        self._contour_baseline_brightness_shift = baseline_db / 10.0
        self._contour_brightness_amount_up = max(0.0, (max_bright_db - baseline_db) / 10.0)
        self._contour_brightness_amount_down = max(0.0, (baseline_db + max_dark_db) / 10.0)
        self._contour_vowel_color_amount_up = self._contour_brightness_amount_up
        self._contour_vowel_color_amount_down = self._contour_brightness_amount_down
        logger.info(
            f"Contour brightness overridden: baseline={baseline_db:+.1f}dB, "
            f"max_brighten={max_bright_db:+.1f}dB, max_darken=-{max_dark_db:.1f}dB"
        )

    def set_forced_mode(self, mode_name: str | None):
        if mode_name is None:
            self._forced_mode = None
            logger.info("Forced accompaniment mode cleared")
            return
        try:
            self._forced_mode = AccompanimentMode(mode_name)
            logger.info(f"Accompaniment mode forced to: {mode_name}")
        except ValueError:
            logger.warning(f"Unknown accompaniment mode '{mode_name}'")

    def set_default_mode(self, mode_name: str):
        try:
            self.mode_selector.default_mode = AccompanimentMode(mode_name)
            logger.info(f"Default accompaniment mode set to {mode_name}")
        except ValueError:
            logger.warning(f"Unknown accompaniment mode '{mode_name}'")
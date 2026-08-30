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


# Output container
@dataclass
class HarmonyDecision:
    target_hz  : float              # the exact frequency the robot should sing
    vocable    : str                # the vowel shape: 'aah', 'ooo', 'mmm', 'hey'
    duration_s : float              # how long to hold the note in seconds
    vowel_color: float              # timbre darkness from Cree profile
    nasality   : float              # nasal resonance from Cree profile
    brightness : float              # spectral brightness from Cree profile
    action     : str                # either of 'sing', 'sustain', or 'rest'
    mode       : AccompanimentMode  # which accompaniment mode produced this decision
    mode_note  : str = ""           # human-readable explanation of what the mode is doing
    texture         : str   = "solo"  # "solo" | "duet" | "choir" — how many voices render this
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

    # Major and natural minor for scale-lock
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    # Loads config and sets up all the state
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

        # Vocable pools for _choose_vocable(). "common" is what gets used
        # almost all the time, "rare" is the special-occasion set, and
        # rare_chance is how often, per NEW NOTE (not per frame -- see the
        # locking logic in decide()), a rare one gets picked instead.
        synth_cfg = cfg.get("synthesis", {})
        self._vocable_common = synth_cfg.get("vocable_set", ["aah", "ooo", "mmm", "hey"])
        self._vocable_rare = synth_cfg.get("vocable_rare_set", [])
        self._vocable_rare_chance = float(synth_cfg.get("vocable_rare_chance", 0.0))
        self._locked_vocable: str | None = None

        # Rolling pitch history to infer key
        self._pitch_history: collections.deque = collections.deque(maxlen=32)
        self._current_key_root: int = 0
        self._current_scale: list = self.MAJOR_SCALE

        # Key inference cooldown — only re-infer key once per second (~43 frames)
        self._key_infer_counter: int = 0
        self._key_update_interval: int = 43

        # Track what the robot is currently doing
        self._current_decision: HarmonyDecision | None = None
        self._frames_on_current_note: int = 0
        self._max_frames_per_note: int = 200  # safety cap

        # Hysteresis for note-onset detection (see _resolve_action) -- a
        # pitch shift has to hold for 2 consecutive frames before it's
        # treated as a genuinely new note, so a single noisy/vibrato frame
        # can't retrigger the synthesizer mid-note.
        self._pending_note_hz: float | None = None
        self._pending_note_frames: int = 0

        # Legacy manual-interval override (see INTERVALS above). Off by default.
        self._manual_override = False
        self._current_interval: str = "fifth"

        # Forces a specific AccompanimentMode on every frame instead of
        # letting AccompanimentModeSelector auto-pick one -- used by the
        # offline whole-track render so a mode explicitly chosen in the
        # UI (e.g. "Octave reinforcement") is actually what gets sung,
        # instead of the selector silently overriding it based on tempo/
        # sustain heuristics. The protocol/sovereignty guard below is
        # NOT bypassed by this -- silence-when-sensitive always wins
        # regardless of what mode was forced.
        self._forced_mode: AccompanimentMode | None = None

        self._t0 = time.monotonic()

        logger.info(
            f"HarmonyEngine ready — default_mode={self.mode_selector.default_mode.value}, "
            f"protocol_enabled={self.protocol.enabled}, fusion_mode={self.mode_selector.fusion_mode}"
        )

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
            self._pitch_history.append(archer_hz)
            self._key_infer_counter += 1
            if self._key_infer_counter >= self._key_update_interval:
                self._key_infer_counter = 0
                self._infer_key()

        key_root_hz = self._key_root_hz()

        self.ctx.update(
            archer_hz=archer_hz,
            now_s=now_s,
            beat_duration_s=beat_duration_s,
            key_root_hz=key_root_hz,
            phrase_just_ended=phrase_just_ended,
        )

        is_protocol_sensitive = (
            phoneme_profile is not None
            and phoneme_profile.detected_class in self.protocol.sensitive_phoneme_classes
        )

        if self._manual_override:
            proposal = self._manual_proposal(archer_hz)
            mode = AccompanimentMode.UNISON
        elif self._forced_mode is not None:
            # Sovereignty guard still applies even with a forced mode --
            # protocol-off or a protocol-sensitive phoneme always wins
            # and forces silence, no matter what was explicitly chosen.
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

        # Note duration: however many beats this mode wants to hold, clipped to a sane max.
        duration = min(max(proposal.hold_beats, 0.25) * beat_duration_s, 4.0)

        action = self._resolve_action(proposal, target_hz)

        # Vocable is only re-rolled when a genuinely new note starts, and
        # held steady for every "sustain" frame of that same note. The old
        # per-frame reroll made sense back when a vocable was just a vowel
        # color (aah/ooo/mmm/hey) that could drift smoothly within a note.
        # Now that a vocable can be a whole pre-recorded word or phrase
        # (neural_wavetable engine), rerolling every frame would mean
        # switching words dozens of times a second mid-note, which is not
        # what you want. This is also what makes vocable_rare_chance mean
        # what it sounds like it means: a chance per NOTE, not per frame.
        if action == "sing":
            self._locked_vocable = self._choose_vocable(phoneme_profile)
        elif action == "rest":
            self._locked_vocable = None
        vocable = self._locked_vocable or self._choose_vocable(phoneme_profile)

        # Blend Cree phoneme influence into timbre params
        influence = getattr(phoneme_profile, "influence", 0.0) or 0.0
        vowel_color = 0.5 * (1 - influence) + phoneme_profile.vowel_color * influence
        nasality    = phoneme_profile.nasality * influence
        brightness  = 0.5 * (1 - influence) + phoneme_profile.brightness * influence

        texture = self._texture_override or DEFAULT_TEXTURE.get(mode, VoiceTexture.SOLO)
        layer = self.texture_params.get(texture)

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
            num_voices=layer.num_voices,
            detune_spread_cents=layer.detune_spread_cents,
            timing_jitter_ms=layer.timing_jitter_ms,
            formant_spread=layer.formant_spread,
            reverb_amount=layer.reverb_amount,
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

    def _finalize_pitch(self, proposal: ModeProposal, mode: AccompanimentMode) -> Optional[float]:
        target_hz = proposal.target_hz
        if target_hz is None or target_hz <= 0:
            return target_hz

        if self.scale_lock and mode != AccompanimentMode.DRONE:
            target_hz = self._snap_to_scale(target_hz)

        if self.detune_cents and mode not in (AccompanimentMode.UNISON, AccompanimentMode.DRONE):
            target_hz *= 2 ** (self.detune_cents / 1200.0)

        return target_hz

    # A genuinely new note is roughly a semitone or more away from the
    # currently-sustained one. Comparing in cents (not raw Hz) makes this
    # threshold mean the same thing at 100 Hz and at 800 Hz -- a fixed Hz
    # gap is a huge musical interval low down and a tiny sliver of a
    # semitone up high, so it fired inconsistently across a singer's range.
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

        # Require the pitch shift to show up on two consecutive frames
        # before treating it as a real new note, not one noisy frame of
        # vibrato/pitch-detector jitter. A single stray frame just resets
        # the pending counter instead of restarting the whole note --
        # every genuine held note, at any real singer's vibrato depth,
        # would otherwise get chopped into dozens of overlapping restarts
        # (this was the actual cause of the "broken and weird" output --
        # each restart re-synthesizes and additively stacks a full note
        # on top of the ones still ringing from a few frames earlier).
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
        if profile is not None and profile.influence >= 0.1:
            # Cree phoneme profile is actively steering timbre, keep it in
            # charge of vocable too. Note: this path still hands back one
            # of the four original vowel names (aah/ooo/mmm/hey) regardless
            # of what's in vocable_set/vocable_rare_set. That's an existing
            # rough edge, not something this change touches, it only
            # matters if cree_tokenizer.enabled gets turned on.
            if profile.vowel_color < 0.25:
                return "hey"
            elif profile.vowel_color < 0.5:
                return "aah"
            elif profile.vowel_color < 0.75:
                return "ooo"
            else:
                return "mmm"

        if self._vocable_rare and self._vocable_rare_chance > 0 and random.random() < self._vocable_rare_chance:
            return random.choice(self._vocable_rare)

        if not self._vocable_common:
            return "aah"
        idx = len(self._pitch_history) % len(self._vocable_common)
        return self._vocable_common[idx]

    def set_interval(self, interval: str):
        if interval in self.INTERVALS:
            self._current_interval = interval
            self._manual_override = True
            logger.info(f"Manual interval override engaged: {interval} "
                        f"(mode-driven accompaniment is now bypassed — "
                        f"call clear_manual_override() to return to it)")
        else:
            logger.warning(f"Unknown interval '{interval}'. Options: {list(self.INTERVALS)}")

    def clear_manual_override(self):
        self._manual_override = False
        logger.info("Manual interval override cleared — back to mode-driven accompaniment")

    def set_fusion_mode(self, enabled: bool):
        self.mode_selector.fusion_mode = bool(enabled)
        logger.info(f"Fusion mode {'ENABLED (triadic harmony)' if enabled else 'DISABLED'}")

    def set_texture(self, texture_name: str):
        try:
            self._texture_override = VoiceTexture(texture_name)
            logger.info(f"Voice texture pinned to: {texture_name}")
        except ValueError:
            logger.warning(f"Unknown texture '{texture_name}'. Options: {[t.value for t in VoiceTexture]}")

    def clear_texture_override(self):
        self._texture_override = None
        logger.info("Voice texture back to per-mode defaults")

    def set_forced_mode(self, mode_name: str | None):
        if mode_name is None:
            self._forced_mode = None
            logger.info("Forced accompaniment mode cleared — back to automatic selection")
            return
        try:
            self._forced_mode = AccompanimentMode(mode_name)
            logger.info(f"Accompaniment mode forced to: {mode_name} "
                        f"(auto-selection bypassed; protocol/silence guard still applies)")
        except ValueError:
            logger.warning(f"Unknown accompaniment mode '{mode_name}' — ignoring forced-mode request")

    def set_default_mode(self, mode_name: str):
        try:
            self.mode_selector.default_mode = AccompanimentMode(mode_name)
            logger.info(f"Default accompaniment mode set to {mode_name}")
        except ValueError:
            logger.warning(f"Unknown accompaniment mode '{mode_name}'")
import time
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
    # Legacy fixed-interval semitone offsets, kept only for the CLI
    # `--interval` flag / `set_interval()` so existing scripts and tests that
    # ask for a specific fixed interval keep working. The primary, default
    # behaviour of the engine is now driven entirely by the accompaniment
    # mode system below (AccompanimentModeSelector + ModeFunctions), which is
    # what actually decides *how* the robot accompanies Archer from moment to
    # moment (shadow him, echo him, drone under him, answer him, etc.)
    # instead of always applying one static interval.
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

        # Sovereignty / protocol guard — this is checked before any mode
        # logic runs. When the protocol is disabled, or the current phoneme
        # is flagged as protocol-sensitive, the robot goes fully silent no
        # matter what mode is selected.
        self.protocol = ProtocolGuard(cfg)

        # Accompaniment mode machinery — decides *which* musical behaviour
        # (unison, octave, delayed echo, timbral thickening, contour
        # following, drone, call-and-response, or fusion-only triadic
        # harmony) the robot uses at any given moment, and computes the
        # actual proposed pitch/action for that behaviour.
        self.mode_selector = AccompanimentModeSelector(cfg)
        self.mode_functions = ModeFunctions(cfg)
        self.ctx = MusicalContext()
        self._fusion_third_semitones = harmony_cfg.get("fusion_third_semitones", 4)

        # Voice texture (solo / duet / choir) — each mode has a sensible
        # default (see DEFAULT_TEXTURE), but it can be pinned explicitly via
        # set_texture(), e.g. so a user can force "concert choir" on demand
        # regardless of which accompaniment mode is currently selected.
        self.texture_params = TextureParams(cfg)
        self._texture_override: Optional[VoiceTexture] = None

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

        # Legacy manual-interval override (see INTERVALS above). Off by default.
        self._manual_override = False
        self._current_interval: str = "fifth"

        self._t0 = time.monotonic()

        logger.info(
            f"HarmonyEngine ready — default_mode={self.mode_selector.default_mode.value}, "
            f"protocol_enabled={self.protocol.enabled}, fusion_mode={self.mode_selector.fusion_mode}"
        )

    # Given the current state of Archer's singing, decide what the robot does
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

        # Update pitch history / key inference only while Archer is actually
        # singing a detected pitch. This intentionally happens *before* the
        # mode is chosen, and regardless of phrase_state, so that modes like
        # call_and_response and drone_support — which specifically care about
        # what happens once Archer goes quiet — still have pitch history and
        # a key to work with.
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

        # Sovereignty check + mode selection happen for every frame, even
        # silent ones — this is what lets drone_support and
        # call_and_response actually trigger once Archer stops singing,
        # instead of the engine going straight to a hardcoded rest.
        is_protocol_sensitive = (
            phoneme_profile is not None
            and phoneme_profile.detected_class in self.protocol.sensitive_phoneme_classes
        )

        if self._manual_override:
            proposal = self._manual_proposal(archer_hz)
            mode = AccompanimentMode.UNISON
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

        # Choose vocable based on phoneme profile
        vocable = self._choose_vocable(phoneme_profile)

        # Blend Cree phoneme influence into timbre params
        influence = getattr(phoneme_profile, "influence", 0.0) or 0.0
        vowel_color = 0.5 * (1 - influence) + phoneme_profile.vowel_color * influence
        nasality    = phoneme_profile.nasality * influence
        brightness  = 0.5 * (1 - influence) + phoneme_profile.brightness * influence

        # Note duration: however many beats this mode wants to hold, clipped to a sane max.
        duration = min(max(proposal.hold_beats, 0.25) * beat_duration_s, 4.0)

        action = self._resolve_action(proposal, target_hz)

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

    # ------------------------------------------------------------------ #
    # Mode dispatch
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Pitch finalization: scale-lock + global detune
    # ------------------------------------------------------------------ #

    def _finalize_pitch(self, proposal: ModeProposal, mode: AccompanimentMode) -> Optional[float]:
        target_hz = proposal.target_hz
        if target_hz is None or target_hz <= 0:
            return target_hz

        # Drone support intentionally sits outside the melodic scale (it's a
        # grounding root, not a harmony note), so it skips scale-lock.
        if self.scale_lock and mode != AccompanimentMode.DRONE:
            target_hz = self._snap_to_scale(target_hz)

        # Unison should match Archer's pitch exactly rather than being
        # nudged by a chorus-style detune.
        if self.detune_cents and mode not in (AccompanimentMode.UNISON, AccompanimentMode.DRONE):
            target_hz *= 2 ** (self.detune_cents / 1200.0)

        return target_hz

    # ------------------------------------------------------------------ #
    # Sing / sustain / rest resolution
    # ------------------------------------------------------------------ #

    def _resolve_action(self, proposal: ModeProposal, target_hz: Optional[float]) -> str:
        if proposal.action == "rest" or target_hz is None or target_hz <= 0:
            self._frames_on_current_note = 0
            return "rest"

        is_new_note = (
            self._current_decision is None
            or self._current_decision.action == "rest"
            or abs(target_hz - self._current_decision.target_hz) > 5.0
            or self._frames_on_current_note >= self._max_frames_per_note
        )

        if is_new_note:
            self._frames_on_current_note = 0
            return "sing"

        self._frames_on_current_note += 1
        return "sustain"

    # ------------------------------------------------------------------ #
    # Scale / key helpers
    # ------------------------------------------------------------------ #

    # Snaps a frequency to the nearest note in the currently detected scale
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

    # Converts the inferred key root (a 0-11 pitch class) into an actual
    # Hz value near Archer's current register, for use as a drone root.
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

    # Figures out what key Archer is singing in — runs at most once per second
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

    # Picks which vowel shape the robot should use based on the Cree phoneme profile
    def _choose_vocable(self, profile: PhonemeProfile) -> str:
        if profile is None or profile.influence < 0.1:
            defaults = ["aah", "ooo", "mmm", "hey"]
            idx = len(self._pitch_history) % len(defaults)
            return defaults[idx]

        if profile.vowel_color < 0.25:
            return "hey"
        elif profile.vowel_color < 0.5:
            return "aah"
        elif profile.vowel_color < 0.75:
            return "ooo"
        else:
            return "mmm"

    # ------------------------------------------------------------------ #
    # External controls (used by server.py / main.py)
    # ------------------------------------------------------------------ #

    # Legacy: force a fixed interval instead of mode-driven behaviour.
    # Mostly here for the CLI (`main.py --interval fifth`) and old tests.
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

    # Explicit opt-in for triadic harmony (contemporary/fusion only — never the default)
    def set_fusion_mode(self, enabled: bool):
        self.mode_selector.fusion_mode = bool(enabled)
        logger.info(f"Fusion mode {'ENABLED (triadic harmony)' if enabled else 'DISABLED'}")

    # Force a specific voice texture (solo/duet/choir) regardless of which
    # accompaniment mode is active — e.g. a "concert" button in the UI that
    # blooms whatever's currently playing into a full choir.
    def set_texture(self, texture_name: str):
        try:
            self._texture_override = VoiceTexture(texture_name)
            logger.info(f"Voice texture pinned to: {texture_name}")
        except ValueError:
            logger.warning(f"Unknown texture '{texture_name}'. Options: {[t.value for t in VoiceTexture]}")

    def clear_texture_override(self):
        self._texture_override = None
        logger.info("Voice texture back to per-mode defaults")

    def set_default_mode(self, mode_name: str):
        try:
            self.mode_selector.default_mode = AccompanimentMode(mode_name)
            logger.info(f"Default accompaniment mode set to {mode_name}")
        except ValueError:
            logger.warning(f"Unknown accompaniment mode '{mode_name}'")
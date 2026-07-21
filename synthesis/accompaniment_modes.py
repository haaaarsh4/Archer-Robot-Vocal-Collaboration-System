from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AccompanimentMode(str, Enum):
    UNISON        = "unison_shadowing"
    OCTAVE        = "octave_reinforcement"
    DELAYED       = "delayed_response"
    TIMBRAL       = "timbral_thickening"
    CONTOUR       = "contour_following"
    DRONE         = "drone_support"
    CALL_RESPONSE = "call_and_response"
    TRIADIC       = "triadic_harmony"      # fusion-only — never selected by default
    SILENT        = "protocol_silence"


@dataclass
class ModeProposal:
    mode: AccompanimentMode
    target_hz: Optional[float]
    action: str                 # "sing" | "sustain" | "rest"
    hold_beats: float = 1.0     # how many beats to hold this note
    note: str = ""              


class MusicalContext:

    def __init__(self, sample_history_len: int = 128):
        self.archer_hz: Optional[float] = None
        self.key_root_hz: Optional[float] = None
        self._timed_pitches: collections.deque = collections.deque(maxlen=sample_history_len)
        self._phrase_final_pitches: collections.deque = collections.deque(maxlen=8)
        self._beat_duration_s: float = 0.5
        self._last_voiced_hz: Optional[float] = None

    def update(self, *, archer_hz: Optional[float], now_s: float,
               beat_duration_s: float, key_root_hz: Optional[float],
               phrase_just_ended: bool) -> None:
        self.archer_hz = archer_hz
        self.key_root_hz = key_root_hz
        self._beat_duration_s = beat_duration_s if beat_duration_s > 0 else 0.5
        if archer_hz:
            self._timed_pitches.append((now_s, archer_hz))
            self._last_voiced_hz = archer_hz
        # phrase_just_ended fires on the frame Archer goes silent, so
        # archer_hz is typically None right here — use the last pitch he
        # was actually singing, not the (empty) current one.
        if phrase_just_ended and self._last_voiced_hz is not None:
            self._phrase_final_pitches.append(self._last_voiced_hz)

    def pitch_at_delay(self, beats: float) -> Optional[float]:
        if not self._timed_pitches:
            return None
        now = self._timed_pitches[-1][0]
        target_time = now - beats * self._beat_duration_s
        best_hz, best_diff = None, float("inf")
        for t, hz in self._timed_pitches:
            diff = abs(t - target_time)
            if diff < best_diff:
                best_diff, best_hz = diff, hz
        return best_hz

    def pitch_direction(self, window: int) -> int:
        pitches = [hz for _, hz in list(self._timed_pitches)[-window:]]
        if len(pitches) < 2:
            return 0
        delta = pitches[-1] - pitches[0]
        if abs(delta) < 1e-6:
            return 0
        return 1 if delta > 0 else -1

    def last_phrase_final_pitch(self) -> Optional[float]:
        return self._phrase_final_pitches[-1] if self._phrase_final_pitches else None


class ModeFunctions:

    def __init__(self, cfg: dict):
        mcfg = cfg.get("modes", {})
        self.octave_direction    = mcfg.get("octave_direction", -1)     # -1 below, +1 above
        self.delay_beats         = mcfg.get("delay_beats", 1.0)
        self.drone_divisor       = mcfg.get("drone_divisor", 2)         # 2 = sub-octave
        self.contour_window      = mcfg.get("contour_window", 6)
        self.contour_step_semis  = mcfg.get("contour_step_semitones", 2)
        self.call_response_semis = mcfg.get("call_response_semitones", 0)

    def unison(self, ctx: MusicalContext) -> ModeProposal:
        return ModeProposal(AccompanimentMode.UNISON, ctx.archer_hz, "sing",
                             note="matching Archer's pitch exactly")

    def octave(self, ctx: MusicalContext) -> ModeProposal:
        target = ctx.archer_hz * (2.0 ** self.octave_direction)
        return ModeProposal(AccompanimentMode.OCTAVE, target, "sing",
                             note=f"octave {'below' if self.octave_direction < 0 else 'above'}")

    def delayed(self, ctx: MusicalContext) -> ModeProposal:
        target = ctx.pitch_at_delay(self.delay_beats)
        if target is None:
            return ModeProposal(AccompanimentMode.DELAYED, None, "rest",
                                 note="not enough pitch history yet for a delayed echo")
        return ModeProposal(AccompanimentMode.DELAYED, target, "sing",
                             hold_beats=self.delay_beats,
                             note=f"echoing what Archer sang {self.delay_beats} beat(s) ago")

    def timbral(self, ctx: MusicalContext) -> ModeProposal:
        return ModeProposal(AccompanimentMode.TIMBRAL, ctx.archer_hz, "sing",
                             note="same pitch, different timbre — thickens rather than harmonizes")

    def contour(self, ctx: MusicalContext) -> ModeProposal:
        direction = ctx.pitch_direction(self.contour_window)
        target = ctx.archer_hz if direction == 0 else (
            ctx.archer_hz * (2 ** (direction * self.contour_step_semis / 12.0))
        )
        shape = "rising" if direction > 0 else "falling" if direction < 0 else "level"
        return ModeProposal(AccompanimentMode.CONTOUR, target, "sing",
                             note=f"tracking melodic shape ({shape})")

    def drone(self, ctx: MusicalContext) -> ModeProposal:
        root = ctx.key_root_hz or ctx.archer_hz
        if root is None:
            return ModeProposal(AccompanimentMode.DRONE, None, "rest",
                                 note="no pitch reference yet for a drone root")
        target = root / self.drone_divisor
        return ModeProposal(AccompanimentMode.DRONE, target, "sustain",
                             hold_beats=4.0, note="grounding drone under the phrase")

    def call_response(self, ctx: MusicalContext) -> ModeProposal:
        last_pitch = ctx.last_phrase_final_pitch()
        if last_pitch is None:
            return ModeProposal(AccompanimentMode.CALL_RESPONSE, None, "rest",
                                 note="no completed phrase to respond to yet")
        target = last_pitch * (2 ** (self.call_response_semis / 12.0))
        return ModeProposal(AccompanimentMode.CALL_RESPONSE, target, "sing",
                             hold_beats=2.0, note="answering Archer's phrase after he finishes")

    def triadic(self, ctx: MusicalContext, third_semitones: int) -> ModeProposal:
        target = ctx.archer_hz * (2 ** (third_semitones / 12.0))
        return ModeProposal(AccompanimentMode.TRIADIC, target, "sing",
                             note="fusion-mode triadic harmony (explicit opt-in only)")

    def silent(self, reason: str) -> ModeProposal:
        return ModeProposal(AccompanimentMode.SILENT, None, "rest", note=reason)


class AccompanimentModeSelector:
    def __init__(self, cfg: dict):
        hcfg = cfg.get("harmony", {})
        self.fusion_mode = bool(hcfg.get("fusion_mode", False))
        self.default_mode = AccompanimentMode(hcfg.get("default_mode", AccompanimentMode.UNISON.value))
        self.fast_tempo_bpm = hcfg.get("fast_tempo_threshold_bpm", 120)
        self.sustain_frames_for_timbral = hcfg.get("sustain_frames_for_timbral", 30)
        self.silence_when_no_singer = bool(hcfg.get("silence_when_no_singer", False))

    def select(self, *, protocol_enabled: bool, is_protocol_sensitive: bool,
               archer_hz: Optional[float], phrase_state: str,
               phrase_just_ended: bool, tempo_bpm: float,
               frames_on_current_note: int) -> AccompanimentMode:

        if not protocol_enabled:
            return AccompanimentMode.SILENT

        if is_protocol_sensitive:
            return AccompanimentMode.SILENT

        if archer_hz is None:
            if phrase_just_ended or phrase_state == "phrase_end":
                return AccompanimentMode.CALL_RESPONSE
            return AccompanimentMode.SILENT if self.silence_when_no_singer else AccompanimentMode.DRONE

        if self.fusion_mode:
            return AccompanimentMode.TRIADIC

        if tempo_bpm >= self.fast_tempo_bpm:
            return AccompanimentMode.CONTOUR

        if frames_on_current_note >= self.sustain_frames_for_timbral:
            return AccompanimentMode.TIMBRAL

        return self.default_mode
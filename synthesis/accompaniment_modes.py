from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import librosa

from synthesis.call_response import CallResponseCfg, LiveCallResponder


class AccompanimentMode(str, Enum):
    UNISON        = "unison_shadowing"
    OCTAVE        = "octave_reinforcement"
    DELAYED       = "delayed_response"
    TIMBRAL       = "timbral_thickening"
    CONTOUR       = "contour_following"
    DRONE         = "drone_support"
    CALL_RESPONSE = "call_and_response"
    TRIADIC       = "triadic_harmony"      # fusion-only: never selected by default
    HUM           = "solo_humming"         # one voice, closed-mouth, barely-there
    SILENT        = "protocol_silence"


class VoiceTexture(str, Enum):
    SOLO   = "solo"     # a single voice
    DUET   = "duet"     # two voices, small detune/timing spread
    CHOIR  = "choir"    # many voices, the "concert" feel


DEFAULT_TEXTURE = {
    AccompanimentMode.UNISON:        VoiceTexture.CHOIR,
    AccompanimentMode.OCTAVE:        VoiceTexture.DUET,
    AccompanimentMode.DELAYED:       VoiceTexture.SOLO,
    AccompanimentMode.TIMBRAL:       VoiceTexture.DUET,
    # Was SOLO. A single dry voice made the (now continuous, smoothed)
    # contour swing hard to hear underneath Archer's own voice, a light
    # duet gives it a touch of width/detune so the lean away from the
    # lead actually registers as a second voice doing something, without
    # going as far as choir (that's the "concert" texture, not this).
    # Easy to revert to VoiceTexture.SOLO if you want it drier.
    AccompanimentMode.CONTOUR:       VoiceTexture.DUET,
    AccompanimentMode.DRONE:         VoiceTexture.DUET,
    # A solo answering voice, : the answer is one other singer (or
    # one small group, if Duet/Choir is picked in the panel) answering Archer,
    # not a section doubling him. The mode's own contrast comes from where the
    # answer sits and how it is phrased, see synthesis/call_response.py.
    AccompanimentMode.CALL_RESPONSE: VoiceTexture.SOLO,
    AccompanimentMode.TRIADIC:       VoiceTexture.CHOIR,
    AccompanimentMode.HUM:           VoiceTexture.SOLO,
    AccompanimentMode.SILENT:        VoiceTexture.SOLO,
}


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
        self.octave_shift_semitones: float = 0.0   # ADD THIS
        # DID A NEW PHRASE JUST START ON THIS FRAME? A drone's held note
        # belongs to a phrase (see ModeFunctions.drone), so it has to be able to
        # tell "the singer is still going" from "the singer has come back after
        # a breath", and it has to be able to do that from the pitch timeline
        # alone: the same test has to work live, where there is no track to look
        # at, and in an offline render, where there is. A gap in the pitched
        # frames longer than phrase_gap_s is that test. Set from config by
        # HarmonyEngine.
        self.phrase_started: bool = False
        self.phrase_gap_s: float = 0.35
        self._last_pitch_time: Optional[float] = None
        # The clock, and whether the phrase ended on this very frame. Both are
        # here because call and response is the one mode that has to know WHEN
        # it is: it schedules an answer phrase over the seconds after the singer
        # stops, rather than reacting to the current frame alone. See
        # synthesis/call_response.py.
        self.now_s: float = 0.0
        self.phrase_just_ended: bool = False

    def update(self, *, archer_hz: Optional[float], now_s: float,
            beat_duration_s: float, key_root_hz: Optional[float],
            phrase_just_ended: bool,
            octave_shift_semitones: float = 0.0) -> None:   # ADD THIS PARAMETER
        self.archer_hz = archer_hz
        self.key_root_hz = key_root_hz
        self.now_s = now_s
        self.phrase_just_ended = phrase_just_ended
        self._beat_duration_s = beat_duration_s if beat_duration_s > 0 else 0.5
        self.octave_shift_semitones = octave_shift_semitones   # STORE IT
        self.phrase_started = False
        if archer_hz:
            # A gap longer than phrase_gap_s between two pitched frames means
            # the singer stopped and came back: that frame opens a phrase.
            gap = None if self._last_pitch_time is None else (now_s - self._last_pitch_time)
            self.phrase_started = (gap is None) or (gap > self.phrase_gap_s)
            self._last_pitch_time = now_s
            self._timed_pitches.append((now_s, archer_hz))
            self._last_voiced_hz = archer_hz
        elif (self._last_pitch_time is not None
                and now_s - self._last_pitch_time > self.phrase_gap_s):
            # Enough silence that the next pitched frame is a new phrase rather
            # than a continuation of this one.
            self._last_pitch_time = None
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

    def pitch_trend_semitones(self, lookback_s: float) -> float:
        """
        How far, and in which direction, the sung pitch has moved over
        the last `lookback_s` SECONDS, in semitones, not raw Hz and
        not raw sample count.

        Two fixes over the original version: (1) pitch is perceived
        logarithmically, so this compares log2(hz) rather than raw Hz
        a 5Hz rise at 200Hz is a much bigger interval than the same 5Hz
        rise at 800Hz. (2) it windows by TIME, not by a fixed number of
        samples. decide() is called once per audio frame (~23ms at the
        default 1024-sample/44.1kHz frame size), so a 6-SAMPLE window
        only covers ~140ms, far shorter than an actual musical phrase,
        which is why a frame-count window can never see real melodic
        contour, only frame-to-frame jitter. Windowing by seconds means
        this keeps meaning the same thing regardless of frame rate.

        Returns the fitted total semitone change across the window
        (least-squares slope * time span), signed: positive = rising,
        negative = falling. A one-off noisy reading can't flip the sign
        on its own since the whole window is fit, not just its endpoints.
        """
        if not self._timed_pitches:
            return 0.0
        now = self._timed_pitches[-1][0]
        recent = [(t, hz) for t, hz in self._timed_pitches if now - t <= lookback_s]
        if len(recent) < 2:
            return 0.0
        times = np.array([t for t, _ in recent], dtype=np.float64)
        hz = np.array([h for _, h in recent], dtype=np.float64)
        if np.any(hz <= 0):
            return 0.0
        semis = 12.0 * np.log2(hz / hz[0])
        span = times[-1] - times[0]
        if span <= 1e-6:
            return 0.0
        slope = np.polyfit(times, semis, 1)[0]  # semitones per second
        return float(slope * span)

    def last_phrase_final_pitch(self) -> Optional[float]:
        return self._phrase_final_pitches[-1] if self._phrase_final_pitches else None


class ModeFunctions:

    def __init__(self, cfg: dict):
        mcfg = cfg.get("modes", {})
        self.octave_semitones = float(mcfg.get("octave_semitones", -12.0))  # continuous, live-adjustable
                                                                              # was "octave_direction", a fixed
                                                                              # -1/+1 exponent meaning exactly one
                                                                              # whole octave with no in-between.
                                                                              # -12 reproduces the old default
                                                                              # (one octave below).
        self.delay_beats         = mcfg.get("delay_beats", 1.0)

        # === drone_support ===
        # Completely redesigned below (see drone()). The old version
        # recomputed the drone's pitch from scratch on EVERY decide()
        # call, straight off whatever octave Archer's most recent single
        # note happened to report, so crossing something as small as a
        # B3->C4 boundary mid-phrase (a semitone of real movement) could
        # flip the drone's whole register by a full octave, and every
        # such flip got picked up by the note-onset logic as a brand new
        # note (a fresh attack + a re-randomized vocable), because it
        # moved by far more than the 70-cent "new note" threshold. That's
        # the "weird", glitchy, randomly-restarting quality, a drone
        # that's supposed to be one continuous grounding tone was
        # actually retriggering itself constantly.
        #
        # A real drone should: (1) lock onto the tonic, (2) sit in a
        # genuinely low, comfortable "grounding" register, not simply
        # "one octave below wherever Archer's last note was", which
        # chases him around, and (3) stay rock-still there for as long
        # as the key doesn't change, ignoring Archer's own note-to-note
        # motion entirely. See drone()'s hysteresis lock for how that's
        # enforced now.
        #
        # Register calibration (revised): two octaves below the tonic,
        # clamped as low as 55Hz, put the lock in genuine sub-bass
        # AT or BELOW a typical singer's lowest note, not comfortably
        # under it. This is a SUNG voice being asked to render there
        # (formant/neural vocal synthesis, not a bass instrument), and
        # pushed that far below its natural register it comes out
        # unnaturally deep and distorted-sounding rather than warm and
        # grounding, a foghorn, not a held "ohh". A real vocal
        # pedal-tone/drone (gospel "oohs" under a lead, a tanpura-style
        # backing) sits about ONE octave below the melody's own
        # register at most, still clearly low, but still a voice.
        legacy_divisor = mcfg.get("drone_divisor", None)
        default_octaves_below = float(np.log2(legacy_divisor)) if legacy_divisor else 1.0
        self.drone_octaves_below = float(mcfg.get("drone_octaves_below", default_octaves_below))
        # Minimum time between two pitched frames before the second one counts
        # as the start of a NEW phrase rather than the same one continuing. This
        # is what gives the drone its one-held-note-per-phrase shape.
        self.drone_phrase_gap_s = float(mcfg.get("drone_phrase_gap_s", 0.35))
        # How long a drone is asked to hold. The offline renderer extends a
        # sustain to the actual length of the phrase, so this is really only the
        # live engine's own top-up interval, kept large because a pedal tone is
        # by definition the thing that does not re-articulate.
        self.drone_hold_beats = float(mcfg.get("drone_hold_beats", 8.0))
        # A pedal sits UNDER a lead voice; it is support, not a duet partner on
        # the same footing. A small, fixed trim (plus the mode's own reverb) is
        # what keeps a note held for a whole phrase from reading as the loudest
        # thing on the panel, which is how the old version came across.
        self.drone_level_trim_db = float(mcfg.get("drone_level_trim_db", 2.5))

        # === octave_reinforcement's three singing fixes ===
        # Singing a whole line an octave below yourself does not sound like a
        # second singer unless you do what a singer actually does down there.
        # The three things are the three things every voice teacher gives for
        # low notes (bright vowel, a narrower vowel, less volume as you go
        # lower), and they are implemented here as timbre and level on the
        # octave voice only, so the mode's pitch behaviour is unchanged:
        #   * bright  -- a high-shelf lift, which is what stops a note an octave
        #                down from reading as "the same thing, but dark and
        #                muddy". The octave voice has half the fundamental and
        #                keeps its own vocal tract, so its energy sits lower and
        #                needs the presence put back .
        #   * narrow  -- the vowel colour is pulled toward the bright/front end,
        #                which slides the voice's formant peak up (see
        #                VocableSynthesizer._apply_phoneme_shaping), the same
        #                thing a singer does by making the mouth a little
        #                narrower on a low note.
        #   * quieter -- a level trim that grows with how far below the phrase
        #                the harmony sits, because the bottom of anybody's range
        #                is where you allow a little volume out rather than
        #                pushing.
        self.octave_low_bright_amount = float(mcfg.get("octave_low_bright_amount", 0.30))
        self.octave_low_narrow_amount = float(mcfg.get("octave_low_narrow_amount", 0.22))
        self.octave_low_trim_db = float(mcfg.get("octave_low_trim_db", 3.0))
        # Extra trim for notes that are low in absolute terms, not just low
        # relative to the singer: the very bottom of the range is where the
        # video's third tip lives. 0 dB at floor_hz and up to the extra amount
        # an octave below it.
        self.octave_low_floor_hz = float(mcfg.get("octave_low_floor_hz", 170.0))
        self.octave_low_floor_extra_db = float(mcfg.get("octave_low_floor_extra_db", 2.0))
        # Comfortable low-vocal floor/ceiling the locked drone pitch is
        # clamped into, in Hz, roughly F#2 to G3, a register most
        # voices (and most vocal synthesis/neural voice models) can
        # actually render cleanly without heavy pitch-shift artifacts.
        # Keeps a lock-in moment from landing somewhere absurdly low/high
        # just because of whatever octave Archer's voice happened to be
        # in at that instant.
        # 70 Hz (D2) is a hard RENDERABILITY floor, not a musical choice: below
        # it a sung voice (and the neural bank's formant-preserving shift in
        # particular) stops sounding like a person at all. It used to be 90 Hz
        # with a rule that pushed the drone UP an octave to stay above the
        # floor, and that rule is what quietly turned a drone on a low voice
        # into a unison with the singer -- the one thing a drone must never be.
        # A low singer now gets a low pedal instead, which is what a real one
        # gets too.
        self.drone_min_hz = float(mcfg.get("drone_min_hz", 70.0))
        # C4, the highest a pedal may sit: a fourth under a line is the closest
        # that still reads as support rather than as a second lead, and the
        # window rule above keeps the pedal at or below that interval anyway.
        self.drone_max_hz = float(mcfg.get("drone_max_hz", 262.0))
        self._drone_locked_hz: float | None = None
        self._drone_note_class: int | None = None

        # === contour_following ===
        # Redesigned as a hysteresis state machine, not a continuous
        # lean: while Archer holds a note, the shadow voice sits in
        # unison with him; the moment his pitch is genuinely moving, it
        # glides up to a full harmony step above (rising) or below
        # (falling) and holds there until the motion settles, then
        # glides back. That's an always-audible, deliberate behavior
        # not a fraction-of-a-semitone wobble that scale-snapping can
        # fold right back onto the unison note.
        self.contour_lookback_s = float(mcfg.get("contour_lookback_seconds", 0.6))
        # Trend magnitude (semitones of real movement across the lookback
        # window) needed to call the melody "moving" and commit to a step.
        self.contour_move_threshold_semis = float(mcfg.get("contour_move_threshold_semitones", 1.5))
        # Trend magnitude below which it's considered "settled" again and
        # glides back to unison. lower than the move
        # threshold (hysteresis) so it doesn't flicker in and out right
        # at the boundary.
        self.contour_release_threshold_semis = float(mcfg.get("contour_release_threshold_semitones", 0.5))
        # contour_step_semitones is the OLD single-value key, now used
        # as the "normal" harmony step size once movement is detected
        # (default a whole tone).
        self.contour_step_semis = float(mcfg.get("contour_step_semitones", 2.0))
        # A bigger, more urgent melodic run escalates to a wider step.
        self.contour_fast_threshold_semis = float(mcfg.get("contour_fast_threshold_semitones", 4.0))
        self.contour_fast_step_semis = float(mcfg.get("contour_fast_step_semitones", 4.0))
        # One-pole glide factor (0-1) applied each decide() call when
        # easing toward the current target offset, this is what makes
        # the transition a swoop into/out of harmony rather than a snap.
        self.contour_glide_alpha = float(mcfg.get("contour_glide_alpha", 0.22))
        self._contour_offset_smoothed = 0.0
        self._contour_state = "unison"   # "unison" | "up" | "down"

        self.contour_window      = mcfg.get("contour_window", 6)  # no longer used by contour(); kept only so old configs don't error on load

        # === call_and_response ===
        # The whole mode lives in synthesis/call_response.py now. What used to
        # be here (one arithmetic step: the phrase's last pitch pulled 65% of
        # the way to an inferred tonic and held for two beats) was deleted
        # outright, not tuned: a single held note is not an answer, and no
        # amount of adjustment makes it one. The live responder below composes
        # a real answering phrase - same rhythm as the call, arriving on the
        # tonic by step - and the offline renderer composes a better one from
        # the whole track at once (server.py calls into the same module).
        self._live_responder = LiveCallResponder(CallResponseCfg.from_config(cfg))

        # === contour_following's continuous timbral signature ===
        # The pitch step above only engages once real melodic movement
        # crosses contour_move_threshold_semis, it HAS to work this way
        # under scale_lock (see HarmonyEngine._snap_to_scale), which
        # rounds pitch to the nearest scale degree and would erase
        # anything subtler than a full step anyway. The consequence:
        # whenever a phrase holds fairly still (a short, mostly-spoken
        # vocal take is often mostly this), contour_following's target
        # pitch is EXACTLY archer_hz, byte-for-byte the same output as
        # unison_shadowing, with nothing else distinguishing it.
        #
        # Brightness/vowel_color are NOT touched by scale_lock, they're
        # timbre, not pitch, so they can afford to react continuously
        # and immediately to melodic direction, at any magnitude, instead
        # of waiting for a threshold crossing. That's what makes this
        # mode's identity permanent rather than intermittent: the voice
        # audibly brightens climbing into a rising phrase and warms/
        # darkens easing into a falling one, tracking melodic SHAPE
        # through timbre continuously, on top of (not instead of) the
        # discrete harmonic step for genuinely large movement. See
        # _contour_timbre_lean below and its use in HarmonyEngine.decide.
        self.contour_trend_scale_semis = float(mcfg.get("contour_trend_scale_semitones", 1.2))
        self.contour_timbre_glide_alpha = float(mcfg.get("contour_timbre_glide_alpha", 0.5))
        self._contour_timbre_lean = 0.0   # -1.0 (falling) .. 0.0 (level) .. +1.0 (rising), continuous

    def unison(self, ctx: MusicalContext) -> ModeProposal:
        return ModeProposal(AccompanimentMode.UNISON, ctx.archer_hz, "sing",
                             note="matching Archer's pitch exactly")

    def octave(self, ctx: MusicalContext) -> ModeProposal:
        if ctx.archer_hz is None:
            return ModeProposal(AccompanimentMode.OCTAVE, None, "rest",
                                note="no pitch to shift octave")
        target = ctx.archer_hz * (2.0 ** (self.octave_semitones / 12.0))
        return ModeProposal(AccompanimentMode.OCTAVE, target, "sing",
                            note=f"{abs(self.octave_semitones):.0f} semitones {'below' if self.octave_semitones < 0 else 'above'}")

    def delayed(self, ctx: MusicalContext) -> ModeProposal:
        # This only decides WHICH pitch gets sung late (what Archer sang
        # delay_beats ago), a single note, once. The actual repeating,
        # decaying echo on top of that (multiple audible taps, each
        # quieter and a little darker than the last) is a separate
        # whole-track effect applied afterward, see apply_echo_effect
        # in server.py, driven by modes.delay_echo_ms /
        # modes.delay_echo_feedback below (or the frontend's Delay/Echo
        # sliders, when the render request includes them). Keeping the
        # repeat/decay effect out of the per-note decision here is
        # deliberate: a real delay pedal doesn't know or care what note
        # is playing, it just repeats whatever came out the other end.
        target = ctx.pitch_at_delay(self.delay_beats)
        if target is None:
            return ModeProposal(AccompanimentMode.DELAYED, None, "rest",
                                 note="not enough pitch history yet for a delayed echo")
        return ModeProposal(AccompanimentMode.DELAYED, target, "sing",
                             hold_beats=self.delay_beats,
                             note=f"echoing what Archer sang {self.delay_beats} beat(s) ago")

    def timbral(self, ctx: MusicalContext) -> ModeProposal:
        return ModeProposal(AccompanimentMode.TIMBRAL, ctx.archer_hz, "sing",
                             note="same pitch, different timbre, thickens rather than harmonizes")

    def contour(self, ctx: MusicalContext) -> ModeProposal:
        if ctx.archer_hz is None:
            # No pitch to react to, ease back toward unison (both the
            # discrete step AND the continuous timbral lean) rather than
            # holding whatever state things were in when the singer
            # dropped out.
            self._contour_offset_smoothed *= (1.0 - self.contour_glide_alpha)
            self._contour_timbre_lean *= (1.0 - self.contour_timbre_glide_alpha)
            self._contour_state = "unison"
            return ModeProposal(AccompanimentMode.CONTOUR, None, "rest",
                                note="no pitch to track contour")

        trend = ctx.pitch_trend_semitones(self.contour_lookback_s)
        abs_trend = abs(trend)

        # The continuous timbral lean reacts to the raw trend at ANY
        # magnitude, updated every call regardless of which discrete
        # state (below) is currently active, see this class's __init__
        # for why this needs to be independent of the threshold/hysteresis
        # logic that the pitch step is stuck using.
        target_lean = float(np.clip(trend / self.contour_trend_scale_semis, -1.0, 1.0))
        self._contour_timbre_lean += self.contour_timbre_glide_alpha * (
            target_lean - self._contour_timbre_lean
        )

        # Hysteresis: commit to a direction once movement clearly crosses
        # the move threshold; only release back to unison once it drops
        # below the (lower) release threshold. In between, keep whatever
        # state we were already in, this is what stops the harmony from
        # flickering in and out right at the boundary.
        if abs_trend >= self.contour_move_threshold_semis:
            self._contour_state = "up" if trend > 0 else "down"
        elif abs_trend <= self.contour_release_threshold_semis:
            self._contour_state = "unison"

        if self._contour_state == "unison":
            target_offset = 0.0
        else:
            step = (self.contour_fast_step_semis
                    if abs_trend >= self.contour_fast_threshold_semis
                    else self.contour_step_semis)
            target_offset = step if self._contour_state == "up" else -step

        # Glide toward the target rather than snapping, this is the
        # audible "swoop" into and out of the harmony step that actually
        # reads as a voice following the melody's shape, as opposed to
        # jumping between fixed intervals with no transition.
        self._contour_offset_smoothed += self.contour_glide_alpha * (
            target_offset - self._contour_offset_smoothed
        )

        target = ctx.archer_hz * (2 ** (self._contour_offset_smoothed / 12.0))

        if self._contour_state == "unison":
            shape = "level (locked to unison while the note holds)"
        else:
            interval_word = "a third" if step >= self.contour_fast_threshold_semis - 1e-6 else "a step"
            direction_word = "above" if self._contour_state == "up" else "below"
            verb = "rising" if self._contour_state == "up" else "falling"
            shape = f"{verb}, shadowing {interval_word} {direction_word} ({self._contour_offset_smoothed:+.1f} st)"
        return ModeProposal(AccompanimentMode.CONTOUR, target, "sing",
                            note=f"tracking melodic shape ({shape}, timbre lean {self._contour_timbre_lean:+.2f})")

    def drone(self, ctx: MusicalContext) -> ModeProposal:
        """ONE VOICE, ONE WORD, ONE NOTE, PER PHRASE.

        What a drone actually is, in the practice this project's own reference
        material comes from (a second singer supporting a lead, the way it is
        taught in part-singing work): the supporting voice waits for the line it
        is supporting, opens its mouth ONCE, takes a note that belongs to that
        line and holds it, and stops when the line stops. It does not sing the
        melody a bit lower -- that is a shadow, not a drone, and it is exactly
        what this mode used to do: it re-articulated its word at every breath
        (the run ended whenever the singer was silent, and a new run is a new
        attack), and it re-locked its own pitch whenever the inferred key's root
        class moved, which mid-phrase means a brand-new note and a brand-new
        word for no musical reason at all.

        So the shape here is:
          * the singer rests -> the drone rests. It breathes with the song.
          * a phrase starts -> ONE utterance: a real onset ('sing'), which is
            the mode's single word for that phrase.
          * everything after that, for the whole phrase -> 'sustain', the same
            note at the same pitch, with no re-articulation.
          * the note itself is the phrase's own tonic, placed one octave under
            the phrase's register and clamped into a band a voice can still
            render warmly, so it matches the melody it is under instead of
            sitting at some fixed depth below the key.
        """
        if ctx.archer_hz is None:
            # The line has stopped. A drone belongs to a phrase; letting go here
            # is the whole difference between a held note and a hum that carries
            # on through every gap in the song.
            self._drone_locked_hz = None
            self._drone_note_class = None
            return ModeProposal(AccompanimentMode.DRONE, None, "rest",
                                 note="silent while the singer breathes, so the drone breathes with the song")

        note_class = self._drone_reference_class(ctx)
        new_phrase = bool(ctx.phrase_started) or self._drone_locked_hz is None

        if new_phrase:
            # The one moment this mode is allowed to choose its pitch: the start
            # of a line. Nothing the singer does after this moves it.
            candidate = self._drone_pitch_for_class(ctx.archer_hz, note_class)
            if candidate is None:
                return ModeProposal(AccompanimentMode.DRONE, None, "rest",
                                     note="no pitch reference yet for a drone root")
            self._drone_locked_hz = candidate
            self._drone_note_class = note_class
            return ModeProposal(AccompanimentMode.DRONE, candidate, "sing",
                                 hold_beats=self.drone_hold_beats,
                                 note=f"one word, once, then held: {candidate:.0f} Hz under this phrase")

        return ModeProposal(AccompanimentMode.DRONE, self._drone_locked_hz, "sustain",
                             hold_beats=self.drone_hold_beats,
                             note="holding that one note under the phrase")

    def _drone_reference_class(self, ctx: MusicalContext) -> Optional[int]:
        """The pitch class the drone sits on: the key's own root when the key is
        known, and otherwise the note the phrase came in on."""
        for ref in (ctx.key_root_hz, ctx.archer_hz):
            if not ref:
                continue
            try:
                return int(round(librosa.hz_to_midi(ref))) % 12
            except Exception:
                continue
        return None

    def _drone_pitch_for_class(self, sung_hz: Optional[float],
                                note_class: Optional[int]) -> Optional[float]:
        """That pitch class, at or just under an octave below the phrase.

        The note is placed RELATIVE TO THE PHRASE, which is what makes the pedal
        match the line it is under: the drone is the singer's own key root in the
        register an octave beneath where they are singing (a fourth to an octave
        under it depending on the key), rather than a fixed depth below the key's
        root somewhere the melody may never go. On this project's own reference
        track that distinction is the whole difference between a drone that sits
        with the song and one that lumbers along a twelfth below it.

        The clamp afterwards only ever moves by whole octaves, into a register a
        sung voice renders warmly (70-196 Hz): a vocal drone an octave and a half
        under a woman's line stops sounding like a voice and starts sounding like
        a foghorn, which is the most audible thing the previous version got wrong.
        """
        if not sung_hz or sung_hz <= 0:
            return self._drone_locked_hz
        try:
            midi_ref = float(librosa.hz_to_midi(sung_hz))
        except Exception:
            return self._drone_locked_hz
        octaves = self.drone_octaves_below if self.drone_octaves_below > 0 else 1.0
        target = midi_ref - 12.0 * octaves
        # THE PEDAL STAYS WITHIN A FOURTH TO AN OCTAVE BELOW THE LINE. Those two
        # bounds are the whole character of the mode: closer than a fourth and
        # it is a duet partner rather than support; further than an octave and a
        # key root can land a twelfth under the melody (a C pedal under a G
        # line, 130 Hz against 392 Hz), which is musically defensible and reads
        # to a listener as the robot having gone off somewhere else entirely.
        window_lo, window_hi = midi_ref - 12.0, midi_ref - 4.0
        midi = None
        if note_class is not None:
            for candidate in range(int(np.floor(window_lo)) - 1, int(np.ceil(window_hi)) + 2):
                if candidate % 12 != int(note_class) % 12:
                    continue
                if not (window_lo <= candidate <= window_hi):
                    continue
                if midi is None or abs(candidate - target) < abs(midi - target):
                    midi = candidate
        if midi is None:
            # No crossing of the key's root in that window (an unusual key for
            # the register, or no key yet): the line's own note, exactly an
            # octave down. Consonant by construction.
            midi = int(round(target))
        while librosa.midi_to_hz(midi) < self.drone_min_hz:
            midi += 12
        while librosa.midi_to_hz(midi) > self.drone_max_hz:
            midi -= 12
        return float(librosa.midi_to_hz(midi))

    def call_response(self, ctx: MusicalContext) -> ModeProposal:
        """The live half of call and response: one frame of a composed answer.

        Every frame is handed to the live responder, which keeps the phrase
        currently being sung, reads its cadence when it ends, composes the
        answer it owes (see synthesis/call_response.py) and then hands back
        whichever note of that answer is sounding right now. All this method
        does is turn that into a proposal; the mode's decisions, and the
        reason for them, are the responder's.
        """
        hz, remaining_s, reason = self._live_responder.step(
            now_s=ctx.now_s,
            hz=ctx.archer_hz,
            beat_s=ctx._beat_duration_s,
        )
        if hz is None:
            return ModeProposal(AccompanimentMode.CALL_RESPONSE, None, "rest", note=reason)
        return ModeProposal(
            AccompanimentMode.CALL_RESPONSE, hz, "sing",
            hold_beats=max(0.25, remaining_s / max(ctx._beat_duration_s, 1e-3)),
            note=reason,
        )

    def triadic(self, ctx: MusicalContext, third_semitones: int) -> ModeProposal:
        if ctx.archer_hz is None:
            return ModeProposal(AccompanimentMode.TRIADIC, None, "rest",
                                note="no pitch for triadic harmony")
        target = ctx.archer_hz * (2 ** (third_semitones / 12.0))
        return ModeProposal(AccompanimentMode.TRIADIC, target, "sing",
                            note="fusion-mode triadic harmony (explicit opt-in only)")

    def hum(self, ctx: MusicalContext) -> ModeProposal:
        """One closed-mouth voice on Archer's own note, held.

        This proposes the note, not the sound: the voice itself is
        synthesis/hum_voice.py, rendered by the synthesizer whenever a decision
        carries this mode (see VocableSynthesizer._render_hum), and rendered as
        a whole part offline for the tracks. Nothing here chooses a vocable,
        because a hum has none -- which is the whole difference from this
        mode's previous implementation, where "humming" was one more recorded
        word sung at the singer's pitch and had no neural path at all.
        """
        return ModeProposal(AccompanimentMode.HUM, ctx.archer_hz, "sing",
                             hold_beats=1.5,
                             note="one voice humming quietly alongside him")

    def silent(self, reason: str) -> ModeProposal:
        return ModeProposal(AccompanimentMode.SILENT, None, "rest", note=reason)


@dataclass
class VoiceLayerParams:
    num_voices: int
    detune_spread_cents: float   # max +/- random detune per voice
    timing_jitter_ms: float      # max +/- onset offset per voice (humanizes ensemble)
    formant_spread: float        # 0-1, how much each voice's vocal-tract size varies
    reverb_amount: float         # 0-1, wet/dry, bigger for "concert hall" choir
    stereo_spread: float         # 0-1, how wide voices are panned (used if output is stereo)


class TextureParams:

    def __init__(self, cfg: dict):
        tcfg = cfg.get("synthesis", {}).get("texture", {})

        solo = tcfg.get("solo", {})
        duet = tcfg.get("duet", {})
        choir = tcfg.get("choir", {})

        self._by_texture = {
            VoiceTexture.SOLO: VoiceLayerParams(
                num_voices=1,
                detune_spread_cents=0.0,
                timing_jitter_ms=0.0,
                formant_spread=0.0,
                reverb_amount=solo.get("reverb_amount", 0.08),
                stereo_spread=0.0,
            ),
            VoiceTexture.DUET: VoiceLayerParams(
                num_voices=duet.get("num_voices", 2),
                detune_spread_cents=duet.get("detune_spread_cents", 9.0),
                timing_jitter_ms=duet.get("timing_jitter_ms", 12.0),
                formant_spread=duet.get("formant_spread", 0.12),
                reverb_amount=duet.get("reverb_amount", 0.16),
                stereo_spread=duet.get("stereo_spread", 0.4),
            ),
            VoiceTexture.CHOIR: VoiceLayerParams(
                num_voices=choir.get("num_voices", 7),
                detune_spread_cents=choir.get("detune_spread_cents", 16.0),
                timing_jitter_ms=choir.get("timing_jitter_ms", 28.0),
                formant_spread=choir.get("formant_spread", 0.22),
                reverb_amount=choir.get("reverb_amount", 0.34),
                stereo_spread=choir.get("stereo_spread", 1.0),
            ),
        }

    def get(self, texture: "VoiceTexture") -> VoiceLayerParams:
        return self._by_texture[texture]


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
"""
Call and response, composed rather than approximated.
=====================================================

This module replaces the whole of what "call and response" used to be in this
project, which was one line of arithmetic: take the last pitch Archer sang
before a pause, pull it 65% of the way toward an inferred tonic, hold that one
note for two beats. One held note is not an answer. It has no phrase, no
cadence, no resolution and no reply in it, and a listener never once hears it
as call and response.

WHAT ACTUALLY MAKES SOMETHING SOUND LIKE CALL AND RESPONSE. Read from the
practice itself (see the sources this design was built from, listed at the
bottom of this docstring):

  - It is a CONVERSATION between two voices: one phrase is stated (the call),
    and a DIFFERENT voice answers with a phrase of its own (the response).
    The answer is a phrase, not a note, and it is not the same performance
    continuing, it is a second speaker.
  - The two halves are joined by a cadence relationship. The call tends to
    hang (a half cadence on the 5th, or an open ending on the 2nd/4th/6th, or
    a leading tone that never arrives); the response finishes the thought by
    arriving on the tonic, and it arrives by step - the 2nd falling to the 1st
    or the 7th rising to the 1st - which is the melodic definition of a
    conclusive cadence in the whole tonal tradition (Caplin, the standard
    period model: antecedent ends weak, consequent ends strong).
  - The response is RECOGNIZABLY DERIVED from the call. The strongest and
    most audible form of that is the parallel period: the answer re-enters on
    the call's own material (its rhythm, its opening degrees) and then goes
    where the call did not go. That is what makes the second phrase read as an
    answer TO THIS CALL rather than as an unrelated line.
  - Where the call already closed (it ended on the tonic), the answer does not
    resolve again, it AGREES: gospel congregations and work-song groups repeat
    the leader's line back. That echo is a real and distinct response, and it
    is what keeps this mode from answering every phrase with the same gesture.
  - Timing is the whole game. A response never overlaps the call. It enters in
    the space the call left - late enough to be heard as a reply, soon enough
    to be heard as an answer - and it stops before the next call arrives.
  - The answering voice is a different voice: a different register, a
    different phrase rhythm, a different word, a different distance. The
    responder has its own identity and keeps it for the whole track.

HOW THIS IS IMPLEMENTED (each step is one class below):

  1.  CallSegmenter        the sung pitch curve becomes NOTES (reusing
                           melody_voice.track_notes, which already corrects
                           octave errors and de-jitters), and the notes become
                           CALLS - phrases separated by real pauses.
  2.  Key                  the track's key, estimated over the whole thing
                           from duration-weighted pitch content, plus the
                           track's own tuning offset (so the answer is in tune
                           with the recording, not with A440).
  3.  read_cadence         what the call DID to us, in cadence terms: closed on
                           the tonic, a half cadence on the 5th, open on the
                           2nd/4th/6th, a leaning 7th, or unresolved.
  4.  Responder.plan       the answer, built in SCALE DEGREES: a degree pattern
                           chosen from the cadence class and the room available,
                           a rhythm taken from the call's own tail (extended on
                           the final note, because a cadence lands on a longer
                           note), a register anchored a fixed interval below
                           the call's own ending, and dynamics that follow the
                           call's own loudness.
  5.  render_answers       the plan becomes audio: one word from the robot's
                           voice bank is sung across the whole answer by the
                           WORLD vocoder (synthesis.world_voice), so the
                           formants and consonants stay the recording's own
                           while the pitch follows the composed line, with a
                           scoop on the entry, portamento between notes,
                           vibrato on held notes and a settle at the end.
  6.  LiveCallResponder    the same rules run causally, one frame at a time,
                           for the live microphone path, which cannot see the
                           future and therefore answers conservatively.

SOURCES THE RULES COME FROM (all read for this implementation, not guessed):

  - "Call and response (music)", Wikipedia: leader/chorus response, question
    and answer, echo response, coro-pregon, work songs and lining out.
  - "Call and Response Explained, With Notation", Motifkit: echo response vs
    varied response, the blues AAB shape ("restate the call, then answer it"),
    and the explicit contrast with a period (a response needs no matching
    length, only to read as a reply).
  - "Call and Response: Antecedent and Consequent Phrases",
    musictheoryauthority.com, and "The Period", musictheory.pugetsound.edu:
    the antecedent ends on a weaker cadence (usually half cadence, sometimes an
    imperfect authentic cadence), the consequent ends on a stronger one with
    the melody arriving on the tonic; parallel periods restate the basic idea
    and make the differing cadence audible precisely because the openings
    match; a repeated phrase is NOT a period.
  - "Periods", SFCM online materials: the antecedent is interrogative and ends
    non-final; Caplin's basic idea + contrasting idea is cited there and in the
    sources above.

Nothing in here holds a note for two beats and calls it an answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ============================================================================
# Scales and keys
# ============================================================================

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)

# Krumhansl-Kessler key profiles, the same pair HarmonyEngine._infer_key
# already correlates against, reused here so the two engines agree about what
# key a track is in.
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# How much a phrase's own resting pitch argues for its pitch class being the
# tonic/dominant, on the same scale as the key-profile correlation it is added
# to (see estimate_key). Enough to break a near-tie (a three-note phrase fits
# several keys about equally) and not enough to overrule the profile: on a real
# track, which has many notes, the correlation is the stronger evidence.
KEY_FINAL_BONUS = 0.30
KEY_DOMINANT_FINAL_BONUS = 0.10

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0) if hz and hz > 0 else 0.0


def midi_to_hz(midi: float) -> float:
    return float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))


def note_name(hz: float) -> str:
    """A note name for a log line or the panel's own text."""
    if not hz or hz <= 0:
        return "-"
    midi = int(round(hz_to_midi(hz)))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class Key:
    """A key, plus the tuning grid the recording itself was sung to."""

    root_pc: int = 0
    minor: bool = False
    tuning_cents: float = 0.0

    @property
    def scale(self) -> tuple[int, ...]:
        return MINOR_SCALE if self.minor else MAJOR_SCALE

    def name(self) -> str:
        return f"{NOTE_NAMES[self.root_pc % 12]} {'minor' if self.minor else 'major'}"

    def degree(self, hz: float, tolerance_cents: float = 65.0) -> Optional[tuple[int, float]]:
        """(scale-degree index 0..6, cents off that degree) for a sung pitch.

        None means the pitch is not close enough to any degree of this key to
        be read as one, which is a real answer in itself (see CADENCE_OPEN).
        """
        if not hz or hz <= 0:
            return None
        rel = (hz_to_midi(hz) - self.root_pc) % 12.0
        best_i, best_off = 0, 1e9
        for i, step in enumerate(self.scale):
            off = rel - step
            if off > 6.0:
                off -= 12.0
            elif off < -6.0:
                off += 12.0
            if abs(off) < abs(best_off):
                best_i, best_off = i, off
        if abs(best_off) * 100.0 > tolerance_cents:
            return None
        return best_i, best_off * 100.0

    def midi_for_degree(self, degree: int, near_midi: float) -> float:
        """The MIDI pitch of a scale degree, at the octave nearest `near_midi`.

        Degrees past 6 continue into the next octave and negative degrees
        continue down (divmod handles both), so a degree pattern can move
        freely between octaves while every pitch in it stays in the key.
        """
        octave, index = divmod(int(degree), 7)
        pc = (self.root_pc + self.scale[index]) % 12
        base = near_midi + 12.0 * octave
        pitch = base - (base % 12.0) + pc
        while pitch - base > 6.0:
            pitch -= 12.0
        while base - pitch > 6.0:
            pitch += 12.0
        return pitch


def estimate_key(note_hz: Sequence[float], weights: Optional[Sequence[float]] = None,
                 tuning_cents: float = 0.0,
                 final_hz: Optional[Sequence[float]] = None,
                 final_weights: Optional[Sequence[float]] = None) -> Key:
    """The track's key from its own sung notes, weighted by how long each was held.

    Weighting by DURATION is the difference between this and
    HarmonyEngine._infer_key (which counts frames of a rolling window): a
    quarter note held for a second should not count the same as a sixteenth in
    a passing turn, and in an offline render the whole track can be measured at
    once, so the key does not wander while the answer is being composed.

    The phrases' own FINAL notes are counted separately and weighted in on top,
    which is the old "final tone" rule of key finding and it matters more than
    it looks here: three or four notes of melody fit several keys equally well,
    and the one thing about a sung phrase that is nearly always decisive is
    where it came to rest. A small share also goes to the 5th, because a track
    that keeps hanging on a half cadence legitimately rests there a lot.
    """
    vals = [h for h in note_hz if h and h > 0]
    if not vals:
        return Key(0, False, tuning_cents)
    if weights is None or len(weights) != len(vals):
        weights = [1.0] * len(vals)

    chroma = np.zeros(12, dtype=np.float64)
    for hz, w in zip(vals, weights):
        midi = hz_to_midi(hz) - tuning_cents / 100.0
        chroma[int(round(midi)) % 12] += max(0.0, float(w))
    total = chroma.sum()
    if total <= 0:
        return Key(0, False, tuning_cents)
    chroma /= total

    finals = np.zeros(12, dtype=np.float64)
    if final_hz:
        weights_f = list(final_weights or [1.0] * len(final_hz))
        for hz, w in zip(final_hz, weights_f):
            if hz and hz > 0:
                midi = hz_to_midi(hz) - tuning_cents / 100.0
                finals[int(round(midi)) % 12] += max(0.0, float(w))
        if finals.sum() > 0:
            finals /= finals.sum()

    best_corr, best_root, best_minor = -np.inf, 0, False
    for root in range(12):
        rolled = np.roll(chroma, -root)
        bonus = (KEY_FINAL_BONUS * float(finals[root])
                 + KEY_DOMINANT_FINAL_BONUS * float(finals[(root + 7) % 12])) if finals.any() else 0.0
        for profile, minor in ((KRUMHANSL_MAJOR, False), (KRUMHANSL_MINOR, True)):
            corr = float(np.corrcoef(rolled, profile)[0, 1]) + bonus
            if corr > best_corr:
                best_corr, best_root, best_minor = corr, root, minor
    return Key(best_root, best_minor, tuning_cents)


def snap_to_grid(hz: float, tuning_cents: float) -> float:
    """Move a pitch onto the recording's own tuning grid.

    A tracked note sits wherever the singer actually sang it; the answer has to
    be tuned to THAT (the same rule the single-word renderer follows), or the
    two voices beat against each other by a few tens of cents for the whole
    track.
    """
    if not hz or hz <= 0:
        return hz
    cents = 1200.0 * math.log2(hz / 440.0)
    snapped = (cents - tuning_cents) / 100.0
    return float(440.0 * (2.0 ** ((round(snapped) * 100.0 + tuning_cents) / 1200.0)))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CallResponseCfg:
    """Every number this mode's musical judgement rests on, in one place."""

    # where a call ends and the answer may begin
    phrase_gap_s: float = 0.34          # silence that ends a call
    release_allowance_s: float = 0.08   # the singer's own release tail, not detected as voiced
    gap_beats: float = 0.50             # nominal space before the answer enters
    gap_min_s: float = 0.22
    gap_max_s: float = 0.42
    gap_jitter_s: float = 0.025         # a human entry is not metronomic
    min_room_s: float = 0.30            # below this there is no answer to give
    tail_guard_s: float = 0.10          # silence left before the next call

    # what counts as a call
    min_call_s: float = 0.30
    min_call_rms_ratio: float = 0.25    # a call this much quieter than the track's median is a cough, not a phrase

    # how big an answer the room allows
    room_medium_s: float = 0.85         # room left after the gap, beyond this a 3+ note answer fits
    room_long_s: float = 1.55
    min_answer_span_s: float = 0.38
    max_answer_span_s: float = 2.80
    short_span_s: float = 0.62
    medium_call_share: float = 0.55     # an answer this long relative to the call ...
    long_call_share: float = 0.72
    long_extra_s: float = 0.35          # ... plus the room the quote/resolve needs

    # the line itself
    resolve_below_semitones: float = 7.0    # where a RESOLVING answer sits, relative to the call's ending
    echo_below_semitones: float = 4.0       # where an ECHOING answer sits
    min_register_drop_semitones: float = 3.0   # an answer never doubles the call in place
    tonic_floor_hz: float = 165.0           # the answer's own tonic is clamped into this band (E3-G4)
    tonic_ceil_hz: float = 415.0
    line_lowest_hz: float = 140.0
    line_highest_hz: float = 640.0
    line_max_span_semitones: float = 14.0
    max_answer_notes: int = 5
    final_note_share: float = 0.42          # the cadence's own landing note, as a share of the answer's span
    final_note_max_share: float = 0.58      # ... and a ceiling on that share, so the rest of the answer keeps its room
    final_note_min_s: float = 0.34
    final_note_max_s: float = 1.20
    final_note_extend: float = 1.55         # how much longer the closing note is than the call's own tail
    min_note_s: float = 0.17
    rhythm_quantum_s: float = 0.02

    # dynamics
    level_ratio: float = 1.0            # the answer's loudness against the call's own
    level_min: float = 0.62
    level_max: float = 1.25
    final_note_db: float = 1.4          # a cadence lands, so the last note is the loudest
    answer_peak: float = 0.72           # the rendered phrase's own peak before the mix

    # the answering voice
    takes: tuple[str, ...] = ("cold", "moon", "need", "remember", "snow",
                              "shell", "voyage", "ohoho", "lalanana")
    long_takes: tuple[str, ...] = ("ohoho", "lalanana", "welcome to reality",
                                   "you know you're going to feel it")
    take_repeat_penalty: float = 1.30   # how much the previous answer's word is avoided
    take_brightness_bias: float = 0.30  # prefer the rounder of two equally-fitting takes
    trim_silence: bool = True
    trim_floor_db: float = -42.0

    # phrasing of the rendered answer
    portamento_s: float = 0.075
    scoop_cents: float = 45.0
    scoop_s: float = 0.10
    vibrato_depth_cents: float = 16.0
    vibrato_rate_hz: float = 5.3
    vibrato_delay_s: float = 0.32
    vibrato_ramp_s: float = 0.26
    drift_depth_cents: float = 5.0
    drift_rate_hz: float = 0.63
    release_fall_cents: float = 30.0
    release_fall_s: float = 0.13
    note_dip: float = 0.34              # how far the level dips between two notes of the answer
    note_dip_s: float = 0.075
    phrase_attack_s: float = 0.024
    phrase_release_s: float = 0.11
    stereo_spread: float = 0.0          # the robot part is mono in this pipeline; kept for the record

    @classmethod
    def from_config(cls, cfg: dict) -> "CallResponseCfg":
        block = (cfg or {}).get("modes", {}).get("call_response", {}) or {}
        out = cls()
        for key, value in block.items():
            if hasattr(out, key) and value is not None:
                current = getattr(out, key)
                try:
                    if isinstance(current, tuple):
                        setattr(out, key, tuple(str(v) for v in value))
                    elif isinstance(current, bool):
                        setattr(out, key, bool(value))
                    elif isinstance(current, int) and not isinstance(current, bool):
                        setattr(out, key, int(value))
                    else:
                        setattr(out, key, float(value))
                except (TypeError, ValueError):
                    pass
        return out


# ============================================================================
# 1. Notes and calls
# ============================================================================

@dataclass
class CallNote:
    start_s: float
    end_s: float
    hz: float
    rms: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class Call:
    """One complete phrase of Archer's: the thing the robot answers."""

    start_s: float
    end_s: float
    notes: list[CallNote]
    mean_rms: float = 0.0

    @property
    def span_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def final_hz(self) -> float:
        return self.notes[-1].hz if self.notes else 0.0

    @property
    def mean_hz(self) -> float:
        """Duration-weighted mean pitch, the phrase's own centre of gravity."""
        total = sum(n.duration_s for n in self.notes) or 1.0
        return float(sum(n.hz * n.duration_s for n in self.notes) / total)

    @property
    def top_hz(self) -> float:
        return max((n.hz for n in self.notes), default=0.0)

    @property
    def tail_durations(self) -> list[float]:
        return [n.duration_s for n in self.notes[-4:]]

    def degrees(self, key: Key) -> list[Optional[int]]:
        return [None if key.degree(n.hz) is None else key.degree(n.hz)[0] for n in self.notes]

    def describe(self) -> str:
        names = " ".join(f"{note_name(n.hz)}({n.duration_s:.2f}s)" for n in self.notes)
        return f"{self.start_s:6.2f}-{self.end_s:6.2f}s  {names}"


@dataclass
class TrackAnalysis:
    calls: list[Call]
    key: Key
    tuning_cents: float
    median_call_rms: float
    first_onset_s: float
    last_end_s: float


def segment_calls(times: np.ndarray, pitch: np.ndarray, rms: np.ndarray,
                  frame_hop_s: float, cfg: CallResponseCfg,
                  audio: Optional[np.ndarray] = None, sr: Optional[int] = None,
                  frame_size: Optional[int] = None) -> list[Call]:
    """A per-frame pitch curve becomes the list of phrases Archer actually sang.

    The note layer is deliberately NOT written here: melody_voice.track_notes
    already does this job properly for this project, including the two things a
    naive segmenter gets wrong on real recorded voice (a detector reading an
    octave out, and vibrato splitting one note into four). It runs with the
    audio in hand so its spectral octave evidence is available, and with its
    own tuning-grid snap off, since the answer is snapped to that grid later,
    at the point where it is placed rather than at the point where it is read.
    """
    from synthesis.melody_voice import track_notes

    times = np.asarray(times, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    voiced = np.isfinite(pitch) & (pitch > 0)
    hz_curve = np.where(voiced, pitch, 0.0)

    if voiced.sum() < 3:
        return []

    _, notes = track_notes(hz_curve, voiced, float(frame_hop_s), audio=audio, sr=sr,
                           hop_samples=frame_size, snap_to_key=False)
    if not notes:
        return []

    def rms_between(a: float, b: float) -> float:
        lo = max(0, int(a / frame_hop_s))
        hi = min(len(rms), int(b / frame_hop_s) + 1)
        if hi <= lo:
            return 0.0
        window = rms[lo:hi]
        return float(np.median(window)) if window.size else 0.0

    calls: list[Call] = []
    current: list[CallNote] = []
    for note in notes:
        entry = CallNote(float(note.start_s), float(note.end_s), float(note.f0_hz),
                         rms_between(note.start_s, note.end_s))
        if current and entry.start_s - current[-1].end_s > cfg.phrase_gap_s:
            calls.append(_close_call(current))
            current = []
        current.append(entry)
    if current:
        calls.append(_close_call(current))

    return [c for c in calls if c.span_s >= cfg.min_call_s and c.mean_rms > 0]


def _close_call(notes: list[CallNote]) -> Call:
    weights = np.array([max(n.duration_s, 1e-3) for n in notes], dtype=np.float64)
    mean_rms = float(np.average([n.rms for n in notes], weights=weights)) if notes else 0.0
    return Call(notes[0].start_s, notes[-1].end_s, list(notes), mean_rms)


def analyze_track(times: np.ndarray, pitch: np.ndarray, rms: np.ndarray,
                  frame_hop_s: float, cfg: CallResponseCfg,
                  audio: Optional[np.ndarray] = None, sr: Optional[int] = None,
                  frame_size: Optional[int] = None) -> TrackAnalysis:
    from synthesis.melody_voice import estimate_tuning_offset_cents

    calls = segment_calls(times, pitch, rms, frame_hop_s, cfg, audio=audio, sr=sr,
                          frame_size=frame_size)
    all_hz: list[float] = []
    all_w: list[float] = []
    for call in calls:
        for note in call.notes:
            all_hz.append(note.hz)
            all_w.append(max(note.duration_s, 1e-3))

    tuning, confidence = estimate_tuning_offset_cents(all_hz)
    if confidence < 0.45:
        tuning = 0.0
    key = estimate_key(all_hz, all_w, tuning_cents=tuning,
                       final_hz=[c.final_hz for c in calls if c.notes],
                       final_weights=[c.notes[-1].duration_s for c in calls if c.notes])

    if calls:
        loud = sorted(c.mean_rms for c in calls)
        median_rms = float(loud[len(loud) // 2])
        floor_rms = median_rms * cfg.min_call_rms_ratio
        calls = [c for c in calls if c.mean_rms >= floor_rms]
        first_onset = calls[0].start_s if calls else 0.0
        last_end = calls[-1].end_s if calls else 0.0
    else:
        median_rms, first_onset, last_end = 0.0, 0.0, 0.0

    return TrackAnalysis(calls, key, tuning, median_rms, first_onset, last_end)


# ============================================================================
# 2. What the call did: its cadence
# ============================================================================

CADENCE_TONIC = "tonic"            # it closed: arrived home on the 1st degree
CADENCE_WEAK_TONIC = "weak_tonic"  # the 3rd: an imperfect arrival, closed but not settled
CADENCE_HALF = "half"              # the 5th: the classic open ending that demands an answer
CADENCE_OPEN = "open"              # the 2nd, 4th or 6th: hanging, unresolved
CADENCE_LEADING = "leading"        # the 7th: leaning, and it never arrived
CADENCE_UNRESOLVED = "unresolved"  # not on the key at all, or no stable final note


@dataclass
class Cadence:
    kind: str
    degree: Optional[int]       # 0..6, the final note's degree index, when it has one
    held_s: float               # how long the closing note was held
    is_cadence: bool            # false when the phrase just stopped rather than closed


def read_cadence(call: Call, key: Key) -> Cadence:
    """Classify how the call ended, which is what decides how the answer answers.

    A phrase whose last note is a short clipped blip is not a cadence: it was
    cut off mid-motion. That reads as open regardless of which degree it
    happened to land on, which is what makes the robot answer a phrase that
    stops abruptly instead of treating it as a statement that finished.
    """
    if not call.notes:
        return Cadence(CADENCE_UNRESOLVED, None, 0.0, False)
    last = call.notes[-1]
    read = key.degree(last.hz)
    held = last.duration_s
    if read is None:
        return Cadence(CADENCE_UNRESOLVED, None, held, True)

    index, _cents = read
    mean_note = float(np.mean([n.duration_s for n in call.notes])) if call.notes else held
    is_cadence = held >= max(0.20, 0.6 * mean_note) or len(call.notes) == 1
    if not is_cadence:
        return Cadence(CADENCE_OPEN, index, held, False)

    if index == 0:
        return Cadence(CADENCE_TONIC, index, held, True)
    if index == 2:
        return Cadence(CADENCE_WEAK_TONIC, index, held, True)
    if index == 4:
        return Cadence(CADENCE_HALF, index, held, True)
    if index == 6:
        return Cadence(CADENCE_LEADING, index, held, True)
    return Cadence(CADENCE_OPEN, index, held, True)


# ============================================================================
# 3. The answer
# ============================================================================

ANSWER_RESOLVE = "resolve"      # the call hung, so the answer finishes it
ANSWER_ECHO = "echo"            # the call closed, so the answer agrees with it
ANSWER_SETTLE = "settle"        # the call ended off the key: land it anyway
ANSWER_SINGLE = "single"        # only a moment of room: one note, and make it count


def _dedupe(degrees: list[int]) -> list[int]:
    """Drop a repeated neighbouring degree, which realisation would collapse anyway."""
    out: list[int] = []
    for degree in degrees:
        if not out or out[-1] != degree:
            out.append(degree)
    return out or [0]


def _compress(line: list[int], budget: int) -> list[int]:
    """Fit a longer cadential run into a smaller note budget without losing its ends.

    What must survive is the note the call actually hung on (the answer's
    pickup, which is what makes the phrase an answer TO THIS CALL), the
    stepwise approach, and the tonic. Notes are dropped from the middle of the
    run, evenly, rather than from its front: trimming the front turns a
    descending answer into a random cluster of notes near the tonic, which is
    precisely the thing that stops sounding like a reply.
    """
    if budget <= 1:
        return [line[-1]]
    if len(line) <= budget:
        return line
    if budget == 2:
        # with two notes there is no room to pick the call's degree up and land
        # as well, so the arrival itself is what survives: a step into the tonic
        return _dedupe(line[-2:])
    head, tail = line[0], line[-2:]
    middle = line[1:-2]
    keep = budget - 3
    if keep <= 0 or not middle:
        return _dedupe([head] + tail)
    step = len(middle) / float(keep)
    chosen = [middle[min(len(middle) - 1, int(i * step))] for i in range(keep)]
    return _dedupe([head] + chosen + tail)


@dataclass
class AnswerNote:
    start_s: float      # relative to the answer's own start
    dur_s: float
    hz: float
    degree: int
    accent: float = 1.0
    scoop: bool = True


@dataclass
class Answer:
    """One composed response, placed on the track's own timeline."""

    start_s: float
    call: Call
    cadence: Cadence
    form: str
    take: str
    notes: list[AnswerNote]
    level: float
    reason: str

    @property
    def span_s(self) -> float:
        return sum(n.dur_s for n in self.notes)

    @property
    def end_s(self) -> float:
        return self.start_s + self.span_s

    def describe(self) -> str:
        line = " ".join(f"{note_name(n.hz)}:{n.dur_s:.2f}" for n in self.notes)
        return (f"{self.start_s:6.2f}s {self.form:6s} take={self.take:9s} "
                f"call[{self.cadence.kind}] {line}")


class Responder:
    """The answering voice: one identity, many answers.

    The identity is chosen once for the track (which words it may sing, how
    loud it answers, which form it favours) so the whole track sounds like ONE
    group answering, while each answer's melody is composed against the call it
    is actually answering. That split - a fixed responder, a tailored answer -
    is exactly the split the practice has: the leader improvises, the group's
    response is stable.
    """

    def __init__(self, key: Key, cfg: CallResponseCfg,
                 brightness: Optional[dict[str, float]] = None,
                 available: Optional[Sequence[str]] = None,
                 lengths: Optional[dict[str, float]] = None):
        self.key = key
        self.cfg = cfg
        self.brightness = brightness or {}
        self.lengths = dict(lengths or {})
        self.available = [str(n).strip().lower() for n in (available or [])]
        self._last_take: Optional[str] = None

    # take choice
    def take_length(self, name: str) -> float:
        key = name.strip().lower()
        if key in self.lengths:
            return float(self.lengths[key])
        return float(_TAKE_LENGTHS.get(key, 1.25))

    def choose_take(self, span_s: float) -> str:
        """The word this answer is sung on: the one whose own length fits it.

        A vocable take is a real recording of a real word, 1.0 to 3.9 seconds
        long. Sung across an answer that lasts about as long as it does, its
        syllables land across the notes of the answer the way a lyric does over
        a phrase. Too long and the word is cut off mid-way; much too short and
        it has to start over inside the phrase. So the fit is the first term,
        and the second term steers toward the rounder-sounding words when two
        takes fit equally well, because the answering voice should be the
        darker of the two voices in the room. The word sung by the previous
        answer is pushed away, so the responder is one voice with more than one
        thing to say rather than a loop.
        """
        pool = list(self.cfg.takes)
        if span_s > 1.7:
            pool = pool + list(self.cfg.long_takes)
        if self.available:
            pool = [n for n in pool if n in self.available]
        if not pool:
            pool = list(self.available) or list(self.cfg.takes)
        if not pool:
            return ""
        scored: list[tuple[float, str]] = []
        for name in pool:
            fit = abs(math.log2(max(span_s, 0.1) / max(self.take_length(name), 0.1)))
            score = fit
            score += self.cfg.take_brightness_bias * float(self.brightness.get(name, 0.5))
            if name == self._last_take:
                score += self.cfg.take_repeat_penalty
            scored.append((score, name))
        scored.sort(key=lambda item: (item[0], item[1]))
        choice = scored[0][1]
        self._last_take = choice
        return choice

    # the degree pattern
    def degree_pattern(self, cadence: Cadence, form: str, call: Call,
                       note_budget: int) -> list[int]:
        """The shape of the answer, in scale degrees, before any octave is chosen.

        The pattern ends on 0 (the tonic) for every form except the echo whose
        own last degree is the tonic already, and every non-echo pattern ends
        with the 2 -> 1 or the 7 -> 1 approach, which is what makes the arrival
        read as a real cadence rather than as a note that happens to be first
        in the scale.
        """
        tail_degrees = [d for d in call.degrees(self.key) if d is not None]
        final_degree = tail_degrees[-1] if tail_degrees else None

        if form == ANSWER_SINGLE:
            return [0]

        if form == ANSWER_ECHO:
            # Agreement: the group sings back what the leader just said, ending
            # where the leader ended. The phrase's last 1-3 degrees, in order,
            # with the tonic appended when the call did not already land there.
            quoted = [d for d in call.degrees(self.key)][-min(3, note_budget):]
            quoted = [d for d in quoted if d is not None]
            if not quoted:
                quoted = [2, 1, 0][-note_budget:]
            if quoted[-1] != 0:
                quoted = quoted[-max(1, note_budget - 1):] + [0]
            if len(quoted) < note_budget:
                # Room for more than the line it is agreeing with, so the group
                # repeats it and then TURNS into the landing: the 2nd falling
                # the step to the tonic (never a repeat of the same note, and
                # never an arrival from nowhere).
                pad = [1] * (note_budget - len(quoted))
                quoted = quoted[:-1] + pad + [quoted[-1]] if len(quoted) > 1 else pad + [0]
            return _dedupe(quoted)[-note_budget:]

        if form == ANSWER_SETTLE or final_degree is None:
            # The call stopped somewhere off the key (or on a pitch no degree
            # can be read from), so there is nothing of the call's own line to
            # pick up. Land the tonic from whichever side is closer to where it
            # actually stopped: the 2nd falling to the 1st if the call ended
            # above the tonic, the 7th rising to it if it ended below. Either
            # way the last move is a step, which is what makes it an arrival.
            anchor = midi_to_hz(self.key.midi_for_degree(0, hz_to_midi(call.final_hz or 220.0)))
            line = [1, 0] if (call.final_hz or 0.0) > anchor else [6, 0]
            return _compress(line, note_budget)

        if final_degree == 6:
            # A leaning 7th is the one ending that must RISE: walking it down
            # would arrive at the tonic a major 7th below and stop sounding
            # like the resolution the ear is already waiting for. The answer
            # picks the leading tone up and steps the semitone home.
            return _compress([6, 0], note_budget)

        # RESOLVE: pick the call's unfinished degree up, and walk it down to the
        # tonic. A scale run from wherever it hung is the oldest and most
        # legible answer in the tradition, and it adapts to any ending because
        # it starts from the ending itself.
        line = list(range(int(final_degree), 0, -1)) + [0]
        if len(line) < 2:
            line = [1, 0]
        return _compress(line, note_budget)

    # the whole answer
    def plan(self, call: Call, cadence: Cadence, start_s: float, room_s: float,
             form: str, span_s: float, level: float, reason: str) -> Optional[Answer]:
        cfg = self.cfg
        # one landing note is a real answer ("yes") and it fits in less room
        # than a phrase does, so it is measured against its own floor
        min_span = (cfg.min_note_s * 1.2) if form == ANSWER_SINGLE else cfg.min_answer_span_s
        if span_s < min_span:
            return None

        # how many notes the room can actually hold: a note below about a third
        # of a second stops reading as a note and starts reading as a trill
        note_budget = max(1, min(cfg.max_answer_notes, int(round(span_s / 0.32))))
        pattern = self.degree_pattern(cadence, form, call, note_budget)
        if not pattern:
            return None

        # Where the answer's own tonic sits: relative to the call's ending, an
        # interval below it (a responder answers from its own place, and its
        # own place is under the leader), clamped into a band that a voice can
        # actually sing, so neither a very low nor a very high call drags the
        # answer out of the responder's register.
        below = cfg.echo_below_semitones if form == ANSWER_ECHO else cfg.resolve_below_semitones
        target_final = call.final_hz * (2.0 ** (-below / 12.0)) if call.final_hz > 0 else 220.0
        anchor_midi = self.key.midi_for_degree(0, hz_to_midi(target_final))
        anchor_hz = midi_to_hz(anchor_midi)
        while anchor_hz < cfg.tonic_floor_hz:
            anchor_midi += 12.0
            anchor_hz = midi_to_hz(anchor_midi)
        while anchor_hz > cfg.tonic_ceil_hz:
            anchor_midi -= 12.0
            anchor_hz = midi_to_hz(anchor_midi)

        hz = self._realize(pattern, anchor_midi)
        if hz is None:
            return None
        hz = self._enforce_register(hz, call)

        durs = self._rhythm(call, len(hz), span_s)
        notes = []
        for index, (pitch_deg, note_hz_, dur) in enumerate(zip(pattern, hz, durs)):
            notes.append(AnswerNote(0.0, dur, note_hz_, pitch_deg))
        # place them on the answer's own clock
        cursor = 0.0
        for note in notes:
            note.start_s = cursor
            cursor += note.dur_s

        # accents: the closing note of a cadence lands, and the entry of the
        # answering voice is very slightly forward so it reads as an entrance
        notes[-1].accent = 10.0 ** (cfg.final_note_db / 20.0)
        if len(notes) > 2:
            notes[0].accent *= 1.03

        take = self.choose_take(cursor)
        return Answer(start_s, call, cadence, form, take, notes, level, reason)

    def _realize(self, pattern: list[int], anchor_midi: float) -> Optional[list[float]]:
        """Degrees to pitches: the last note is the anchor, the rest walk back to it.

        Walking backwards (rather than forwards from the first note) is what
        guarantees the answer LANDS where it was meant to land. Each earlier
        note is then placed at the octave nearest the note after it, which is
        the placement that keeps every interval in the line as small as it can
        be: 5 falling to 2 to 1 (G4 D4 C4) is one small step down after another,
        where anything cleverer about octaves turns it into a lurch.
        """
        cfg = self.cfg
        midis = [0.0] * len(pattern)
        midis[-1] = anchor_midi
        for i in range(len(pattern) - 2, -1, -1):
            midis[i] = self.key.midi_for_degree(pattern[i], midis[i + 1])

        span = max(midis) - min(midis)
        if span > cfg.line_max_span_semitones:
            # the pattern wants more than an octave, which means the answer
            # would stop sounding like one voice's phrase: pull the upper notes
            # down an octave rather than let the line sprawl
            for i in range(len(midis) - 1, -1, -1):
                if midis[i] - min(midis) > cfg.line_max_span_semitones:
                    midis[i] -= 12.0

        hz = [midi_to_hz(m) for m in midis]
        if min(hz) < cfg.line_lowest_hz or max(hz) > cfg.line_highest_hz:
            shift = 0
            while min(midi_to_hz(m + 12.0 * shift) for m in midis) < cfg.line_lowest_hz:
                shift += 1
            while max(midi_to_hz(m + 12.0 * shift) for m in midis) > cfg.line_highest_hz:
                shift -= 1
            if shift:
                midis = [m + 12.0 * shift for m in midis]
                hz = [midi_to_hz(m) for m in midis]
        return [snap_to_grid(h, self.key.tuning_cents) for h in hz]

    def _enforce_register(self, hz: list[float], call: Call) -> list[float]:
        """Keep the answer under the call, so it never doubles it in place.

        The one failure that would sink the whole idea: an answer realised at
        the call's own pitches in the call's own register does not sound like a
        second voice answering, it sounds like the same voice doubled or
        delayed. A responder answers from lower down, so if the realised line's
        centre of gravity is not clearly under the call's, it is dropped an
        octave - as far as the vocal band allows, and no further, since a very
        low call has nowhere lower to go and the answer is simply sung above
        it, which is also what a real group does with a low leader.
        """
        if not hz or call.mean_hz <= 0:
            return hz
        for _ in range(3):
            mean = float(np.mean(np.log2(np.asarray(hz, dtype=np.float64))))
            limit = math.log2(call.mean_hz) - self.cfg.min_register_drop_semitones / 12.0
            if mean <= limit or min(hz) / 2.0 < self.cfg.line_lowest_hz:
                break
            hz = [h / 2.0 for h in hz]
        return hz

    def _rhythm(self, call: Call, count: int, span_s: float) -> list[float]:
        """The answer's rhythm: the call's own tail, with its landing extended.

        The rhythm is the most audible proof that the second phrase is an
        answer TO THIS CALL, so it is not invented: it is the durations Archer
        just sang, as many of them as the answer has notes, with the closing
        note stretched (a cadence lands and holds) and the whole thing scaled
        to the room there actually is.
        """
        cfg = self.cfg
        if count <= 1:
            return [span_s]

        tail = [d for d in call.tail_durations if d > 0.05]
        if not tail:
            tail = [0.4]
        take_from = tail[-min(count - 1, len(tail)):]
        weights = [max(take_from[i % len(take_from)], 0.08) for i in range(count - 1)]
        total_w = sum(weights)

        # The closing note is the cadence's landing, so it is held: at least the
        # call's own final note held longer, and at least its share of the span,
        # but never so much of the span that the notes before it are crushed
        # into blips.
        final = max(span_s * cfg.final_note_share,
                    min(tail[-1] * cfg.final_note_extend, span_s * cfg.final_note_max_share))
        final = float(np.clip(final, cfg.final_note_min_s, cfg.final_note_max_s))
        final = min(final, max(cfg.min_note_s, span_s - cfg.min_note_s * (count - 1)))
        rest = max(span_s - final, cfg.min_note_s * (count - 1))
        durs = [max(cfg.min_note_s, rest * w / total_w) for w in weights]
        if sum(durs) > rest:
            # the floor pushed the run over its own budget: keep the shape and
            # bring it back inside, the floor holding where it has to
            factor = rest / sum(durs)
            durs = [max(cfg.min_note_s, d * factor) for d in durs]
        # quantize to a 20 ms grid, then let the landing note take up whatever
        # is left, so the answer's total length is exactly the length it was
        # composed at and it lands where the plan said it would
        q = cfg.rhythm_quantum_s
        durs = [max(cfg.min_note_s, round(d / q) * q) for d in durs]
        final = max(cfg.min_note_s, span_s - sum(durs))
        return durs + [final]


# Rough length of each loaded vocable take, in seconds, as measured from the
# bank on this project's own samples. Used only to pick which word fits an
# answer; the renderer reads the real take and its real length.
_TAKE_LENGTHS = {
    "cold": 1.23, "moon": 1.20, "need": 1.30, "remember": 1.00, "snow": 1.12,
    "shell": 1.71, "voyage": 1.26, "ohoho": 2.23, "lalanana": 3.26,
    "welcome to reality": 3.91, "you know you're going to feel it": 3.50,
}


# ============================================================================
# 4. Composing the whole track
# ============================================================================

def compose_answers(track: TrackAnalysis, cfg: CallResponseCfg,
                    tempo_bpm: float = 0.0, total_duration_s: float = 0.0,
                    brightness: Optional[dict[str, float]] = None,
                    available: Optional[Sequence[str]] = None,
                    lengths: Optional[dict[str, float]] = None,
                    log: bool = True) -> list[Answer]:
    """Every answer the track earns, in order.

    The room each answer gets is the real silence between this call and the
    next one - the singer's own breathing space, measured - which is what makes
    the responses land in the conversation's gaps instead of over the top of
    it. The final call of a track is allowed to answer past the end of the
    track, because the last answer is the one the listener is left with.
    """
    calls = track.calls
    if not calls:
        return []

    responder = Responder(track.key, cfg, brightness, available=available, lengths=lengths)
    beat_s = (60.0 / tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 0.5
    gap = float(np.clip(cfg.gap_beats * beat_s, cfg.gap_min_s, cfg.gap_max_s))

    answers: list[Answer] = []
    for index, call in enumerate(calls):
        cadence = read_cadence(call, track.key)

        next_start = calls[index + 1].start_s if index + 1 < len(calls) else None
        call_end = call.end_s + cfg.release_allowance_s
        if next_start is None:
            # the last answer of the track: the room is whatever is left of the
            # track, plus the render's own tail - it never gets cut off
            room = max(cfg.max_answer_span_s * 1.4,
                       (total_duration_s - call_end) if total_duration_s else 2.0)
        else:
            room = next_start - call_end

        # A tight breathing space gets the answer in as early as the singer's
        # own release allows, rather than a fixed entry that would leave the
        # responder with nothing it can say.
        this_gap = min(gap, max(cfg.gap_min_s, room - cfg.min_answer_span_s - cfg.tail_guard_s))
        usable = room - this_gap - cfg.tail_guard_s
        jitter = cfg.gap_jitter_s * math.sin(index * 2.399963)  # deterministic, no RNG
        start_s = call_end + this_gap + jitter
        if usable < cfg.min_room_s:
            if log:
                print(f"[call-response] no room to answer '{call.describe()}' "
                      f"({room:.2f}s of space)")
            continue

        form, span = _choose_form(call, cadence, usable, cfg)
        if form is None:
            continue
        level = float(np.clip(cfg.level_ratio * (call.mean_rms / track.median_call_rms)
                              if track.median_call_rms > 0 else 1.0, cfg.level_min, cfg.level_max))
        reason = _reason_text(form, cadence)
        answer = responder.plan(call, cadence, start_s, usable, form, span, level, reason)
        if answer is None:
            continue
        answers.append(answer)

    if log:
        print(f"[call-response] key {track.key.name()} (tuning {track.tuning_cents:+.0f} cents), "
              f"{len(calls)} call(s), {len(answers)} answer(s)")
        for answer in answers:
            print(f"[call-response]   {answer.describe()}")
    return answers


def _choose_form(call: Call, cadence: Cadence, usable_s: float,
                 cfg: CallResponseCfg) -> tuple[Optional[str], float]:
    """Pick how much answer this much room can hold, and how long it should be."""
    if cadence.kind in (CADENCE_TONIC, CADENCE_WEAK_TONIC):
        base_form = ANSWER_ECHO
    elif cadence.kind == CADENCE_UNRESOLVED:
        base_form = ANSWER_SETTLE
    else:
        base_form = ANSWER_RESOLVE

    # how long the answer wants to be, before the room gets a say
    if usable_s >= cfg.room_long_s:
        span = cfg.long_call_share * call.span_s + cfg.long_extra_s
    elif usable_s >= cfg.room_medium_s:
        span = cfg.medium_call_share * call.span_s
    else:
        span = cfg.short_span_s
    span = float(np.clip(span, cfg.min_answer_span_s, min(cfg.max_answer_span_s, usable_s)))

    if usable_s < cfg.min_answer_span_s:
        return None, 0.0
    if span < cfg.min_answer_span_s or span < cfg.min_note_s * 1.5:
        # not enough room for a phrase; a single landing note can still fit,
        # and one note that lands the tonic is a real answer (that is the
        # shortest response there is: "yes")
        if usable_s >= cfg.min_note_s * 1.2:
            return ANSWER_SINGLE, min(usable_s, max(cfg.short_span_s, cfg.min_note_s * 1.6))
        return None, 0.0
    return base_form, span


def _reason_text(form: str, cadence: Cadence) -> str:
    if form == ANSWER_ECHO:
        return "it closed on the tonic, so the answer sings it back instead of resolving it"
    if form == ANSWER_SETTLE:
        return "the phrase stopped off the key, so the answer lands it on the tonic"
    if form == ANSWER_SINGLE:
        return "a small space to answer in, so one note that lands the tonic"
    return f"the phrase hung on the {cadence.kind} cadence, so the answer walks it down to the tonic"


# ============================================================================
# 5. Singing an answer: the renderer
# ============================================================================

@dataclass
class TakeData:
    name: str
    audio: np.ndarray
    perf: Optional[dict]
    brightness: float = 0.5


class AnswerRenderer:
    """Turns composed answers into audio, in the answering voice.

    The voice is the robot's own: one of the recorded vocable takes from the
    neural bank (the same takes every other mode sings through), sung across
    the whole answer by the WORLD vocoder rather than re-triggered per note.
    That choice is the reason this mode can sound like a sung phrase at all: the
    take's own formants, consonants and breath are kept frame by frame while
    only its pitch is rewritten, so a word 1.2 seconds long is placed across an
    answer 1.2 seconds long and its syllables land where the composed notes do,
    the way a lyric sits over a melody. Nothing is time-stretched, nothing is
    resampled, and the notes of the answer are not copies of the same sample.

    The phrasing on top is what a singer does and a sampler does not: a scoop
    into the first note, a portamento between notes, vibrato that fades in on
    held notes, a slight settle at the end of the final note, and a level that
    dips between notes so the phrase is heard as notes rather than as one
    smeared word.
    """

    def __init__(self, cfg: CallResponseCfg, sample_rate: int, synthesizer=None):
        self.cfg = cfg
        self.sr = int(sample_rate)
        self.synth = synthesizer
        self._takes: dict[str, TakeData] = {}
        self._reverb = None

    # takes
    def _bank_bases(self) -> dict:
        bank = getattr(self.synth, "_neural_bank", None)
        return getattr(bank, "_bases", None) or {}

    def available_takes(self) -> list[str]:
        return sorted(self._bank_bases().keys())

    def measured_lengths(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, bases in self._bank_bases().items():
            if bases:
                try:
                    out[name] = float(len(np.asarray(bases[0].audio))) / self.sr
                except Exception:
                    continue
        return out

    def brightness_map(self) -> dict[str, float]:
        try:
            if self.synth is not None:
                return dict(self.synth.get_vocable_brightness_map() or {})
        except Exception:
            pass
        return {}

    def get_take(self, name: str) -> Optional[TakeData]:
        key = (name or "").strip().lower()
        if not key:
            return None
        if key in self._takes:
            return self._takes[key]
        bases = self._bank_bases().get(key)
        if not bases:
            return None
        try:
            audio = np.asarray(bases[0].audio, dtype=np.float32)
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.cfg.trim_silence:
            audio = _trim_silence(audio, self.sr, self.cfg.trim_floor_db)
        perf = None
        try:
            from synthesis import world_voice
            from synthesis.melody_voice import take_f0
            register = take_f0(audio, self.sr)
            perf = world_voice.analyze_performance(audio, self.sr, register_hz=register)
        except Exception as exc:  # pragma: no cover - WORLD is an optional path
            print(f"[call-response] WORLD analysis unavailable for '{key}' ({exc})")
        data = TakeData(key, audio, perf, brightness=float(self.brightness_map().get(key, 0.5)))
        self._takes[key] = data
        return data

    # one answer
    def render_phrase(self, answer: Answer, voices: int = 1,
                      detune_spread_cents: float = 0.0, timing_jitter_ms: float = 0.0,
                      reverb_amount: float = 0.0) -> Optional[np.ndarray]:
        take = self.get_take(answer.take)
        if take is None:
            return None
        n_samples = int(round(answer.span_s * self.sr))
        if n_samples <= 0 or not answer.notes:
            return None

        if take.perf is not None:
            audio = self._sing_with_world(take, answer, n_samples)
        else:
            audio = self._sing_with_bank(answer, n_samples)
        if audio is None or audio.size == 0:
            return None

        # Normalise by LEVEL, not by peak: a take that opens softly (a word
        # starting on a quiet consonant) would otherwise come out several dB
        # under every other answer, so the answering voice would change
        # loudness from phrase to phrase for no musical reason.
        loud = _sung_rms(audio, self.sr)
        target = self.cfg.answer_peak * 0.42 * float(np.clip(answer.level, 0.2, 1.6))
        if loud > 1e-6:
            audio = audio * (target / loud)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)

        voices = max(1, int(voices))
        if voices > 1:
            audio = self._ensemble(audio, answer, voices, detune_spread_cents, timing_jitter_ms)

        if reverb_amount > 0 and self.synth is not None:
            try:
                wet = np.asarray(self.synth._apply_reverb(audio, float(reverb_amount)), dtype=np.float32)
                audio = wet if wet.shape == audio.shape else audio
            except Exception:
                pass

        return np.asarray(audio, dtype=np.float32)

    def _sing_with_world(self, take: TakeData, answer: Answer, n_samples: int) -> Optional[np.ndarray]:
        from synthesis import world_voice

        curve_hz = _phrase_pitch_curve(answer, n_samples, self.sr, self.cfg)
        cents = 1200.0 * np.log2(np.maximum(curve_hz, 1e-6))
        audio = world_voice.sing_along(take.perf, self.sr, n_samples,
                                       base_hz=1.0, curve_cents=cents)
        if audio is None or not np.any(audio):
            return None
        gain = _phrase_gain_curve(answer, n_samples, self.sr, self.cfg)
        return np.asarray(audio, dtype=np.float64) * gain

    def _sing_with_bank(self, answer: Answer, n_samples: int) -> Optional[np.ndarray]:
        """Fallback for a machine without WORLD: the bank's own per-note renderer.

        Not the version this mode is meant to sound like (each note re-says the
        word rather than one word carrying the phrase), but it keeps the mode
        working, in the right rhythm at the right pitches, on any install.
        """
        if self.synth is None:
            return None
        from synthesis.harmony_engine import HarmonyDecision
        from synthesis.accompaniment_modes import AccompanimentMode

        out = np.zeros(n_samples, dtype=np.float32)
        for note in answer.notes:
            a = int(round(note.start_s * self.sr))
            b = min(n_samples, int(round((note.start_s + note.dur_s) * self.sr)))
            if b - a < 64:
                continue
            decision = HarmonyDecision(
                target_hz=float(note.hz), vocable=answer.take or "aah",
                duration_s=(b - a) / self.sr, vowel_color=0.58, nasality=0.06,
                brightness=0.44, action="sing", mode=AccompanimentMode.CALL_RESPONSE,
                mode_note=answer.reason, texture="solo", num_voices=1,
                detune_spread_cents=0.0, timing_jitter_ms=0.0, formant_spread=0.0,
                reverb_amount=0.0,
            )
            try:
                chunk = np.asarray(self.synth.synthesize(decision, blocking=True, apply_crossfade=False),
                                   dtype=np.float32)
            except Exception:
                continue
            length = min(len(chunk), b - a)
            out[a:a + length] += chunk[:length] * note.accent
        return out

    def _ensemble(self, audio: np.ndarray, answer: Answer, voices: int,
                  detune_spread_cents: float, timing_jitter_ms: float) -> np.ndarray:
        """A texture picker's worth of extra voices on the same phrase.

        Duet and Choir are real choices in the panel and this mode has to honour
        them; the answer becomes a small group singing the same line, each voice
        a few cents off centre and a few milliseconds behind, with the outer
        voices quieter. The phrasing (the curve) is already in the audio, so the
        extra voices are placed as whole samples, exactly like the bank's own
        ensemble does for every other mode.
        """
        spread = float(max(0.0, detune_spread_cents))
        jitter = int(max(0.0, timing_jitter_ms) * self.sr / 1000.0)
        base = audio.copy()
        mixed = base.astype(np.float64)
        peaks = float(np.max(np.abs(mixed))) or 1.0
        for i in range(1, max(1, voices)):
            share = 1.0 / (1.0 + 0.12 * i)
            detune = spread * (i / max(1, voices - 1)) if voices > 1 else 0.0
            shift = _pitch_shift_cents(base, self.sr, detune)
            if jitter > 0:
                shift = _time_shift(shift, int(round(jitter * math.sin(i * 2.399963))))
            mixed += shift * share
        peak = float(np.max(np.abs(mixed)))
        if peak > 0:
            mixed = mixed / peak * peaks
        return mixed.astype(np.float32)

    # the whole track
    def render_track(self, answers: Sequence[Answer], total_samples: int,
                     voices: int = 1, detune_spread_cents: float = 0.0,
                     timing_jitter_ms: float = 0.0, reverb_amount: float = 0.0) -> np.ndarray:
        """All the answers placed on the track's own clock.

        Placement is by absolute sample position, so the timing the answers were
        composed with (enter a beat after the call, land the cadence, stop
        before the next call) is the timing that reaches the mix.
        """
        tail = int(0.6 * self.sr)
        out = np.zeros(max(total_samples, 0) + tail, dtype=np.float32)
        for answer in answers:
            phrase = self.render_phrase(answer, voices=voices, detune_spread_cents=detune_spread_cents,
                                        timing_jitter_ms=timing_jitter_ms, reverb_amount=reverb_amount)
            if phrase is None:
                continue
            start = int(round(answer.start_s * self.sr))
            if start < 0:
                continue
            end = start + len(phrase)
            if end > len(out):
                out = np.pad(out, (0, end - len(out)))
            out[start:end] += phrase
        return out


def _pitch_shift_cents(audio: np.ndarray, sr: int, cents: float) -> np.ndarray:
    if abs(cents) < 0.5:
        return np.asarray(audio, dtype=np.float64)
    try:
        from synthesis.neural_vocable_bank import _formant_preserving_pitch_shift
        return np.asarray(_formant_preserving_pitch_shift(np.asarray(audio, dtype=np.float32),
                                                          sr, cents / 100.0), dtype=np.float64)
    except Exception:
        return np.asarray(audio, dtype=np.float64)


def _sung_rms(audio: np.ndarray, sr: int) -> float:
    """RMS of the part of a finished phrase that is actually sounding.

    The same measurement melody_voice.normalize_loudness uses (-26 dB below the
    phrase's own loud frames), so this agrees with how the rest of the project
    decides how loud a sung part is.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size < 64:
        return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    frame = max(64, int(0.02 * sr))
    n_frames = audio.size // frame
    if n_frames < 2:
        return float(np.sqrt(np.mean(audio ** 2)))
    trimmed = audio[:n_frames * frame].reshape(n_frames, frame)
    level = np.sqrt(np.mean(trimmed ** 2, axis=1))
    gate = float(np.max(level)) * 0.05
    sung = level[level >= gate]
    if sung.size == 0:
        return float(np.sqrt(np.mean(audio ** 2)))
    return float(np.sqrt(np.mean(sung ** 2)))


def _time_shift(audio: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return audio
    out = np.zeros_like(audio)
    if shift > 0:
        out[shift:] = audio[:len(audio) - shift]
    else:
        out[:shift] = audio[-shift:]
    return out


def _trim_silence(audio: np.ndarray, sr: int, floor_db: float = -42.0) -> np.ndarray:
    """Drop the silence either end of a take, so its word starts where the answer does.

    Measured need, not a guess: the phrase is placed by absolute sample
    position, so a take that opens with a tenth of a second of room tone makes
    every answer enter a tenth of a second late, which is exactly the amount
    that turns a response into a late response.
    """
    if audio.size < 64:
        return audio
    frame = max(64, int(0.01 * sr))
    n_frames = audio.size // frame
    if n_frames < 3:
        return audio
    trimmed = audio[:n_frames * frame].reshape(n_frames, frame)
    level = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    peak = float(np.max(level)) or 1.0
    gate = peak * (10.0 ** (floor_db / 20.0))
    loud = np.nonzero(level >= gate)[0]
    if loud.size == 0:
        return audio
    start = int(loud[0]) * frame
    end = min(audio.size, (int(loud[-1]) + 1) * frame)
    if end - start < 0.2 * sr:
        return audio
    return audio[start:end]


def _phrase_pitch_curve(answer: Answer, n_samples: int, sr: int, cfg: CallResponseCfg) -> np.ndarray:
    """The answer's pitch, sample by sample, the way a singer sings it.

    Levels come from the composed notes; between them, a portamento (a singer
    arrives at a new note, never jumps to it); the first note is entered with a
    small scoop from below; held notes carry vibrato that fades in rather than
    switching on; every note drifts a few cents so nothing is digitally flat;
    and the final note settles slightly flat at the very end, which is how a
    sung cadence actually finishes.
    """
    n = max(2, int(n_samples))
    t = np.arange(n, dtype=np.float64) / sr
    level = np.empty(n, dtype=np.float64)

    notes = [note for note in answer.notes if note.dur_s > 0]
    if not notes:
        return np.full(n, 220.0)

    for i, note in enumerate(notes):
        a = int(round(note.start_s * sr))
        b = n if i == len(notes) - 1 else int(round((note.start_s + note.dur_s) * sr))
        a = max(0, min(n - 1, a))
        b = max(a + 1, min(n, b))
        level[a:b] = note.hz
    if n > 1:
        level[:int(round(notes[0].start_s * sr))] = notes[0].hz

    # portamento: ease into each note over the glide time before it, so the
    # new pitch arrives instead of cutting in
    glide = max(2, int(cfg.portamento_s * sr))
    for i, note in enumerate(notes[1:], start=1):
        boundary = int(round(note.start_s * sr))
        previous = notes[i - 1].hz
        if boundary <= 0 or boundary >= n:
            continue
        start = max(0, boundary - glide)
        seg = np.arange(start, boundary, dtype=np.float64)
        if seg.size == 0:
            continue
        u = (seg - start) / max(1.0, float(boundary - start))
        ease = 0.5 - 0.5 * np.cos(np.pi * u)      # raised cosine, fast then settled
        level[start:boundary] = previous + (note.hz - previous) * ease

    curve = level.copy()

    # scoop: the first note is entered from below (a singer's onset, not a
    # switch), only where the answer begins with a real note and not a drone
    first = notes[0]
    if first.dur_s >= 0.30 and first.scoop:
        scoop_len = min(int(cfg.scoop_s * sr), n)
        if scoop_len > 2:
            u = np.linspace(0.0, 1.0, scoop_len)
            ease = 0.5 - 0.5 * np.cos(np.pi * u)
            curve[:scoop_len] *= (1.0 + (cfg.scoop_cents / 1200.0) * (1.0 - ease))

    # vibrato and drift, per note
    for i, note in enumerate(notes):
        a = int(round(note.start_s * sr))
        b = n if i == len(notes) - 1 else int(round((note.start_s + note.dur_s) * sr))
        a = max(0, min(n - 1, a))
        b = max(a + 1, min(n, b))
        dur = (b - a) / sr
        seg_t = t[a:b]
        depth = cfg.vibrato_depth_cents * float(np.clip(dur / 1.1, 0.0, 1.0))
        vib = np.zeros(b - a)
        if depth > 0.3 and (b - a) > int(cfg.vibrato_delay_s * sr):
            onset = int(cfg.vibrato_delay_s * sr)
            ramp = np.zeros(b - a)
            ramp_len = max(1, min(b - a - onset, int(cfg.vibrato_ramp_s * sr)))
            ramp[onset:onset + ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            vib = depth * ramp * np.sin(2 * np.pi * cfg.vibrato_rate_hz * seg_t + i * 0.9)
        drift = cfg.drift_depth_cents * np.sin(2 * np.pi * cfg.drift_rate_hz * seg_t + i * 1.7)
        curve[a:b] *= np.exp2((vib + drift) / 1200.0)

    # the settle at the end of the final note
    fall = min(int(cfg.release_fall_s * sr), n)
    if fall > 2 and answer.notes[-1].dur_s > 0.25:
        u = np.linspace(0.0, 1.0, fall)
        curve[-fall:] *= np.exp2((-cfg.release_fall_cents * u * u) / 1200.0)

    return np.clip(curve, 55.0, 2000.0)


def _phrase_gain_curve(answer: Answer, n_samples: int, sr: int, cfg: CallResponseCfg) -> np.ndarray:
    """The answer's loudness, sample by sample: notes, not one smeared word."""
    n = max(2, int(n_samples))
    gain = np.ones(n, dtype=np.float64)
    notes = answer.notes

    for i, note in enumerate(notes):
        a = int(round(note.start_s * sr))
        b = n if i == len(notes) - 1 else int(round((note.start_s + note.dur_s) * sr))
        a = max(0, min(n - 1, a))
        b = max(a + 1, min(n, b))
        gain[a:b] *= note.accent
        if a > 0:
            attack = min(int(cfg.phrase_attack_s * sr), b - a)
            if attack > 1:
                gain[a:a + attack] *= np.linspace(0.65, 1.0, attack)

    # a small dip at every note boundary inside the phrase: this is the
    # articulation that lets a listener count the notes of the answer
    for i, note in enumerate(notes[1:], start=1):
        boundary = int(round(note.start_s * sr))
        half = max(2, int(cfg.note_dip_s * sr / 2))
        lo, hi = max(0, boundary - half), min(n, boundary + half)
        if hi - lo < 4:
            continue
        u = np.linspace(0.0, 1.0, hi - lo)
        shaft = np.sin(np.pi * u)
        gain[lo:hi] *= 1.0 - cfg.note_dip * shaft

    # the cadence swells into its landing note
    if len(notes) > 1:
        last_start = int(round(notes[-1].start_s * sr))
        approach = max(0, last_start - int(0.18 * sr))
        if last_start - approach > 4:
            gain[approach:last_start] *= np.linspace(0.94, 1.0, last_start - approach)

    # release, so the phrase ends rather than stops
    rel = min(int(cfg.phrase_release_s * sr), n // 2)
    if rel > 2:
        gain[-rel:] *= np.linspace(1.0, 0.0, rel)
    return gain


def render_call_response(times: np.ndarray, pitch: np.ndarray, rms: np.ndarray,
                         frame_hop_s: float, sample_rate: int, total_samples: int,
                         cfg: CallResponseCfg, synthesizer=None, tempo_bpm: float = 0.0,
                         audio: Optional[np.ndarray] = None,
                         voices: int = 1, detune_spread_cents: float = 0.0,
                         timing_jitter_ms: float = 0.0, reverb_amount: float = 0.0,
                         log: bool = True) -> tuple[np.ndarray, list[Answer], TrackAnalysis]:
    """The whole offline mode: analyse, compose, sing, and place.

    Returns the robot's part for this track on its own (one channel, the same
    shape every other mode's offline part has), plus what it decided, so the
    caller can refuse the render honestly when there was nothing to answer.
    """
    track = analyze_track(times, pitch, rms, frame_hop_s, cfg,
                          audio=audio, sr=sample_rate,
                          frame_size=int(round(frame_hop_s * sample_rate)))
    if total_samples <= 0 and len(times):
        total_samples = int(round(times[-1] * sample_rate)) + sample_rate

    renderer = AnswerRenderer(cfg, sample_rate, synthesizer=synthesizer)
    lengths = renderer.measured_lengths()
    brightness = renderer.brightness_map()
    answers = compose_answers(track, cfg, tempo_bpm=tempo_bpm,
                              total_duration_s=total_samples / sample_rate,
                              brightness=brightness,
                              available=renderer.available_takes(),
                              lengths=lengths, log=log)
    part = renderer.render_track(answers, total_samples, voices=voices,
                                 detune_spread_cents=detune_spread_cents,
                                 timing_jitter_ms=timing_jitter_ms, reverb_amount=reverb_amount)
    return part, answers, track


# ============================================================================
# 6. The live path: the same rules, without a future
# ============================================================================

class LiveCallResponder:
    """Call and response one frame at a time, for the microphone.

    The offline engine above can see the whole track, so it knows exactly how
    much room the singer left before the next phrase. A live engine cannot, and
    that single missing fact changes the design: it never quotes the call's
    opening (a long answer it might have to abandon), it answers conservatively,
    and if the singer comes back in before the answer has finished, it drops the
    notes it has not reached instead of singing over them. Everything else is
    the same machinery: the same key, the same cadence reading, the same
    stepwise arrival on the tonic.
    """

    def __init__(self, cfg: Optional[CallResponseCfg] = None):
        self.cfg = cfg or CallResponseCfg()
        self._phrase: list[CallNote] = []
        self._last_voiced_s: Optional[float] = None
        self._schedule: list[AnswerNote] = []
        self._schedule_start_s = 0.0
        self._schedule_reason = ""
        self._key = Key(0, False)
        self._pitch_history: list[float] = []
        self._tuning = 0.0
        self._last_take: Optional[str] = None
        self._responder: Optional[Responder] = None

    # input
    def push(self, now_s: float, hz: Optional[float], beat_s: float = 0.5) -> None:
        """One frame of the live pitch stream.

        Everything this class knows arrives here: a pitch when the singer is
        being tracked, None when they are not. The two decisions that matter -
        "that phrase just ended" and "the singer is back, so stop answering" -
        are both made against the clock, which is all a live engine has.
        """
        if hz and hz > 0:
            if self._last_voiced_s is None:
                self._phrase = []
            elif now_s - self._last_voiced_s > 0.30:
                # a real breath inside what the tracker reported as one phrase:
                # that is two calls, and the first one is already answered
                self._close_phrase(now_s, beat_s)
            self._phrase.append(CallNote(now_s - 0.02, now_s, float(hz), 0.0))
            self._last_voiced_s = now_s
            self._trim_schedule(now_s)
        else:
            if self._last_voiced_s is not None and (now_s - self._last_voiced_s) > self.cfg.phrase_gap_s:
                self._close_phrase(now_s, beat_s)
                self._last_voiced_s = None
            # NOTE: the schedule is deliberately NOT trimmed here. Silence is
            # what the answer is sung into, so treating it as "the singer is
            # back" would cancel every answer in its own lead-in. Only a voiced
            # frame means the singer returned early; see the branch above.

    def _close_phrase(self, now_s: float, beat_s: float) -> None:
        notes = self._coalesce(self._phrase)
        self._phrase = []
        if not notes:
            return
        call = _close_call(notes)
        if call.span_s < self.cfg.min_call_s:
            return
        for note in notes:
            self._pitch_history.append(note.hz)
        if len(self._pitch_history) > 64:
            self._pitch_history = self._pitch_history[-64:]
        self._key = estimate_key(self._pitch_history, tuning_cents=self._tuning)

        cadence = read_cadence(call, self._key)
        gap = float(np.clip(self.cfg.gap_beats * beat_s, self.cfg.gap_min_s, self.cfg.gap_max_s))
        # No lookahead, so the answer is planned against a conservative room: a
        # short phrase that resolves. The single-note form is never used here
        # (too little to hear as an answer from where the singer is standing),
        # and the long quote-and-resolve form is never used either, because
        # quoting a call needs room the live engine cannot promise.
        usable = min(self.cfg.room_medium_s, self.cfg.max_answer_span_s)
        form, span = _choose_form(call, cadence, usable, self.cfg)
        if form in (ANSWER_SINGLE, ANSWER_ECHO):
            form = ANSWER_RESOLVE if cadence.kind not in (CADENCE_TONIC, CADENCE_WEAK_TONIC) else ANSWER_ECHO
            span = self.cfg.short_span_s
        if self._responder is None:
            self._responder = Responder(self._key, self.cfg)
        self._responder.key = self._key
        self._responder._last_take = self._last_take
        answer = self._responder.plan(call, cadence, now_s + gap, usable,
                                      form, span, 1.0, _reason_text(form, cadence))
        if answer is None:
            return
        self._last_take = answer.take
        self._schedule = answer.notes
        self._schedule_start_s = answer.start_s
        self._schedule_reason = answer.reason

    def _trim_schedule(self, now_s: float) -> None:
        """Drop the part of the answer the singer has just sung over.

        Two cases, and they are different: if the answer has not begun yet the
        whole thing goes (a reply is never worth interrupting the person you
        are replying to), and if it is already sounding, the note being sung
        finishes and the notes after it are dropped.
        """
        if not self._schedule:
            return
        elapsed = now_s - self._schedule_start_s
        if elapsed < 0.0:
            self._schedule = []
            return
        kept: list[AnswerNote] = []
        cursor = 0.0
        for note in self._schedule:
            if cursor <= elapsed:
                kept.append(note)
            cursor += note.dur_s
        self._schedule = kept

    def _coalesce(self, notes: list[CallNote]) -> list[CallNote]:
        """The live pitch stream is one reading per frame; group it into notes."""
        out: list[CallNote] = []
        for note in notes:
            if out and abs(1200.0 * math.log2(note.hz / out[-1].hz)) < 70.0:
                out[-1] = CallNote(out[-1].start_s, note.end_s, out[-1].hz, 0.0)
            else:
                out.append(note)
        merged: list[CallNote] = []
        for note in out:
            if merged and note.duration_s < 0.09:
                merged[-1] = CallNote(merged[-1].start_s, note.end_s, merged[-1].hz, 0.0)
            else:
                merged.append(note)
        return merged

    # output
    def step(self, now_s: float, hz: Optional[float], beat_s: float = 0.5
             ) -> tuple[Optional[float], float, str]:
        """One frame in, one sounding note out: push this reading, then answer."""
        self.push(now_s, hz, beat_s)
        return self.current(now_s)

    def current(self, now_s: float) -> tuple[Optional[float], float, str]:
        """(pitch, remaining duration in seconds, what to say about it).

        The text is the panel's own explanation line, so it says what the mode
        is doing right now rather than what it is called: listening to a phrase,
        answering it, or waiting for one.
        """
        if not self._schedule:
            if self._last_voiced_s is not None:
                return None, 0.0, ("Listening to this phrase. It answers after the pause, "
                                   "never over the singing.")
            return None, 0.0, ("Waiting for a phrase to finish. The answer comes after the "
                               "pause, in its own voice, and always ends on the tonic.")
        elapsed = now_s - self._schedule_start_s
        if elapsed < 0.0:
            return None, 0.0, ("Phrase heard. Holding back for a beat so the answer lands "
                               "in the space rather than over the singer.")
        cursor = 0.0
        for note in self._schedule:
            if cursor <= elapsed < cursor + note.dur_s:
                remaining = cursor + note.dur_s - elapsed
                return note.hz, remaining, self._schedule_reason
            cursor += note.dur_s
        if elapsed >= cursor:
            self._schedule = []
        return None, 0.0, "Waiting for the next phrase to answer."

    def reset(self) -> None:
        self._phrase = []
        self._schedule = []
        self._last_voiced_s = None
        self._pitch_history = []

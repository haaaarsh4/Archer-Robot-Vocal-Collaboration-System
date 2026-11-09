"""Turn a chord sequence into an arrangement, audio and a MIDI file.

MySong's paper stops at "a simple, pre-defined pattern of chord-dependent piano
notes" and points at Band-in-a-Box for better audio.  This module goes further:

* a voice-leading search over inversions, so the chordal part does not jump
  around;
* a band per style rather than one block-chord pattern: hand drum, round dance,
  Métis fiddle and cedar flute come first (the music this project is for),
  with pop, ballad, folk, jazz, waltz, ambient and rock as the other bands;
* a small additive piano, a plucked bass, an optional pad and synthesised
  drums, all rendered offline at 44.1 kHz;
* a convolution reverb, vocal ducking and a soft limiter for a mix that can be
  listened to rather than just inspected;
* a Standard MIDI File, so the arrangement can be replaced with real
  instruments.

The synthesis is intentionally dependency-free (numpy + scipy).  It is not a
sampled grand piano; it is a clean, quiet accompaniment that sits under a
voice, which is what the task needs.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Iterable, Sequence

import numpy as np

from mysong.expression import apply_expression
from mysong.harmonize import ChordSegment, Harmonization
from mysong.percussion import render_percussion
from mysong.theory import Chord, voice_leading_cost

# #
# Data
# #

@dataclass
class ArrangementNote:
    start: float       # beats from the start
    duration: float    # beats
    pitch: int         # MIDI
    velocity: int      # 1..127
    instrument: str    # "piano", "guitar", "bass", "pad", "drums"
    channel: int = 0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class ArrangementStyle:
    name: str = "pop"
    intensity: float = 0.7       # 0..1, density and loudness
    swing: float = 0.0           # 0..1, delays off-beats
    humanize: float = 0.5        # 0..1, timing/velocity jitter
    reverb: float = 0.22         # wet amount
    #: How much this band moves from the common mix level.  A dense band at the
    #: same RMS as a sparse one masks more of the voice, so the busiest of the
    #: indigenous bands sits a little further back and the drum-and-bass one a
    #: little closer; everything else stays where ``BAND_OFFSET_DB`` puts it.
    level_db: float = 0.0
    piano: bool = True           # the chordal or melodic slot (see programmes)
    bass: bool = True
    pad: bool = False
    guitar: bool = False         # strummed acoustic guitar, its own slot
    drums: bool = True
    count_in_bars: int = 0
    percussion: str = ""         # "" = the drum kit for this style's name,
    #                              or "hand", "round_dance", "light"
    programmes: dict[str, int] = field(default_factory=dict)  # slot -> GM programme

    #: The bands this project belongs to come first in the app, in the code and
    #: in every list the CLI prints: the hand drum, round dance, grass dance,
    #: jingle dress, Métis fiddle and cedar flute, for a Cree and Métis artist
    #: who has sung with a hand drum since 1982.  The Western kit is available
    #: under "Other bands" and is not the default anywhere.
    INDIGENOUS: ClassVar[tuple[str, ...]] = (
        "hand_drum", "round_dance", "grass_dance", "jingle_dress",
        "metis_fiddle", "cedar_flute",
    )

    @classmethod
    def preset(cls, name: str, **overrides) -> "ArrangementStyle":
        name = (name or "pop").lower().strip()
        presets = {
            # --- the bands this project is for, first in every list --------
            # Hand drum: the heartbeat pulse, a drone under it and two strums
            # that leave the middle of the bar to the voice.  The high
            # humanize is deliberate: a drum a person plays moves, and the
            # grid-perfect pulse is the single biggest reason a generated
            # backing reads as a machine.
            "hand_drum": dict(intensity=0.50, drums=True, pad=False, bass=True,
                              reverb=0.15, humanize=0.78, percussion="hand",
                              level_db=-3.0, programmes={"piano": 25}),
            # Round dance: steady hand drum with rattles on the eighths and a
            # boom-chuck guitar, the shape of a northern round-dance song.
            "round_dance": dict(intensity=0.62, drums=True, pad=False, piano=False,
                                guitar=True, bass=True, reverb=0.16, humanize=0.72,
                                percussion="round_dance"),
            # Grass dance: the big drum on every beat -- the honour beat -- with
            # the dancers' bells on the eighths and a pickup into the next bar.
            # Drum and bass only; the harmony moves under the voices.
            "grass_dance": dict(intensity=0.68, drums=True, pad=False, piano=False,
                                guitar=False, bass=True, reverb=0.17, humanize=0.82,
                                level_db=2.5, percussion="grass_dance"),
            # Jingle dress: the side-step a jingle dancer steps on, a hand drum
            # on the pulse with the cone jingles ringing with each step and a
            # soft boom-chuck guitar holding the chords.
            "jingle_dress": dict(intensity=0.55, drums=True, pad=False, piano=False,
                                 guitar=True, bass=True, reverb=0.20, humanize=0.78,
                                 percussion="jingle_dress"),
            # Métis fiddle: a reel line on the fiddle over a two-step guitar
            # and an upright two-feel bass.  No drum kit.
            # GM 110 is the solo fiddle, not 41's violin section: one player
            # with a bow and a wooden body, which is the instrument this band is.
            "metis_fiddle": dict(intensity=0.58, drums=False, pad=False, guitar=True,
                                 bass=True, reverb=0.18, humanize=0.66,
                                 programmes={"piano": 110}),
            # Cedar flute: long tones with air around them, a soft drum and a
            # warm pad underneath.
            # GM 75 is the pan flute: wood and breath.  Programme 73 is the
            # bright metal flute, which is a different instrument in a
            # different material.
            "cedar_flute": dict(intensity=0.40, drums=True, pad=True, bass=True,
                                reverb=0.30, humanize=0.62, percussion="light",
                                programmes={"piano": 75}),
            # --- other bands -------------------------------------------------
            "pop": dict(intensity=0.60, drums=True, pad=False, reverb=0.20),
            "ballad": dict(intensity=0.46, drums=False, pad=True, reverb=0.32),
            "folk": dict(intensity=0.54, drums=True, pad=False, reverb=0.14),
            "jazz": dict(intensity=0.52, drums=True, pad=False, reverb=0.18, swing=0.55),
            "waltz": dict(intensity=0.48, drums=False, pad=False, reverb=0.22),
            "ambient": dict(intensity=0.32, drums=False, pad=True, reverb=0.45),
            "rock": dict(intensity=0.78, drums=True, pad=False, reverb=0.12),
        }
        if name not in presets:
            raise ValueError(f"unknown style {name!r}; choose from {sorted(presets)}")
        values = dict(presets[name])
        values.update(overrides)
        return cls(name=name, **values)

    @classmethod
    def presets(cls) -> tuple[str, ...]:
        """Every band name, the ones for this project's own music first."""
        return (
            "hand_drum", "round_dance", "grass_dance", "jingle_dress",
            "metis_fiddle", "cedar_flute",
            "pop", "ballad", "folk", "jazz", "waltz", "ambient", "rock",
        )


@dataclass
class Arrangement:
    notes: list[ArrangementNote]
    tempo_bpm: float
    beats_per_bar: float
    harmony: Harmonization
    sample_rate: int = 44100
    duration_beats: float = 0.0
    melody: Sequence | None = None
    style: ArrangementStyle = field(default_factory=ArrangementStyle)

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / max(float(self.tempo_bpm), 1.0)

    @property
    def duration_seconds(self) -> float:
        return max(self.duration_beats, 0.0) * self.seconds_per_beat

    def notes_for(self, instrument: str) -> list[ArrangementNote]:
        return [note for note in self.notes if note.instrument == instrument]

    # -- MIDI ------------------------------------------------------------ #
    def to_midi_bytes(self, *, melody_notes: Sequence | None = None) -> bytes:
        return write_midi(self, melody_notes if melody_notes is not None else self.melody)

    def save_midi(self, path: str, *, melody_notes: Sequence | None = None) -> str:
        data = self.to_midi_bytes(melody_notes=melody_notes)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    # -- audio ----------------------------------------------------------- #
    def render(self, *, sample_rate: int | None = None, seed: int = 0) -> np.ndarray:
        return render_arrangement(self, sample_rate=sample_rate or self.sample_rate, seed=seed)

    def save_wav(self, path: str, *, sample_rate: int | None = None) -> str:
        import soundfile as sf

        audio = self.render(sample_rate=sample_rate)
        sf.write(path, audio, sample_rate or self.sample_rate, subtype="PCM_16")
        return path


# #
# Voicing
# #

def _chord_pitch_classes(chord: Chord) -> list[int]:
    pcs = list(dict.fromkeys(chord.pitch_classes(False)))
    # A four-note voicing sounds fuller than all five tones of a ninth chord;
    # drop the fifth before dropping the seventh or ninth.
    if len(pcs) > 4:
        intervals = [i % 12 for i in chord.intervals]
        keep = []
        for pc in pcs:
            interval = (pc - chord.root) % 12
            if interval == 7 and len(keep) >= 3:
                continue
            keep.append(pc)
        pcs = keep[:4]
    return pcs


def voicing_candidates(
    chord: Chord,
    *,
    low: int = 52,
    high: int = 79,
    max_candidates: int = 8,
) -> list[list[int]]:
    """Inversions and octave placements that fit a comfortable piano register."""
    pcs = _chord_pitch_classes(chord)
    if not pcs:
        return []
    candidates: list[list[int]] = []
    for rotation in range(len(pcs)):
        ordered = pcs[rotation:] + pcs[:rotation]
        for base in (0, 12):
            notes = []
            previous = None
            for pc in ordered:
                pitch = 12 * (base + 4) + pc  # around C4/C5
                while pitch < low:
                    pitch += 12
                if previous is not None:
                    while pitch <= previous:
                        pitch += 12
                notes.append(pitch)
                previous = pitch
            while notes and notes[-1] > high:
                notes[-1] -= 12
            if notes and max(notes) <= high and min(notes) >= low:
                candidates.append(sorted(notes))
    # Deduplicate and rank by compactness.
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    unique.sort(key=lambda notes: (max(notes) - min(notes), abs(sum(notes) / len(notes) - 64)))
    return unique[:max_candidates]


def choose_voicings(
    segments: Sequence[ChordSegment],
    *,
    low: int = 52,
    high: int = 79,
) -> list[list[int]]:
    """Dynamic programme over inversions, minimising voice movement."""
    if not segments:
        return []
    candidates = [voicing_candidates(segment.chord, low=low, high=high) for segment in segments]
    for index, options in enumerate(candidates):
        if not options:
            candidates[index] = [sorted({min(high, 60 + pc) for pc in segments[index].chord.pitch_classes(False)})]

    # dp[i][v] = best cost ending at candidate v of segment i
    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []
    first = candidates[0]
    dp.append(np.asarray([abs(sum(notes) / len(notes) - 64) for notes in first], dtype=np.float64))
    back.append(np.full(len(first), -1, dtype=int))
    for index in range(1, len(segments)):
        options = candidates[index]
        previous = candidates[index - 1]
        costs = np.full((len(previous), len(options)), np.inf)
        for p_index, p_notes in enumerate(previous):
            for c_index, c_notes in enumerate(options):
                costs[p_index, c_index] = dp[-1][p_index] + voice_leading_cost(p_notes, c_notes)
        best_previous = np.argmin(costs, axis=0)
        dp.append(costs[best_previous, np.arange(len(options))])
        back.append(best_previous)
    state = int(np.argmin(dp[-1]))
    chosen: list[list[int]] = []
    for index in range(len(segments) - 1, -1, -1):
        chosen.append(candidates[index][state])
        state = int(back[index][state]) if index > 0 else -1
    chosen.reverse()
    return chosen


# #
# Patterns
# #

def _swing_offset(beat: float, swing: float) -> float:
    """Delay the second eighth of every beat by up to a triplet feel."""
    if swing <= 0:
        return 0.0
    fraction = beat - math.floor(beat)
    return swing * 0.16 * fraction if abs(fraction - 0.5) < 0.26 else 0.0


def _velocity(base: int, intensity: float, jitter: np.random.Generator, humanize: float) -> int:
    value = base * (0.65 + 0.45 * intensity) + jitter.normal(0, 6 * humanize)
    return int(np.clip(round(value), 20, 126))


def _piano_pattern(
    segment: ChordSegment,
    voicing: Sequence[int],
    style: ArrangementStyle,
    meter: float,
    rng: np.random.Generator,
) -> list[ArrangementNote]:
    notes: list[ArrangementNote] = []
    length = segment.duration
    if length <= 0 or not voicing:
        return notes
    start = segment.start_beat
    intensity = style.intensity

    def hit(at: float, duration: float, velocity: int, pitches: Sequence[int] | None = None) -> None:
        offset = _swing_offset(at, style.swing)
        onset = at + offset + rng.normal(0, 0.008 * style.humanize)
        for pitch in (pitches or voicing):
            notes.append(ArrangementNote(
                start=max(0.0, onset),
                duration=max(0.08, duration),
                pitch=int(pitch),
                velocity=_velocity(velocity, intensity, rng, style.humanize),
                instrument="piano",
            ))

    name = style.name
    bar = 0
    while bar * meter < length - 1e-6:
        origin = start + bar * meter
        bar_length = min(meter, start + length - origin)
        if name in ("pop", "rock"):
            hit(origin, min(1.6, bar_length), 92)
            if intensity > 0.45:
                hit(origin + 1.5, 0.45, 72)
            if intensity > 0.65:
                hit(origin + 2.5, 0.8, 80)
            if meter >= 4 and intensity > 0.8:
                hit(origin + 3.5, 0.4, 66)
        elif name == "ballad":
            # Broken chord: root, fifth, tenth, fifth, across the bar.
            ordered = list(voicing)
            pattern = [0, 2, 1, 2, 0, 2, 1, 2]
            step = max(0.5, meter / 8.0)
            for index in range(8):
                at = origin + index * step
                if at >= origin + bar_length - 1e-6:
                    break
                pitch = ordered[pattern[index] % len(ordered)]
                hit(at, step * 1.8, 58 + (8 if index % 4 == 0 else 0), [pitch])
        elif name == "folk":
            hit(origin, 1.0, 86, voicing[1:] if len(voicing) > 1 else voicing)
            for at in (origin + 1.0, origin + 3.0):
                if at < origin + bar_length:
                    hit(at, 0.8, 74)
            if intensity > 0.6:
                hit(origin + 2.0, 0.6, 78, voicing[1:] if len(voicing) > 1 else voicing)
        elif name == "jazz":
            # Charleston-ish comping, with the off-beat delayed by swing.
            hit(origin, 1.4, 78)
            hit(origin + 1.66, 0.7, 68)
            if meter >= 4:
                hit(origin + 2.66, 0.9, 72)
        elif name == "waltz":
            hit(origin, 1.0, 88)
            for at in (origin + 1.0, origin + 2.0):
                if at < origin + bar_length:
                    hit(at, 0.75, 62, voicing[1:] if len(voicing) > 1 else voicing)
        elif name == "ambient":
            hit(origin, max(0.5, bar_length - 0.1), 52)
        elif name == "hand_drum":
            # Two off-beat strums: the middle of the bar belongs to the voice.
            hit(origin + 1.5, 0.4, 56, voicing[:2] if len(voicing) >= 2 else voicing)
            hit(origin + 3.5, 0.4, 48, voicing[:2] if len(voicing) >= 2 else voicing)
        elif name == "metis_fiddle":
            # A reel line instead of block chords: eighths through the chord in
            # the fiddle's own register, so the slot answers the voice rather
            # than competing with it in the middle of the keyboard.
            pcs = sorted({int(pitch) % 12 for pitch in voicing})
            ladder = [pitch for pitch in range(67, 88) if pitch % 12 in pcs] or [67, 69, 71, 72]
            steps = 8 if meter >= 4 else 6
            step = meter / float(steps)
            for index in range(steps):
                at = origin + index * step
                if at >= origin + bar_length - 1e-6:
                    break
                pitch = ladder[(index * 3) % len(ladder)]
                hit(at, step * 0.92, 66 + (10 if index % 4 == 0 else 0), [pitch])
        elif name == "cedar_flute":
            # Long tones with air around them: the root, then a tone above it.
            ordered = list(voicing)
            hit(origin, min(bar_length, meter) * 0.9, 64, [ordered[0]])
            if meter >= 4 and bar_length > meter * 0.8 and len(ordered) > 2:
                hit(origin + meter * 0.5, meter * 0.4, 58, [ordered[2]])
        else:  # pragma: no cover - preset validation catches this
            hit(origin, min(1.5, bar_length), 80)
        bar += 1
    return notes


def _guitar_pattern(
    segment: ChordSegment,
    voicing: Sequence[int],
    style: ArrangementStyle,
    meter: float,
    rng: np.random.Generator,
) -> list[ArrangementNote]:
    """Strummed acoustic guitar: a boom-chuck, not a block chord.

    Each string lands a few milliseconds after the one below it, which is most
    of what separates a strummed chord from a triggered one.
    """
    notes: list[ArrangementNote] = []
    if not voicing or segment.duration <= 0:
        return notes
    start = segment.start_beat
    length = segment.duration
    strings = [int(pitch) for pitch in voicing if pitch >= 52] or [int(pitch) for pitch in voicing]
    low = int(min(voicing)) - 12

    def strum(at: float, duration: float, velocity: int, pitches: Sequence[int]) -> None:
        onset = at + rng.normal(0, 0.007 * style.humanize)
        for index, pitch in enumerate(pitches):
            notes.append(ArrangementNote(
                start=max(0.0, onset + index * 0.011),
                duration=max(0.1, duration),
                pitch=int(pitch),
                velocity=_velocity(velocity - index * 3, style.intensity, rng, style.humanize),
                instrument="guitar",
                channel=3,
            ))

    bar = 0
    while bar * meter < length - 1e-6:
        origin = start + bar * meter
        bar_length = min(meter, start + length - origin)
        if style.name == "metis_fiddle":
            # Two-step: a bass note on 1 and 3, the chord on 2 and 4.
            strum(origin, 0.5, 78, [low])
            if bar_length > 1.0:
                strum(origin + 1.0, 0.4, 70, strings)
            if bar_length > 2.0:
                strum(origin + 2.0, 0.5, 76, [low])
            if bar_length > 3.0:
                strum(origin + 3.0, 0.4, 68, strings)
        else:
            # Round dance: low note on the beat, chord on the off-beat.
            for index in range(max(1, int(round(meter)))):
                at = origin + index
                if at >= origin + bar_length - 1e-6:
                    break
                if index % 2 == 0:
                    strum(at, 0.35, 62, [low])
                elif at + 0.5 < origin + bar_length:
                    strum(at + 0.5, 0.3, 66, strings)
        bar += 1
    return notes


def _bass_pattern(
    segment: ChordSegment,
    next_chord: Chord | None,
    style: ArrangementStyle,
    meter: float,
    rng: np.random.Generator,
) -> list[ArrangementNote]:
    notes: list[ArrangementNote] = []
    root = segment.chord.bass if segment.chord.bass is not None else segment.chord.root
    base = 36 + (root % 12)
    while base < 36:
        base += 12
    fifth = 36 + ((root + 7) % 12)
    while fifth < 40:
        fifth += 12
    third = 36 + ((root + (4 if segment.chord.core_quality == "maj" else 3)) % 12)
    while third < 40:
        third += 12

    def add(at: float, duration: float, pitch: int, velocity: int) -> None:
        onset = at + rng.normal(0, 0.006 * style.humanize)
        notes.append(ArrangementNote(
            start=max(0.0, onset),
            duration=max(0.08, duration),
            pitch=int(pitch),
            velocity=_velocity(velocity, style.intensity, rng, style.humanize),
            instrument="bass",
            channel=1,
        ))

    length = segment.duration
    start = segment.start_beat
    name = style.name
    if name in ("hand_drum", "cedar_flute"):
        # One low root per chord: a drone to sing over, not a bass line.
        add(start, max(0.75, length - 0.05), base, 66)
    elif name in ("grass_dance", "jingle_dress"):
        # The drum's own pulse: one low root per beat, the fifth answering it,
        # so the harmony moves with the dancers instead of walking anywhere.
        beats = max(1, int(math.ceil(length - 1e-6)))
        for index in range(beats):
            at = start + index
            if at >= start + length - 1e-6:
                break
            add(at, min(0.9, start + length - at),
                base if index % 2 == 0 else fifth,
                82 if index % 2 == 0 else 66)
    elif name == "round_dance":
        add(start, min(1.9, max(0.5, length)), base, 82)
        if length > 2.0:
            add(start + 2.0, min(1.8, length - 2.0), fifth, 72)
    elif name == "metis_fiddle":
        add(start, min(1.8, max(0.5, length)), base, 84)
        if length > 2.0:
            add(start + 2.0, min(1.7, length - 2.0), fifth, 74)
    elif name == "jazz":
        # Walking quarter notes, with a chromatic approach into the next root.
        approach = None
        if next_chord is not None:
            target = next_chord.root
            candidates = [(target - 1) % 12, (target + 1) % 12, (target + 7) % 12]
            approach = 36 + min(candidates, key=lambda pc: abs(((pc - root) + 6) % 12 - 6))
        steps = [base, third, fifth, approach if approach is not None else base]
        for index in range(int(math.ceil(length))):
            at = start + index
            if at >= start + length - 1e-6:
                break
            add(at, 0.92, steps[index % len(steps)], 76)
    elif name == "waltz":
        add(start, 0.95, base, 82)
        for beat in (1.0, 2.0):
            if beat < length:
                add(start + beat, 0.8, fifth if beat < 2 else third, 58)
    elif name == "ballad":
        add(start, min(2.0, length), base, 78)
        if length > 2:
            add(start + 2.0, min(2.0, length - 2.0), fifth, 64)
    elif name == "ambient":
        add(start, max(1.0, length), base, 62)
    else:
        add(start, min(1.8, length), base, 84)
        if length > 2:
            add(start + 2.0, min(1.6, length - 2.0), fifth if style.name != "rock" else base, 72)
        if length > 3.5 and style.name in ("pop", "rock"):
            add(start + 3.5, 0.45, third, 62)
    return notes


def _drum_pattern(
    total_beats: float,
    meter: float,
    style: ArrangementStyle,
    rng: np.random.Generator,
) -> list[ArrangementNote]:
    notes: list[ArrangementNote] = []
    if not style.drums or style.name == "ambient":
        return notes

    def hit(at: float, instrument: str, velocity: int) -> None:
        pitch = {
            "kick": 36, "snare": 38, "hat": 42, "open_hat": 46, "ride": 51,
            "crash": 49, "tom_low": 45, "tom_mid": 47, "tom_high": 50,
            "shaker": 69, "rattle": 54,
        }[instrument]
        onset = at + rng.normal(0, 0.006 * style.humanize)
        notes.append(ArrangementNote(
            start=max(0.0, onset), duration=0.25, pitch=pitch,
            velocity=_velocity(velocity, style.intensity, rng, style.humanize),
            instrument="drums", channel=9,
        ))

    bars = int(math.ceil(total_beats / meter))
    for bar in range(bars):
        origin = bar * meter
        if origin >= total_beats:
            break
        if not style.percussion and (bar == 0 or bar % 8 == 0):
            # A cymbal is the Western kit's punctuation; the drum-led bands
            # below never ask for one.
            hit(origin, "crash", 74)
        if style.percussion == "hand":
            # The heartbeat a hand drum keeps under a song: an accented low
            # stroke on the beat, a lighter answer to it, and space between
            # them for the voice.
            for beat in range(max(1, int(round(meter)))):
                at = origin + beat
                if at >= total_beats:
                    break
                accent = beat % 2 == 0
                # Under a voice the drum is played, not struck for its own
                # sake: the accent carries the pulse and the answer stays
                # under the consonants instead of on top of them.
                hit(at, "tom_low", 82 if accent else 58)
                if not accent:
                    hit(at + 0.5, "tom_mid", 36)
        elif style.percussion == "round_dance":
            # Steady pulse with rattles on the eighths.
            for index in range(max(2, int(round(meter * 2)))):
                at = origin + index * 0.5
                if at >= total_beats:
                    break
                if index % 2 == 0:
                    hit(at, "tom_low", 94 if index % 4 == 0 else 80)
                else:
                    hit(at, "tom_mid", 52)
                hit(at, "shaker", 44 if index % 2 == 0 else 34)
        elif style.percussion == "grass_dance":
            # The big drum on every beat (the honour beat), the dancers' bells on
            # the eighths and a pickup into the next bar.  No cymbal: the
            # drum-led bands in this list never ask for one.
            for beat in range(max(1, int(round(meter)))):
                at = origin + beat
                if at >= total_beats:
                    break
                accent = beat % 2 == 0
                hit(at, "tom_low", 104 if accent else 88)
                if not accent and bar % 2:
                    hit(at + 0.5, "tom_mid", 58)
            for index in range(max(2, int(round(meter * 2)))):
                at = origin + index * 0.5
                if at >= total_beats:
                    break
                hit(at, "rattle", 50 if index % 2 == 0 else 38)
            if bar % 4 == 3 and meter >= 4:
                for offset, drum, velocity in (
                    (3.0, "tom_mid", 74), (3.5, "tom_high", 82),
                ):
                    if origin + offset < total_beats:
                        hit(origin + offset, drum, velocity)
        elif style.percussion == "jingle_dress":
            # The side-step: a hand drum on the pulse with the cone jingles
            # ringing on every step the dancer takes, and a lighter shift of
            # weight on the half beat between them.  A dress full of cones does
            # not stop ringing between steps -- the jingles of the step that has
            # just landed are still sounding under the next one -- so both the
            # beat and its half carry cones, never a hi-hat pattern.
            for beat in range(max(1, int(round(meter)))):
                at = origin + beat
                if at >= total_beats:
                    break
                hit(at, "tom_low", 92 if beat % 2 == 0 else 68)
                hit(at, "shaker", 58 if beat % 2 == 0 else 48)
                if at + 0.5 < total_beats:
                    hit(at + 0.5, "shaker", 30 if beat % 2 == 0 else 38)
        elif style.percussion == "light":
            # Two soft strokes a bar and a quiet rattle, under a flute.
            hit(origin, "tom_low", 68)
            if meter >= 4:
                hit(origin + 2.0, "tom_low", 58)
            for index in range(max(2, int(round(meter * 2)))):
                at = origin + index * 0.5
                if at < total_beats:
                    hit(at, "shaker", 36 if index % 2 == 0 else 30)
        elif style.name == "jazz":
            for beat in np.arange(0, meter, 0.5):
                if origin + beat >= total_beats:
                    break
                hit(origin + beat + _swing_offset(beat, 0.66), "ride", 58 if beat % 1 else 66)
            for beat in (1.0, 3.0):
                if beat < meter and origin + beat < total_beats:
                    hit(origin + beat, "snare", 38)
            if meter >= 3:
                for beat in (0.0, 2.0):
                    if origin + beat < total_beats:
                        hit(origin + beat, "hat", 46)
        elif style.name == "waltz":
            hit(origin, "kick", 70)
            for beat in (1.0, 2.0):
                if origin + beat < total_beats:
                    hit(origin + beat, "snare", 40)
        elif style.name == "rock":
            hit(origin, "kick", 96)
            if meter >= 4:
                hit(origin + 2.5, "kick", 80)
            for beat in (1.0, 3.0):
                if beat < meter and origin + beat < total_beats:
                    hit(origin + beat, "snare", 88)
            for beat in np.arange(0, meter, 0.5):
                if origin + beat < total_beats:
                    hit(origin + beat, "hat", 56 if beat % 1 else 66)
            if bar % 4 == 3 and meter >= 4:
                for offset, drum, velocity in (
                    (2.5, "tom_high", 72), (3.0, "tom_mid", 76),
                    (3.5, "tom_low", 82),
                ):
                    if origin + offset < total_beats:
                        hit(origin + offset, drum, velocity)
        else:  # pop / folk / ballad grid
            hit(origin, "kick", 84)
            if meter >= 4:
                hit(origin + 2.0, "kick", 70)
            for beat in (1.0, 3.0):
                if beat < meter and origin + beat < total_beats:
                    hit(origin + beat, "snare", 66 if style.name == "ballad" else 78)
            if style.intensity > 0.55:
                for beat in np.arange(0, meter, 0.5):
                    if origin + beat < total_beats:
                        hit(origin + beat, "hat", 48 if beat % 1 else 58)
            if bar % 4 == 3 and meter >= 4:
                for offset, drum, velocity in (
                    (2.5, "tom_high", 64), (3.0, "tom_mid", 70),
                    (3.5, "tom_low", 76),
                ):
                    if origin + offset < total_beats:
                        hit(origin + offset, drum, velocity)
    return notes


# #
# Arrangement assembly
# #

def arrange(
    harmony: Harmonization,
    *,
    style: ArrangementStyle | str | None = None,
    melody=None,
    count_in: bool = False,
) -> Arrangement:
    if style is None:
        style = ArrangementStyle()
    elif isinstance(style, str):
        style = ArrangementStyle.preset(style)
    style = ArrangementStyle(**{**style.__dict__})
    if count_in:
        style.count_in_bars = max(1, style.count_in_bars)
    if harmony.beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be positive")
    meter = float(harmony.beats_per_bar)
    rng = np.random.default_rng(7)
    voicings = choose_voicings(harmony.segments)
    notes: list[ArrangementNote] = []
    for index, segment in enumerate(harmony.segments):
        next_chord = harmony.segments[index + 1].chord if index + 1 < len(harmony.segments) else None
        if style.piano:
            notes.extend(_piano_pattern(segment, voicings[index], style, meter, rng))
        if style.bass:
            notes.extend(_bass_pattern(segment, next_chord, style, meter, rng))
        if style.guitar:
            notes.extend(_guitar_pattern(segment, voicings[index], style, meter, rng))
        if style.pad:
            for pitch in voicings[index]:
                notes.append(ArrangementNote(
                    start=segment.start_beat,
                    duration=max(0.5, segment.duration - 0.05),
                    pitch=int(pitch + 12),
                    velocity=_velocity(52, style.intensity, rng, style.humanize),
                    instrument="pad",
                ))
    total = harmony.duration_beats
    if style.drums:
        notes.extend(_drum_pattern(total, meter, style, rng))
    notes.sort(key=lambda note: (note.start, note.instrument, note.pitch))
    return Arrangement(
        notes=notes,
        tempo_bpm=float(harmony.tempo_bpm or 100.0),
        beats_per_bar=meter,
        harmony=harmony,
        duration_beats=total,
        melody=melody,
        style=style,
    )


# #
# Synthesis
# #

def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """A compact ADSR envelope for one note of ``n`` samples."""
    attack_n = max(1, int(attack * sr))
    decay_n = max(1, int(decay * sr))
    release_n = max(1, int(release * sr))
    if attack_n + decay_n + release_n >= n:
        # Short note: scale the stages proportionally.
        scale = n / float(attack_n + decay_n + release_n + 1)
        attack_n = max(1, int(attack_n * scale))
        decay_n = max(1, int(decay_n * scale))
        release_n = max(1, n - attack_n - decay_n)
    sustain_n = max(0, n - attack_n - decay_n - release_n)
    env = np.concatenate([
        np.linspace(0.0, 1.0, attack_n, endpoint=False),
        np.linspace(1.0, sustain, decay_n, endpoint=False),
        np.full(sustain_n, sustain),
        np.linspace(sustain, 0.0, release_n),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def _add(buffer: np.ndarray, signal: np.ndarray, start: int, gain: float = 1.0, pan: float = 0.0) -> None:
    if start >= len(buffer) or len(signal) == 0:
        return
    end = min(len(buffer), start + len(signal))
    if end <= 0:
        return
    left = gain * math.sqrt(max(0.0, 0.5 * (1.0 - pan)))
    right = gain * math.sqrt(max(0.0, 0.5 * (1.0 + pan)))
    buffer[start:end, 0] += signal[: end - start] * left
    buffer[start:end, 1] += signal[: end - start] * right


def _piano_note(midi: int, velocity: int, seconds: float, sr: int) -> np.ndarray:
    n = max(1, int(seconds * sr))
    t = np.arange(n) / sr
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    amplitude = (velocity / 127.0) ** 1.4
    tone = np.zeros(n)
    # Piano strings are slightly inharmonic; a touch of that is what keeps an
    # additive tone from sounding like an organ.
    inharmonicity = 0.00035
    for harmonic in range(1, 9):
        frequency = f0 * harmonic * math.sqrt(1.0 + inharmonicity * harmonic * harmonic)
        if frequency >= sr / 2.0 - 200:
            break
        decay = 2.2 + 0.55 * harmonic
        envelope = np.exp(-t * decay) * (1.0 - np.exp(-t * 900.0))
        tone += (amplitude / (harmonic ** 1.15)) * envelope * np.sin(2 * np.pi * frequency * t)
    # A short hammer noise transient.
    noise_n = min(n, int(0.012 * sr))
    if noise_n > 2:
        noise = np.random.default_rng(int(midi) * 7919 + velocity).normal(0.0, 1.0, noise_n)
        noise *= np.linspace(0.08, 0.0, noise_n) * amplitude
        tone[:noise_n] += noise
    tone *= _adsr(n, sr, 0.004, 0.05, 0.55, 0.32)
    return tone.astype(np.float32)


def _bass_note(midi: int, velocity: int, seconds: float, sr: int) -> np.ndarray:
    n = max(1, int(seconds * sr))
    t = np.arange(n) / sr
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    amplitude = (velocity / 127.0) ** 1.3
    tone = np.sin(2 * np.pi * f0 * t)
    tone += 0.28 * np.sin(2 * np.pi * 2 * f0 * t)
    tone += 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
    # A sub-octave gives body without mud at these pitches.
    tone += 0.18 * np.sin(2 * np.pi * 0.5 * f0 * t)
    tone *= _adsr(n, sr, 0.008, 0.08, 0.62, 0.18)
    return (amplitude * tone).astype(np.float32)


def _pad_note(midi: int, velocity: int, seconds: float, sr: int, detune_cents: float = 7.0) -> np.ndarray:
    n = max(1, int(seconds * sr))
    t = np.arange(n) / sr
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    tone = np.zeros(n)
    for offset in (-detune_cents, 0.0, detune_cents):
        f = f0 * 2.0 ** (offset / 1200.0)
        tone += 0.35 * np.sin(2 * np.pi * f * t)
        tone += 0.15 * np.sin(2 * np.pi * 2 * f * t)
        tone += 0.06 * np.sin(2 * np.pi * 3 * f * t)
    tone *= _adsr(n, sr, 0.35, 0.2, 0.7, 0.55)
    return ((velocity / 127.0) * 0.4 * tone).astype(np.float32)


#: General MIDI drum note -> the fallback synth's voice.  This is the numpy
#: engine only: the indigenous bands keep their notes but are played by
#: mysong.percussion, and a note with no entry here is skipped rather than
#: raising, so a pattern may write a voice the fallback does not have.
_DRUM_VOICES = {
    36: "kick", 38: "snare", 42: "hat", 46: "open_hat", 51: "ride",
    49: "crash", 45: "tom_low", 47: "tom_mid", 50: "tom_high",
    69: "shaker", 54: "rattle",
}


def _drum_hit(kind: str, velocity: int, sr: int) -> np.ndarray:
    rng = np.random.default_rng({
        "kick": 1, "snare": 2, "hat": 3, "open_hat": 4, "ride": 5,
        "crash": 6, "tom_low": 7, "tom_mid": 8, "tom_high": 9,
        "shaker": 10, "rattle": 11,
    }[kind])
    amplitude = (velocity / 127.0) ** 1.3
    if kind == "kick":
        n = int(0.36 * sr)
        t = np.arange(n) / sr
        f = 120.0 * np.exp(-t * 18.0) + 45.0
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = np.sin(phase) * np.exp(-t * 9.0)
        click = rng.normal(0, 1, n) * np.exp(-t * 180.0) * 0.25
        return (amplitude * (tone + click)).astype(np.float32)
    if kind == "snare":
        n = int(0.22 * sr)
        t = np.arange(n) / sr
        noise = rng.normal(0, 1, n) * np.exp(-t * 22.0)
        tone = 0.35 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30.0)
        return (amplitude * 0.7 * (noise + tone)).astype(np.float32)
    if kind in ("hat", "open_hat"):
        n = int((0.055 if kind == "hat" else 0.28) * sr)
        t = np.arange(n) / sr
        noise = rng.normal(0, 1, n)
        # Crude high-pass: difference of the noise with its short moving mean.
        kernel = np.ones(12) / 12.0
        noise = noise - np.convolve(noise, kernel, mode="same")
        return (amplitude * 0.35 * noise * np.exp(-t * (70.0 if kind == "hat" else 14.0))).astype(np.float32)
    if kind in ("shaker", "rattle"):
        # A gourd rattle and a cone cluster are played by mysong.percussion for
        # the bands that use them; this is only the numpy fallback for a machine
        # without FluidSynth, and it must not raise for a note the patterns emit.
        n = int((0.12 if kind == "shaker" else 0.2) * sr)
        t = np.arange(n) / sr
        noise = rng.normal(0, 1, n)
        kernel = np.ones(4) / 4.0
        noise = noise - np.convolve(noise, kernel, mode="same")
        return (amplitude * 0.3 * noise
                * np.exp(-t * (34.0 if kind == "shaker" else 20.0))).astype(np.float32)
    if kind in ("tom_low", "tom_mid", "tom_high"):
        base = {"tom_low": 110.0, "tom_mid": 145.0, "tom_high": 190.0}[kind]
        n = int(0.35 * sr)
        t = np.arange(n) / sr
        f = base * (1.0 + 0.55 * np.exp(-t * 22.0))
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = np.sin(phase) * np.exp(-t * 8.0)
        click = rng.normal(0, 1, n) * np.exp(-t * 90.0) * 0.12
        return (amplitude * 0.8 * (tone + click)).astype(np.float32)
    if kind == "crash":
        n = int(1.5 * sr)
        t = np.arange(n) / sr
        noise = rng.normal(0, 1, n)
        noise -= np.convolve(noise, np.ones(5) / 5.0, mode="same")
        tone = sum(0.12 * np.sin(2 * np.pi * f * t) * np.exp(-t * 3.0)
                   for f in (430, 690, 1010, 1470))
        return (amplitude * 0.28 * (noise + tone) * np.exp(-t * 2.4)).astype(np.float32)
    # Ride: a bright, ringing wash.
    n = int(0.8 * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0, 1, n)
    noise -= np.convolve(noise, np.ones(8) / 8.0, mode="same")
    tone = sum(0.18 * np.sin(2 * np.pi * f * t) * np.exp(-t * 6.0) for f in (520, 790, 1180))
    return (amplitude * 0.32 * (noise + tone) * np.exp(-t * 4.0)).astype(np.float32)


def _synthetic_ir(sr: int, seconds: float = 1.9, seed: int = 11) -> np.ndarray:
    """A small stereo impulse response: early reflections plus a diffuse tail."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = np.zeros((n, 2), dtype=np.float32)
    for channel in range(2):
        noise = rng.normal(0, 1, n)
        envelope = np.exp(-t * 3.2) * (1.0 - np.exp(-t * 60.0))
        ir[:, channel] = (noise * envelope).astype(np.float32)
        for delay, gain in ((0.011, 0.35), (0.023, 0.22), (0.037, 0.15), (0.053, 0.10)):
            index = int(delay * sr) + channel * 17
            if index < n:
                ir[index, channel] += gain
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True) + 1e-9)
    return ir


def _convolve_reverb(audio: np.ndarray, sr: int, wet: float) -> np.ndarray:
    if wet <= 0.001 or len(audio) == 0:
        return audio
    from scipy.signal import fftconvolve

    ir = _synthetic_ir(sr)
    out = np.zeros_like(audio)
    for channel in range(2):
        out[:, channel] = fftconvolve(audio[:, channel], ir[:, channel], mode="full")[: len(audio)]
    return (1.0 - 0.35 * wet) * audio + wet * out


def _soft_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return np.tanh(audio * 1.02) / math.tanh(1.02)


def _render_arrangement_synth(
    arrangement: Arrangement,
    *,
    sample_rate: int = 44100,
    seed: int = 0,
) -> np.ndarray:
    """The dependency-free fallback: additive piano, bass, pad and drums."""
    spb = arrangement.seconds_per_beat
    total_beats = arrangement.duration_beats + 4.0  # let the last chord ring
    total_samples = max(1, int(total_beats * spb * sample_rate) + sample_rate // 2)
    buffer = np.zeros((total_samples, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    # The hand drum and the cone jingles of a jingle dress are played by
    # mysong.percussion.  No General MIDI bank has either instrument, which is
    # why these bands used to come back as a tom and a cabasa.
    percussion = render_percussion(arrangement, sample_rate=sample_rate, seed=seed + 1)
    for note in arrangement.notes:
        start = max(0, int(note.start * spb * sample_rate))
        seconds = max(0.05, note.duration * spb + 0.35)
        if note.instrument == "piano":
            signal = _piano_note(note.pitch, note.velocity, seconds, sample_rate)
            pan = float(np.clip((note.pitch - 62) / 45.0, -0.45, 0.45))
            _add(buffer, signal, start, gain=0.42, pan=pan)
        elif note.instrument == "bass":
            signal = _bass_note(note.pitch, note.velocity, seconds, sample_rate)
            _add(buffer, signal, start, gain=0.55, pan=0.0)
        elif note.instrument == "guitar":
            signal = _piano_note(note.pitch, note.velocity, seconds * 0.8, sample_rate)
            _add(buffer, signal, start, gain=0.30, pan=float(np.clip((note.pitch - 62) / 45.0, -0.5, 0.5)))
        elif note.instrument == "pad":
            signal = _pad_note(note.pitch, note.velocity, seconds + 0.8, sample_rate)
            _add(buffer, signal, start, gain=0.22, pan=float(np.clip((note.pitch - 60) / 50.0, -0.6, 0.6)))
        elif note.instrument == "drums":
            if percussion is not None:
                continue
            drum = _DRUM_VOICES.get(note.pitch)
            if drum is None:
                continue
            signal = _drum_hit(drum, note.velocity, sample_rate)
            _add(buffer, signal, start, gain=0.5, pan=0.0)
    if percussion is not None:
        span = min(len(buffer), len(percussion))
        buffer[:span] += percussion[:span]
    # A touch of stereo width from a short decorrelation delay.
    delay = int(0.012 * sample_rate)
    if len(buffer) > delay:
        buffer[delay:, 1] += 0.06 * buffer[:-delay, 0]
    buffer = _convolve_reverb(buffer, sample_rate, arrangement.style.reverb)
    # Gentle bus compression, then a limiter.
    envelope = np.abs(buffer).max(axis=1)
    attack = 0.02
    release = 0.25
    smoothed = np.copy(envelope)
    for index in range(1, len(smoothed)):
        coefficient = math.exp(-1.0 / (sample_rate * (attack if smoothed[index] > smoothed[index - 1] else release)))
        smoothed[index] = coefficient * smoothed[index - 1] + (1 - coefficient) * envelope[index]
    gain = np.where(smoothed > 0.25, 0.25 / (smoothed + 1e-6), 1.0)
    gain = np.clip(gain, 0.55, 1.0)
    buffer *= gain[:, None]
    buffer = _soft_limit(buffer)
    fade = min(len(buffer), int(0.05 * sample_rate))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        buffer[:fade] *= ramp
        buffer[-fade:] *= ramp[::-1]
    return buffer.astype(np.float32)


# #
# SoundFont rendering
# #

_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf3/default-GM.sf3",
)


def find_soundfont(path: str | None = None) -> str | None:
    """Locate a General MIDI SoundFont without requiring one in the repo."""
    requested = path or os.environ.get("MYSONG_SOUNDFONT")
    if requested and Path(requested).exists():
        return str(requested)
    for candidate in _SOUNDFONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def soundfont_available(path: str | None = None) -> bool:
    return shutil.which("fluidsynth") is not None and find_soundfont(path) is not None


def _render_midi(
    midi_path: Path,
    raw_path: Path,
    soundfont: str,
    sample_rate: int,
    gain: float,
) -> np.ndarray:
    """Render one MIDI file through FluidSynth and return a stereo buffer.

    ``-F`` fast-renders **headerless signed 16-bit stereo PCM**, not WAV.
    Requesting ``-O s16`` makes that explicit and the samples are decoded here.
    """
    command = [
        shutil.which("fluidsynth") or "fluidsynth",
        "-ni",
        "-F", str(raw_path),
        "-O", "s16",
        "-r", str(int(sample_rate)),
        "-g", f"{float(gain):.4f}",
        "-o", "synth.reverb.active=0",
        "-o", "synth.chorus.active=0",
        soundfont,
        str(midi_path),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or not raw_path.exists():
        message = completed.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"fluidsynth failed: {message}")
    samples = np.fromfile(str(raw_path), dtype="<i2")
    if samples.size % 2:
        samples = samples[:-1]
    audio = samples.reshape(-1, 2).astype(np.float32) / 32768.0
    return audio


def _stem_gain(instrument: str) -> float:
    # FluidSynth's master gain is conservative; these levels put the sampled
    # stems in the same range as the additive engine before bus compression.
    return {"piano": 1.55, "guitar": 1.30, "bass": 1.35, "pad": 0.82, "drums": 1.22}.get(instrument, 0.9)


def render_arrangement_soundfont(
    arrangement: Arrangement,
    *,
    sample_rate: int = 44100,
    seed: int = 0,
    soundfont: str | None = None,
) -> np.ndarray:
    """Render the arrangement with a real General MIDI instrument bank.

    Each instrument is rendered as its own MIDI stem so the piano, bass, pad
    and drums can be balanced and processed separately.  This is the path that
    makes the output sound like an instrument rather than an additive organ;
    the numpy synth remains the fallback when FluidSynth is absent.
    """
    sf_path = find_soundfont(soundfont)
    if sf_path is None or shutil.which("fluidsynth") is None:
        raise RuntimeError("no SoundFont renderer available")

    spb = arrangement.seconds_per_beat
    total_beats = arrangement.duration_beats + 4.0
    total_samples = max(1, int(total_beats * spb * sample_rate) + sample_rate // 2)
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    # A hide drum and a dress full of brass cones exist in no General MIDI bank,
    # so those two instruments are rendered from their acoustics and summed with
    # the sampled stems below (see mysong.percussion).
    percussion = render_percussion(arrangement, sample_rate=sample_rate, seed=seed + 1)

    with tempfile.TemporaryDirectory(prefix="mysong-sf2-") as tmp:
        tmp_path = Path(tmp)
        present = {note.instrument for note in arrangement.notes}
        for instrument in [slot for slot in ("piano", "guitar", "bass", "pad", "drums")
                           if slot in present
                           and not (slot == "drums" and percussion is not None)]:
            notes = arrangement.notes_for(instrument)
            if not notes:
                continue
            # Give the piano a little more release and the pad a longer tail;
            # MIDI note-off otherwise clips a sampled decay much earlier than
            # the same gesture would ring on a real instrument.
            if instrument == "piano":
                notes = [replace(note, duration=note.duration + 0.18) for note in notes]
            elif instrument == "guitar":
                notes = [replace(note, duration=note.duration + 0.10) for note in notes]
            elif instrument == "pad":
                notes = [replace(note, duration=note.duration + 0.5) for note in notes]
            stem_arrangement = replace(arrangement, notes=notes)
            midi_path = tmp_path / f"{instrument}.mid"
            raw_path = tmp_path / f"{instrument}.raw"
            stem_arrangement.save_midi(str(midi_path), melody_notes=None)
            stem = _render_midi(
                midi_path, raw_path, sf_path, sample_rate, _stem_gain(instrument)
            )
            # Trim or pad the stem to the arrangement length.  FluidSynth adds
            # its own release tail, which is welcome but must not extend the
            # project past the last beat indefinitely.
            if len(stem) > total_samples:
                stem = stem[:total_samples]
            elif len(stem) < total_samples:
                stem = np.pad(stem, ((0, total_samples - len(stem)), (0, 0)))
            if instrument == "piano":
                # The MIDI has one piano channel, so build a little width from
                # a decorrelated delay instead of hard-panning every note.
                delay = int(0.008 * sample_rate)
                if len(stem) > delay:
                    stem[delay:, 1] += 0.16 * stem[:-delay, 0]
                    stem[delay:, 0] += 0.05 * stem[:-delay, 1]
            elif instrument == "pad":
                delay = int(0.017 * sample_rate)
                if len(stem) > delay:
                    stem[delay:, 1] += 0.22 * stem[:-delay, 0]
            elif instrument == "guitar":
                # A strummed guitar wants width; a short decorrelation does it
                # without smearing the attack.
                delay = int(0.011 * sample_rate)
                if len(stem) > delay:
                    stem[delay:, 1] += 0.20 * stem[:-delay, 0]
            elif instrument == "bass":
                stem *= 0.92  # keep the low end centered and out of the way
            # A sampled patch has no bow, breath or pick.  Those are what make
            # a fiddle read as a person and a flute as wood rather than a sine,
            # so they are generated from the notes actually played.
            layer = EXPRESSIVE.get(arrangement.style.name, {}).get(instrument)
            if layer:
                stem = apply_expression(
                    stem, notes, kind=layer, sample_rate=sample_rate,
                    seconds_per_beat=spb, total=total_samples, seed=seed,
                )
            mix += stem

    if percussion is not None:
        span = min(len(mix), len(percussion))
        mix[:span] += percussion[:span]

    # Shared bus treatment: a short, natural room, gentle glue compression and
    # a limiter.  The SoundFont supplies the timbre; this supplies the record.
    mix = _convolve_reverb(mix, sample_rate, min(0.30, arrangement.style.reverb))
    envelope = np.abs(mix).max(axis=1)
    attack = 0.018
    release = 0.32
    smoothed = np.copy(envelope)
    for index in range(1, len(smoothed)):
        coefficient = math.exp(
            -1.0 / (sample_rate * (attack if smoothed[index] > smoothed[index - 1] else release))
        )
        smoothed[index] = coefficient * smoothed[index - 1] + (1 - coefficient) * envelope[index]
    gain = np.where(smoothed > 0.22, 0.22 / (smoothed + 1e-6), 1.0)
    gain = np.clip(gain, 0.62, 1.0)
    mix *= gain[:, None]
    mix = _soft_limit(mix)
    fade = min(len(mix), int(0.05 * sample_rate))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        mix[:fade] *= ramp
        mix[-fade:] *= ramp[::-1]
    return mix.astype(np.float32)


def render_arrangement(
    arrangement: Arrangement,
    *,
    sample_rate: int = 44100,
    seed: int = 0,
    renderer: str = "auto",
    soundfont: str | None = None,
) -> np.ndarray:
    """Render an arrangement as stereo audio.

    ``renderer`` is ``"auto"`` (SoundFont when available, numpy otherwise),
    ``"soundfont"`` (raise if unavailable) or ``"synth"`` (the old additive
    engine, useful for tests and for machines without FluidSynth).
    """
    choice = (renderer or "auto").lower()
    if choice == "auto":
        choice = "soundfont" if soundfont_available(soundfont) else "synth"
    if choice == "soundfont":
        return render_arrangement_soundfont(
            arrangement, sample_rate=sample_rate, seed=seed, soundfont=soundfont
        )
    if choice != "synth":
        raise ValueError(f"unknown renderer {renderer!r}; use auto, soundfont or synth")
    return _render_arrangement_synth(arrangement, sample_rate=sample_rate, seed=seed)


# #
# Mixing with the voice
# #

#: Which rendered slot of which band gets a playing layer from
#: mysong.expression, and what kind.  A sample has no bow, breath or pick, so
#: these are the slots where a General MIDI patch alone sounds synthetic.
EXPRESSIVE: dict[str, dict[str, str]] = {
    "metis_fiddle": {"piano": "fiddle", "guitar": "guitar"},
    "cedar_flute": {"piano": "flute"},
    "round_dance": {"guitar": "guitar"},
    "jingle_dress": {"guitar": "guitar"},
}

#: The band every path uses unless it is told otherwise: the hand drum this
#: project's artist has sung with since 1982, not a generic pop kit.
DEFAULT_BAND = "hand_drum"

#: Level targets for the voice-over-band mix.  The voice is levelled to
#: ``VOICE_RMS`` and the band is then placed ``BAND_OFFSET_DB`` under it.  A
#: dense polyphonic band at the same RMS as a single voice still buries that
#: voice, which is exactly what this mix got wrong when it levelled nothing and
#: relied on the recording's own level.
VOICE_RMS = 0.18
#: How far under the voice the band sits, in dB, before ducking.  Public
#: because the app lets the user trim it from the panel.
BAND_OFFSET_DB = -10.0
_MAX_VOICE_BOOST = 16.0   # +24 dB, enough for a quiet phone recording

#: Every generated band the app plays is written at one reference RMS, the same
#: way the DSP vocable is levelled to its own target, so a slider means the
#: same thing from one render and one band to the next.  The panel then places
#: that stem under the take with :func:`band_placement_gain_db`.
BAND_STEM_RMS = 0.1


def _highpass_voice(vocal: np.ndarray, sample_rate: int) -> np.ndarray:
    """The take without its rumble, and the signal every level is read from."""
    from scipy.signal import butter, sosfiltfilt

    vocal = np.asarray(vocal, dtype=np.float64).reshape(-1)
    if vocal.size == 0:
        return vocal.astype(np.float32)
    sos = butter(2, 75.0, btype="high", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, vocal).astype(np.float32)


def _voice_gain(voice: np.ndarray, *, voice_rms: float = VOICE_RMS) -> float:
    """How far a take has to move to reach ``voice_rms`` (boost is capped)."""
    rms = float(np.sqrt(np.mean(np.asarray(voice, dtype=np.float64) ** 2)) + 1e-9)
    return float(min(_MAX_VOICE_BOOST, voice_rms / rms))


def level_band_stem(band: np.ndarray, *, reference_rms: float = BAND_STEM_RMS) -> np.ndarray:
    """Write a rendered band at the one level the app's player expects.

    A renderer writes a full-scale band whichever style it rendered, while the
    take beside it is whatever the person recorded, so the raw render says
    nothing about how loud the band should sit.  Normalising the stem to
    ``reference_rms`` and letting the page place it with
    :func:`band_placement_gain_db` is what keeps the two sliders meaning the
    same thing on every render, and keeps the downloaded band a usable stem
    rather than a file scaled down for one particular take.

    The RMS is measured and set *after* the peak is made safe, not before: a
    soft limiter that drives low levels up would otherwise return a stem about
    2 dB hotter than the placement arithmetic assumes.
    """
    band = np.asarray(band, dtype=np.float32)
    if band.size == 0:
        return band
    if band.ndim == 1:
        band = np.stack([band, band], axis=1)
    peak = float(np.max(np.abs(band)))
    if peak > 0.95:
        band = band * (0.95 / peak)
    rms = float(np.sqrt(np.mean(band.astype(np.float64) ** 2)) + 1e-9)
    if rms > 1e-6:
        band = band * float(reference_rms / rms)
    return band.astype(np.float32)


def band_placement_gain_db(
    vocal: np.ndarray,
    *,
    sample_rate: int,
    band_offset_db: float = BAND_OFFSET_DB,
    reference_rms: float = BAND_STEM_RMS,
) -> float:
    """How far to pull the band stem down so it sits under *this* take.

    The page plays the take itself, untouched, at its own recorded level, so
    the band has to be placed relative to that level rather than to any fixed
    number: a quiet phone recording and a loud studio take both get the same
    balance, which is the one the rendered mix has (``band_offset_db``).
    """
    voice = _highpass_voice(vocal, sample_rate)
    if voice.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(voice.astype(np.float64) ** 2)) + 1e-9)
    target = rms * 10 ** (band_offset_db / 20.0)
    if target <= 0 or reference_rms <= 0:
        return 0.0
    return float(20.0 * math.log10(target / reference_rms))


def mix_with_vocal(
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    sample_rate: int,
    offset_seconds: float = 0.0,
    vocal_gain_db: float = 0.0,
    duck_db: float = 6.0,
    band_offset_db: float = BAND_OFFSET_DB,
    voice_rms: float = VOICE_RMS,
) -> np.ndarray:
    """Place a vocal over an arrangement with the voice on top.

    Both parts are levelled explicitly instead of trusting the absolute level
    of the recording or of the renderer: the voice is normalised to
    ``voice_rms`` and the band is set ``band_offset_db`` under it, then ducked
    further while the voice is singing.  ``vocal_gain_db`` is the user's trim
    on the voice ("track volume"); ``band_offset_db`` moves the whole band.
    """
    if vocal.ndim > 1:
        vocal = vocal.mean(axis=1)
    if accompaniment.ndim == 1:
        accompaniment = np.stack([accompaniment, accompaniment], axis=1)
    start = max(0, int(round(offset_seconds * sample_rate)))
    length = max(len(accompaniment), start + len(vocal))
    out = np.zeros((length, 2), dtype=np.float32)
    if len(vocal) == 0:
        out[: len(accompaniment)] += accompaniment
        return out

    voice = _highpass_voice(vocal, sample_rate)
    gain = 10 ** (vocal_gain_db / 20.0)

    # 1. Level the voice.  A real take can be 20 dB quieter than the band, so
    #    the boost is generous; the cap only stops a silent file from being
    #    amplified into noise.
    voice = voice * (_voice_gain(voice, voice_rms=voice_rms) * gain)
    voice = _soft_limit(voice, ceiling=0.98)

    # 2. Level the band against the levelled voice, whatever the renderer
    #    produced, so every style sits in the same place under the voice.
    band = accompaniment.astype(np.float32)
    band_rms = float(np.sqrt(np.mean(band.astype(np.float64) ** 2)) + 1e-9)
    if band_rms > 1e-6:
        target = voice_rms * gain * 10 ** (band_offset_db / 20.0)
        band = band * float(target / band_rms)

    # 3. Duck the band with a smoothed vocal envelope.
    envelope = np.abs(voice)
    window = max(1, int(0.08 * sample_rate))
    kernel = np.ones(window) / window
    smoothed = np.convolve(envelope, kernel, mode="same")
    duck = 1.0 - (1.0 - 10 ** (-duck_db / 20.0)) * np.clip(smoothed / 0.08, 0.0, 1.0)
    span_end = min(start + len(voice), len(band))
    band[start:span_end] *= duck[: span_end - start, None]

    out[: len(band)] += band
    out[start:start + len(voice), 0] += voice
    out[start:start + len(voice), 1] += voice

    # A little vocal reverb, without drowning the words.
    wet = np.zeros_like(out)
    wet[start:start + len(voice), 0] = voice
    wet[start:start + len(voice), 1] = voice
    wet = _convolve_reverb(wet, sample_rate, 0.13)
    out = 0.87 * out + 0.13 * wet
    return _soft_limit(out)


def mix_stems(stems: dict[str, np.ndarray], sample_rate: int) -> np.ndarray:
    """Sum named stems (each mono or stereo) into a limited stereo mix."""
    length = max((len(stem) for stem in stems.values()), default=0)
    out = np.zeros((length, 2), dtype=np.float32)
    for name, stem in stems.items():
        if len(stem) == 0:
            continue
        if stem.ndim == 1:
            stem = np.stack([stem, stem], axis=1)
        out[: len(stem)] += stem.astype(np.float32)
    return _soft_limit(out)


# #
# Standard MIDI File
# #

def _vlq(value: int) -> bytes:
    if value < 0:
        value = 0
    buffer = [value & 0x7F]
    value >>= 7
    while value:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buffer))


def _track_name(name: str) -> bytes:
    """A MIDI track-name meta event, including its length byte.

    ``b"\xff\x03" + name`` is a common mistake: meta type 0x03 is followed by
    a variable-length *length*, not by the text directly.  Without the length
    byte every parser sees the first character as a huge length and the whole
    track is misread, which is why FluidSynth rendered silence from a file that
    looked plausible in a hex dump.
    """
    data = name.encode("utf-8", "replace")[:127]
    return b"\xff\x03" + _vlq(len(data)) + data


def _track_chunk(events: list[tuple[int, bytes]], *, end_of_track: bool = True) -> bytes:
    events = sorted(events, key=lambda item: item[0])
    payload = bytearray()
    previous = 0
    for tick, data in events:
        payload += _vlq(max(0, tick - previous))
        payload += data
        previous = tick
    if end_of_track:
        payload += _vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(payload)) + bytes(payload)


def write_midi(
    arrangement: Arrangement,
    melody_notes: Sequence | None = None,
    *,
    ppq: int = 480,
) -> bytes:
    """A type-1 MIDI file with melody, piano, bass and drums."""
    spb = arrangement.seconds_per_beat
    beats_to_tick = lambda beat: max(0, int(round(float(beat) * ppq)))

    meta = [
        (0, b"\xff\x51\x03" + int(60_000_000 / max(arrangement.tempo_bpm, 1)).to_bytes(3, "big")),
        (0, b"\xff\x58\x04" + bytes([int(round(arrangement.beats_per_bar)), 2, 24, 8])),
        (0, _track_name("Conductor")),
    ]
    tracks = [_track_chunk(meta)]

    melody: list[tuple[int, bytes]] = [(0, _track_name("Melody"))]
    if melody_notes:
        for note in melody_notes:
            pitch = int(getattr(note, "pitch", getattr(note, "midi", 60)))
            start = float(getattr(note, "onset", getattr(note, "start", 0.0)))
            duration = float(getattr(note, "duration", 0.5))
            velocity = int(getattr(note, "velocity", 80))
            begin = beats_to_tick(start)
            finish = beats_to_tick(start + max(0.05, duration))
            melody.append((begin, bytes([0x90, pitch & 0x7F, velocity & 0x7F])))
            melody.append((finish, bytes([0x80, pitch & 0x7F, 0])))
    tracks.append(_track_chunk(melody))

    programs = {"piano": (0, 0), "bass": (1, 32), "pad": (2, 89), "guitar": (3, 25),
                "drums": (9, 0)}
    # A band can re-point a slot at a different General MIDI instrument: the
    # "piano" slot carries the fiddle in the Métis fiddle band and the flute in
    # the cedar flute band, which is why the slot is not named after one voice.
    overrides = dict(getattr(arrangement.style, "programmes", {}) or {})
    for instrument, (channel, program) in programs.items():
        events: list[tuple[int, bytes]] = [(0, _track_name(instrument.title()))]
        program = int(overrides.get(instrument, program))
        if channel != 9:
            events.append((0, bytes([0xC0 | channel, program & 0x7F])))
        for note in arrangement.notes_for(instrument):
            begin = beats_to_tick(note.start)
            finish = beats_to_tick(note.end)
            events.append((begin, bytes([0x90 | channel, note.pitch & 0x7F, note.velocity & 0x7F])))
            events.append((finish, bytes([0x80 | channel, note.pitch & 0x7F, 0])))
        tracks.append(_track_chunk(events))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq)
    return header + b"".join(tracks)

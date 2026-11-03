"""Music-theory primitives used by the MySong harmonizer.

Everything in this module is deterministic, dependency-free (apart from the
standard library) and written in pitch classes (0 = C .. 11 = B).  Octaves only
appear when we need to place notes in a register, which is the renderer's job.

The vocabulary separates two things:

``core`` qualities
    The five triads MySong (Simon, Morris & Basu, CHI 2008) models:
    major, minor, diminished, augmented and suspended.  A chord's *core*
    is what the statistical model predicts, because that is the level at
    which a lead-sheet corpus has enough data to estimate transitions.

``quality``
    The full chord symbol, including sevenths, sixths and ninths.  The
    decoder predicts a core triad and :mod:`mysong.accompaniment` colours it
    with an extension that fits the melody and the style.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# #
# Pitch classes
# #

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

#: Krumhansl & Kessler (1982) key profiles.  Used for the key prior, never as
#: the chord model itself.
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def pc_name(pc: int, *, prefer_flats: bool = False) -> str:
    """Return the name of a pitch class (mod 12)."""
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[int(pc) % 12]


def spell_pc(pc: int, key: "Key | None" = None) -> str:
    """Spell a pitch class the way musicians write it in ``key``.

    Flat keys get flats, sharp keys get sharps.  C and A minor are the
    ambiguous case: pop/jazz lead sheets there use Bb/Eb/Ab/Db far more often
    than A#/D#/G#/C#, but F# rather than Gb, so 6 stays sharp.
    """
    if key is None:
        return pc_name(pc)
    fifths = key_fifths(key.tonic, key.mode)
    if fifths < 0:
        return pc_name(pc, prefer_flats=True)
    if fifths > 0:
        return pc_name(pc, prefer_flats=False)
    return pc_name(pc, prefer_flats=int(pc) % 12 in (1, 3, 8, 10))


def pc_from_name(name: str) -> int:
    """Parse a note name such as ``C#``, ``Bb``, ``F`` or ``G##``."""
    text = name.strip()
    if not text:
        raise ValueError("empty note name")
    letter = text[0].upper()
    if letter not in "ABCDEFG":
        raise ValueError(f"not a note name: {name!r}")
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    for accidental in text[1:]:
        if accidental in ("#", "\u266f"):
            base += 1
        elif accidental in ("b", "\u266d", "-"):
            base -= 1
        elif accidental == "\u266e":
            pass
        else:
            raise ValueError(f"bad accidental in {name!r}")
    return base % 12


def fifths_distance(a: int, b: int) -> int:
    """Signed circle-of-fifths distance from pitch class ``a`` to ``b``.

    ``+1`` means a perfect fifth up, ``-1`` a perfect fifth down.  The result
    is in ``[-6, 5]`` and is the feature the transition back-off uses: root
    motion by fifths is the strongest regularity in tonal harmony.
    """
    return ((int(b) - int(a)) * 7 + 6) % 12 - 6


def key_fifths(tonic: int, mode: str) -> int:
    """Signed key-signature count in ``[-7, 7]`` for display/spelling."""
    tonic = int(tonic) % 12
    relative = (tonic + 3) % 12 if mode == "minor" else tonic
    # pitch class -> fifths position on the line of fifths
    table = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6,
             5: -1, 10: -2, 3: -3, 8: -4, 1: -5}
    return table[relative]


def metric_weight(offset_quarters: float, beats_per_bar: float = 4.0) -> float:
    """Salience of a position inside a bar.

    Downbeats are the most informative place to hear a chord tone, off-beats
    the least.  The values are multipliers on a melody note's duration when it
    is accumulated into an observation vector; they are learned indirectly,
    since the same weighting is applied to the training corpus.
    """
    if beats_per_bar <= 0:
        beats_per_bar = 4.0
    position = float(offset_quarters) % beats_per_bar
    if abs(position) < 1e-6 or abs(position - beats_per_bar) < 1e-6:
        return 1.6
    # Half-bar, then any other beat, then the off-beats.
    half = beats_per_bar / 2.0
    if abs(position - half) < 1e-6:
        return 1.25
    if abs(position - round(position)) < 1e-6:
        return 1.0
    return 0.55


# #
# Chord qualities
# #

#: Full quality name -> semitone intervals above the root.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "mMaj7": (0, 3, 7, 11),
    "add9": (0, 2, 4, 7),
    "madd9": (0, 2, 3, 7),
    "9": (0, 2, 4, 7, 10),
    "maj9": (0, 2, 4, 7, 11),
    "min9": (0, 2, 3, 7, 10),
    "11": (0, 2, 4, 5, 7, 10),
    "13": (0, 2, 4, 7, 9, 10),
    "7sus4": (0, 5, 7, 10),
    "9sus4": (0, 2, 5, 7, 10),
    "m11": (0, 2, 3, 5, 7, 10),
}

#: The five triads the MySong model predicts.  Order is the state order used
#: by every saved model, so do not reorder these.
CORE_QUALITIES: tuple[str, ...] = ("maj", "min", "dim", "aug", "sus")
CORE_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus": (0, 5, 7),
}

#: Full quality -> core triad.
CORE_OF_QUALITY: dict[str, str] = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "aug",
    "sus2": "sus", "sus4": "sus", "7sus4": "sus", "9sus4": "sus",
    "5": "maj",
    "7": "maj", "6": "maj", "maj7": "maj", "add9": "maj", "9": "maj",
    "maj9": "maj", "11": "maj", "13": "maj",
    "m6": "min", "min7": "min", "mMaj7": "min", "madd9": "min",
    "min9": "min", "m11": "min",
    "m7b5": "dim", "dim7": "dim",
}

#: music21 chord-kind strings (``ChordSymbol.chordKind``) -> full quality.
MUSIC21_KIND_TO_QUALITY: dict[str, str] = {
    "major": "maj",
    "minor": "min",
    "diminished": "dim",
    "augmented": "aug",
    "suspended-second": "sus2",
    "suspended-fourth": "sus4",
    "power": "5",
    "major-sixth": "6",
    "minor-sixth": "m6",
    "major-seventh": "maj7",
    "minor-seventh": "min7",
    "dominant-seventh": "7",
    "half-diminished-seventh": "m7b5",
    "diminished-seventh": "dim7",
    "minor-major": "mMaj7",
    "major-ninth": "maj9",
    "minor-ninth": "min9",
    "dominant-ninth": "9",
    "major-11th": "11",
    "dominant-11th": "11",
    "minor-11th": "m11",
    "dominant-13th": "13",
    "augmented-seventh": "aug",
    "augmented-ninth": "aug",
}

#: Symbols used when printing.  The empty string is the major triad.
QUALITY_SYMBOL: dict[str, str] = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4", "5": "5",
    "6": "6", "m6": "m6", "maj7": "maj7", "min7": "m7", "7": "7",
    "m7b5": "m7b5", "dim7": "dim7", "mMaj7": "mMaj7",
    "add9": "add9", "madd9": "madd9", "9": "9", "maj9": "maj9",
    "min9": "m9", "11": "11", "13": "13", "7sus4": "7sus4",
    "9sus4": "9sus4", "m11": "m11",
}

#: Common textual spellings accepted by :func:`parse_quality`.
_QUALITY_ALIASES: dict[str, str] = {
    "": "maj", "M": "maj", "maj": "maj", "major": "maj",
    "m": "min", "min": "min", "major-minor": "mMaj7",
    "-": "min", "mi": "min",
    "dim": "dim", "o": "dim", "°": "dim",
    "aug": "aug", "+": "aug",
    "sus": "sus4", "sus4": "sus4", "sus2": "sus2",
    "5": "5", "6": "6", "M6": "6", "m6": "m6", "min6": "m6",
    "7": "7", "dom7": "7", "M7": "maj7", "maj7": "maj7",
    "Δ7": "maj7", "Δ": "maj7",
    "m7": "min7", "min7": "min7", "-7": "min7",
    "ø": "m7b5", "ø7": "m7b5", "m7b5": "m7b5", "half-dim": "m7b5",
    "o7": "dim7", "dim7": "dim7", "°7": "dim7",
    "mM7": "mMaj7", "mMaj7": "mMaj7", "minMaj7": "mMaj7",
    "add9": "add9", "madd9": "madd9",
    "9": "9", "maj9": "maj9", "M9": "maj9", "m9": "min9", "min9": "min9",
    "11": "11", "m11": "m11", "13": "13", "7sus4": "7sus4",
    "9sus4": "9sus4",
}


def parse_quality(text: str) -> str:
    """Normalise a quality spelling to a key of :data:`QUALITY_INTERVALS`."""
    token = (text or "").strip()
    if token in _QUALITY_ALIASES:
        return _QUALITY_ALIASES[token]
    lowered = token.lower()
    if lowered in _QUALITY_ALIASES:
        return _QUALITY_ALIASES[lowered]
    if token in QUALITY_INTERVALS:
        return token
    raise ValueError(f"unknown chord quality: {text!r}")


def core_of(quality: str) -> str:
    """Return the MySong triad of a full quality."""
    if quality in CORE_INTERVALS:
        return quality
    try:
        return CORE_OF_QUALITY[quality]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"no core triad for quality {quality!r}") from exc


def quality_from_semitones(semitones: Iterable[int]) -> str:
    """Best full quality for a set of intervals (used by the chord parser)."""
    wanted = frozenset(int(s) % 12 for s in semitones)
    if not wanted:
        return "maj"
    # Prefer smaller chord symbols when several interpretations fit.
    for name in sorted(QUALITY_INTERVALS, key=lambda n: len(QUALITY_INTERVALS[n])):
        if frozenset(QUALITY_INTERVALS[name]) == wanted:
            return name
    return "maj"


# #
# Chords
# #

_CHORD_RE = re.compile(
    r"^\s*([A-Ga-g])([#b\u266f\u266d-]?)\s*(.*?)\s*(?:/\s*([A-Ga-g])([#b\u266f\u266d-]?)\s*)?$"
)


@dataclass(frozen=True, order=True)
class Chord:
    """A chord symbol: root pitch class, quality and optional bass."""

    root: int
    quality: str = "maj"
    bass: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", int(self.root) % 12)
        if self.quality not in QUALITY_INTERVALS:
            object.__setattr__(self, "quality", parse_quality(self.quality))
        if self.bass is not None:
            object.__setattr__(self, "bass", int(self.bass) % 12)

    # -- structure ------------------------------------------------------- #
    @property
    def core(self) -> "Chord":
        """The triad this chord reduces to (bass is preserved)."""
        return Chord(self.root, core_of(self.quality), self.bass)

    @property
    def core_quality(self) -> str:
        return core_of(self.quality)

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITY_INTERVALS[self.quality]

    def pitch_classes(self, include_bass: bool = True) -> tuple[int, ...]:
        pcs = [(self.root + i) % 12 for i in self.intervals]
        if include_bass and self.bass is not None and self.bass not in pcs:
            pcs.append(self.bass)
        return tuple(dict.fromkeys(pcs))

    def contains(self, pc: int) -> bool:
        return int(pc) % 12 in self.pitch_classes()

    def is_diatonic(self, key: "Key") -> bool:
        return all(pc in key.scale_pcs() for pc in self.pitch_classes(False))

    # -- names ----------------------------------------------------------- #
    def figure(self, key: "Key | None" = None) -> str:
        text = spell_pc(self.root, key) + QUALITY_SYMBOL[self.quality]
        if self.bass is not None and self.bass != self.root:
            text += "/" + spell_pc(self.bass, key)
        return text

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.figure()

    # -- transformations -------------------------------------------------- #
    def transpose(self, semitones: int) -> "Chord":
        return Chord(
            (self.root + semitones) % 12,
            self.quality,
            None if self.bass is None else (self.bass + semitones) % 12,
        )

    def with_bass(self, bass: int | None) -> "Chord":
        return Chord(self.root, self.quality, bass)

    def simplified(self) -> "Chord":
        """Drop the bass note and extensions, keeping the core triad."""
        return Chord(self.root, self.core_quality)

    @classmethod
    def parse(cls, figure: str) -> "Chord":
        match = _CHORD_RE.match(figure or "")
        if not match:
            raise ValueError(f"cannot parse chord: {figure!r}")
        letter, accidental, quality_text, bass_letter, bass_acc = match.groups()
        root = pc_from_name(letter + (accidental or ""))
        quality = parse_quality(quality_text.strip())
        bass = None
        if bass_letter:
            bass = pc_from_name(bass_letter + (bass_acc or ""))
        return cls(root, quality, bass)


def core_chords() -> list[Chord]:
    """All 60 core triads in model order (root-major, root-minor, ...)."""
    return [Chord(root, quality) for root in range(12) for quality in CORE_QUALITIES]


def core_index(chord: Chord) -> int:
    root = chord.root % 12
    quality = chord.core_quality
    return root * len(CORE_QUALITIES) + CORE_QUALITIES.index(quality)


def chord_from_core_index(index: int) -> Chord:
    root, quality = divmod(int(index), len(CORE_QUALITIES))
    return Chord(root, CORE_QUALITIES[quality])


# #
# Keys
# #

_MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
_NATURAL_MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)
_HARMONIC_MINOR_STEPS = (0, 2, 3, 5, 7, 8, 11)


@dataclass(frozen=True, order=True)
class Key:
    """A tonic pitch class plus ``"major"`` or ``"minor"``."""

    tonic: int
    mode: str = "major"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tonic", int(self.tonic) % 12)
        mode = (self.mode or "major").lower()
        if mode not in ("major", "minor"):
            raise ValueError(f"unknown mode: {self.mode!r}")
        object.__setattr__(self, "mode", mode)

    # -- scales ---------------------------------------------------------- #
    def scale_pcs(self, harmonic_minor: bool = False) -> frozenset[int]:
        if self.mode == "major":
            steps = _MAJOR_STEPS
        else:
            steps = _HARMONIC_MINOR_STEPS if harmonic_minor else _NATURAL_MINOR_STEPS
        return frozenset((self.tonic + s) % 12 for s in steps)

    def degree_pc(self, degree: int) -> int:
        """Pitch class of scale degree ``1..7``."""
        steps = _MAJOR_STEPS if self.mode == "major" else _NATURAL_MINOR_STEPS
        return (self.tonic + steps[(int(degree) - 1) % 7]) % 12

    def relative(self) -> "Key":
        if self.mode == "major":
            return Key((self.tonic + 9) % 12, "minor")
        return Key((self.tonic + 3) % 12, "major")

    def parallel(self) -> "Key":
        return Key(self.tonic, "minor" if self.mode == "major" else "major")

    def transpose(self, semitones: int) -> "Key":
        return Key((self.tonic + semitones) % 12, self.mode)

    # -- names ----------------------------------------------------------- #
    def name(self) -> str:
        prefer_flats = key_fifths(self.tonic, self.mode) < 0
        base = pc_name(self.tonic, prefer_flats=prefer_flats)
        return base + ("m" if self.mode == "minor" else "")

    def __str__(self) -> str:  # pragma: no cover
        return self.name()

    # -- harmony --------------------------------------------------------- #
    def diatonic_triads(self) -> list[Chord]:
        """The seven diatonic triads, in degree order."""
        qualities = ("maj", "min", "min", "maj", "maj", "min", "dim")
        if self.mode == "minor":
            qualities = ("min", "dim", "maj", "min", "min", "maj", "maj")
        return [
            Chord(self.degree_pc(i + 1), qualities[i])
            for i in range(7)
        ]

    def diatonic_sevenths(self) -> list[Chord]:
        qualities = ("maj7", "min7", "min7", "maj7", "7", "min7", "m7b5")
        if self.mode == "minor":
            qualities = ("min7", "m7b5", "maj7", "min7", "min7", "maj7", "7")
        return [Chord(self.degree_pc(i + 1), qualities[i]) for i in range(7)]

    def roman(self, chord: Chord) -> str:
        """Roman-numeral analysis, e.g. ``V7/vi`` is reported as ``V7`` on F#."""
        triads = self.diatonic_triads()
        core = chord.core
        for i, triad in enumerate(triads):
            if triad.root == core.root and triad.core_quality == core.core_quality:
                numeral = ("I", "II", "III", "IV", "V", "VI", "VII")[i]
                if core.core_quality in ("min", "dim"):
                    numeral = numeral.lower()
                if core.core_quality == "dim":
                    numeral += "°"
                elif core.core_quality == "aug":
                    numeral += "+"
                if chord.quality not in ("maj", "min") and QUALITY_SYMBOL.get(chord.quality):
                    suffix = QUALITY_SYMBOL[chord.quality]
                    if suffix.startswith("m") and core.core_quality == "min":
                        suffix = suffix[1:]
                    numeral += suffix
                if chord.bass is not None and chord.bass != chord.root:
                    numeral += "/" + pc_name(chord.bass)
                return numeral
        # Chromatic chord: name it relative to the tonic.
        return f"{pc_name(chord.root)}" + QUALITY_SYMBOL[chord.quality]

    def distance_to(self, other: "Key") -> int:
        """Circle-of-fifths distance between two keys (mode included)."""
        if self.mode != other.mode:
            return 6  # parallel/relative pairs are a special case
        return abs(fifths_distance(self.tonic, other.tonic))


def parse_key(text: str) -> Key:
    """Parse ``"C"``, ``"Am"``, ``"F# minor"``, ``"Bb major"`` ..."""
    raw = (text or "").strip()
    match = re.match(r"^([A-Ga-g][#b\u266f\u266d-]?)\s*(.*)$", raw)
    if not match:
        raise ValueError(f"cannot parse key: {text!r}")
    tonic = pc_from_name(match.group(1))
    rest = match.group(2).strip().lower()
    if rest.startswith("min") or rest in ("m", "-", "aeolian"):
        mode = "minor"
    elif rest.startswith("maj") or rest in ("", "ionian"):
        mode = "major"
    else:
        raise ValueError(f"cannot parse key mode: {text!r}")
    return Key(tonic, mode)


def key_profile(key: Key) -> tuple[float, ...]:
    """Krumhansl-Kessler profile rotated onto ``key``'s tonic."""
    source = KK_MAJOR if key.mode == "major" else KK_MINOR
    return tuple(source[(pc - key.tonic) % 12] for pc in range(12))


def profile_correlation(chroma: Sequence[float], profile: Sequence[float]) -> float:
    """Pearson correlation, returning 0 for a flat (silent) chroma."""
    import math

    x = [float(v) for v in chroma]
    y = [float(v) for v in profile]
    n = len(x)
    if n == 0 or len(y) != n:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def estimate_key(
    chroma: Sequence[float],
    modes: Sequence[str] = ("major", "minor"),
) -> Key:
    """Krumhansl-Schmuckler key estimate from a 12-bin chroma vector."""
    best = Key(0, "major")
    best_score = -2.0
    for mode in modes:
        for tonic in range(12):
            key = Key(tonic, mode)
            score = profile_correlation(chroma, key_profile(key))
            if score > best_score:
                best, best_score = key, score
    return best


def key_scores(chroma: Sequence[float], modes: Sequence[str] = ("major", "minor")) -> list[tuple[Key, float]]:
    """All 24 key correlations, strongest first."""
    scored = []
    for mode in modes:
        for tonic in range(12):
            key = Key(tonic, mode)
            scored.append((key, profile_correlation(chroma, key_profile(key))))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def key_from_signature(sharps: int, mode: str = "major") -> Key:
    """Convert a MusicXML key signature (fifths) to a :class:`Key`."""
    major_by_fifths = {
        0: 0, 1: 7, 2: 2, 3: 9, 4: 4, 5: 11, 6: 6, 7: 1,
        -1: 5, -2: 10, -3: 3, -4: 8, -5: 1, -6: 6, -7: 11,
    }
    # Note: -5 is Db major, -6 Gb, -7 Cb; the table above is only used for
    # display, and the ambiguous enharmonic spellings share a pitch class.
    tonic = major_by_fifths.get(int(sharps), 0)
    if mode == "minor":
        tonic = (tonic + 9) % 12
    return Key(tonic, mode)


# #
# Voicing helpers (used by the renderer and by tests)
# #

def chord_midi_notes(chord: Chord, octave: int = 4) -> list[int]:
    """A close-position voicing rooted at ``octave`` (C4 = 60)."""
    base = 12 * (octave + 1) + chord.root
    notes = [base + interval for interval in chord.intervals]
    if chord.bass is not None and chord.bass != chord.root:
        bass_note = 12 * (octave + 1) + chord.bass
        while bass_note >= notes[0]:
            bass_note -= 12
        notes.insert(0, bass_note)
    return sorted(notes)


def voice_leading_cost(previous: Sequence[int], current: Sequence[int]) -> float:
    """Cost of moving from one voicing to another.

    Lower is smoother: penalise total voice motion, large leaps and wide
    spacing.  Used by the renderer's dynamic programme over inversions.
    """
    if not previous or not current:
        return 0.0
    prev = sorted(previous)
    cur = sorted(current)
    # Compare the nearest notes of each voicing, allowing a different note
    # count (a seventh chord has one more voice than a triad).
    count = min(len(prev), len(cur))
    movement = sum(abs(a - b) for a, b in zip(prev[:count], cur[:count]))
    movement += 1.5 * abs(len(prev) - len(cur))
    spacing = sum(max(0, b - a - 12) for a, b in zip(cur, cur[1:]))
    low_penalty = max(0, 48 - cur[0]) * 0.6 + max(0, cur[-1] - 84) * 0.6
    return float(movement + 2.0 * spacing + low_penalty)


def diatonic_function(chord: Chord, key: Key) -> str:
    """Coarse harmonic function: tonic, subdominant or dominant."""
    triads = key.diatonic_triads()
    core = chord.core
    for i, triad in enumerate(triads):
        if triad.root == core.root and triad.core_quality == core.core_quality:
            if i in (0, 5):
                return "tonic"
            if i in (1, 3):
                return "subdominant"
            if i in (4, 6):
                return "dominant"
            return "mediant"
    # Chromatic chords: use root motion relative to the tonic.
    if fifths_distance(key.tonic, core.root) in (1, -1):
        return "dominant"
    return "chromatic"

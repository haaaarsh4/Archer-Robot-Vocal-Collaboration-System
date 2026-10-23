"""Read lead sheets (MusicXML) into the small symbolic form MySong trains on.

The training corpus is OpenEWLD (a public-domain subset of Wikifonia, the same
kind of source MySong used).  Only three things are needed from each score:

* a monophonic melody, with note onsets and durations in quarter notes;
* the chord symbols, as spans of time;
* the key and the meter.

``music21`` does the MusicXML decoding.  Everything after that is plain
dataclasses so the rest of the package does not depend on music21 -- a trained
model can be used on a machine that only has numpy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mysong.theory import (
    Chord,
    Key,
    MUSIC21_KIND_TO_QUALITY,
    key_from_signature,
    key_profile,
    metric_weight,
    profile_correlation,
)

DEFAULT_CORPUS = Path("data/mysong/openewld")
DEFAULT_CACHE = Path("data/mysong/corpus.jsonl")


@dataclass
class NoteEvent:
    onset: float          # quarter notes from the start of the score
    duration: float       # quarter notes
    pitch: int            # MIDI note number
    velocity: int = 80

    @property
    def end(self) -> float:
        return self.onset + self.duration

    @property
    def pc(self) -> int:
        return self.pitch % 12


@dataclass
class ChordSpan:
    onset: float
    duration: float
    chord: Chord

    @property
    def end(self) -> float:
        return self.onset + self.duration


@dataclass
class Song:
    song_id: str
    title: str
    composer: str
    key: Key
    time_signature: str
    beats_per_bar: float
    notes: list[NoteEvent]
    chords: list[ChordSpan]
    source: str = ""
    errors: list[str] = field(default_factory=list)

    # -- convenience ----------------------------------------------------- #
    @property
    def duration(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end for n in self.notes)

    @property
    def chord_duration(self) -> float:
        return sum(c.duration for c in self.chords)

    def melody_span(self) -> tuple[float, float]:
        if not self.notes:
            return (0.0, 0.0)
        return (min(n.onset for n in self.notes), max(n.end for n in self.notes))

    def chord_at(self, offset: float) -> Chord | None:
        for span in self.chords:
            if span.onset <= offset < span.end:
                return span.chord
        return None

    def transposed(self, semitones: int) -> "Song":
        return Song(
            song_id=self.song_id,
            title=self.title,
            composer=self.composer,
            key=self.key.transpose(semitones),
            time_signature=self.time_signature,
            beats_per_bar=self.beats_per_bar,
            notes=[NoteEvent(n.onset, n.duration, n.pitch + semitones, n.velocity) for n in self.notes],
            chords=[ChordSpan(c.onset, c.duration, c.chord.transpose(semitones)) for c in self.chords],
            source=self.source,
            errors=list(self.errors),
        )

    def in_tonic_relative(self) -> "Song":
        """Transpose so the tonic is C, keeping the mode separate."""
        return self.transposed((-self.key.tonic) % 12)

    def to_dict(self) -> dict:
        return {
            "id": self.song_id,
            "title": self.title,
            "composer": self.composer,
            "key": [self.key.tonic, self.key.mode],
            "time_signature": self.time_signature,
            "beats_per_bar": self.beats_per_bar,
            "source": self.source,
            "notes": [[round(n.onset, 5), round(n.duration, 5), n.pitch, n.velocity] for n in self.notes],
            "chords": [
                [round(c.onset, 5), round(c.duration, 5), c.chord.root, c.chord.quality, c.chord.bass]
                for c in self.chords
            ],
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Song":
        tonic, mode = payload["key"]
        return cls(
            song_id=payload["id"],
            title=payload.get("title", ""),
            composer=payload.get("composer", ""),
            key=Key(int(tonic), mode),
            time_signature=payload.get("time_signature", "4/4"),
            beats_per_bar=float(payload.get("beats_per_bar", 4.0)),
            notes=[NoteEvent(*n) for n in payload.get("notes", [])],
            chords=[ChordSpan(o, d, Chord(r, q, b)) for o, d, r, q, b in payload.get("chords", [])],
            source=payload.get("source", ""),
            errors=list(payload.get("errors", [])),
        )


# --------------------------------------------------------------------------- #
# MusicXML -> Song
# --------------------------------------------------------------------------- #

def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:80] or "song"


def _time_signature(flat) -> tuple[str, float]:
    from music21 import meter

    signature = flat.getElementsByClass(meter.TimeSignature).first()
    if signature is None:
        return "4/4", 4.0
    numerator = int(signature.numerator)
    denominator = int(signature.denominator)
    beats_per_bar = numerator * 4.0 / denominator  # in quarter notes
    return f"{numerator}/{denominator}", beats_per_bar


def _melody_part(score):
    """The part carrying the tune.

    OpenEWLD scores are monophonic, but a few contain an empty second staff.
    Picking the part with the most notes is robust to both.
    """
    parts = list(score.parts)
    if not parts:
        return score
    return max(parts, key=lambda part: len(part.flatten().notes))


def _merge_tied_notes(flat) -> list[NoteEvent]:
    """Flatten a part into monophonic note events, joining tied notes."""
    from music21 import note as m21note

    raw = [n for n in flat.notes if isinstance(n, m21note.Note)]
    raw.sort(key=lambda n: float(n.offset))
    events: list[NoteEvent] = []
    for item in raw:
        duration = float(item.duration.quarterLength)
        if duration <= 0:
            continue
        midi = int(round(item.pitch.midi))
        onset = float(item.offset)
        velocity = 80
        volume = getattr(item, "volume", None)
        if volume is not None and getattr(volume, "velocity", None) is not None:
            velocity = int(volume.velocity)
        tie = getattr(item, "tie", None)
        tie_type = getattr(tie, "type", None) if tie is not None else None
        if tie_type in ("stop", "continue") and events:
            previous = events[-1]
            if previous.pitch == midi and abs(previous.end - onset) < 1e-3:
                previous.duration += duration
                continue
        # A note that starts a tie is merged when its continuation arrives.
        events.append(NoteEvent(onset, duration, midi, velocity))
    return events


def _chord_spans(flat, end: float) -> list[ChordSpan]:
    from music21 import harmony

    symbols = [h for h in flat.getElementsByClass(harmony.ChordSymbol)]
    symbols.sort(key=lambda h: float(h.offset))
    spans: list[ChordSpan] = []
    cleaned: list[tuple[float, Chord | None]] = []
    for symbol in symbols:
        try:
            root = symbol.root()
            if root is None:
                continue
            kind = str(symbol.chordKind or "")
            quality = MUSIC21_KIND_TO_QUALITY.get(kind)
            if quality is None:
                # Unknown kind: keep the root and treat it as a major triad
                # rather than dropping a harmony the melody still needs.
                quality = "maj"
            bass = symbol.bass()
            bass_pc = int(bass.pitchClass) if bass is not None else None
            chord = Chord(int(root.pitchClass), quality, bass_pc)
        except Exception:
            continue
        offset = float(symbol.offset)
        cleaned.append((offset, chord))

    if not cleaned:
        return spans

    # Collapse duplicate symbols at the same offset (last wins).
    merged: list[tuple[float, Chord]] = []
    for offset, chord in cleaned:
        if chord is None:
            continue
        if merged and abs(merged[-1][0] - offset) < 1e-6:
            merged[-1] = (offset, chord)
        else:
            merged.append((offset, chord))

    first_offset = merged[0][0]
    if first_offset > 0:
        # A pickup: the first chord governs the notes before it.
        merged.insert(0, (0.0, merged[0][1]))

    for index, (offset, chord) in enumerate(merged):
        stop = merged[index + 1][0] if index + 1 < len(merged) else max(end, offset + 1.0)
        duration = stop - offset
        if duration <= 1e-6:
            continue
        # Do not emit trailing silence after the melody has ended.
        if offset >= end:
            break
        spans.append(ChordSpan(offset, min(duration, end - offset), chord))
    return spans


def _chroma_for_key(song_notes: Sequence[NoteEvent], spans: Sequence[ChordSpan],
                    beats_per_bar: float) -> np.ndarray:
    """A key-finding chroma: chord tones dominate, the tune refines it."""
    chroma = np.zeros(12, dtype=np.float64)
    for span in spans:
        weight = span.duration
        for pc in span.chord.pitch_classes():
            chroma[pc] += weight
        # Root emphasis helps the profile lock onto the tonic.
        chroma[span.chord.root] += 0.35 * weight
    if spans:
        chroma[spans[-1].chord.root] += 3.0
    for item in song_notes:
        chroma[item.pc] += 0.45 * item.duration * metric_weight(item.onset, beats_per_bar)
    return chroma


def estimate_key_from_events(
    notes: Sequence[NoteEvent],
    spans: Sequence[ChordSpan],
    beats_per_bar: float = 4.0,
) -> Key:
    """Choose the tonic and mode that best explain a whole lead sheet.

    Krumhansl-Schmuckler on a chord-tone chroma gets the key signature right
    most of the time, but relative major/minor pairs share a signature and the
    two are *not* interchangeable for a transposition-invariant chord model.
    The extra terms below break that tie: a final tonic chord, a first tonic
    chord, and how many chord tones lie inside each candidate scale.  On
    OpenEWLD the final chord is the tonic in 450 of 502 songs.
    """
    chroma = _chroma_for_key(notes, spans, beats_per_bar)
    melody = np.zeros(12, dtype=np.float64)
    for item in notes:
        melody[item.pc] += item.duration

    best = Key(0, "major")
    best_score = -1e18
    for mode in ("major", "minor"):
        for tonic in range(12):
            key = Key(tonic, mode)
            scale = key.scale_pcs()
            score = 0.0
            for span in spans:
                pcs = span.chord.pitch_classes(False)
                inside = sum(1 for pc in pcs if pc in scale)
                score += span.duration * (0.7 * inside - 0.5 * (len(pcs) - inside))
                score += (0.35 if span.chord.root in scale else -0.2) * span.duration
            inside = sum(melody[pc] for pc in scale)
            score += 0.06 * (inside - 0.6 * (float(melody.sum()) - inside))
            score += 2.0 * profile_correlation(chroma, key_profile(key))
            if spans:
                last = spans[-1].chord
                first = spans[0].chord
                if last.root == tonic:
                    score += 3.0
                    if last.core_quality == ("maj" if mode == "major" else "min"):
                        score += 1.2
                if first.root == tonic:
                    score += 0.8
            if score > best_score:
                best, best_score = key, score
    return best


def _quality_filter(spans: Sequence[ChordSpan]) -> list[ChordSpan]:
    """Keep spans whose symbol we can render; drop nonsense durations."""
    out = []
    for span in spans:
        if span.duration <= 1e-6:
            continue
        out.append(span)
    return out


def parse_musicxml(path: str | Path, *, song_id: str | None = None) -> Song:
    """Parse one MusicXML/MXL lead sheet.

    Raises ``ValueError`` if the file has no usable melody or harmony.  A score
    with a bad key signature is still returned, with the key inferred from the
    harmony instead; that is common in Wikifonia and is why we do not trust the
    signature alone.
    """
    import warnings

    from music21 import converter

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = converter.parse(str(path))
    part = _melody_part(score).flatten()
    end = float(score.highestTime)
    notes = _merge_tied_notes(part)
    if len(notes) < 6:
        raise ValueError(f"{path.name}: only {len(notes)} melody notes")
    if not notes:
        raise ValueError(f"{path.name}: no melody")

    time_signature, beats_per_bar = _time_signature(part)

    spans = _quality_filter(_chord_spans(part, max(end, notes[-1].end)))
    if not spans:
        # A few scores put the harmony on a separate (empty-melody) staff.
        spans = _quality_filter(_chord_spans(score.flatten(), max(end, notes[-1].end)))
    if not spans:
        raise ValueError(f"{path.name}: no chord symbols")

    melody_start = min(n.onset for n in notes)
    melody_end = max(n.end for n in notes)
    covered = sum(
        max(0.0, min(span.end, melody_end) - max(span.onset, melody_start))
        for span in spans
    )
    if melody_end - melody_start > 0 and covered / (melody_end - melody_start) < 0.75:
        raise ValueError(f"{path.name}: chords cover only {covered / (melody_end - melody_start):.0%} of the melody")

    key = estimate_key_from_events(notes, spans, beats_per_bar)

    errors: list[str] = []
    from music21 import key as m21key

    notated = part.getElementsByClass(m21key.KeySignature).first()
    if notated is None:
        notated = score.flatten().getElementsByClass(m21key.KeySignature).first()
    if notated is not None:
        signature_key = key_from_signature(int(notated.sharps), "major")
        if signature_key.tonic != key.tonic and signature_key.relative().tonic != key.tonic:
            errors.append(
                f"key signature implies {signature_key.name()}, harmony implies {key.name()}"
            )

    title = path.parent.name.replace("_", " ")
    composer = path.parent.parent.name.replace("_", " ")
    return Song(
        song_id=song_id or _slug(f"{composer}-{title}"),
        title=title,
        composer=composer,
        key=key,
        time_signature=time_signature,
        beats_per_bar=beats_per_bar,
        notes=notes,
        chords=spans,
        source=str(path),
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Corpus handling
# --------------------------------------------------------------------------- #

def find_scores(root: str | Path = DEFAULT_CORPUS) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob("dataset/*/*/*.mxl")) + sorted(root.glob("dataset/*/*/*.musicxml"))


def build_corpus(
    root: str | Path = DEFAULT_CORPUS,
    cache: str | Path = DEFAULT_CACHE,
    *,
    limit: int = 0,
    verbose: bool = True,
) -> tuple[list[Song], list[tuple[str, str]]]:
    """Parse every score under ``root`` and write a JSONL cache.

    Returns ``(songs, failures)``.  Parsing is idempotent; a cache can be
    reused by :func:`load_corpus`.
    """
    paths = find_scores(root)
    if limit:
        paths = paths[:limit]
    songs: list[Song] = []
    failures: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for index, path in enumerate(paths, 1):
        try:
            song = parse_musicxml(path)
        except Exception as exc:  # noqa: BLE001 - a corpus build reports, never crashes
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        if song.song_id in seen_ids:
            song.song_id = f"{song.song_id}-{len(seen_ids)}"
        seen_ids.add(song.song_id)
        songs.append(song)
        if verbose and index % 100 == 0:
            print(f"  parsed {index}/{len(paths)}")
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as handle:
        for song in songs:
            handle.write(json.dumps(song.to_dict(), ensure_ascii=False) + "\n")
    if verbose:
        print(f"wrote {len(songs)} songs to {cache} ({len(failures)} failed)")
    return songs, failures


def load_corpus(cache: str | Path = DEFAULT_CACHE) -> list[Song]:
    path = Path(cache)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m mysong.cli corpus` first"
        )
    songs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                songs.append(Song.from_dict(json.loads(line)))
    return songs


def corpus_stats(songs: Sequence[Song]) -> dict:
    """Summary used by the CLI and by the docs."""
    if not songs:
        return {"songs": 0}
    chord_counts: dict[str, int] = {}
    qualities: dict[str, int] = {}
    modes: dict[str, int] = {}
    meters: dict[str, int] = {}
    total_chords = 0
    total_notes = 0
    total_quarters = 0.0
    for song in songs:
        total_chords += len(song.chords)
        total_notes += len(song.notes)
        total_quarters += song.duration
        modes[song.key.mode] = modes.get(song.key.mode, 0) + 1
        meters[song.time_signature] = meters.get(song.time_signature, 0) + 1
        for span in song.chords:
            chord_counts[span.chord.figure()] = chord_counts.get(span.chord.figure(), 0) + 1
            qualities[span.chord.quality] = qualities.get(span.chord.quality, 0) + 1
    return {
        "songs": len(songs),
        "melody_notes": total_notes,
        "chords": total_chords,
        "quarters": round(total_quarters, 1),
        "hours_at_120bpm": round(total_quarters / 120.0, 2),
        "modes": modes,
        "meters": dict(sorted(meters.items(), key=lambda kv: -kv[1])[:8]),
        "top_chords": sorted(chord_counts.items(), key=lambda kv: -kv[1])[:15],
        "qualities": dict(sorted(qualities.items(), key=lambda kv: -kv[1])),
        "unique_chords": len(chord_counts),
        "mean_chord_quarters": round(total_quarters / max(total_chords, 1), 2),
    }


def melody_signature(song: Song, notes: int = 12) -> tuple | None:
    """A transposition-invariant fingerprint of a lead-sheet melody.

    Two archives of the same Wikifonia score often disagree on title spelling
    and composer formatting while carrying the same tune.  A hash of the first
    notes, expressed as intervals from the first pitch, catches those duplicates
    without matching unrelated songs.  On OpenEWLD it finds 449 of the 502
    scores inside the AutoHarmonizer archive, including 40 of the 44 test
    songs that a title-only deduplication missed.
    """
    ordered = sorted(song.notes, key=lambda note: note.onset)[: max(6, int(notes))]
    if len(ordered) < 6:
        return None
    first = ordered[0].pitch
    return tuple(
        (
            int(round(note.pitch - first)),
            int(round(note.onset * 4.0)),
            int(round(note.duration * 4.0)),
        )
        for note in ordered
    )


def _group_key(song: Song) -> str:
    composer = (song.composer or "unknown").strip().lower()
    return composer


def split_corpus(
    songs: Sequence[Song],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[Song]]:
    """Composer-disjoint train/dev/test split.

    A single composer with a large catalogue would otherwise leak the same
    harmonic habits into both sides of the split; groups larger than 8% of the
    corpus are split by song id instead.
    """
    groups: dict[str, list[Song]] = {}
    for song in songs:
        groups.setdefault(_group_key(song), []).append(song)

    threshold = max(8, int(len(songs) * 0.08))
    buckets: list[list[Song]] = [[], [], []]
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda s: s.song_id)
        if len(members) > threshold:
            for song in members:
                digest = hashlib.sha256(f"{seed}:{song.song_id}".encode()).digest()
                roll = int.from_bytes(digest[:4], "big") / 2**32
                if roll < ratios[0]:
                    buckets[0].append(song)
                elif roll < ratios[0] + ratios[1]:
                    buckets[1].append(song)
                else:
                    buckets[2].append(song)
            continue
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        roll = int.from_bytes(digest[:4], "big") / 2**32
        if roll < ratios[0]:
            bucket = 0
        elif roll < ratios[0] + ratios[1]:
            bucket = 1
        else:
            bucket = 2
        buckets[bucket].extend(members)
    return {"train": buckets[0], "dev": buckets[1], "test": buckets[2]}


def chord_change_rate(song: Song) -> float:
    """Chords per bar, a simple measure of harmonic density."""
    bars = max(song.duration / song.beats_per_bar, 1e-6)
    return len(song.chords) / bars


def span_melody_chroma(
    song: Song,
    span: ChordSpan,
    *,
    soft: bool = False,
    sigma_semitones: float = 0.45,
    metric: bool = True,
) -> np.ndarray:
    """Duration-weighted pitch-class histogram of the melody over ``span``.

    With ``soft=True`` each note contributes a Gaussian in cents rather than
    landing entirely in one bin.  The decoder uses the soft form; training uses
    the same accumulation so the learned distributions match inference.
    """
    chroma = np.zeros(12, dtype=np.float64)
    for item in song.notes:
        overlap = min(item.end, span.end) - max(item.onset, span.onset)
        if overlap <= 1e-9:
            continue
        weight = overlap
        if metric:
            weight *= metric_weight(item.onset, song.beats_per_bar)
        if soft:
            # Gaussian on the pitch-class circle (wrapped, so B and C mix).
            for pc in range(12):
                delta = (item.pitch - pc) % 12
                delta = min(delta, 12 - delta)
                chroma[pc] += weight * math.exp(-0.5 * (delta / sigma_semitones) ** 2)
        else:
            chroma[item.pc] += weight
    return chroma

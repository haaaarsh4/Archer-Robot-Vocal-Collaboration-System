"""Objective evaluation on held-out lead sheets.

Melody harmonization has no single correct answer, so accuracy against the
corpus's chord symbols is a lower bound on musical quality, not a verdict.
It is still the only honest way to compare models without a listening panel,
and it catches regressions.

Metrics
-------
root / core / full
    Frame-weighted accuracy at quarter-beat resolution.  ``core`` requires the
    root and one of the five MySong triads; ``full`` also requires the exact
    extension (7th, 6th, ...).
key
    Exact key and mode, and tonic-only accuracy, against the corpus label.

Baselines
---------
``fixed_bar``
    MySong's model: one chord per bar, Viterbi, same classifier and transition
    matrix.  This isolates the effect of flexible harmonic rhythm.
``generative``
    The classifier replaced by the paper's ``P(note | chord)`` dot product,
    with everything else identical.  This isolates the discriminative
    observation model.
``template``
    Per bar, the diatonic triad whose tones best match the melody (cosine).
    A simple, key-aware baseline with no sequence model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mysong.corpus import Song, load_corpus, split_corpus
from mysong.harmonize import Harmonizer, Harmonization, Style, chords_for_bars
from mysong.melody import Melody
from mysong.model import MySongModel
from mysong.theory import Chord, core_index


def chord_change_f1(song: Song, harmony: Harmonization, *, tolerance: float = 0.5) -> dict:
    """Precision/recall of chord-change positions, one-to-one within tolerance.

    Frame accuracy rewards holding a safe chord; this catches a model that
    never changes harmony at all (or changes on every beat), which is exactly
    the failure mode a frame-weighted score hides.
    """
    reference = [span.onset for span in song.chords[1:]]
    predicted = [segment.start_beat for segment in harmony.segments[1:]]
    used: set[int] = set()
    hits = 0
    for position in predicted:
        for index, target in enumerate(reference):
            if index in used:
                continue
            if abs(position - target) <= tolerance:
                used.add(index)
                hits += 1
                break
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def chord_accuracy(song: Song, harmony: Harmonization, *, resolution: float = 0.25) -> dict:
    """Frame-weighted accuracy of a predicted chord sequence."""
    total = root = core = full = 0.0
    for span in song.chords:
        if span.duration <= 0:
            continue
        for beat in np.arange(span.onset, span.end, resolution):
            predicted = harmony.chord_at(float(beat))
            if predicted is None:
                continue
            weight = min(resolution, span.end - float(beat))
            total += weight
            if predicted.root == span.chord.root:
                root += weight
                if predicted.core_quality == span.chord.core_quality:
                    core += weight
                    if predicted.quality == span.chord.quality:
                        full += weight
    if total <= 0:
        return {"root": 0.0, "core": 0.0, "full": 0.0, "beats": 0.0}
    return {"root": root / total, "core": core / total, "full": full / total, "beats": total}


@dataclass
class EvalRow:
    song: str
    reference_key: str
    predicted_key: str
    key_exact: bool
    key_tonic: bool
    root: float
    core: float
    full: float
    segments: int
    change_f1: float = 0.0

    def to_dict(self) -> dict:
        return {
            "song": self.song,
            "reference_key": self.reference_key,
            "predicted_key": self.predicted_key,
            "key_exact": self.key_exact,
            "key_tonic": self.key_tonic,
            "root": round(self.root, 4),
            "core": round(self.core, 4),
            "full": round(self.full, 4),
            "segments": self.segments,
            "change_f1": round(self.change_f1, 4),
        }


def _predictions_accuracy(song: Song, predictions: Sequence[tuple[float, float, Chord]]) -> dict:
    total = root = core = 0.0
    for span in song.chords:
        for beat in np.arange(span.onset, span.end, 0.25):
            hit = None
            for start, end, chord in predictions:
                if start <= beat < end:
                    hit = chord
                    break
            if hit is None:
                continue
            total += 0.25
            if hit.root == span.chord.root:
                root += 0.25
                if hit.core_quality == span.chord.core_quality:
                    core += 0.25
    return {
        "root": root / total if total else 0.0,
        "core": core / total if total else 0.0,
        "full": 0.0,
        "beats": total,
    }


def template_baseline(song: Song) -> list[tuple[float, float, Chord]]:
    """Per-bar diatonic triad by chroma cosine, using the reference key."""
    melody = Melody.from_notes(song.notes, beats_per_bar=song.beats_per_bar)
    triads = song.key.diatonic_triads()
    bars = int(np.ceil(melody.total_beats / song.beats_per_bar))
    predictions: list[tuple[float, float, Chord]] = []
    for bar in range(bars):
        start = bar * song.beats_per_bar
        end = min(melody.total_beats, start + song.beats_per_bar)
        chroma = melody.slice_chroma(start, end)
        norm = float(np.linalg.norm(chroma))
        best: tuple[float, Chord] | None = None
        for triad in triads:
            mask = np.isin(np.arange(12), list(triad.pitch_classes(False))).astype(float)
            mask /= np.linalg.norm(mask)
            score = float(chroma @ mask) / (norm + 1e-9)
            if best is None or score > best[0]:
                best = (score, triad)
        assert best is not None
        predictions.append((start, end, best[1]))
    return predictions


def evaluate(
    model: MySongModel,
    songs: Sequence[Song],
    *,
    style: Style | None = None,
    oracle_key: bool = False,
    include_baselines: bool = True,
    limit: int = 0,
) -> dict:
    """Evaluate one model configuration on a list of songs."""
    style = style or Style()
    harmonizer = Harmonizer(model)
    if limit:
        songs = songs[:limit]
    rows: list[EvalRow] = []
    fixed_rows: list[EvalRow] = []
    template_rows: list[dict] = []

    for song in songs:
        melody = Melody.from_notes(song.notes, beats_per_bar=song.beats_per_bar)
        harmony = harmonizer.harmonize(
            melody,
            key=song.key if oracle_key else None,
            style=style,
            beats_per_bar=song.beats_per_bar,
            with_marginals=False,
        )
        metrics = chord_accuracy(song, harmony)
        changes = chord_change_f1(song, harmony)
        rows.append(EvalRow(
            song=song.title,
            reference_key=song.key.name(),
            predicted_key=harmony.key.name(),
            key_exact=(harmony.key.tonic == song.key.tonic and harmony.key.mode == song.key.mode),
            key_tonic=(harmony.key.tonic == song.key.tonic),
            root=metrics["root"], core=metrics["core"], full=metrics["full"],
            segments=len(harmony.segments), change_f1=changes["f1"],
        ))
        if include_baselines:
            fixed = harmonizer.harmonize_fixed_bars(
                melody, key=song.key, style=style, beats_per_bar=song.beats_per_bar
            )
            fixed_metrics = chord_accuracy(song, fixed)
            fixed_rows.append(EvalRow(
                song=song.title,
                reference_key=song.key.name(),
                predicted_key=song.key.name(),
                key_exact=True, key_tonic=True,
                root=fixed_metrics["root"], core=fixed_metrics["core"],
                full=fixed_metrics["full"], segments=len(fixed.segments),
            ))
            template_rows.append(_predictions_accuracy(song, template_baseline(song)))

    def summarise(items: Sequence[EvalRow] | Sequence[dict]) -> dict:
        if not items:
            return {}
        if isinstance(items[0], EvalRow):
            return {
                "n": len(items),
                "root": round(float(np.mean([item.root for item in items])), 4),
                "core": round(float(np.mean([item.core for item in items])), 4),
                "full": round(float(np.mean([item.full for item in items])), 4),
                "key_exact": round(float(np.mean([item.key_exact for item in items])), 4),
                "key_tonic": round(float(np.mean([item.key_tonic for item in items])), 4),
                "segments_per_song": round(float(np.mean([item.segments for item in items])), 2),
                "change_f1": round(float(np.mean([item.change_f1 for item in items])), 4),
            }
        return {
            "n": len(items),
            "root": round(float(np.mean([item["root"] for item in items])), 4),
            "core": round(float(np.mean([item["core"] for item in items])), 4),
        }

    result = {
        "mode": "oracle_key" if oracle_key else "auto_key",
        "ours": summarise(rows),
        "rows": [row.to_dict() for row in rows],
    }
    if include_baselines:
        result["fixed_bar"] = summarise(fixed_rows)
        result["template"] = summarise(template_rows)
    return result


def evaluate_generative(
    model: MySongModel,
    songs: Sequence[Song],
    *,
    style: Style | None = None,
    limit: int = 0,
) -> dict:
    """The paper's generative observation, with everything else unchanged."""
    saved_classifier = model.classifier
    saved_tagger = getattr(model, "tagger", None)
    model.classifier = None
    model.tagger = None
    try:
        return evaluate(model, songs, style=style, oracle_key=True,
                        include_baselines=False, limit=limit)
    finally:
        model.classifier = saved_classifier
        model.tagger = saved_tagger


def evaluate_corpus(
    model: MySongModel,
    *,
    cache: str | Path = "data/mysong/corpus.jsonl",
    split: str = "test",
    limit: int = 0,
    include_generative: bool = True,
) -> dict:
    songs = load_corpus(cache)
    splits = split_corpus(songs)
    items = splits.get(split, [])
    if not items:
        raise ValueError(f"split {split!r} is empty")
    report = {
        "split": split,
        "corpus_songs": len(songs),
        "auto": evaluate(model, items, oracle_key=False, limit=limit),
        "oracle": evaluate(model, items, oracle_key=True, limit=limit),
    }
    if include_generative:
        report["generative"] = evaluate_generative(model, items, limit=limit)
    return report


def format_report(report: dict) -> str:
    lines = []
    split = report.get("split", "")
    lines.append(f"split: {split}" + (f" ({report.get('corpus_songs')} songs in corpus)" if report.get("corpus_songs") else ""))
    for name in ("auto", "oracle", "generative"):
        block = report.get(name)
        if not block:
            continue
        ours = block["ours"]
        lines.append(
            f"  {name:10} n={ours['n']:3}  root={ours['root']:.3f}  "
            f"core={ours['core']:.3f}  full={ours['full']:.3f}"
            + (f"  key={ours['key_exact']:.3f}" if "key_exact" in ours else "")
            + (f"  changes={ours['change_f1']:.3f}" if "change_f1" in ours else "")
        )
        for baseline in ("fixed_bar", "template"):
            if baseline in block:
                value = block[baseline]
                lines.append(
                    f"  {baseline:10} n={value['n']:3}  root={value['root']:.3f}  core={value['core']:.3f}"
                )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the MySong harmonizer")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", default="data/mysong/corpus.jsonl")
    parser.add_argument("--model", default="data/mysong/model/mysong.npz")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate_corpus(
        MySongModel.load(args.model),
        cache=args.cache,
        split=args.split,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

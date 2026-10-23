"""Estimate the MySong potentials from a lead-sheet corpus.

The trainer is deliberately transparent: every potential is a normalised count
table with an explicit, music-theoretic prior.  Nothing here needs a GPU, and
the whole model fits in a few hundred kilobytes.

Model
-----
For a chord sequence ``c_1 .. c_T`` in a known key/mode:

``P(c | melody) ∝ P(c_1) P(c_T | c_{T-1}) ... P(c_T) prod_t P(melody_t | c_t)``

with a semi-Markov duration term for how long each chord lasts.  The corpus is
transposed to the tonic before counting, so the model generalises across all
twelve keys without sharing statistics across relative major/minor pairs.

Counts are smoothed toward structured priors, not toward a uniform
distribution:

* transitions toward a factorised prior over circle-of-fifths root motion and
  quality pairs (so an unseen ``bVII -> I`` still gets a sensible probability);
* melody observations toward a chord-tone template blended with the corpus
  pitch-class distribution;
* durations toward the corpus marginal for the meter, with a separate prior
  for *where* in the bar changes happen.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mysong.corpus import (
    DEFAULT_CACHE,
    Song,
    corpus_stats,
    load_corpus,
    span_melody_chroma,
    split_corpus,
)
from mysong.chord_model import train_classifier
from mysong.key_model import song_chroma, train_key_classifier
from mysong.model import (
    DEFAULT_DURATIONS,
    DEFAULT_METERS,
    DEFAULT_MODEL,
    MODEL_VERSION,
    ModelMeta,
    MySongModel,
    now_iso,
)
from mysong.theory import (
    CORE_QUALITIES,
    QUALITY_INTERVALS,
    Chord,
    core_index,
    fifths_distance,
)


@dataclass
class TrainConfig:
    corpus: Path = DEFAULT_CACHE
    out: Path = DEFAULT_MODEL
    seed: int = 0
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    alpha_transition: float = 6.0
    beta_observation: float = 25.0
    alpha_duration: float = 4.0
    alpha_extension: float = 12.0
    soft_observation: bool = True
    sigma_semitones: float = 0.5
    durations: tuple[float, ...] = DEFAULT_DURATIONS
    meters: tuple[float, ...] = DEFAULT_METERS
    train_classifier: bool = True
    classifier_C: float = 0.2
    classifier_max_iter: int = 300
    train_key_classifier: bool = True
    key_classifier_C: float = 0.1
    train_tagger: bool = True
    tagger_epochs: int = 25
    tagger_hidden: int = 128
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #

def _normalise_rows(counts: np.ndarray, alpha: float = 0.0, prior: np.ndarray | None = None) -> np.ndarray:
    if prior is None:
        prior = np.full_like(counts, 1.0 / max(counts.shape[-1], 1))
    smoothed = counts + alpha * prior
    totals = smoothed.sum(axis=-1, keepdims=True)
    totals[totals <= 0] = 1.0
    return smoothed / totals


def _log(array: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(array, floor))


def transition_prior(counts: np.ndarray) -> np.ndarray:
    """Factorised prior over root motion and quality pairs.

    ``counts`` is ``(states, states)`` in tonic-relative core-state order.
    Returns a row-stochastic ``(states, states)`` matrix.
    """
    n = counts.shape[0]
    qualities = len(CORE_QUALITIES)
    root_interval = np.zeros((len(CORE_QUALITIES), 12), dtype=np.float64)
    quality_pair = np.zeros((len(CORE_QUALITIES), len(CORE_QUALITIES)), dtype=np.float64)
    for previous in range(n):
        qa = previous % qualities
        for following in range(n):
            qb = following % qualities
            delta = fifths_distance(previous // qualities, following // qualities)
            root_interval[qa, delta + 6] += counts[previous, following]
            quality_pair[qa, qb] += counts[previous, following]
    root_interval = _normalise_rows(root_interval)
    quality_pair = _normalise_rows(quality_pair)
    prior = np.zeros_like(counts)
    for previous in range(n):
        qa = previous % qualities
        root_a = previous // qualities
        for following in range(n):
            qb = following % qualities
            delta = fifths_distance(root_a, following // qualities)
            prior[previous, following] = root_interval[qa, delta + 6] * quality_pair[qa, qb]
    return _normalise_rows(prior, alpha=1e-3)


def observation_base() -> np.ndarray:
    """Chord-tone template for each of the 60 core states."""
    base = np.zeros((60, 12), dtype=np.float64)
    for root in range(12):
        for qindex, quality in enumerate(CORE_QUALITIES):
            state = root * len(CORE_QUALITIES) + qindex
            intervals = {
                "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
                "aug": (0, 4, 8), "sus": (0, 5, 7),
            }[quality]
            weights = np.full(12, 0.04, dtype=np.float64)
            for index, interval in enumerate(intervals):
                pc = (root + interval) % 12
                weights[pc] = (1.6, 1.05, 1.0)[min(index, 2)]
            # The leading tone and the ninth are common melodic colours over
            # dominant-function triads; a light bump keeps those from being
            # treated as errors when a chord has few observations.
            if quality == "maj":
                weights[(root + 11) % 12] += 0.18
                weights[(root + 2) % 12] += 0.12
            elif quality == "min":
                weights[(root + 10) % 12] += 0.12
            base[state] = weights / weights.sum()
    return base


def duration_base(durations: Sequence[float]) -> np.ndarray:
    """A gentle prior favouring whole bars over very short chords."""
    values = np.asarray(durations, dtype=np.float64)
    # 4 quarter notes is the mode in 4/4; express the prior in bars-agnostic
    # terms so 3/4 does not get a different shape.
    strength = 0.6 + 0.4 * np.exp(-0.5 * ((np.log2(values / 4.0)) / 0.9) ** 2)
    return strength / strength.sum()


def extension_base(core: str, qualities: Sequence[str]) -> np.ndarray:
    preferred = {
        "maj": {"maj": 0.52, "7": 0.14, "6": 0.07, "maj7": 0.15, "9": 0.05,
                "add9": 0.03, "13": 0.015, "11": 0.005, "5": 0.01, "maj9": 0.01},
        "min": {"min": 0.50, "min7": 0.31, "m6": 0.05, "min9": 0.05,
                "mMaj7": 0.01, "m11": 0.01, "madd9": 0.02, "min7": 0.31},
        "dim": {"dim": 0.62, "dim7": 0.30, "m7b5": 0.08},
        "aug": {"aug": 0.94, "7": 0.06},
        "sus": {"sus4": 0.58, "sus2": 0.22, "7sus4": 0.16, "9sus4": 0.04},
    }.get(core, {})
    values = np.zeros(len(qualities), dtype=np.float64)
    for index, quality in enumerate(qualities):
        values[index] = preferred.get(quality, 0.001)
    if values.sum() <= 0:
        values[:] = 1.0
    return values / values.sum()


def chord_prior_base(mode: int) -> np.ndarray:
    """Weak tonic-relative prior over the 60 core triads.

    Without this, a bag-of-pitch-class likelihood happily explains a melody
    containing only C, D and E with an E augmented triad: the triad covers C
    and E, and the absent G# costs it almost nothing.  A unigram term is the
    standard fix, and it also keeps the decoder from inventing diminished
    chords in a major-key song.
    """
    scale = (0, 2, 4, 5, 7, 9, 11) if mode == 0 else (0, 2, 3, 5, 7, 8, 10)
    quality_weight = {"maj": 1.0, "min": 0.9, "dim": 0.22, "aug": 0.10, "sus": 0.30}
    weights = np.zeros(60, dtype=np.float64)
    for root in range(12):
        if root == 0:
            root_weight = 1.7
        elif root == 7:
            root_weight = 1.3
        elif root == 5:
            root_weight = 1.2
        elif root in scale:
            root_weight = 0.9
        else:
            root_weight = 0.25
        for qindex, quality in enumerate(CORE_QUALITIES):
            weights[root * len(CORE_QUALITIES) + qindex] = root_weight * quality_weight[quality]
    return weights / weights.sum()


def bass_base(qualities: Sequence[str] | None = None) -> np.ndarray:
    # Root, third, fifth, seventh, other.
    return np.asarray([0.72, 0.14, 0.10, 0.03, 0.01], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #

def _phase_bucket(offset: float, beats_per_bar: float) -> int:
    phase = float(offset) % float(beats_per_bar)
    if phase < 0.25 or abs(phase - beats_per_bar) < 0.25:
        return 0
    if abs(phase - beats_per_bar / 2.0) < 0.25:
        return 1
    return 2


def _snap_duration(value: float, durations: Sequence[float]) -> int | None:
    array = np.asarray(durations, dtype=np.float64)
    index = int(np.argmin(np.abs(array - float(value))))
    # Accept anything within half a beat; the corpus writes ties and triplets
    # that should not create a new duration class.
    if abs(array[index] - float(value)) <= 0.5:
        return index
    return None


def _bass_class(bass: int | None, chord: Chord) -> int:
    if bass is None or bass == chord.root:
        return 0
    interval = (bass - chord.root) % 12
    if interval in (3, 4):
        return 1
    if interval == 7:
        return 2
    if interval in (10, 11):
        return 3
    return 4


@dataclass
class Counts:
    transition: np.ndarray = field(default_factory=lambda: np.zeros((2, 60, 60)))
    start: np.ndarray = field(default_factory=lambda: np.zeros((2, 60)))
    end: np.ndarray = field(default_factory=lambda: np.zeros((2, 60)))
    observation: np.ndarray = field(default_factory=lambda: np.zeros((60, 12)))
    duration: dict[int, np.ndarray] = field(default_factory=dict)
    phase: dict[int, np.ndarray] = field(default_factory=dict)
    extension: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    bass: dict[str, np.ndarray] = field(default_factory=dict)
    unigram: np.ndarray = field(default_factory=lambda: np.zeros((2, 60)))
    modes: np.ndarray = field(default_factory=lambda: np.zeros(2))
    songs: int = 0
    transitions: int = 0


def count_corpus(
    songs: Sequence[Song],
    config: TrainConfig,
    counts: Counts | None = None,
) -> Counts:
    counts = counts or Counts()
    durations = np.asarray(config.durations, dtype=np.float64)
    for song in songs:
        if not song.chords:
            continue
        mode = 1 if song.key.mode == "minor" else 0
        rel = song.in_tonic_relative()
        meter_index = _meter_index(song.beats_per_bar, config.meters)
        for index, span in enumerate(rel.chords):
            core = span.chord.core
            state = core_index(core)
            if index == 0:
                counts.start[mode, state] += 1
            else:
                previous = core_index(rel.chords[index - 1].chord.core)
                counts.transition[mode, previous, state] += 1
                counts.transitions += 1
            if index == len(rel.chords) - 1:
                counts.end[mode, state] += 1
            counts.unigram[mode, state] += span.duration

            chroma = span_melody_chroma(
                rel, span,
                soft=config.soft_observation,
                sigma_semitones=config.sigma_semitones,
                metric=True,
            )
            if chroma.sum() > 0:
                counts.observation[state] += chroma

            duration_index = _snap_duration(span.duration, durations)
            if duration_index is not None:
                bucket = counts.duration.setdefault(
                    meter_index, np.zeros(len(durations), dtype=np.float64)
                )
                bucket[duration_index] += 1
                phase = counts.phase.setdefault(
                    meter_index, np.zeros(3, dtype=np.float64)
                )
                phase[_phase_bucket(span.onset, rel.beats_per_bar)] += 1

            key = (mode, core.core_quality)
            table = counts.extension.setdefault(
                key, np.zeros(len(QUALITY_INTERVALS), dtype=np.float64)
            )
            table[list(QUALITY_INTERVALS).index(span.chord.quality)] += 1

            bass_table = counts.bass.setdefault(
                core.core_quality, np.zeros(5, dtype=np.float64)
            )
            bass_table[_bass_class(span.chord.bass, span.chord)] += 1

        counts.modes[mode] += 1
        counts.songs += 1
    return counts


def _meter_index(beats_per_bar: float, meters: Sequence[float]) -> int:
    values = np.asarray(meters, dtype=np.float64)
    return int(np.argmin(np.abs(values - float(beats_per_bar))))


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def fit_model(
    songs: Sequence[Song],
    config: TrainConfig | None = None,
    *,
    split: dict[str, int] | None = None,
    verbose: bool = False,
) -> tuple[MySongModel, dict]:
    config = config or TrainConfig()
    if not songs:
        raise ValueError("no training songs")
    counts = count_corpus(songs, config)

    n_modes = 2
    n_states = 60
    transitions = np.zeros((n_modes, n_states, n_states), dtype=np.float64)
    for mode in range(n_modes):
        prior = transition_prior(counts.transition[mode])
        transitions[mode] = _log(_normalise_rows(
            counts.transition[mode], alpha=config.alpha_transition, prior=prior
        ))

    start = _log(_normalise_rows(counts.start, alpha=0.5))
    end = _log(_normalise_rows(counts.end, alpha=0.5))
    mode_prior = (counts.modes + 0.5) / (counts.modes.sum() + 1.0)
    chord_logprior = np.zeros((n_modes, n_states), dtype=np.float64)
    for mode in range(n_modes):
        chord_logprior[mode] = _log(_normalise_rows(
            counts.unigram[mode][None, :],
            alpha=40.0,
            prior=chord_prior_base(mode)[None, :],
        )[0])

    # Melody observation: chord-tone prior blended with the corpus marginal.
    global_pc = counts.observation.sum(axis=0)
    global_pc = (global_pc + 0.5) / (global_pc.sum() + 6.0)
    base = 0.85 * observation_base() + 0.15 * global_pc[None, :]
    base = base / base.sum(axis=1, keepdims=True)
    observation = _log(_normalise_rows(
        counts.observation, alpha=config.beta_observation, prior=base
    ))

    durations = np.asarray(config.durations, dtype=np.float64)
    duration_prior_base = duration_base(durations)
    duration_logprob = np.zeros((len(config.meters), len(durations)), dtype=np.float64)
    phase_logprob = np.zeros((len(config.meters), 3), dtype=np.float64)
    for meter_index in range(len(config.meters)):
        duration_counts = counts.duration.get(
            meter_index, np.zeros(len(durations), dtype=np.float64)
        )
        duration_logprob[meter_index] = _log(
            (duration_counts + config.alpha_duration * duration_prior_base)
            / (duration_counts.sum() + config.alpha_duration)
        )
        phase_counts = counts.phase.get(meter_index, np.zeros(3, dtype=np.float64))
        phase_logprob[meter_index] = _log(
            (phase_counts + 1.0) / (phase_counts.sum() + 3.0)
        )

    qualities = list(QUALITY_INTERVALS)
    extension: dict[str, np.ndarray] = {}
    for mode in range(n_modes):
        for core in CORE_QUALITIES:
            key = (mode, core)
            table = counts.extension.get(key, np.zeros(len(qualities), dtype=np.float64))
            prior = extension_base(core, qualities)
            extension[f"{mode}:{core}"] = _log(
                (table + config.alpha_extension * prior)
                / (table.sum() + config.alpha_extension)
            )

    bass_tables: dict[str, np.ndarray] = {}
    for core in CORE_QUALITIES:
        table = counts.bass.get(core, np.zeros(5, dtype=np.float64))
        prior = bass_base()
        bass_tables[core] = _log((table + 2.0 * prior) / (table.sum() + 2.0))

    classifier = None
    classifier_report: dict = {}
    if config.train_classifier:
        classifier, classifier_report = train_classifier(
            songs,
            C=config.classifier_C,
            max_iter=config.classifier_max_iter,
            seed=config.seed,
        )

    key_classifier = None
    key_report: dict = {}
    key_profiles = None
    tagger_report: dict = {}
    if config.train_key_classifier:
        key_classifier, key_report = train_key_classifier(
            songs, C=config.key_classifier_C, seed=config.seed
        )
        key_profiles = np.zeros((2, 12), dtype=np.float64)
        for mode_name, mode_value in (("major", 0), ("minor", 1)):
            rows = []
            for song in songs:
                if song.key.mode != mode_name:
                    continue
                chroma = song_chroma(song)
                rows.append([chroma[(pc + song.key.tonic) % 12] for pc in range(12)])
            if rows:
                profile = np.mean(rows, axis=0)
                key_profiles[mode_value] = profile / max(profile.sum(), 1e-12)
            else:
                key_profiles[mode_value] = 1.0 / 12.0

    if config.train_tagger:
        try:
            from mysong.neural import train_tagger

            tagger_path = Path(config.out).with_name(Path(config.out).stem + "_tagger.pt")
            tagger_report = train_tagger(
                songs,
                tagger_path,
                epochs=config.tagger_epochs,
                hidden=config.tagger_hidden,
                seed=config.seed,
                log=(print if verbose else (lambda *args, **kwargs: None)),
            ).to_dict()
            tagger_report["path"] = str(tagger_path)
        except Exception as exc:  # pragma: no cover - optional dependency
            tagger_report = {"error": str(exc)}

    stats = corpus_stats(songs)
    meta = ModelMeta(
        version=MODEL_VERSION,
        trained_at=now_iso(),
        songs=len(songs),
        transitions=int(counts.transitions),
        melody_notes=sum(len(s.notes) for s in songs),
        split=split or {},
        corpus=stats,
        config={
            "alpha_transition": config.alpha_transition,
            "beta_observation": config.beta_observation,
            "alpha_duration": config.alpha_duration,
            "alpha_extension": config.alpha_extension,
            "soft_observation": config.soft_observation,
            "sigma_semitones": config.sigma_semitones,
            "durations": list(config.durations),
            "meters": list(config.meters),
            "split_ratios": list(config.split_ratios),
            "seed": config.seed,
            "classifier": classifier_report,
            "key_classifier": key_report,
            "tagger": tagger_report,
        },
        notes=list(config.notes),
    )
    model = MySongModel(
        transition=transitions,
        start=start,
        end=end,
        observation=observation,
        durations=durations,
        duration_logprob=duration_logprob,
        phase_logprob=phase_logprob,
        meters=np.asarray(config.meters, dtype=np.float64),
        mode_logprior=_log(mode_prior),
        chord_logprior=chord_logprior,
        extension=extension,
        extension_qualities=qualities,
        bass=bass_tables,
        meta=meta,
        classifier=classifier,
        key_classifier=key_classifier,
        key_profiles=key_profiles,
    )
    report = {
        "classifier": classifier_report,
        "key_classifier": key_report,
        "tagger": tagger_report,
        "songs": len(songs),
        "transitions": int(counts.transitions),
        "top_transitions": top_transitions(transitions[0], limit=10),
        "duration_counts": {
            str(config.meters[index]): {
                str(durations[i]): int(counts.duration.get(index, np.zeros(len(durations)))[i])
                for i in range(len(durations))
            }
            for index in range(len(config.meters))
            if index in counts.duration
        },
    }
    return model, report


def top_transitions(matrix: np.ndarray, limit: int = 10) -> list[tuple[str, str, float]]:
    """Human-readable most likely transitions (major mode by default)."""
    flat = np.argsort(matrix.ravel())[::-1][:limit]
    out = []
    for flat_index in flat:
        previous, following = divmod(int(flat_index), matrix.shape[1])
        pa, qa = divmod(previous, len(CORE_QUALITIES))
        pb, qb = divmod(following, len(CORE_QUALITIES))
        out.append((
            str(Chord(pa, CORE_QUALITIES[qa])),
            str(Chord(pb, CORE_QUALITIES[qb])),
            float(matrix[previous, following]),
        ))
    return out


def train_from_cache(
    config: TrainConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[MySongModel, dict, dict[str, list[Song]]]:
    config = config or TrainConfig()
    songs = load_corpus(config.corpus)
    splits = split_corpus(songs, ratios=config.split_ratios, seed=config.seed)
    model, report = fit_model(
        splits["train"],
        config,
        split={name: len(items) for name, items in splits.items()},
        verbose=verbose,
    )
    path = model.save(config.out)
    report["model_path"] = str(path)
    report["split"] = {name: len(items) for name, items in splits.items()}
    if verbose:
        print(model.describe())
        print(f"paths: {path} ({path.stat().st_size / 1024:.0f} KiB)")
        print("top major transitions:")
        for previous, following, probability in report["top_transitions"]:
            print(f"  {previous:>6} -> {following:<6} {probability:.3f}")
        print("split:", report["split"])
    return model, report, splits


def dump_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

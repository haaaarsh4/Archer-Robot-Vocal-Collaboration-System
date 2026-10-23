"""The trained MySong model: arrays plus the small amount of metadata needed.

The model is deliberately tiny (a few hundred kilobytes).  It is a generative
model of chord sequences with a melody-observation likelihood, not a neural
network, so it runs on a CPU at interactive speed and can be inspected.

State space
-----------
60 core triads: ``root * 5 + quality`` in :data:`mysong.theory.CORE_QUALITIES`
order.  The mode (major/minor) selects a different transition matrix, exactly
as in MySong, but the melody-observation matrix is shared because the corpus
does not support two reliable matrices.

Everything stored as a "log" array is a natural logarithm of a probability
(rows normalised unless documented otherwise).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mysong.chord_model import ChordClassifier
from mysong.key_model import KeyClassifier
from mysong.theory import CORE_QUALITIES, Chord, core_index

MODEL_VERSION = 1
DEFAULT_MODEL = Path("data/mysong/model/mysong.npz")

#: Chord lengths the decoder may use, in quarter notes.  The corpus rarely
#: writes a chord shorter than a beat or longer than four bars.
DEFAULT_DURATIONS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)

#: Meter buckets (beats per bar) with enough corpus support to estimate a
#: duration prior.  Anything else falls back to the 4/4 bucket.
DEFAULT_METERS: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0)


@dataclass
class ModelMeta:
    version: int = MODEL_VERSION
    trained_at: str = ""
    songs: int = 0
    transitions: int = 0
    melody_notes: int = 0
    split: dict[str, int] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


class MySongModel:
    """Container for the learned potentials.

    Attributes
    ----------
    transition : (2, 60, 60)
        ``transition[mode, previous, next]`` log-probabilities.
    start, end : (2, 60)
        Virtual start/end chord distributions.
    observation : (60, 12)
        ``log P(melody pitch class | chord)``.
    durations : (D,)
        Candidate chord lengths in quarter notes.
    duration_logprob : (M, D)
        Prior over lengths for each meter bucket.
    phase_logprob : (M, 3)
        Per-change prior for downbeat / half-bar / other positions.
    extension : dict[str, (Q,)]
        ``log P(full quality | core quality)`` over ``extension_qualities``.
    """

    def __init__(
        self,
        *,
        transition: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        observation: np.ndarray,
        durations: np.ndarray,
        duration_logprob: np.ndarray,
        phase_logprob: np.ndarray,
        meters: np.ndarray,
        mode_logprior: np.ndarray,
        chord_logprior: np.ndarray,
        extension: dict[str, np.ndarray],
        extension_qualities: list[str],
        bass: dict[str, np.ndarray],
        meta: ModelMeta,
        classifier: ChordClassifier | None = None,
        key_classifier: KeyClassifier | None = None,
        key_profiles: np.ndarray | None = None,
        tagger: Any | None = None,
    ) -> None:
        self.transition = np.asarray(transition, dtype=np.float64)
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.observation = np.asarray(observation, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.duration_logprob = np.asarray(duration_logprob, dtype=np.float64)
        self.phase_logprob = np.asarray(phase_logprob, dtype=np.float64)
        self.meters = np.asarray(meters, dtype=np.float64)
        self.mode_logprior = np.asarray(mode_logprior, dtype=np.float64)
        self.chord_logprior = np.asarray(chord_logprior, dtype=np.float64)
        self.extension = {k: np.asarray(v, dtype=np.float64) for k, v in extension.items()}
        self.extension_qualities = list(extension_qualities)
        self.bass = {k: np.asarray(v, dtype=np.float64) for k, v in bass.items()}
        self.classifier = classifier
        self.key_classifier = key_classifier
        self.key_profiles = (
            np.asarray(key_profiles, dtype=np.float64)
            if key_profiles is not None else None
        )
        self.tagger = tagger
        self.meta = meta

    # -- basic properties ------------------------------------------------ #
    @property
    def n_states(self) -> int:
        return int(self.transition.shape[-1])

    @property
    def core_qualities(self) -> tuple[str, ...]:
        return CORE_QUALITIES

    def mode_index(self, mode: str) -> int:
        return 1 if (mode or "").lower().startswith("min") else 0

    def meter_index(self, beats_per_bar: float) -> int:
        buckets = [float(m) for m in self.meters]
        best = 0
        for index, bucket in enumerate(buckets):
            if abs(bucket - float(beats_per_bar)) < abs(buckets[best] - float(beats_per_bar)):
                best = index
        return best

    # -- serialisation --------------------------------------------------- #
    def save(self, path: str | Path = DEFAULT_MODEL) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "transition": self.transition,
            "start": self.start,
            "end": self.end,
            "observation": self.observation,
            "durations": self.durations,
            "duration_logprob": self.duration_logprob,
            "phase_logprob": self.phase_logprob,
            "meters": self.meters,
            "mode_logprior": self.mode_logprior,
            "chord_logprior": self.chord_logprior,
            "extension_qualities": np.asarray(self.extension_qualities, dtype=object),
            "meta": np.asarray(self.meta.to_json()),
        }
        for core, values in self.extension.items():
            arrays[f"extension:{core}"] = values
        for core, values in self.bass.items():
            arrays[f"bass:{core}"] = values
        if self.classifier is not None:
            arrays.update(self.classifier.to_arrays())
        if self.key_classifier is not None:
            arrays.update(self.key_classifier.to_arrays())
        if self.key_profiles is not None:
            arrays["key_profiles"] = self.key_profiles
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> "MySongModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; train a model with `python -m mysong.cli train`"
            )
        with np.load(path, allow_pickle=True) as data:
            extension = {
                key.split(":", 1)[1]: data[key]
                for key in data.files
                if key.startswith("extension:")
            }
            bass = {
                key.split(":", 1)[1]: data[key]
                for key in data.files
                if key.startswith("bass:")
            }
            meta = ModelMeta(**json.loads(str(data["meta"])))
            classifier = None
            if "clf_weights" in data.files:
                classifier = ChordClassifier.from_arrays({
                    key: data[key]
                    for key in (
                        "clf_weights", "clf_bias", "clf_mean", "clf_scale",
                        "clf_classes", "clf_log_prior",
                    )
                    if key in data.files
                })
            key_classifier = None
            if "key_weights" in data.files:
                key_classifier = KeyClassifier.from_arrays({
                    key: data[key]
                    for key in (
                        "key_weights", "key_bias", "key_mean", "key_scale",
                        "key_classes", "key_log_prior",
                    )
                    if key in data.files
                })
            tagger = None
            tagger_candidates = [
                path.with_name(path.stem + "_tagger.npz"),
                path.with_name(path.stem + "_tagger.pt"),
            ]
            for tagger_path in tagger_candidates:
                if not tagger_path.exists():
                    continue
                try:
                    from mysong.neural import Tagger

                    tagger = Tagger.load(tagger_path)
                    break
                except Exception:  # pragma: no cover - optional dependency
                    tagger = None
            # An optional second tagger turns the observation model into a
            # two-model ensemble.  This is how the large-corpus tagger can be
            # combined with the small one without retraining either.
            second = path.with_name(path.stem + "_tagger2.npz")
            if tagger is not None and second.exists():
                try:
                    from mysong.neural import Tagger, TaggerEnsemble

                    tagger = TaggerEnsemble([tagger, Tagger.load(second)])
                except Exception:  # pragma: no cover - optional dependency
                    pass
            return cls(
                transition=data["transition"],
                start=data["start"],
                end=data["end"],
                observation=data["observation"],
                durations=data["durations"],
                duration_logprob=data["duration_logprob"],
                phase_logprob=data["phase_logprob"],
                meters=data["meters"],
                mode_logprior=data["mode_logprior"],
                chord_logprior=data["chord_logprior"],
                extension=extension,
                extension_qualities=[str(q) for q in data["extension_qualities"]],
                bass=bass,
                meta=meta,
                classifier=classifier,
                key_classifier=key_classifier,
                key_profiles=(
                    data["key_profiles"] if "key_profiles" in data.files else None
                ),
                tagger=tagger,
            )

    # -- convenience ----------------------------------------------------- #
    def state_chord(self, state: int, bass: int | None = None) -> Chord:
        root, qindex = divmod(int(state), len(CORE_QUALITIES))
        return Chord(root, CORE_QUALITIES[qindex], bass)

    def chord_state(self, chord: Chord) -> int:
        return core_index(chord)

    def duration_index(self, beats: float) -> int | None:
        if len(self.durations) == 0:
            return None
        index = int(np.argmin(np.abs(self.durations - float(beats))))
        if abs(self.durations[index] - float(beats)) > 1e-6:
            return None
        return index

    def describe(self) -> str:
        mode_count = ""
        if self.meta.split:
            mode_count = " " + " ".join(f"{k}={v}" for k, v in self.meta.split.items())
        extras = []
        if self.classifier is not None:
            extras.append("discriminative chords")
        if self.key_classifier is not None:
            extras.append("24-way key model")
        if self.tagger is not None:
            extras.append("BiGRU tagger")
        return (
            f"MySong model v{self.meta.version}: {self.meta.songs} songs, "
            f"{self.meta.transitions} chord transitions, "
            f"{len(self.durations)} durations{mode_count}"
            + (f" [{', '.join(extras)}]" if extras else "")
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MySongModel {self.meta.songs} songs, states={self.n_states}>"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

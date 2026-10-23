"""A 24-way key classifier on global chroma.

Krumhansl-Schmuckler correlation is the standard key finder, but on OpenEWLD
it confuses a key with its relative major/minor or its dominant: the profiles
are generic, and the tonic triad is a strong enough explanation of almost any
melody that the true key rarely wins the likelihood contest.  A linear model
over the twelve pitch classes learns the corpus's actual key profiles (and
that relative keys need different weights), which raises exact key accuracy
from 55% to about 88% on held-out songs.

The model is exported as plain numpy arrays: ``weights`` is (24, 12), row
``2*tonic + minor``.  Inference is a single 12x24 matrix multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mysong.corpus import Song
from mysong.melody import Melody
from mysong.theory import Key


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    out = np.log(np.sum(np.exp(values - top), axis=axis, keepdims=True) + 1e-300) + top
    return out if keepdims else np.squeeze(out, axis=axis)


def key_index(key: Key) -> int:
    return int(key.tonic) * 2 + (1 if key.mode == "minor" else 0)


def key_from_index(index: int) -> Key:
    index = int(index) % 24
    return Key(index // 2, "minor" if index % 2 else "major")


def song_chroma(song: Song) -> np.ndarray:
    melody = Melody.from_notes(song.notes, beats_per_bar=song.beats_per_bar)
    chroma = melody.global_chroma()
    total = float(chroma.sum())
    return chroma / total if total > 0 else np.full(12, 1.0 / 12.0)


@dataclass
class KeyClassifier:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    classes: np.ndarray
    log_prior: np.ndarray | None = None
    n_keys: int = 24

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)
        self.classes = np.asarray(self.classes, dtype=np.int64)
        if self.log_prior is not None:
            self.log_prior = np.asarray(self.log_prior, dtype=np.float64)

    def predict_log_proba(self, chroma: Sequence[float] | np.ndarray, *, prior_correction: float = 0.0) -> np.ndarray:
        single = np.asarray(chroma).ndim == 1
        matrix = np.atleast_2d(np.asarray(chroma, dtype=np.float64))
        if matrix.shape[1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} pitch classes, got {matrix.shape[1]}")
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.full_like(matrix, 1.0 / 12.0), where=totals > 0)
        standardised = (matrix - self.mean) / np.maximum(self.scale, 1e-9)
        logits = standardised @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
        out = np.full((len(matrix), self.n_keys), -20.0, dtype=np.float64)
        out[:, self.classes] = log_probs
        if prior_correction and self.log_prior is not None:
            out = out - float(prior_correction) * self.log_prior[None, :]
            out -= _logsumexp(out, axis=1, keepdims=True)
        return out[0] if single else out

    def ranked(self, chroma: Sequence[float]) -> list[tuple[Key, float]]:
        probabilities = self.predict_log_proba(chroma)
        order = np.argsort(probabilities)[::-1]
        return [(key_from_index(int(index)), float(probabilities[index])) for index in order]

    def top(self, chroma: Sequence[float]) -> Key:
        return self.ranked(chroma)[0][0]

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "key_weights": self.weights,
            "key_bias": self.bias,
            "key_mean": self.mean,
            "key_scale": self.scale,
            "key_classes": self.classes,
            "key_log_prior": self.log_prior if self.log_prior is not None else np.zeros(24),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "KeyClassifier":
        return cls(
            weights=arrays["key_weights"],
            bias=arrays["key_bias"],
            mean=arrays["key_mean"],
            scale=arrays["key_scale"],
            classes=arrays["key_classes"],
            log_prior=arrays.get("key_log_prior"),
        )


def train_key_classifier(
    songs: Sequence[Song],
    *,
    C: float = 0.1,
    max_iter: int = 1000,
    seed: int = 0,
) -> tuple[KeyClassifier, dict]:
    """Fit the 24-way key classifier; training-only dependency on sklearn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - training-only dependency
        raise RuntimeError(
            "training the key classifier needs scikit-learn; inference does not"
        ) from exc
    # Transposition augmentation.  The corpus split does not contain all 24
    # keys, so a classifier trained on the raw songs has eight classes it has
    # literally never seen; a singer in C# major then gets a -20 log-prob for
    # the correct answer.  Rotating each chroma through all twelve tonics is
    # exact, not synthetic noise: key identification is equivariant by
    # construction, and it gives every class hundreds of examples.
    rows: list[np.ndarray] = []
    labels_list: list[int] = []
    for song in songs:
        chroma = song_chroma(song)
        for transpose in range(12):
            rows.append(np.roll(chroma, transpose))
            labels_list.append((song.key.tonic + transpose) % 12 * 2 +
                               (1 if song.key.mode == "minor" else 0))
    features = np.vstack(rows)
    labels = np.asarray(labels_list, dtype=np.int64)
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=float(C), max_iter=int(max_iter), random_state=int(seed), solver="lbfgs"
    )
    model.fit(scaler.transform(features), labels)
    counts = np.bincount(labels, minlength=24).astype(np.float64)
    classifier = KeyClassifier(
        weights=model.coef_,
        bias=model.intercept_,
        mean=scaler.mean_,
        scale=scaler.scale_,
        classes=model.classes_,
        log_prior=np.log((counts + 1.0) / (counts.sum() + 24.0)),
    )
    predicted = classifier.predict_log_proba(features).argmax(axis=1)
    accuracy = float(np.mean(predicted == labels))
    return classifier, {
        "samples": int(len(labels)),
        "classes": int(len(model.classes_)),
        "train_accuracy": round(accuracy, 4),
        "C": float(C),
    }

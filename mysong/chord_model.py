"""A discriminative chord classifier, used as the melody potential.

Why this exists
---------------
MySong's melody potential is generative: ``P(note | chord)`` learned from note
durations, scored with a dot product.  That model cannot distinguish a C major
triad from an E augmented triad on a melody of C and E -- both explain the
notes, and the missing third costs almost nothing.  On OpenEWLD the generative
potential ranks the reference triad first for only about 5% of chord spans.

A small multinomial logistic regression on the same span features ranks the
reference triad first for 34% of held-out spans and in the top three for 66%.
It learns, from the corpus, that some pitch-class sets imply a root, that
downbeats matter more than passing notes, and that rare triads are rare.  The
weights are exported to plain numpy arrays, so inference needs no scikit-learn.

The classifier is small: 52 features, 60 classes, a few thousand
weights.  It trains in about fifteen seconds on a laptop CPU and cannot
plausibly memorise a song.  That matters because the same corpus supplies both
the classifier and the transition matrix; a transformer would need far more
data and would make the system harder to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mysong.corpus import Song
from mysong.melody import Melody
from mysong.theory import core_index

#: Order matters only for reproducibility of old checkpoints.
FEATURE_NAMES: tuple[str, ...] = (
    *(f"chroma_{i}" for i in range(12)),
    *(f"strong_{i}" for i in range(12)),
    *(f"onset_{i}" for i in range(12)),
    *(f"tail_{i}" for i in range(12)),
    "log_duration",
    "phase_sin",
    "phase_cos",
    "mode",
)
N_FEATURES = len(FEATURE_NAMES)


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    out = np.log(np.sum(np.exp(values - top), axis=axis, keepdims=True) + 1e-300) + top
    return out if keepdims else np.squeeze(out, axis=axis)


def _normalise(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0:
        return np.zeros(12, dtype=np.float64)
    return vector / total


def extract_features(
    melody: Melody,
    start: float,
    end: float,
    *,
    mode: str | int = "major",
    rotation: int = 0,
) -> np.ndarray:
    """Features for one candidate chord span.

    ``rotation`` maps absolute pitch classes into the candidate key: index
    ``q`` reads absolute pitch class ``(q + rotation) % 12``.  The classifier
    is trained in tonic-relative coordinates, so a C major model applied to a
    song in F passes ``rotation=5``.

    Everything is computed from prefix sums except the duration and phase, so
    scoring the thousands of spans the decoder considers costs milliseconds.
    """
    index = [(pc + int(rotation)) % 12 for pc in range(12)]
    frame = melody.frame_beats
    total_frames = len(melody.chroma)
    i = min(total_frames, max(0, int(round(float(start) / frame))))
    j = min(total_frames, max(0, int(round(float(end) / frame))))
    if j <= i:
        j = min(total_frames, i + 1)
    prefix = melody.prefix
    strong = melody.strong_prefix
    step = max(1, int(round(1.0 / frame)))

    chroma = _normalise((prefix[j] - prefix[i])[index])
    strong_chroma = _normalise((strong[j] - strong[i])[index])
    onset = _normalise((prefix[min(j, i + step)] - prefix[i])[index])
    tail = _normalise((prefix[j] - prefix[max(i, j - step)])[index])

    meter = max(melody.beats_per_bar, 1.0)
    duration = np.log2(max(1.0, float(end) - float(start))) / 5.0
    phase = ((float(start) % meter) / meter) * 2.0 * np.pi
    mode_value = 1.0 if (isinstance(mode, int) and mode == 1) or str(mode).startswith("min") else 0.0

    return np.concatenate([
        chroma,
        strong_chroma,
        onset,
        tail,
        np.asarray([duration, np.sin(phase), np.cos(phase), mode_value]),
    ]).astype(np.float64)


@dataclass
class ChordClassifier:
    """Small numpy-only classifier.

    ``weights`` is ``(n_classes, N_FEATURES)`` and ``classes`` holds the core
    state index each row predicts.  Missing states get a strongly negative
    log-probability rather than being impossible, so a model trained before a
    rare triad appeared can still emit it if the transition and context insist.
    """

    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    classes: np.ndarray
    log_prior: np.ndarray | None = None
    n_states: int = 60

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)
        self.classes = np.asarray(self.classes, dtype=np.int64)
        if self.log_prior is not None:
            self.log_prior = np.asarray(self.log_prior, dtype=np.float64)

    def predict_log_proba(
        self, features: np.ndarray, *, prior_correction: float = 0.0
    ) -> np.ndarray:
        """Class log-probabilities, optionally removing the learned prior.

        In tonic-relative space the tonic triad is 33% of all windows, so the
        classifier learns a large prior for it.  The transition model already
        supplies a prior; subtracting part of the classifier's own prior
        (``prior_correction`` around 0.3) stops the two from double-counting
        and measurably improves top-1 and top-3 on held-out songs.
        """
        single = features.ndim == 1
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if matrix.shape[1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} features, got {matrix.shape[1]}")
        standardised = (matrix - self.mean) / np.maximum(self.scale, 1e-9)
        logits = standardised @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
        out = np.full((matrix.shape[0], self.n_states), -20.0, dtype=np.float64)
        out[:, self.classes] = log_probs
        if prior_correction and self.log_prior is not None:
            # Flatten the classifier's learned class prior, but renormalise
            # afterwards.  Subtracting alpha*log(prior) without this step does
            # not just reweight states: rare classes gain positive log-mass,
            # so the model pays *less* for adding segments and prefers keys
            # whose chord vocabulary is rare.  On OpenEWLD that made every
            # song decode in a minor key.  Renormalising keeps every row a
            # proper distribution and leaves the segment-count trade-off to
            # the duration model, where it belongs.
            adjusted = out - float(prior_correction) * self.log_prior[None, :]
            adjusted -= _logsumexp(adjusted, axis=1, keepdims=True)
            out = adjusted
        return out[0] if single else out

    def log_proba(self, features: np.ndarray, *, prior_correction: float = 0.0) -> np.ndarray:
        return self.predict_log_proba(features, prior_correction=prior_correction)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "clf_weights": self.weights,
            "clf_bias": self.bias,
            "clf_mean": self.mean,
            "clf_scale": self.scale,
            "clf_classes": self.classes,
            "clf_log_prior": self.log_prior if self.log_prior is not None else np.zeros(60),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "ChordClassifier":
        return cls(
            weights=arrays["clf_weights"],
            bias=arrays["clf_bias"],
            mean=arrays["clf_mean"],
            scale=arrays["clf_scale"],
            classes=arrays["clf_classes"],
            log_prior=arrays.get("clf_log_prior"),
        )


def build_dataset(
    songs: Sequence[Song],
    *,
    min_duration: float = 1.0,
    window_lengths: Sequence[int] = (1, 2, 4),
    window_step: int = 1,
    include_spans: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Training rows: annotated spans plus beat-aligned windows.

    The decoder scores spans of one to four beats at every beat boundary, so a
    classifier trained only on the corpus's annotated spans sees a different
    distribution at inference time.  Adding every aligned window (labelled by
    the chord covering at least 60% of it) cuts the mismatch; on OpenEWLD it
    raises held-out window top-1 from 34% to 42%.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for song in songs:
        # Features and labels must live in the same coordinate system.  The
        # chords are transposed to the tonic, so the melody has to be too;
        # using the absolute melody here silently teaches the classifier that
        # every key looks like C, which collapses it to the most common key.
        relative = song.in_tonic_relative()
        melody = Melody.from_notes(relative.notes, beats_per_bar=relative.beats_per_bar)
        if include_spans:
            for span in relative.chords:
                if span.duration < min_duration:
                    continue
                rows.append(extract_features(
                    melody,
                    span.onset,
                    span.onset + span.duration,
                    mode=relative.key.mode,
                ))
                labels.append(core_index(span.chord))

        total = int(np.floor(melody.total_beats))
        for start in range(0, total, max(1, int(window_step))):
            for length in window_lengths:
                end = start + length
                if end > total:
                    continue
                best = None
                overlap = 0.0
                for span in relative.chords:
                    shared = min(end, span.end) - max(start, span.onset)
                    if shared > overlap:
                        best, overlap = span.chord, shared
                if best is None or overlap < 0.6 * length:
                    continue
                rows.append(extract_features(
                    melody, start, end, mode=relative.key.mode
                ))
                labels.append(core_index(best))
    if not rows:
        raise ValueError("no usable chord spans")
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


def train_classifier(
    songs: Sequence[Song],
    *,
    C: float = 0.2,
    max_iter: int = 300,
    seed: int = 0,
) -> tuple[ChordClassifier, dict]:
    """Fit the logistic regression and export it to numpy arrays."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - training-only dependency
        raise RuntimeError(
            "training the chord classifier needs scikit-learn "
            "(pip install scikit-learn); inference does not"
        ) from exc

    features, labels = build_dataset(songs)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    model = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter),
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(scaled, labels)
    counts = np.bincount(labels, minlength=60).astype(np.float64)
    log_prior = np.log((counts + 1.0) / (counts.sum() + 60.0))
    classifier = ChordClassifier(
        weights=model.coef_,
        bias=model.intercept_,
        mean=scaler.mean_,
        scale=scaler.scale_,
        classes=model.classes_,
        log_prior=log_prior,
    )
    probabilities = classifier.predict_log_proba(features)
    top = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    top1 = float(np.mean(top[:, 0] == labels))
    top3 = float(np.mean(np.any(top == labels[:, None], axis=1)))
    report = {
        "samples": int(len(labels)),
        "features": int(features.shape[1]),
        "classes": int(len(model.classes_)),
        "train_top1": round(top1, 4),
        "train_top3": round(top3, 4),
        "C": float(C),
    }
    return classifier, report

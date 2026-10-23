"""Offline lab for the MySong chord classifier.

Compares feature sets and model families on the held-out dev split.  Nothing
here is imported by the package; it exists so model choices are measured
rather than guessed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np

from mysong.chord_model import FEATURE_NAMES, extract_features
from mysong.corpus import Song, load_corpus, split_corpus
from mysong.melody import Melody
from mysong.theory import CORE_QUALITIES, core_index


def _norm(v):
    s = float(v.sum())
    return v / s if s > 0 else np.zeros(12)


def base_features(melody: Melody, start: float, end: float, mode: str, rotation: int = 0) -> np.ndarray:
    index = [(pc + int(rotation)) % 12 for pc in range(12)]
    frame = melody.frame_beats
    n = len(melody.chroma)
    i = min(n, max(0, int(round(start / frame))))
    j = min(n, max(0, int(round(end / frame))))
    if j <= i:
        j = min(n, i + 1)
    prefix = melody.prefix
    step = max(1, int(round(1.0 / frame)))
    chroma = _norm((prefix[j] - prefix[i])[index])
    strong = _norm((melody.strong_prefix[j] - melody.strong_prefix[i])[index])
    onset = _norm((prefix[min(j, i + step)] - prefix[i])[index])
    tail = _norm((prefix[j] - prefix[max(i, j - step)])[index])
    meter = max(melody.beats_per_bar, 1.0)
    duration = np.log2(max(1.0, end - start)) / 5.0
    phase = ((start % meter) / meter) * 2 * np.pi
    mode_value = 1.0 if str(mode).startswith("min") else 0.0
    return np.concatenate([chroma, strong, onset, tail, [duration, np.sin(phase), np.cos(phase), mode_value]])


def extra_features(melody: Melody, start: float, end: float, mode: str, rotation: int = 0, variant: str = "v1") -> np.ndarray:
    base = base_features(melody, start, end, mode, rotation)
    if variant == "base":
        return base
    index = [(pc + int(rotation)) % 12 for pc in range(12)]
    frame = melody.frame_beats
    n = len(melody.chroma)
    i = min(n, max(0, int(round(start / frame))))
    j = min(n, max(0, int(round(end / frame))))
    if j <= i:
        j = min(n, i + 1)
    step = max(1, int(round(1.0 / frame)))
    prefix = melody.prefix
    meter = max(melody.beats_per_bar, 1.0)
    prev = _norm((prefix[i] - prefix[max(0, i - step)])[index]) if i > 0 else np.zeros(12)
    nxt = _norm((prefix[min(n, j + step)] - prefix[j])[index]) if j < n else np.zeros(12)
    total = prefix[j] - prefix[i]
    raw = total[index]
    presence = (raw > 1e-9).astype(float)
    entropy = 0.0
    s = raw.sum()
    if s > 0:
        p = raw / s
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    # chord-tone cosine against all 60 core triads in tonic-relative space
    templates = np.zeros((60, 12))
    for root in range(12):
        for qi, q in enumerate(CORE_QUALITIES):
            intervals = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
                         "aug": (0, 4, 8), "sus": (0, 5, 7)}[q]
            for iv in intervals:
                templates[root * 5 + qi, (root + iv) % 12] = 1.0
    c = _norm(raw)
    tn = np.linalg.norm(templates, axis=1)
    cos = templates @ c / np.maximum(tn * np.linalg.norm(c), 1e-9)
    # scale-degree indicator (relative to tonic; chroma is already rotated)
    degrees = np.zeros(12)
    scale = (0, 2, 3, 5, 7, 8, 10) if str(mode).startswith("min") else (0, 2, 4, 5, 7, 9, 11)
    for d in scale:
        degrees[d] = 1.0
    phase2 = ((start % meter) / meter) * 4 * np.pi
    bar_end = (end % meter)
    extras = [prev, nxt, presence, [entropy, len(np.nonzero(presence)[0]) / 12.0,
                                   np.sin(phase2), np.cos(phase2),
                                   1.0 if abs(start % meter) < 1e-6 else 0.0,
                                   1.0 if abs(end % meter) < 1e-6 else 0.0,
                                   np.log2(max(2.0, end - start)) / 5.0,
                                   float(c @ degrees) / max(np.linalg.norm(degrees), 1e-9),
                                   float((raw * np.arange(12)).sum() / max(s, 1e-9)) / 11.0]]
    out = np.concatenate([base, *extras])
    if variant == "templates":
        out = np.concatenate([out, cos])
    return out


def rows_for_song(song: Song, variant: str, windows=(1, 2, 3, 4, 6, 8), spans=True, fully_inside=False):
    rel = song.in_tonic_relative()
    melody = Melody.from_notes(rel.notes, beats_per_bar=rel.beats_per_bar)
    rows, labels, kinds = [], [], []
    if spans:
        for sp in rel.chords:
            if sp.duration < 1.0:
                continue
            rows.append(extra_features(melody, sp.onset, sp.onset + sp.duration, rel.key.mode, 0, variant))
            labels.append(core_index(sp.chord))
            kinds.append("span")
    total = int(np.floor(melody.total_beats))
    for start in range(total):
        for length in windows:
            end = start + length
            if end > total:
                continue
            best, overlap = None, 0.0
            inside = False
            for sp in rel.chords:
                shared = min(end, sp.end) - max(start, sp.onset)
                if shared > overlap:
                    best, overlap = sp.chord, shared
                if sp.onset <= start and sp.end >= end:
                    inside = True
            if best is None or overlap < (length if fully_inside else 0.6 * length):
                continue
            rows.append(extra_features(melody, start, end, rel.key.mode, 0, variant))
            labels.append(core_index(best))
            kinds.append("window")
    return np.vstack(rows), np.asarray(labels), np.asarray(kinds)


def dataset(songs, variant, **kw):
    parts = [rows_for_song(s, variant, **kw) for s in songs]
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def scores(logp, labels):
    top = np.argsort(logp, axis=1)[:, ::-1][:, :5]
    top1 = float(np.mean(top[:, 0] == labels))
    top3 = float(np.mean(np.any(top[:, :3] == labels[:, None], axis=1)))
    top5 = float(np.mean(np.any(top[:, :5] == labels[:, None], axis=1)))
    ranks = np.argmax(top == labels[:, None], axis=1)
    mrr = float(np.mean(1.0 / (ranks + 1)))
    return dict(top1=round(top1, 4), top3=round(top3, 4), top5=round(top5, 4), mrr=round(mrr, 4))


def run(Xtr, ytr, Xdev, ydev, kinds, label=""):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    scaler = StandardScaler().fit(Xtr)
    a, b = scaler.transform(Xtr), scaler.transform(Xdev)
    out = {}
    for name, model in [
        ("lr0.2", LogisticRegression(C=0.2, max_iter=400, solver="lbfgs")),
        ("lr0.05", LogisticRegression(C=0.05, max_iter=400, solver="lbfgs")),
        ("lr1", LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")),
        ("mlp128", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=600,
                                 early_stopping=True, n_iter_no_change=30, random_state=0)),
        ("mlp256x2", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=3e-3, max_iter=600,
                                   early_stopping=True, n_iter_no_change=30, random_state=0)),
        ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                               max_leaf_nodes=31, l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.12,
                                               random_state=0)),
    ]:
        t = time.time()
        model.fit(a, ytr)
        p = model.predict_proba(b)
        s = scores(np.log(np.maximum(p, 1e-12)), ydev)
        for kind in ("span", "window"):
            mask = kinds == kind
            if mask.any():
                s2 = scores(np.log(np.maximum(p[mask], 1e-12)), ydev[mask])
                s[f"{kind}_top1"] = s2["top1"]
        s["secs"] = round(time.time() - t, 1)
        out[name] = s
        print(label, name, s, flush=True)
    return out


if __name__ == "__main__":
    songs = load_corpus()
    splits = split_corpus(songs)
    train, dev = splits["train"], splits["dev"]
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    kw = {}
    if len(sys.argv) > 2 and sys.argv[2] == "clean":
        kw = dict(fully_inside=True)
    t = time.time()
    Xtr, ytr, ktr = dataset(train, variant, **kw)
    Xdev, ydev, kdev = dataset(dev, variant, **kw)
    print("features", Xtr.shape, "dev", Xdev.shape, "rows", dict(zip(*np.unique(kdev, return_counts=True))),
          "build", round(time.time() - t, 1), "s", flush=True)
    run(Xtr, ytr, Xdev, ydev, kdev, label=variant)

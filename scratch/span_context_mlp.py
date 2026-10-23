"""Prototype: span MLP on frozen tagger context plus span chroma features."""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mysong.chord_model import extract_features
from mysong.corpus import load_corpus, split_corpus
from mysong.melody import Melody
from mysong.model import MySongModel
from mysong.neural import tagger_features
from mysong.theory import core_index


def label_for(rel, start, end):
    best = None
    best_overlap = 0.0
    for span in rel.chords:
        overlap = min(end, span.end) - max(start, span.onset)
        if overlap > best_overlap:
            best_overlap = overlap
            best = span.chord
    if best is None or best_overlap < 0.5 * (end - start):
        return None
    return core_index(best)


def song_examples(song, encoder):
    rel = song.in_tonic_relative()
    melody = Melody.from_notes(rel.notes, beats_per_bar=rel.beats_per_bar)
    beats = int(math.floor(melody.total_beats))
    if beats < 4:
        return None
    x = tagger_features(melody, 0, rel.key.mode)[:beats]
    with torch.no_grad():
        h = encoder(torch.from_numpy(x)[None])[0].numpy()
    examples = []
    seen = set()
    for span in rel.chords:
        start = max(0, int(round(span.onset)))
        end = min(beats, int(round(span.end)))
        if end <= start:
            continue
        label = core_index(span.chord)
        if (start, end, label) not in seen:
            seen.add((start, end, label))
            examples.append((start, end, label, 1.0))
    for length in (1, 2, 3, 4, 6, 8):
        for start in range(0, beats - length + 1):
            label = label_for(rel, start, start + length)
            if label is None:
                continue
            if (start, start + length, label) not in seen:
                seen.add((start, start + length, label))
                examples.append((start, start + length, label, 0.35))
    if not examples:
        return None
    rows = []
    for s, e, _, _ in examples:
        piece = h[s:e]
        chroma = extract_features(melody, s, e, mode=rel.key.mode, rotation=0)
        rows.append(np.concatenate([
            piece.mean(0), piece.max(0), h[s], h[e - 1],
            np.asarray([(e - s) / 8.0,
                        math.sin(2 * math.pi * (s % 4) / 4),
                        math.cos(2 * math.pi * (s % 4) / 4)]),
            chroma,
        ]))
    return (np.stack(rows).astype(np.float32),
            np.asarray([label for _, _, label, _ in examples], dtype=np.int64),
            np.asarray([weight for _, _, _, weight in examples], dtype=np.float32))


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_in),
            nn.Linear(n_in, 512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 60),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    model = MySongModel.load()
    tagger = model.tagger
    assert tagger is not None
    encoder = tagger.model
    songs = load_corpus()
    split = split_corpus(songs)
    train = [d for d in (song_examples(s, encoder) for s in split["train"]) if d]
    val = [d for d in (song_examples(s, encoder) for s in split["test"][:20]) if d]
    X = np.concatenate([x for x, _, _ in train])
    Y = np.concatenate([y for _, y, _ in train])
    W = np.concatenate([w for _, _, w in train])
    Xv = np.concatenate([x for x, _, _ in val])
    Yv = np.concatenate([y for _, y, _ in val])
    Wv = np.concatenate([w for _, _, w in val])
    print("train", X.shape, "val", Xv.shape, flush=True)
    clf = MLP(X.shape[1])
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=2e-3)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    rng = np.random.default_rng(0)
    for epoch in range(30):
        clf.train()
        order = rng.permutation(len(X))
        losses = []
        t0 = time.time()
        for start in range(0, len(order), 512):
            idx = order[start:start + 512]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(Y[idx])
            wb = torch.from_numpy(W[idx])
            loss = (ce(clf(xb), yb) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.from_numpy(Xv)).argmax(1)
            mask = torch.from_numpy(Wv) >= 1.0
            top1 = float(((pred == torch.from_numpy(Yv)) & mask).sum()) / float(mask.sum())
        print(f"epoch {epoch+1:2d} loss {np.mean(losses):.4f} val-top1 {top1:.3f} {time.time()-t0:.0f}s", flush=True)
        torch.save({"state": clf.state_dict()}, "scratch/span_context_mlp.pt")


if __name__ == "__main__":
    main()

"""Prototype: BiGRU encoder + span scorer, trained on annotated spans and windows.

This is an experiment, not the production trainer.  It answers one question:
does scoring a *span* with surrounding context beat the per-beat tagger and the
52-feature logistic classifier?
"""
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

from mysong.corpus import load_corpus, split_corpus
from mysong.melody import Melody
from mysong.neural import FEATURE_DIM, tagger_features
from mysong.theory import core_index


def label_for(rel, start: float, end: float) -> int | None:
    """Core chord covering the largest part of [start, end), if at least half."""
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


def song_data(song):
    rel = song.in_tonic_relative()
    melody = Melody.from_notes(rel.notes, beats_per_bar=rel.beats_per_bar)
    beats = int(math.floor(melody.total_beats))
    if beats < 4:
        return None
    x = tagger_features(melody, 0, rel.key.mode)
    x = x[:beats]
    # Annotated spans, plus windows on the beat grid.  We keep every window of
    # length 1..8 whose majority chord is unambiguous.  This matches the spans
    # the semimarkov decoder actually asks about.
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
    return x, examples


class SpanModel(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.span = nn.Sequential(
            nn.LayerNorm(hidden * 8 + 4),
            nn.Linear(hidden * 8 + 4, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 60),
        )
        self.beat = nn.Linear(hidden * 2, 60)

    def encode(self, x):
        h, _ = self.gru(self.inp(x))
        return h

    def score(self, h, start, end):
        # Pool the encoder states over the candidate span.  Concat mean, max,
        # start and end; the decoder's span lengths are short, so this is cheap.
        piece = h[start:end]
        mean = piece.mean(dim=0)
        mx = piece.max(dim=0).values
        first = h[start]
        last = h[end - 1]
        length = torch.tensor([(end - start) / 8.0], device=h.device)
        phase = torch.tensor([math.sin(2 * math.pi * (start % 4) / 4.0),
                              math.cos(2 * math.pi * (start % 4) / 4.0)], device=h.device)
        mode = h.new_zeros(1)
        return self.span(torch.cat([mean, mx, first, last, length, phase, mode]))


def batches(data, batch_size, shuffle, rng):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for offset in range(0, len(order), batch_size):
        yield [data[i] for i in order[offset:offset + batch_size]]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    songs = load_corpus()
    split = split_corpus(songs)
    train_songs = split["train"]
    val_songs = split["test"][:20]
    train = [d for d in (song_data(s) for s in train_songs) if d]
    val = [d for d in (song_data(s) for s in val_songs) if d]
    print(f"train songs {len(train)} val songs {len(val)}", flush=True)
    model = SpanModel()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = random.Random(0)

    # Precompute encoder states once per epoch is not possible because weights
    # change; instead train song by song, accumulating examples.
    for epoch in range(12):
        model.train()
        losses = []
        t0 = time.time()
        order = list(range(len(train)))
        rng.shuffle(order)
        for idx in order:
            x, examples = train[idx]
            h = model.encode(torch.from_numpy(x)[None])[0]
            # beat auxiliary loss
            # labels are constructed here from example spans is expensive; skip
            # the auxiliary head for now and use only span CE.
            sample = examples
            rng.shuffle(sample)
            for chunk_start in range(0, len(sample), 64):
                chunk = sample[chunk_start:chunk_start + 64]
                logits = torch.stack([model.score(h, s, e) for s, e, _, _ in chunk])
                y = torch.tensor([label for _, _, label, _ in chunk], dtype=torch.long)
                w = torch.tensor([weight for _, _, _, weight in chunk], dtype=torch.float32)
                loss = (ce(logits, y) * w).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, examples in val:
                h = model.encode(torch.from_numpy(x)[None])[0]
                for s, e, label, weight in examples:
                    if weight < 1.0:
                        continue
                    logits = model.score(h, s, e)
                    correct += int(int(logits.argmax()) == label)
                    total += 1
        print(f"epoch {epoch+1:2d} loss {np.mean(losses):.4f} "
              f"span-top1 {correct/max(total,1):.3f} time {time.time()-t0:.0f}s", flush=True)
        torch.save({"state": model.state_dict()}, "scratch/span_proto.pt")


if __name__ == "__main__":
    main()

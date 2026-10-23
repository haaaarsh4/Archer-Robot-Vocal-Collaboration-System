"""Beat-level bidirectional GRU chord tagger, prototype.

Compares a sequence model (which sees the whole melodic phrase) against the
per-span logistic classifier.  Run from the project root with PYTHONPATH=.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from mysong.corpus import Song, load_corpus, split_corpus
from mysong.melody import Melody
from mysong.theory import core_index

torch.manual_seed(0)
torch.set_num_threads(4)


def song_sequence(song: Song):
    rel = song.in_tonic_relative()
    mel = Melody.from_notes(rel.notes, beats_per_bar=rel.beats_per_bar)
    n = int(np.floor(mel.total_beats))
    if n < 2:
        return None
    frame = mel.frame_beats
    per = max(1, int(round(1.0 / frame)))
    rows, labels = [], []
    mode = 1.0 if rel.key.mode == "minor" else 0.0
    for beat in range(n):
        i, j = beat * per, (beat + 1) * per
        chroma = mel.prefix[j] - mel.prefix[i]
        total = chroma.sum()
        if total > 0:
            chroma = chroma / total
        strong = mel.strong_prefix[j] - mel.strong_prefix[i]
        st = strong.sum()
        if st > 0:
            strong = strong / st
        onset = mel.prefix[min(j, i + per)] - mel.prefix[i]
        ot = onset.sum()
        if ot > 0:
            onset = onset / ot
        phase = (beat % rel.beats_per_bar) / rel.beats_per_bar * 2 * np.pi
        rows.append(np.concatenate([chroma, strong, onset,
                                    [np.sin(phase), np.cos(phase), mode, total > 0]]))
        midpoint = beat + 0.5
        label = -1
        for span in rel.chords:
            if span.onset <= midpoint < span.end:
                label = core_index(span.chord)
                break
        labels.append(label)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    mask = y >= 0
    return x[mask], y[mask]


class Tagger(nn.Module):
    def __init__(self, n_in=40, hidden=128, layers=2, n_out=60):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.15 if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, n_out))

    def forward(self, x):
        h = self.inp(x)
        h, _ = self.gru(h)
        return self.out(h)


def batches(seqs, bs, shuffle=True):
    order = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(order), bs):
        chunk = [seqs[k] for k in order[i:i + bs]]
        length = max(len(x) for x, _ in chunk)
        x = np.zeros((len(chunk), length, chunk[0][0].shape[1]), dtype=np.float32)
        y = np.full((len(chunk), length), -100, dtype=np.int64)
        for r, (sx, sy) in enumerate(chunk):
            x[r, :len(sx)] = sx
            y[r, :len(sy)] = sy
        yield torch.from_numpy(x), torch.from_numpy(y)


def evaluate(model, seqs):
    model.eval()
    correct = total = root_ok = 0
    with torch.no_grad():
        for x, y in batches(seqs, 16, shuffle=False):
            logits = model(x)
            pred = logits.argmax(-1)
            valid = y >= 0
            correct += int(((pred == y) & valid).sum())
            total += int(valid.sum())
            root_ok += int(((pred // 5 == y // 5) & valid).sum())
    return root_ok / max(total, 1), correct / max(total, 1)


if __name__ == "__main__":
    songs = load_corpus()
    splits = split_corpus(songs)
    print("building", flush=True)
    train = [s for s in (song_sequence(x) for x in splits["train"]) if s]
    dev = [s for s in (song_sequence(x) for x in splits["dev"]) if s]
    print("songs", len(train), len(dev), "beats", sum(len(y) for _, y in train), flush=True)
    model = Tagger()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0
    for epoch in range(40):
        model.train()
        t = time.time()
        losses = []
        for x, y in batches(train, 24):
            opt.zero_grad()
            logits = model(x)
            loss = lossf(logits.reshape(-1, 60), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        root, acc = evaluate(model, dev)
        print(f"epoch {epoch} loss {np.mean(losses):.3f} dev root {root:.4f} acc {acc:.4f} {time.time()-t:.1f}s", flush=True)
        if root > best:
            best = root
            torch.save(model.state_dict(), "scratch/gru_best.pt")
    print("best", best)

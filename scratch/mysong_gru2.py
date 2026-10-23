"""Improved beat-level tagger experiments.

Adds bar/global context and chord-template features to the per-beat vector,
class-balanced loss, and light augmentation.  Caches the prepared sequences so
architecture changes do not rebuild features every run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mysong.corpus import Song, load_corpus, split_corpus
from mysong.melody import Melody
from mysong.theory import CORE_QUALITIES, core_index

torch.manual_seed(0)
torch.set_num_threads(4)

TEMPLATES = np.zeros((60, 12), dtype=np.float32)
for _root in range(12):
    for _qi, _q in enumerate(CORE_QUALITIES):
        for _iv in {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
                    "aug": (0, 4, 8), "sus": (0, 5, 7)}[_q]:
            TEMPLATES[_root * 5 + _qi, (_root + _iv) % 12] = 1.0
TEMPLATE_NORMS = np.linalg.norm(TEMPLATES, axis=1)


def sequence(song: Song, rich: bool = True):
    rel = song.in_tonic_relative()
    mel = Melody.from_notes(rel.notes, beats_per_bar=rel.beats_per_bar)
    n = int(np.floor(mel.total_beats))
    if n < 2:
        return None
    per = max(1, int(round(1.0 / mel.frame_beats)))
    mode = 1.0 if rel.key.mode == "minor" else 0.0
    glob = mel.global_chroma()
    glob = glob / glob.sum() if glob.sum() > 0 else glob
    rows, labels = [], []
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
        parts = [chroma, strong, onset, [np.sin(phase), np.cos(phase), mode, total > 0]]
        if rich:
            bar = beat - (beat % int(round(rel.beats_per_bar)))
            bc = mel.slice_chroma(bar, bar + rel.beats_per_bar)
            bs = bc.sum()
            parts.append(bc / bs if bs > 0 else bc)
            parts.append(glob)
            # two-beat template similarity (more stable than one beat)
            lo = max(0, beat - 1)
            hi = min(n, beat + 2)
            w = mel.slice_chroma(lo, hi)
            ws = w.sum()
            if ws > 0:
                w = w / ws
            cos = TEMPLATES @ w / np.maximum(TEMPLATE_NORMS * np.linalg.norm(w), 1e-9)
            parts.append(cos)
        rows.append(np.concatenate(parts))
        mid = beat + 0.5
        label = -1
        for span in rel.chords:
            if span.onset <= mid < span.end:
                label = core_index(span.chord)
                break
        labels.append(label)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    mask = y >= 0
    return x[mask], y[mask]


class Tagger(nn.Module):
    def __init__(self, n_in, hidden=160, layers=2, n_out=60, dropout=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
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
            pred = model(x).argmax(-1)
            valid = y >= 0
            correct += int(((pred == y) & valid).sum())
            total += int(valid.sum())
            root_ok += int(((pred // 5 == y // 5) & valid).sum())
    return root_ok / max(total, 1), correct / max(total, 1)


def augment(x):
    if np.random.random() < 0.5:
        return x
    x = x.copy()
    # drop a pitch class or two, as a singer might under-articulate one
    for _ in range(np.random.randint(1, 3)):
        pc = np.random.randint(0, 12)
        x[..., pc] *= np.random.uniform(0.0, 0.4)
    x[..., :12] += np.random.normal(0, 0.008, size=x.shape[:2] + (12,)).astype(np.float32)
    return x


if __name__ == "__main__":
    rich = "--rich" in sys.argv
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 25
    cache = Path(f"scratch/gru2_{'rich' if rich else 'plain'}.npz")
    songs = load_corpus()
    splits = split_corpus(songs)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        train, dev = list(d["train"]), list(d["dev"])
    else:
        print("building", flush=True)
        train = [s for s in (sequence(x, rich) for x in splits["train"]) if s]
        dev = [s for s in (sequence(x, rich) for x in splits["dev"]) if s]
        np.savez_compressed(cache, train=np.asarray(train, dtype=object), dev=np.asarray(dev, dtype=object))
    n_in = train[0][0].shape[1]
    counts = np.bincount(np.concatenate([y for _, y in train]), minlength=60).astype(float)
    weights = (counts.sum() / np.maximum(counts, 1)) ** 0.35
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    model = Tagger(n_in, hidden=160, layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        t = time.time()
        losses = []
        for x, y in batches(train, 24):
            x = torch.from_numpy(augment(x.numpy()))
            opt.zero_grad()
            loss = lossf(model(x).reshape(-1, 60), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        root, acc = evaluate(model, dev)
        print(f"epoch {epoch} loss {np.mean(losses):.3f} dev root {root:.4f} acc {acc:.4f} {time.time()-t:.1f}s", flush=True)
        if root > best:
            best = root
            torch.save({"state": model.state_dict(), "n_in": n_in, "hidden": 160, "layers": 2},
                       "scratch/gru2_best.pt")
    print("best", best)

"""A beat-level bidirectional GRU chord tagger.

The per-span logistic classifier in :mod:`mysong.chord_model` sees only the
candidate chord span.  That is enough to rank a triad, but not to resolve the
places where a melody is genuinely ambiguous without the phrase around it --
a passing tone, a cadence, a repeated motive.  A small bidirectional GRU over
the whole song sees both sides of every beat and makes those decisions with
context.  On held-out OpenEWLD it raises root accuracy from about 0.57 to 0.64
with the same decoder.

The model is deliberately small (two 128-unit layers, roughly 0.6 MB) and is
trained from scratch on the corpus.  Training uses a validation split carved
out of the training songs, never the dev or test split.  Inference needs
PyTorch; if it is unavailable the package falls back to the numpy classifier,
so the app still runs on a bare install.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mysong.corpus import Song
from mysong.melody import Melody
from mysong.theory import Key, core_index

try:  # pragma: no cover - exercised by presence/absence in the environment
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False

FEATURE_DIM = 40
_ModuleBase = nn.Module if TORCH_AVAILABLE else object


class BeatTagger(_ModuleBase):  # type: ignore[misc]
    """Two-layer BiGRU over per-beat melody features."""

    def __init__(self, n_in: int = FEATURE_DIM, hidden: int = 128, layers: int = 2,
                 n_out: int = 60, dropout: float = 0.15) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, n_out))

    def forward(self, x):  # pragma: no cover - thin torch wrapper
        hidden, _ = self.gru(self.inp(x))
        return self.out(hidden)


def _per_beat(melody: Melody) -> int:
    return max(1, int(round(1.0 / melody.frame_beats)))


def tagger_features(melody: Melody, tonic: int = 0, mode: str = "major") -> np.ndarray:
    """Per-beat features in tonic-relative coordinates, shape ``(beats, 40)``."""
    per = _per_beat(melody)
    beats = int(np.floor(melody.total_beats))
    if beats <= 0:
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)
    index = [(pc + int(tonic)) % 12 for pc in range(12)]
    minor = 1.0 if str(mode).startswith("min") else 0.0
    rows = np.zeros((beats, FEATURE_DIM), dtype=np.float32)
    for beat in range(beats):
        i, j = beat * per, (beat + 1) * per
        chroma = melody.prefix[j] - melody.prefix[i]
        total = float(chroma.sum())
        if total > 0:
            chroma = chroma / total
        strong = melody.strong_prefix[j] - melody.strong_prefix[i]
        st = float(strong.sum())
        if st > 0:
            strong = strong / st
        onset = melody.prefix[min(j, i + per)] - melody.prefix[i]
        ot = float(onset.sum())
        if ot > 0:
            onset = onset / ot
        phase = (beat % melody.beats_per_bar) / melody.beats_per_bar * 2 * np.pi
        rows[beat, :12] = chroma[index]
        rows[beat, 12:24] = strong[index]
        rows[beat, 24:36] = onset[index]
        rows[beat, 36] = np.sin(phase)
        rows[beat, 37] = np.cos(phase)
        rows[beat, 38] = minor
        rows[beat, 39] = 1.0 if total > 0 else 0.0
    return rows


def song_sequence(song: Song) -> tuple[np.ndarray, np.ndarray] | None:
    """Training sequence and per-beat core-state labels for one lead sheet."""
    relative = song.in_tonic_relative()
    melody = Melody.from_notes(relative.notes, beats_per_bar=relative.beats_per_bar)
    beats = int(np.floor(melody.total_beats))
    if beats < 2:
        return None
    x = tagger_features(melody, 0, relative.key.mode)
    y = np.full(beats, -1, dtype=np.int64)
    for span in relative.chords:
        start = max(0, int(np.floor(span.onset)))
        end = min(beats, int(np.ceil(span.end)))
        label = core_index(span.chord)
        for beat in range(start, end):
            if span.onset <= beat + 0.5 < span.end:
                y[beat] = label
    mask = y >= 0
    if mask.sum() < 2:
        return None
    return x[mask], y[mask]


def _batches(sequences, batch_size: int, shuffle: bool = True, device=None):
    order = list(range(len(sequences)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [sequences[index] for index in order[start:start + batch_size]]
        length = max(len(x) for x, _ in chunk)
        x = np.zeros((len(chunk), length, FEATURE_DIM), dtype=np.float32)
        y = np.full((len(chunk), length), -100, dtype=np.int64)
        for row, (sx, sy) in enumerate(chunk):
            x[row, :len(sx)] = sx
            y[row, :len(sy)] = sy
        xb = torch.from_numpy(x)
        yb = torch.from_numpy(y)
        if device is not None:
            xb, yb = xb.to(device), yb.to(device)
        yield xb, yb


def _evaluate(model, sequences, device=None) -> tuple[float, float]:
    model.eval()
    root_ok = state_ok = total = 0
    with torch.no_grad():
        for x, y in _batches(sequences, 16, shuffle=False, device=device):
            predicted = model(x).argmax(-1)
            valid = y >= 0
            root_ok += int(((predicted // 5 == y // 5) & valid).sum())
            state_ok += int(((predicted == y) & valid).sum())
            total += int(valid.sum())
    return root_ok / max(total, 1), state_ok / max(total, 1)


@dataclass
class TaggerReport:
    epochs: int = 0
    best_val_root: float = 0.0
    best_val_state: float = 0.0
    train_songs: int = 0
    val_songs: int = 0
    parameters: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def train_tagger(
    songs: Sequence[Song],
    path: str | Path,
    *,
    epochs: int = 25,
    hidden: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 0,
    validation_fraction: float = 0.1,
    device: str | None = None,
    log=print,
) -> TaggerReport:
    """Train the GRU on lead sheets, early-stopping on a held-out subset."""
    sequences = [item for item in (song_sequence(song) for song in songs) if item is not None]
    return train_tagger_sequences(
        sequences,
        path,
        epochs=epochs,
        hidden=hidden,
        learning_rate=learning_rate,
        seed=seed,
        validation_fraction=validation_fraction,
        device=device,
        log=log,
    )


def train_tagger_sequences(
    sequences: Sequence[tuple[np.ndarray, np.ndarray]],
    path: str | Path,
    *,
    epochs: int = 25,
    hidden: int = 128,
    learning_rate: float = 2e-3,
    seed: int = 0,
    validation_fraction: float = 0.1,
    max_sequence_beats: int = 1024,
    device: str | None = None,
    init_path: str | Path | None = None,
    log=print,
) -> TaggerReport:
    """Train from pre-built ``(features, labels)`` sequences.

    This is the memory-bounded entry point for large corpora: callers can
    stream scores, convert each one with :func:`song_sequence`, and keep only
    the compact arrays.  ``device`` accepts ``"cuda"`` on a GPU server; the
    checkpoint is always written as CPU tensors.

    A handful of lead sheets in large archives contain a 20,000-beat repeat or
    a broken measure.  One such score is enough to make a batch allocate a
    multi-gigabyte GRU graph, so training skips sequences above
    ``max_sequence_beats``.  It is one score out of thousands, not a corpus
    filter that changes the distribution.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("training the chord tagger needs PyTorch")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    if max_sequence_beats:
        before = len(sequences)
        sequences = [item for item in sequences if len(item[0]) <= int(max_sequence_beats)]
        if len(sequences) != before:
            log(f"  skipped {before - len(sequences)} sequence(s) over {max_sequence_beats} beats", flush=True)
    if len(sequences) < 8:
        raise ValueError("not enough songs to train the chord tagger")
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    n_val = max(2, int(round(len(order) * validation_fraction)))
    val = [sequences[index] for index in order[:n_val]]
    train = [sequences[index] for index in order[n_val:]]

    model = BeatTagger(hidden=hidden)
    if init_path is not None:
        payload = torch.load(str(init_path), map_location="cpu", weights_only=False)
        if int(payload.get("hidden", hidden)) != int(hidden):
            raise ValueError(
                f"checkpoint hidden={payload.get('hidden')} does not match requested hidden={hidden}"
            )
        model.load_state_dict(payload["state"])
        log(f"  initialised from {init_path}", flush=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_root = -1.0
    best_state = None
    report = TaggerReport(epochs=epochs, train_songs=len(train), val_songs=len(val))
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for x, y in _batches(train, 24, device=device):
            optimizer.zero_grad()
            loss = loss_fn(model(x).reshape(-1, 60), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        root, state = _evaluate(model, val, device=device)
        log(f"  tagger epoch {epoch + 1:2d}/{epochs} loss {np.mean(losses):.3f} "
            f"val root {root:.3f} state {state:.3f}", flush=True)
        if root > best_root:
            best_root = root
            report.best_val_root = root
            report.best_val_state = state
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": best_state,
        "n_in": FEATURE_DIM,
        "hidden": int(hidden),
        "layers": 2,
        "report": report.to_dict(),
    }, path)
    # A numpy checkpoint keeps the weights reviewable and avoids a pickle load
    # at inference.  The .pt file is still written for people who prefer it.
    Tagger(model, hidden=int(hidden), layers=2).save_npz(path.with_suffix(".npz"))
    report.parameters = int(sum(p.numel() for p in model.parameters()))
    return report


class TaggerEnsemble:
    """Average the log-probabilities of independently trained taggers.

    Two taggers with different initializations make different mistakes.  A
    simple arithmetic mean of their log-probabilities is a cheap, deterministic
    ensemble and needs no extra training.  ``MySongModel.load`` builds one when
    a sibling ``<model>_tagger2.npz`` exists.
    """

    def __init__(self, taggers: Sequence["Tagger"], weights: Sequence[float] | None = None) -> None:
        if not taggers:
            raise ValueError("an ensemble needs at least one tagger")
        self.taggers = list(taggers)
        values = np.asarray(weights if weights is not None else [1.0] * len(taggers), dtype=np.float64)
        self.weights = values / max(values.sum(), 1e-9)

    def clear(self) -> None:
        for tagger in self.taggers:
            tagger.clear()

    def log_proba(self, melody: Melody, key: Key) -> np.ndarray:
        stack = np.stack([tagger.log_proba(melody, key) for tagger in self.taggers])
        average = np.tensordot(self.weights, stack, axes=(0, 0))
        average -= average.max(axis=-1, keepdims=True)
        return average - np.log(np.exp(average).sum(axis=-1, keepdims=True) + 1e-300)

    def log_span(self, melody: Melody, key: Key, start: float, end: float) -> np.ndarray:
        log_probs = self.log_proba(melody, key)
        first = max(0, int(np.floor(start)))
        last = min(len(log_probs), int(np.ceil(end)))
        if last <= first:
            return np.zeros(60, dtype=np.float64)
        return log_probs[first:last].sum(axis=0)


class Tagger:
    """Inference wrapper: per-beat chord log-probabilities for a melody and key."""

    def __init__(self, model: "BeatTagger", *, hidden: int = 128, layers: int = 2) -> None:
        self.model = model.eval()
        self.hidden = hidden
        self.layers = layers
        self._cache: dict[tuple[int, int, str], np.ndarray] = {}

    @classmethod
    def load(cls, path: str | Path) -> "Tagger":
        if not TORCH_AVAILABLE:
            raise RuntimeError("loading the chord tagger needs PyTorch")
        path = Path(path)
        if path.suffix == ".npz":
            return cls.load_npz(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        model = BeatTagger(
            n_in=int(payload.get("n_in", FEATURE_DIM)),
            hidden=int(payload.get("hidden", 128)),
            layers=int(payload.get("layers", 2)),
        )
        model.load_state_dict(payload["state"])
        return cls(model, hidden=int(payload.get("hidden", 128)),
                   layers=int(payload.get("layers", 2)))

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value.detach().cpu().numpy() for name, value in self.model.state_dict().items()}
        arrays["meta:n_in"] = np.asarray(FEATURE_DIM)
        arrays["meta:hidden"] = np.asarray(self.hidden)
        arrays["meta:layers"] = np.asarray(self.layers)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "Tagger":
        if not TORCH_AVAILABLE:
            raise RuntimeError("loading the chord tagger needs PyTorch")
        with np.load(path, allow_pickle=False) as data:
            n_in = int(data["meta:n_in"])
            hidden = int(data["meta:hidden"])
            layers = int(data["meta:layers"])
            state = {
                key: torch.from_numpy(data[key])
                for key in data.files if not key.startswith("meta:")
            }
        model = BeatTagger(n_in=n_in, hidden=hidden, layers=layers)
        model.load_state_dict(state)
        return cls(model, hidden=hidden, layers=layers)

    def clear(self) -> None:
        self._cache.clear()

    def log_proba(self, melody: Melody, key: Key) -> np.ndarray:
        cache_key = (id(melody), int(key.tonic), key.mode)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        features = tagger_features(melody, key.tonic, key.mode)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(features)[None])[0]
            log_probs = logits.log_softmax(-1).numpy()
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[cache_key] = log_probs
        return log_probs

    def log_span(self, melody: Melody, key: Key, start: float, end: float) -> np.ndarray:
        log_probs = self.log_proba(melody, key)
        first = max(0, int(np.floor(start)))
        last = min(len(log_probs), int(np.ceil(end)))
        if last <= first:
            return np.zeros(60, dtype=np.float64)
        return log_probs[first:last].sum(axis=0)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """State dict as numpy arrays, for callers that want one checkpoint."""
        return {name: value.detach().cpu().numpy() for name, value in self.model.state_dict().items()}

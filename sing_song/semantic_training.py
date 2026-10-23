"""Fit the 1024-centre semantic codebook on a corpus.

The paper quantizes w2v-BERT features with k-means trained on the corpus.  This
step is independent of the transformer stages and can be run once: iterate
audio, take a bounded random sample of frames, and fit the codebook.  Sampling
is deliberate; a 46k-hour corpus produces billions of frames and k-means only
needs a representative subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig, SemanticConfig
from .semantic import SemanticEncoder, train_kmeans


def _audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)


def training_audio_root(data: DataConfig) -> Path:
    """Return the corpus split used for training, never the held-out test set.

    ``DataConfig.corpus`` points at the corpus root, which for MUSDB contains
    both ``train/`` and ``test/``.  The k-means must not see test audio, or the
    semantic codebook leaks the evaluation split into every later stage.
    """
    root = Path(data.corpus)
    split = root / data.musdb_split
    return split if split.is_dir() else root


def extract_features(
    audio_dir: str | Path,
    encoder: SemanticEncoder,
    *,
    max_frames: int = 2_000_000,
    clip_seconds: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    import librosa

    files = _audio_files(audio_dir)
    if not files:
        raise RuntimeError(f"no audio files under {audio_dir}")
    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    total = 0
    clip_samples = int(round(clip_seconds * 16000))
    for path in files:
        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        if audio.size < clip_samples:
            audio = np.pad(audio, (0, clip_samples - audio.size))
        for start in range(0, audio.size - clip_samples + 1, clip_samples):
            chunk = torch.from_numpy(audio[start:start + clip_samples]).unsqueeze(0)
            emb = encoder.embeddings(chunk, sample_rate=16000)
            feats = emb.reshape(-1, emb.shape[-1]).float().cpu().numpy()
            # Downsample frames before collecting; adjacent 20 ms frames are
            # highly correlated and add little to a 1024-centre codebook.
            take = max(1, feats.shape[0] // 5)
            idx = rng.choice(feats.shape[0], size=take, replace=False)
            collected.append(feats[idx])
            total += take
            if total >= max_frames:
                break
        if total >= max_frames:
            break
    if not collected:
        raise RuntimeError("no semantic features extracted")
    features = np.concatenate(collected, axis=0)[:max_frames]
    return np.ascontiguousarray(features, dtype=np.float32)


def train_semantic_kmeans(
    data: DataConfig,
    semantic_cfg: SemanticConfig,
    *,
    max_frames: int = 2_000_000,
    device: str = "auto",
) -> str:
    device_obj = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = SemanticEncoder(semantic_cfg, device=device_obj).load()
    features = extract_features(
        training_audio_root(data), encoder, max_frames=max_frames
    )
    tokenizer = train_kmeans(
        features, codebook_size=semantic_cfg.codebook_size, use_faiss=True
    )
    tokenizer.save(semantic_cfg.kmeans_path)
    return semantic_cfg.kmeans_path


def main() -> None:
    from .config import load_sing_song_config

    parser = argparse.ArgumentParser(description="Fit the SingSong semantic k-means codebook")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-frames", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_sing_song_config() if args.config is None else load_sing_song_config(
        __import__("yaml").safe_load(open(args.config, encoding="utf-8"))
    )
    path = train_semantic_kmeans(
        cfg.data, cfg.semantic, max_frames=args.max_frames, device=args.device
    )
    print(f"[singsong] semantic k-means written to {path}")


if __name__ == "__main__":
    main()

"""Train the SoundStream codec from scratch.

This is the part Google pre-trained on their corpus before SingSong.  The
objective is the standard SoundStream one: multi-scale mel reconstruction,
residual-VQ commitment/codebook loss, a multi-scale STFT discriminator with a
hinge GAN loss, and feature matching.  It is intentionally independent of the
SingSong transformer stages so the codec can be trained once and reused.

Example (single GPU, 1 s segments)::

    python -m sing_song.codec_training --audio data/singsong/corpus --out checkpoints/singsong/soundstream.pt
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .codec import SoundStream, build_codec, save_codec
from .config import CodecConfig


@dataclass
class CodecTrainConfig:
    steps: int = 200_000
    batch_size: int = 16
    segment_seconds: float = 1.0
    sample_rate: int = 16_000
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.9)
    grad_clip: float = 1.0
    save_every: int = 2000
    log_every: int = 50
    num_workers: int = 4
    device: str = "auto"
    seed: int = 0
    adversarial_weight: float = 1.0
    feature_weight: float = 100.0
    mel_weight: float = 1.0


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int = 1024, hop: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, (3, 9), padding=(1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, (3, 8), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, (3, 8), stride=(2, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, (3, 8), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, (3, 3), padding=(1, 1)),
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spec = torch.stft(
            wav.squeeze(1), n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=torch.hann_window(self.n_fft, device=wav.device),
            return_complex=True, center=True,
        )
        x = torch.view_as_real(spec).permute(0, 3, 1, 2)  # [B, 2, F, T]
        features: list[torch.Tensor] = []
        for layer in self.net[:-1]:
            x = layer(x)
            features.append(x)
        return self.net[-1](x), features


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, fft_sizes: tuple[int, ...], hops: tuple[int, ...]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            STFTDiscriminator(n_fft, hop) for n_fft, hop in zip(fft_sizes, hops)
        )

    def forward(self, wav: torch.Tensor):
        outs = [d(wav) for d in self.discriminators]
        return [o[0] for o in outs], [o[1] for o in outs]


class MultiScaleMelLoss(nn.Module):
    def __init__(self, sample_rate: int, fft_sizes: tuple[int, ...] = (512, 1024, 2048)):
        super().__init__()
        import torchaudio

        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 4,
                n_mels=80, power=1.0, center=True,
            )
            for n_fft in fft_sizes
        ])

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        loss = real.new_zeros(())
        for mel in self.mels:
            r = mel(real.squeeze(1)).clamp_min(1e-5).log()
            f = mel(fake.squeeze(1)).clamp_min(1e-5).log()
            loss = loss + F.l1_loss(f, r)
            loss = loss + torch.linalg.vector_norm(r - f) / (torch.linalg.vector_norm(r) + 1e-7)
        return loss / len(self.mels)


class AudioClips(Dataset):
    """Random one-second crops without decoding a whole track per step.

    The naive implementation loaded and resampled the full song in every
    ``__getitem__``. At 200k steps x batch 16 that is millions of full-track
    decodes, which turns a GPU run into a disk benchmark. This class reads only
    the crop it needs from WAV/FLAC; formats libsndfile cannot seek (MP3, M4A)
    fall back to a full decode, which is correct but slower.
    """

    _FAST_SUFFIXES = {".wav", ".flac", ".ogg"}

    def __init__(
        self,
        audio_dir: str | Path,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.root = Path(audio_dir)
        self.sample_rate = int(sample_rate)
        self.segment_samples = int(round(segment_seconds * self.sample_rate))
        if self.segment_samples <= 0:
            raise ValueError("segment_seconds must be positive")
        self.rng = np.random.default_rng(seed)
        self.files: list[Path] = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        )
        if not self.files:
            raise RuntimeError(f"no audio files under {audio_dir}")
        self._meta: list[tuple[Path, int | None, int | None]] = []
        for path in self.files:
            try:
                info = sf.info(str(path))
                self._meta.append((path, int(info.samplerate), int(info.frames)))
            except Exception:
                # libsndfile refused the container; librosa will handle it.
                self._meta.append((path, None, None))

    def __len__(self) -> int:
        return len(self.files)

    def _read_crop(self, path: Path, src_rate: int, frames: int) -> np.ndarray:
        src_frames = max(1, int(round(self.segment_samples * src_rate / self.sample_rate)))
        start = int(self.rng.integers(0, max(1, frames - src_frames + 1)))
        audio, actual_rate = sf.read(
            str(path), start=start, frames=src_frames,
            dtype="float32", always_2d=True,
        )
        audio = audio.mean(axis=1).astype(np.float32)
        if actual_rate != self.sample_rate:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=actual_rate, target_sr=self.sample_rate,
                res_type="soxr_hq",
            ).astype(np.float32)
        return audio

    def _read_full(self, path: Path) -> np.ndarray:
        import librosa

        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return np.asarray(audio, dtype=np.float32)

    def __getitem__(self, index: int) -> torch.Tensor:
        path, src_rate, frames = self._meta[index % len(self._meta)]
        if src_rate is None or frames is None or path.suffix.lower() not in self._FAST_SUFFIXES:
            audio = self._read_full(path)
            if audio.size <= self.segment_samples:
                audio = np.pad(audio, (0, self.segment_samples - audio.size))
                start = 0
            else:
                start = int(self.rng.integers(0, audio.size - self.segment_samples + 1))
            clip = audio[start:start + self.segment_samples]
        else:
            clip = self._read_crop(path, src_rate, frames)
            if clip.size < self.segment_samples:
                clip = np.pad(clip, (0, self.segment_samples - clip.size))
            else:
                clip = clip[:self.segment_samples]
        return torch.from_numpy(clip).float()


class SoundStreamTrainer:
    def __init__(self, cfg: CodecConfig, train: CodecTrainConfig):
        self.cfg = cfg
        self.train_cfg = train
        self.device = torch.device(
            train.device if train.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator = build_codec(cfg).to(self.device)
        self.discriminator = MultiScaleSTFTDiscriminator(
            cfg.discriminator_fft_sizes, cfg.discriminator_hops
        ).to(self.device)
        self.mel = MultiScaleMelLoss(cfg.sample_rate).to(self.device)
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=train.lr_g, betas=train.betas
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=train.lr_d, betas=train.betas
        )
        self.step = 0

    def _d_loss(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, list]:
        real_logits, real_feats = self.discriminator(real)
        fake_logits, _ = self.discriminator(fake.detach())
        loss = real.new_zeros(())
        for r, f in zip(real_logits, fake_logits):
            loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
        return loss / len(real_logits), real_feats

    def _g_loss(self, real: torch.Tensor, fake: torch.Tensor, real_feats: list) -> torch.Tensor:
        fake_logits, fake_feats = self.discriminator(fake)
        adv = fake.new_zeros(())
        for f in fake_logits:
            adv = adv - f.mean()
        adv = adv / len(fake_logits)

        fm = fake.new_zeros(())
        n = 0
        for rf, ff in zip(real_feats, fake_feats):
            for a, b in zip(rf, ff):
                fm = fm + F.l1_loss(b, a.detach())
                n += 1
        fm = fm / max(1, n)
        return adv, fm

    def train_step(self, real: torch.Tensor) -> dict:
        self.generator.train()
        self.discriminator.train()
        real = real.to(self.device)
        with torch.no_grad():
            _, _, _ = self.generator(real)

        # Discriminator update
        recon, _, _ = self.generator(real)
        d_loss, real_feats = self._d_loss(real, recon)
        self.opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_cfg.grad_clip)
        self.opt_d.step()

        # Generator update
        recon, _, vq_loss = self.generator(real)
        mel = self.mel(real, recon)
        adv, fm = self._g_loss(real, recon, real_feats)
        g_loss = (
            self.train_cfg.mel_weight * mel
            + vq_loss
            + self.train_cfg.adversarial_weight * adv
            + self.train_cfg.feature_weight * fm
        )
        self.opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.train_cfg.grad_clip)
        self.opt_g.step()
        self.step += 1
        return {
            "step": self.step,
            "d_loss": float(d_loss.detach()),
            "g_loss": float(g_loss.detach()),
            "mel": float(mel.detach()),
            "vq": float(vq_loss.detach()),
            "adv": float(adv.detach()),
            "fm": float(fm.detach()),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.generator.state_dict(),
            "config": self.cfg.__dict__,
            "step": self.step,
        }, path)


def train_codec(
    audio_dir: str | Path,
    output_path: str | Path,
    *,
    cfg: CodecConfig | None = None,
    train: CodecTrainConfig | None = None,
) -> dict:
    """Train on a directory tree of audio files; returns the final metrics."""
    cfg = cfg or CodecConfig()
    train = train or CodecTrainConfig()
    torch.manual_seed(train.seed)

    loader = DataLoader(
        AudioClips(
            audio_dir, sample_rate=cfg.sample_rate,
            segment_seconds=train.segment_seconds, seed=train.seed,
        ),
        batch_size=train.batch_size, shuffle=True,
        num_workers=train.num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    trainer = SoundStreamTrainer(cfg, train)
    output_path = Path(output_path)
    last: dict = {}
    start = time.time()
    while trainer.step < train.steps:
        for batch in loader:
            last = trainer.train_step(batch)
            if last["step"] % train.log_every == 0:
                print(
                    f"[codec] step {last['step']} d={last['d_loss']:.3f} "
                    f"g={last['g_loss']:.3f} mel={last['mel']:.3f} "
                    f"vq={last['vq']:.3f} ({(time.time()-start)/max(1,last['step']):.2f}s/step)"
                )
            if last["step"] % train.save_every == 0:
                trainer.save(output_path)
            if trainer.step >= train.steps:
                break
    trainer.save(output_path)
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SoundStream codec for SingSong")
    parser.add_argument("--audio", required=True, help="directory of training audio")
    parser.add_argument("--out", default="checkpoints/singsong/soundstream.pt")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train_codec(
        args.audio, args.out,
        train=CodecTrainConfig(
            steps=args.steps, batch_size=args.batch_size,
            segment_seconds=args.segment_seconds, device=args.device,
            num_workers=args.workers,
        ),
    )


if __name__ == "__main__":
    main()

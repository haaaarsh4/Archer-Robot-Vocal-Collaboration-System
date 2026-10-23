"""A SoundStream-shaped neural audio codec, trainable from scratch.

SingSong does not model waveforms.  It models the discrete codes of a
SoundStream codec: 16 kHz mono, a 320-sample hop (50 frames/second), twelve
residual vector quantizers with 1024 entries each, of which the first four are
the "coarse" codes and the last eight are the "fine" codes.  Google's paper
uses a SoundStream checkpoint pre-trained on their corpus; this module is the
same architecture and training objective so the project can train its own
codec instead of borrowing someone else's weights.

The encoder/decoder are convolutional, the quantizer is residual VQ with
straight-through gradients, and ``codec_training.py`` supplies the multi-scale
STFT discriminator and the reconstruction/adversarial/feature-matching losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CodecConfig
from .tokens import NUM_COARSE_QUANTIZERS


def _prod(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.elu(self.conv1(F.elu(x))))


class SoundStreamEncoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        channels = cfg.channels
        self.stem = nn.Conv1d(1, channels[0], 7, padding=3)
        blocks: list[nn.Module] = []
        for i, stride in enumerate(cfg.strides):
            in_ch = channels[i]
            out_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            blocks.append(
                nn.Sequential(
                    ResidualUnit(in_ch),
                    ResidualUnit(in_ch),
                    ResidualUnit(in_ch),
                    nn.ELU(),
                    nn.Conv1d(
                        in_ch, out_ch, kernel_size=2 * stride, stride=stride,
                        padding=(stride + 1) // 2,
                    ),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv1d(channels[-1], cfg.latent_dim, 3, padding=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.stem(wav)
        for block in self.blocks:
            x = block(x)
        return self.final(F.elu(x))


class SoundStreamDecoder(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        channels = cfg.channels
        self.stem = nn.Conv1d(cfg.latent_dim, channels[-1], 3, padding=1)
        blocks: list[nn.Module] = []
        # Mirror the encoder: the last stride is undone first.
        for i in range(len(cfg.strides) - 1, -1, -1):
            stride = cfg.strides[i]
            in_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            out_ch = channels[i]
            blocks.append(
                nn.Sequential(
                    nn.ELU(),
                    nn.ConvTranspose1d(
                        in_ch, out_ch, kernel_size=2 * stride, stride=stride,
                        padding=(stride + 1) // 2,
                        output_padding=1 if stride % 2 else 0,
                    ),
                    ResidualUnit(out_ch),
                    ResidualUnit(out_ch),
                    ResidualUnit(out_ch),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv1d(channels[0], 1, 7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.stem(z)
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.final(x))


class VectorQuantizer(nn.Module):
    """One VQ book with a straight-through estimator and commitment loss."""

    def __init__(self, codebook_size: int, dim: int, commitment: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment = commitment
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, D, T] -> distances against [K, D]
        flat = x.transpose(1, 2).reshape(-1, self.dim)
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(x.shape[0], x.shape[2], self.dim).transpose(1, 2)
        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment * commitment_loss
        quantized = x + (quantized - x).detach()  # straight-through
        return quantized, indices.view(x.shape[0], x.shape[2]), loss


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.quantizers = nn.ModuleList(
            VectorQuantizer(cfg.codebook_size, cfg.latent_dim, cfg.commitment_loss_weight)
            for _ in range(cfg.num_quantizers)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        indices: list[torch.Tensor] = []
        loss = z.new_zeros(())
        for quantizer in self.quantizers:
            q, idx, q_loss = quantizer(residual)
            residual = residual - q
            quantized = quantized + q
            indices.append(idx)
            loss = loss + q_loss
        return quantized, torch.stack(indices, dim=1), loss / len(self.quantizers)


class SoundStream(nn.Module):
    """Full codec.  ``encode`` returns [B, 12, T] integer codes."""

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SoundStreamEncoder(cfg)
        self.quantizer = ResidualVectorQuantizer(cfg)
        self.decoder = SoundStreamDecoder(cfg)
        self.hop = _prod(cfg.strides)

    def _pad(self, wav: torch.Tensor) -> tuple[torch.Tensor, int]:
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.ndim != 3 or wav.shape[1] != 1:
            raise ValueError("expected mono audio [B, T] or [B, 1, T]")
        length = wav.shape[-1]
        remainder = length % self.hop
        if remainder:
            wav = F.pad(wav, (0, self.hop - remainder))
        return wav, length

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        wav, _ = self._pad(wav)
        z = self.encoder(wav)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode(self, indices: torch.Tensor, length: int | None = None) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError("indices must be [B, quantizers, frames]")
        quantized = torch.zeros(
            (indices.shape[0], self.cfg.latent_dim, indices.shape[2]),
            dtype=self.quantizer.quantizers[0].embedding.weight.dtype,
            device=indices.device,
        )
        for q, index in zip(self.quantizer.quantizers, indices.unbind(dim=1)):
            quantized = quantized + q.embedding(index).transpose(1, 2)
        wav = self.decoder(quantized)
        if length is not None:
            wav = wav[..., :length]
        return wav

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav, length = self._pad(wav)
        z = self.encoder(wav)
        quantized, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(quantized)[..., :length]
        return recon, indices, vq_loss

    def reconstruct_from_codes(
        self, coarse: torch.Tensor, fine: torch.Tensor, length: int | None = None
    ) -> torch.Tensor:
        """Combine the two SingSong code groups and decode them together.

        ``coarse`` is [B, 4, T] and ``fine`` is [B, 8, T].  Both are codebook
        indices, not offsets; the token layout adds its own offsets.
        """
        if coarse.shape[1] != NUM_COARSE_QUANTIZERS:
            raise ValueError(f"expected {NUM_COARSE_QUANTIZERS} coarse streams")
        if fine.shape[1] != self.cfg.num_fine:
            raise ValueError(f"expected {self.cfg.num_fine} fine streams")
        return self.decode(torch.cat((coarse, fine), dim=1), length=length)


@dataclass
class CodecOutput:
    recon: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor


def build_codec(cfg: CodecConfig) -> SoundStream:
    return SoundStream(cfg)


def save_codec(model: SoundStream, path: str, *, config: CodecConfig | None = None) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": (config or model.cfg).__dict__},
        path,
    )


def load_codec(path: str, *, map_location="cpu") -> SoundStream:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = CodecConfig.from_dict(payload.get("config", {}))
    model = SoundStream(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

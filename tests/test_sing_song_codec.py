from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sing_song.codec import SoundStream
from sing_song.codec_training import AudioClips
from sing_song.config import CodecConfig


def _tiny() -> SoundStream:
    return SoundStream(CodecConfig(
        channels=(8, 16, 32, 64), latent_dim=16,
        num_quantizers=12, codebook_size=64,
    ))


def test_codec_is_exactly_fifty_hertz_at_sixteen_khz():
    cfg = CodecConfig()
    assert cfg.sample_rate == 16000
    assert cfg.frame_rate == 50
    assert cfg.num_coarse == 4
    assert cfg.num_fine == 8


def test_encode_decode_and_reconstruct_shapes():
    model = _tiny()
    wav = torch.randn(2, 1, 3200) * 0.1
    recon, indices, vq = model(wav)
    assert recon.shape == wav.shape
    assert indices.shape == (2, 12, 10)
    assert vq.ndim == 0 and torch.isfinite(vq)
    assert model.encode(wav).shape == (2, 12, 10)
    assert model.decode(indices, length=3200).shape == wav.shape
    assert model.reconstruct_from_codes(indices[:, :4], indices[:, 4:], length=3200).shape == wav.shape


def test_codec_has_a_usable_gradient():
    model = _tiny()
    wav = torch.randn(1, 1, 1600) * 0.1
    recon, _, vq = model(wav)
    loss = recon.pow(2).mean() + vq
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_audio_clips_reads_a_one_second_crop_without_full_decode(tmp_path: Path):
    sr = 44_100
    t = np.arange(sr * 3, dtype=np.float32) / sr
    audio = np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)], axis=1)
    path = tmp_path / "track.wav"
    sf.write(path, audio, sr)

    dataset = AudioClips(tmp_path, sample_rate=16_000, segment_seconds=1.0, seed=0)
    assert len(dataset) == 1
    clip = dataset[0]
    assert clip.shape == (16_000,)
    assert torch.isfinite(clip).all()


def test_audio_clips_pads_a_short_file(tmp_path: Path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(4_000, dtype=np.float32), 16_000)
    dataset = AudioClips(tmp_path, sample_rate=16_000, segment_seconds=1.0, seed=0)
    clip = dataset[0]
    assert clip.shape == (16_000,)
    assert float(clip.abs().max()) == 0.0

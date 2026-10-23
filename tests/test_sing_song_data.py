from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sing_song.config import DataConfig
from sing_song.semantic_training import training_audio_root
from sing_song.data import (
    TokenDataset,
    add_noise,
    decide_clip,
    discover_tracks,
    iter_clips,
    peak_rms_db,
    prepare_dataset,
    tokenize_clip,
    write_codec_corpus,
)


class FakeCodec:
    class cfg:
        sample_rate = 16000
        frame_rate = 50

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        b, _, n = wav.shape
        return torch.zeros(b, 12, n // 320, dtype=torch.long)


class FakeSemantic:
    def encode(self, wav: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        return torch.zeros(wav.shape[0], wav.shape[-1] // 640, dtype=torch.long)


def test_peak_rms_distinguishes_silence_from_a_tone():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    assert peak_rms_db(np.zeros(sr, dtype=np.float32), sr) < -100
    assert -12 < peak_rms_db(tone, sr) < -5


def test_clip_filter_matches_the_paper_thresholds():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    vocal = 0.1 * np.sin(2 * np.pi * 220 * t)
    quiet = np.zeros_like(vocal)
    loud_vocal = 0.4 * np.sin(2 * np.pi * 220 * t)
    quiet_inst = np.zeros_like(vocal)
    normal_inst = 0.2 * np.sin(2 * np.pi * 110 * t)
    assert not decide_clip(vocal, quiet_inst, sr).keep
    assert decide_clip(vocal, normal_inst, sr).keep
    assert not decide_clip(loud_vocal, normal_inst, sr).keep


def test_clips_are_non_overlapping_and_drop_the_remainder():
    sr = 16000
    audio = np.zeros(sr * 25, dtype=np.float32)
    clips = list(iter_clips(audio, audio, sr, clip_seconds=10.0))
    assert len(clips) == 2
    assert [c.start_sample for c in clips] == [0, sr * 10]
    assert all(len(c.vocal) == sr * 10 for c in clips)


def test_noise_has_the_requested_sigma_and_does_not_mutate_input():
    audio = np.zeros(16000, dtype=np.float32)
    rng = np.random.default_rng(0)
    noisy = add_noise(audio, 0.01, rng)
    assert noisy.shape == audio.shape
    assert abs(float(noisy.std()) - 0.01) < 0.002
    assert float(audio.std()) == 0.0


def test_discover_tracks_recognises_a_vocal_instrumental_pair(tmp_path: Path):
    track = tmp_path / "song"
    track.mkdir()
    sf.write(track / "vocals.wav", np.zeros(16000, dtype=np.float32), 16000)
    sf.write(track / "instrumental.wav", np.zeros(16000, dtype=np.float32), 16000)
    pairs = discover_tracks(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].name == "song"
    assert pairs[0].vocals.name == "vocals.wav"


def test_tokenize_clip_produces_the_paper_shapes():
    sr = 16000
    t = np.arange(sr * 10, dtype=np.float32) / sr
    clip = next(iter_clips(
        0.2 * np.sin(2 * np.pi * 220 * t),
        0.2 * np.sin(2 * np.pi * 110 * t),
        sr, clip_seconds=10.0,
    ))
    tokens = tokenize_clip(
        clip, codec=FakeCodec(), semantic=FakeSemantic(),
        target_sample_rate=16000, semantic_rate=25,
        noise_sigma=0.01, rng=np.random.default_rng(0),
    )
    assert tokens["semantic"].shape == (250,)
    assert tokens["coarse"].shape == (4, 500)
    assert tokens["fine"].shape == (8, 500)


def test_semantic_training_never_reads_the_test_split(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    cfg = replace(DataConfig(), corpus=str(tmp_path), musdb_split="train")
    assert training_audio_root(cfg) == tmp_path / "train"

    flat = tmp_path / "flat"
    flat.mkdir()
    cfg = replace(cfg, corpus=str(flat))
    assert training_audio_root(cfg) == flat


def test_write_codec_corpus_writes_instrumental_only_audio(tmp_path: Path):
    corpus = tmp_path / "corpus"
    track = corpus / "train" / "one"
    track.mkdir(parents=True)
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    sf.write(track / "vocals.wav", 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    sf.write(track / "instrumental.wav", 0.2 * np.sin(2 * np.pi * 110 * t), sr)

    cfg = replace(
        DataConfig(), corpus=str(corpus), sample_rate=sr, musdb_split="train"
    )
    out = tmp_path / "codec_audio"
    result = write_codec_corpus(cfg, out, log=lambda *_: None)
    assert result["tracks"] == 1
    assert result["warnings"] == []
    assert (out / "one.wav").exists()
    audio, actual_sr = sf.read(out / "one.wav")
    assert actual_sr == sr
    assert audio.shape == (sr * 2,)


def test_prepare_dataset_writes_an_index_and_a_loadable_dataset(tmp_path: Path):
    corpus = tmp_path / "corpus"
    track = corpus / "one"
    track.mkdir(parents=True)
    sr = 16000
    t = np.arange(sr * 10, dtype=np.float32) / sr
    sf.write(track / "vocals.wav", 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    sf.write(track / "instrumental.wav", 0.2 * np.sin(2 * np.pi * 110 * t), sr)

    token_dir = tmp_path / "tokens"
    cfg = replace(
        DataConfig(), corpus=str(corpus), token_dir=str(token_dir),
        clip_seconds=10.0, sample_rate=sr, overwrite=True,
    )
    manifest = prepare_dataset(
        cfg, codec=FakeCodec(), semantic=FakeSemantic(), log=lambda *_: None
    )
    assert manifest["clips"] == 1
    index = [json.loads(line) for line in (token_dir / "index.jsonl").read_text().splitlines()]
    assert len(index) == 1
    dataset = TokenDataset(token_dir, expected_frames=250)
    item = dataset[0]
    assert item["semantic"].shape == (250,)
    assert item["coarse"].shape == (4, 500)
    assert item["fine"].shape == (8, 500)

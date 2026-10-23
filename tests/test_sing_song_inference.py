from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from sing_song.config import FineConfig, InferenceConfig, SingSongConfig, Stage12Config
from sing_song.inference import SingSongEngine, _fit_frames, _window_weights
from sing_song.model import build_fine, build_stage12
from sing_song.tokens import FineLayout, Stage12Layout


def test_window_weights_sum_to_one_in_the_overlap():
    n, hop = 200, 100
    a = _window_weights(n, hop, first=True, last=False)
    b = _window_weights(n, hop, first=False, last=True)
    # The first window's tail and the second window's head overlap.
    assert np.allclose(a[-hop:] + b[:hop], 1.0, atol=1e-5)


def test_fit_frames_pads_and_trims():
    tokens = torch.arange(5).view(1, 5)
    assert _fit_frames(tokens, 5).shape == (1, 5)
    assert _fit_frames(tokens, 3).tolist() == [[0, 1, 2]]
    assert _fit_frames(tokens, 7).shape == (1, 7)


def test_engine_status_is_honest_before_any_weights_exist():
    cfg = SingSongConfig()
    engine = SingSongEngine(cfg)
    status = engine.status()
    assert status["ready"] is False
    assert status["loaded"] is False
    assert len(status["missing"]) == 5


class FakeCodec:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(sample_rate=16000, frame_rate=50)

    def reconstruct_from_codes(self, coarse, fine, length=None):
        return torch.zeros(coarse.shape[0], 1, length or 1)


class FakeSemantic:
    def encode(self, wav, sample_rate=16000):
        return torch.zeros(wav.shape[0], 8, dtype=torch.long)


def _fake_engine() -> SingSongEngine:
    cfg = SingSongConfig(
        inference=InferenceConfig(
            window_seconds=0.32, overlap_seconds=0.16, max_seconds=1.0,
            temperature=0.0, add_noise=False, seed=0,
        )
    )
    engine = SingSongEngine(cfg)
    engine.codec = FakeCodec()
    engine.semantic = FakeSemantic()
    engine.stage12 = build_stage12(Stage12Config(
        d_model=32, encoder_layers=1, decoder_layers=1, heads=4, d_ff=64, dropout=0.0
    )).eval()
    engine.fine = build_fine(FineConfig(
        d_model=32, layers=1, heads=4, d_ff=64, dropout=0.0
    )).eval()
    engine.device = torch.device("cpu")
    engine.dtype = torch.float32
    engine.loaded = True
    return engine


def test_generate_runs_the_whole_cascade_and_returns_the_input_length():
    engine = _fake_engine()
    vocal = np.zeros(8000, dtype=np.float32)
    result = engine.generate(vocal, 16000)
    assert result["sample_rate"] == 16000
    assert result["instrumental"].shape == (8000,)
    assert result["mix"].shape == (8000,)
    assert np.isfinite(result["instrumental"]).all()
    assert len(result["windows"]) >= 1
    assert result["windows"][0]["semantic_tokens"] == 8


def test_generate_caps_at_the_configured_max_seconds():
    engine = _fake_engine()
    engine.cfg = replace(engine.cfg, inference=replace(
        engine.cfg.inference, max_seconds=0.25
    ))
    result = engine.generate(np.zeros(16000, dtype=np.float32), 16000)
    assert result["mix"].shape == (4000,)

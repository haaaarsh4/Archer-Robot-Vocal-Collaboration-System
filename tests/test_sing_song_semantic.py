"""Semantic-token adapter tests that do not need a downloaded checkpoint.

These guard two mistakes that are easy to make and invisible in a shape-only
test: resampling audio to the feature *frame rate* instead of 16 kHz, and
sending a waveform to w2v-BERT 2.0, which expects mel ``input_features``.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from sing_song.config import SemanticConfig
from sing_song.semantic import SemanticEncoder


class _FakeKMeans:
    def predict(self, features):
        import numpy as np

        return np.zeros(len(features), dtype=np.int64)


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.inputs: list[int] = []

    def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask):
        self.inputs.append(int(wav.shape[-1]))
        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        assert return_attention_mask is True
        return {
            "input_features": torch.zeros(1, 50, 160),
            "attention_mask": torch.ones(1, 50, dtype=torch.long),
        }


class _FakeFeatureModel:
    def __init__(self, frames: int = 50) -> None:
        self.frames = frames
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert "input_features" in kwargs
        return SimpleNamespace(
            hidden_states=[torch.zeros(1, self.frames, 8) for _ in range(9)]
        )


class _FakeWaveformModel:
    def __init__(self, frames: int = 50) -> None:
        self.frames = frames
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert "input_values" in kwargs
        return SimpleNamespace(
            hidden_states=[torch.zeros(1, self.frames, 8) for _ in range(9)]
        )


def _encoder(features: bool) -> SemanticEncoder:
    enc = SemanticEncoder(SemanticConfig(), device=torch.device("cpu"))
    enc.kmeans = _FakeKMeans()
    enc.uses_features = features
    if features:
        enc.feature_extractor = _FakeFeatureExtractor()
        enc.model = _FakeFeatureModel()
    else:
        enc.model = _FakeWaveformModel()
    return enc


def test_w2v_bert_uses_mel_features_and_reduces_50hz_to_25hz():
    enc = _encoder(features=True)
    tokens = enc.encode(torch.zeros(1, 16_000), sample_rate=16_000)
    assert tokens.shape == (1, 25)
    assert enc.feature_extractor.inputs == [16_000]
    call = enc.model.calls[0]
    assert call["input_features"].shape == (1, 50, 160)
    assert call["attention_mask"].shape == (1, 50)


def test_waveform_models_still_use_input_values():
    enc = _encoder(features=False)
    tokens = enc.encode(torch.zeros(1, 16_000), sample_rate=16_000)
    assert tokens.shape == (1, 25)
    assert enc.model.calls[0]["input_values"].shape == (1, 16_000)


def test_audio_is_resampled_to_16k_not_to_the_feature_frame_rate():
    enc = _encoder(features=True)
    enc.encode(torch.zeros(1, 8_000), sample_rate=8_000)
    # The fake extractor sees the upsampled 16 kHz waveform, not 50 samples.
    assert enc.feature_extractor.inputs == [16_000]

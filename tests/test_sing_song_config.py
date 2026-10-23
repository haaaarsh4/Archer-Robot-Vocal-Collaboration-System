from __future__ import annotations

import os

import yaml

from sing_song.config import (
    FineConfig,
    SingSongConfig,
    Stage12Config,
    load_sing_song_config,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raw() -> dict:
    with open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_section_is_present_and_matches_the_paper():
    cfg = load_sing_song_config(_raw())
    assert cfg.enabled is True
    assert cfg.codec.sample_rate == 16000
    assert cfg.codec.frame_rate == 50
    assert cfg.codec.num_quantizers == 12
    assert cfg.semantic.output_hz == 25
    assert cfg.data.clip_seconds == 10.0
    assert cfg.data.noise_sigma == 0.01
    assert cfg.data.instrumental_peak_rms_db == -25.0
    assert cfg.data.vocal_over_instrumental_db == 5.0
    assert cfg.inference.temperature == 0.85
    assert cfg.inference.overlap_seconds == 5.0


def test_cree_profile_points_at_paired_stems_and_separate_checkpoints():
    with open(os.path.join(ROOT, "config", "cree.yaml"), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = load_sing_song_config(raw)
    assert cfg.data.corpus == "data/cree_singsong/train_audio"
    assert cfg.data.token_dir == "data/cree_tokens"
    assert cfg.paths.codec == "checkpoints/cree/soundstream.pt"
    assert cfg.semantic.kmeans_path == "checkpoints/cree/semantic_kmeans.joblib"
    assert cfg.stage12.gradient_checkpointing is True
    assert cfg.fine.gradient_checkpointing is True


def test_status_reports_missing_artifacts_without_raising():
    cfg = load_sing_song_config({"singsong": {}})
    status = cfg.status()
    assert status["ready"] is False
    assert len(status["missing"]) == 5
    assert all(":" in item for item in status["missing"])


def test_presets_have_the_published_depths():
    base = Stage12Config.base()
    xl = Stage12Config.xl()
    assert base.d_model == 768 and base.encoder_layers == 12 and base.heads == 12
    assert base.dropout == 0.1
    assert xl.d_model == 1024 and xl.encoder_layers == 24 and xl.heads == 32
    fine = FineConfig.xl()
    assert fine.layers == 24 and fine.d_model == 1024


def test_unknown_keys_are_rejected_instead_of_ignored():
    import pytest

    with pytest.raises(ValueError):
        SingSongConfig.from_dict({"inference": {"temperatur": 0.5}})

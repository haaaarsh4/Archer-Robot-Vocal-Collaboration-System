"""Typed configuration for the in-repo SingSong implementation.

Everything here is optional at import time: the server can ask for a status
report on a machine with no GPU and no checkpoints, and the training scripts can
run on a cluster with a different YAML.  No torch import, so ``--help`` and
status checks stay fast.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _as_path(value: Any, *, root: Path = PROJECT_ROOT) -> Path | None:
    if value in (None, "", False):
        return None
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = root / path
    return path


@dataclass(frozen=True)
class Stage12Config:
    """The T5 encoder-decoder that predicts semantic + coarse instrumental codes.

    ``base`` follows the paper's t5.1.1.base shape (12 encoder and 12 decoder
    layers, d_model 768, 12 heads, d_ff 3072, dropout 0.1).  ``xl`` follows the
    published 3B scaling direction (24 layers, d_model 1024, 32 heads,
    d_ff 16384).  The vocabulary here is 5122 rather than T5's 32k sentencepiece
    vocabulary, so the parameter count is not identical to stock T5; the
    transformer shape, depth, dropout and fixed positional encodings are.
    """

    preset: str = "base"
    d_model: int = 768
    encoder_layers: int = 12
    decoder_layers: int = 12
    heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    max_seq_len: int = 4096
    gradient_checkpointing: bool = False

    @classmethod
    def base(cls) -> "Stage12Config":
        return cls()

    @classmethod
    def xl(cls) -> "Stage12Config":
        return cls(
            preset="xl",
            d_model=1024,
            encoder_layers=24,
            decoder_layers=24,
            heads=32,
            d_ff=16384,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "Stage12Config":
        raw = dict(raw or {})
        preset = str(raw.get("preset", "base")).lower()
        base = cls.xl() if preset == "xl" else cls.base()
        allowed = {
            "preset", "d_model", "encoder_layers", "decoder_layers", "heads",
            "d_ff", "dropout", "attention_dropout", "max_seq_len",
            "gradient_checkpointing",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown stage12 model keys: {sorted(unknown)}")
        return replace(base, **raw)


@dataclass(frozen=True)
class FineConfig:
    """The decoder-only transformer that predicts fine codes from coarse codes."""

    preset: str = "base"
    d_model: int = 768
    layers: int = 12
    heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    max_seq_len: int = 8192
    gradient_checkpointing: bool = False

    @classmethod
    def base(cls) -> "FineConfig":
        return cls()

    @classmethod
    def xl(cls) -> "FineConfig":
        return cls(
            preset="xl",
            d_model=1024,
            layers=24,
            heads=32,
            d_ff=16384,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "FineConfig":
        raw = dict(raw or {})
        preset = str(raw.get("preset", "base")).lower()
        base = cls.xl() if preset == "xl" else cls.base()
        allowed = {
            "preset", "d_model", "layers", "heads", "d_ff", "dropout",
            "attention_dropout", "max_seq_len", "gradient_checkpointing",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown fine model keys: {sorted(unknown)}")
        return replace(base, **raw)


@dataclass(frozen=True)
class CodecConfig:
    """SoundStream-shaped 16 kHz codec, 12 RVQ books, 50 Hz frames."""

    sample_rate: int = 16000
    latent_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 4, 5, 8)
    num_quantizers: int = 12
    codebook_size: int = 1024
    commitment_loss_weight: float = 0.25
    codebook_loss_weight: float = 1.0
    discriminator_fft_sizes: tuple[int, ...] = (2048, 1024, 512)
    discriminator_hops: tuple[int, ...] = (512, 256, 128)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CodecConfig":
        raw = dict(raw or {})
        for key in ("channels", "strides", "discriminator_fft_sizes", "discriminator_hops"):
            if key in raw:
                raw[key] = tuple(int(x) for x in raw[key])
        allowed = {
            "sample_rate", "latent_dim", "channels", "strides", "num_quantizers",
            "codebook_size", "commitment_loss_weight", "codebook_loss_weight",
            "discriminator_fft_sizes", "discriminator_hops",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown codec keys: {sorted(unknown)}")
        return cls(**raw)

    @property
    def frame_rate(self) -> int:
        return self.sample_rate // int(_prod(self.strides))

    @property
    def num_coarse(self) -> int:
        return 4

    @property
    def num_fine(self) -> int:
        return self.num_quantizers - self.num_coarse


def _prod(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


@dataclass(frozen=True)
class SemanticConfig:
    """w2v-BERT / HuBERT features plus the k-means codebook from the paper."""

    backend: str = "w2vbert"
    model_path: str = "data/singsong/semantic/w2v-bert-2.0"
    sample_rate: int = 16_000
    hidden_layer: int = 7
    output_hz: int = 25
    native_hz: int = 50
    codebook_size: int = 1024
    kmeans_path: str = "checkpoints/singsong/semantic_kmeans.joblib"
    normalize_embeddings: bool = True
    device: str = "auto"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SemanticConfig":
        raw = dict(raw or {})
        allowed = {
            "backend", "model_path", "sample_rate", "hidden_layer", "output_hz",
            "native_hz", "codebook_size", "kmeans_path", "normalize_embeddings", "device",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown semantic keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class DataConfig:
    corpus: str = "data/singsong/corpus"
    token_dir: str = "data/singsong/tokens"
    clip_seconds: float = 10.0
    sample_rate: int = 16000
    semantic_rate: int = 25
    noise_sigma: float = 0.01
    instrumental_peak_rms_db: float = -25.0
    vocal_over_instrumental_db: float = 5.0
    workers: int = 4
    separation_backend: str = "demucs"
    separation_command: str | None = None
    musdb_split: str = "train"
    overwrite: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DataConfig":
        raw = dict(raw or {})
        allowed = {
            "corpus", "token_dir", "clip_seconds", "sample_rate", "semantic_rate", "noise_sigma",
            "instrumental_peak_rms_db", "vocal_over_instrumental_db", "workers",
            "separation_backend", "separation_command", "musdb_split", "overwrite",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown data keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class InferenceConfig:
    temperature: float = 0.85
    top_k: int = 0
    window_seconds: float = 10.0
    overlap_seconds: float = 5.0
    max_seconds: float = 120.0
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 0
    add_noise: bool = True
    mix_vocal: bool = True
    vocal_gain_db: float = 0.0
    instrumental_gain_db: float = 0.0
    crossfade_ms: float = 50.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "InferenceConfig":
        raw = dict(raw or {})
        allowed = {
            "temperature", "top_k", "window_seconds", "overlap_seconds",
            "max_seconds", "device", "dtype", "seed", "add_noise",
            "mix_vocal", "vocal_gain_db", "instrumental_gain_db", "crossfade_ms",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown inference keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class PathsConfig:
    codec: str = "checkpoints/singsong/soundstream.pt"
    stage12: str = "checkpoints/singsong/stage12.pt"
    fine: str = "checkpoints/singsong/fine.pt"
    semantic_kmeans: str = "checkpoints/singsong/semantic_kmeans.joblib"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "PathsConfig":
        raw = dict(raw or {})
        allowed = {"codec", "stage12", "fine", "semantic_kmeans"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown artifact keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True)
class SingSongConfig:
    enabled: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    stage12: Stage12Config = field(default_factory=Stage12Config)
    fine: FineConfig = field(default_factory=FineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "SingSongConfig":
        raw = dict(raw or {})
        allowed = {
            "enabled", "paths", "codec", "semantic", "stage12", "fine",
            "data", "inference", "model",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown singsong keys: {sorted(unknown)}")

        model_raw = dict(raw.get("model") or {})
        stage12_raw = model_raw.get("stage12", raw.get("stage12"))
        fine_raw = model_raw.get("fine", raw.get("fine"))

        return cls(
            enabled=bool(raw.get("enabled", True)),
            paths=PathsConfig.from_dict(raw.get("paths")),
            codec=CodecConfig.from_dict(raw.get("codec")),
            semantic=SemanticConfig.from_dict(raw.get("semantic")),
            stage12=Stage12Config.from_dict(stage12_raw),
            fine=FineConfig.from_dict(fine_raw),
            data=DataConfig.from_dict(raw.get("data")),
            inference=InferenceConfig.from_dict(raw.get("inference")),
        )

    # Artifact paths and accurate readiness
    def resolve(self, value: str | None) -> Path | None:
        return _as_path(value)

    @property
    def codec_path(self) -> Path | None:
        return self.resolve(self.paths.codec)

    @property
    def stage12_path(self) -> Path | None:
        return self.resolve(self.paths.stage12)

    @property
    def fine_path(self) -> Path | None:
        return self.resolve(self.paths.fine)

    @property
    def semantic_model_path(self) -> Path | None:
        return self.resolve(self.semantic.model_path)

    @property
    def semantic_kmeans_path(self) -> Path | None:
        return self.resolve(self.semantic.kmeans_path)

    def missing_artifacts(self) -> list[str]:
        """Every file this configuration needs but cannot find.

        A missing checkpoint is not an exception: it is the normal state of a
        fresh clone, and the UI is supposed to say exactly which artifact is
        missing rather than pretend the mode works.  The semantic model may be
        a Hugging Face directory or a local directory; either way it has to
        exist on disk because this project never downloads at inference time.
        """
        missing: list[str] = []
        for label, path in (
            ("SoundStream codec checkpoint", self.codec_path),
            ("semantic k-means", self.semantic_kmeans_path),
            ("stage 1+2 transformer", self.stage12_path),
            ("fine transformer", self.fine_path),
        ):
            if path is None or not path.exists():
                missing.append(f"{label}: {path}")
        model_path = self.semantic_model_path
        if model_path is None or not model_path.exists():
            missing.append(f"semantic encoder (w2v-BERT/HuBERT): {model_path}")
        return missing

    @property
    def ready(self) -> bool:
        return self.enabled and not self.missing_artifacts()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "missing": self.missing_artifacts(),
            "paths": {
                "codec": str(self.codec_path),
                "stage12": str(self.stage12_path),
                "fine": str(self.fine_path),
                "semantic_model": str(self.semantic_model_path),
                "semantic_kmeans": str(self.semantic_kmeans_path),
            },
            "model": {
                "stage12_preset": self.stage12.preset,
                "fine_preset": self.fine.preset,
                "stage12_parameters_m": round(_stage12_params(self.stage12) / 1e6, 1),
                "fine_parameters_m": round(_fine_params(self.fine) / 1e6, 1),
            },
            "codec": {
                "sample_rate": self.codec.sample_rate,
                "frame_rate": self.codec.frame_rate,
                "num_quantizers": self.codec.num_quantizers,
                "num_coarse": self.codec.num_coarse,
                "num_fine": self.codec.num_fine,
            },
            "semantic": {
                "backend": self.semantic.backend,
                "output_hz": self.semantic.output_hz,
                "codebook_size": self.semantic.codebook_size,
            },
        }


def _stage12_params(cfg: Stage12Config) -> int:
    d = cfg.d_model
    emb = 5122 * d
    layer = 4 * d * d + 2 * d * cfg.d_ff + 2 * d * d  # self, cross, ffn
    return emb + cfg.encoder_layers * (4 * d * d + 2 * d * cfg.d_ff) + \
        cfg.decoder_layers * layer + d


def _fine_params(cfg: FineConfig) -> int:
    d = cfg.d_model
    emb = 4096 * d + 8194 * d
    return emb + cfg.layers * (4 * d * d + 2 * d * cfg.d_ff) + d


def load_sing_song_config(raw: Mapping[str, Any] | None = None) -> SingSongConfig:
    """Load the ``singsong`` block from config.yaml (or a supplied mapping)."""
    if raw is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    section = (raw or {}).get("singsong", {})
    return SingSongConfig.from_dict(section)

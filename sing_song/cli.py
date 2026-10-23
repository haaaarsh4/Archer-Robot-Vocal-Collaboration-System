"""Command line entry point for the in-repo SingSong pipeline.

Examples::

    python -m sing_song.cli status
    python -m sing_song.cli prepare-codec
    python -m sing_song.cli prepare --separate --limit 20
    python -m sing_song.cli train-codec --audio data/singsong/codec_audio
    python -m sing_song.cli train-kmeans
    python -m sing_song.cli train-stage12 --steps 200000 --batch-size 8 --grad-accum 64
    python -m sing_song.cli train-fine --steps 200000 --batch-size 1 --grad-accum 512
    python -m sing_song.cli render --input vocal.wav --out renders/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from .config import PROJECT_ROOT, SingSongConfig, load_sing_song_config
from .inference import SingSongEngine
from .training import TrainConfig


def _load(path: str | None) -> SingSongConfig:
    if not path:
        return load_sing_song_config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return load_sing_song_config(raw)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    print(json.dumps(cfg.status(), indent=2))
    return 0 if cfg.ready else 2


def cmd_prepare_codec(args: argparse.Namespace) -> int:
    from .data import write_codec_corpus

    cfg = _load(args.config)
    result = write_codec_corpus(
        cfg.data, args.out, log=lambda msg: print(msg, file=sys.stderr)
    )
    print(json.dumps(result, indent=2))
    for warning in result["warnings"]:
        print(f"[singsong] warning: {warning}", file=sys.stderr)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .codec import load_codec
    from .data import prepare_dataset
    from .semantic import load_semantic

    cfg = _load(args.config)
    missing = [p for p in (cfg.codec_path, cfg.semantic_model_path, cfg.semantic_kmeans_path) if not p or not p.exists()]
    if missing:
        print("[singsong] cannot prepare data; missing:\n  " + "\n  ".join(map(str, missing)), file=sys.stderr)
        return 2
    codec = load_codec(str(cfg.codec_path))
    semantic = load_semantic(cfg.semantic, kmeans_path=cfg.semantic_kmeans_path)
    manifest = prepare_dataset(
        cfg.data, codec=codec, semantic=semantic,
        separate=args.separate, limit=args.limit,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_train_codec(args: argparse.Namespace) -> int:
    from .codec_training import CodecTrainConfig, train_codec

    cfg = _load(args.config)
    out = args.out or str(cfg.codec_path)
    result = train_codec(
        args.audio, out,
        cfg=cfg.codec,
        train=CodecTrainConfig(
            steps=args.steps, batch_size=args.batch_size,
            segment_seconds=args.segment_seconds, device=args.device,
            num_workers=args.workers,
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_train_kmeans(args: argparse.Namespace) -> int:
    from .semantic_training import train_semantic_kmeans

    cfg = _load(args.config)
    path = train_semantic_kmeans(
        cfg.data, cfg.semantic, max_frames=args.max_frames, device=args.device
    )
    print(f"[singsong] semantic k-means written to {path}")
    return 0


def _train_cfg(args: argparse.Namespace, cfg: SingSongConfig) -> TrainConfig:
    base = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir or str(PROJECT_ROOT / "checkpoints" / "singsong"),
        resume_from=args.resume,
        compile=args.compile,
    )
    return base


def cmd_train_stage12(args: argparse.Namespace) -> int:
    from .training import train_stage12

    cfg = _load(args.config)
    result = train_stage12(cfg.stage12, cfg.data, _train_cfg(args, cfg))
    print(json.dumps(result, indent=2))
    return 0


def cmd_train_fine(args: argparse.Namespace) -> int:
    from .training import train_fine

    cfg = _load(args.config)
    result = train_fine(cfg.fine, cfg.data, _train_cfg(args, cfg))
    print(json.dumps(result, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    engine = SingSongEngine(cfg)
    engine.load()
    out = Path(args.out)
    result = engine.render_file(args.input, out, return_mix=not args.instrumental_only)
    printable = {k: v for k, v in result.items() if k not in ("instrumental", "mix")}
    print(json.dumps(printable, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sing_song.cli", description=__doc__)
    p.add_argument("--config", default=None, help="optional YAML file with a singsong: block")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="show readiness and the exact missing artifacts")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser(
        "prepare-codec",
        help="write one instrumental WAV per track for codec training",
    )
    s.add_argument("--out", default="data/singsong/codec_audio")
    s.set_defaults(func=cmd_prepare_codec)

    s = sub.add_parser("prepare", help="separate, filter, tokenize and cache a corpus")
    s.add_argument("--separate", action="store_true", help="run Demucs on mixture-only tracks")
    s.add_argument("--limit", type=int, default=None, help="only the first N tracks")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train-codec", help="train the SoundStream codec")
    s.add_argument("--audio", required=True)
    s.add_argument("--out", default=None)
    s.add_argument("--steps", type=int, default=200_000)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--segment-seconds", type=float, default=1.0)
    s.add_argument("--device", default="auto")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_train_codec)

    s = sub.add_parser("train-kmeans", help="fit the semantic codebook")
    s.add_argument("--max-frames", type=int, default=2_000_000)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_train_kmeans)

    for name, func, default_batch, default_accum in (
        ("train-stage12", cmd_train_stage12, 8, 64),
        ("train-fine", cmd_train_fine, 1, 512),
    ):
        s = sub.add_parser(name)
        s.add_argument("--steps", type=int, default=200_000)
        s.add_argument("--batch-size", type=int, default=default_batch)
        s.add_argument("--grad-accum", type=int, default=default_accum)
        s.add_argument("--device", default="auto")
        s.add_argument("--dtype", default="auto")
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--checkpoint-dir", default=None)
        s.add_argument("--resume", default=None)
        s.add_argument("--compile", action="store_true")
        s.set_defaults(func=func)

    s = sub.add_parser("render", help="generate an instrumental for a vocal file")
    s.add_argument("--input", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--instrumental-only", action="store_true")
    s.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

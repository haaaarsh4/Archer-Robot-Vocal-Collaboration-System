#!/usr/bin/env python3
"""Verify that a trained SingSong installation is actually usable.

This is the check to run after the five training commands. It verifies:

- the config loads and every required artifact exists;
- the codec checkpoint loads and encodes/decodes a one-second waveform;
- the semantic encoder and k-means load;
- stage 1+2 and stage 3 load with the configured architecture;
- the engine reports ready;
- optionally, a held-out vocal renders finite, non-silent audio.

It does not replace the unit tests. Run both:

    python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py
    python tools/check_singsong_training.py --config config/config.yaml

Smoke-render 10 seconds of a held-out vocal:

    python tools/check_singsong_training.py \
      --config config/config.yaml --render heldout_vocal.wav --seconds 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sing_song.codec import load_codec
from sing_song.config import load_sing_song_config
from sing_song.inference import SingSongEngine


def _check_artifacts(cfg) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    missing = cfg.missing_artifacts()
    errors.extend(missing)
    for label, path in (
        ("codec", cfg.codec_path),
        ("stage12", cfg.stage12_path),
        ("fine", cfg.fine_path),
        ("semantic_kmeans", cfg.semantic_kmeans_path),
    ):
        if path is None or not path.exists():
            continue
        size = path.stat().st_size
        if size <= 0:
            errors.append(f"{label} checkpoint is empty: {path}")
        else:
            notes.append(f"{label}: {size / 1e6:.1f} MB")
    model = cfg.semantic_model_path
    if model is not None and model.exists():
        weights = list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
        if not weights:
            errors.append(f"semantic model directory has no weights: {model}")
        else:
            notes.append(f"semantic weights: {len(weights)} file(s)")
    return errors, notes


def _check_codec(cfg) -> list[str]:
    codec = load_codec(str(cfg.codec_path), map_location="cpu").eval()
    wav = torch.zeros(1, 1, 16_000)
    with torch.no_grad():
        codes = codec.encode(wav)
        recon = codec.decode(codes, length=16_000)
    errors = []
    if tuple(codes.shape) != (1, 12, 50):
        errors.append(f"codec codes shape {tuple(codes.shape)}, expected (1, 12, 50)")
    if tuple(recon.shape) != (1, 1, 16_000):
        errors.append(f"codec recon shape {tuple(recon.shape)}, expected (1, 1, 16000)")
    if not torch.isfinite(recon).all():
        errors.append("codec produced non-finite samples")
    return errors


def _check_engine(cfg) -> tuple[list[str], SingSongEngine | None]:
    errors: list[str] = []
    engine = SingSongEngine(cfg)
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001 - reported as a verification failure
        return [f"engine load failed: {exc}"], None
    status = engine.status()
    if not status.get("ready"):
        errors.append(f"engine reports not ready: {status.get('missing')}")
    if not status.get("loaded"):
        errors.append("engine did not report loaded=True")
    return errors, engine


def _check_render(
    engine: SingSongEngine,
    audio_path: str,
    *,
    seconds: float,
    out_path: str | None,
) -> list[str]:
    import librosa
    import soundfile as sf

    errors: list[str] = []
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if audio.size == 0:
        return [f"render input is empty: {audio_path}"]
    audio = audio[: int(round(seconds * sr))]
    if engine.cfg.inference.max_seconds < seconds:
        engine.cfg = replace(
            engine.cfg,
            inference=replace(engine.cfg.inference, max_seconds=float(seconds)),
        )
    result = engine.generate(audio, sr)
    instrumental = result["instrumental"]
    mix = result["mix"]
    if instrumental.shape != audio.shape:
        errors.append(
            f"instrumental shape {instrumental.shape} != input {audio.shape}"
        )
    if mix.shape != audio.shape:
        errors.append(f"mix shape {mix.shape} != input {audio.shape}")
    if not np.isfinite(instrumental).all() or not np.isfinite(mix).all():
        errors.append("render contains NaN or infinity")
    peak = float(np.max(np.abs(instrumental))) if instrumental.size else 0.0
    if peak < 1e-6:
        errors.append("generated instrumental is silent")
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, mix, sr, subtype="PCM_16")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify trained SingSong artifacts")
    parser.add_argument("--config", default=None, help="optional YAML with a singsong: block")
    parser.add_argument("--render", default=None, help="held-out vocal WAV for a smoke render")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", default=None, help="optional output WAV for the smoke mix")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_sing_song_config() if args.config is None else load_sing_song_config(
        __import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))
    )
    errors, notes = _check_artifacts(cfg)
    if not errors:
        errors.extend(_check_codec(cfg))
    engine = None
    if not errors:
        engine_errors, engine = _check_engine(cfg)
        errors.extend(engine_errors)
    if engine is not None and args.render and not errors:
        errors.extend(
            _check_render(
                engine, args.render, seconds=args.seconds, out_path=args.out
            )
        )

    payload = {
        "ok": not errors,
        "ready": cfg.ready,
        "artifacts": notes,
        "rendered": bool(args.render and not errors),
        "errors": errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"ready: {payload['ready']}")
        for note in notes:
            print(f"  {note}")
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print("OK" if not errors else "FAILED")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

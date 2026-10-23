#!/usr/bin/env python3
"""Validate a Cree SingSong dataset before any training run.

The dataset is a directory of paired stems plus one metadata row per song:

    data/cree_singsong/
      audio/<song_id>/vocals.wav
      audio/<song_id>/instrumental.wav
      metadata.jsonl

This checks the things that silently ruin a fine-tune: missing stems, different
lengths, a singer leaking across train/test, duplicate song IDs, missing
consent/licence fields, and too little audio. It reads audio headers only; it
does not load or alter the recordings.

Usage:
    python tools/check_cree_singsong_dataset.py --root data/cree_singsong
    python tools/check_cree_singsong_dataset.py --root data/cree_singsong \
        --min-hours 5 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

REQUIRED = (
    "song_id",
    "singer_id",
    "dialect",
    "style",
    "language",
    "consent_id",
    "license",
    "split",
)
SPLITS = {"train", "dev", "test"}
STEMS = ("vocals", "instrumental")
EXTS = (".wav", ".flac")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    songs: int = 0
    seconds: float = 0.0
    by_split: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    singers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> dict:
        return {
            "ok": not self.errors,
            "songs": self.songs,
            "paired_hours": round(self.seconds / 3600, 3),
            "hours_by_split": {
                split: round(seconds / 3600, 3)
                for split, seconds in sorted(self.by_split.items())
            },
            "singers": {
                singer: sorted(splits) for singer, splits in sorted(self.singers.items())
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _metadata(path: Path) -> list[dict]:
    if path.suffix == ".jsonl" or path.name.endswith(".jsonl"):
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "songs" in data:
        data = data["songs"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list or JSONL")
    return data


def _find(root: Path, song_id: str, stem: str) -> Path | None:
    for ext in EXTS:
        path = root / "audio" / song_id / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def validate(root: str | Path, *, min_hours: float = 0.0) -> Report:
    root = Path(root)
    report = Report()
    meta_path = root / "metadata.jsonl"
    if not meta_path.exists():
        meta_path = root / "metadata.json"
    if not meta_path.exists():
        report.add(f"missing metadata file: {root / 'metadata.jsonl'}")
        return report
    try:
        rows = _metadata(meta_path)
    except Exception as exc:  # noqa: BLE001 - turned into a validation error
        report.add(str(exc))
        return report

    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            report.add(f"row {index}: expected an object")
            continue
        missing = [key for key in REQUIRED if not str(row.get(key, "")).strip()]
        if missing:
            report.add(f"row {index}: missing required fields: {', '.join(missing)}")
            continue
        song_id = str(row["song_id"])
        if song_id in seen:
            report.add(f"duplicate song_id: {song_id}")
            continue
        seen.add(song_id)
        split = str(row["split"])
        if split not in SPLITS:
            report.add(f"{song_id}: split must be one of {sorted(SPLITS)}, got {split!r}")
            continue
        singer = str(row["singer_id"])
        report.singers[singer].add(split)

        found = {stem: _find(root, song_id, stem) for stem in STEMS}
        for stem, path in found.items():
            if path is None:
                report.add(f"{song_id}: missing audio/{song_id}/{stem}.wav (or .flac)")
        if any(path is None for path in found.values()):
            continue

        infos = {}
        for stem, path in found.items():
            try:
                info = sf.info(str(path))
            except Exception as exc:  # noqa: BLE001
                report.add(f"{song_id}: cannot read {path}: {exc}")
                continue
            if info.frames <= 0:
                report.add(f"{song_id}: {stem} is empty")
            infos[stem] = info

        if len(infos) != len(STEMS):
            continue
        vocal, instrumental = infos["vocals"], infos["instrumental"]
        if vocal.samplerate != instrumental.samplerate:
            report.add(
                f"{song_id}: sample rates differ "
                f"({vocal.samplerate} vs {instrumental.samplerate})"
            )
        if abs(vocal.frames - instrumental.frames) > max(2, int(0.05 * vocal.samplerate)):
            report.add(
                f"{song_id}: stem lengths differ "
                f"({vocal.frames / vocal.samplerate:.3f}s vs "
                f"{instrumental.frames / instrumental.samplerate:.3f}s)"
            )
        duration = min(
            vocal.frames / vocal.samplerate, instrumental.frames / instrumental.samplerate
        )
        report.songs += 1
        report.seconds += duration
        report.by_split[split] += duration

    for singer, splits in report.singers.items():
        if len(splits) > 1:
            report.add(
                f"singer {singer!r} appears in multiple splits: {sorted(splits)}; "
                "use singer-disjoint train/dev/test"
            )

    if report.seconds / 3600 < min_hours:
        report.add(
            f"only {report.seconds / 3600:.3f} paired hours; "
            f"requested at least {min_hours:.3f}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate paired Cree SingSong stems and metadata"
    )
    parser.add_argument("--root", default="data/cree_singsong")
    parser.add_argument("--min-hours", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = validate(args.root, min_hours=args.min_hours)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"songs: {report.songs}")
        print(f"paired hours: {report.seconds / 3600:.3f}")
        for split, seconds in sorted(report.by_split.items()):
            print(f"  {split}: {seconds / 3600:.3f} h")
        for warning in report.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        for error in report.errors:
            print(f"error: {error}", file=sys.stderr)
        print("OK" if not report.errors else "FAILED")
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

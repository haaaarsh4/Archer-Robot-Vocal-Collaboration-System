from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "check_cree_singsong_dataset.py"


def _write_dataset(root: Path, *, leak: bool = False) -> None:
    sr = 16_000
    audio = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    rows = []
    for index, (song, singer, split) in enumerate(
        [("song_a", "singer_1", "train"), ("song_b", "singer_1" if leak else "singer_2", "test")]
    ):
        folder = root / "audio" / song
        folder.mkdir(parents=True)
        sf.write(folder / "vocals.wav", audio, sr)
        sf.write(folder / "instrumental.wav", audio * 0.5, sr)
        rows.append({
            "song_id": song,
            "singer_id": singer,
            "dialect": "Plains Cree",
            "style": "round dance",
            "language": "cree",
            "consent_id": f"consent_{index}",
            "license": "research",
            "split": split,
        })
    (root / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(root), "--json", *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_valid_paired_dataset_passes(tmp_path: Path):
    _write_dataset(tmp_path)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["songs"] == 2
    assert payload["paired_hours"] > 0


def test_singer_leakage_fails(tmp_path: Path):
    _write_dataset(tmp_path, leak=True)
    result = _run(tmp_path)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("multiple splits" in error for error in payload["errors"])


def test_minimum_hours_is_enforced(tmp_path: Path):
    _write_dataset(tmp_path)
    result = _run(tmp_path, "--min-hours", "10")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any("paired hours" in error for error in payload["errors"])

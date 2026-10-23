"""What the trained voice adds to a hum, band by band, across pitches.

Fetches the takes the panel's live bridge actually asks for (the same body the
page posts, see NeuralNoteBridge._fireRender) and compares them with the murmur
the DSP path renders at the same pitch. The difference is the correction
restore_closed_mouth has to make; this is the number that decides the filter.

Run:  ARCHER_PORT=8000 .venv/bin/python scratch/hum_restore_fit.py
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402

PORT = os.environ.get("ARCHER_PORT", "8000")
PITCHES = [123.47, 146.83, 174.61, 220.0, 261.63, 329.63, 392.0]
NF = 6.0   # bands per octave


def steady(y: np.ndarray, sr: int, seconds: float = 0.5) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(sr * seconds)
    if y.size <= n:
        return y
    start = max(0, int(y.size * 0.5) - n // 2)
    return y[start:start + n]


def envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    x = steady(y, sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    centers = 100.0 * (2 ** (np.arange(0, 4.65 * NF) / NF))
    vals = []
    for c in centers:
        lo, hi = c / 2 ** (0.5 / NF), c * 2 ** (0.5 / NF)
        sel = (freqs >= lo) & (freqs < hi)
        vals.append(20.0 * np.log10(float(mag[sel].max()) if sel.any() else 1e-15))
    v = np.asarray(vals)
    return centers, v - v.max()


def fetch(hz: float, seconds: float = 1.8) -> tuple[np.ndarray, int, str]:
    body = json.dumps({"target_hz": hz, "vocable": "aah", "duration_s": seconds,
                       "mode": "solo_humming", "texture": "solo",
                       "neural_voice_count": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/neural/render", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        applied = resp.headers.get("X-Neural-Applied")
        raw = resp.read()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return y, sr, applied


def main() -> int:
    cfg = HumConfig.from_config()
    diffs: list[np.ndarray] = []
    centers: np.ndarray | None = None
    print(f"port {PORT}: converted hum minus murmur, in dB, per band\n")
    for hz in PITCHES:
        y, sr, applied = fetch(hz)
        c, env_conv = envelope(y, sr)
        _, env_ref = envelope(synthesize_hum_note(hz, int(1.8 * sr), sr, cfg), sr)
        if centers is None:
            centers = c
        d = env_conv - env_ref
        # only bands that carry real energy in the murmur are meaningful
        d = np.where(env_ref > -45.0, d, np.nan)
        diffs.append(d)
        keep = (c >= 100) & (c <= 2500)
        print(f"  {hz:6.1f} Hz (applied={applied}): " +
              " ".join(f"{v:+.0f}" for v in d[keep]))
        print(f"  {'':10s}center Hz: " + " ".join(f"{v:.0f}" for v in c[keep]))
    arr = np.asarray(diffs, dtype=np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(arr, axis=0)
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
    keep = (centers >= 100) & (centers <= 2600)
    print("\n  band  median  min  max   (across the seven pitches)")
    for c, m, a, b in zip(centers[keep], med[keep], lo[keep], hi[keep]):
        print(f"  {c:7.0f} {m:+7.1f} {a:+5.1f} {b:+5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The correction a converted hum needs, as a function of frequency.

Per harmonic, not per third-octave band: every partial of the converted take is
compared with the SAME partial of the murmur at that pitch, both referenced to
their own fundamental, so the number at frequency f is "how much louder the
trained voice is at f than a murmur is". Binning those samples by frequency
across seven pitches gives the response the restore filter has to have.

Run:  ARCHER_PORT=8000 .venv/bin/python scratch/hum_restore_measure.py
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
CACHE = "/tmp/hum_takes"
PITCHES = [123.47, 146.83, 174.61, 220.0, 261.63, 329.63, 392.0]


def steady(y: np.ndarray, sr: int, seconds: float = 0.6) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(sr * seconds)
    if y.size <= n:
        return y
    start = max(0, int(y.size * 0.5) - n // 2)
    return y[start:start + n]


def partials(y: np.ndarray, sr: int, f0: float, n_harm: int = 30) -> tuple[np.ndarray, np.ndarray]:
    x = steady(y, sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    fs, vs = [], []
    for k in range(1, n_harm + 1):
        f = k * f0
        if f >= sr / 2 - 400:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.98, f * 1.02])
        fs.append(f)
        vs.append(20.0 * np.log10(float(mag[max(lo, 0):max(hi, lo + 1)].max())))
    v = np.asarray(vs)
    return np.asarray(fs), v - v.max()


def fetch(hz: float, seconds: float = 1.8) -> tuple[np.ndarray, int]:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{hz:.2f}.wav"
    if os.path.exists(path):
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        return y, sr
    body = json.dumps({"target_hz": hz, "vocable": "aah", "duration_s": seconds,
                       "mode": "solo_humming", "texture": "solo",
                       "neural_voice_count": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/neural/render", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()
    with open(path, "wb") as fh:
        fh.write(raw)
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return y, sr


def main() -> int:
    cfg = HumConfig.from_config()
    samples: list[tuple[float, float]] = []
    for hz in PITCHES:
        y, sr = fetch(hz)
        f_c, p_c = partials(y, sr, hz)
        _, p_r = partials(synthesize_hum_note(hz, int(1.8 * sr), sr, cfg), sr, hz)
        n = min(len(p_c), len(p_r))
        for f, a, b in zip(f_c[:n], p_c[:n], p_r[:n]):
            if b > -55.0 and f < 3000.0:      # only where the murmur has content
                samples.append((float(f), float(a - b)))
        print(f"  {hz:6.1f} Hz  " + " ".join(f"{f:.0f}:{a - b:+.0f}" for f, a, b in
                                             zip(f_c[:12], p_c[:12], p_r[:12])))

    arr = np.asarray(samples)
    np.savez("/tmp/hum_correction_samples.npz", freq=arr[:, 0], diff=arr[:, 1])
    print(f"\n{len(arr)} partial samples; correction needed, by frequency band")
    print("   band   n   median   min    max")
    edges = np.asarray([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                        1250, 1600, 2000, 2500, 3000], dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if not sel.any():
            continue
        v = arr[sel, 1]
        print(f"  {lo:5.0f}-{hi:<5.0f} {sel.sum():3d}  {np.median(v):+6.1f} "
              f"{v.min():+6.1f} {v.max():+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

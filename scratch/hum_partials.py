"""Compare the converted hum the panel plays with the DSP murmur, cheaply.

No librosa, no server import: the takes are already on disk (fetched by curl,
see logs), and the murmur is rendered by hum_voice alone. Memory is the reason
this is a separate, small script -- importing the app in this environment gets
killed.

Run:  .venv/bin/python scratch/hum_partials.py /tmp/hum220.wav ...
"""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402


def steady(y: np.ndarray, sr: int, seconds: float = 0.5) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(sr * seconds)
    if y.size <= n:
        return y
    start = max(0, int(y.size * 0.5) - n // 2)
    return y[start:start + n]


def partial_table(y: np.ndarray, sr: int, f0: float, n_harm: int = 20) -> np.ndarray:
    x = steady(y, sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    vals = []
    for k in range(1, n_harm + 1):
        f = k * f0
        if f >= sr / 2 - 300:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.97, f * 1.03])
        vals.append(20.0 * np.log10(float(mag[max(lo, 0):max(hi, lo + 1)].max())))
    v = np.asarray(vals)
    return v - v.max()


def smoothed_db(y: np.ndarray, sr: int, octave_frac: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Third-octave-ish envelope in dB, from 80 Hz up: shows formant structure."""
    x = steady(y, sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    centers = 80.0 * (2 ** (np.arange(0, 6.0 * octave_frac) / octave_frac))
    out = []
    for c in centers:
        lo, hi = c / 2 ** (0.5 / octave_frac), c * 2 ** (0.5 / octave_frac)
        sel = (freqs >= lo) & (freqs < hi)
        out.append(20.0 * np.log10(float(mag[sel].max()) if sel.any() else 1e-15))
    env = np.asarray(out)
    return centers, env - env.max()


def describe(name: str, y: np.ndarray, sr: int, f0: float) -> None:
    p = partial_table(y, sr, f0)
    idx = np.arange(1, len(p) + 1)
    above = idx * f0 > 700.0
    share = float((10 ** (p[above] / 10)).sum() / (10 ** (p / 10)).sum()) if above.any() else 0.0
    gap = float(p.max() - p[above].max()) if above.any() else 99.0
    # the /o/ tell: how loud the second partial is next to the first
    print(f"  {name:34s} f0 {f0:6.1f} | H2-H1 {p[1]:6.1f} H3-H1 {p[2]:6.1f} H4-H1 "
          f"{p[3]:6.1f} dB | top>700 {gap:5.1f} dB under | >700 share {share * 100:5.2f}%")
    centers, env = smoothed_db(y, sr)
    show = [(c, v) for c, v in zip(centers, env) if 100 <= c <= 2500]
    print("      envelope (Hz:dBr) " + "  ".join(f"{c:.0f}:{v:.0f}" for c, v in show[::2]))


def main(argv: list[str]) -> int:
    cfg = HumConfig.from_config()
    import re
    takes = [(float(re.search(r"(\d+(?:\.\d+)?)\.wav", a).group(1)), a) for a in argv[1:]]
    print("=== converted hum the panel plays, against the DSP murmur ===")
    for hz, path in sorted(takes):
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        describe(f"panel take {hz} Hz", y, sr, float(hz))
        ref = synthesize_hum_note(float(hz), int(1.8 * sr), sr, cfg)
        describe(f"DSP murmur {hz} Hz", ref, sr, float(hz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

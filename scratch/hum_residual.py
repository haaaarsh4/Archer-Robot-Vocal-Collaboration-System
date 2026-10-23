"""Where the hum the panel plays still differs from the murmur, pitch by pitch.

Uses the takes already cached from the running app (/tmp/hum_takes, fetched from
/api/neural/render with the page's own body) and the murmur the DSP path renders
at the same pitch. Residual = converted - murmur, per harmonic, both referenced
to their own fundamental: positive means the trained voice is louder there than a
hum is, which is exactly what the ear reads as a vowel.

Run:  .venv/bin/python scratch/hum_residual.py
"""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402

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


def partials(y: np.ndarray, sr: int, f0: float, n_harm: int = 40) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> int:
    cfg = HumConfig.from_config()
    sr = 44100
    print("harmonic:  frequency  murmur(dB)  converted(dB)  residual(dB)")
    for hz in PITCHES:
        y, ysr = sf.read(f"/tmp/hum_takes/{hz:.2f}.wav", dtype="float32", always_2d=False)
        f_c, p_c = partials(y, ysr, hz)
        ref = synthesize_hum_note(hz, int(1.8 * sr), sr, cfg)
        _, p_r = partials(ref, sr, hz)
        n = min(len(p_c), len(p_r))
        f, p_c, p_r = f_c[:n], p_c[:n], p_r[:n]
        aud = p_r > -30.0
        pow_c, pow_r = 10 ** (p_c / 10), 10 ** (p_r / 10)
        above = f > 700.0
        print(f"\n  pitch {hz:6.1f} Hz  (audible harmonics: {int(aud.sum())})"
              f"   >700 Hz share {pow_c[above].sum() / pow_c.sum() * 100:5.2f}% "
              f"vs murmur {pow_r[above].sum() / pow_r.sum() * 100:5.2f}%"
              f"   top>700 {p_c.max() - p_c[above].max():5.1f} dB under "
              f"(murmur {p_r.max() - p_r[above].max():5.1f})")
        for fr, a, b in zip(f, p_c, p_r):
            if b < -55.0 and a < -55.0:
                continue
            mark = "  <-- excess" if (a - b) > 6.0 else ""
            print(f"     {fr:7.0f} {b:9.1f} {a:12.1f} {a - b:+11.1f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

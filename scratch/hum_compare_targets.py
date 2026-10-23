"""The server's own hum note, measured with the exact definitions the in-page
JavaScript measurement uses, so the browser's live hum can be held to it.

Definitions (both sides):
  * partial level  = peak of |DFT|^2 in a +-15 Hz neighbourhood of k*f0, Hann
    window over 1.0-2.6 s of a 3 s note, reported relative to the loudest
    partial anywhere;
  * hi_peak_db     = loudest partial above 700 Hz, relative to the loudest;
  * f2_worst_db    = loudest partial in 500-1300 Hz, relative to the loudest;
  * hi_share       = share of the partial energy above 700 Hz;
  * centroid / hf  = Welch (nperseg 4096), as synthesis.hum_voice.hum_measurements.

    .venv/bin/python scratch/hum_compare_targets.py
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.signal as sps

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402

SR = 44100
PITCHES = (196.0, 262.0)


NFFT = 65536


def partial_table(y: np.ndarray, f0: float, tol: float = 15.0) -> dict:
    # exactly the window the in-page measurement uses: a power-of-two FFT
    # (65536, 1.49 s) starting 1 s into the note, Hann.
    a = int(1.0 * SR)
    seg = np.asarray(y[a:a + NFFT], dtype=np.float64)
    win = np.hanning(seg.size)
    freqs = np.fft.rfftfreq(seg.size, 1.0 / SR)
    spec = np.abs(np.fft.rfft(seg * win)) ** 2
    rows = []
    for k in range(1, 41):
        hz = k * f0
        if hz > 0.45 * SR:
            break
        m = np.abs(freqs - hz) <= tol
        rows.append((hz, float(spec[m].max()) if np.any(m) else 0.0))
    top = max(p for _, p in rows)
    db = lambda p: 10.0 * np.log10(max(p, 1e-30) / top)  # noqa: E731
    hi = [r for r in rows if r[0] > 700.0]
    f2 = [r for r in rows if 500.0 <= r[0] < 1300.0]
    tot = sum(p for _, p in rows) or 1.0
    return {
        "hi_peak_db": round(max((db(p) for _, p in hi), default=-99.0), 1),
        "f2_worst_db": round(max((db(p) for _, p in f2), default=-99.0), 1),
        "hi_share": round(sum(p for hz, p in hi if hz > 700.0) / tot, 4),
        "peak_hz": rows[int(np.argmax([p for _, p in rows]))][0],
        "levels": " ".join(f"{hz:.0f}:{db(p):+.0f}" for hz, p in rows),
    }


def welch_metrics(y: np.ndarray) -> dict:
    f, p = sps.welch(np.asarray(y, dtype=np.float64), fs=SR, nperseg=4096)
    m = f > 40.0
    p, f = p[m], f[m]
    total = float(p.sum()) + 1e-18
    return {
        "centroid_hz": round(float((f * p).sum() / total), 1),
        "hf_ratio": round(float(p[f > 2500.0].sum() / (float(np.sum(p)) + 1e-18)), 4),
        "rms": round(float(np.sqrt(np.mean(np.asarray(y, np.float64) ** 2))), 4),
        "noise_share": round(float(p[(f > 700.0) & (f < 2500.0)].sum() / total), 4),
    }


def main() -> int:
    cfg = HumConfig.from_config()
    for f0 in PITCHES:
        y = np.asarray(synthesize_hum_note(f0, int(3.0 * SR), SR, cfg), dtype=np.float64)
        t = partial_table(y, f0)
        w = welch_metrics(y[int(1.0 * SR):int(2.6 * SR)])
        print(f"\nserver hum note at {f0:.0f} Hz")
        print(f"    hi_peak {t['hi_peak_db']:+.1f} dB   f2_worst {t['f2_worst_db']:+.1f} dB   "
              f"hi_share {100 * t['hi_share']:.2f}%   loudest partial {t['peak_hz']:.0f} Hz")
        print(f"    centroid {w['centroid_hz']} Hz   above 2.5 kHz {100 * w['hf_ratio']:.3f}%   "
              f"700-2500 Hz energy share {100 * w['noise_share']:.3f}%   rms {w['rms']}")
        print(f"    partials: {t['levels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

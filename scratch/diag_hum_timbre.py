"""Is the hum a murmur or a vowel? Measured on the partials, which is what the
ear actually reads.

A closed-mouth hum keeps its energy at the bottom: the partials above about
700 Hz are all well under the loudest one, and their share of the sound's total
harmonic energy is small. A vowel puts a second group of comparable partials up
there -- that group IS the vowel colour, which is how "mmm" and "ohhh" differ
while both stay dark (a spectral centroid barely moves; this moves a lot).

    .venv/bin/python scratch/diag_hum_timbre.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

PITCHES = (98.0, 117.5, 131.0, 147.0, 165.0, 196.0, 220.0, 262.0, 294.0, 330.0, 392.0)
CUTOFF_HZ = 700.0
F2_BAND = (800.0, 1300.0)


def _partials(y: np.ndarray, sr: int, f0: float, n_harm: int = 40):
    y = np.asarray(y, dtype=np.float64)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    tol = sr / len(y) * 2.0
    hz, pw = [], []
    for k in range(1, n_harm + 1):
        if k * f0 > sr * 0.45:
            break
        m = np.abs(freqs - k * f0) <= tol
        hz.append(k * f0)
        pw.append(float(spec[m].max()) if np.any(m) else 0.0)
    return np.array(hz), np.array(pw)


def _render(cfg, f0: float, sr: int = 44100, seconds: float = 2.5) -> np.ndarray:
    from synthesis.hum_voice import synthesize_hum_note
    y = synthesize_hum_note(f0, int(seconds * sr), sr, cfg)
    return np.asarray(y, dtype=np.float64)[int(0.6 * sr):int(2.4 * sr)]


def _summary(cfg, sr: int = 44100) -> dict:
    rows = []
    for f0 in PITCHES:
        hz, pw = _partials(_render(cfg, f0, sr), sr, f0)
        if pw.size < 4 or pw.max() <= 0:
            continue
        top = pw.max()
        db = 10.0 * np.log10((pw + 1e-30) / top)
        hi = hz > CUTOFF_HZ
        f2 = (hz >= F2_BAND[0]) & (hz < F2_BAND[1])
        rows.append({
            "f0": f0,
            "hi_peak_db": float(db[hi].max()) if np.any(hi) else -99.0,
            "hi_share": float(pw[hi].sum() / pw.sum()) if np.any(hi) else 0.0,
            "f2_db": float(db[f2].max()) if np.any(f2) else -99.0,
            "peak_hz": float(hz[int(np.argmax(pw))]),
        })
    from synthesis.hum_voice import hum_measurements
    m = hum_measurements(_render(cfg, 196.0, sr), np.zeros(int(2.4 * sr), np.float32), sr, cfg)
    return {
        "worst_hi_peak_db": max(r["hi_peak_db"] for r in rows),
        "median_hi_peak_db": float(np.median([r["hi_peak_db"] for r in rows])),
        "worst_hi_share": max(r["hi_share"] for r in rows),
        "median_hi_share": float(np.median([r["hi_share"] for r in rows])),
        "worst_f2_db": max(r["f2_db"] for r in rows),
        "centroid": m.get("hum_centroid_hz"),
        "rows": rows,
    }


def _print(title: str, cfg) -> None:
    s = _summary(cfg)
    print(f"\n{title}")
    print(f"    above {CUTOFF_HZ:.0f} Hz vs the loudest partial anywhere:  median "
          f"{s['median_hi_peak_db']:+.1f} dB, worst {s['worst_hi_peak_db']:+.1f} dB"
          f"   |  energy share median {100 * s['median_hi_share']:.1f}%, worst {100 * s['worst_hi_share']:.1f}%")
    print(f"    loudest partial in {F2_BAND[0]:.0f}-{F2_BAND[1]:.0f} Hz: worst "
          f"{s['worst_f2_db']:+.1f} dB   |  centroid {s['centroid']} Hz")
    print("      " + "  ".join(f"{r['f0']:.0f}Hz:{r['hi_peak_db']:+.0f}/{100 * r['hi_share']:.0f}%"
                               for r in s["rows"]))


def main() -> int:
    from synthesis.hum_voice import HumConfig

    _print("AS CONFIGURED NOW  (F1 260/60, F2 1000/100, zero 1600 -11 Q1.4, tilt -9 over 900, lp 3400)",
           HumConfig())

    def cand(**kw):
        c = HumConfig()
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    _print("A: F2 damped, zero over F2, steeper tilt",
           cand(f2_hz=1000.0, b2_hz=320.0, nasal_zero_hz=1050.0, nasal_zero_depth_db=-20.0,
                nasal_zero_q=1.6, tilt_hz=600.0, tilt_db_per_oct=-12.0, lowpass_hz=2600.0))
    _print("B: A + wider F2, deeper zero, softer source, quieter/narrower breath",
           cand(f2_hz=1000.0, b2_hz=520.0, nasal_zero_hz=1000.0, nasal_zero_depth_db=-24.0,
                nasal_zero_q=1.2, tilt_hz=500.0, tilt_db_per_oct=-12.0, lowpass_hz=2400.0,
                open_quotient=0.50, breath_cap_db=-32.0, breath_hp_hz=250.0, breath_lp_hz=1200.0))
    _print("C: B + a lower, wider F1 (nasal damping) and more body",
           cand(f2_hz=1000.0, b2_hz=520.0, nasal_zero_hz=1000.0, nasal_zero_depth_db=-24.0,
                nasal_zero_q=1.2, tilt_hz=500.0, tilt_db_per_oct=-12.0, lowpass_hz=2400.0,
                open_quotient=0.50, breath_cap_db=-32.0, breath_hp_hz=250.0, breath_lp_hz=1200.0,
                f1_hz=280.0, b1_hz=95.0, body_gain_db=4.5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

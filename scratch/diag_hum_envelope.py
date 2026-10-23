"""Line up the browser's live hum voice with the server's murmur.

The two hums are separate implementations on purpose (the page cannot run
scipy), and the project keeps them within a couple of dB of each other by hand.
This prints the SPECTRAL ENVELOPE of both -- the server's cascade of resonators
plus shelves against the browser's chain of peaking filters -- as a level at
each frequency against that chain's own maximum, with the source's own tilt
divided out (the server's differentiated Rosenberg pulse and the browser's
sawtooth both fall about 6 dB per octave, so the two are comparable).

    .venv/bin/python scratch/diag_hum_envelope.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")

POINTS = [100, 150, 200, 260, 330, 400, 500, 600, 700, 800, 1000, 1200,
          1400, 1600, 1800, 2000, 2500, 3000, 4000]


def _server_db(cfg, sr: int = 44100) -> np.ndarray:
    from scipy import signal as s

    from synthesis.hum_voice import _antiresonance, _resonator, _shelf

    stages = [_resonator(cfg.f1_hz, cfg.b1_hz, sr), _resonator(cfg.f2_hz, cfg.b2_hz, sr),
              _resonator(cfg.f3_hz, cfg.b3_hz, sr), _resonator(cfg.f4_hz, cfg.b4_hz, sr),
              _antiresonance(cfg.nasal_zero_hz, cfg.nasal_zero_q, cfg.nasal_zero_depth_db, sr),
              _shelf(cfg.body_hz, cfg.body_gain_db, sr, q=0.8, high=False),
              _shelf(cfg.tilt_hz, cfg.tilt_db_per_oct, sr, q=0.7, high=True),
              _shelf(cfg.hf_shelf_hz, cfg.hf_shelf_db, sr, q=0.7, high=True)]
    ours = s.butter(int(cfg.lowpass_order), cfg.lowpass_hz / (sr / 2.0), btype="low", output="sos")
    w = 2.0 * np.pi * np.array(POINTS, dtype=np.float64) / sr
    mag = np.ones_like(w)
    for st in stages:
        _, h = s.freqz(st[:3], np.concatenate([[1.0], st[3:]]), worN=w)
        mag *= np.abs(h)
    _, h = s.sosfreqz(ours, worN=w)
    mag *= np.abs(h)
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def _browser_db(stages, sr: int = 44100) -> np.ndarray:
    from scipy import signal as s

    from synthesis.hum_voice import _peaking, _shelf

    w = 2.0 * np.pi * np.array(POINTS, dtype=np.float64) / sr
    mag = np.ones_like(w)
    for st in stages:
        kind = st[0]
        if kind == "peaking":
            coeffs = _peaking(st[1], st[2], st[3], sr)
        elif kind == "lowshelf":
            coeffs = _shelf(st[1], st[2], sr, q=0.7, high=False)
        elif kind == "highshelf":
            coeffs = _shelf(st[1], st[2], sr, q=0.7, high=True)
        elif kind == "lowpass":
            # Web Audio's lowpass biquad is a 2nd-order filter at Q
            b, a = s.butter(2, min(0.99, st[1] / (sr / 2.0)), btype="low")
            coeffs = np.concatenate([[b[0]], b[1:], a[1:]]) if st[1] else None
            if coeffs is None:
                continue
            _, h = s.freqz(coeffs[:3], np.concatenate([[1.0], coeffs[3:]]), worN=w)
            mag *= np.abs(h)
            continue
        else:
            continue
        _, h = s.freqz(coeffs[:3], np.concatenate([[1.0], coeffs[3:]]), worN=w)
        mag *= np.abs(h)
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def _show(title: str, db: np.ndarray) -> np.ndarray:
    ref = db.max()
    rel = db - ref
    print(f"\n{title}   (peak at {POINTS[int(np.argmax(db))]} Hz)")
    print("    " + "  ".join(f"{p}:{v:+.0f}" for p, v in zip(POINTS, rel)))
    return rel


def main() -> int:
    from synthesis.hum_voice import HumConfig

    server = _show("server murmur (HumConfig as shipped)", _server_db(HumConfig()))

    browser = _browser_db([("peaking", 260.0, 15.0, 1.69), ("peaking", 1000.0, 9.0, 4.55),
                           ("peaking", 2250.0, 7.0, 7.3), ("peaking", 3200.0, 5.0, 7.3),
                           ("lowshelf", 250.0, 3.0), ("lowpass", 3400.0, 0.7)])
    _show("browser hum voice, AS IT IS NOW (peaking 260+15/1000+9/2250+7/3200+5, "
          "notch 1600, chest +3, lp 3400)", browser)
    print("    diff vs server: " + "  ".join(f"{p}:{b - a:+.0f}"
                                             for p, a, b in zip(POINTS, server, browser)))

    for label, stages in (
            ("A: peaking 260+14 Q1.7, 1000 -7 Q1.0, 2250 -15 Q0.9; chest +3; tilt 800 -8; lp 2400",
             [("peaking", 260.0, 14.0, 1.7), ("peaking", 1000.0, -7.0, 1.0),
              ("peaking", 2250.0, -15.0, 0.9), ("lowshelf", 250.0, 3.0),
              ("highshelf", 800.0, -8.0), ("lowpass", 2400.0, 0.7)]),
            ("B: A + a second lowpass at 2 kHz and an hf shelf",
             [("peaking", 260.0, 14.0, 1.7), ("peaking", 1000.0, -7.0, 1.0),
              ("peaking", 2250.0, -15.0, 0.9), ("lowshelf", 250.0, 3.0),
              ("highshelf", 500.0, -9.0), ("highshelf", 1800.0, -6.0),
              ("lowpass", 2400.0, 0.7), ("lowpass", 2000.0, 0.7)]),
            ("C: darker dip, steeper tilt",
             [("peaking", 260.0, 14.0, 1.7), ("peaking", 1000.0, -9.0, 0.9),
              ("peaking", 2250.0, -18.0, 0.9), ("lowshelf", 250.0, 4.0),
              ("highshelf", 500.0, -10.0), ("highshelf", 1500.0, -7.0),
              ("lowpass", 2200.0, 0.7), ("lowpass", 2200.0, 0.7)])):
        c = _browser_db(stages)
        _show(f"browser hum voice, {label}", c)
        print("    diff vs server: " + "  ".join(f"{p}:{b - a:+.0f}"
                                                 for p, a, b in zip(POINTS, server, c)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

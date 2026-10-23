"""Re-fit the page's live humming tract to the server's.

The browser builds the murmur from a fixed list of Web Audio biquads
(HUM_TRACT in frontend/index.html) that approximates the server's cascade of
four resonators, the nasal zero, the body shelf, the tilt, the high shelf and a
4th-order low-pass. The server's tract has just changed (a nasal's second
resonance is damped to near-nothing, the zero moved onto 700 Hz, the tilt and
the rolloff are steeper), so the page's stages are fitted again here against the
same relative response tools/check_hum_live_tract.py compares them on.

Run: .venv/bin/python scratch/fit_hum_live_tract.py
"""
import math
import sys

import numpy as np
import scipy.optimize as opt
import scipy.signal as sps

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, _antiresonance, _resonator, _shelf  # noqa: E402

SR = 44100
NORM_HZ = 196.0
PROBE = np.unique(np.concatenate([
    np.linspace(90, 2400, 90), np.geomspace(90, 6000, 60),
    np.array([130, 196, 262, 330, 392, 588, 784, 980, 1000, 1200, 1500, 2000]),
]))


def biquad_db(freqs, b, a):
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / SR
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    return 20.0 * np.log10(np.abs((b[0] + b[1] * z1 + b[2] * z2) /
                                  (a[0] + a[1] * z1 + a[2] * z2)) + 1e-30)


def page_stage_db(freqs, kind, f0, q, gain_db):
    w0 = 2.0 * math.pi * f0 / SR
    cw, sw = math.cos(w0), math.sin(w0)
    if kind == "peaking":
        A = 10.0 ** (gain_db / 40.0)
        al = sw / (2.0 * max(q, 0.05))
        return biquad_db(freqs, [1 + al * A, -2 * cw, 1 - al * A], [1 + al / A, -2 * cw, 1 - al / A])
    if kind == "lowpass":
        al = sw / (2.0 * max(q, 0.05))
        return biquad_db(freqs, [(1 - cw) / 2, 1 - cw, (1 - cw) / 2], [1 + al, -2 * cw, 1 - al])
    A = 10.0 ** (gain_db / 40.0)
    al = sw / 2.0 * math.sqrt(2.0)
    t = 2.0 * math.sqrt(A) * al
    if kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * cw + t), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - t)]
        a = [(A + 1) - (A - 1) * cw + t, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - t]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + t), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - t)]
        a = [(A + 1) + (A - 1) * cw + t, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - t]
    return biquad_db(freqs, b, a)


def server_tract_db(freqs, cfg):
    stages = [_resonator(cfg.f1_hz, cfg.b1_hz, SR), _resonator(cfg.f2_hz, cfg.b2_hz, SR),
              _resonator(cfg.f3_hz, cfg.b3_hz, SR), _resonator(cfg.f4_hz, cfg.b4_hz, SR),
              _antiresonance(cfg.nasal_zero_hz, cfg.nasal_zero_q, cfg.nasal_zero_depth_db, SR),
              _shelf(cfg.body_hz, cfg.body_gain_db, SR, q=0.8, high=False),
              _shelf(cfg.tilt_hz, cfg.tilt_db_per_oct, SR, q=0.7, high=True),
              _shelf(cfg.hf_shelf_hz, cfg.hf_shelf_db, SR, q=0.7, high=True)]
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / SR
    total = np.ones(len(freqs), dtype=complex)
    for c in stages:
        _, h = sps.freqz(c[:3], np.concatenate([[1.0], c[3:]]), worN=w)
        total *= h
    lp = sps.butter(cfg.lowpass_order, min(0.99, cfg.lowpass_hz / (SR / 2.0)), btype="low", output="sos")
    _, h = sps.sosfreqz(lp, worN=w)
    total *= h
    return 20.0 * np.log10(np.maximum(np.abs(total), 1e-12))


# [kind, hz, q, gain, tracks] -- the same eight stages the page builds, in order.
START = np.array([
    265.1, 1.09, 23.4,        # F1
    1004.3, 2.26, 5.4,        # F2
    3044.4, 0.7, -28.8,       # F3, damped away
    320.7, 0.8, 6.9,          # body shelf
    596.3, 0.7, 6.0,          # tilt
    2687.8, 0.7, -34.5,       # the shut lips
    2636.8, 0.541, 0.0,       # low-pass section 1
    2636.8, 1.307, 0.0,       # low-pass section 2
])
KINDS = ["peaking", "peaking", "peaking", "lowshelf", "highshelf", "highshelf", "lowpass", "lowpass"]
BOUNDS = np.array([
    [[200, 340], [0.5, 3.0], [10, 30]],
    [[500, 1400], [0.5, 3.0], [-14, 12]],
    [[1200, 4000], [0.4, 2.0], [-40, -8]],
    [[150, 600], [0.5, 2.0], [-6, 12]],
    [[300, 1400], [0.5, 2.0], [-2, 14]],
    [[900, 4000], [0.5, 2.0], [-45, -12]],
    [[700, 4000], [0.4, 1.5], [-0.001, 0.001]],
    [[700, 4000], [0.4, 2.5], [-0.001, 0.001]],
])


def response(p, freqs=PROBE):
    stages = p.reshape(8, 3)
    out = np.zeros(len(freqs))
    for (hz, q, g), kind in zip(stages, KINDS):
        out += page_stage_db(freqs, kind, hz, q, g)
    return out


def residual(p, target_rel):
    rel = response(p) - float(np.interp(NORM_HZ, PROBE, response(p)))
    body = (PROBE >= 130) & (PROBE <= 2000)
    weight = np.where(body, 1.0, 0.35)
    return (rel - target_rel) * weight


def main():
    cfg = HumConfig.from_config()
    srv = server_tract_db(PROBE, cfg)
    srv_rel = srv - float(np.interp(NORM_HZ, PROBE, srv))

    lo = BOUNDS[:, :, 0].ravel()
    hi = BOUNDS[:, :, 1].ravel()
    x0 = np.clip(START.ravel(), lo, hi)
    best = None
    rng = np.random.default_rng(4)
    for trial in range(4):
        start = x0 if trial == 0 else np.clip(x0 * (1 + 0.25 * rng.standard_normal(x0.size)), lo, hi)
        try:
            res = opt.least_squares(residual, start, args=(srv_rel,), bounds=(lo, hi),
                                    max_nfev=3000, xtol=1e-10, ftol=1e-10)
        except Exception as e:
            print("  trial failed:", e)
            continue
        rel = response(res.x)
        rel -= float(np.interp(NORM_HZ, PROBE, rel))
        body = (PROBE >= 130) & (PROBE <= 2000)
        worst = float(np.max(np.abs(rel - srv_rel)[body]))
        if best is None or worst < best[0]:
            best = (worst, res.x.copy())
        print(f"  trial {trial}: worst {worst:.2f} dB")

    worst, p = best
    print(f"\nbest worst-case difference {worst:.2f} dB (the check allows 6.0)")
    stages = p.reshape(8, 3)
    print("\nconst HUM_TRACT = [")
    for (hz, q, g), kind in zip(stages, KINDS):
        tracks = "true" if kind == "peaking" or kind == "lowshelf" else "false"
        line = f"  ['{kind}', {hz:.1f}, {q:.3f}, {g:.1f}, {tracks}],"
        print(line)
    print("];")
    rel = response(p)
    rel -= float(np.interp(NORM_HZ, PROBE, rel))
    print("\n  Hz      " + " ".join(f"{f:>6.0f}" for f in [196, 330, 392, 588, 700, 784, 980, 1200, 1500, 2000, 3000]))
    for name, curve in (("page", rel), ("server", srv_rel)):
        print(f"  {name:<7} " + " ".join(f"{np.interp(f, PROBE, curve):>6.1f}"
                                         for f in [196, 330, 392, 588, 700, 784, 980, 1200, 1500, 2000, 3000]))
    peak = float(PROBE[int(np.argmax(rel))])
    high = np.max(rel[PROBE >= 1000])
    print(f"\n  page: loudest at {peak:.0f} Hz (must be <= 420), best above 1 kHz {high:+.1f} dB (must be < -8)")


if __name__ == "__main__":
    main()

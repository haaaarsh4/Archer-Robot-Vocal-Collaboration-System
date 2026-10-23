"""Score a candidate page-side HUM_TRACT against the server's tract.

Prints exactly what tools/check_hum_live_tract.py compares: the relative
response (each side normalised to 0 dB at 196 Hz), the worst difference between
130 and 2000 Hz, where the page's loudest response sits, and its best response
above 1 kHz. Cheap on purpose: this environment kills anything long.
"""
import math
import sys

import numpy as np
import scipy.signal as sps

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, _antiresonance, _resonator, _shelf  # noqa: E402

SR = 44100
PROBE = np.array([110, 130, 150, 175, 196, 240, 262, 300, 330, 392, 470, 550, 650, 780, 900,
                  1000, 1100, 1300, 1500, 1700, 2000, 2400, 3000])


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


def score(tract, label):
    cfg = HumConfig.from_config()
    page = np.zeros(len(PROBE))
    for kind, f0, q, g, _t in tract:
        page += page_stage_db(PROBE, kind, f0, q, g)
    srv = server_tract_db(PROBE, cfg)
    p_rel = page - np.interp(196.0, PROBE, page)
    s_rel = srv - np.interp(196.0, PROBE, srv)
    body = (PROBE >= 130) & (PROBE <= 2000)
    worst = float(np.max(np.abs(p_rel - s_rel)[body]))
    peak = float(PROBE[int(np.argmax(p_rel))])
    high = float(np.max(p_rel[PROBE >= 1000]))
    print(f"\n=== {label}: worst {worst:.1f} dB (limit 6.0), loudest at {peak:.0f} Hz "
          f"(limit 420), best above 1 kHz {high:+.1f} dB (limit -8.0) ===")
    for i, f in enumerate(PROBE):
        flag = "  <--" if body[i] and abs(p_rel[i] - s_rel[i]) > 4.0 else ""
        print(f"   {f:6.0f} Hz  page {p_rel[i]:7.1f}  server {s_rel[i]:7.1f}  "
              f"diff {p_rel[i] - s_rel[i]:+6.1f}{flag}")
    return worst, peak, high


CANDIDATES = {
    "v1": [( "peaking", 262.0, 1.2, 22.0, True), ("peaking", 700.0, 1.3, -16.0, True),
           ("peaking", 1900.0, 0.8, -4.0, True), ("lowshelf", 300.0, 0.8, 6.0, True),
           ("highshelf", 620.0, 0.7, 6.0, False), ("highshelf", 1800.0, 0.7, -22.0, False),
           ("lowpass", 1800.0, 0.541, 0, False), ("lowpass", 1800.0, 1.307, 0, False)],
    "v2": [("peaking", 262.0, 1.2, 22.0, True), ("peaking", 700.0, 0.9, -10.0, True),
           ("peaking", 1900.0, 0.8, -6.0, True), ("lowshelf", 300.0, 0.8, 6.0, True),
           ("highshelf", 620.0, 0.7, 6.0, False), ("highshelf", 1200.0, 0.7, -26.0, False),
           ("lowpass", 1400.0, 0.541, 0, False), ("lowpass", 1400.0, 1.307, 0, False)],
    "v3": [("peaking", 262.0, 1.2, 20.0, True), ("peaking", 680.0, 0.85, -8.0, True),
           ("peaking", 1500.0, 0.7, -12.0, True), ("lowshelf", 300.0, 0.8, 5.0, True),
           ("highshelf", 600.0, 0.7, 7.0, False), ("highshelf", 1100.0, 0.7, -24.0, False),
           ("lowpass", 1300.0, 0.541, 0, False), ("lowpass", 1300.0, 1.307, 0, False)],
    "v4": [("peaking", 262.0, 1.1, 19.0, True), ("peaking", 660.0, 0.8, -7.0, True),
           ("peaking", 1300.0, 0.7, -16.0, True), ("lowshelf", 300.0, 0.8, 5.0, True),
           ("highshelf", 600.0, 0.7, 7.0, False), ("highshelf", 1000.0, 0.7, -26.0, False),
           ("lowpass", 1200.0, 0.541, 0, False), ("lowpass", 1200.0, 1.307, 0, False)],
}

if __name__ == "__main__":
    only = sys.argv[1:] or list(CANDIDATES)
    for name in only:
        score(CANDIDATES[name], f"candidate {name}")

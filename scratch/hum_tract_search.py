"""The finer search: which tract keeps the murmur as closed as a plain one?

Each candidate is rendered at the pitch sweep, optionally with an extra low-pass
whose corner follows the note (`lp_track` x f0, never above the tract's own), and
scored against the plain reference from hum_tract_tune: the 700-1500 Hz share and
the loudest partial above 700 Hz.

An LTI filter added after the voice's own tract filters the same signal the tract
would, so the sweep can be measured first and the winning setting built into the
voice afterwards.
"""
import itertools
import math

import numpy as np
from scipy import signal as s

from synthesis.hum_voice import HumConfig, synthesize_hum

SR = 44100
PITCHES = [72.0, 94.0, 140.0, 196.0, 247.0, 280.0, 330.0, 392.0, 440.0]
# what the plain reference closed mouth measures at each pitch (hum_tract_tune)
REFERENCE = {72.0: (0.01, -41.5), 94.0: (0.01, -40.4), 140.0: (0.03, -47.0),
             196.0: (0.16, -25.9), 247.0: (0.61, -21.8), 280.0: (0.06, -32.1),
             330.0: (0.01, -42.0), 392.0: (1.00, -20.0), 440.0: (0.41, -24.3)}


def metrics(x, f0, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    power = spec ** 2
    total = power.sum() + 1e-20
    m = (freqs >= 700) & (freqs < 1500)
    share = 100.0 * float(power[m].sum()) / total
    peaks = []
    for k in range(1, 40):
        f = f0 * k
        if f > sr / 2 - 200:
            break
        sel = (freqs >= f * 0.97) & (freqs <= f * 1.03)
        if sel.any():
            peaks.append((f, float(spec[sel].max())))
    if not peaks:
        return share, 0.0
    top = max(p for _, p in peaks)
    above = [p for f, p in peaks if f > 700]
    return share, (20 * math.log10(max(above) / top + 1e-20) if above else -99.0)


def score(cfg, lp_track=None, lp_floor=600.0):
    worst_share = 0.0
    worst_db = 0.0
    over = 0
    rows = []
    for f0 in PITCHES:
        x = synthesize_hum(np.full(int(2.0 * SR), f0), SR, cfg, rng=np.random.default_rng(5))
        seg = x[int(0.5 * SR):int(1.6 * SR)]
        if lp_track:
            corner = min(float(cfg.lowpass_hz), max(lp_floor, lp_track * f0))
            sos = s.butter(4, min(0.99, corner / (SR / 2)), btype="low", output="sos")
            seg = s.sosfilt(sos, seg)
        share, loud = metrics(seg, f0)
        ref_share, ref_loud = REFERENCE[f0]
        bad = share > max(1.5 * ref_share, 0.5) or loud < min(ref_loud - 2.0, -17.0)
        over += 1 if bad else 0
        worst_share = max(worst_share, share)
        worst_db = min(worst_db, loud)
        rows.append(f"    {f0:5.0f} Hz: 700-1500 {share:5.2f}% (ref {ref_share:5.2f})  "
                    f"loudest>700 {loud:6.1f} dB (ref {ref_loud:6.1f}){'  <-- worse' if bad else ''}")
    return worst_share, worst_db, over, rows


def main():
    base = HumConfig.from_config()
    base_kw = {k: getattr(base, k) for k in base.__dataclass_fields__}
    grid = list(itertools.product([700.0, 900.0, 1100.0],      # b2
                                  [700.0, 760.0],              # nasal zero
                                  [-26.0, -30.0],              # zero depth
                                  [900.0, 1000.0],             # f2
                                  [None, 1.5, 2.0, 2.6]))      # lp_track
    results = []
    for b2, zero, depth, f2, lp_track in grid:
        over = {**base_kw, "b2_hz": b2, "nasal_zero_hz": zero,
                "nasal_zero_depth_db": depth, "f2_hz": f2,
                "lowpass_hz": 1800.0, "tilt_db_per_oct": -15.0}
        cfg = HumConfig(**over)
        worst_share, worst_db, bad, rows = score(cfg, lp_track=lp_track)
        results.append((bad, worst_share, worst_db, (b2, zero, depth, f2, lp_track), rows))
    results.sort(key=lambda r: (r[0], r[1]))
    for bad, share, db, params, rows in results[:5]:
        print(f"\n=== b2 {params[0]:.0f}, zero {params[1]:.0f} @ {params[2]:.0f} dB, "
              f"f2 {params[3]:.0f}, lp_track {params[4]} -> {bad} pitches worse than the "
              f"reference, worst share {share:.2f}%, worst loudest>700 {db:.1f} dB ===")
        print("\n".join(rows))
    print("\n--- worst three ---")
    for bad, share, db, params, rows in results[-3:]:
        print(f"  {params}: {bad} worse, worst {share:.2f}% / {db:.1f} dB")


if __name__ == "__main__":
    main()

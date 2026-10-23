"""A CLOSED MOUTH, AT EVERY PITCH.

The murmur measures like a hum from 72 to 140 Hz and like a vowel from 196 up
(the 700-1500 Hz share goes 0.12 -> 3.4 -> 19.7 per cent as the note rises),
while the melody the panel's hum is actually asked to sing on the reference
track sits between 200 and 400 Hz. So this builds the plainest possible
closed-mouth reference -- a glottal pulse through ONE low resonance and a steep
rolloff above it, nothing else -- and tunes the murmur's own tract until its
spectrum is at least as closed as that at every pitch.

Scored per pitch on
  * how far the loudest partial above 700 Hz sits under the loudest anywhere,
  * the energy share in 700-1500 Hz and above 3 kHz,
  * whether the envelope is monotone above its peak (a second hump is a vowel).
"""
import itertools
import math

import numpy as np
from scipy import signal as s

from synthesis.hum_voice import HumConfig, synthesize_hum, _rosenberg

SR = 44100
PITCHES = [72.0, 94.0, 140.0, 196.0, 247.0, 280.0, 330.0, 392.0, 440.0]


def reference_voice(f0, sr=SR, seconds=2.0, f1=300.0, bw=90.0, lp=700.0):
    """The plainest closed mouth there is: one resonance, then a shut lip."""
    n = int(seconds * sr)
    phase = np.cumsum(np.full(n, f0)) / sr
    pulse = _rosenberg(np.mod(phase, 1.0), 0.48)
    pulse = np.diff(pulse, prepend=pulse[0])
    pulse /= np.max(np.abs(pulse)) + 1e-12
    # one two-pole resonance
    r = math.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f1 / sr
    a = [1.0, -2 * r * math.cos(theta), r * r]
    y = s.lfilter([1.0 - r], a, pulse)
    sos = s.butter(6, min(0.99, lp / (sr / 2)), btype="low", output="sos")
    y = s.sosfilt(sos, y)
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    return y


def metrics(x, f0, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    power = spec ** 2
    total = power.sum() + 1e-20
    m = (freqs >= 700) & (freqs < 1500)
    share = 100.0 * float(power[m].sum()) / total
    m = freqs >= 3000
    hi = 100.0 * float(power[m].sum()) / total
    peaks = []
    for k in range(1, 40):
        f = f0 * k
        if f > sr / 2 - 200:
            break
        sel = (freqs >= f * 0.97) & (freqs <= f * 1.03)
        if sel.any():
            peaks.append((f, float(spec[sel].max())))
    if not peaks:
        return share, hi, 0.0, False
    top = max(p for _, p in peaks)
    above = [p for f, p in peaks if f > 700]
    loud_above = 20 * math.log10(max(above) / top + 1e-20) if above else -99.0
    db = [20 * math.log10(p / top + 1e-20) for _, p in peaks]
    # monotone above the peak: is there a second hump?
    second_hump = False
    if len(db) >= 3:
        peak_at = int(np.argmax(db))
        tail = db[peak_at:]
        for i in range(1, len(tail) - 1):
            if tail[i] > tail[i - 1] + 1.5 and tail[i] > tail[i + 1] + 1.5 and tail[i] > tail[peak_at] - 25:
                second_hump = True
    return share, hi, loud_above, second_hump


def evaluate(cfg, label):
    worst_share = 0.0
    worst_hi = 0.0
    worst_db = 0.0
    humps = []
    rows = []
    for f0 in PITCHES:
        x = synthesize_hum(np.full(int(2.0 * SR), f0), SR, cfg, rng=np.random.default_rng(5))
        seg = x[int(0.5 * SR):int(1.6 * SR)]
        share, hi, loud, hump = metrics(seg, f0)
        worst_share = max(worst_share, share)
        worst_hi = max(worst_hi, hi)
        worst_db = min(worst_db, loud)
        if hump:
            humps.append(f0)
        rows.append(f"      {f0:5.0f} Hz: 700-1500 {share:5.2f}%  3k+ {hi:5.2f}%  "
                    f"loudest>700 {loud:6.1f} dB{'   SECOND HUMP' if hump else ''}")
    print(f"\n  --- {label} ---")
    print("\n".join(rows))
    print(f"    worst 700-1500 {worst_share:.2f}%   worst 3k+ {worst_hi:.2f}%   "
          f"worst loudest-above-700 {worst_db:.1f} dB   second humps at {humps or 'none'}")
    return worst_share, worst_db


def main():
    print("=== the plain reference closed mouth (F1 300, 6th-order LP at 700) ===")
    for f0 in PITCHES:
        x = reference_voice(f0)
        share, hi, loud, hump = metrics(x, f0)
        print(f"  {f0:5.0f} Hz: 700-1500 {share:5.2f}%  3k+ {hi:5.2f}%  "
              f"loudest>700 {loud:6.1f} dB{'   SECOND HUMP' if hump else ''}")
    print("\n  (that is the target: whatever the murmur does at 72 Hz, it should")
    print("   keep doing at 400 Hz.)")

    base = HumConfig.from_config()
    base_kw = {k: getattr(base, k) for k in base.__dataclass_fields__}
    evaluate(HumConfig(**base_kw), "current")

    candidates = {
        "zero at 760, -26, F2 damped 700": {"nasal_zero_hz": 760.0, "nasal_zero_depth_db": -26.0,
                                            "b2_hz": 700.0},
        "... + record LP 1800": {"nasal_zero_hz": 760.0, "nasal_zero_depth_db": -26.0,
                                 "b2_hz": 700.0, "lowpass_hz": 1800.0},
        "... + F2 down to 900": {"nasal_zero_hz": 760.0, "nasal_zero_depth_db": -26.0,
                                 "b2_hz": 700.0, "lowpass_hz": 1800.0, "f2_hz": 900.0},
        "... + steeper HF shelf": {"nasal_zero_hz": 760.0, "nasal_zero_depth_db": -26.0,
                                   "b2_hz": 700.0, "lowpass_hz": 1800.0, "f2_hz": 900.0,
                                   "hf_shelf_hz": 2600.0, "hf_shelf_db": -24.0},
        "... + HP-normalised hole": {"nasal_zero_hz": 760.0, "nasal_zero_depth_db": -26.0,
                                     "b2_hz": 700.0, "lowpass_hz": 1800.0, "f2_hz": 900.0,
                                     "hf_shelf_hz": 2600.0, "hf_shelf_db": -24.0,
                                     "tilt_db_per_oct": -15.0},
    }
    for label, over in candidates.items():
        evaluate(HumConfig(**{**base_kw, **over}), label)


if __name__ == "__main__":
    main()

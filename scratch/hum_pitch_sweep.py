"""THE MURMUR AT EVERY PITCH, measured on the voice alone.

The panel's hum at 16 s of the reference track came back with 3.6 % of its
energy between 700 Hz and 1.5 kHz and partials there only 15 dB under its
loudest -- a formant pair, which is what an ear hears as a vowel, and the
complaint is exactly "it sounds like OH". The same voice at 32 s measured 0.13 %
in that band. So: render the murmur by itself, constant pitch, and find out
whether the closed mouth only closes at some pitches.

For each pitch it prints
  * the partials above 700 Hz against the loudest partial,
  * the energy share in the bands that decide "hum" against "vowel",
  * the tract's own response at the frequencies where the peaks landed, so a
    peak can be attributed to the tract or to the source.
"""
import numpy as np
import soundfile as sf

from synthesis.hum_voice import HumConfig, synthesize_hum, _resonator, _antiresonance, _shelf

SR = 44100
PITCHES = [72.0, 94.0, 110.0, 140.0, 196.0, 247.0, 280.0, 330.0, 392.0]


def bands(x, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    p = spec / (spec.sum() + 1e-20)
    out = {}
    for lo, hi in ((0, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        out[(lo, hi)] = 100.0 * float(p[m].sum())
    centroid = float((freqs * spec).sum() / (spec.sum() + 1e-20))
    return out, centroid


def partial_table(x, f0, sr=SR, n=20):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    rows = []
    for k in range(1, n + 1):
        f = f0 * k
        if f > sr / 2 - 200:
            break
        m = (freqs >= f * 0.97) & (freqs <= f * 1.03)
        rows.append((k, f, float(spec[m].max()) if m.any() else 0.0))
    if not rows:
        return rows, 0.0
    top = max(p for _, _, p in rows)
    return rows, top


def tract_response_at(cfg, f0, sr=SR, probe=None):
    """The murmur's tract response, stage by stage, from the voice's own helpers."""
    from scipy.signal import sosfreqz
    tracking = float(np.clip(cfg.formant_tracking, 0.0, 0.6))
    scale = float(np.clip((f0 / 200.0) ** tracking, 0.7, 1.35))
    stages = [("res1", _resonator(cfg.f1_hz * scale, cfg.b1_hz, sr)),
              ("res2", _resonator(cfg.f2_hz * scale, cfg.b2_hz, sr)),
              ("res3", _resonator(cfg.f3_hz * scale, cfg.b3_hz, sr)),
              ("res4", _resonator(cfg.f4_hz * scale, cfg.b4_hz, sr)),
              ("nasal", _antiresonance(cfg.nasal_zero_hz * scale, cfg.nasal_zero_q,
                                       cfg.nasal_zero_depth_db, sr)),
              ("body", _shelf(cfg.body_hz * scale, cfg.body_gain_db, sr, q=0.8, high=False)),
              ("tilt", _shelf(cfg.tilt_hz * scale, cfg.tilt_db_per_oct, sr, q=0.7, high=True)),
              ("hf", _shelf(cfg.hf_shelf_hz * scale, cfg.hf_shelf_db, sr, q=0.7, high=True))]
    probe = probe if probe is not None else np.array([100.0, 250.0, 400.0, 600.0, 800.0,
                                                      1000.0, 1200.0, 1500.0, 2000.0, 2500.0])
    total = np.ones_like(probe)
    for name, sos in stages:
        try:
            w, h = sosfreqz(sos, worN=probe, fs=sr)
        except ValueError:
            from scipy.signal import freqz
            b, a = (sos if isinstance(sos, tuple) else (np.asarray(sos)[0][:3], np.asarray(sos)[0][3:]))
            w, h = freqz(b, a, worN=probe, fs=sr)
        total = total * np.abs(h)
    total_db = 20 * np.log10(total + 1e-20)
    return scale, {float(f): float(d) for f, d in zip(probe, total_db)}


def main():
    cfg = HumConfig.from_config()
    print("cfg:", {k: getattr(cfg, k) for k in
                   ("f1_hz", "b1_hz", "f2_hz", "b2_hz", "nasal_zero_hz",
                    "nasal_zero_depth_db", "nasal_zero_q", "formant_tracking",
                    "tilt_hz", "tilt_db_per_oct", "lowpass_hz", "lowpass_order",
                    "hf_shelf_hz", "hf_shelf_db", "breath_db", "breath_cap_db")})
    worst = 0.0
    for f0 in PITCHES:
        n = int(2.5 * SR)
        curve = np.full(n, f0, dtype=np.float64)
        x = synthesize_hum(curve, SR, cfg, rng=np.random.default_rng(7))
        seg = x[int(0.6 * SR):int(2.1 * SR)]
        b, cen = bands(seg)
        rows, top = partial_table(seg, f0)
        above = [(k, f, 20 * np.log10((p + 1e-12) / (top + 1e-12))) for k, f, p in rows if f > 700]
        loudest_above = max((d for _, _, d in above), default=-99.0)
        worst = max(worst, b[(700, 1500)])
        print(f"\nf0 {f0:6.1f} Hz  centroid {cen:6.0f} Hz  700-1500 Hz share {b[(700, 1500)]:6.2f}%  "
              f"3000+ {b[(3000, 8000)]:5.2f}%   0-300 {b[(0, 300)]:5.1f}%  300-700 {b[(300, 700)]:5.1f}%")
        print("   loudest partial above 700 Hz: "
              + (f"{loudest_above:5.1f} dB under the loudest anywhere" if above else "none above 700 Hz"))
        print("   partials: " + " ".join(f"H{k}:{f:.0f}:{d:5.1f}" for k, f, d in rows))
        scale = float(np.clip((f0 / 200.0) ** float(np.clip(cfg.formant_tracking, 0.0, 0.6)), 0.7, 1.35))
        print(f"   tract_scale {scale:.3f} (F1 {cfg.f1_hz * scale:.0f}, F2 {cfg.f2_hz * scale:.0f}, "
              f"zero {cfg.nasal_zero_hz * scale:.0f})")
    print(f"\nWORST 700-1500 Hz share across the sweep: {worst:.2f}%")


if __name__ == "__main__":
    main()

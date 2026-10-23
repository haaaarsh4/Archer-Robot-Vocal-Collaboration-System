"""Two questions, one pass:

A. WHAT REGISTER IS THE TRACK ACTUALLY SUNG IN, and what does the hum's own
   melody tracker say about it? The panel's hum chunk at 16 s came back with a
   second partial group (an /o/) while the very same voice at 32 s was a closed
   murmur, and the tracked melody over that chunk swings 50-400 Hz, so the
   register the hum is asked to sing is the first thing to check.

B. CAN THE MURMUR'S TRACT BE TUNED SO IT STAYS CLOSED AT EVERY PITCH? Candidates
   are rendered at a pitch sweep and scored on the two numbers that decide
   "hum" against "vowel": the energy share between 700 Hz and 1.5 kHz, and how
   far the loudest partial above 700 Hz sits under the loudest partial anywhere.
"""
import numpy as np
import soundfile as sf

from synthesis.hum_voice import HumConfig, track_melody, synthesize_hum

SR = 44100


def band_shares(x, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    p = spec / (spec.sum() + 1e-20)
    out = {}
    for lo, hi in ((0, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        out[(lo, hi)] = 100.0 * float(p[m].sum())
    return out, float((freqs * spec).sum() / (spec.sum() + 1e-20))


def near_partials(x, f0, sr=SR, n=30):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    vals = []
    for k in range(1, n + 1):
        f = f0 * k
        if f > sr / 2 - 200:
            break
        m = (freqs >= f * 0.97) & (freqs <= f * 1.03)
        if m.any():
            vals.append((f, float(spec[m].max())))
    if not vals:
        return 0.0, []
    top = max(v for _, v in vals)
    return 20 * np.log10(top + 1e-20), [(f, 20 * np.log10(v / (top + 1e-20) + 1e-20)) for f, v in vals]


def register_report():
    audio, sr = sf.read("frontend/assets/audio/hummingbird.mp3", dtype="float32", always_2d=False)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    cfg = HumConfig.from_config()
    print("=== A. the tracked melody over the whole track ===")
    for start, end in ((0.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 49.0)):
        chunk = audio[int(start * SR):int(end * SR)]
        mel = track_melody(chunk, SR, cfg)
        v = mel.voiced
        hz = mel.f0_hz[v]
        if not hz.size:
            print(f"  {start:4.0f}-{end:4.0f}s: nothing voiced")
            continue
        print(f"  {start:4.0f}-{end:4.0f}s: {100 * v.mean():3.0f}% voiced, "
              f"p10 {np.percentile(hz, 10):6.1f}  median {np.median(hz):6.1f}  "
              f"p90 {np.percentile(hz, 90):6.1f}  min {hz.min():6.1f}  max {hz.max():6.1f} Hz")
    whole = track_melody(audio, SR, cfg)
    v = whole.voiced
    hz = whole.f0_hz[v]
    print(f"  whole track: {100 * v.mean():.0f}% voiced, median {np.median(hz):.1f} Hz")
    hist, edges = np.histogram(np.log2(hz), bins=24, range=(np.log2(50), np.log2(500)))
    print("  pitch histogram (log-spaced, frames per semitone-ish bin):")
    for h, e in zip(hist, edges):
        print(f"    {2 ** e:6.1f} Hz  {'#' * int(60 * h / max(hist.max(), 1))} {h}")


CANDIDATES = {
    "current": {},
    "wide F2": {"b2_hz": 700.0},
    "deep low zero": {"nasal_zero_hz": 820.0, "nasal_zero_depth_db": -26.0},
    "wide F2 + deep low zero": {"b2_hz": 700.0, "nasal_zero_hz": 820.0, "nasal_zero_depth_db": -26.0},
    "wide F2 + deep low zero + tilt": {"b2_hz": 700.0, "nasal_zero_hz": 820.0,
                                       "nasal_zero_depth_db": -26.0, "tilt_db_per_oct": -15.0},
    "wide F2 + deep low zero + tilt + lp": {"b2_hz": 700.0, "nasal_zero_hz": 820.0,
                                            "nasal_zero_depth_db": -26.0, "tilt_db_per_oct": -15.0,
                                            "lowpass_hz": 1800.0},
}
PITCHES = [72.0, 94.0, 140.0, 196.0, 247.0, 280.0, 330.0, 392.0]


def tract_report():
    base = HumConfig.from_config()
    print("\n=== B. the murmur at every pitch, per candidate tract ===")
    for name, overrides in CANDIDATES.items():
        cfg = HumConfig(**{**{k: getattr(base, k) for k in
                              base.__dataclass_fields__}, **overrides})
        worst_share, worst_db, lowest_share = 0.0, 0.0, 0.0
        lines = []
        for f0 in PITCHES:
            n = int(2.5 * SR)
            x = synthesize_hum(np.full(n, f0), SR, cfg, rng=np.random.default_rng(11))
            seg = x[int(0.6 * SR):int(2.1 * SR)]
            b, cen = band_shares(seg)
            _, parts = near_partials(seg, f0)
            above = [d for f, d in parts if f > 700]
            loud = max(above) if above else -99.0
            worst_share = max(worst_share, b[(700, 1500)])
            lowest_share = max(lowest_share, b[(0, 300)])
            worst_db = min(worst_db, loud) if above else worst_db
            lines.append(f"    {f0:5.0f} Hz: 700-1500 {b[(700, 1500)]:5.2f}%  "
                         f"centroid {cen:5.0f}  loudest>700 {loud:6.1f} dB")
        print(f"\n  --- {name} {overrides} ---")
        print("\n".join(lines))
        print(f"    worst 700-1500 share {worst_share:.2f}%, worst loudest-above-700 "
              f"{worst_db:.1f} dB, best low-band share {lowest_share:.1f}%")


if __name__ == "__main__":
    register_report()
    tract_report()

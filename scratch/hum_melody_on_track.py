"""WHERE THE BRIGHTNESS COMES FROM, on the murmur's own trail.

The panel's hum chunk at 16 s of the reference track came back with 3.6 % of its
energy between 700 Hz and 1.5 kHz (the same voice at 32 s: 0.13 %). The murmur
rendered by itself, constant pitch, is dark at every pitch up to ~190 Hz and
gets a second partial group above that. So this walks the pipeline's own steps
on that exact chunk -- the tracked melody, then the render, then the panel's own
vocal-fold vibration -- and measures each one, to see which step brightens it.
"""
import numpy as np
import soundfile as sf

from synthesis.hum_voice import HumConfig, track_melody, synthesize_hum

SR = 44100


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


def loudest_above_700(x, f0, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    best = 0.0
    for k in range(1, 60):
        f = f0 * k
        if f > sr / 2 - 200:
            break
        m = (freqs >= f * 0.97) & (freqs <= f * 1.03)
        if m.any():
            best = max(best, float(spec[m].max()))
    return 20 * np.log10(best / (spec.max() + 1e-20) + 1e-20)


def report(name, x, f0):
    b, cen = bands(x)
    print(f"  {name}: centroid {cen:6.0f} Hz  700-1500 {b[(700, 1500)]:6.2f}%  "
          f"300-700 {b[(300, 700)]:5.1f}%  0-300 {b[(0, 300)]:5.1f}%  "
          f"loudest partial >700Hz {loudest_above_700(x, f0):6.1f} dB under the loudest")


def main():
    track = "frontend/assets/audio/hummingbird.mp3"
    audio, sr = sf.read(track, dtype="float32", always_2d=False)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    cfg = HumConfig.from_config()

    for start in (16.0, 24.0, 32.0):
        chunk = audio[int(start * SR):int((start + 8.0) * SR)]
        mel = track_melody(chunk, SR, cfg)
        v = mel.voiced
        f0 = mel.f0_hz
        voiced_hz = f0[v] if v.any() else np.array([])
        print(f"\n=== chunk @ {start:.0f}s ===")
        if voiced_hz.size:
            print(f"  tracked melody: {100 * v.mean():.0f}% voiced, median {np.median(voiced_hz):.1f} Hz, "
                  f"p10 {np.percentile(voiced_hz, 10):.1f}, p90 {np.percentile(voiced_hz, 90):.1f}, "
                  f"min {voiced_hz.min():.1f}, max {voiced_hz.max():.1f}, hop {mel.hop_s:.3f}s")
            # how often does it jump a whole octave between neighbouring frames?
            jumps = 0
            for i in range(1, len(f0)):
                if v[i] and v[i - 1] and f0[i] > 0 and f0[i - 1] > 0:
                    r = f0[i] / f0[i - 1]
                    if r > 1.7 or r < 0.59:
                        jumps += 1
            print(f"  octave-or-worse jumps between neighbouring voiced frames: {jumps} "
                  f"of {int(v.sum())} voiced frames")
        frames = max(1, int(np.ceil(chunk.size / (mel.hop_s * SR))))
        idx = np.clip(np.arange(frames), 0, f0.size - 1)
        curve = f0[idx]
        rms = mel.rms[idx]
        vv = v[idx].astype(np.float64)
        med = float(np.median(curve[vv > 0.5])) if np.any(vv > 0.5) else 200.0

        y = synthesize_hum(curve, SR, cfg, rms_curve=rms, hop_s=mel.hop_s, voiced=vv,
                           rng=np.random.default_rng(4242))
        report("pipeline render (as _render_hum_ensemble does it)", y, med)

        # the page's vocal-fold-vibration slider, as the offline route applies it
        import server
        y2 = server._apply_vocal_vibration_amount(y.copy(), SR, 45.0)
        report("... after the panel's vibrato (45%)", y2, med)

        # the same melody rendered at its own median pitch, held flat
        flat = np.full_like(curve, med)
        y3 = synthesize_hum(flat, SR, cfg, rms_curve=rms, hop_s=mel.hop_s,
                            voiced=np.ones_like(vv), rng=np.random.default_rng(3))
        report(f"... the same length at a FLAT {med:.0f} Hz", y3, med)


if __name__ == "__main__":
    main()

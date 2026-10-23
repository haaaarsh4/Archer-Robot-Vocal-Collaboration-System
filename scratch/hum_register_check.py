"""Does the hum now follow ONE line, and does it sound like a murmur?

track_melody per window (the settled register, and how far the line moves), then
the hum rendered from that melody the way _render_hum_ensemble does it, measured
on the two numbers that decide hum against vowel.
"""
import numpy as np
import soundfile as sf

from synthesis.hum_voice import HumConfig, track_melody, synthesize_hum

SR = 44100


def shares(x, sr=SR):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    p = spec / (spec.sum() + 1e-20)
    out = {}
    for lo, hi in ((0, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        out[(lo, hi)] = 100.0 * float(p[m].sum())
    return out, float((freqs * spec).sum() / (spec.sum() + 1e-20))


def main():
    audio, sr = sf.read("frontend/assets/audio/hummingbird.mp3", dtype="float32", always_2d=False)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    cfg = HumConfig.from_config()

    for start, end in ((0.0, 8.0), (16.0, 24.0), (32.0, 40.0)):
        chunk = np.ascontiguousarray(audio[int(start * SR):int(end * SR)], dtype=np.float32)
        mel = track_melody(chunk, SR, cfg)
        v = mel.voiced
        hz = mel.f0_hz[v]
        print(f"\n=== {start:.0f}-{end:.0f}s ===")
        if not hz.size:
            print("  nothing voiced at all")
            continue
        print(f"  line: {100 * v.mean():3.0f}% voiced, p10 {np.percentile(hz, 10):6.1f}  "
              f"median {np.median(hz):6.1f}  p90 {np.percentile(hz, 90):6.1f}  "
              f"min {hz.min():6.1f}  max {hz.max():6.1f} Hz")
        # how many semitones does one frame move to the next? A hum's line moves
        # by a note at a time; the old one moved by octaves.
        steps = []
        for i in range(1, len(mel.f0_hz)):
            if v[i] and v[i - 1] and mel.f0_hz[i] > 0 and mel.f0_hz[i - 1] > 0:
                steps.append(abs(12 * np.log2(mel.f0_hz[i] / mel.f0_hz[i - 1])))
        if steps:
            steps = np.array(steps)
            print(f"  frame-to-frame move: median {np.median(steps):4.2f} semitones, "
                  f"p90 {np.percentile(steps, 90):5.2f}, {100 * np.mean(steps > 6):4.1f}% of frames "
                  f"move more than a tritone")

        frames = max(1, int(np.ceil(chunk.size / (mel.hop_s * SR))))
        idx = np.clip(np.arange(frames), 0, mel.f0_hz.size - 1)
        y = synthesize_hum(mel.f0_hz[idx], SR, cfg, rms_curve=mel.rms[idx], hop_s=mel.hop_s,
                           voiced=mel.voiced[idx].astype(np.float64),
                           rng=np.random.default_rng(4242))
        b, cen = shares(y)
        print(f"  hum: centroid {cen:5.0f} Hz  700-1500 {b[(700, 1500)]:5.2f}%  "
              f"300-700 {b[(300, 700)]:5.1f}%  0-300 {b[(0, 300)]:5.1f}%  "
              f"rms {float(np.sqrt(np.mean(y ** 2))):.3f}")


if __name__ == "__main__":
    main()

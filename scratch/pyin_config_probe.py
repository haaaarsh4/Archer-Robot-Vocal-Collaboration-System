"""WHY THE MELODY READS 50-400 Hz.

track_melody's own output, per eight-second window of the reference track, is
p10 = 50.0 Hz and p90 = 300-400 Hz -- half the voiced frames sitting on the
tracker's own floor and the rest scattered an octave or two up. A hum asked to
follow that wails. This runs pyin itself under the pipeline's settings and
under saner ones, on the same material, and prints what each reads.
"""
import numpy as np
import soundfile as sf
import librosa

SR = 44100


def stats(f0, voiced, label):
    v = voiced & ~np.isnan(f0) & (f0 > 0)
    if not v.any():
        print(f"  {label}: nothing voiced")
        return
    hz = f0[v]
    at_floor = 100.0 * float(np.mean(hz < 55.0))
    print(f"  {label}: {100 * v.mean():3.0f}% voiced | p10 {np.percentile(hz, 10):6.1f} "
          f"med {np.median(hz):6.1f} p90 {np.percentile(hz, 90):6.1f} "
          f"max {hz.max():6.1f} Hz | {at_floor:4.0f}% of voiced frames under 55 Hz")


def run(x, sr, frame_length, hop, fmin, fmax, label):
    f0, flag, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=sr,
                               frame_length=frame_length, hop_length=hop,
                               center=True, fill_na=0.0)
    stats(np.nan_to_num(f0, nan=0.0), np.asarray(flag, dtype=bool), label)


def main():
    audio, sr = sf.read("frontend/assets/audio/hummingbird.mp3", dtype="float32", always_2d=False)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    harmonic, _ = librosa.effects.hpss(audio, margin=1.6)

    for name, x in (("mix", audio), ("harmonic part", harmonic)):
        for start, end in ((0.0, 8.0), (16.0, 24.0), (32.0, 40.0)):
            seg = np.ascontiguousarray(x[int(start * SR):int(end * SR)], dtype=np.float32)
            print(f"\n=== {name} {start:.0f}-{end:.0f}s ===")
            # the pipeline's own settings: 16 kHz analysis, pyin_frame_size scaled by 0.363
            low = librosa.resample(seg, orig_sr=SR, target_sr=16000)
            run(low, 16000, 1487, 93, 50.0, 1100.0, "pipeline (16 kHz, 1487/93, fmin 50)")
            run(low, 16000, 2048, 256, 50.0, 1100.0, "16 kHz, frame 2048 hop 256, fmin 50")
            run(seg, SR, 2048, 256, 60.0, 1100.0, "audio rate, frame 2048 hop 256, fmin 60")
            run(seg, SR, 4096, 256, 70.0, 1000.0, "audio rate, frame 4096 hop 256, fmin 70")


if __name__ == "__main__":
    main()

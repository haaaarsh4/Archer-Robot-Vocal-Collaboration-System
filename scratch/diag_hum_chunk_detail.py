"""One live hum chunk: what the page gets back, saved and taken apart.

Writes /tmp/hum_chunk_<start>s.wav for the chunk the panel actually plays in
Live Neural + Humming, then prints
  * the harmonic partials under the loudest one (a closed murmur is a low
    harmonic series with a HOLE where its antiformant sits; a vowel has a
    formant PEAK there),
  * its pitch, read with the project's own tracker,
  * its loudness over time (does the line arrive as one continuous note or as
    a row of syllables?),
  * the raw difference against the DSP engine's hum of the same seconds.
"""
import io
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "scratch")
from diag_live_hum_chunks import load, measure, post_multipart, encode_wav, SR, BASE  # noqa: E402


def track_melody_pitches(x, sr):
    """The project's own pitch reading of a stretch of audio (librosa pyin)."""
    import librosa
    f0, voiced, _ = librosa.pyin(x, fmin=60.0, fmax=800.0, sr=sr, frame_length=2048, hop_length=256)
    v = voiced & ~np.isnan(f0)
    if not v.any():
        return np.array([])
    return f0[v]


def partials(x, sr, f0, n=16):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    out = []
    for k in range(1, n + 1):
        f = f0 * k
        if f > sr / 2 - 100:
            break
        lo, hi = f * 0.97, f * 1.03
        m = (freqs >= lo) & (freqs <= hi)
        out.append((k, f, float(spec[m].max()) if m.any() else 0.0))
    return out


def report(name, x, sr=SR):
    print(f"\n--- {name} ---")
    measure(name, x)
    f0s = track_melody_pitches(x, sr)
    if not f0s.size:
        print("       pitch: no voiced frames at all")
        return
    med = float(np.median(f0s))
    print(f"       pitch (pyin): median {med:6.1f} Hz  p10 {np.percentile(f0s, 10):6.1f}  "
          f"p90 {np.percentile(f0s, 90):6.1f}  ({100 * len(f0s) / (len(x) / 256):.0f}% of frames voiced)")
    ps = partials(x, sr, med)
    if ps:
        top = max(p for _, _, p in ps) or 1.0
        print("       partials (dB under the loudest): " + "  ".join(
            f"H{k}:{20 * np.log10((p + 1e-12) / (top + 1e-12)):5.1f}" for k, _, p in ps))
    hop = int(0.25 * sr)
    env = np.array([float(np.sqrt(np.mean(np.square(x[i:i + hop], dtype=np.float64))))
                    for i in range(0, max(1, len(x) - hop), hop)])
    print("       rms every 0.25 s: " + " ".join(f"{v:.3f}" for v in env))


def main():
    track = "frontend/assets/audio/hummingbird.mp3"
    audio = load(track)
    for start in (16.0, 32.0):
        slice_ = audio[int(start * SR):int((start + 8) * SR)]
        fields = {"voice_index": "0", "transpose_semitones": "0", "pad_seconds": "0.3",
                  "instruments_enabled": "false", "vibrato_amount": "45", "mode": "solo_humming"}
        data, headers = post_multipart(f"{BASE}/api/neural/convert-track", fields,
                                       {"file": ("chunk.wav", encode_wav(slice_))})
        out, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        out = np.asarray(out, dtype=np.float32)
        sf.write(f"/tmp/hum_chunk_{int(start)}s.wav", out, sr)
        print(f"\n===== live hum chunk @ {start:.0f}s "
              f"(X-Neural-Applied={headers.get('X-Neural-Applied')}) =====")
        report(f"the mix sent ({start:.0f}s)", slice_)
        report(f"what the page plays ({start:.0f}s)", out, sr)


if __name__ == "__main__":
    main()

"""The saved live-hum chunks, peak by peak, against the murmur's own tract.

Answers one question: the hum at 16 s came back with 8 % of its energy between
700 Hz and 1.5 kHz while the same voice at 32 s came back with 0.13 %. Is there
a FORMANTS peak in the 16 s one (that is what an ear hears as a vowel, and this
whole complaint is "it sounds like OH") and where does it come from -- the tract
the hum is supposed to be sung through, or something added on top of it?
"""
import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, "synthesis")


def peaks(x, sr, n=14, floor_db=-70.0):
    n_fft = 1 << int(np.ceil(np.log2(len(x))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    db = 20 * np.log10(spec / (spec.max() + 1e-20) + 1e-20)
    out = []
    for i in range(2, len(db) - 2):
        if db[i] > floor_db and db[i] >= db[i - 1] and db[i] >= db[i + 1] and \
           db[i] > db[i - 2] and db[i] > db[i + 2]:
            out.append((float(freqs[i]), float(db[i])))
    out.sort(key=lambda p: -p[1])
    merged = []
    for f, d in out:
        if all(abs(f - g) > 60 for g, _ in merged):
            merged.append((f, d))
        if len(merged) >= n:
            break
    return sorted(merged)


def harm_spacing(pks):
    """The spacing of the low peaks = the hum's real fundamental."""
    low = [f for f, d in pks if f < 900]
    if len(low) < 2:
        return 0.0, []
    diffs = np.diff(sorted(low))
    return float(np.median(diffs)), sorted(low)


def tract_response(sr=44100):
    """The murmur's own spectral envelope, straight out of the voice's code."""
    from scipy.signal import sosfreqz
    from hum_voice import HumConfig, _apply_sos, _resonator, _antiresonance, _peaking, _shelf
    cfg = HumConfig.from_config()
    stages = []
    for name in dir(cfg):
        pass
    # The tract is built in synthesize_hum from these cfg fields; rebuild the
    # same chain by hand from the code's own helpers (see synthesize_hum).
    trace = []
    for attr in ("f1_hz", "f1_bw_hz", "f2_hz", "f2_bw_hz", "f3_hz", "f3_bw_hz",
                 "f4_hz", "f4_bw_hz", "zero_hz", "zero_q", "zero_depth_db",
                 "tilt_hz", "tilt_db_per_oct", "shelf_hz", "shelf_db", "rolloff_hz"):
        if hasattr(cfg, attr):
            trace.append(f"{attr}={getattr(cfg, attr)}")
    return trace


def main():
    for path in ("/tmp/hum_chunk_16s.wav", "/tmp/hum_chunk_32s.wav"):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        x = np.asarray(x, np.float32)
        # the loudest 1.5 s, so a chunk with silence at its head is not measured on silence
        win = int(1.5 * sr)
        best, at = 0.0, 0
        for i in range(0, max(1, len(x) - win), sr // 4):
            r = float(np.sqrt(np.mean(np.square(x[i:i + win], dtype=np.float64))))
            if r > best:
                best, at = r, i
        seg = x[at:at + win]
        pks = peaks(seg, sr)
        spacing, lows = harm_spacing(pks)
        print(f"\n=== {path} (loudest 1.5 s: {best:.3f} rms at {at / sr:.1f}s) ===")
        print("  strongest spectral peaks (Hz @ dB under the loudest):")
        for f, d in pks:
            print(f"    {f:8.1f} Hz  {d:6.1f} dB")
        print(f"  lowest peaks: {[round(f) for f in lows]}  median spacing {spacing:.1f} Hz")
        tot = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
        p = tot / (tot.sum() + 1e-20)
        for lo, hi in ((0, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 8000)):
            m = (fr >= lo) & (fr < hi)
            print(f"    {lo:5d}-{hi:5d} Hz: {100 * p[m].sum():6.2f}%")

    print("\n=== the murmur's own tract parameters (what it is supposed to be) ===")
    for line in tract_response():
        print("   ", line)


if __name__ == "__main__":
    main()

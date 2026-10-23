"""Which measurement tells a drum from a voice on the user's own track?

The track's first sixteen seconds are drums alone (measured: energy nearly all
below 300 Hz, the tracker reporting 100% voiced at ~83 Hz), and the singing
starts at 16 s. Any rule that is going to stop the hum following a drum has to
separate those two stretches, so the candidate measurements are printed per two
seconds, side by side, and the one that separates them is the one to use.

Run:  .venv/bin/python scratch/voice_vs_drum.py "frontend/assets/audio/hummingbird.mp3"
"""
from __future__ import annotations

import sys

import numpy as np
import librosa
import soundfile as sf

WIN = 2048
HOP = 512
SR = 22050


def vox_features(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    stft = np.abs(librosa.stft(y, n_fft=WIN, hop_length=HOP)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=WIN)
    power = stft ** 2
    f0, voiced, _ = librosa.pyin(y, fmin=50.0, fmax=1100.0, sr=sr,
                                 frame_length=WIN, hop_length=HOP)
    n = min(power.shape[1], f0.size)
    flatness = librosa.feature.spectral_flatness(S=stft[:, :n])[0]
    band_lo = (freqs >= 60) & (freqs < 300)
    band_hi = (freqs >= 300) & (freqs < 1500)
    band_vhi = (freqs >= 1500) & (freqs < 4000)
    ratio = power[band_hi, :n].sum(axis=0) / np.maximum(power[band_lo, :n].sum(axis=0), 1e-12)
    vhi = power[band_vhi, :n].sum(axis=0) / np.maximum(power.sum(axis=0)[:n], 1e-12)
    # comb support: how many of harmonics 2..8 stand 20x above the frame's own
    # mean bin power (the rule hum_voice._harmonic_fit uses, see _COMB_PEAK)
    comb = np.zeros(n)
    slope = np.zeros(n)
    for i in range(n):
        hz = f0[i]
        if not np.isfinite(hz):
            comb[i] = np.nan
            slope[i] = np.nan
            continue
        col = power[:, i]
        mean = float(col.mean())
        levels, ks = [], []
        for k in range(1, 9):
            f = k * hz
            if f >= sr / 2 - 200:
                break
            j = int(np.argmin(np.abs(freqs - f)))
            lo, hi = max(0, j - 1), min(col.size, j + 2)
            lv = float(col[lo:hi].max())
            levels.append(20 * np.log10(lv / mean))
            ks.append(k)
        if len(levels) > 1:
            comb[i] = float(np.mean(np.asarray(levels[1:]) > 20.0))
            a = np.asarray(ks, dtype=float)
            b = np.asarray(levels)
            slope[i] = float(np.polyfit(np.log2(a), b, 1)[0])
        else:
            comb[i] = np.nan
            slope[i] = np.nan
    return {"f0": f0[:n], "voiced": voiced[:n], "flat": flatness, "ratio": ratio,
            "vhi": vhi, "comb": comb, "slope": slope}


def main(path: str) -> int:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y22 = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    f = vox_features(y22, SR)
    hop_s = HOP / SR
    print(f"{path}: {len(y) / sr:.1f} s")
    print("\n  window    f0Hz   voiced   combsup  slope(dB/oct)  300-1500/60-300  >1500Hz share  flatness")
    for t0 in np.arange(0, len(y) / sr - 2.0, 2.0):
        a, b = int(t0 / hop_s), int((t0 + 2.0) / hop_s)
        def med(key):
            v = f[key][a:b]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        print(f"  {t0:4.0f}-{t0 + 2:4.0f}s {med('f0'):7.1f} "
              f"{float(np.nanmean(f['voiced'][a:b])) * 100:7.1f}%  {med('comb'):7.2f}  "
              f"{med('slope'):12.1f}  {med('ratio'):15.2f}  {med('vhi'):13.4f}  {med('flat'):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else
                          "frontend/assets/audio/hummingbird.mp3"))

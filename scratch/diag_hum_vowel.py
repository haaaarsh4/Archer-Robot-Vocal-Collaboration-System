"""Is the humming the listener hears a murmur, or a vowel in the robot's timbre?

The page does not play the DSP hum when the neural sidecar is up: on every note
onset the NeuralNoteBridge swaps in a server take and ducks the local voice to
12%. So the sound in Humming mode is the CONVERTED hum, and the question "does
it sound like mmm or ohhh" has to be asked of that audio, not of the DSP render.

Measured here, for the DSP murmur (the reference), the converted hum the page
actually plays, and real recorded vowel takes:

  * the partial table -- every harmonic in dB below the loudest one, which is
    what the ear reads as a vowel when the second partial is nearly as strong as
    the first (that is /o/), and
  * LPC formants -- a murmur's first formant sits near 260 Hz; a sung /o/ puts
    it at 400-500 and an /a/ higher still.

Run:  .venv/bin/python scratch/diag_hum_vowel.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from config.config_loader import get_config  # noqa: E402
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402

PITCHES = [147.0, 220.0, 330.0]


def mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y.mean(axis=1) if y.ndim > 1 else y


def steady(y: np.ndarray, sr: int, seconds: float = 0.5) -> np.ndarray:
    y = mono(y)
    n = int(sr * seconds)
    if y.size <= n:
        return y
    start = min(max(0, int(y.size * 0.45)), y.size - n)
    return y[start:start + n]


def track_f0(y: np.ndarray, sr: int, near: float) -> float:
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(y, np.float32), fmin=near * 0.5, fmax=near * 2.0,
                                 sr=sr, frame_length=8192, hop_length=1024)
    ok = (~np.isnan(f0)) & np.asarray(voiced, bool)
    return float(np.median(f0[ok])) if np.any(ok) else near


def partials(y: np.ndarray, sr: int, f0: float, n_harm: int = 16) -> np.ndarray:
    x = steady(y, sr)
    win = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * win)) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    vals = []
    for k in range(1, n_harm + 1):
        f = k * f0
        if f >= sr / 2 - 200:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.96, f * 1.04])
        vals.append(20.0 * np.log10(float(mag[max(lo, 0):max(hi, lo + 1)].max())))
    v = np.asarray(vals)
    return v - v.max()


def formants(y: np.ndarray, sr: int) -> list[float]:
    """LPC formants, averaged over short frames, robust to a singular solve."""
    import librosa
    order = 2 + int(sr / 1000)
    x = np.asarray(mono(y), np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])          # pre-emphasis
    win = int(0.04 * sr)
    hop = int(0.02 * sr)
    found: list[list[float]] = []
    for start in range(0, max(1, x.size - win), hop):
        frame = x[start:start + win]
        if frame.size < win or float(np.abs(frame).max()) < 1e-4:
            continue
        frame = frame * np.hanning(frame.size)
        try:
            a = librosa.lpc(frame, order=order)
            if not np.all(np.isfinite(a)):
                continue
            roots = [r for r in np.roots(a) if np.imag(r) > 0.01 and r.real > 0]
            fs = sorted(2.0 * np.angle(r) * sr / (2 * np.pi) for r in roots)
            if fs:
                found.append(fs[:3])
        except Exception:  # noqa: BLE001
            continue
    if not found:
        return []
    arr = np.asarray([f + [np.nan] * (3 - len(f)) for f in found], dtype=np.float64)
    return [round(float(v), 0) if np.isfinite(v) else -1.0 for v in np.nanmedian(arr, axis=0)]


def report(name: str, y: np.ndarray, sr: int, f0: float | None = None) -> None:
    f0 = f0 or track_f0(y, sr, 220.0)
    p = partials(y, sr, f0)
    fs = formants(y, sr)
    above = [i for i, k in enumerate(range(1, len(p) + 1)) if k * f0 > 700.0]
    gap = float(max(p) - max(p[above])) if above else 99.0
    # share of the note's harmonic energy sitting above 700 Hz
    power = 10.0 ** (p / 10.0)
    share = float(power[above].sum() / power.sum()) if above else 0.0
    print(f"  {name:32s} f0 {f0:6.1f}  H2 {p[1]:6.1f}  H3 {p[2]:6.1f}  H4 {p[3]:6.1f} dB  "
          f"| top>700Hz {gap:5.1f} dB under  | >700Hz share {share * 100:5.2f}%  | "
          f"LPC formants {fs}")


import os

PORT = os.environ.get("ARCHER_PORT", "8000")


def neural_hum(hz: float, seconds: float = 1.8) -> tuple[np.ndarray, int]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/neural/render",
        data=json.dumps({"target_hz": hz, "vocable": "aah", "duration_s": seconds,
                         "mode": "solo_humming", "texture": "solo",
                         "neural_voice_count": 1}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        applied = resp.headers.get("X-Neural-Applied")
        raw = resp.read()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return mono(y), sr


def main() -> int:
    cfg = get_config()
    sr = int(cfg["audio"]["sample_rate"])
    hum_cfg = HumConfig.from_config()
    n = int(1.8 * sr)

    print("=== the murmur (server DSP), the converted hum the page plays, and real takes ===")
    for hz in PITCHES:
        report(f"DSP murmur {hz:.0f} Hz",
               synthesize_hum_note(hz, n, sr, hum_cfg), sr, hz)
    for hz in PITCHES:
        try:
            y, ysr = neural_hum(hz)
            report(f"converted hum (page) {hz:.0f} Hz", y, ysr, hz)
        except Exception as exc:  # noqa: BLE001
            print(f"  converted hum {hz:.0f} Hz: unavailable ({type(exc).__name__}: {exc})")

    print("\n--- real recorded vowels for scale (a murmur should not look like these) ---")
    import glob
    for path in sorted(glob.glob("synthesis/source_takes/*.wav")):
        y, ysr = sf.read(path, dtype="float32", always_2d=False)
        report(path.split("/")[-1][:30], y, ysr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

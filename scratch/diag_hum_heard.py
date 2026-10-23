"""Which hum does the page actually play, and does it have vowel formants?

The DSP hum is one thing; the live page, when the neural sidecar is up, swaps
in a server-rendered take at full volume and ducks the local voice to 12%. So
the sound in the listener's ear in Humming mode is the *converted* hum. Here
the three candidates are measured with the same yardstick -- a third-octave
spectral envelope in dB, and, because a murmur is defined by what is missing
above its fundamental, the level of the first formant region against the
fundamental's own.

Run:  .venv/bin/python scratch/diag_hum_heard.py
"""
from __future__ import annotations

import io
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from config.config_loader import get_config  # noqa: E402
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402

BANDS = [(100, 160), (160, 250), (250, 400), (400, 560), (560, 800),
         (800, 1130), (1130, 1600), (1600, 2260), (2260, 3200), (3200, 4500)]


def steady(y: np.ndarray, sr: int) -> np.ndarray:
    """The loud middle of a note, where the timbre has settled."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(sr * 0.4)
    if y.size <= n:
        return y
    start = int(y.size * 0.45)
    return y[start:start + n]


def envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Third-octave envelope in dB, referenced to its own loudest band."""
    x = steady(y, sr)
    win = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * win)) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    out = []
    for lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        out.append(20.0 * np.log10(float(mag[sel].max()) if sel.any() else 1e-15))
    out = np.asarray(out)
    return out - out.max()


def fingerprint(name: str, y: np.ndarray, sr: int, f0: float | None = None) -> None:
    env = envelope(y, sr)
    print(f"  {name:26s} " + " ".join(f"{v:7.1f}" for v in env))
    # Where the sound is 20 dB down, in Hz: a murmur is dark well below 1 kHz,
    # a vowel keeps its first two formants alive out to ~1 kHz and beyond.
    down20 = None
    for (lo, hi), v in zip(BANDS, env):
        if v <= -20.0:
            down20 = lo
            break
    rms = float(np.sqrt(np.mean(steady(y, sr) ** 2)))
    print(f"  {'':26s} -> -20 dB by {down20} Hz, rms {rms:.4f}")


def read(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    return y, sr


def main() -> int:
    cfg = get_config()
    sr = int(cfg["audio"]["sample_rate"])
    hum_cfg = HumConfig.from_config()
    f0 = 220.0
    n = int(1.8 * sr)

    print("=== harmonic envelope, third-octave bands (dB re: own max) ===")
    print("  " + " " * 26 + " ".join(f"{lo:7d}" for lo, _ in BANDS))

    # 1. the server's DSP hum: the reference murmur
    hum = synthesize_hum_note(f0, n, sr, hum_cfg)
    fingerprint("server DSP hum", hum, sr, f0)

    # 2. recorded vowel takes, the thing a listener calls "ahhh"/"ohhh"
    import glob
    for path in sorted(glob.glob("synthesis/source_takes/*.wav"))[:4]:
        y, ysr = read(path)
        fingerprint(f"recorded {path.split('/')[-1][:16]}", y, ysr)

    # 3. the LIVE neural take, exactly what the page swaps in for a hum onset
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/neural/render",
            data=json.dumps({"target_hz": f0, "vocable": "aah", "duration_s": 1.8,
                             "mode": "solo_humming", "texture": "solo",
                             "neural_voice_count": 1}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            applied = resp.headers.get("X-Neural-Applied")
            raw = resp.read()
        y, ysr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        fingerprint(f"LIVE neural hum (applied={applied})", y, ysr)
    except Exception as exc:  # noqa: BLE001
        print(f"  LIVE neural hum: unavailable ({type(exc).__name__}: {exc})")

    print("\n(A murmur: nearly all energy in the bottom two bands and nothing "
          "left by 800 Hz. A vowel: a plateau carried up to 800-1130 Hz.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

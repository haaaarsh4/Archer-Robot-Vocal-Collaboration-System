"""What the hum does over a real track, window by window.

The panel's Humming mode on a track that opens with sixteen seconds of drums:
does the hum wait for the singer, or does it hum the drum? Asked of the very
code the render uses (track_melody then render_hum_track), with the track's own
melody printed beside the hum's level so a drum-following hum cannot hide behind
an average.

Run:  .venv/bin/python scratch/hum_on_track.py "frontend/assets/audio/hummingbird.mp3"
"""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from synthesis.hum_voice import (HumConfig, render_hum_track,  # noqa: E402
                                 track_melody)

WIN = 2.0


def main(path: str) -> int:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    cfg = HumConfig.from_config()
    print(f"{path}: {len(y) / sr:.1f} s at {sr} Hz")
    melody = track_melody(y, sr, cfg)
    hum = render_hum_track(y, sr, cfg)
    print(f"melody frames: {melody.f0_hz.size}, voiced share "
          f"{float(np.mean(melody.voiced)) * 100:.1f}%, hop {melody.hop_s:.3f} s")
    print(f"hum length {len(hum) / sr:.1f} s, rms {float(np.sqrt(np.mean(hum ** 2))):.4f}")
    print("\n  window      track rms   tracked Hz  voiced%   hum rms   hum/track dB")
    hop = melody.hop_s
    for t0 in np.arange(0, len(y) / sr - WIN, WIN):
        a, b = int(t0 * sr), int((t0 + WIN) * sr)
        seg = y[a:b]
        seg_h = hum[a:b] if len(hum) > b else np.zeros(1)
        f0 = melody.f0_hz[int(t0 / hop):int((t0 + WIN) / hop)]
        vo = melody.voiced[int(t0 / hop):int((t0 + WIN) / hop)]
        voiced = f0[vo]
        hz = float(np.median(voiced)) if voiced.size else float("nan")
        tr = float(np.sqrt(np.mean(seg ** 2)))
        hr = float(np.sqrt(np.mean(seg_h ** 2))) if seg_h.size else 0.0
        ratio = 20 * np.log10((hr + 1e-9) / (tr + 1e-9))
        print(f"  {t0:4.0f}-{t0 + WIN:4.0f}s  {tr:9.4f}  {hz:10.1f}  "
              f"{float(np.mean(vo)) * 100:6.1f}  {hr:8.4f}  {ratio:+11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else
                          "frontend/assets/audio/hummingbird.mp3"))

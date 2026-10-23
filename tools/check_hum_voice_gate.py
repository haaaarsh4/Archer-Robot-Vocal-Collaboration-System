"""Verify that a hum only follows a VOICE, in both implementations.

A struck drum's own low note is richly harmonic, so every pitch tracker in this
project follows it. Measured on this project's own reference track, whose first
sixteen seconds are drums alone, pyin answered "voiced" on 100 per cent of those
frames at 79-85 Hz (near the drum's own E2) and the hum droned the drum at full
level -- 7 dB under the track -- for sixteen seconds, while the singing that
starts at 16 s went unhummed. What a listener hears there is the robot not
humming at all.

The gate is one rule in two places: HumConfig.voice_ratio_min in
synthesis/hum_voice.py, which takes the frames off the server's melody, and the
HUM_VOICE_* constants beside _HumVoice in frontend/index.html, which keeps the
live panel quiet. This file reads both, then measures the render:

  * the page's constants equal the server's, value for value and band for band,
    and the page's live loop is actually gated on them;
  * on real material a drum measures under the threshold and a voice over it --
    the numbers the comments claim, re-derived rather than trusted;
  * the rendered hum is far below the track where the track has no voice, and
    with it where it has one.

Run:  .venv/bin/python tools/check_hum_voice_gate.py [track]
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
from synthesis.hum_voice import HumConfig, _voice_like, render_hum_track  # noqa: E402

PAGE = pathlib.Path("frontend/index.html")
SOLO_TAKE = "synthesis/source_takes/Lalanana.wav"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str) -> None:
    print(f"    [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


def page_number(name: str) -> float | None:
    m = re.search(rf"const\s+{name}\s*=\s*([0-9.]+)\s*;", PAGE.read_text())
    return float(m.group(1)) if m else None


def window_ratio(audio: np.ndarray, sr: int, cfg: HumConfig, start_s: float,
                 end_s: float) -> float:
    """The gate's own measurement over one stretch: voiced frames that pass it."""
    a, b = int(start_s * sr), int(end_s * sr)
    seg = np.asarray(audio[a:b], dtype=np.float64)
    n = max(1, int(len(seg) / (cfg.hop_size)))
    like = _voice_like(seg, sr, cfg, cfg.hop_size / sr, n)
    return float(np.mean(like)) if like.size else 0.0


def window_rms(x: np.ndarray, sr: int, start_s: float, end_s: float) -> float:
    a, b = int(start_s * sr), int(end_s * sr)
    seg = np.asarray(x[a:b], dtype=np.float64)
    return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0


def main() -> int:
    track = sys.argv[1] if len(sys.argv) > 1 else "frontend/assets/audio/hummingbird.mp3"
    cfg = HumConfig.from_config()

    # ------------------------------------------------------------------
    print("\n=== one rule, two implementations (the page's constants)")
    # ------------------------------------------------------------------
    for name, server_value in (("HUM_VOICE_RATIO_MIN", cfg.voice_ratio_min),
                               ("HUM_VOICE_BAND_LO_HZ", cfg.voice_band_lo_hz),
                               ("HUM_VOICE_BAND_HI_HZ", cfg.voice_band_hi_hz),
                               ("HUM_VOICE_REF_LO_HZ", cfg.voice_ref_lo_hz),
                               ("HUM_VOICE_REF_HI_HZ", cfg.voice_ref_hi_hz)):
        got = page_number(name)
        check(f"{name} in the page matches HumConfig", got is not None and abs(got - server_value) < 1e-9,
              f"page {got}, server {server_value}" if got is not None else "not found in the page")
    text = PAGE.read_text()
    check("the page's live loop is gated on that rule",
          "isLikelyVoiceFrame(demoFreqBuf" in text
          and "result.action !== 'rest' && voiceHere" in text,
          "isLikelyVoiceFrame is asked of the analyser every frame and voiceHere guards "
          "whether the robot sounds at all")

    # ------------------------------------------------------------------
    print("\n=== what a drum measures, and what a voice measures")
    # ------------------------------------------------------------------
    y, sr = sf.read(track, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    drum = window_ratio(y, sr, cfg, 0.0, 14.0)
    voice = window_ratio(y, sr, cfg, 20.0, 46.0)
    print(f"    {track}: drum-only stretch {drum * 100:.0f}% of frames pass the gate, "
          f"the sung stretch {voice * 100:.0f}%")
    check("a drum is under the threshold", drum < 0.5,
          f"{drum * 100:.0f}% of the drum's frames read as a voice")
    check("a voice is over the threshold", voice > 0.5,
          f"{voice * 100:.0f}% of the sung frames read as a voice")

    if pathlib.Path(SOLO_TAKE).exists():
        z, zsr = sf.read(SOLO_TAKE, dtype="float32", always_2d=False)
        if z.ndim > 1:
            z = z.mean(axis=1)
        solo = window_ratio(z, zsr, cfg, 0.0, len(z) / zsr)
        check("a solo voice is not gated out", solo > 0.5,
              f"{solo * 100:.0f}% of {SOLO_TAKE}'s frames pass; a gate that silences a "
              f"solo singer would be worse than the drum it is there to stop")

    # ------------------------------------------------------------------
    print("\n=== the render: quiet where there is no voice, there where there is")
    # ------------------------------------------------------------------
    hum = render_hum_track(y, sr, cfg)
    rows = []
    for t0 in np.arange(0.0, len(y) / sr - 2.0, 2.0):
        ratio = window_ratio(y, sr, cfg, t0, t0 + 2.0)
        tr = window_rms(y, sr, t0, t0 + 2.0)
        hr = window_rms(hum, sr, t0, t0 + 2.0)
        rows.append((t0, ratio, 20 * np.log10((hr + 1e-9) / (tr + 1e-9))))
    quiet = [d for _, ratio, d in rows if ratio < 0.35]
    singing = [d for _, ratio, d in rows if ratio >= 0.6]
    for t0, ratio, d in rows[:9]:
        print(f"      {t0:5.0f}s  gate {ratio * 100:5.0f}%  hum/track {d:+6.1f} dB")
    if quiet:
        worst = max(quiet)
        check("over the drums the hum is under the track by more than 12 dB",
              worst < -12.0, f"its loudest such window sits {worst:+.1f} dB under")
    else:
        check("the track has a stretch with no voice in it to check", False,
              "no window of this track measured under the gate's own threshold")
    if singing:
        best = max(singing)
        check("where the voice is, the hum is with it", best > -12.0,
              f"its loudest window sits {best:+.1f} dB under the track")
    else:
        check("the track has a sung stretch to check", False,
              "no window of this track measured as a voice")

    print(f"\n{'ALL VOICE-GATE CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURES'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

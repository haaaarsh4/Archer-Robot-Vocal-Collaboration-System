"""Runs the project's OWN audit criteria on the modes this change touched.

tools/check_demo_matrix.py measures every render the panel can ask for (clicks,
dropouts, DC, clipping, length, level, whether the render sings the source's own
note) and passes it through `verdict`. The full matrix is dozens of renders plus
the Neural sidecar, so this runs the same measurement and the same verdict on
just the paths that changed, through the same endpoint the page uses:

  * the two rebuilt modes (octave below, drone support) and unison as the
    control, at the panel's own default vocal-fold-vibration setting;
  * the same three at 0%, which is the previous, straight-tone behaviour.

Anything the matrix would have failed -- a click, a dropout, a DC offset, a part
that is the wrong length or the wrong level -- fails here too.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

from check_demo_matrix import analyse, make_inputs, verdict  # noqa: E402


def main() -> int:
    import server
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    inputs, refs = make_inputs()
    raw = inputs["take.wav"]
    ref, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    expect_s = len(ref) / sr
    print(f"input: {expect_s:.2f}s of take, the panel's own test input\n", flush=True)

    for amount in ("45", "0"):
        for mode in ("unison_shadowing", "octave_reinforcement", "drone_support"):
            resp = client.post(
                "/api/dsp/render-track",
                files={"file": ("take.wav", raw, "audio/wav")},
                data={
                    "texture": "solo",
                    "pitch_method": "yin",
                    "mode": mode,
                    "instruments_enabled": "true",
                    "vocable": "mix",
                    "vibrato_amount": amount,
                },
            )
            label = f"DSP {mode} vibrato={amount}%"
            if resp.status_code != 200:
                print(f"[FAIL] {label}: HTTP {resp.status_code} {resp.text[:160]}", flush=True)
                continue
            y, ysr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
            mono = np.ascontiguousarray(y[:, -1], dtype=np.float32)
            info = analyse(mono, ysr, reference=ref, reference_sr=sr)
            info["channels"] = y.shape[1]
            print(f"  {label}: {info['sec']}s, {info['channels']}ch, peak {info['peak']}, "
                  f"rms {info['rms']}, dc {info['dc']}, clicks {info['clicks']}, "
                  f"dropouts {info['dropouts_ms']}, note-fit {info.get('note_fit_pct')}% at f0, "
                  f"{info.get('octave_down_pct')}% an octave below", flush=True)
            verdict(label, info, expect_sec=expect_s,
                    minsrc=(mode != "call_and_response"))

    import check_demo_matrix as cdm
    if cdm.FAILURES:
        print(f"\nFAILED ({len(cdm.FAILURES)}):")
        for f in cdm.FAILURES:
            print("  -", f)
        return 1
    print("\nALL CHANGED-MODE RENDER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

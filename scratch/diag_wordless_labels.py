"""Why does a stretch of a track come back as <wordless vocals> instead of the
instrument that is actually playing?

Three sources can put that label on a row:

  1. `_add_percussive_segments`'s gap fill, when PANNs' top classes for a
     2 s window include one of `_PANNS_VOCAL_CLASSES` (Singing, Chant, ...).
  2. the same pass, when the fallback path runs because PANNs is not
     installed at all (`<instruments playing>` / `<percussion playing>`).
  3. the forced-decode path, when the pitch scan flagged a stretch as sung
     and Whisper came back with nothing (or with its own caption for it).

This prints, per 2 s window over the first N seconds of a file: PANNs' top
classes with scores, the pitch-scan verdict for that window, and what the
aggregated instrument spans look like. Run:

    .venv/bin/python scratch/diag_wordless_labels.py <path> [seconds]
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "music samples/shortlorezarchersongs/Ahasiw F 120.mp3"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    with open(path, "rb") as fh:
        raw = fh.read()

    import server

    print(f"\n=== {path} ({seconds:.0f}s window) ===", flush=True)
    audio = server._decode_audio_via_ffmpeg(raw, 32000)
    dur = len(audio) / 32000
    print(f"decoded {dur:.1f}s at 32000 Hz, PANNs installed: {server.panns_model is not None}",
          flush=True)

    if server.panns_model is not None:
        labels = server.panns_labels
        win = int(2.0 * 32000)
        hop = int(1.0 * 32000)
        i = 0
        while i < len(audio) and i / 32000 < seconds:
            chunk = audio[i:i + win]
            if len(chunk) < 32000 * 0.5:
                break
            out, _ = server.panns_model.inference(chunk[None, :])
            order = np.argsort(out[0])[::-1][:5]
            tops = [(labels[j], float(out[0][j])) for j in order]
            named = [f"{l}={s:.2f}->{server._PANNS_LABELS.get(l, '?')}" for l, s in tops]
            chosen = None
            for l, s in tops:
                if s < server.PANNS_MIN_SCORE:
                    break
                if l in server._PANNS_LABELS:
                    chosen = (l, s, server._PANNS_LABELS[l])
                    break
            print(f"  {i/32000:5.1f}s  top5={named}", flush=True)
            print(f"          -> names this window {chosen}", flush=True)
            i += hop

    spans = server._detect_instrument_spans(raw, 2.0)
    print("\n-> instrument spans from _detect_instrument_spans:", flush=True)
    for s in spans:
        print(f"   {s['start']:6.2f}-{s['end']:6.2f}  {s['label']}", flush=True)

    pitch_spans = server._detect_vocal_pitch_spans(raw)
    print("\n-> pitch-scan 'sung' spans (first 12):", flush=True)
    for s, e in pitch_spans[:12]:
        print(f"   {s:6.2f}-{e:6.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

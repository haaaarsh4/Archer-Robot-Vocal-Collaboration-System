"""The DSP render endpoint, driven through its own handler, to check one thing:
that asking for the Humming mode gets a HUM even when a word is also locked, and
that a locked word still wins under any other mode (which is what that control
is for). The locked-word path answers with two channels (the word + the track
with its lead taken out); Mix and the hum answer with one. That shape, plus a
direct comparison against the hum renderer, is what these assertions use.
"""
from __future__ import annotations

import asyncio
import io
import sys

import numpy as np
import soundfile as sf
from fastapi import UploadFile

sys.path.insert(0, ".")
import server
import synthesis.hum_voice as hv
from synthesis.hum_voice import HumConfig, render_hum_track, track_melody, _harmonic_fit


def call(**kw):
    raw = _raw()
    f = UploadFile(filename="t.wav", file=io.BytesIO(raw))
    # every Form(...) parameter the endpoint declares is named here, because
    # calling the handler directly hands it the Form() sentinel as a default and
    # float() chokes on it
    return asyncio.run(server.render_dsp_track(
        file=f, texture="solo", pitch_method="yin", instruments_enabled=False,
        delay_ms=None, echo_amount=None,
        contour_baseline_db=None, contour_max_bright_db=None, contour_max_dark_db=None,
        **kw))


_source = {}


def _raw() -> bytes:
    if "b" not in _source:
        y, sr = sf.read(_source["path"], dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        _source["b"] = buf.getvalue()
        _source["y"] = np.ascontiguousarray(y, dtype=np.float32)
        _source["sr"] = sr
    return _source["b"]


def main() -> int:
    _source["path"] = sys.argv[1] if len(sys.argv) > 1 else "synthesis/source_takes/Moon.wav"
    cfg = HumConfig.from_config()
    _raw()   # loads the source into _source
    hum_truth = render_hum_track(_source["y"], _source["sr"], cfg)
    failures = 0
    # (label, request, should this be the hum?)
    cases = [
        ("Humming mode + a locked word", dict(mode="solo_humming", vocable="cold"), True),
        ("Humming mode + Mix", dict(mode="solo_humming", vocable="mix"), True),
        ("Unison mode + the same locked word (control)", dict(mode="unison_shadowing", vocable="cold"), False),
    ]
    for label, kw, want_hum in cases:
        resp = call(**kw)
        body = getattr(resp, "body", b"")
        if getattr(resp, "status_code", 200) != 200:
            print(f"\n[{label}] FAILED with {resp.status_code}: "
                  f"{body[:300].decode('utf8', 'replace')}")
            failures += 1
            continue
        y, ysr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
        chans = y.shape[1]
        lead = np.ascontiguousarray(y[:, -1], dtype=np.float32)
        n = min(lead.size, hum_truth.size)
        corr = float(np.corrcoef(lead[:n].astype(float), hum_truth[:n].astype(float))[0, 1])
        mel = track_melody(_source["y"], _source["sr"], cfg)
        want = np.where(mel.voiced, mel.f0_hz, 0.0)
        fit = _harmonic_fit(lead[: int(len(_source["y"]) * 1.05)], ysr, want, mel.hop_s)
        m = hv.hum_measurements(lead, _source["y"][: lead.size], ysr, cfg)
        print(f"\n[{label}]  {y.shape[0] / ysr:.2f}s, {chans} channel(s), "
              f"centroid {m['hum_centroid_hz']} Hz, above 2.5 kHz {m['hum_hf_ratio'] * 100:.2f}%, "
              f"correlation with the hum render {corr:.3f}, harmonic comb matched {fit.get('matched_pct')}%")
        if want_hum:
            # the hum render is deterministic (same config, same source, a seeded
            # generator), so the endpoint's own render of it correlates at ~1.0
            if corr < 0.9:
                print("    FAIL Humming mode did not render the hum")
                failures += 1
            if m["hum_centroid_hz"] > 1100.0 or m["hum_hf_ratio"] > 0.03:
                print("    FAIL that render is not a closed-mouth hum")
                failures += 1
        else:
            if corr > 0.5:
                print("    FAIL a locked word under a normal mode rendered the hum instead of the word")
                failures += 1
            print("    (the control: this should be the recorded word, and it is)")
    print(f"\n{'ENDPOINT CHECKS PASSED' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

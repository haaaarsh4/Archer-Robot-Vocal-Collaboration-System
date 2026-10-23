"""Verify the app's real offline humming render, in both engines, from the same
entry point the UI's render endpoints use (server._render_track_offline).

Measures what came out the far end of each path: is it a hum (dark, no
sibilance), is it in tune with the melody it was handed, and does the hum the
Neural engine returns equal the one the DSP engine renders.

That last check is the interesting one. A hum used to be handed to the trained
voice in the Neural engine, and measured on the takes the panel asks for, a
singing converter answers a closed-mouth hum with a vowel: +12 to +40 dB on the
model's own formants, a broadband tail to 4 kHz, and sometimes a re-sung pitch
1.9 semitones off the note it was handed. No post-filter fixes a formant or a
pitch, so the murmur IS the hum in both engines now, and this check exists to
keep that true: the two engines must not drift back apart.
"""
from __future__ import annotations

import io
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
import server  # noqa: E402
from synthesis.hum_voice import HumConfig, track_melody, _harmonic_fit, hum_measurements  # noqa: E402


def check(label: str, audio: np.ndarray, sr: int, reference: np.ndarray, cfg: HumConfig) -> dict:
    out: dict = {}
    mel = track_melody(reference[: audio.size], sr, cfg)
    want = np.where(mel.voiced, mel.f0_hz, 0.0)
    fit = _harmonic_fit(audio, sr, want, mel.hop_s)
    m = hum_measurements(audio, reference[: audio.size], sr, cfg)
    # and a second opinion that shares nothing with the comb: an independent
    # tracker on the OUTPUT, compared with the melody in Hz (NOT folded by
    # octaves -- the point here is to catch an octave mistake, the one thing an
    # octave-agnostic comparison cannot see).
    import librosa

    f0, voiced, _ = librosa.pyin(audio.astype(np.float32), fmin=45.0, fmax=1200.0,
                                 sr=sr, frame_length=4096, hop_length=1024)
    want_at = np.interp(np.arange(f0.size) * 1024.0 / sr,
                        np.arange(want.size) * mel.hop_s, want, left=0.0, right=0.0)
    ok = (~np.isnan(f0)) & (want_at > 0) & np.asarray(voiced, bool)
    if np.any(ok):
        cents = 1200.0 * np.log2(f0[ok] / want_at[ok])
        octave_off = float(np.mean(np.abs(cents) > 600.0) * 100.0)
        tracked_median = float(np.median(np.abs(cents)))
    else:
        octave_off, tracked_median = 100.0, 999.0
    out["octave_off_pct"] = round(octave_off, 1)
    out["tracker_median_cents"] = round(tracked_median, 1)
    print(f"\n=== {label}")
    print(f"    rms {m['hum_rms']} (source {m['source_rms']})  centroid {m['hum_centroid_hz']} Hz  "
          f"above 2.5 kHz {m['hum_hf_ratio'] * 100:.2f}%")
    print(f"    comb matched {fit.get('matched_pct')}%  lag {fit.get('lag_ms')} ms  "
          f"median {fit.get('median_abs_cents')} cents  p90 {fit.get('p90_abs_cents')}  "
          f"held {fit.get('held_frames')} frames median {fit.get('held_median_abs_cents')}  "
          f"gaps {fit.get('gap_db')} dB suboct {fit.get('pct_fundamental_wins')}%")
    print(f"    an independent tracker on the output vs the melody: median "
          f"{out['tracker_median_cents']} cents, an octave out on {out['octave_off_pct']}% of frames")
    return {"fit": fit, "measure": m, "extra": out}


def main() -> int:
    src_path = sys.argv[1] if len(sys.argv) > 1 else "synthesis/source_takes/Lalanana.wav"
    raw, sr = sf.read(src_path, dtype="float32", always_2d=False)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, raw, sr, format="WAV")

    cfg = HumConfig.from_config()
    # ONE CASE PER VOICE IN CONFIG, not a hard-coded pair: a hum must be the
    # same murmur in every engine and every trained voice, and a third voice
    # added to model_paths should be checked without editing this script.
    from pathlib import Path as _Path
    import yaml as _yaml
    with open(_Path(__file__).resolve().parent.parent / "config" / "config.yaml", encoding="utf-8") as fh:
        _ncfg = _yaml.safe_load(fh)["synthesis"]["neural"]
    _models = list(_ncfg.get("model_paths", []))
    cases = [("DSP (closed-mouth voice straight)", False, 0)]
    for _i, _p in enumerate(_models):
        cases.append((f"Neural voice {_i} ({_Path(_p).stem})", True, _i))
    failures = 0
    dsp_render: np.ndarray | None = None
    for label, neural, voice in cases:
        t0 = time.time()
        out = server._render_track_offline(
            buf.getvalue(), "rmvpe", "solo", voice, mode_override="solo_humming",
            apply_neural=neural,
        )
        dt = time.time() - t0
        y, out_sr = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = np.ascontiguousarray(y, dtype=np.float32)
        print(f"\n[{label}] rendered {y.size / out_sr:.2f}s in {dt:.2f}s at {out_sr} Hz")
        r = check(label, y, out_sr, raw, cfg)
        fit = r["fit"]
        m = r["measure"]
        if not fit.get("frames"):
            print("    FAIL no frames of the hum could be read at all")
            failures += 1
            continue
        # The comb share is a diagnostic, not a verdict: how comb-like the output
        # is depends on the trained voice (the DSP hum and voice 0 fit on 77-82%
        # of frames, voice 1 on 43%). What is checked is that the frames that DO
        # fit are in tune, that an independent tracker agrees in Hz, and that
        # nothing is an octave out.
        if (fit.get("matched_pct") or 0) < 30.0:
            print("    FAIL a harmonic series barely fits the output anywhere")
            failures += 1
        if (fit.get("median_abs_cents") or 999) > 35.0:
            print("    FAIL the output is not on the melody it was handed")
            failures += 1
        # note the explicit None tests: 0.0 is a legitimate value here and a
        # plain `or` fallback turns "no octave errors at all" into a failure.
        tracked_median = r["extra"].get("tracker_median_cents")
        octave_off = r["extra"].get("octave_off_pct")
        if tracked_median is not None and tracked_median > 60.0:
            print("    FAIL an independent tracker disagrees with the melody")
            failures += 1
        if octave_off is None or octave_off > 5.0:
            print("    FAIL the output is an octave off the melody somewhere")
            failures += 1
        if m["hum_centroid_hz"] > 1100.0:
            print("    FAIL the output is brighter than a closed mouth")
            failures += 1
        if m["hum_hf_ratio"] > 0.03:
            print("    FAIL the output has sibilance a closed mouth cannot make")
            failures += 1
        if m["hum_rms"] > m["source_rms"] + 0.06:
            print("    FAIL the output is louder than the singer")
            failures += 1
        # THE SAME HUM IN BOTH ENGINES. The Neural engine asks for the same
        # murmur the DSP engine renders (server._render_track_offline), so a
        # difference here is either a conversion that has crept back in or a
        # render that is no longer deterministic -- both worth knowing about.
        if not neural:
            dsp_render = y.copy()
        elif dsp_render is not None:
            if y.size != dsp_render.size or not np.allclose(y, dsp_render, atol=1e-6):
                same = (y.size == dsp_render.size)
                diff = float(np.max(np.abs(y[:min(y.size, dsp_render.size)]
                                          - dsp_render[:min(y.size, dsp_render.size)]))) \
                    if same else float("inf")
                print(f"    FAIL the Neural hum is not the DSP hum (lengths equal: {same}, "
                      f"largest sample difference {diff:.4f}) -- a closed-mouth hum must "
                      f"not be converted (see server.render_neural_note)")
                failures += 1
            else:
                print("    the trained voice did not touch it: this is the DSP murmur, "
                      "sample for sample")
        sf.write(f"/tmp/hum_path_{label.split(' ')[0]}_{voice}.wav", y, out_sr)
    print(f"\n{'ALL PATH CHECKS PASSED' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

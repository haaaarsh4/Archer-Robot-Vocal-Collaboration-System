"""Verify the LIVE one-note-at-a-time humming interface, in both engines.

(The hum is one voice in both: the live Neural door answers with the same
closed-mouth murmur the DSP door renders, which is what this file checks.)

The live engine never hands the synthesizer a whole performance: `_render_hum`
asks the hum voice for ONE note, and the page's live Neural loop posts one note
at a time to /api/neural/render in the page's own JSON body. Those are the two
doors the live app actually uses while it is playing, and they are not the same
doors as the offline whole-track render.

Measured here, through those doors:
  * does a hum come back a hum (dark, closed-mouth, no vowel brightness), against
    the same note asked for as a normal voice on the same endpoint;
  * is it ON the note that was asked for, read back by an independent tracker,
    with octave mistakes counted separately (the one error a hum's own comb
    measurement cannot see);
  * does a held note come back long enough to be held (the live hum asks for 1.8 s
    so the following onset crossfades over the tail instead of leaving silence);
  * does a mid-note top-up (legato) continue the note without a click, which is
    what the live loop does every second or two while a note is held.
"""
from __future__ import annotations

import io
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
import server  # noqa: E402
from config.config_loader import get_config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from synthesis.hum_voice import HumConfig, synthesize_hum_note  # noqa: E402
from synthesis.vocable_synthesizer import VocableSynthesizer  # noqa: E402

PAGE_DURATION_HUM = 1.8     # what the live Neural loop asks for in Humming mode
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str) -> None:
    print(f"    [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


def spectrum(audio: np.ndarray, sr: int) -> dict:
    """Centroid and high-frequency share, both off the whole note."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(2 ** np.floor(np.log2(max(2048, x.size))))
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(x[:n] * win)) + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    p = mag ** 2
    total = float(p.sum())
    centroid = float((freqs * p).sum() / total)
    hf = float(p[freqs >= 2500.0].sum() / total)
    # the hum's own signature: where the energy sits relative to the note's
    # second partial, which a closed mouth keeps low and a vowel pushes up.
    return {"centroid_hz": centroid, "hf_ratio": hf, "rms": float(np.sqrt(np.mean(x ** 2)))}


def pitch_readback(audio: np.ndarray, sr: int, target_hz: float) -> dict:
    """Independent tracker on the OUTPUT, in Hz against the asked-for note."""
    import librosa

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, voiced, _ = librosa.pyin(x, fmin=55.0, fmax=1200.0, sr=sr,
                                 frame_length=4096, hop_length=1024)
    ok = (~np.isnan(f0)) & np.asarray(voiced, bool)
    if not np.any(ok):
        return {"frames": 0, "median_cents": 999.0, "octave_pct": 100.0, "p90_cents": 999.0}
    cents = 1200.0 * np.log2(f0[ok] / target_hz)
    return {
        "frames": int(ok.sum()),
        "median_cents": round(float(np.median(np.abs(cents))), 1),
        "p90_cents": round(float(np.percentile(np.abs(cents), 90)), 1),
        "octave_pct": round(float(np.mean(np.abs(cents) > 600.0) * 100.0), 1),
    }


def wav_of(response) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def live_hum_note(client: TestClient, target_hz: float, duration_s: float = PAGE_DURATION_HUM,
                  mode: str = "solo_humming", vocable: str = "aah",
                  texture: str = "solo") -> object:
    """Exactly the body the page's live Neural loop posts (frontend/index.html)."""
    return client.post("/api/neural/render", json={
        "target_hz": target_hz,
        "vocable": vocable,
        "duration_s": duration_s,
        "mode": mode,
        "texture": texture,
        "neural_voice_count": 1,      # the page sends this; the route must not choke on it
    })


def main() -> int:
    cfg = get_config()
    sample_rate = int(cfg["audio"]["sample_rate"])
    hum_cfg = HumConfig.from_config()
    print(f"HumConfig in use: breath {hum_cfg.breath_db} dB (cap {hum_cfg.breath_cap_db}), "
          f"vibrato {hum_cfg.vibrato_rate_hz} Hz, note window {hum_cfg.note_window_s} s, "
          f"pitch method {hum_cfg.pitch_method}")

    if server.synthesizer is None:
        print(f"FATAL: synthesizer not loaded: {server.synthesizer_load_error}")
        return 2
    neural_up = (server.neural_timbre is not None and server.neural_timbre.enabled
                 and getattr(server.neural_timbre, "_reachable", False))
    print(f"Neural sidecar reachable: {neural_up}")
    if not neural_up:
        print("FATAL: the live Neural path cannot be measured without the sidecar "
              f"({server.neural_timbre_load_error})")
        return 2

    notes = [220.00, 277.18, 329.63]      # A3, C#4, E4 -- a small live phrase
    with TestClient(server.app) as client:
        # ------------------------------------------------------------------
        # 1. the LIVE NEURAL door: one note per request, the page's own body
        # ------------------------------------------------------------------
        print("\n=== live Neural: one note at a time (/api/neural/render) ===")
        hum_neural: dict[float, np.ndarray] = {}
        for hz in notes:
            t0 = time.perf_counter()
            r = live_hum_note(client, hz)
            ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                check(f"neural hum {hz:.1f} Hz", False, f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            hum_neural[hz], sr = wav_of(r)
            applied = r.headers.get("X-Neural-Applied")
            sp = spectrum(hum_neural[hz], sr)
            pi = pitch_readback(hum_neural[hz], sr, hz)
            held = len(hum_neural[hz]) / sr
            print(f"  {hz:.2f} Hz: {held:.2f} s in {ms:.0f} ms  neural_applied={applied}  "
                  f"centroid {sp['centroid_hz']:.0f} Hz  above 2.5 kHz {sp['hf_ratio'] * 100:.2f}%  "
                  f"pitch median {pi['median_cents']} cents (p90 {pi['p90_cents']}), "
                  f"octave-out {pi['octave_pct']}% of {pi['frames']} frames")
            check(f"holds for the note it answers ({hz:.1f} Hz)",
                  held >= PAGE_DURATION_HUM * 0.98,
                  f"{held:.2f} s against the {PAGE_DURATION_HUM} s the live loop asks for")
            # A HUM IS NOT CONVERTED (see server.render_neural_note): a
            # converter trained on singing answers a closed-mouth hum with a
            # sung vowel, measured at +12 to +40 dB on the model's own formants
            # with a broadband tail to 4 kHz, so the DSP murmur is what this
            # endpoint hands back. This check is here to keep that true -- the
            # earlier version of this file asserted the opposite.
            check(f"the hum was NOT converted ({hz:.1f} Hz)", applied == "0",
                  f"X-Neural-Applied={applied}")
            check(f"is a hum, not a vowel ({hz:.1f} Hz)",
                  sp["centroid_hz"] < 900.0 and sp["hf_ratio"] < 0.01,
                  f"centroid {sp['centroid_hz']:.0f} Hz, {sp['hf_ratio'] * 100:.2f}% above 2.5 kHz")
            check(f"is on the asked-for note ({hz:.1f} Hz)",
                  pi["median_cents"] <= 35.0 and pi["octave_pct"] <= 2.0,
                  f"median {pi['median_cents']} cents, octave-out on {pi['octave_pct']}% of frames")

        # the same endpoint, same note, a normal voice: the contrast that says
        # the hum above is the hum path and not the endpoint being dark for
        # everything.
        mid = notes[1]
        t0 = time.perf_counter()
        r_uni = live_hum_note(client, mid, duration_s=0.45, mode="unison_shadowing")
        ms = (time.perf_counter() - t0) * 1000.0
        if r_uni.status_code != 200:
            check("control: unison on the same endpoint", False,
                  f"HTTP {r_uni.status_code}: {r_uni.text[:200]}")
        else:
            uni, sr = wav_of(r_uni)
            sp_u = spectrum(uni, sr)
            print(f"  control (unison {mid:.2f} Hz, 0.45 s in {ms:.0f} ms): "
                  f"centroid {sp_u['centroid_hz']:.0f} Hz  above 2.5 kHz {sp_u['hf_ratio'] * 100:.2f}%")
            hum_sp = spectrum(hum_neural[mid], sr) if mid in hum_neural else {"centroid_hz": 0.0,
                                                                                 "hf_ratio": 0.0}
            check("control: the same endpoint stays bright for a sung word",
                  sp_u["centroid_hz"] > hum_sp["centroid_hz"] * 1.3
                  and sp_u["hf_ratio"] > hum_sp["hf_ratio"] * 1.3,
                  f"unison centroid {sp_u['centroid_hz']:.0f} Hz / {sp_u['hf_ratio'] * 100:.2f}% "
                  f"above 2.5 kHz vs the hum's {hum_sp['centroid_hz']:.0f} Hz / "
                  f"{hum_sp['hf_ratio'] * 100:.2f}% on the same note")

    # ----------------------------------------------------------------------
    # 2. the LIVE DSP door: the synthesizer asked for one note, then topped up
    # ----------------------------------------------------------------------
    print("\n=== live DSP: one note, then a mid-note top-up (legato) ===")
    synth = VocableSynthesizer()
    note_s = 1.0
    prev = None
    for hz in notes:
        req = server.RenderNoteRequest(target_hz=hz, vocable="aah", duration_s=note_s,
                                       mode="solo_humming", texture="solo")
        decision = server._build_render_decision(req, cfg)
        note = np.asarray(synth.synthesize(decision), dtype=np.float32)
        sp = spectrum(note, sample_rate)
        pi = pitch_readback(note, sample_rate, hz)
        print(f"  {hz:.2f} Hz: {len(note) / sample_rate:.3f} s  centroid {sp['centroid_hz']:.0f} Hz  "
              f"above 2.5 kHz {sp['hf_ratio'] * 100:.2f}%  pitch median {pi['median_cents']} cents, "
              f"octave-out {pi['octave_pct']}%")
        check(f"live DSP note length ({hz:.1f} Hz)",
              abs(len(note) / sample_rate - note_s) <= 0.02,
              f"{len(note) / sample_rate:.3f} s for the {note_s} s asked for")
        check(f"live DSP hum is dark and closed-mouth ({hz:.1f} Hz)",
              sp["centroid_hz"] < 900.0 and sp["hf_ratio"] < 0.01,
              f"centroid {sp['centroid_hz']:.0f} Hz, {sp['hf_ratio'] * 100:.2f}% above 2.5 kHz")
        check(f"live DSP hum is on the asked-for note ({hz:.1f} Hz)",
              pi["median_cents"] <= 35.0 and pi["octave_pct"] <= 2.0,
              f"median {pi['median_cents']} cents, octave-out {pi['octave_pct']}%")
        prev = hz

    # ----------------------------------------------------------------------
    # 3. a HELD note, the way the live loop holds one: the same note asked for
    # again and again by top-ups. This is the one thing a hum does that a word
    # cannot (a word restarts, a hum just keeps going), so it is measured on
    # the voice's own output, where nothing but the hum is in the signal.
    # ----------------------------------------------------------------------
    print("\n=== live DSP: a HELD note, chunk after chunk ===")
    held_hz, chunk_s = 329.63, 0.6
    n_chunk = int(chunk_s * sample_rate)
    state: dict = {}
    # the engine knows the note is holding when it renders its first chunk
    # (`action == sustain`), so the end of that chunk is not a note end.
    chunks = [np.asarray(synthesize_hum_note(held_hz, n_chunk, sample_rate, cfg=hum_cfg,
                                             state=state, hold=True), dtype=np.float32)]
    for _ in range(2):
        chunks.append(np.asarray(synthesize_hum_note(held_hz, n_chunk, sample_rate, cfg=hum_cfg,
                                                     legato=True, state=state, hold=True),
                                 dtype=np.float32))
    held = np.concatenate(chunks)
    d = np.abs(np.diff(held.astype(np.float64)))
    own = float(np.percentile(d, 99.9))
    seams = [i * n_chunk for i in (1, 2)]
    worst = max(float(d[max(0, j - 3):j + 3].max()) for j in seams)
    # the level either side of a seam, which is what a re-normalised or
    # re-gated chunk would step on
    win = int(0.02 * sample_rate)
    levels = []
    for j in seams:
        a = float(np.sqrt(np.mean(held[j - win:j] ** 2)))
        b = float(np.sqrt(np.mean(held[j:j + win] ** 2)))
        levels.append(20.0 * np.log10((b + 1e-12) / (a + 1e-12)))
    print(f"  {len(held) / sample_rate:.2f} s from {len(chunks)} chunks; worst step at a seam "
          f"{worst:.5f} vs the note's own p99.9 step {own:.5f}; level either side "
          f"{levels[0]:+.2f} / {levels[1]:+.2f} dB")
    check("a held hum continues across a top-up with no step",
          worst <= max(own * 1.5, 1e-4),
          f"seam step {worst:.5f} against the note's own p99.9 slope {own:.5f}")
    check("a held hum keeps its level across a top-up",
          all(abs(dl) <= 0.5 for dl in levels),
          f"level step {levels[0]:+.2f} / {levels[1]:+.2f} dB across the seams")
    pi = pitch_readback(held, sample_rate, held_hz)
    check("a held hum stays on its pitch across the top-ups",
          pi["median_cents"] <= 35.0 and pi["octave_pct"] <= 2.0,
          f"median {pi['median_cents']} cents over {len(held) / sample_rate:.2f} s, "
          f"octave-out {pi['octave_pct']}% of {pi['frames']} frames")
    sp_h = spectrum(held, sample_rate)
    check("a long held hum is still dark",
          sp_h["centroid_hz"] < 900.0 and sp_h["hf_ratio"] < 0.01,
          f"centroid {sp_h['centroid_hz']:.0f} Hz, {sp_h['hf_ratio'] * 100:.2f}% above 2.5 kHz")

    # the same held note through the synthesizer the live engine actually
    # calls, with the crossfade left to the caller (_render_hum applies it).
    req = server.RenderNoteRequest(target_hz=held_hz, vocable="aah", duration_s=chunk_s,
                                   mode="solo_humming", texture="solo")
    decision = server._build_render_decision(req, cfg)
    decision.action = "sustain"          # the engine says: this note is holding
    synth2 = VocableSynthesizer()
    pieces = [np.asarray(synth2.synthesize(decision, apply_crossfade=False), dtype=np.float32)]
    for _ in range(2):
        pieces.append(np.asarray(synth2.synthesize(decision, legato=True, apply_crossfade=False),
                                 dtype=np.float32))
    s_held = np.concatenate(pieces)
    d2 = np.abs(np.diff(s_held.astype(np.float64)))
    own2 = float(np.percentile(d2, 99.9))
    worst2 = max(float(d2[max(0, j - 3):j + 3].max()) for j in [i * n_chunk - 1 for i in (1, 2)])
    pi2 = pitch_readback(s_held, sample_rate, held_hz)
    print(f"  through the synthesizer's own legato path: worst step at a seam {worst2:.5f} "
          f"vs its own p99.9 {own2:.5f}; pitch median {pi2['median_cents']} cents, "
          f"octave-out {pi2['octave_pct']}%")
    check("the synthesizer's legato top-up does not click",
          worst2 <= max(own2 * 2.0, 0.01),
          f"seam step {worst2:.5f} against its own p99.9 slope {own2:.5f}")
    check("the synthesizer's held hum stays on its pitch",
          pi2["median_cents"] <= 35.0 and pi2["octave_pct"] <= 2.0,
          f"median {pi2['median_cents']} cents, octave-out {pi2['octave_pct']}%")

    # and the same again through the DEFAULT path (_render_hum crossfades a new
    # note and deliberately does not crossfade a top-up), which is the one the
    # live engine is on.
    synth3 = VocableSynthesizer()
    pieces = [np.asarray(synth3.synthesize(decision), dtype=np.float32)]
    for _ in range(2):
        pieces.append(np.asarray(synth3.synthesize(decision, legato=True), dtype=np.float32))
    d3 = np.abs(np.diff(np.concatenate(pieces).astype(np.float64)))
    lv = []
    win3 = int(0.02 * sample_rate)
    for i in (1, 2):
        j = i * len(pieces[0])
        a = float(np.sqrt(np.mean(pieces[i - 1][-win3:] ** 2)))
        b = float(np.sqrt(np.mean(pieces[i][:win3] ** 2)))
        lv.append(20.0 * np.log10((b + 1e-12) / (a + 1e-12)))
    worst3 = max(float(d3[i * len(pieces[0]) - 3:i * len(pieces[0]) + 3].max()) for i in (1, 2))
    print(f"  through the default live path: worst step at a seam {worst3:.5f} vs p99.9 "
          f"{float(np.percentile(d3, 99.9)):.5f}; level {lv[0]:+.2f} / {lv[1]:+.2f} dB")
    check("the default live hold path has no step at a top-up",
          worst3 <= max(float(np.percentile(d3, 99.9)) * 2.0, 0.01),
          f"seam step {worst3:.5f} against the signal's own p99.9 slope")
    check("the default live hold path keeps its level",
          all(abs(dl) <= 0.5 for dl in lv),
          f"level step {lv[0]:+.2f} / {lv[1]:+.2f} dB across the seams")

    print()
    if FAILURES:
        print(f"LIVE NOTE CHECKS FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("LIVE NOTE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

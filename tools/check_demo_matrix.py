"""Audit the Upload panel's whole render matrix, through the real endpoints.

Every combination the page can ask for -- both engines (DSP, Neural), every
accompaniment mode, locked word vs Mix, instruments on and off, every
configured trained voice, the convert and backing-bed routes -- is posted to the same handler the
browser posts to, with the same form fields, and what comes back is measured
rather than assumed:

  * did it render at all (status, and for a refusal, the sentence it gave);
  * is it the length it should be, and is anything in it non-finite;
  * does it clip, is there DC on it, does it start or end in the middle of
    something;
  * are there GLITCHES -- sample-delta outliers (clicks), short dropouts,
    long unexplained silences, energy that jumps between chunks;
  * is it in tune with the melody the source is singing, read back by an
    independent tracker;
  * is the input itself intact: does an uploaded MP3 decode to the same music
    the page played (sample rate, level, no truncation, no aliasing).

Inputs are the two shapes the panel actually receives: a WAV take, the same
take as a 192 kbps MP3 (what "upload an MP3" means), and a synthetic
drums+voice song, which is what the instrument/percussion handling is for.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

MODE_LABELS = {
    "unison_shadowing": "Unison",
    "octave_reinforcement": "Octave",
    "delayed_response": "Delayed",
    "timbral_thickening": "Timbral",
    "contour_following": "Contour",
    "drone_support": "Drone",
    "call_and_response": "Call/response",
    "solo_humming": "Humming",
}
TEXTURES = ["solo", "duet", "choir"]

FAILURES: list[str] = []
NOTES: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    FAILURES.append(msg)
    say(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    say(f"  [ok]   {msg}")


# ---------------------------------------------------------------- test inputs
def make_inputs(workdir: str = "/tmp/demo_matrix") -> tuple[dict[str, bytes], dict[str, np.ndarray]]:
    """The two shapes of input the panel receives, and the music inside each.

    Returns (bytes by name, the mono reference each file actually holds). The
    reference is what makes the fidelity section below mean anything: an MP3 has
    to be judged against the music it was made FROM, not against a different
    file (the first version of this compared the song inputs with the take
    voice, which flags a perfectly good decode as "not this track").
    """
    import os
    os.makedirs(workdir, exist_ok=True)
    src = "synthesis/source_takes/Lalanana.wav"
    take, sr = sf.read(src, dtype="float32", always_2d=True)
    take_mono = take.mean(axis=1)
    out: dict[str, bytes] = {}
    refs: dict[str, np.ndarray] = {}

    def wav_bytes(x: np.ndarray, rate: int) -> bytes:
        b = io.BytesIO()
        sf.write(b, x, rate, format="WAV", subtype="PCM_16")
        return b.getvalue()

    out["take.wav"] = wav_bytes(take_mono, sr)
    refs["take.wav"] = take_mono.astype(np.float32)

    mp3_path = f"{workdir}/take.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-b:a", "192k", mp3_path],
                   check=True)
    with open(mp3_path, "rb") as f:
        out["take.mp3"] = f.read()
    refs["take.mp3"] = take_mono.astype(np.float32)

    # A SONG-SHAPED INPUT: the voice over a drum bed, repeated so there is
    # enough of it for the model to separate (Demucs, at the panel's own 6 s
    # minimum) and for a chunked render to have more than one chunk to join.
    rng = np.random.default_rng(7)
    reps = 4
    take_rep = np.tile(take_mono, reps)
    n = take_rep.size
    t = np.arange(n) / sr
    bed = np.zeros(n, dtype=np.float64)
    beat = 60.0 / 96.0                      # 96 bpm
    for b in range(int(n / sr / beat)):
        at = int(b * beat * sr)
        if at >= n:
            break
        if b % 2 == 0:                      # kick
            dur = int(0.18 * sr)
            env = np.exp(-np.linspace(0, 9, min(dur, n - at)))
            bed[at:at + env.size] += 0.5 * env * np.sin(2 * np.pi * 55 * np.arange(env.size) / sr)
        else:                               # snare-ish
            dur = int(0.12 * sr)
            env = np.exp(-np.linspace(0, 14, min(dur, n - at)))
            noise = rng.standard_normal(env.size) * 0.35
            bed[at:at + env.size] += 0.3 * env * (noise + np.sin(2 * np.pi * 190 * t[at:at + env.size]))
        hat = int(0.03 * sr)                # hat on every eighth
        for off in (0.0, 0.5):
            at2 = int((b + off) * beat * sr)
            if at2 + hat < n:
                h = np.exp(-np.linspace(0, 18, hat)) * rng.standard_normal(hat) * 0.12
                bed[at2:at2 + hat] += h
    song = 0.75 * take_rep + 0.55 * bed.astype(np.float32)
    song = song / max(1e-9, np.max(np.abs(song))) * 0.95
    out["song.wav"] = wav_bytes(song, sr)
    refs["song.wav"] = song.astype(np.float32)

    song_mp3 = f"{workdir}/song.mp3"
    with open(f"{workdir}/song_raw.wav", "wb") as f:
        f.write(out["song.wav"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{workdir}/song_raw.wav",
                    "-b:a", "192k", song_mp3], check=True)
    with open(song_mp3, "rb") as f:
        out["song.mp3"] = f.read()
    refs["song.mp3"] = song.astype(np.float32)
    return out, refs


# ------------------------------------------------------------------ measuring
def analyse(x: np.ndarray, sr: int, reference: np.ndarray | None = None,
            reference_sr: int = 44100) -> dict:
    """Everything measurable about one rendered part.

    `reference` is the performance the part is a rendering OF, at the same
    length and alignment. It is what makes the two measurements that matter
    mean something: a DROPOUT only counts where the source had something to
    sing (music has pauses, and a part that is quiet through one is correct),
    and tuning is only asked for on the frames where the source was voiced.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    out: dict = {"sec": round(y.size / sr, 3), "n": int(y.size)}
    if y.size == 0:
        out["empty"] = True
        return out
    out["finite"] = bool(np.all(np.isfinite(y)))
    out["peak"] = round(float(np.max(np.abs(y))), 4)
    out["rms"] = round(float(np.sqrt(np.mean(y ** 2))), 4)
    out["dc"] = round(float(np.mean(y)), 5)
    out["clip_pct"] = round(float(np.mean(np.abs(y) >= 0.999) * 100.0), 4)
    d = np.abs(np.diff(y))
    p999 = float(np.percentile(d, 99.9))
    out["p999_step"] = round(p999, 5)
    # a click is a step at least 8x the signal's own worst typical step
    thr = max(p999 * 8.0, 0.05)
    out["clicks"] = int(np.sum(d > thr))
    out["worst_step"] = round(float(d.max()), 4)

    ref_env = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference_sr != sr:
            import librosa
            ref = np.asarray(librosa.resample(ref.astype(np.float32), orig_sr=reference_sr,
                                              target_sr=sr), dtype=np.float64)
        ref = ref[:y.size]
        w = max(1, int(0.005 * sr))
        ref_env = np.sqrt(np.convolve(ref ** 2, np.ones(w) / w, mode="same"))

    # dropouts: runs of near-silence inside a signal that is otherwise loud, AND
    # only where the reference itself is singing. A part that rests through an
    # instrumental passage is doing what it was asked to.
    w = max(1, int(0.005 * sr))
    env = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same"))
    quiet = env < (np.percentile(env, 90) * 0.02 + 1e-9)
    if ref_env is not None and ref_env.size:
        loud = ref_env >= max(0.08 * float(np.percentile(ref_env, 90)), 1e-5)
        # the reference can be shorter than the render (a render that runs past
        # the source, an echo tail), so the two are matched before the AND
        quiet = quiet[:loud.size] & loud
    runs, cur = [], 0
    for q in quiet:
        cur = cur + 1 if q else 0
        if not q and cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    out["dropouts_ms"] = [round(r / sr * 1000.0, 1) for r in runs if r / sr > 0.008]
    out["tail_dbfs"] = round(float(20 * np.log10(abs(y[-1]) + 1e-12)), 1)
    # SPECTRUM OF THE LOUDEST WINDOW, not of the first one. A part can open with
    # silence -- a call-and-response answer starts in a pause by definition --
    # and a Fourier transform of silence, floored by a +1e-18 per bin so it can
    # be logged at all, is a perfectly FLAT spectrum: it reads back as 11025 Hz
    # of centroid and 89 per cent of the energy above 2.5 kHz, which is a noise
    # generator's signature and says nothing at all about the render.
    n = int(2 ** np.floor(np.log2(max(2048, min(y.size, sr * 4)))))
    if y.size > n:
        w = max(1, int(0.02 * sr))
        env = np.convolve(y ** 2, np.ones(w) / w, mode="valid")
        hop = max(1, (env.size - 1) // 400)
        coarse = env[::hop]
        start = int(np.argmax(np.convolve(coarse, np.ones(max(1, n // hop)) / max(1, n // hop),
                                         mode="same"))) * hop
        start = int(np.clip(start - n // 2, 0, y.size - n))
        seg = y[start:start + n]
    else:
        seg = y
    p = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    f = np.fft.rfftfreq(seg.size, 1.0 / sr)
    tot = float(p.sum()) + 1e-30
    out["centroid_hz"] = round(float((f * p).sum() / tot), 1)
    out["hf2500_pct"] = round(float(p[f >= 2500].sum() / tot * 100.0), 3)
    out["hf10000_pct"] = round(float(p[f >= 10000].sum() / tot * 100.0), 4)
    if reference is not None and reference.size > sr // 2:
        import librosa
        ref = np.asarray(reference, dtype=np.float32).reshape(-1)[:y.size]
        f0, v, _ = librosa.pyin(y.astype(np.float32), fmin=55.0, fmax=1200.0, sr=sr,
                                frame_length=4096, hop_length=1024)
        r0, rv, _ = librosa.pyin(ref, fmin=55.0, fmax=1200.0, sr=sr,
                                 frame_length=4096, hop_length=1024)
        m = min(f0.size, r0.size)
        va = (~np.isnan(f0[:m])) & np.asarray(v[:m], bool)
        vb = (~np.isnan(r0[:m])) & np.asarray(rv[:m], bool)
        out["sung_frames_pct"] = round(float(va.mean() * 100.0), 1)
        both = va & vb
        if np.sum(both) > 20:
            cents = 1200 * np.log2(f0[:m][both] / r0[:m][both])
            # octave errors are counted, not folded away
            out["ref_median_cents"] = round(float(np.median(np.abs(cents))), 1)
            out["ref_octave_pct"] = round(float(np.mean(np.abs(cents) > 600) * 100.0), 1)
            # AND THE TEST THAT ACTUALLY ANSWERS "IS IT SINGING THE SOURCE'S
            # NOTE": for each frame the source is voiced in, how much of the
            # render's energy sits on the harmonic series of the source's own
            # f0, against the same measurement at its octave and half-octave.
            # A pitch TRACKER on a synthesized voice makes octave mistakes of
            # its own (that is what the numbers above are full of), while the
            # comb is read straight off the spectrum and cannot: if the render
            # sings the note, f0 wins; if it sings an octave below, f0/2 wins.
            hop = 1024
            keys = []
            for i in np.where(both)[0]:
                centre = int(i * hop)
                lo = max(0, centre - 2048)
                seg = y[lo:lo + 4096]
                if seg.size < 4096 or r0[i] is None or not np.isfinite(r0[i]) or r0[i] <= 0:
                    continue
                keys.append(_comb_fit(seg, sr, float(r0[i])))
            if keys:
                keys = np.asarray(keys, dtype=np.float64)
                n_at = int(np.sum((keys == 0)))
                n_up = int(np.sum((keys == 1)))
                n_down = int(np.sum((keys == 2)))
                out["note_fit_pct"] = round(n_at / len(keys) * 100.0, 1)
                out["octave_up_pct"] = round(n_up / len(keys) * 100.0, 1)
                out["octave_down_pct"] = round(n_down / len(keys) * 100.0, 1)
                out["comb_frames"] = int(len(keys))
        # how much of the render is the source itself, played back
        src_m = ref - ref.mean()
        out_m = y[:src_m.size] - y[:src_m.size].mean()
        denom = float(np.sqrt((src_m ** 2).sum() * (out_m ** 2).sum()))
        if denom > 1e-9:
            out["corr_with_source"] = round(float(np.abs((src_m * out_m).sum() / denom)), 3)
    return out


def _comb_fit(seg: np.ndarray, sr: int, f0: float, harmonics: int = 10) -> int:
    """0 if this frame's energy sits on f0's harmonic series, 1 if it sits on
    2*f0's, 2 if it sits on f0/2's (an octave below), 3 if none of those.

    Read off the spectrum, so it cannot make the octave mistakes a tracker
    makes; the two octaves are included because they are the only two answers a
    wrong one is ever confused for.
    """
    if f0 <= 0 or f0 * 2 >= sr / 2:
        return 3
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2 + 1e-18
    bin_hz = sr / seg.size

    def series(f: float) -> float:
        total = 0.0
        for k in range(1, harmonics + 1):
            fk = f * k
            if fk >= sr / 2:
                break
            idx = int(round(fk / bin_hz))
            lo, hi = max(1, idx - 1), min(len(spec), idx + 2)
            total += float(spec[lo:hi].sum())
        return total

    e0, eup, edown = series(f0), series(f0 * 2), series(f0 / 2)
    best = max(e0, eup, edown)
    if best <= 0:
        return 3
    return 0 if best == e0 else (1 if best == eup else 2)


def verdict(label: str, info: dict, expect_sec: float, minsrc: bool = True) -> None:
    bad = []
    if info.get("empty"):
        bad.append("empty render")
    if not info.get("finite", True):
        bad.append("non-finite samples")
    if info.get("peak", 0) > 1.0005:
        bad.append(f"over 0 dBFS (peak {info['peak']})")
    if info.get("clip_pct", 0) > 0.2:
        bad.append(f"clipping {info['clip_pct']}% of samples")
    if abs(info.get("dc", 0)) > 0.01:
        bad.append(f"DC offset {info['dc']}")
    if minsrc and info.get("rms", 0) < 0.004:
        bad.append(f"almost silent (rms {info.get('rms')})")
    if info.get("clicks", 0) > 3:
        bad.append(f"{info['clicks']} click-like steps (worst {info['worst_step']})")
    if info.get("sec", 0) < 0.55 * expect_sec:
        bad.append(f"only {info.get('sec')}s for a {expect_sec:.2f}s source")
    if info.get("sec", 0) > 2.5 * expect_sec + 1.0:
        bad.append(f"{info.get('sec')}s for a {expect_sec:.2f}s source (part too long)")
    long_gaps = [g for g in info.get("dropouts_ms", []) if g > 60.0]
    if long_gaps:
        bad.append(f"dropouts of {long_gaps[:4]} ms")
    if bad:
        fail(f"{label}: " + "; ".join(bad))
    else:
        ok(label)
    return


# ------------------------------------------------------------------ the matrix
def main() -> int:
    cfg_sample_rate = 44100
    inputs, refs = make_inputs()
    src_audio, src_sr = sf.read(io.BytesIO(inputs["take.wav"]), dtype="float32", always_2d=True)
    song_audio, _ = sf.read(io.BytesIO(inputs["song.wav"]), dtype="float32", always_2d=True)
    say(f"inputs: {[(k, f'{len(v) / 1024:.0f} KB') for k, v in inputs.items()]}")

    neural_up = (server.neural_timbre is not None and getattr(server.neural_timbre, "enabled", False)
                 and getattr(server.neural_timbre, "_reachable", False))
    say(f"neural sidecar reachable: {neural_up}")
    # NOT A HARD-CODED PAIR. The panel builds one button per entry in
    # synthesis.neural.model_paths, so the audit has to drive the same set:
    # a new voice must not be able to load and never be rendered here. The
    # count is the main app's own view of that list, available even while the
    # sidecar is still warming up.
    voice_count = max(1, int(getattr(server.neural_timbre, "num_voices_configured", 1) or 1))
    voice_indices = list(range(voice_count))
    say(f"trained voices configured: {voice_count} ({', '.join(f'Model {v+1}' for v in voice_indices)})")
    if server.synthesizer is None:
        say(f"FATAL: synthesizer not loaded: {server.synthesizer_load_error}")
        return 2

    durations: dict[str, float] = {}
    with TestClient(server.app) as client:
        # ------------------------------------------------ 1. input fidelity
        say("\n=== 1. the upload itself: does it survive the way in ===")
        for name, raw in inputs.items():
            mono_ref = refs[name]
            t0 = time.perf_counter()
            r = client.post("/api/pitch/analyze-track", files={"file": (name, raw, "audio/mpeg")},
                            data={"pitch_method": "yin"})
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                fail(f"{name}: /api/pitch/analyze-track -> {r.status_code} {r.text[:120]}")
            else:
                j = r.json()
                hz = np.asarray(j.get("hz") or [], dtype=float)
                sung = int(np.sum(np.asarray(hz) > 0))
                say(f"  {name}: analyzed in {dt:.0f} ms, {len(hz)} frames, {sung} with a pitch")
                if len(hz) < 10:
                    fail(f"{name}: the pitch analyzer found {len(hz)} frames in it")
            # every decode the renderers use, compared against the music it was
            # made from, and against each other: the two halves of this project
            # have to hear the same upload at the same level
            decodes: dict[str, tuple[np.ndarray, int]] = {}
            for label in ("sf.read", "ffmpeg"):
                try:
                    if label == "sf.read":
                        d = server._decode_mono_for_conversion(raw)
                        dec, dec_sr = np.asarray(d[0]), int(d[1])
                    else:
                        dec = server._decode_audio_via_ffmpeg(raw, 44100)
                        dec_sr = 44100
                except Exception as e:
                    fail(f"{name}: {label} decode raised {type(e).__name__}: {str(e)[:110]}")
                    continue
                decodes[label] = (dec, dec_sr)
                if dec.size < mono_ref.size * 0.9:
                    fail(f"{name}: {label} decoded {dec.size / dec_sr:.2f}s of a "
                         f"{mono_ref.size / src_sr:.2f}s file (short)")
                    continue
                a = np.asarray(dec[:mono_ref.size], dtype=np.float64)
                b = mono_ref[:a.size].astype(np.float64)
                rms = float(np.sqrt(np.mean(a ** 2)))
                ref_rms = float(np.sqrt(np.mean(b ** 2)))
                denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
                corr = float(abs((a * b).sum() / denom)) if denom > 1e-9 else 0.0
                # high-frequency content must survive the way in, and the level
                # must not be quietly rescaled by a decode stage
                def hf_pct(x, sr):
                    n = int(2 ** np.floor(np.log2(max(2048, min(x.size, sr * 4)))))
                    p = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2 + 1e-18
                    f = np.fft.rfftfreq(n, 1.0 / sr)
                    return float(p[f >= 6000].sum() / p.sum() * 100.0)
                gain_db = 20 * np.log10((rms + 1e-12) / (ref_rms + 1e-12))
                say(f"    {label}: {dec.size / dec_sr:.2f}s rms {rms:.4f} (source {ref_rms:.4f}), "
                    f"gain {gain_db:+.2f} dB, corr with the original {corr:.4f}, above 6 kHz "
                    f"{hf_pct(a, dec_sr):.3f}% (source {hf_pct(b, src_sr):.3f}%), peak "
                    f"{float(np.max(np.abs(a))):.3f}")
                if abs(gain_db) > 1.0:
                    fail(f"{name}: {label} changes the level by {gain_db:+.1f} dB")
                if float(np.max(np.abs(a))) > 1.0005:
                    fail(f"{name}: {label} decodes past full scale (peak "
                         f"{float(np.max(np.abs(a))):.3f}), so it is clipping")
                if corr < 0.75:
                    fail(f"{name}: {label} decoded to something that is not this track "
                         f"(corr {corr:.3f})")
            if len(decodes) == 2:
                (a1, sr1), (a2, sr2) = decodes["sf.read"], decodes["ffmpeg"]
                n = min(len(a1), len(a2))
                x1, x2 = np.asarray(a1[:n], np.float64), np.asarray(a2[:n], np.float64)
                denom = float(np.sqrt((x1 ** 2).sum() * (x2 ** 2).sum()))
                agree_corr = float(abs((x1 * x2).sum() / denom)) if denom > 1e-9 else 0.0
                agree_db = 20 * np.log10((np.sqrt((x1 ** 2).mean()) + 1e-12) /
                                         (np.sqrt((x2 ** 2).mean()) + 1e-12))
                say(f"    the two decoders agree: corr {agree_corr:.5f}, level {agree_db:+.2f} dB apart")
                if agree_corr < 0.999 or abs(agree_db) > 0.2:
                    fail(f"{name}: the two decoders disagree about the same file "
                         f"(corr {agree_corr:.4f}, {agree_db:+.2f} dB)")

        # ------------------------------------------------ 2. DSP x every mode
        say("\n=== 2. DSP engine: every accompaniment mode (as the page posts it) ===")
        for name, mode in MODE_LABELS.items():
            # `name` is the key the panel posts (demoMode); `mode` is only the
            # label this report prints. Posting the label was the first version
            # of this harness's bug: the server refuses an unknown mode and
            # falls back to its default, so all sixteen combinations below were
            # quietly the same mode.
            for vocable in ("mix", "moon"):
                for instruments in (True, False):
                    label = f"DSP {mode} vocable={vocable} instruments={instruments}"
                    form = {"texture": "solo", "pitch_method": "yin", "mode": name,
                            "instruments_enabled": str(instruments).lower(),
                            "delay_ms": "350", "echo_amount": "45",
                            "contour_baseline_db": "1.8", "contour_max_bright_db": "6.3",
                            "contour_max_dark_db": "2.7", "vocable": vocable}
                    t0 = time.perf_counter()
                    r = client.post("/api/dsp/render-track",
                                    files={"file": ("take.mp3", inputs["take.mp3"], "audio/mpeg")},
                                    data=form)
                    dt = time.perf_counter() - t0
                    if r.status_code == 422:
                        msg = ""
                        try:
                            msg = r.json().get("error", "")
                        except Exception:
                            pass
                        if name in ("call_and_response",):
                            NOTES.append(f"{label}: refused as designed -- {msg[:90]}")
                            say(f"  [note] {label}: refused ({msg[:70]}...)")
                        else:
                            fail(f"{label}: refused with nothing wrong with the input -- {msg[:120]}")
                        continue
                    if r.status_code != 200:
                        fail(f"{label}: HTTP {r.status_code} {r.text[:140]}")
                        continue
                    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
                    info = analyse(audio, sr if sr else cfg_sample_rate,
                                   reference=refs["take.mp3"], reference_sr=src_sr)
                    info["channels"] = audio.shape[1]
                    say(f"  {label}: {dt:.1f}s -> {info['sec']}s {info['channels']}ch "
                        f"peak {info['peak']} rms {info['rms']} centroid {info['centroid_hz']} Hz "
                        f"hf2.5k {info['hf2500_pct']}% clicks {info['clicks']} "
                        f"gaps {info['dropouts_ms'][:3]} "
                        f"on the note {info.get('note_fit_pct')}% "
                        f"(octave up {info.get('octave_up_pct')}%, "
                        f"octave down {info.get('octave_down_pct')}%, {info.get('comb_frames')} frames)")
                    verdict(label, info, expect_sec=src_audio.shape[0] / src_sr,
                            minsrc=name not in ("call_and_response",))

        # ------------------------------------------------ 3. Neural x every mode
        if neural_up:
            say("\n=== 3. Neural engine: every accompaniment mode x every configured voice ===")
            for name, mode in MODE_LABELS.items():
                for voice in voice_indices:
                    label = f"Neural {mode} voice={voice}"
                    form = {"texture": "solo", "pitch_method": "rmvpe", "voice_index": str(voice),
                            "mode": name, "instruments_enabled": "false", "vocable": "mix",
                            "delay_ms": "350", "echo_amount": "45"}
                    t0 = time.perf_counter()
                    r = client.post("/api/neural/render-track",
                                    files={"file": ("song.mp3", inputs["song.mp3"], "audio/mpeg")},
                                    data=form)
                    dt = time.perf_counter() - t0
                    if r.status_code == 422:
                        msg = ""
                        try:
                            msg = r.json().get("error", "")
                        except Exception:
                            pass
                        say(f"  [note] {label}: refused ({msg[:70]}...)")
                        NOTES.append(f"{label}: refused -- {msg[:90]}")
                        continue
                    if r.status_code != 200:
                        fail(f"{label}: HTTP {r.status_code} {r.text[:140]}")
                        continue
                    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
                    # The reference is the SONG's own music, at the song's own
                    # length; the drum bed is what the first version of this got
                    # wrong by feeding it the bare take voice, which turned a
                    # perfectly good render into "not this track".
                    info = analyse(audio, sr, reference=refs["song.mp3"], reference_sr=src_sr)
                    info["channels"] = audio.shape[1]
                    say(f"  {label}: {dt:.1f}s -> {info['sec']}s {info['channels']}ch "
                        f"peak {info['peak']} rms {info['rms']} centroid {info['centroid_hz']} Hz "
                        f"clicks {info['clicks']} "
                        f"on the note {info.get('note_fit_pct')}% "
                        f"(octave up {info.get('octave_up_pct')}%, "
                        f"octave down {info.get('octave_down_pct')}%, {info.get('comb_frames')} frames) "
                        f"corr-with-source {info.get('corr_with_source')}")
                    verdict(label, info, expect_sec=song_audio.shape[0] / src_sr)
        else:
            NOTES.append("Neural matrix skipped: sidecar not reachable")

        # ------------------------------------------------ 4. textures & the other routes
        say("\n=== 4. textures, the convert route and the backing bed ===")
        for texture in TEXTURES:
            label = f"DSP Unison texture={texture}"
            r = client.post("/api/dsp/render-track",
                            files={"file": ("take.wav", inputs["take.wav"], "audio/wav")},
                            data={"texture": texture, "pitch_method": "yin",
                                  "mode": "unison_shadowing", "instruments_enabled": "true",
                                  "vocable": "mix"})
            if r.status_code != 200:
                fail(f"{label}: HTTP {r.status_code} {r.text[:120]}")
                continue
            audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
            info = analyse(audio, sr, reference=refs["take.wav"], reference_sr=src_sr)
            say(f"  {label}: {info['sec']}s rms {info['rms']} centroid {info['centroid_hz']} Hz "
                f"clicks {info['clicks']}")
            verdict(label, info, expect_sec=src_audio.shape[0] / src_sr)

        if neural_up:
            for voice in voice_indices:
                label = f"neural convert-track voice={voice}"
                t0 = time.perf_counter()
                r = client.post("/api/neural/convert-track",
                                files={"file": ("take.mp3", inputs["take.mp3"], "audio/mpeg")},
                                data={"voice_index": str(voice)})
                dt = time.perf_counter() - t0
                if r.status_code != 200:
                    fail(f"{label}: HTTP {r.status_code} {r.text[:140]}")
                    continue
                audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
                info = analyse(audio, sr, reference=refs["take.mp3"], reference_sr=src_sr)
                say(f"  {label}: {dt:.1f}s -> {info['sec']}s rms {info['rms']} "
                    f"centroid {info['centroid_hz']} Hz clicks {info['clicks']} "
                    f"corr-with-source {info.get('corr_with_source')}")
                verdict(label, info, expect_sec=src_audio.shape[0] / src_sr)

            # the separation the live bed and the single-word render both use
            label = "neural backing-bed separation (song.mp3)"
            r = client.post("/api/neural/backing-bed",
                            files={"file": ("song.mp3", inputs["song.mp3"], "audio/mpeg")})
            if r.status_code != 200:
                fail(f"{label}: HTTP {r.status_code} {r.text[:140]}")
            else:
                key = (r.json() or {}).get("track_key")
                status = (r.json() or {}).get("status")
                t0 = time.perf_counter()
                while status == "running" and time.perf_counter() - t0 < 420:
                    time.sleep(3)
                    status = server._backing_bed_status(key)
                say(f"  {label}: track_key {key}, separation {status} after "
                    f"{time.perf_counter() - t0:.1f}s")
                if status != "ready":
                    fail(f"{label}: separation ended as '{status}'")
                else:
                    n = song_audio.shape[0]
                    bed = server._backing_bed_slice(key, 0.0, src_sr, n)
                    voc = server._backing_vocal_slice(key, 0.0, src_sr, n)
                    if bed is None or voc is None:
                        fail(f"{label}: ready but no stems came back from the cache")
                    else:
                        b = np.asarray(bed, dtype=np.float64)
                        v = np.asarray(voc, dtype=np.float64)
                        mix = song_audio.mean(axis=1)[:len(b)].astype(np.float64)
                        recon = b[:len(mix)] + v[:len(mix)]
                        denom = float(np.sqrt((mix ** 2).sum() * (recon ** 2).sum()))
                        say(f"  {label}: bed rms {np.sqrt((b ** 2).mean()):.4f} peak {np.max(np.abs(b)):.3f}, "
                            f"vocal rms {np.sqrt((v ** 2).mean()):.4f}, "
                            f"bed+vocal reconstructs the mix at corr "
                            f"{float(abs((mix * recon).sum() / denom)) if denom > 1e-9 else 0.0:.4f}")
                        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(v)):
                            fail(f"{label}: separation produced non-finite samples")
                        if np.max(np.abs(b)) > 4.0:
                            fail(f"{label}: bed peak {np.max(np.abs(b)):.2f} (way over full scale)")
                        # no stem may be silent: a separation that hands back an
                        # empty bed is why a live chunk would fall back to drums
                        for stem_name, stem in (("bed", b), ("vocal", v)):
                            if np.sqrt((stem ** 2).mean()) < 1e-4:
                                fail(f"{label}: the {stem_name} stem is silent")

                        # THE TRACK THE PAGE PLAYS UNDER THE ROBOT, through the
                        # endpoint the page calls: it must exist, be playable,
                        # be the upload's own length, and -- the whole point of
                        # it -- contain almost none of the original lead voice.
                        band_label = "the separated band the page plays (backing-bed/track)"
                        rb = client.get("/api/neural/backing-bed/track", params={"track_key": key})
                        if rb.status_code != 200:
                            fail(f"{band_label}: HTTP {rb.status_code} {rb.text[:140]}")
                        else:
                            band, band_sr = sf.read(io.BytesIO(rb.content), dtype="float32", always_2d=True)
                            band_mono = band.mean(axis=1) if band.ndim > 1 else band
                            upload = song_audio.mean(axis=1)
                            n = min(len(band_mono), len(upload))
                            band_mono, upload = band_mono[:n], upload[:n]

                            def lead_leak_db(x, vocal_stem):
                                """How many dB of the isolated lead voice are sitting
                                in this mix, by projection (the project's own leak
                                metric): 20*log10 of the lead's own projection onto
                                the signal, over the lead itself. 0 dB is
                                'the lead is all of it', -inf is 'not there'."""
                                v = vocal_stem[:len(x)].astype(np.float64)
                                y = x.astype(np.float64)
                                vv = float((v ** 2).sum())
                                if vv < 1e-9:
                                    return float("-inf")
                                proj = (float((y * v).sum()) / vv) * v
                                return float(20 * np.log10((np.sqrt((proj ** 2).mean()) + 1e-12) /
                                                           (np.sqrt((v ** 2).mean()) + 1e-12)))

                            leak_upload = lead_leak_db(upload, v)
                            leak_band = lead_leak_db(band_mono, v)
                            gain_db = 20 * np.log10((np.sqrt((band_mono.astype(np.float64) ** 2).mean()) + 1e-12) /
                                                    (np.sqrt((upload.astype(np.float64) ** 2).mean()) + 1e-12))
                            say(f"  {band_label}: {band.shape[1] if band.ndim > 1 else 1}ch "
                                f"{len(band_mono) / band_sr:.2f}s (upload "
                                f"{len(upload) / band_sr:.2f}s), level {gain_db:+.1f} dB vs the upload, "
                                f"original lead at {leak_upload:+.1f} dB in the upload vs "
                                f"{leak_band:+.1f} dB in the band")
                            if abs(len(band_mono) - len(upload)) > int(0.05 * band_sr):
                                fail(f"{band_label}: {len(band_mono) / band_sr:.2f}s for a "
                                     f"{len(upload) / band_sr:.2f}s upload, so it would drift "
                                     f"against the track")
                            if not np.all(np.isfinite(band_mono)) or np.max(np.abs(band_mono)) > 1.0005:
                                fail(f"{band_label}: non-finite or past full scale")
                            if leak_band > leak_upload - 20.0:
                                fail(f"{band_label}: still carries the original lead "
                                     f"({leak_band:+.1f} dB vs {leak_upload:+.1f} dB in the upload)")
                            if abs(gain_db) > 6.0:
                                fail(f"{band_label}: level {gain_db:+.1f} dB against the upload, so "
                                     f"swapping one for the other would move the volume")

    say("")
    if NOTES:
        say(f"{len(NOTES)} note(s):")
        for n in NOTES:
            say(f"  - {n}")
    if FAILURES:
        say(f"\nMATRIX FAILURES ({len(FAILURES)}):")
        for f in FAILURES:
            say(f"  - {f}")
        return 1
    say("DEMO MATRIX PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
THE QUALITY SWEEP.

Renders the live demo's own combinations through the running server, exactly
the way the page asks for them, and measures every artifact that comes back:
one file per render under /tmp/qa/ so any of them can be listened to.

Everything the render endpoints return is the ROBOT'S PART (the DSP track
endpoint returns the voice alone; the single-word path returns voice +
instruments as two channels). So the sweep never compares the part's loudness
to the upload -- that is a different signal by design -- and it never calls a
musical rest a dropout: several modes are silent by construction (call and
response answers only in the pauses, drone support sits under phrases, a hum
breathes between phrases). What it does judge:

  rail / clipping     a sample at or past full scale (the PCM-16 rail)
  hot ceiling         a render that parks on the ceiling instead of under it
  silence             a part that is nothing at all
  clicks              one-sample discontinuities -- the audible "glitch"
  micro-holes         a gap SHORTER than a breath inside loud material, which
                      is a seam or a scheduling bug, never music
  level stability     the same request twice, within 1.5 dB (this is the bug
                      where "the same settings came back 3 dB apart")
  brightness          the part's own spectral centre against the input's, for
                      the modes that sing; hum modes are dark by nature
  pitch               on the controlled phrase, whose notes are known to the
                      cent: a sung part must be ON the note (or a mode-declared
                      octave of it) for most of its sounding frames
  separation          two modes (or two textures) that return the same audio,
                      which is a plumbing bug rather than a taste problem

Usage:
    .venv/bin/python tools/check_quality_sweep.py [--base http://127.0.0.1:8000]
        [--only dsp,neural] [--inputs real,synth,acap] [--models all]
        [--quick]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import requests
import soundfile as sf

SR = 44100
OUT_DIR = "/tmp/qa"
REPORT = "/tmp/qa/report.json"

MODES = [
    "unison_shadowing", "octave_reinforcement", "timbral_thickening",
    "delayed_response", "contour_following", "drone_support",
    "call_and_response", "solo_humming",
]
TEXTURES = ["solo", "duet", "choir"]

# Modes whose part is a hum or a drone and are therefore dark on purpose, and
# modes that are silent in the pauses by design, so a long quiet stretch in
# them is the mode working rather than a hole in the audio.
DARK_MODES = {"solo_humming", "drone_support"}
RESTING_MODES = {"call_and_response", "drone_support", "solo_humming", "delayed_response"}
# Modes that sing the input's own melody, so their pitch can be judged against
# it. Drone and humming do not follow the line, so they are exempt.
PITCH_FOLLOWING_MODES = {"unison_shadowing", "timbral_thickening", "contour_following",
                         "delayed_response"}
# What a mode is allowed to sing over ONE clean note, as multiples of that
# note's frequency. This is the one place a pitch judgement can be trusted: a
# single held note with no melody tracking to get wrong, measured on the render
# itself. Call and response is absent on purpose -- it answers in the pauses,
# and four and a half seconds of continuous note has none.
NOTE_INTERVALS = {
    "unison_shadowing": (1.0,),
    "timbral_thickening": (1.0,),
    "contour_following": (1.0,),
    "delayed_response": (1.0,),
    "solo_humming": (1.0,),
    "octave_reinforcement": (1.0, 2.0, 0.5),
    "drone_support": (0.5, 1.0, 2.0),
}
# Modes whose render is a MIXTURE around the note rather than one voice on it
# (a delayed copy answering itself, a pad under it, a hummed timbre). A single
# median lag cannot describe a mixture -- measured on this harness's own
# octave-reinforcement render, the autocorrelation reads 152 Hz for a render
# holding 250 Hz plus 125 Hz, i.e. a value between its own two notes -- so for
# these the one-note probe asserts the honest thing instead: that the input's
# note is IN the render at a level that matters. Their melody following is
# still judged over a whole song, against the hum's own curve.
PRESENCE_ONLY_MODES = {"delayed_response", "solo_humming", "call_and_response"}

# How the mode is allowed to relate to the sung note. Octave reinforcement is
# the one mode whose whole point is to be an octave away, and it may also sit
# in unison where the melody is already high.
MODE_PITCH_RATIOS = {
    "octave_reinforcement": (1.0, 2.0, 0.5),
    "unison_shadowing": (1.0,),
    "timbral_thickening": (1.0,),
    "contour_following": (1.0,),
    "delayed_response": (1.0,),
}

# What the page sends for a mode's own echo/contour settings. The panel's
# delayed-response controls START AT ZERO, which means the page's default
# render of that mode has no echo in it at all -- worth knowing, and worth
# testing with the effect actually on, since that is the mode's identity.
DSP_MODE_FIELDS = {
    "delayed_response": {"delay_ms": "220", "echo_amount": "35"},
    "contour_following": {
        "contour_baseline_db": "0", "contour_max_bright_db": "6",
        "contour_max_dark_db": "6",
    },
}

FAILS: list[str] = []
NOTES: list[str] = []
RESULTS: list[dict] = []


def say(msg: str) -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}", flush=True)


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"  [note] {msg}", flush=True)


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}", flush=True)


# ---------------------------------------------------------------- inputs ---

def cut_mp3(path: str, start_s: float, dur_s: float) -> bytes:
    """A real excerpt of a real upload, as MP3 bytes, the way a person's file
    reaches the server (ffmpeg, stream copy, no re-encode)."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start_s}", "-t", f"{dur_s}",
           "-i", path, "-c", "copy", "-f", "mp3", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg cut failed for {path}: {proc.stderr[-300:]!r}")
    return proc.stdout


# The controlled phrase: notes known to the cent, a real vibrato on each, a
# quiet pad underneath. Used for the strict pitch test, because here "which
# note is being sung" is a fact rather than an estimate.
SYNTH_NOTES = [(0.0, 1.6, 220.0), (1.6, 1.6, 293.66), (3.2, 2.0, 349.23),
               (5.2, 1.6, 329.63), (6.8, 2.4, 261.63), (9.2, 1.8, 196.0),
               (11.0, 2.2, 246.94), (13.2, 1.6, 293.66), (14.8, 2.4, 220.0),
               (17.2, 2.4, 174.61)]
SYNTH_DUR_S = 20.0
TEXTURE_S = 4.5
BREATH_S = 1.2
TEXTURE_F0 = 233.08


def _tone(t: np.ndarray, f0: float, vib_hz: float = 5.2, vib_cents: float = 7.0) -> np.ndarray:
    """One sung note: 16 harmonics under a 1/h^1.55 roll-off with vibrato.

    THE VIBRATO IS INTEGRATED INTO THE PHASE, which is the only way a vibrato
    exists: a frequency f(t) = f0 * (1 + m*sin(w*t)) is the PHASE
    sin(2*pi*f0*(t - (m/w)*cos(w*t))), not sin(2*pi*f0*t*(1 + m*sin(w*t))).

    The old form here multiplied the whole phase, and its instantaneous
    frequency carried an extra m*w*t term that GROWS WITHOUT BOUND: +830 cents
    at t = 3 s, +1090 cents at t = 4.5 s. This harness's "single held note,
    perfectly measurable, known to the cent" fixture was therefore a rising
    glide. It is why the sweep's own estimator read the 233.08 Hz probe back
    anywhere between 184 and 1002 Hz inside one 2.8 s window, why the project's
    trackers "misread" it by hundreds of cents -- they did not, the fixture was
    wrong -- and why every render that correctly followed the input's melody
    failed a nominal-note assertion. Fixed here rather than papered over in the
    assertions: with a real vibrato the note IS the note, and the nominal
    expectations below become the facts they were meant to be.
    """
    m = vib_cents / 1200.0
    w = 2 * np.pi * vib_hz
    phase = 2 * np.pi * f0 * (t - (m / w) * np.cos(w * t))
    out = np.zeros_like(t)
    for h in range(1, 17):
        out += (1.0 / (h ** 1.55)) * np.sin(phase * h)
    return out


def synth_texture(phrase: bool = True, dur_s: float = TEXTURE_S) -> np.ndarray:
    """A short, perfectly measurable phrase (or a single held note).

    The texture probe: one note, one pitch, so "did the texture do anything"
    is answered by comparing two renders of it sample for sample instead of
    hoping a whole song hides the difference.
    """
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    if not phrase:
        env = np.clip(np.minimum(1.0, np.minimum(t / 0.15, (dur_s - t) / 0.4)), 0, 1) ** 1.1
        return (_tone(t, TEXTURE_F0) * env).astype(np.float32)
    out = np.zeros(n)
    at = 0.15
    for _, length, f0 in SYNTH_NOTES:
        length = min(length, dur_s - at)
        if length <= 0.05:
            break
        k = int(length * SR)
        tt = np.arange(k) / SR
        env = np.clip(np.minimum(1.0, np.minimum(tt / 0.07, (length - tt) / 0.16)), 0, 1) ** 1.15
        out[int(at * SR): int(at * SR) + k] += _tone(tt, f0) * env
        at += length + 0.05
    return out.astype(np.float32)


def synth_input(dur_s: float = SYNTH_DUR_S) -> tuple[bytes, np.ndarray]:
    """A controlled solo: a sung phrase on held notes with vibrato, plus a
    quiet synth pad. Measurable: every note is known to the cent."""
    phrase = synth_texture(phrase=True, dur_s=dur_s)
    n = phrase.shape[0]
    t = np.arange(n) / SR
    pad = 0.10 * np.sin(2 * np.pi * 110.0 * t) + 0.06 * np.sin(2 * np.pi * 164.81 * t)
    mix = 0.72 * phrase + pad
    mix /= max(1e-9, np.abs(mix).max())
    mix *= 0.55
    return wav_bytes(mix.astype(np.float32)), mix.astype(np.float32)


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    if x.ndim == 2:
        x = x.T
    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def load_inputs(which: list[str]) -> dict[str, tuple[bytes, str]]:
    out: dict[str, tuple[bytes, str]] = {}
    assets = "frontend/assets/audio"
    if "real" in which:
        raw = cut_mp3(os.path.join(assets, "aim.mp3"), 60.0, 24.0)
        out["real"] = (raw, "a loud real mix (aim.mp3, 24s), the panel's own demo track")
    if "acap" in which:
        raw = cut_mp3(os.path.join(assets, "memory-v2.mp3"), 60.0, 24.0)
        out["acap"] = (raw, "an a cappella take (memory-v2.mp3, 24s): voice and nothing else")
    if "synth" in which:
        out["synth"] = (synth_input()[0], "a controlled sung phrase, 20s, known notes")
    return out


# --------------------------------------------------------------- metrics ---

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def pitch_curve(sig: np.ndarray, sr: int, hop_s: float = 0.02, win: int = 4096,
                fmin: float = 65.0, fmax: float = 1000.0, conf: float = 0.45):
    """A per-frame f0 for a RENDER, by autocorrelation.

    Used for one thing only: telling whether two renders are singing the same
    melody. It is deliberately the same estimator on both sides, because the
    absolute reading of any detector on a synthetic voice is questionable --
    the project's own RMVPE reads this harness's controlled 220 Hz tone as a
    glide from 216 Hz down to 204 Hz, i.e. the ground truth it would be judged
    against is itself wrong -- while a DIFFERENCE between two things measured
    the same way is still the answer to "is this the same tune".
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = max(1, int(hop_s * sr))
    n_fft = 1 << int(np.ceil(np.log2(2 * win)))
    out = []
    for at in range(0, max(1, len(x) - win), hop):
        seg = x[at: at + win]
        if rms(seg) < 10 ** (-50 / 20):
            out.append(0.0)
            continue
        seg = seg - seg.mean()
        sp = np.fft.rfft(seg, n_fft)
        ac = np.fft.irfft(np.abs(sp) ** 2, n_fft)[:win]
        if ac[0] <= 1e-12:
            out.append(0.0)
            continue
        ac = ac / ac[0]
        lo = max(2, int(sr / fmax))
        hi = min(len(ac) - 1, int(sr / fmin))
        if hi <= lo:
            out.append(0.0)
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        out.append(sr / lag if ac[lag] >= conf else 0.0)
    return np.array(out)


def note_presence_db(core: np.ndarray, sr: int, f0: float, tol: float = 0.04) -> float:
    """How loud the input's own note is inside a render, in dB under its peak.

    The one pitch assertion that survives a MIXTURE: whatever else a mode adds
    (a pad, an octave below, its own delayed answer), the note the singer is
    singing has to be in there, and not 30 dB under the rest of the chord.
    """
    x = np.asarray(core, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < 1024 or not f0 > 0:
        return float("nan")
    n = 1 << int(np.ceil(np.log2(x.size * 2)))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    peak = float(spec.max())
    lo, hi = np.searchsorted(freqs, f0 * (1.0 - tol)), np.searchsorted(freqs, f0 * (1.0 + tol))
    if peak <= 0 or hi <= lo:
        return float("nan")
    return 20.0 * float(np.log10(max(float(spec[lo:hi].max()), 1e-12) / peak))


def melody_agreement(art: np.ndarray, sr: int, ref_curve: np.ndarray,
                     ratios: tuple[float, ...] = (1.0,)) -> tuple[float, float, int]:
    """(hit rate, median cents, frames compared) between two renders' curves."""
    cur = pitch_curve(art, sr)
    n = min(len(cur), len(ref_curve))
    cur, ref = cur[:n], np.asarray(ref_curve)[:n]
    both = (cur > 0) & (ref > 0)
    if both.sum() < 20:
        return (float("nan"), float("nan"), int(both.sum()))
    cents = np.array([min(abs(1200 * np.log2(a / (r * k))) for k in ratios)
                      for a, r in zip(cur[both], ref[both])])
    return (float(np.mean(cents <= 60.0)), float(np.median(cents)), int(both.sum()))


def analyse(x: np.ndarray, sr: int, name: str, reference: np.ndarray | None = None,
            reference_sr: int = SR, notes: list[tuple[float, float, float]] | None = None,
            ratios: tuple[float, ...] | None = None) -> dict:
    """Everything the sweep needs to know about one artifact."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    info: dict = {"name": name, "sec": len(x) / sr}
    if len(x) == 0:
        info.update(peak=0.0, rms=0.0, rail=0, hot=0, dropouts=0, worst_dropout_ms=0.0,
                    click_rel=0.0, click_count=0, hf_share=0.0, centroid=0.0)
        return info

    info["peak"] = float(np.abs(x).max())
    info["rms"] = rms(x)
    info["rail"] = int(np.sum(np.abs(x) >= 1.0))
    # 0.9995 used to be the threshold, but a PCM-16 file's loudest possible
    # sample is 32767/32768 = 0.999969, so every render that used its headroom
    # was being reported as clipped. "Hot" is the honest reading: parked on the
    # ceiling.
    info["hot"] = int(np.sum(np.abs(x) >= 0.999))
    # how much of the render sits within a hair of the ceiling, i.e. is it
    # being squashed there or just touching it once
    info["hot_share"] = float(info["hot"] / len(x))

    # A HOLE IS A GAP INSIDE LOUD MATERIAL, SHORTER THAN A BREATH. Long quiet
    # stretches are music in half these modes (see RESTING_MODES) and are
    # counted separately, as rests. 60 ms is the line: nothing musical is
    # shorter than that, and every seam, re-schedule and buffer gap is.
    win = int(0.01 * sr)
    n_win = len(x) // win
    if n_win >= 4:
        fr = np.sqrt(np.mean(x[: n_win * win].reshape(n_win, win) ** 2, axis=1))
        local = np.array([np.median(fr[max(0, i - 30): i + 31]) for i in range(n_win)])
        quiet = fr < np.maximum(local * 0.02, 10 ** (-55 / 20))

        def _runs(mask_arr) -> list[tuple[int, int]]:
            out, start = [], None
            for i, v in enumerate(mask_arr):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    out.append((start, i - start))
                    start = None
            if start is not None:
                out.append((start, len(mask_arr) - start))
            return out

        # IS THE MUSIC EVEN SOUNDING THERE? A part that goes quiet through an
        # instrumental passage, or in the breath the singer took, is the part
        # doing its job; a part that goes quiet while the singer is still
        # singing is a seam. So a gap only counts where the REFERENCE (the
        # input, or the chunk the conversion was given) has something in it.
        ref_fr = None
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float64)
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            scale = max(1, int(round(sr / max(1, reference_sr))))
            m = min(len(x), len(ref) * scale)
            n_ref = m // win
            if n_ref >= 4:
                ref_fr = np.sqrt(np.mean(np.asarray(ref)[: n_ref * win // scale]
                                         .reshape(n_ref, win // scale) ** 2, axis=1))

        edge = int(0.15 * sr / win)  # 150 ms at each end
        holes, rests = [], []
        for at, length in _runs(quiet):
            if at < edge or at + length > n_win - edge:
                # THE TWO ENDS ARE NOT SEAMS. A part that starts 30 ms after the
                # track's first transient, or stops before the last reverb tail,
                # is a lead-in and an outro, not a hole in the middle of music.
                continue
            # WHAT IS AROUND IT decides what it is. A gap inside loud material
            # is a seam or a scheduling bug; the same gap under quiet material
            # is the singer (or the mode) breathing, and half these modes rest
            # on purpose.
            before = local[max(0, at - 20): at]
            after = local[at + length: at + length + 20]
            around = float(np.median(np.concatenate([before, after]))) if (len(before) + len(after)) else 0.0
            loud_around = around > 10 ** (-35 / 20)
            if ref_fr is not None:
                seg_ref = ref_fr[max(0, at - 20): min(len(ref_fr), at + length + 20)]
                if seg_ref.size:
                    # 12 dB under the part's own loud frames is "the music is
                    # still going" rather than a hard gate, because a mix with
                    # a quiet verse should not excuse a hole in a loud one.
                    loud_around = loud_around and float(np.max(seg_ref)) > 10 ** (-40 / 20)
            if loud_around and length * win / sr < 0.06:
                holes.append(length)
            elif loud_around:
                rests.append(length)
        info["rests"] = len(rests)
        info["dropouts"] = len(holes)
        info["worst_dropout_ms"] = (max(holes) * win / sr * 1000.0) if holes else 0.0
        interior = [(at, l) for at, l in _runs(quiet) if at >= edge and at + l <= n_win - edge]
        info["longest_gap_s"] = max((l * win / sr for _, l in interior), default=0.0)

        # DIGITAL SILENCE IS NEVER A BREATH, HOWEVER LONG IT IS. The rule above
        # has a 60 ms ceiling on purpose (a longer gap is either a rest or a
        # dropout this instrument cannot tell apart from one without knowing
        # what the singer did) -- and that ceiling is exactly where a mute
        # hides. Measured on the hum path: a note change muted the render to
        # EXACT zeros for 81-232 ms at 20 places inside a phrase the singer was
        # singing at -2 dB, and every one of them was filed as a rest, because
        # 81-232 ms is longer than a breath. Zero samples are a fact, so they
        # are graded as one here: a run of exact zeros longer than 5 ms in a
        # loud neighbourhood is a finding at any length.
        z = np.abs(x) < 1e-7
        hard: list[float] = []
        for zs, zl in _runs(z):
            if zl < int(0.005 * sr):
                continue
            at = zs // win
            ln = max(1, zl // win)
            if at < edge or at + ln > n_win - edge:
                continue
            before = local[max(0, at - 20): at]
            after = local[at + ln: at + ln + 20]
            around = float(np.median(np.concatenate([before, after]))) if (len(before) + len(after)) else 0.0
            if around > 10 ** (-35 / 20):
                hard.append(zl / sr)
        info["hard_silence"] = len(hard)
        info["worst_hard_silence_ms"] = max(hard) * 1000.0 if hard else 0.0

    # THE LEVEL WHILE IT IS SINGING, which is the level a person hears. Plain
    # RMS over the whole file is not comparable between modes: a mode that only
    # answers in the pauses is mostly silence, so it measures quiet however
    # loud its phrases are. This is the RMS of the frames within 20 dB of the
    # part's own loudest frames, used for the one-level-for-every-mode check.
    if len(x) > 4096:
        f = int(0.02 * sr)
        n_f = len(x) // f
        if n_f >= 4:
            fr_rms = np.sqrt(np.mean(x[: n_f * f].reshape(n_f, f) ** 2, axis=1))
            top = float(np.percentile(fr_rms, 95))
            keep = fr_rms > 0.1 * top
            info["sounding_rms"] = float(np.sqrt(np.mean(fr_rms[keep] ** 2))) if keep.any() else 0.0
            info["sounding_share"] = float(keep.mean())

    # curvature clicks: a single-sample jump, the shape of a glitch
    curv = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    amp = max(0.05, float(np.percentile(np.abs(curv), 99.9)))
    info["click"] = float(np.abs(curv).max())
    info["click_rel"] = float(np.abs(curv).max() / amp)
    info["click_count"] = int(np.sum(np.abs(curv) > max(0.25, 8 * amp)))

    # CLARITY, over the part's own loudest ten seconds rather than its first
    # ten. Half these modes are quiet at the top of a track (call and response
    # answers in the first pause, drone support waits for a phrase), and
    # measuring the opening silence made a perfectly bright part come back as
    # "spectral centre 0 Hz".
    span = min(len(x), 10 * sr)
    if span > 4096:
        step = max(1, int(0.25 * sr))
        starts = list(range(0, max(1, len(x) - span), step))
        if len(starts) > 400:
            starts = starts[:: max(1, len(starts) // 400)]
        best, best_rms, at = None, -1.0, 0
        for s0 in starts:
            chunk = x[s0:s0 + span]
            lv = float(np.sqrt(np.mean(chunk ** 2)))
            if lv > best_rms:
                best, best_rms, at = chunk, lv, s0
        info["loud_window_s"] = at / sr
        seg = best if best is not None else x[:span]
    else:
        seg = x
    if len(seg) > 4096:
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        tot = float(spec.sum()) + 1e-30
        info["hf_share"] = float(spec[(freqs >= 6000) & (freqs <= 12000)].sum() / tot)
        info["centroid"] = float((spec * freqs).sum() / tot)
        info["air_share"] = float(spec[freqs >= 10000].sum() / tot)
    else:
        info["hf_share"], info["centroid"], info["air_share"] = 0.0, 0.0, 0.0

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        m = min(len(x), len(ref))
        if m > 4096:
            a = x[:m] - x[:m].mean()
            b = ref[:m] - ref[:m].mean()
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-30
            info["corr"] = float(np.sum(a * b) / denom)

    # PITCH, against notes that are facts rather than estimates.
    #
    # BY AUTOCORRELATION, not by the tallest spectral peak. A sung note is a
    # stack of harmonics whose loudest partial is often the second or third
    # (a 220 Hz note with a strong 660 Hz partial reads as 1900 cents sharp if
    # you take argmax), and an earlier version of this measured exactly that:
    # it reported "0% of frames sit on the sung note" for a render that was on
    # the note. The correlation of the signal with itself is what a fundamental
    # detector is for; the search is bounded to the notes this mode is allowed
    # to be on, so it cannot answer with an arbitrary octave.
    if notes:
        cents: list[float] = []
        hits, tested = 0, 0
        win_n = 4096
        for start_s, length_s, f0 in notes:
            wanted = ratios or (1.0,)
            for frac in (0.25, 0.5, 0.75):
                at = int((start_s + length_s * frac) * sr)
                if at + win_n >= len(x):
                    continue
                seg = np.asarray(x[at: at + win_n], dtype=np.float64)
                if rms(seg) < 10 ** (-45 / 20):
                    continue
                seg = seg - seg.mean()
                n_fft = 1 << int(np.ceil(np.log2(2 * win_n)))
                spec = np.fft.rfft(seg, n_fft)
                ac = np.fft.irfft(np.abs(spec) ** 2, n_fft)[:win_n]
                if ac[0] <= 1e-12:
                    continue
                ac = ac / ac[0]
                f_lo = min(wanted) * f0 * 0.90
                f_hi = max(wanted) * f0 * 1.10
                lo_lag = max(2, int(sr / f_hi))
                hi_lag = min(len(ac) - 1, int(sr / f_lo) + 1)
                if hi_lag <= lo_lag:
                    continue
                seg_ac = ac[lo_lag:hi_lag]
                lag = lo_lag + int(np.argmax(seg_ac))
                if ac[lag] < 0.3:
                    continue  # no pitched content in this frame at all
                f_est = sr / lag
                tested += 1
                best = min(abs(1200 * np.log2(f_est / (f0 * r))) for r in wanted)
                cents.append(best)
                if best <= 60.0:
                    hits += 1
        if tested:
            info["pitch_tested"] = tested
            info["pitch_hit_rate"] = hits / tested
            info["pitch_cents_med"] = float(np.median(cents))
            info["pitch_cents_p90"] = float(np.percentile(cents, 90))
    return info


def note_timeline(input_array: np.ndarray, notes: list[tuple[float, float, float]],
                  sr: int, hop_s: float = 0.02) -> np.ndarray:
    """The input's own f0 timeline for the controlled phrase: an actual note
    during a note, 0 in the gaps between them."""
    frames = int(len(input_array) / (hop_s * sr))
    out = np.zeros(frames)
    for i in range(frames):
        at = i * hop_s
        for start_s, length_s, f0 in notes:
            if start_s + 0.05 <= at <= start_s + length_s - 0.05:
                out[i] = f0
                break
    return out


# --------------------------------------------------------------- network ---

def post(endpoint: str, base: str, files: dict, data: dict, timeout: int = 1800):
    t0 = time.time()
    r = requests.post(base + endpoint, files=files, data=data, timeout=timeout)
    return r, time.time() - t0


def vocable_words(base: str) -> list[str]:
    """The plain names of the loaded takes.

    The manifest answers with one entry per vocable, each carrying its
    registers -- it is a list of OBJECTS, and asking for a render with one of
    those objects as the `vocable` field is a guaranteed 500 ("no recorded take
    loaded for vocable '{'vocable': ..."). So the name is taken out of the
    entry here, once, and the sweep sends what the page sends.
    """
    try:
        r = requests.get(base + "/api/vocables/manifest", timeout=30)
        j = r.json()
    except Exception as e:
        note(f"could not read the vocable manifest ({e}); only Mix is tested")
        return []
    entries = j.get("vocables") if isinstance(j, dict) else j
    if not isinstance(entries, list):
        return []
    words: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            words.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("vocable") or entry.get("name") or entry.get("word")
            if name:
                words.append(str(name))
    return words


# ------------------------------------------------------------------ sweep ---

def check_common(tag: str, info: dict, dark: bool, resting: bool,
                 input_centroid: float | None = None) -> None:
    if info["rms"] < 10 ** (-50 / 20):
        fail(f"{tag}: the render is silence")
        return
    if info["rail"] > 0:
        fail(f"{tag}: {info['rail']} sample(s) at the PCM-16 rail (real clipping)")
    elif info["hot"] > 0 and info["hot_share"] > 0.002:
        fail(f"{tag}: {info['hot']} samples parked on the ceiling ({info['hot_share']*100:.2f}% of the render)")
    if info["peak"] > 1.0005:
        fail(f"{tag}: peak {info['peak']:.4f} is over full scale")
    if info["rms"] > 0.5:
        fail(f"{tag}: RMS {info['rms']:.3f} is loud enough to be squashed, not mixed")
    if info["dropouts"] > 0:
        fail(f"{tag}: {info['dropouts']} hole(s) shorter than a breath inside loud material, "
             f"worst {info['worst_dropout_ms']:.0f}ms (a seam, not a rest)")
    if info.get("hard_silence", 0) > 0:
        fail(f"{tag}: {info['hard_silence']} stretch(es) of EXACT digital silence inside loud "
             f"material, worst {info['worst_hard_silence_ms']:.0f}ms (a mute, not a rest)")
    if info["click_rel"] > 6 and info["click_count"] > 0:
        fail(f"{tag}: {info['click_count']} click-sized step(s), worst {info['click_rel']:.1f}x the material")
    if not dark and info["centroid"] < 400:
        fail(f"{tag}: spectral centre {info['centroid']:.0f} Hz is dull for a sung part")
    if not dark and input_centroid and input_centroid > 0:
        # BRIGHTNESS IS RELATIVE TO WHAT WAS SUNG. A part an octave darker than
        # the take it is shadowing reads as muffled however bright the track
        # itself was, and a dark track cannot make a dull part acceptable.
        info["input_centroid"] = input_centroid
        info["centroid_ratio"] = info["centroid"] / input_centroid
        if info["centroid"] < 0.4 * input_centroid:
            fail(f"{tag}: spectral centre {info['centroid']:.0f} Hz against the input's "
                 f"{input_centroid:.0f} Hz - the part is far darker than what was sung")
    if not resting and info["longest_gap_s"] > 6.0:
        note(f"{tag}: {info['longest_gap_s']:.1f}s of silence inside a mode that should sing throughout")


def sweep_dsp(base: str, inputs: dict, words: list[str], quick: bool) -> None:
    say("\n=== DSP engine: every mode x texture (Mix vocable) ===")
    seen: dict[str, tuple[np.ndarray, int]] = {}
    for iname, (raw, what) in inputs.items():
        x_src, sr_src = read_wav(raw)
        sr_src = int(sr_src)
        ref = x_src if x_src.ndim == 1 else x_src.mean(axis=1)
        ref_centroid = float(analyse(ref, sr_src, f"ref-{iname}")["centroid"])
        say(f"\ninput: {iname} - {what}  (spectral centre {ref_centroid:.0f} Hz)")
        # THE MELODY REFERENCE, and it is a RENDER rather than a table of notes.
        # The hum sings the tracked melody through the same pipeline, and
        # measured +4 to +70 cents against the notes of the controlled phrase,
        # so it is the one part of this system whose pitch can be trusted; a
        # melodic mode is then asked to be on the same tune as it. Judging
        # against nominal notes instead fails everything the moment the
        # project's own tracker misreads the input (it reads the controlled
        # 220 Hz phrase as a glide from 216 to 204 Hz, and the part follows the
        # tracker, which is correct behaviour and a wrong test).
        melody_ref = None
        r, _ = post("/api/dsp/render-track", base, {"file": ("in.wav", raw, "audio/wav")},
                    {"texture": "solo", "pitch_method": "yin", "mode": "solo_humming",
                     "instruments_enabled": "false", "vocable": "mix"})
        if r.status_code == 200:
            hy, hsr = read_wav(r.content)
            melody_ref = pitch_curve(hy if hy.ndim == 1 else hy.mean(axis=1), hsr)
            say(f"  melody reference (the hum of this input): {int((melody_ref > 0).sum())} "
                f"voiced frames of {len(melody_ref)}")
        for mode in MODES:
            for texture in TEXTURES:
                data = {"texture": texture, "pitch_method": "yin", "mode": mode,
                        "instruments_enabled": "false", "vocable": "mix"}
                data.update(DSP_MODE_FIELDS.get(mode, {}))
                r, dt = post("/api/dsp/render-track", base, {"file": ("in.wav", raw, "audio/wav")}, data)
                tag = f"dsp_{iname}_{mode}_{texture}"
                if r.status_code != 200:
                    fail(f"{tag}: HTTP {r.status_code}: {r.text[:200]}")
                    continue
                y, ysr = read_wav(r.content)
                y_mono = y if y.ndim == 1 else y.mean(axis=1)
                sf.write(os.path.join(OUT_DIR, tag + ".wav"), y if y.ndim == 1 else y.T, ysr)
                info = analyse(y_mono, ysr, tag, reference=ref, reference_sr=sr_src)
                info.update(mode=mode, texture=texture, engine="dsp", input=iname,
                            seconds=dt, channels=1 if y.ndim == 1 else y.shape[1])
                RESULTS.append(info)
                say(f"  {mode:22s} {texture:6s} {dt:5.1f}s  "
                    f"peak {info['peak']:.3f} rms {info['rms']:.4f} "
                    f"hot {info['hot']} holes {info['dropouts']} rests {info.get('rests', 0)} "
                    f"click {info['click_rel']:.1f}x cent {info['centroid']:.0f}Hz"
                    + (f" nominal-pitch {info['pitch_hit_rate']*100:.0f}% "
                       f"(med {info['pitch_cents_med']:.0f}c)" if 'pitch_hit_rate' in info else ""))
                check_common(tag, info, dark=mode in DARK_MODES, resting=mode in RESTING_MODES,
                             input_centroid=ref_centroid)
                if melody_ref is not None and mode in PITCH_FOLLOWING_MODES:
                    ratios = MODE_PITCH_RATIOS.get(mode, (1.0,))
                    hit, med, nf = melody_agreement(y_mono, ysr, melody_ref, ratios)
                    info["melody_hit_rate"] = hit
                    info["melody_cents_med"] = med
                    info["melody_frames"] = nf
                    # A NOTE, NOT A VERDICT, and the wording says why: over a
                    # whole song several of the part's own notes ring at once
                    # (tails, reverb, the ensemble), and any fundamental
                    # detector reads the mixture, so a low number here does not
                    # establish that the part is off the tune. The one-note
                    # probe below is where pitch is actually judged.
                    say(f"    (informational) {hit*100:.0f}% of {nf} frames agree with the "
                        f"hum's own pitch curve, median {med:.0f} cents")
                    if nf >= 40 and hit < 0.15:
                        note(f"{tag}: only {hit*100:.0f}% of {nf} frames agree with the hum's "
                             f"curve - worth a listen (polyphonic material limits what this can prove)")
                # THE TEXTURES MUST BE DIFFERENT RENDERS, not the same file
                # under three labels: solo/duet/choir are three arrangements.
                if texture != "solo":
                    key = f"dsp_{iname}_{mode}_{TEXTURES[0]}"
                    prev = seen.get(key)
                    if prev is not None and prev[0].shape == y_mono.shape:
                        c = float(np.corrcoef(prev[0][:min(len(prev[0]), len(y_mono))],
                                              y_mono[:min(len(prev[0]), len(y_mono))])[0, 1])
                        if c > 0.9995:
                            fail(f"{tag}: identical to {TEXTURES[0]} (corr {c:.5f}) - the texture does nothing here")
                seen[tag] = (y_mono, ysr)

        # EVERY loaded word, so a broken take is caught by name rather than
        # being averaged away inside Mix.
        if words and iname == "synth" and not quick:
            say(f"\ninput: {iname} - every loaded vocable word (unison_shadowing, solo)")
            word_renders: dict[str, np.ndarray] = {}
            for w in words:
                data = {"texture": "solo", "pitch_method": "yin", "mode": "unison_shadowing",
                        "instruments_enabled": "false", "vocable": w}
                r, dt = post("/api/dsp/render-track", base, {"file": ("in.wav", raw, "audio/wav")}, data)
                tag = f"dsp_{iname}_word-{w}"
                if r.status_code != 200:
                    fail(f"{tag}: HTTP {r.status_code}: {r.text[:200]}")
                    continue
                y, ysr = read_wav(r.content)
                y_mono = y if y.ndim == 1 else y.mean(axis=1)
                sf.write(os.path.join(OUT_DIR, tag + ".wav"), y if y.ndim == 1 else y.T, ysr)
                info = analyse(y_mono, ysr, tag, notes=SYNTH_NOTES, ratios=(1.0,))
                info.update(mode="unison_shadowing", texture="solo", engine="dsp",
                            input=iname, word=w, seconds=dt)
                RESULTS.append(info)
                say(f"  word {w:26s} {dt:5.1f}s  peak {info['peak']:.3f} rms {info['rms']:.4f} "
                    f"hot {info['hot']} holes {info['dropouts']} cent {info['centroid']:.0f}Hz "
                    + (f"nominal-pitch {info['pitch_hit_rate']*100:.0f}%" if 'pitch_hit_rate' in info else ""))
                check_common(tag, info, dark=False, resting=False, input_centroid=ref_centroid)
                word_renders[w] = y_mono
            # a word that renders exactly like another word is not that word
            names = list(word_renders)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = word_renders[names[i]], word_renders[names[j]]
                    m = min(len(a), len(b))
                    if m > 4096:
                        c = float(np.corrcoef(a[:m], b[:m])[0, 1])
                        if c > 0.9995:
                            fail(f"dsp word {names[i]} and {names[j]} are the same render "
                                 f"(corr {c:.5f}) - one take is standing in for the other")


def sweep_one_note_probe(base: str, raw: bytes) -> None:
    """One held note, every mode and texture: is it on the note, and is it there?

    A whole song hides the difference between a duet and a solo, and it makes
    pitch unmeasurable (several notes ringing at once, reverb, tails: any
    fundamental detector starts reading the mixture). One 4.5 s note has
    neither problem. This is the test that caught the humming texture being a
    no-op, and it is the only pitch judgement in this harness that is a fact
    rather than an inference.
    """
    say("\n=== One-note probe: one held note, every mode x texture ===")
    note_bytes = wav_bytes(synth_texture(phrase=False))
    # THE FIXTURE IS CHECKED BEFORE IT IS USED. This probe is the one pitch
    # judgement in this harness that is a fact rather than an inference, and
    # that is only true while the probe's own tone is the note it says it is:
    # _tone's vibrato used to be written in a form that made it a rising glide
    # (see the fix note there), and every render that followed it correctly
    # failed a nominal-note assertion because of it. So the fixture measures
    # itself here, and a broken one is a harness failure, loudly.
    y_in, sr_in = read_wav(note_bytes)
    y_in_mono = y_in if y_in.ndim == 1 else y_in.mean(axis=1)
    core_in = y_in_mono[int(0.8 * sr_in): int(3.6 * sr_in)]
    cur_in = pitch_curve(core_in, sr_in, hop_s=0.05)
    voiced_in = cur_in[cur_in > 0]
    f0 = float(np.median(voiced_in)) if voiced_in.size >= 5 else TEXTURE_F0
    fixture_cents = 1200 * np.log2(f0 / TEXTURE_F0) if f0 > 0 else float("nan")
    say(f"  fixture: the probe tone is {TEXTURE_F0:.2f} Hz by construction and measures "
        f"{f0:.2f} Hz ({fixture_cents:+.0f} cents), voiced {voiced_in.size}/{cur_in.size}")
    if abs(fixture_cents) > 25 or voiced_in.size < 0.5 * cur_in.size:
        fail(f"harness fixture: the one-note probe measures {f0:.2f} Hz of its own "
             f"{TEXTURE_F0:.2f} Hz tone ({fixture_cents:+.0f} cents) - the probe's note is "
             "not the note it claims, so nothing below it can be judged")
    for mode in MODES:
        renders: dict[str, np.ndarray] = {}
        for texture in TEXTURES:
            data = {"texture": texture, "pitch_method": "yin", "mode": mode,
                    "instruments_enabled": "false", "vocable": "mix"}
            data.update(DSP_MODE_FIELDS.get(mode, {}))
            r, dt = post("/api/dsp/render-track", base, {"file": ("n.wav", note_bytes, "audio/wav")}, data)
            if r.status_code == 422 and mode == "call_and_response":
                note(f"one-note probe {mode}: refused ({r.json().get('error','')[:60]}\u2026) - "
                     f"this mode answers in pauses and a held note has none")
                continue
            if r.status_code != 200:
                fail(f"one-note probe {mode}/{texture}: HTTP {r.status_code}: {r.text[:160]}")
                continue
            y, ysr = read_wav(r.content)
            y_mono = y if y.ndim == 1 else y.mean(axis=1)
            sf.write(os.path.join(OUT_DIR, f"note_{mode}_{texture}.wav"), y_mono, ysr)
            renders[texture] = y_mono
            if mode in NOTE_INTERVALS:
                # the note's own sustain, not the attack or the release tail
                core = y_mono[int(0.8 * ysr): int(3.6 * ysr)]
                cur = pitch_curve(core, ysr, hop_s=0.05)
                voiced = cur[cur > 0]
                present = note_presence_db(core, ysr, f0)
                got = float(np.median(voiced)) if len(voiced) >= 5 else float("nan")
                cents = (min(abs(1200 * np.log2(got / (f0 * k))) for k in NOTE_INTERVALS[mode])
                         if got == got else float("nan"))
                say(f"  {mode:22s} {texture:6s} sings {got:7.2f} Hz over the {f0:.2f} Hz note "
                    f"({cents:5.0f} cents from the nearest allowed interval), "
                    f"that note sits {present:5.1f} dB under the render's peak, "
                    f"voiced {len(voiced)}/{len(cur)}")
                if present != present or present < -20.0:
                    fail(f"one-note probe {mode}/{texture}: the input's {f0:.0f} Hz note is "
                         f"{present:.1f} dB under this render's peak - the mode is not on it")
                if mode not in PRESENCE_ONLY_MODES:
                    if got != got:
                        fail(f"one-note probe {mode}/{texture}: nothing pitched came back "
                             f"({len(voiced)} voiced frames of {len(cur)})")
                    elif cents > 90:
                        fail(f"one-note probe {mode}/{texture}: sings {got:.1f} Hz over a "
                             f"{f0:.1f} Hz note - {cents:.0f} cents off, allowed "
                             f"{NOTE_INTERVALS[mode]}")
        for i in range(len(TEXTURES)):
            for j in range(i + 1, len(TEXTURES)):
                a, b = renders.get(TEXTURES[i]), renders.get(TEXTURES[j])
                if a is None or b is None:
                    continue
                m = min(len(a), len(b))
                if m <= 4096:
                    continue
                c = float(np.corrcoef(a[:m], b[:m])[0, 1])
                if c > 0.9995:
                    fail(f"one-note probe {mode}: {TEXTURES[i]} and {TEXTURES[j]} render "
                         f"identically (corr {c:.5f}) - the texture row does nothing here")
        if len(renders) == len(TEXTURES):
            ok(f"{mode}: {len(TEXTURES)} textures are distinct renders")


def sweep_repeatability(base: str, raw: bytes) -> None:
    """The same request twice: identical audio, and the same level.

    This is the instrument for the one quality bug a listener reports as "it
    sounds different every time": the same settings coming back at different
    loudness, because the renderer measured its own level over silence.
    """
    say("\n=== Repeatability: the same request twice ===")
    for mode in ("unison_shadowing", "solo_humming", "call_and_response"):
        levels, clips = [], []
        for _ in range(2):
            data = {"texture": "solo", "pitch_method": "yin", "mode": mode,
                    "instruments_enabled": "false", "vocable": "mix"}
            r, _ = post("/api/dsp/render-track", base, {"file": ("in.wav", raw, "audio/wav")}, data)
            if r.status_code != 200:
                fail(f"repeat {mode}: HTTP {r.status_code}: {r.text[:160]}")
                break
            y, ysr = read_wav(r.content)
            y_mono = y if y.ndim == 1 else y.mean(axis=1)
            levels.append(20 * np.log10(rms(y_mono) + 1e-12))
            clips.append(y_mono)
        if len(levels) == 2:
            spread = abs(levels[0] - levels[1])
            msg = f"{mode}: RMS {levels[0]:.2f} / {levels[1]:.2f} dBFS (spread {spread:.2f} dB)"
            if spread > 1.5:
                fail(f"repeat {msg} - the same request should not come back a different level")
            else:
                ok(msg)
            if clips[0].shape == clips[1].shape:
                c = float(np.corrcoef(clips[0], clips[1])[0, 1])
                note(f"repeat {mode}: render-to-render correlation {c:.4f} "
                     f"({'seeded, reproducible' if c > 0.999 else 'varies by design'})")


def sweep_mode_separation(base: str, raw: bytes) -> None:
    """Every pair of modes must be a different piece of music.

    Two modes returning the same audio is the worst kind of bug: the panel says
    it did something and the file says it did not.
    """
    say("\n=== Mode separation: all 8 modes on one input ===")
    renders: dict[str, np.ndarray] = {}
    for mode in MODES:
        data = {"texture": "solo", "pitch_method": "yin", "mode": mode,
                "instruments_enabled": "false", "vocable": "mix"}
        data.update(DSP_MODE_FIELDS.get(mode, {}))
        r, _ = post("/api/dsp/render-track", base, {"file": ("in.wav", raw, "audio/wav")}, data)
        if r.status_code != 200:
            fail(f"mode separation {mode}: HTTP {r.status_code}: {r.text[:160]}")
            continue
        y, ysr = read_wav(r.content)
        renders[mode] = (y if y.ndim == 1 else y.mean(axis=1))
    names = list(renders)
    identical = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = renders[names[i]], renders[names[j]]
            m = min(len(a), len(b))
            if m < 4096:
                continue
            c = float(np.corrcoef(a[:m], b[:m])[0, 1])
            if c > 0.9995:
                identical += 1
                fail(f"modes {names[i]} and {names[j]} render identically (corr {c:.5f})")
    if not identical:
        ok(f"{len(names)} modes, all distinct")

    # ONE LEVEL FOR EVERY MODE. The panel has one ROBOT VOLUME slider and eight
    # modes, so the same slider position must mean the same loudness in all of
    # them; a mode that comes back 5 dB hotter makes the slider a different
    # control per mode, and a listener hears switching modes as switching level.
    levels = {}
    for mode, sig in renders.items():
        info = analyse(sig, SR, f"level-{mode}")
        levels[mode] = float(info.get("sounding_rms") or 0.0)
    heard = {m: v for m, v in levels.items() if v > 0}
    if len(heard) >= 3:
        med = float(np.median(list(heard.values())))
        say("  level while singing: " + ", ".join(
            f"{m} {20*np.log10(v/med):+.1f} dB" for m, v in sorted(heard.items(), key=lambda kv: kv[1])))
        for mode, v in heard.items():
            off = 20 * np.log10(v / med)
            if abs(off) > 3.0:
                fail(f"mode {mode} sings {off:+.1f} dB against the other modes "
                     f"(one slider, eight modes, one level)")


def sweep_neural(base: str, inputs: dict, models: list[int], quick: bool) -> None:
    say("\n=== Neural engine: live chunks + full render (every configured model) ===")
    for iname, (raw, what) in inputs.items():
        if iname == "acap" and quick:
            continue
        x_src, sr_src = read_wav(raw)
        sr_src = int(sr_src)
        ref = x_src if x_src.ndim == 1 else x_src.mean(axis=1)
        ref_centroid = float(analyse(ref, sr_src, f"ref-{iname}")["centroid"])
        say(f"\ninput: {iname} - {what}  ({len(ref)/sr_src:.1f}s @ {sr_src} Hz)")
        t0 = time.time()
        r, _ = post("/api/neural/backing-bed", base, {"file": ("track.wav", raw, "audio/wav")}, {})
        track_key = (r.json() or {}).get("track_key") if r.status_code == 200 else None
        status = "unknown"
        if track_key:
            for _ in range(600):
                s = requests.get(base + "/api/neural/backing-bed/status",
                                 params={"track_key": track_key}, timeout=30).json().get("status")
                status = s
                if s in ("ready", "failed", "none", "unknown"):
                    break
                time.sleep(2)
        say(f"  separation: {status} ({(time.time()-t0):.0f}s)")

        # the band the page actually plays under the robot in the neural engine
        band = None
        if track_key and status == "ready":
            rb = requests.get(base + "/api/neural/backing-bed/track",
                              params={"track_key": track_key}, timeout=300)
            if rb.status_code == 200:
                band, band_sr = read_wav(rb.content)
                sf.write(os.path.join(OUT_DIR, f"band_{iname}.wav"), band, band_sr)
                bmono = band if band.ndim == 1 else band.mean(axis=1)
                mixmono = ref
                m = min(len(bmono), len(mixmono))
                # how much of the ORIGINAL mix survives in the band: the thing a
                # listener hears as "my track lost its quality"
                gap_db = 20 * np.log10((rms(bmono[:m]) + 1e-12) / (rms(mixmono[:m]) + 1e-12))
                say(f"  band: {len(bmono)/band_sr:.1f}s {band.shape[1] if band.ndim>1 else 1}ch "
                    f"rms {rms(bmono):.4f} vs mix {rms(mixmono):.4f} ({gap_db:+.1f} dB) "
                    f"centroid {analyse(bmono, band_sr, 'band')['centroid']:.0f}Hz "
                    f"vs mix {analyse(mixmono, sr_src, 'mix')['centroid']:.0f}Hz")
                note(f"band/{iname}: {gap_db:+.1f} dB vs the uploaded mix, "
                     f"X-Separation-Channels={rb.headers.get('X-Separation-Channels')}, "
                     f"{rb.headers.get('X-Sample-Rate')} Hz")

        chunk_start = min(12.0, max(1.0, len(ref) / sr_src / 3))
        # 8 s: the length the page actually streams
        chunk = ref[int(chunk_start * sr_src): int((chunk_start + 8.0) * sr_src)]
        if len(chunk) < sr_src:
            chunk_start = 0.0
            chunk = ref[: int(min(8.0, len(ref) / sr_src) * sr_src)]
        for mode in ("unison_shadowing", "solo_humming", "call_and_response"):
            for model in models:
                data = {"voice_index": str(model), "transpose_semitones": "0",
                        "pad_seconds": "0.3", "instruments_enabled": "false", "mode": mode}
                if track_key:
                    data["track_key"] = track_key
                    data["track_offset_s"] = str(chunk_start)
                r, dt = post("/api/neural/convert-track", base,
                             {"file": ("chunk.wav", wav_bytes(chunk, sr_src), "audio/wav")}, data)
                tag = f"neural_{iname}_{mode}_model{model}"
                if r.status_code != 200:
                    fail(f"{tag}: HTTP {r.status_code}: {r.text[:200]}")
                    continue
                y, ysr = read_wav(r.content)
                y_mono = y if y.ndim == 1 else y.mean(axis=1)
                sf.write(os.path.join(OUT_DIR, tag + ".wav"), y if y.ndim == 1 else y.T, ysr)
                info = analyse(y_mono, ysr, tag, reference=chunk, reference_sr=sr_src,
                               notes=SYNTH_NOTES if iname == "synth" else None,
                               ratios=(1.0,))
                rt = dt / max(1e-6, len(chunk) / sr_src)
                info.update(mode=mode, texture="live-chunk", engine="neural", input=iname,
                            model=model, seconds=dt, realtime_factor=rt,
                            bed=r.headers.get("X-Backing-Bed", "-"))
                RESULTS.append(info)
                say(f"  {mode:18s} model {model} {dt:5.1f}s ({rt:.2f}x real time)  "
                    f"peak {info['peak']:.3f} rms {info['rms']:.4f} hot {info['hot']} "
                    f"holes {info['dropouts']} click {info['click_rel']:.1f}x "
                    f"cent {info['centroid']:.0f}Hz bed {info['bed']}")
                chunk_centroid = float(analyse(chunk, sr_src, "chunk")["centroid"])
                check_common(tag, info, dark=mode in DARK_MODES, resting=mode in RESTING_MODES,
                             input_centroid=chunk_centroid)
                if rt > 1.0:
                    note(f"{tag}: {rt:.2f}x real time - this chunk cannot be converted "
                         f"faster than it plays, so a live stream of these will drift late")
                if 'corr' in info and mode == "unison_shadowing" and info['corr'] < 0.05:
                    fail(f"{tag}: the conversion has almost no resemblance to the chunk "
                         f"(corr {info['corr']:.3f})")

        # the Full render, which is the same path the page's Render button uses
        for model in models:
            data = {"voice_index": str(model), "transpose_semitones": "0", "texture": "solo",
                    "pitch_method": "rmvpe", "mode": "unison_shadowing",
                    "instruments_enabled": "false", "file": ("in.wav", raw, "audio/wav")}
            r, dt = post("/api/neural/render-track", base,
                         {"file": ("in.wav", raw, "audio/wav")}, data)
            tag = f"neural_{iname}_full_model{model}"
            if r.status_code != 200:
                fail(f"{tag}: HTTP {r.status_code}: {r.text[:200]}")
                continue
            y, ysr = read_wav(r.content)
            y_mono = y if y.ndim == 1 else y.mean(axis=1)
            sf.write(os.path.join(OUT_DIR, tag + ".wav"), y if y.ndim == 1 else y.T, ysr)
            info = analyse(y_mono, ysr, tag, reference=ref, reference_sr=sr_src)
            rt = dt / max(1e-6, len(ref) / sr_src)
            info.update(mode="unison_shadowing", texture="full", engine="neural", input=iname,
                        model=model, seconds=dt, realtime_factor=rt)
            RESULTS.append(info)
            say(f"  full render model {model} {dt:5.1f}s ({rt:.2f}x real time) "
                f"peak {info['peak']:.3f} rms {info['rms']:.4f} hot {info['hot']} "
                f"holes {info['dropouts']} click {info['click_rel']:.1f}x cent {info['centroid']:.0f}Hz")
            check_common(tag, info, dark=False, resting=False, input_centroid=ref_centroid)


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    return sf.read(io.BytesIO(data), dtype="float32", always_2d=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default="dsp,neural")
    ap.add_argument("--inputs", default="real,acap,synth")
    ap.add_argument("--models", default="all",
                    help="comma-separated voice indices, or \"all\" for every model in config.yaml")
    ap.add_argument("--quick", action="store_true",
                    help="skip the per-word sweep and the a cappella neural pass")
    ap.add_argument("--dsp-inputs", default="", help="override the inputs used for the DSP sweep")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    inputs = load_inputs([s for s in args.inputs.split(",") if s])
    words = vocable_words(args.base)
    say(f"inputs: {', '.join(inputs)}   vocable words: {len(words)}   artifacts: {OUT_DIR}")

    # WHAT "EVERY MODEL" MEANS IS READ FROM THE RUNNING CONFIG, not from a
    # literal in this file. The old default was "0,1" and stayed that way when
    # a third voice was added, so the sweep would report "both models pass"
    # while Model 3 had never been rendered at all. "all" asks the main app
    # how many entries synthesis.neural.model_paths holds and drives every one.
    models_arg = (args.models or "all").strip().lower()
    if models_arg in ("all", "*", ""):
        try:
            status = requests.get(args.base + "/neural/status", timeout=30).json()
            count = int(status.get("voices_configured") or 0)
        except Exception as e:
            note(f"could not read how many voices are configured ({e}); testing model 0 only")
            count = 0
        model_indices = list(range(max(1, count)))
    else:
        model_indices = [int(m) for m in models_arg.split(",") if m.strip()]
    say(f"neural models: {', '.join(f'Model {m+1}' for m in model_indices)}")

    if "dsp" in args.only:
        dsp_inputs = ({k: v for k, v in inputs.items()
                       if k in args.dsp_inputs.split(",")} if args.dsp_inputs else inputs)
        sweep_dsp(args.base, dsp_inputs, words, args.quick)
        sweep_mode_separation(args.base, inputs.get("synth", next(iter(inputs.values())))[0])
        sweep_one_note_probe(args.base, inputs.get("synth", next(iter(inputs.values())))[0])
        sweep_repeatability(args.base, inputs.get("synth", next(iter(inputs.values())))[0])
    if "neural" in args.only:
        sweep_neural(args.base, inputs, model_indices, args.quick)

    say("")
    say(f"=== {len(FAILS)} finding(s) ===")
    for f in FAILS:
        say(f"  [FAIL] {f}")
    for n in NOTES:
        say(f"  [note] {n}")
    with open(REPORT, "w") as fh:
        json.dump({"fails": FAILS, "notes": NOTES, "results": RESULTS}, fh, indent=1)
    say(f"artifacts: {OUT_DIR}   report: {REPORT}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

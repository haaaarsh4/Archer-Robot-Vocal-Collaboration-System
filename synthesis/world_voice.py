"""
Singing one recorded word along a melody, with the WORLD vocoder.
=================================================================

This is the engine behind the offline "one word, sung along the track" render
(_render_track_single_vocable_offline in server.py). It replaces the chain
that used to be there, a bucket phase vocoder for a constant shift, plus a
variable-rate re-read (apply_cents_curve) for the melody's movement, and it
exists because that chain could not stop sounding like a machine, no matter
how carefully its parameters were tuned.

WHY THE OLD CHAIN COULD NOT SOUND HUMAN. Two separate reasons, both
structural rather than a matter of tuning:

  - A variable-rate re-read changes the SPEED of the audio as it changes its
    pitch. Following a melody therefore also time-warped the word: as the
    tune moved, the syllables inside it were read faster or slower, so the
    rhythm the singer actually recorded, the thing that makes "lalala"
    legible as a word, was scrambled. Measured on this project's own take,
    the melody's own movement inside one phrase is several semitones, i.e.
    up to a third faster or slower at the edges, and the renormalization
    only guarantees the total length, never the syllable timing inside it.
  - A phase vocoder rebuilds the sound from its own frame-by-frame phase
    guesswork. On a voice that is exactly where the "metallic"/"robotic"
    colour comes from, and a formant correction on top of it does not remove
    it: it moves the spectral ENVELOPE back to where a throat keeps it, but
    the phases underneath are the vocoder's, not the singer's.

WHAT THIS DOES INSTEAD. WORLD (pyworld) is a source-filter model built for
exactly this job, modifying a human voice's pitch. One analysis of the take
gives three things:

  - f0: the pitch the word was actually sung at, frame by frame.
  - sp: the spectral ENVELOPE, i.e. the vocal tract itself. Keeping this
    untouched is what keeps the result sounding like the same person: the
    formants, the nasality, the vowel colours are the recording's own, not
    reconstructed, and they do not travel with the pitch the way a plain
    resample drags them (no chipmunk).
  - ap: the aperiodicity, i.e. how much of each frame is breath rather than
    tone. This is where the consonants and the voice's own airiness live, so
    the "l" and the "n" of the word stay real consonants instead of being
    rebuilt as tones.

The word is then re-synthesized from those three curves with a NEW f0: the
melody's own pitch, frame by frame, plus the take's own fast pitch life
(vibrato and micro-movement, see below) so it still breathes like the
performance. The length never changes: a frame of the take always occupies
exactly one frame of output, so the syllables land where the singer put them
and "lalala" stays audible as a word at any pitch. There is no resampling and
no phase vocoder anywhere in this path.

WHAT IS DELIBERATELY KEPT FROM THE TAKE'S OWN PITCH CURVE. The SLOW part of
the take's pitch movement (a scoop, a drift: anything slower than SLOW_S) is
removed, because it is exactly what a fixed note cannot follow, the same
split flatten_syllable_pitch uses, and for the same reason. The FAST part is
kept, scaled and capped (DEV_KEEP / DEV_MAX_CENTS): a real sung vibrato runs
4-7 Hz, which is faster than the split, so it survives as the take's own
life, while the cap keeps the placed note from drifting audibly off pitch.

Frames the take sang as unvoiced (its consonants, its breath) are synthesized
unvoiced here too, which is what keeps the articulation intact rather than
turning the whole word into one continuous vowel.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import pyworld as _pw
    HAS_WORLD = True
except Exception:  # pragma: no cover - optional dependency
    _pw = None
    HAS_WORLD = False

# One analysis/synthesis frame per 5 ms. WORLD's own default, and fine enough
# that a note onset or a consonant edge is placed where it was sung.
FRAME_PERIOD_MS = 5.0

# The slow/fast split of the take's own pitch movement, in seconds. Movement
# slower than this is removed (a constant note cannot follow it); movement
# faster than it, vibrato, micro-movement, is the performance's own life
# and is kept. Same value as melody_voice.FLATTEN_SLOW_S, deliberately: the
# two paths must agree on what "the take's own movement" means.
SLOW_S = 0.40

# How much of that kept fast movement survives, and its cap, in cents. A few
# tens of cents is a singer's own vibrato; the raw fast component of this
# project's takes runs to ~90 cents RMS, which simply measures as out of tune.
DEV_KEEP = 0.6
DEV_MAX_CENTS = 30.0

# How far either side of the take's own register the pitch detector is
# allowed to look, as a ratio. The take is one person singing one word, so a
# tight window around its own register stops the detector from reporting an
# octave (or two) away, which on a female voice with a weak fundamental is
# the single most common failure, and which reads to a listener as a second,
# unrelated singer.
F0_FLOOR_RATIO = 0.45
F0_CEIL_RATIO = 2.2

# A short fade at each point where the word starts over, so a repeated word
# is re-articulated cleanly instead of clicking across a waveform step. This
# is a fade, not a crossfade: at a word boundary a real singer's mouth closes
# and opens, and blending the end of one word into the start of the next is
# what smears a consonant.
SEAM_FADE_S = 0.008

# The melody's own movement is applied through this short a smoothing, so a
# note change arrives as the small slide a singer makes rather than as an
# instant step in the middle of a vowel.
PORTAMENTO_S = 0.05

# Where everything under this frequency is removed from the finish of a sung
# phrase. Measured directly on this project's own take (its spectral balance
# against the same take re-synthesized by WORLD with its own pitch curve: the
# two agree within half a dB everywhere from 200 Hz to 20 kHz, which is the
# proof that WORLD keeps the voice itself), WORLD's synthesis carries 6 to 17
# dB of extra energy below 150 Hz that the recording does not have, a low
# rumble under an otherwise identical voice. A 4th-order zero-phase high-pass
# there removes it completely, and cannot touch the voice: the lowest
# fundamental the detector is allowed to report (see F0_FLOOR_RATIO) is about
# 150 Hz on this take, and the recording's own energy in this band sits 50 dB
# below its vowels, so nothing that was ever the singer lives down there.
LF_CUT_HZ = 90.0


def _moving_average(vals: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with edge padding (the slow/fast split)."""
    win = max(1, int(win))
    if win <= 1 or vals.size <= 1:
        return vals.copy()
    pad = win // 2
    padded = np.pad(vals, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(padded, kernel, mode="valid")[:vals.size]


def analyze_performance(take: np.ndarray, sr: int, register_hz: float = 0.0):
    """Analyse the take once. Returns a dict of WORLD curves, or None.

    None means WORLD is unavailable or the take is too short to analyse, and
    the caller falls back to its older shifting path, this function never
    raises for a take it cannot handle.
    """
    if not HAS_WORLD:
        return None
    x = np.ascontiguousarray(np.asarray(take, dtype=np.float64))
    if x.size < int(0.25 * sr) or not np.any(np.isfinite(x)):
        return None
    nyq = 0.5 * float(sr)
    if register_hz and register_hz > 0:
        floor = max(50.0, float(register_hz) * F0_FLOOR_RATIO)
        ceil = min(nyq * 0.90, float(register_hz) * F0_CEIL_RATIO)
        if ceil <= floor * 1.25:
            ceil = min(nyq * 0.90, floor * 1.25)
    else:
        floor, ceil = 65.0, min(nyq * 0.90, 1200.0)
    try:
        f0, t = _pw.harvest(x, sr, f0_floor=float(floor), f0_ceil=float(ceil),
                            frame_period=FRAME_PERIOD_MS)
        f0 = np.ascontiguousarray(_pw.stonemask(x, f0, t, sr), dtype=np.float64)
        sp = np.ascontiguousarray(_pw.cheaptrick(x, f0, t, sr), dtype=np.float64)
        ap = np.ascontiguousarray(_pw.d4c(x, f0, t, sr), dtype=np.float64)
    except Exception:
        return None
    n = int(len(f0))
    if n < 4 or sp.size == 0 or ap.size == 0:
        return None

    voiced = np.isfinite(f0) & (f0 > 0)
    hz = np.where(voiced, f0, 0.0)
    # The take's own FAST pitch life, in cents: the movement left after the
    # slow component is averaged away. Detector dropouts are bridged first by
    # holding the neighbouring reading, so a consonant cannot pull the average
    # off the note being held either side of it.
    dev = np.zeros(n, dtype=np.float64)
    if int(voiced.sum()) >= 4:
        idx = np.arange(n)
        filled = np.interp(idx, idx[voiced], hz[voiced])
        slow = _moving_average(filled, int(round(SLOW_S * 1000.0 / FRAME_PERIOD_MS)))
        fast = 1200.0 * np.log2(np.maximum(filled, 1e-6) / np.maximum(slow, 1e-6))
        dev = np.clip(fast * DEV_KEEP, -DEV_MAX_CENTS, DEV_MAX_CENTS)
        dev[~voiced] = 0.0

    frame_period_s = FRAME_PERIOD_MS / 1000.0
    return {
        "f0": hz,
        "sp": sp,
        "ap": ap,
        "dev_cents": dev,
        "voiced": voiced,
        "n_frames": n,
        "frame_period_s": frame_period_s,
        "register_hz": float(register_hz or 0.0),
        "f0_floor": float(floor),
        "f0_ceil": float(ceil),
    }


def _high_pass(x: np.ndarray, sr: int, cut_hz: float = LF_CUT_HZ) -> np.ndarray:
    """Zero-phase high-pass, so the filter itself adds no phase smear.

    scipy is what this project already uses for its filters; if it is not
    importable the audio is passed through untouched rather than failed.
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:  # pragma: no cover - scipy is a project dependency
        return x
    nyq = 0.5 * float(sr)
    if not (0.0 < cut_hz < nyq * 0.9) or x.size < 64:
        return x
    try:
        sos = butter(4, cut_hz / nyq, btype="highpass", output="sos")
        return sosfiltfilt(sos, x)
    except Exception:  # pragma: no cover - a filter that cannot run must not kill a render
        return x


def sing_along(perf: dict, sr: int, n_samples: int, base_hz: float,
               curve_cents: np.ndarray | None = None) -> np.ndarray:
    """The take, sung for `n_samples` at `base_hz` plus its own movement.

    The take is read continuously from its own start and starts over whenever
    it runs out (with a short re-articulation fade at that point), so the
    whole word is always heard at the pace it was recorded at. The pitch of
    each frame is `base_hz` moved by the melody's own curve and by the take's
    own kept micro-movement; formants and breathiness are the take's own and
    never move. Frames the take sang unvoiced stay unvoiced.
    """
    sr_f = float(sr)
    frame_period_s = float(perf["frame_period_s"])
    frame_len = max(1, int(round(sr_f * frame_period_s)))
    n_frames = max(1, int(math.ceil(n_samples / frame_len)))
    take_n = int(perf["n_frames"])
    sp_take = perf["sp"]
    ap_take = perf["ap"]

    if take_n <= 0 or sp_take.size == 0:
        return np.zeros(n_samples, dtype=np.float32)

    idx = np.arange(n_frames, dtype=np.int64) % take_n
    sp = np.ascontiguousarray(sp_take[idx], dtype=np.float64)
    ap = np.ascontiguousarray(ap_take[idx], dtype=np.float64)
    voiced = perf["voiced"][idx]
    dev = perf["dev_cents"][idx]

    target = np.full(n_frames, float(base_hz) if base_hz > 0 else 0.0, dtype=np.float64)
    if curve_cents is not None:
        cents = np.asarray(curve_cents, dtype=np.float64)
        if cents.size != n_frames:
            cents = np.interp(np.linspace(0.0, 1.0, n_frames),
                              np.linspace(0.0, 1.0, max(cents.size, 1)), cents)
        target = target * np.exp2(_moving_average(cents, int(round(PORTAMENTO_S / frame_period_s))) / 1200.0)
    target = target * np.exp2(dev / 1200.0)
    f0 = np.where(voiced & (target > 0), target, 0.0)
    f0 = np.ascontiguousarray(f0, dtype=np.float64)

    if not np.any(f0 > 0):
        # nothing to sing from (a take that analysed as entirely unvoiced):
        # rather than emit a wall of noise, leave the phrase empty
        return np.zeros(n_samples, dtype=np.float32)

    try:
        y = _pw.synthesize(f0, sp, ap, int(sr), frame_period=FRAME_PERIOD_MS)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)[:n_samples]
    if y.size < n_samples:
        y = np.pad(y, (0, n_samples - y.size))

    # The sub-vocal rumble WORLD leaves under every phrase (see LF_CUT_HZ).
    # This is the only processing the synthesis itself gets.
    y = _high_pass(y, int(round(sr_f)))

    # Where the word starts over: a short fade on both sides, so the boundary
    # is a re-articulation rather than a step in the waveform. Only the fade
    # is applied, the take's own material either side of it is untouched.
    fade = int(round(SEAM_FADE_S * sr_f))
    if fade > 1 and n_frames > take_n:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        for seam in range(take_n, n_frames, take_n):
            a = seam * frame_len
            b = min(n_samples, a + fade)
            if b > a:
                y[a:b] *= ramp[: b - a]
            a0 = max(0, a - fade)
            if a > a0:
                y[a0:a] *= ramp[: a - a0][::-1]
    return y.astype(np.float32)


def describe(perf: dict) -> str:
    """One log line's worth of what the analysis found."""
    if not perf:
        return "no WORLD analysis"
    voiced = int(np.count_nonzero(perf["voiced"]))
    hz = perf["f0"][perf["voiced"]]
    med = float(np.median(hz)) if hz.size else 0.0
    return (f"{perf['n_frames']} frames, {voiced} voiced, take {med:.0f} Hz, "
            f"detector window {perf['f0_floor']:.0f}-{perf['f0_ceil']:.0f} Hz")

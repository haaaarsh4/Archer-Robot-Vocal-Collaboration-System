"""VOCAL FOLD VIBRATION: the movement in a held note.

WHAT IT IS, PHYSICALLY. Air from the lungs pushes past the folds in the throat,
and they open and close rapidly under that steady pressure to make the sound of
a voice. That cycle is not perfectly rigid: the folds' own tension swings a few
times a second as the muscles holding them re-balance, so the pitch of a held
note moves up and down in a small, regular way. A listener hears that as
vibrato, and it is one of the strongest single cues that a note is coming from a
body rather than from an oscillator. A note with no movement in it at all is
what "held by a machine" sounds like.

THE NUMBERS, AND WHERE THEY COME FROM. The literature on trained classical
singing puts the rate at roughly 4.5-6.5 Hz (5.4 Hz used here, which is the
middle of that band and also the value the project's own hum voice and
call-response composer already used), and the extent at 50-120 cents
peak-to-peak, i.e. 0.5-1.2 semitones. Below about 20 cents peak-to-peak the ear
reads it as a straight tone; above about 150 it starts to read as a wobble
rather than as a voice. So 100% on the panel is 130 cents peak-to-peak, a full,
deliberate classical vibrato, and the default 45% is 58.5 cents peak-to-peak:
an ordinary trained singer holding a note. The extent grows a little with the
length of the note (a short note gets almost none, because a real singer does
not have time to open one up), and it ARRIVES AFTER THE ATTACK -- the folds
settle into their oscillation over the first fraction of a second rather than
being in it from the first cycle, which is the difference between "sings" and
"warbles".

WHAT THIS FILE DOES NOT DO. It does not try to *find* the notes. It is applied
to a part that has already been rendered, and it treats that part as a voice:
each stretch of continuous sound is one held note, and the vibrato fades in a
moment after each such stretch begins and fades out with it. That keeps it
mode-agnostic -- every accompaniment mode, in both engines, goes through the
same door -- instead of each mode having to grow its own vibrato separately.

HOW IT IS APPLIED. A small, regular pitch swing is a frequency modulation, so
the part is re-read at a slightly varying rate: the read position advances a
touch faster while the curve is above the note and a touch slower while it is
below, which raises and lowers the instantaneous pitch without ever changing
the total duration (the curve's mean is exactly 1.0, so it oscillates around the
part's own length rather than drifting through it). An intensity swing rides
along with it, because a real vibrato is not a pure pitch effect: the same
change in the folds' closure that moves the pitch also moves how loud the note
is, by a decibel or so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The panel's own scale: 0-100, where 100 is a full classical vibrato. The
# default here is what the server uses when a request doesn't say (an older
# client, a script, the live mic pipeline), and it matches the panel's own
# default so both ends of the wire agree without either having to be told.
DEFAULT_VIBRATO_AMOUNT = 45.0


@dataclass
class VocalVibrationConfig:
    """Every number behind the movement in a held note."""

    # Cycles per second. 5.4 Hz sits in the middle of the 4.5-6.5 Hz band the
    # singing-voice literature reports for trained classical singers, and it is
    # the value this project's hum voice and call-response composer already use.
    rate_hz: float = 5.4
    # How much faster or slower an individual note's own rate can be, relative
    # to rate_hz. Two singers never have the same vibrato rate, and neither do
    # two notes from the same singer.
    rate_jitter_fraction: float = 0.07
    # Peak-to-peak pitch swing at amount=100, in cents (130 = 1.3 semitones).
    extent_cents_at_full: float = 130.0
    # The fold's oscillation takes hold over the first part of a note rather
    # than being present in its very first cycles.
    onset_s: float = 0.26
    ramp_s: float = 0.34
    # ...and lets go as the note does, slightly faster than it arrived.
    release_s: float = 0.10
    # A note has to last at least this long before any vibrato is added to it:
    # there is nothing to open up inside a sixteenth note.
    min_note_s: float = 0.22
    # Full extent is reached on a note held this long; shorter notes get a
    # proportionally shallower vibrato.
    full_depth_note_s: float = 1.1
    # Slow, non-periodic wander of the whole curve, in cents. Keeps the
    # oscillation from reading as a test signal.
    drift_cents: float = 5.0
    drift_rate_hz: float = 0.33
    # The loudness swing that rides with the pitch swing, as a fraction of the
    # amplitude (0.03 = about +/-0.25 dB at full amount, which is the order a
    # real vibrato measures).
    shimmer_fraction: float = 0.03
    # Analysis/render hop for the curve. One audio frame at this project's
    # frame size; the curve is interpolated up to sample rate afterwards.
    frame_hop_s: float = 1024.0 / 44100.0
    # Level below which a frame counts as silence (relative to the part's own
    # loud frames), used to find where the notes are.
    silence_fraction: float = 0.04

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "VocalVibrationConfig":
        """The project's own tuning, from config.yaml's
        synthesis.vocal_vibration block, falling back to the defaults above for
        anything the file does not name. Every field is optional there.
        """
        if cfg is None:
            try:
                from config.config_loader import get_config
                cfg = get_config()
            except Exception:
                cfg = {}
        raw = (cfg.get("synthesis", {}) or {}).get("vocal_vibration", {}) or {}
        out = cls()
        for name in out.__dataclass_fields__:
            if name in raw:
                try:
                    setattr(out, name, float(raw[name]))
                except (TypeError, ValueError):
                    pass
        return out


def _frame_rms(audio: np.ndarray, frame_len: int) -> np.ndarray:
    n = len(audio)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    n_frames = max(1, int(np.ceil(n / frame_len)))
    padded = np.zeros(n_frames * frame_len, dtype=np.float64)
    padded[:n] = audio
    frames = padded.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def _one_pole(target: np.ndarray, tau_s: float, hop_s: float) -> np.ndarray:
    """Feed `target` through a one-pole follower with time constant `tau_s`."""
    out = np.empty_like(target, dtype=np.float64)
    if target.size == 0:
        return out
    if tau_s <= 0:
        return target.astype(np.float64, copy=True)
    coeff = float(np.exp(-max(hop_s, 1e-6) / tau_s))
    level = float(target[0])
    for i in range(target.size):
        level = coeff * level + (1.0 - coeff) * float(target[i])
        out[i] = level
    return out


def _note_runs(voiced: np.ndarray, hop_s: float, min_note_s: float) -> np.ndarray:
    """For each frame, how far into its own continuous sounding stretch it is."""
    elapsed = np.zeros(voiced.shape, dtype=np.float64)
    run_start = None
    for i, on in enumerate(voiced):
        if on:
            if run_start is None:
                run_start = i
            elapsed[i] = (i - run_start + 1) * hop_s
        else:
            run_start = None
            elapsed[i] = 0.0
    # A stretch shorter than min_note_s is a consonant, not a note: it gets no
    # vibrato at all, and neither does the stretch right after it that started
    # too late to be its own note. Rather than a second pass to find runs, each
    # frame's own depth is gated on the run it belongs to being long enough,
    # which is the same decision made without a second scan.
    starts = np.flatnonzero(voiced & (np.concatenate(([False], voiced[:-1])) == False))
    for s in starts:
        e = s
        while e + 1 < len(voiced) and voiced[e + 1]:
            e += 1
        if (e - s + 1) * hop_s < min_note_s:
            elapsed[s:e + 1] = 0.0
    return elapsed


def vibration_curve_cents(n_frames: int, cfg: VocalVibrationConfig,
                          voiced: np.ndarray, amount: float) -> np.ndarray:
    """The pitch swing itself, one value per frame, in cents."""
    if n_frames <= 0 or amount <= 0:
        return np.zeros(max(0, n_frames), dtype=np.float64)

    amt = float(np.clip(amount, 0.0, 100.0)) / 100.0
    hop = cfg.frame_hop_s
    t = (np.arange(n_frames) + 0.5) * hop

    # Where the notes are, and how far into each one each frame sits.
    elapsed = _note_runs(voiced if voiced.size == n_frames else np.ones(n_frames, dtype=bool),
                         hop, cfg.min_note_s)

    # Depth: the panel's amount, scaled by how much of a note there is to open
    # up. A note held for a second gets the full extent, a quarter-second note a
    # fraction of it.
    length_factor = np.clip(elapsed / max(cfg.full_depth_note_s, 1e-6), 0.0, 1.0)
    depth = 0.5 * cfg.extent_cents_at_full * amt * length_factor      # peak, in cents

    # It arrives after the attack (onset + ramp) and lets go at the release.
    onset_env = np.clip((elapsed - cfg.onset_s) / max(cfg.ramp_s, 1e-6), 0.0, 1.0)
    # Smooth the onset rather than gating it, so the first cycle fades in.
    onset_env = _one_pole(onset_env, 0.05, hop)

    # A per-note rate and a slow wander, both seeded from the material itself so
    # that two runs of the same render produce the same movement (a re-render of
    # one setting must not wobble differently from the one before it).
    seed = int(abs(np.sum(np.abs(voiced.astype(np.float64))) * 1e3)) % (2 ** 31)
    rng = np.random.default_rng(seed)
    rate = cfg.rate_hz * (1.0 + cfg.rate_jitter_fraction * float(rng.uniform(-1.0, 1.0)))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    # Slow drift, so the oscillation is never exactly periodic.
    drift_f = cfg.drift_rate_hz * (1.0 + 0.23 * float(rng.uniform(-1.0, 1.0)))
    drift = cfg.drift_cents * amt * np.sin(2.0 * np.pi * drift_f * t + phase * 0.5)

    swing = depth * onset_env * np.sin(2.0 * np.pi * rate * t + phase)
    return swing + drift


def apply_vocal_vibration(audio: np.ndarray, sample_rate: int,
                          amount: float = DEFAULT_VIBRATO_AMOUNT,
                          cfg: VocalVibrationConfig | None = None) -> np.ndarray:
    """Add the movement of real folds to an already-rendered part.

    `amount` is the panel's 0-100 scale (100 = a full classical vibrato of 130
    cents peak-to-peak). Returns the part unchanged at 0, and unchanged if it is
    too short to hold a note.
    """
    cfg = cfg or VocalVibrationConfig()
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0 or float(amount) <= 0.0:
        return data

    mono = data if data.ndim == 1 else data.mean(axis=-1)
    frame_len = max(64, int(round(cfg.frame_hop_s * sample_rate)))
    if len(mono) < frame_len * 2:
        return data

    rms = _frame_rms(mono, frame_len)
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 0.0:
        return data
    voiced = rms >= max(peak * cfg.silence_fraction, 1e-6)

    cents = vibration_curve_cents(rms.size, cfg, voiced, amount)
    if not np.any(cents != 0.0):
        return data

    hop_s = frame_len / float(sample_rate)
    frame_times = (np.arange(cents.size) + 0.5) * hop_s
    n = len(mono)
    sample_times = np.arange(n) / float(sample_rate)
    curve = np.interp(sample_times, frame_times, cents,
                      left=cents[0] if cents.size else 0.0,
                      right=cents[-1] if cents.size else 0.0)

    ratio = 2.0 ** (curve / 1200.0)
    # The read position advances by the ratio every sample. Because the curve's
    # mean is 1.0 this oscillates around n and never drifts through it: the
    # part keeps its own length and its own rhythm, and only its momentary
    # pitch moves.
    pos = np.cumsum(ratio)
    pos -= pos[0]
    pos = np.clip(pos, 0.0, n - 1.0)

    idx = np.arange(n, dtype=np.float64)
    if data.ndim > 1:
        out = np.empty_like(data)
        for ch in range(data.shape[-1]):
            out[..., ch] = np.interp(pos, idx, data[..., ch].astype(np.float64))
    else:
        out = np.interp(pos, idx, data.astype(np.float64))

    # The loudness that rides with the pitch, taken straight off the same curve
    # the pitch comes from so the two are one gesture rather than two effects:
    # the same change in the folds' closure that moves the pitch moves how loud
    # the note is. Normalised by the full-scale peak, so +/-3% at amount=100
    # (about a quarter of a decibel) and proportionally less below it.
    full_scale_peak = 0.5 * cfg.extent_cents_at_full
    shimmer = 1.0 + cfg.shimmer_fraction * (curve / max(full_scale_peak, 1e-9))
    out = out * shimmer

    return out.astype(np.float32)

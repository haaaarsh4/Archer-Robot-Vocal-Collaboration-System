"""Bow, breath and pick: the layers a sampled patch does not have.

A General MIDI "violin" or "flute" is one recorded note stretched over a
keyboard: it has no bow catching the string, no air crossing the embouchure, no
vibrato that a player adds as a note settles, and no wooden body.  Those four
things are most of what tells an ear that a person is playing rather than a
patch being triggered -- which is why a GM fiddle sounds like a synth violin and
an unprocessed wooden flute sounds like a sine.

Each layer here is generated from the arrangement's own note list, so it lands
on the notes that are actually played, at their velocities and durations:

* **fiddle** -- bow attack noise on every stroke, a low bed of bow hair on the
  string while the note is held, a player's vibrato, and the wooden body's
  low-pass and resonance;
* **flute** -- the breath onset (the chiff as the air starts), air in the tone
  while it is held, a slower, smaller vibrato, and a softer body;
* **guitar** -- the pick on the string, which is all a strummed chord needs to
  stop sounding like a sample being triggered.

Nothing is sampled and nothing is downloaded: this is the part of playing that
a sample library encodes in its recordings, rebuilt from the note list.
"""

from __future__ import annotations

import math

import numpy as np


def _band_sos(sample_rate: int, low: float, high: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter

    nyquist = 0.5 * sample_rate
    low = max(20.0, min(low, nyquist * 0.9))
    high = max(low * 1.05, min(high, nyquist * 0.95))
    return butter(order, [low, high], btype="band", fs=sample_rate, output="sos")


def _noise_layer(
    notes,
    *,
    seconds_per_beat: float,
    total: int,
    sample_rate: int,
    rng: np.random.Generator,
    band: tuple[float, float],
    burst_seconds: float,
    burst_decay: float,
    burst_gain: float,
    sustain_gain: float,
    fade_seconds: float = 0.05,
    sustain_attack: float = 0.05,
):
    """Per-note noise: a transient at each onset and air while the note holds."""
    from scipy.signal import sosfilt

    sos = _band_sos(sample_rate, *band)
    layer = np.zeros(total, dtype=np.float32)
    for note in notes:
        start = int(max(0.0, note.start) * seconds_per_beat * sample_rate)
        if start >= total - 64:
            continue
        seconds = max(0.08, note.duration * seconds_per_beat)
        length = min(total - start, int((seconds + fade_seconds) * sample_rate))
        if length <= 64:
            continue
        velocity = max(1, min(127, int(note.velocity))) / 127.0
        t = np.arange(length) / sample_rate
        segment = sosfilt(sos, rng.normal(0.0, 1.0, length))

        # The note sounds while the player holds it, then stops: a fast rise, a
        # release at the end, nothing after.
        rise = max(1, int(min(sustain_attack, seconds / 3.0) * sample_rate))
        envelope = np.ones(length)
        envelope[:rise] = np.linspace(0.0, 1.0, rise)
        release_from = max(rise, length - int(fade_seconds * sample_rate))
        envelope[release_from:] *= np.linspace(1.0, 0.0, length - release_from)
        segment *= envelope * sustain_gain * velocity

        burst = min(length, int(burst_seconds * sample_rate))
        if burst > 16:
            segment[:burst] += (
                sosfilt(sos, rng.normal(0.0, 1.0, burst))
                * np.exp(-np.arange(burst) / sample_rate * burst_decay)
                * burst_gain
                * velocity
            )
        layer[start:start + length] += segment
    return layer


def _stereo(layer: np.ndarray, spread: float = 0.12) -> np.ndarray:
    """Place a generated layer across both channels, slightly decorrelated."""
    mono = np.asarray(layer, dtype=np.float32)
    left = mono * (1.0 - spread)
    right = mono * (1.0 + spread)
    return np.stack([left, right], axis=1)


def _vibrato(audio: np.ndarray, sample_rate: int, *, rate_hz: float,
             cents: float) -> np.ndarray:
    """A player's vibrato, as a modulated delay line.

    A delay that swings by ``A`` samples at ``R`` Hz moves the pitch by about
    ``f * 2*pi*R*A / sr``, so the depth that gives ``cents`` on a mid-register
    note is solved for directly instead of being tuned by ear.
    """
    if cents <= 0:
        return audio
    mid_hz = 480.0
    deviation = mid_hz * (2.0 ** (cents / 1200.0) - 1.0)
    depth = deviation * sample_rate / (mid_hz * 2.0 * math.pi * rate_hz)
    n = len(audio)
    if n < 64 or depth < 0.05:
        return audio
    t = np.arange(n) / sample_rate
    delay = depth * np.sin(2.0 * math.pi * rate_hz * t)
    index = np.arange(n) - delay
    base = np.floor(index).astype(np.int64)
    fraction = (index - base).astype(np.float32)
    base = np.clip(base, 0, n - 2)
    out = np.empty_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        column = audio[:, channel].astype(np.float32)
        out[:, channel] = (1.0 - fraction) * column[base] + fraction * column[base + 1]
    return out


def _body(audio: np.ndarray, sample_rate: int, *, lowpass: float,
          peak_hz: float, boost_db: float) -> np.ndarray:
    """The instrument's own body: a wooden low-pass and one resonance."""
    from scipy.signal import butter, iirpeak, sosfilt, lfilter

    sos = butter(2, min(lowpass, sample_rate * 0.45), btype="low", fs=sample_rate, output="sos")
    out = sosfilt(sos, audio, axis=0)
    if boost_db > 0.0:
        b, a = iirpeak(peak_hz / (0.5 * sample_rate), Q=1.4)
        resonance = lfilter(b, a, out, axis=0)
        out = out + resonance * (10.0 ** (boost_db / 20.0) - 1.0)
    return out.astype(np.float32)


def apply_expression(
    stem: np.ndarray,
    notes,
    *,
    kind: str,
    sample_rate: int = 44100,
    seconds_per_beat: float = 0.5,
    total: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Add the playing to a rendered stem.  ``kind`` is fiddle, flute or guitar."""
    audio = np.asarray(stem, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    audio = audio.copy()
    total = int(total or len(audio))
    if len(audio) < total:
        audio = np.pad(audio, ((0, total - len(audio)), (0, 0)))
    elif len(audio) > total:
        audio = audio[:total]
    rng = np.random.default_rng(seed + 13)
    if not notes:
        return audio

    if kind == "fiddle":
        # A player's vibrato arrives as the note settles, so it is applied to
        # the whole line at a depth a fiddle actually uses (~9 cents).
        audio = _vibrato(audio, sample_rate, rate_hz=5.6, cents=9.0)
        bow = _noise_layer(
            notes, seconds_per_beat=seconds_per_beat, total=total,
            sample_rate=sample_rate, rng=rng, band=(1800.0, 6800.0),
            burst_seconds=0.040, burst_decay=130.0, burst_gain=0.32,
            sustain_gain=0.075, sustain_attack=0.08,
        )
        audio += _stereo(bow, spread=0.10)
        audio = _body(audio, sample_rate, lowpass=7200.0, peak_hz=430.0, boost_db=2.0)
    elif kind == "flute":
        audio = _vibrato(audio, sample_rate, rate_hz=4.6, cents=5.0)
        breath = _noise_layer(
            notes, seconds_per_beat=seconds_per_beat, total=total,
            sample_rate=sample_rate, rng=rng, band=(900.0, 5200.0),
            burst_seconds=0.055, burst_decay=55.0, burst_gain=0.20,
            sustain_gain=0.070, sustain_attack=0.09,
        )
        audio += _stereo(breath, spread=0.14)
        audio = _body(audio, sample_rate, lowpass=8200.0, peak_hz=900.0, boost_db=1.5)
    elif kind == "guitar":
        # The pick is a transient only: no noise while the string rings.
        pick = _noise_layer(
            notes, seconds_per_beat=seconds_per_beat, total=total,
            sample_rate=sample_rate, rng=rng, band=(1600.0, 9000.0),
            burst_seconds=0.007, burst_decay=420.0, burst_gain=0.20,
            sustain_gain=0.0, fade_seconds=0.0,
        )
        audio += _stereo(pick, spread=0.30)
    else:  # pragma: no cover - callers pass a known kind
        raise ValueError(f"unknown expression kind {kind!r}")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio *= 0.95 / peak
    return audio.astype(np.float32)

"""What every trained voice gets on the way out, so the words survive the model.

An RVC checkpoint returns the voice it was trained to return, and that voice's
spectral balance is the model's own: a take trained on close, bright singing
comes back bright, while a take trained on a darker or more distant source
comes back with the 2-5 kHz band -- the burst that carries sibilants and the
front half of every consonant -- several dB under the voice it was handed. The
words are still in the output; they are just harder to make out, which is
exactly the "some models you can barely hear the words" complaint.

This stage is deliberately not an EQ preset. It measures the presence the
singer's own material has (2-5 kHz against the 300-1500 Hz body of the voice)
and restores only the part of it the model lost, with a zero-phase high shelf
that is capped at +3 dB by default. A model that already returns the input's
presence gets no filter at all -- the code checks the measurement before it
touches a sample, so this cannot make a warm voice brittle. A model that comes
back dull gets a gentle lift and nothing more.

It also owns peak safety. A converted take is written straight to a WAV, and a
sample above full scale is clipped by whatever plays it, which is both a click
and a loss of the transient that makes a consonant readable. The ceiling here
only ever turns a too-hot take down; it never normalises a quiet one up.

Measured on the project's own material: the baked neural takes carry between
0.4 and 3.0 per cent of their energy in 2-5 kHz, i.e. the same order as the
differences this is here to remove. The stage is held to its promises in
tests/test_voice_clarity.py, which builds a reference vowel, a deliberately
dull copy of it, and checks the restored copy on band ratio, loudness, length
and peak rather than on how it "sounds".
"""
from __future__ import annotations

import numpy as np
from scipy import signal

# The two bands the presence measurement is made from. The reference band is
# where a voice's body lives; the presence band is where its consonants live.
_PRESENCE_BAND_HZ = (2000.0, 5000.0)
_REFERENCE_BAND_HZ = (300.0, 1500.0)
_SHELF_HZ = 2200.0
_SHELF_Q = 0.707


def _as_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def _band_power(power: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    # Summing the bins of an STFT frame is proportional to the energy in the
    # band; the constant factor cancels in the ratio below.
    return float(np.sum(power[mask]))


def presence_ratio(x: np.ndarray, sr: int) -> float:
    """2-5 kHz energy over 300-1500 Hz energy, over the sounding frames only.

    A whole-file FFT would let a long quiet tail or a room-tone lead-in decide
    the answer for a two-second word. The measurement is made per frame and
    averaged over the frames that are actually carrying the voice, using the
    same -45 dB-from-peak convention the rest of this project calls "sounding".
    """
    x = _as_mono_float(x)
    if len(x) < 256 or sr <= 0:
        return 0.0
    nperseg = int(min(2048, len(x)))
    noverlap = int(nperseg * 3 // 4)
    try:
        freqs, _t, z = signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap,
                                   boundary=None, padded=False)
    except Exception:
        return 0.0
    power = np.abs(z) ** 2
    if power.size == 0:
        return 0.0
    frame_energy = power.sum(axis=0)
    peak = float(frame_energy.max()) if frame_energy.size else 0.0
    if peak <= 0.0:
        return 0.0
    keep = frame_energy >= peak * 10 ** (-45 / 10.0)
    if not np.any(keep):
        keep = np.ones_like(frame_energy, dtype=bool)
    mean_power = power[:, keep].mean(axis=1)
    body = _band_power(mean_power, freqs, _REFERENCE_BAND_HZ)
    presence = _band_power(mean_power, freqs, _PRESENCE_BAND_HZ)
    if body <= 1e-30:
        return 0.0
    return presence / body


def _high_shelf_sos(sr: int, gain_db: float, f0: float = _SHELF_HZ,
                    q: float = _SHELF_Q) -> np.ndarray:
    """RBJ high-shelf biquad, as one SOS row.

    Zero-phase application (sosfiltfilt) matters here: an ordinary causal
    shelf smears the onset of a consonant across the filter's own group delay,
    and a consonant's onset is the part that makes it legible.
    """
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def limit_peak(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Turn a too-hot take down; leave anything else exactly as it is."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak > ceiling > 0.0:
        x = x * (ceiling / peak)
    return x.astype(np.float32)


def apply_voice_clarity(out: np.ndarray, out_sr: int, reference: np.ndarray | None = None,
                        reference_sr: int | None = None, presence_match: bool = True,
                        max_lift_db: float = 3.0, ceiling: float = 0.99) -> np.ndarray:
    """Restore the presence a trained voice lost, then guarantee headroom.

    `reference` is the material the model was handed (the isolated vocal, or
    the DSP part in the offline renderer). When it is present and has a usable
    presence ratio, the output is lifted toward that ratio by at most
    `max_lift_db`; there is never a cut, because a model that came back
    brighter than its input is not the failure this exists to fix.
    """
    out = _as_mono_float(out)
    if out.size == 0:
        return np.asarray(out, dtype=np.float32)

    if presence_match and reference is not None and len(np.asarray(reference)) >= 256:
        ref_sr = int(reference_sr or out_sr)
        ref_ratio = presence_ratio(reference, ref_sr)
        out_ratio = presence_ratio(out, out_sr)
        if ref_ratio > 1e-9:
            if out_ratio > 1e-9:
                # The shelf multiplies the presence band's amplitude by
                # 10^(dB/20), so its energy by 10^(dB/10); that is the
                # conversion from a ratio gap to the filter gain that closes
                # it.
                needed_db = 10.0 * np.log10(ref_ratio / out_ratio)
            else:
                # A model can come back with essentially NO energy in the
                # presence band (a very dark checkpoint, or a heavily
                # low-passed training set). There is no ratio to divide by, so
                # ask for the most this stage is allowed to give rather than
                # giving up: that is the case it exists for.
                needed_db = max(0.0, float(max_lift_db))
            gain_db = float(np.clip(needed_db, 0.0, max(0.0, float(max_lift_db))))
            if gain_db >= 0.1:
                sos = _high_shelf_sos(out_sr, gain_db)
                before_rms = float(np.sqrt(np.mean(out ** 2)))
                try:
                    # sosfiltfilt needs a signal longer than its pad; short
                    # live chunks fall back to the causal filter rather than
                    # failing the whole conversion.
                    if out.size > 3 * (len(sos[0]) - 1):
                        filtered = signal.sosfiltfilt(sos, out)
                    else:
                        filtered = signal.sosfilt(sos, out)
                    if np.all(np.isfinite(filtered)):
                        after_rms = float(np.sqrt(np.mean(filtered ** 2)))
                        if before_rms > 1e-12 and after_rms > 1e-12:
                            # A shelf changes loudness slightly. This is a
                            # timbre fix, not a fader: hand back the level that
                            # went in, so the ROBOT VOLUME slider keeps meaning
                            # the same thing for every model.
                            filtered *= before_rms / after_rms
                        out = filtered
                except Exception:
                    pass

    return limit_peak(out, ceiling=ceiling)

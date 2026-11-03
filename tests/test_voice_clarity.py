"""What the shared clarity stage promises, measured rather than listened to.

Run with:  python -m pytest tests/test_voice_clarity.py -v

The fixtures are simple and made of exactly the thing the stage
cares about: a voice-like harmonic body with a consonant band above it. The
"dull" copy is that same signal through a low-pass, which is what a checkpoint
trained on darker material does to the band the words live in. The stage is
then asked for the four things it says it does -- restore presence, never cut,
leave the level alone, keep the peak under full scale -- and each one is a
number in the assertions below. No trained model is needed, so this runs on
any machine with numpy and scipy.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neural_env"))

from voice_clarity import (  # noqa: E402
    apply_voice_clarity,
    limit_peak,
    presence_ratio,
)

SR = 44100


def _voice_like(seconds: float = 2.0, f0: float = 220.0, consonant: float = 0.22) -> np.ndarray:
    """A steady vowel with a little band-limited air above it.

    The harmonic stack is the body (300-1500 Hz at this pitch), and the
    filtered noise is the 2-5 kHz presence a consonant would put there. Both
    are needed: presence_ratio would divide by zero on a bare noise burst.
    """
    t = np.arange(int(seconds * SR)) / SR
    body = sum((1.0 / (h ** 1.2)) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 9))
    body /= np.max(np.abs(body))
    rng = np.random.default_rng(7)
    air = rng.standard_normal(len(t))
    sos = signal.butter(4, [2000 / (SR / 2), 5000 / (SR / 2)], btype="bandpass", output="sos")
    air = signal.sosfiltfilt(sos, air)
    air /= max(1e-9, np.max(np.abs(air)))
    return (0.75 * body + consonant * air).astype(np.float32)


def _dull(x: np.ndarray, cutoff_hz: float = 1500.0) -> np.ndarray:
    sos = signal.butter(6, cutoff_hz / (SR / 2), btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def test_presence_ratio_responds_to_the_consonant_band():
    voice = _voice_like()
    dull = _dull(voice)
    assert presence_ratio(voice, SR) > 0.02
    assert presence_ratio(dull, SR) < presence_ratio(voice, SR) * 0.35


def test_dull_voice_is_lifted_toward_the_input():
    ref = _voice_like()
    dull = _dull(ref, cutoff_hz=1200.0)
    before = presence_ratio(dull, SR)
    fixed = apply_voice_clarity(dull, SR, reference=ref, reference_sr=SR,
                                presence_match=True, max_lift_db=3.0)
    after = presence_ratio(fixed, SR)
    target = presence_ratio(ref, SR)

    assert len(fixed) == len(dull)
    assert np.all(np.isfinite(fixed))
    assert after > before * 1.15, "the stage did not restore any presence"
    # It may not close the whole gap in one +3 dB shelf, but it must move
    # toward the input rather than past it.
    assert abs(np.log(after / target)) < abs(np.log(before / target))


def test_level_is_preserved_when_presence_is_lifted():
    ref = _voice_like()
    dull = _dull(ref, cutoff_hz=1200.0) * 0.4
    fixed = apply_voice_clarity(dull, SR, reference=ref, reference_sr=SR)
    rms_before = float(np.sqrt(np.mean(dull ** 2)))
    rms_after = float(np.sqrt(np.mean(fixed ** 2)))
    assert abs(20 * np.log10(rms_after / rms_before)) < 1.0


def test_a_voice_that_already_has_the_presence_is_not_touched():
    ref = _voice_like()
    fixed = apply_voice_clarity(ref, SR, reference=ref, reference_sr=SR)
    # The filter is skipped when the measured gap is under 0.1 dB, so the only
    # possible change is the peak safety stage, which is a no-op here.
    assert np.array_equal(fixed, limit_peak(ref))
    assert presence_ratio(fixed, SR) == pytest.approx(presence_ratio(ref, SR), rel=0.01)


def test_the_lift_is_capped():
    ref = _voice_like()
    dull = _dull(ref, cutoff_hz=700.0)
    before = presence_ratio(dull, SR)
    fixed = apply_voice_clarity(dull, SR, reference=ref, reference_sr=SR,
                                presence_match=True, max_lift_db=1.0)
    after = presence_ratio(fixed, SR)
    # A +1 dB shelf raises the band's energy by 10^(1/10); allow a little
    # measurement slack, but nothing near the +3 dB default.
    assert after / before < 10 ** (1.0 / 10.0) * 1.25


def test_peak_safety_turns_a_hot_take_down_without_clipping_it():
    t = np.arange(SR) / SR
    hot = (2.0 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    safe = limit_peak(hot, ceiling=0.99)
    assert float(np.max(np.abs(safe))) <= 0.99 + 1e-6
    # Scaling down is linear: the waveform's shape survives.
    corr = float(np.corrcoef(hot, safe)[0, 1])
    assert corr > 0.9999


def test_silence_and_empty_input_are_safe():
    assert apply_voice_clarity(np.zeros(0, dtype=np.float32), SR).size == 0
    tiny = np.zeros(64, dtype=np.float32)
    assert apply_voice_clarity(tiny, SR, reference=tiny, reference_sr=SR).size == 64


def test_presence_match_can_be_disabled():
    ref = _voice_like()
    dull = _dull(ref, cutoff_hz=1200.0)
    fixed = apply_voice_clarity(dull, SR, reference=ref, reference_sr=SR,
                                presence_match=False)
    assert np.array_equal(fixed, limit_peak(dull))

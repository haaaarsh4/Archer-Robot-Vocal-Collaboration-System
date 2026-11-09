"""Regression tests for the hand drum and the jingle dress.

The bands these belong to used to render through a General MIDI kit, which has
neither instrument: the drum came back as a pitched tom and the jingles as a
cabasa.  These tests pin the two properties that make each sound like itself and
that no drum-machine patch has -- where its energy sits and how it decays -- so
the kit cannot quietly take the bands back.

Two measurement notes, because both were learned the hard way:

* band shares come from summed STFT frames, never one Hann window over the whole
  hit.  A drum's spectrum lives in its first few tens of milliseconds and a
  windowed 0.85 s FFT has faded the signal to nothing exactly there, so every
  drum measures as a pure sine and the test would bless a tom;
* the fundamental is read from an unwindowed FFT with parabolic interpolation,
  because a 1024-sample frame at 44.1 kHz puts the next bin 43 Hz away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mysong.accompaniment import ArrangementStyle, arrange
from mysong.harmonize import ChordSegment, Harmonization
from mysong.percussion import (
    drum_fundamental_hz,
    percussion_hit,
    percussion_voices,
    render_percussion,
)
from mysong.theory import Chord, Key


def _bands(signal: np.ndarray, sample_rate: int = 44100):
    """Band energy shares and spectral centroid, from summed STFT frames."""
    import librosa

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = signal - signal.mean()
    power = np.abs(librosa.stft(signal, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
    total = float(power.sum()) or 1e-12
    shares = power.sum(axis=1) / total
    return freqs, shares, float((freqs * shares).sum())


def _share(freqs, shares, low: float, high: float) -> float:
    return float(shares[(freqs >= low) & (freqs < high)].sum())


def _fundamental(signal: np.ndarray, sample_rate: int = 44100,
                 low: float = 40.0, high: float = 400.0) -> float:
    """The strongest partial below ``high``, interpolated between bins."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    window = (freqs >= low) & (freqs <= high)
    index = int(np.argmax(np.where(window, spectrum, 0.0)))
    if 0 < index < len(spectrum) - 1:
        left, centre, right = spectrum[index - 1], spectrum[index], spectrum[index + 1]
        denominator = left - 2 * centre + right
        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            return float(freqs[index] + offset * (freqs[1] - freqs[0]))
    return float(freqs[index])


def _tail_ms(signal: np.ndarray, sample_rate: int = 44100, floor: float = 0.1) -> float:
    envelope = np.abs(np.asarray(signal, dtype=np.float64))
    quiet = np.argmax(envelope[::-1] > envelope.max() * floor)
    return 1000.0 * quiet / sample_rate


def _arrangement(style: str, *, tonic: int = 0, seconds: float = 4.0):
    """A four-beat arrangement in one key, without going through the model."""
    harmony = Harmonization(
        key=Key(tonic, "major"),
        segments=[ChordSegment(start_beat=0.0, end_beat=seconds, chord=Chord(tonic, "maj"))],
        score=0.0,
        key_scores=[],
        key_confidence=1.0,
        beats_per_bar=4.0,
        tempo_bpm=100.0,
    )
    return arrange(harmony, style=ArrangementStyle.preset(style))


def test_drum_is_tuned_to_the_key():
    for tonic in (0, 2, 4, 5, 7, 9, 11):
        fundamental = drum_fundamental_hz(tonic)
        assert 55.0 <= fundamental <= 100.0
        # The pitch class of the head is the tonic, which is what lets the drum
        # sit in the harmony instead of fighting it.
        midi = 69 + 12 * np.log2(fundamental / 440.0)
        assert round(midi) % 12 == tonic % 12


def test_drum_speaks_in_the_register_of_a_hide_drum():
    target = drum_fundamental_hz(0)
    signal = percussion_hit("boom", 100, 44100, fundamental=target)
    assert abs(_fundamental(signal) - target) < 0.03 * target

    freqs, shares, centroid = _bands(signal)
    # A struck hide is mostly low end ...
    assert _share(freqs, shares, 0, 250) > 0.55
    # ... and it is not a sine: the stick on skin under the modes is what carries
    # the stroke past a phone speaker, and without it this reads as a synth tom.
    assert _share(freqs, shares, 250, 1000) > 0.04
    assert centroid < 600.0
    # And it is damped: a drum rings for a moment, it does not sustain.
    assert 100.0 < _tail_ms(signal) < 900.0


def test_tap_is_brighter_and_shorter_than_the_accent():
    fundamental = drum_fundamental_hz(0)
    boom = percussion_hit("boom", 100, 44100, fundamental=fundamental)
    tap = percussion_hit("tap", 100, 44100, fundamental=fundamental)
    assert _bands(tap)[2] > _bands(boom)[2]
    assert _tail_ms(tap) < _tail_ms(boom)


def test_jingles_are_a_bright_metallic_cloud():
    signal = percussion_hit("jingle", 90, 44100)
    freqs, shares, centroid = _bands(signal)
    # The whole point: a jingle dress lives between about 2.5 and 8 kHz, where a
    # tom and a cabasa have nothing.
    assert _share(freqs, shares, 2500, 8000) > 0.70
    assert centroid > 3000.0
    assert _share(freqs, shares, 0, 1000) < 0.02
    # Cones keep ringing after the step, which is why the steps overlap.
    assert _tail_ms(signal) > 120.0


def test_two_steps_are_not_the_same_audio():
    first = percussion_hit("jingle", 80, 44100, index=0)
    second = percussion_hit("jingle", 80, 44100, index=1)
    assert first.shape == second.shape
    assert not np.allclose(first, second)


def test_only_the_indigenous_bands_take_these_instruments():
    assert set(percussion_voices("jingle_dress")) == {45, 47, 69, 54}
    assert percussion_voices("") == {}
    # A Western kit keeps its SoundFont drums.
    assert render_percussion(_arrangement("pop")) is None


@pytest.mark.parametrize("style", ["hand_drum", "jingle_dress", "grass_dance", "round_dance"])
def test_indigenous_bands_render_with_the_engine(style):
    arrangement = _arrangement(style)
    stem = render_percussion(arrangement)
    assert stem is not None
    assert stem.shape[1] == 2
    assert np.isfinite(stem).all()
    assert float(np.max(np.abs(stem))) > 0.1
    audio = __import__("mysong.accompaniment", fromlist=["render_arrangement"]).render_arrangement(
        arrangement, sample_rate=22050, renderer="synth"
    )
    assert np.isfinite(audio).all()
    assert float(np.sqrt(np.mean(audio ** 2))) > 1e-4


def test_jingle_dress_band_is_brighter_than_the_hand_drum_band():
    drum = render_percussion(_arrangement("hand_drum"))
    jingle = render_percussion(_arrangement("jingle_dress"))
    assert drum is not None and jingle is not None
    drum_high = _share(*_bands(drum.mean(axis=1))[:2], 2500, 8000)
    jingle_high = _share(*_bands(jingle.mean(axis=1))[:2], 2500, 8000)
    assert jingle_high > max(0.05, drum_high * 5)


def test_fallback_synth_can_play_the_shaker_notes():
    """The numpy fallback must not raise for a note the patterns emit."""
    from mysong.accompaniment import _drum_hit

    for kind in ("shaker", "rattle", "tom_low", "tom_mid", "tom_high"):
        hit = _drum_hit(kind, 80, 22050)
        assert hit.size > 0 and np.isfinite(hit).all()


def test_stem_is_written_at_a_level_the_mixer_expects():
    stem = render_percussion(_arrangement("jingle_dress"))
    assert stem is not None
    peak = float(np.max(np.abs(stem)))
    assert 0.45 <= peak <= 0.60

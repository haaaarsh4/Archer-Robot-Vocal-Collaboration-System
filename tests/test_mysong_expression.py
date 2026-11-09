"""Regression tests for the playing layers on the melodic indigenous bands.

A General MIDI fiddle or flute is one sampled note: no bow catching the string,
no air, no vibrato, no body.  These tests pin that the layers which turn those
patches into a person playing are actually in the signal, and that they land on
the notes the arrangement wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mysong.accompaniment import ArrangementNote, ArrangementStyle, arrange, render_arrangement
from mysong.expression import _vibrato, apply_expression
from mysong.harmonize import ChordSegment, Harmonization
from mysong.theory import Chord, Key

SR = 44100


def _band_energy(signal: np.ndarray, low: float, high: float, sr: int = SR) -> float:
    import librosa

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    power = np.abs(librosa.stft(signal, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    return float(power[(freqs >= low) & (freqs < high)].sum())


def _generated_stem(seconds: float = 2.0, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.35 * np.sin(2 * np.pi * hz * t) * np.exp(-t * 0.5)
    return np.stack([tone, tone], axis=1).astype(np.float32)


def _held_notes(count: int = 4) -> list[ArrangementNote]:
    return [
        ArrangementNote(start=float(i), duration=0.5, pitch=69 + i, velocity=80,
                        instrument="piano")
        for i in range(count)
    ]


def test_bands_use_a_solo_fiddle_and_a_wooden_flute():
    fiddle = ArrangementStyle.preset("metis_fiddle")
    flute = ArrangementStyle.preset("cedar_flute")
    # 110 is the solo fiddle; 41 is a violin section in a hall.
    assert fiddle.programmes.get("piano") == 110
    # 75 is the pan flute: wood and breath, not the metal flute at 73.
    assert flute.programmes.get("piano") == 75


def test_fiddle_expression_adds_a_bow():
    stem = _generated_stem()
    out = apply_expression(stem, _held_notes(), kind="fiddle", sample_rate=SR,
                           seconds_per_beat=0.5, total=len(stem), seed=1)
    assert out.shape == stem.shape
    assert np.isfinite(out).all()
    before = _band_energy(stem, 1800, 6800)
    after = _band_energy(out, 1800, 6800)
    # Bow noise is broadband and sits where the string's own harmonics are thin;
    # without it a GM violin is a filtered sawtooth with a slow attack.
    assert after > before * 4.0, f"bow energy {after:.3e} vs {before:.3e}"


def test_flute_expression_adds_breath():
    stem = _generated_stem()
    out = apply_expression(stem, _held_notes(), kind="flute", sample_rate=SR,
                           seconds_per_beat=0.5, total=len(stem), seed=2)
    assert np.isfinite(out).all()
    before = _band_energy(stem, 900, 5200)
    after = _band_energy(out, 900, 5200)
    assert after > before * 3.0, f"breath energy {after:.3e} vs {before:.3e}"


def test_guitar_expression_is_a_transient_not_a_hiss():
    stem = np.zeros_like(_generated_stem())
    notes = _held_notes(1)
    out = apply_expression(stem, notes, kind="guitar", sample_rate=SR,
                           seconds_per_beat=0.5, total=len(stem), seed=3)
    mono = np.abs(out.mean(axis=1))
    onset = float(mono[: int(0.03 * SR)].max())
    # 0.5 beat = 0.25 s here; by the end of the note there must be nothing left.
    tail = float(mono[int(0.2 * SR): int(0.25 * SR)].max())
    assert onset > 0.01
    assert tail < onset * 0.05


def test_vibrato_actually_moves_the_pitch():
    from scipy.signal import hilbert

    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 480.0 * t).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    out = _vibrato(stereo, SR, rate_hz=5.6, cents=9.0)
    assert out.shape == stereo.shape
    phase = np.unwrap(np.angle(hilbert(out[:, 0].astype(np.float64))))
    frequency = np.diff(phase) * SR / (2 * np.pi)
    # 9 cents on 480 Hz is about 2.5 Hz of swing either way.
    assert float(np.std(frequency[SR // 10:])) > 0.6
    assert abs(float(np.mean(frequency)) - 480.0) < 3.0


def test_expression_stays_inside_full_scale():
    stem = _generated_stem() * 4.0
    out = apply_expression(stem, _held_notes(), kind="fiddle", sample_rate=SR,
                           seconds_per_beat=0.5, total=len(stem), seed=4)
    assert float(np.max(np.abs(out))) <= 0.9501  # float32 rounding on the ceiling


def _harmonization(seconds: float = 4.0, tonic: int = 0) -> Harmonization:
    return Harmonization(
        key=Key(tonic, "major"),
        segments=[ChordSegment(start_beat=0.0, end_beat=seconds, chord=Chord(tonic, "maj"))],
        score=0.0, key_scores=[], key_confidence=1.0, beats_per_bar=4.0, tempo_bpm=100.0,
    )


@pytest.mark.parametrize("style", ArrangementStyle.INDIGENOUS)
def test_every_indigenous_band_renders(style):
    from mysong.accompaniment import soundfont_available

    if not soundfont_available():
        pytest.skip("FluidSynth or a General MIDI SoundFont is not installed")
    arrangement = arrange(_harmonization(), style=ArrangementStyle.preset(style))
    assert arrangement.notes
    audio = render_arrangement(arrangement, sample_rate=22050, renderer="soundfont")
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert np.isfinite(audio).all()
    assert float(np.sqrt(np.mean(audio ** 2))) > 1e-4
    assert float(np.max(np.abs(audio))) <= 1.0

"""Regression tests for the MySong harmonizer.

These are deliberately small: a synthetic melody for the decoder, a few
hundred corpus songs for the key classifier, and a random-weight tagger for
the inference path.  None of them trains a neural network.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mysong.accompaniment import ArrangementStyle, arrange, render_arrangement
from mysong.chord_model import ChordClassifier
from mysong.corpus import load_corpus, split_corpus
from mysong.harmonize import Harmonizer, Style
from mysong.key_model import KeyClassifier, key_from_index, key_index, song_chroma, train_key_classifier
from mysong.melody import Melody
from mysong.model import MySongModel
from mysong.theory import Chord, Key, parse_key


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def model():
    return MySongModel.load()


def _melody(beats: float = 8.0) -> Melody:
    # C major arpeggio, one note per beat, ending on a non-multiple-of-four
    # length so the tail-truncation path is exercised.
    notes = []
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for index in range(int(np.ceil(beats))):
        notes.append(__import__("mysong.corpus", fromlist=["NoteEvent"]).NoteEvent(
            onset=float(index), duration=1.0, pitch=pitches[index % len(pitches)]
        ))
    return Melody.from_notes(notes, beats_per_bar=4.0)


def test_key_roundtrip():
    for index in range(24):
        key = key_from_index(index)
        assert key_index(key) == index
        assert key.tonic in range(12)
        assert key.mode in ("major", "minor")


def test_key_classifier_is_strong(corpus):
    splits = split_corpus(corpus)
    classifier, report = train_key_classifier(splits["train"])
    correct = 0
    for song in splits["dev"]:
        predicted = classifier.top(song_chroma(song))
        correct += key_index(predicted) == key_index(song.key)
    accuracy = correct / len(splits["dev"])
    assert accuracy > 0.70, f"key classifier dev accuracy {accuracy:.3f}"
    assert report["train_accuracy"] > 0.80
    # transposition augmentation must cover every key, or a singer in one of
    # the corpus's missing keys would receive -20 log-probability
    assert len(classifier.classes) == 24
    assert classifier.predict_log_proba(song_chroma(splits["dev"][0])).shape == (24,)


def test_classifier_prior_correction_is_normalised():
    rng = np.random.default_rng(0)
    classifier = ChordClassifier(
        weights=rng.normal(size=(51, 4)),
        bias=np.zeros(51),
        mean=np.zeros(4),
        scale=np.ones(4),
        classes=np.arange(51),
        log_prior=np.log(np.ones(60) / 60.0),
    )
    log_probs = classifier.log_proba(np.zeros(4), prior_correction=0.7)
    total = float(np.exp(log_probs).sum())
    assert np.isclose(total, 1.0, atol=1e-6)
    assert log_probs.max() < 0.0


def test_decoder_covers_a_ragged_horizon(model):
    melody = _melody(7.0)
    harmony = Harmonizer(model).harmonize(
        melody,
        key=Key(0, "major"),
        style=Style(durations=(4.0,), max_duration=4.0, tagger_weight=12.0),
        beats_per_bar=4.0,
        with_marginals=False,
    )
    assert harmony.segments
    assert harmony.segments[0].start_beat == 0.0
    assert harmony.segments[-1].end_beat == pytest.approx(7.0)
    assert all(segment.end_beat > segment.start_beat for segment in harmony.segments)
    # every beat is covered exactly once
    covered = sum(segment.duration for segment in harmony.segments)
    assert covered == pytest.approx(7.0)


def test_harmonizer_uses_key_classifier(model):
    melody = _melody(8.0)
    harmony = Harmonizer(model).harmonize(
        melody, style=Style(key_candidates=1), beats_per_bar=4.0,
        with_marginals=False,
    )
    assert harmony.key.tonic in range(12)
    assert harmony.key.mode in ("major", "minor")
    assert harmony.key_confidence > 0.0


def test_fixed_bar_baseline(model):
    melody = _melody(9.0)
    harmony = Harmonizer(model).harmonize_fixed_bars(
        melody, key=Key(0, "major"), beats_per_bar=4.0
    )
    assert len(harmony.segments) == 3
    assert harmony.segments[-1].end_beat == pytest.approx(9.0)


def test_model_save_load(tmp_path, model):
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = MySongModel.load(path)
    assert loaded.key_classifier is not None
    assert np.allclose(loaded.classifier.weights, model.classifier.weights)
    assert np.allclose(loaded.key_classifier.weights, model.key_classifier.weights)
    assert loaded.tagger is None  # the .pt file was not copied


def test_tagger_features_and_inference(model):
    pytest.importorskip("torch")
    from mysong.neural import BeatTagger, Tagger, tagger_features

    melody = _melody(8.0)
    features = tagger_features(melody, 0, "major")
    assert features.shape == (8, 40)
    assert np.allclose(features[:, :12].sum(axis=1), 1.0)

    tagger = Tagger(BeatTagger(hidden=16))
    log_probs = tagger.log_proba(melody, Key(0, "major"))
    assert log_probs.shape == (8, 60)
    span = tagger.log_span(melody, Key(0, "major"), 1.0, 3.0)
    assert span.shape == (60,)
    assert np.isclose(span, log_probs[1:3].sum(axis=0)).all()


def test_colorize_returns_a_real_chord(model):
    melody = _melody(8.0)
    harmony = Harmonizer(model).harmonize(
        melody, key=Key(0, "major"), style=Style(color=1.0), beats_per_bar=4.0,
        with_marginals=True,
    )
    for segment in harmony.segments:
        assert isinstance(segment.chord, Chord)
        assert 0 <= segment.chord.root < 12
        assert segment.chord.quality
        assert 0.0 <= segment.confidence <= 1.0


def test_arrangement_renders_audio_and_midi(model):
    melody = _melody(8.0)
    harmony = Harmonizer(model).harmonize(
        melody, key=Key(0, "major"), beats_per_bar=4.0, with_marginals=False
    )
    arrangement = arrange(harmony, style=ArrangementStyle.preset("pop"))
    assert arrangement.notes
    midi = arrangement.to_midi_bytes()
    assert midi[:4] == b"MThd" and len(midi) > 64
    audio = render_arrangement(arrangement, sample_rate=22050, renderer="synth")
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert float(np.max(np.abs(audio))) > 0.0
    assert np.isfinite(audio).all()


def test_extra_corpus_has_no_openewld_melody_overlap():
    import json
    from pathlib import Path

    from mysong.corpus import load_corpus, melody_signature

    extra_path = Path("data/mysong/corpus_extra.jsonl")
    if not extra_path.exists():
        pytest.skip("auxiliary corpus is not built in this checkout")
    base = load_corpus()
    base_signatures = {
        signature for signature in (melody_signature(song) for song in base)
        if signature is not None
    }
    with extra_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            song = type(base[0]).from_dict(payload)
            signature = melody_signature(song)
            assert signature not in base_signatures, payload.get("title")


def test_midi_track_names_are_well_formed(model, tmp_path):
    from music21 import converter

    melody = _melody(8.0)
    harmony = Harmonizer(model).harmonize(
        melody, key=Key(0, "major"), beats_per_bar=4.0
    )
    arrangement = arrange(harmony, style=ArrangementStyle.preset("pop"))
    path = tmp_path / "song.mid"
    path.write_bytes(arrangement.to_midi_bytes())
    parsed = converter.parse(str(path))
    assert parsed.highestTime > 0
    assert len(parsed.flatten().notes) > 0


def test_soundfont_renderer_when_available(model):
    from mysong.accompaniment import soundfont_available

    if not soundfont_available():
        pytest.skip("FluidSynth or a General MIDI SoundFont is not installed")
    melody = _melody(8.0)
    harmony = Harmonizer(model).harmonize(
        melody, key=Key(0, "major"), beats_per_bar=4.0
    )
    arrangement = arrange(harmony, style=ArrangementStyle.preset("pop"))
    audio = render_arrangement(arrangement, sample_rate=22050, renderer="soundfont")
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert np.isfinite(audio).all()
    assert float(np.sqrt(np.mean(audio ** 2))) > 1e-4

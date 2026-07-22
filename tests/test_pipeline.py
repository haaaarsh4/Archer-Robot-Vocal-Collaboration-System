import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Two helper functions to generate fake audio
def make_sine(freq_hz: float, duration_s: float = 0.1, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)


def make_silence(duration_s: float = 0.1, sample_rate: int = 44100) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


# Tests the audio cleaner, total of four checks
class TestPreprocessor:
    # Creates a fresh instance of the module being tested
    def setup_method(self):
        from core.preprocessor import Preprocessor
        self.p = Preprocessor()

    def test_silence_is_gated(self):
        frame = make_silence()
        _, is_voiced = self.p.process(frame)
        assert is_voiced is False

    def test_signal_passes_gate(self):
        frame = make_sine(440.0)
        _, is_voiced = self.p.process(frame)
        assert is_voiced is True

    def test_output_is_normalized(self):
        # Very loud input
        frame = make_sine(440.0) * 10.0
        cleaned, is_voiced = self.p.process(frame)
        assert is_voiced is True
        # Peak should be clipped to [-1, 1]
        assert np.max(np.abs(cleaned)) <= 1.0

    def test_output_dtype(self):
        frame = make_sine(440.0)
        cleaned, _ = self.p.process(frame)
        assert cleaned.dtype == np.float32


# Tests the pitch detector, three checks
class TestPitchDetector:

    def setup_method(self):
        from analysis.pitch_detector import PitchDetector
        self.d = PitchDetector()

    def test_hz_to_note_name(self):
        note = self.d.hz_to_note_name(440.0)
        assert "A" in note

    def test_hz_to_midi(self):
        midi = self.d.hz_to_midi(440.0)
        assert abs(midi - 69.0) < 0.5

    def test_silence_returns_none(self):
        frame = make_silence()
        hz, conf = self.d.detect(frame)
        # Silence should give no reliable pitch
        assert conf < 0.5 or hz is None


# Tests the rhythm tracker, four checks:
class TestRhythmAnalyzer:

    def setup_method(self):
        from analysis.rhythm_analyzer import RhythmAnalyzer
        self.r = RhythmAnalyzer()

    def test_silence_sets_phrase_state(self):
        silence = make_silence(duration_s=0.5)
        frame_size = 1024
        for i in range(0, len(silence), frame_size):
            chunk = silence[i:i + frame_size]
            if len(chunk) == frame_size:
                self.r.push_frame(chunk, is_voiced=False)
        assert self.r.phrase_state in ("phrase_end", "silence")

    def test_voiced_frames_set_singing_state(self):
        frame = make_sine(440.0, duration_s=0.023)
        self.r.push_frame(frame, is_voiced=True)
        assert self.r.phrase_state == "singing"

    def test_reset_clears_state(self):
        frame = make_sine(440.0, duration_s=0.023)
        self.r.push_frame(frame, is_voiced=True)
        self.r.reset()
        assert self.r.phrase_state == "silence"
        assert self.r.current_tempo == 0.0

    def test_beat_duration_default(self):
        # When no tempo detected, should return a sensible default
        dur = self.r.beat_duration_seconds
        assert dur > 0


# Tests the musical brain, six checks:
class TestHarmonyEngine:

    def setup_method(self):
        from synthesis.harmony_engine import HarmonyEngine
        from analysis.phonetic_analysis import PhonemeProfile
        self.engine = HarmonyEngine()
        self.neutral_profile = PhonemeProfile()

    def test_rest_on_silence(self):
        decision = self.engine.decide(
            archer_hz=None,
            phrase_state="silence",
            tempo_bpm=120.0,
            phoneme_profile=self.neutral_profile,
        )
        assert decision.action == "rest"

    def test_sing_on_voiced(self):
        decision = self.engine.decide(
            archer_hz=440.0,
            phrase_state="singing",
            tempo_bpm=120.0,
            phoneme_profile=self.neutral_profile,
        )
        assert decision.action == "sing"
        assert decision.target_hz > 0

    def test_third_interval(self):
        self.engine.set_interval("third")
        decision = self.engine.decide(
            archer_hz=440.0,
            phrase_state="singing",
            tempo_bpm=120.0,
            phoneme_profile=self.neutral_profile,
        )
        # A major third above 440Hz is 440 * 2^(4/12) ≈ 554Hz
        # Scale lock may shift it slightly, so allow ±50Hz
        assert 500 < decision.target_hz < 610

    def test_fifth_interval(self):
        self.engine.set_interval("fifth")
        decision = self.engine.decide(
            archer_hz=440.0,
            phrase_state="singing",
            tempo_bpm=120.0,
            phoneme_profile=self.neutral_profile,
        )
        # Perfect fifth above 440Hz is 440 * 2^(7/12) ≈ 659Hz
        assert 620 < decision.target_hz < 700

    def test_vocable_is_valid(self):
        decision = self.engine.decide(
            archer_hz=440.0,
            phrase_state="singing",
            tempo_bpm=120.0,
            phoneme_profile=self.neutral_profile,
        )
        assert decision.vocable in ["aah", "ooo", "mmm", "hey"]

    def test_set_invalid_interval_does_not_crash(self):
        self.engine.set_interval("invalid_interval")
        # Should log warning but not raise
        assert self.engine._current_interval == "fifth"  # unchanged


# Tests the audio generator, five checks:
class TestVocableSynthesizer:

    def setup_method(self):
        from synthesis.vocable_synthesizer import VocableSynthesizer
        from synthesis.harmony_engine import HarmonyDecision
        self.synth = VocableSynthesizer()
        self.HarmonyDecision = HarmonyDecision

    def _make_decision(self, hz=440.0, action="sing"):
        return self.HarmonyDecision(
            target_hz=hz,
            vocable="aah",
            duration_s=0.2,
            vowel_color=0.5,
            nasality=0.0,
            brightness=0.5,
            action=action,
        )

    def test_rest_returns_silence(self):
        d = self._make_decision(action="rest")
        audio = self.synth.synthesize(d)
        assert np.all(audio == 0)

    def test_sing_returns_nonzero_audio(self):
        d = self._make_decision(action="sing")
        audio = self.synth.synthesize(d)
        assert len(audio) > 0
        assert np.any(audio != 0)

    def test_output_dtype(self):
        d = self._make_decision()
        audio = self.synth.synthesize(d)
        assert audio.dtype == np.float32

    def test_output_clipped(self):
        d = self._make_decision()
        audio = self.synth.synthesize(d)
        assert np.max(np.abs(audio)) <= 1.0

    def test_different_pitches_produce_different_audio(self):
        d1 = self._make_decision(hz=220.0)
        d2 = self._make_decision(hz=440.0)
        a1 = self.synth.synthesize(d1)
        a2 = self.synth.synthesize(d2)
        min_len = min(len(a1), len(a2))
        assert not np.allclose(a1[:min_len], a2[:min_len])


# Tests the Cree phoneme analyzer, two checks:
class TestCreeTokenizer:

    def setup_method(self):
        from analysis.phonetic_analysis import CreeTokenizer
        self.tok = CreeTokenizer()

    def test_disabled_returns_neutral(self):
        # Default config has enabled=false
        frame = make_sine(440.0)
        profile = self.tok.analyze(frame)
        assert profile.influence == 0.0
        assert profile.detected_class == "neutral"

    def test_profile_fields_in_range(self):
        frame = make_sine(440.0)
        profile = self.tok.analyze(frame)
        assert 0.0 <= profile.vowel_color <= 1.0
        assert 0.0 <= profile.nasality <= 1.0
        assert 0.0 <= profile.brightness <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
import numpy as np
import collections
from dataclasses import dataclass
from loguru import logger
from config.config_loader import get_config
from analysis.cree_tokenizer import PhonemeProfile
import librosa

# Output container
@dataclass
class HarmonyDecision:
    target_hz  : float              # the exact frequency the robot should sing
    vocable    : str                # the vowel shape: 'aah', 'ooo', 'mmm', 'hey'
    duration_s : float              # how long to hold the note in seconds
    vowel_color: float              # timbre darkness from Cree profile
    nasality   : float              # nasal resonance from Cree profile
    brightness : float              # spectral brightness from Cree profile
    action     : str                # either of 'sing', 'sustain', or 'rest'

class HarmonyEngine:
    # Semitone intervals above Archer's note
    INTERVALS = {
        "unison": 0,
        "third":  4,   
        "fifth":  7,
        "octave": 12,
    }

    # Pentatonic scale degrees (in semitones from root)
    PENTATONIC = [0, 2, 4, 7, 9]

    # Major and natural minor for scale-lock
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    # Loads config and sets up all the state
    def __init__(self):
        cfg = get_config()
        harmony_cfg = cfg["harmony"]
        self.default_interval = harmony_cfg["default_interval"]
        self.detune_cents = harmony_cfg["detune_cents"]
        self.scale_lock = harmony_cfg["scale_lock"]
        self.sample_rate = cfg["audio"]["sample_rate"]

        # Rolling pitch history to infer key
        self._pitch_history: collections.deque = collections.deque(maxlen=32)
        self._current_key_root: int = 0       # MIDI note of detected key root
        self._current_scale: list = self.PENTATONIC
        self._current_interval: str = self.default_interval

        # Track what the robot is currently doing
        self._current_decision: HarmonyDecision | None = None
        self._frames_on_current_note: int = 0
        self._max_frames_per_note: int = 200  # safety cap

    # Given the current state of Archer's singing, decide what the robot does
    def decide(
        self,
        archer_hz: float | None,
        phrase_state: str,
        tempo_bpm: float,
        phoneme_profile: PhonemeProfile,
    ):

        # If Archer is silent then the robot rests
        if archer_hz is None or phrase_state in ("phrase_end", "silence"):
            return HarmonyDecision(
                target_hz=0.0,
                vocable="mmm",
                duration_s=0.0,
                vowel_color=0.5,
                nasality=0.0,
                brightness=0.5,
                action="rest",
            )

        # Update pitch history and infer key
        self._pitch_history.append(archer_hz)
        self._infer_key()

        # Choose harmony note
        harmony_hz = self._choose_harmony_pitch(archer_hz)

        # Choose vocable based on phoneme profile
        vocable = self._choose_vocable(phoneme_profile)

        # Note duration: one beat, clipped to max
        beat_dur = (60.0 / tempo_bpm) if tempo_bpm > 0 else 0.5
        duration = min(beat_dur, 4.0)

        # Blend Cree phoneme influence into timbre params
        influence = phoneme_profile.influence
        vowel_color = 0.5 * (1 - influence) + phoneme_profile.vowel_color * influence
        nasality    = phoneme_profile.nasality * influence
        brightness  = 0.5 * (1 - influence) + phoneme_profile.brightness * influence

        decision = HarmonyDecision(
            target_hz=harmony_hz,
            vocable=vocable,
            duration_s=duration,
            vowel_color=vowel_color,
            nasality=nasality,
            brightness=brightness,
            action="sing",
        )

        self._current_decision = decision
        self._frames_on_current_note += 1
        return decision

    # Decide what the robot's note should be
    def _choose_harmony_pitch(self, archer_hz: float) -> float:
        semitones = self.INTERVALS.get(self._current_interval, 4)
        harmony_hz = archer_hz * (2 ** (semitones / 12.0))

        if self.scale_lock:
            harmony_hz = self._snap_to_scale(harmony_hz)

        if self.detune_cents != 0:
            harmony_hz *= 2 ** (self.detune_cents / 1200.0)

        return harmony_hz

    # Snaps a frequency to the nearest note in the currently detected scale
    def _snap_to_scale(self, freq_hz: float) -> float:
        try:
            midi = librosa.hz_to_midi(freq_hz)
            note_class = int(round(midi)) % 12

            # Find the nearest scale degree
            distances = [abs(note_class - (self._current_key_root + d) % 12)
                         for d in self._current_scale]
            nearest_degree = self._current_scale[int(np.argmin(distances))]
            snapped_note_class = (self._current_key_root + nearest_degree) % 12

            # Reconstruct the MIDI note in the same octave
            octave = int(midi) // 12
            snapped_midi = octave * 12 + snapped_note_class
            return float(librosa.midi_to_hz(snapped_midi))

        except Exception as e:
            logger.error(f"Scale snap error: {e}")
            return freq_hz

    # Figures out what key Archer is singing in by analyzing his last 32 pitches
    def _infer_key(self):
        if len(self._pitch_history) < 8:
            return
        try:
            midi_notes = [int(round(librosa.hz_to_midi(hz)))
                          for hz in self._pitch_history if hz > 0]
            if not midi_notes:
                return

            # Build a chroma distribution from recent notes
            chroma = np.zeros(12)
            for n in midi_notes:
                chroma[n % 12] += 1
            chroma /= chroma.sum()

            # Simple Krumhansl-style key finding:
            # correlate against major and minor key profiles
            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,
                                       2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,
                                       2.54,4.75,3.98,2.69,3.34,3.17])

            best_corr = -np.inf
            best_root = 0
            best_scale = self.MAJOR_SCALE

            for root in range(12):
                shifted_chroma = np.roll(chroma, -root)
                major_corr = float(np.corrcoef(shifted_chroma, major_profile)[0, 1])
                minor_corr = float(np.corrcoef(shifted_chroma, minor_profile)[0, 1])

                if major_corr > best_corr:
                    best_corr, best_root = major_corr, root
                    best_scale = self.MAJOR_SCALE
                if minor_corr > best_corr:
                    best_corr, best_root = minor_corr, root
                    best_scale = self.MINOR_SCALE

            if best_root != self._current_key_root or best_scale != self._current_scale:
                self._current_key_root = best_root
                self._current_scale = best_scale
                logger.debug(f"Key updated: root={best_root}, scale={best_scale}")

        except Exception as e:
            logger.error(f"Key inference error: {e}")

    # Picks which vowel shape the robot should use based on the Cree phoneme profile
    def _choose_vocable(self, profile: PhonemeProfile) -> str:
        if profile.influence < 0.1:
            # No Cree influence the cycle through defaults
            defaults = ["aah", "ooo", "mmm", "hey"]
            idx = len(self._pitch_history) % len(defaults)
            return defaults[idx]

        # Map vowel color to vocable
        if profile.vowel_color < 0.25:
            return "hey"     # front/bright vowel
        elif profile.vowel_color < 0.5:
            return "aah"     # mid vowel
        elif profile.vowel_color < 0.75:
            return "ooo"     # back vowel
        else:
            return "mmm"     # dark/nasal

    # function to let the pipeline change the harmony interval live during a performance without restarting anything
    def set_interval(self, interval: str):
        if interval in self.INTERVALS:
            self._current_interval = interval
            logger.info(f"Interval set to {interval}")
        else:
            logger.warning(f"Unknown interval '{interval}'. Options: {list(self.INTERVALS)}")

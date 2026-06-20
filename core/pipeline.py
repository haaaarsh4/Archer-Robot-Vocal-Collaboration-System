import threading
import queue
import time
import numpy as np
from loguru import logger
from config.config_loader import get_config

from core.audio_capture    import AudioCapture
from core.preprocessor     import Preprocessor
from analysis.pitch_detector  import PitchDetector
from analysis.rhythm_analyzer import RhythmAnalyzer
from analysis.cree_tokenizer  import CreeTokenizer, PhonemeProfile
from synthesis.harmony_engine    import HarmonyEngine
from synthesis.vocable_synthesizer import VocableSynthesizer
from output.timing_sync    import TimingSync


class Pipeline:

    def __init__(self):
        logger.info("Initializing Archer-Robot Vocal Collaboration pipeline")

        self.capture      = AudioCapture()
        self.preprocessor = Preprocessor()
        self.pitch        = PitchDetector()
        self.rhythm       = RhythmAnalyzer()
        self.cree         = CreeTokenizer()
        self.harmony      = HarmonyEngine()
        self.synth        = VocableSynthesizer()
        self.timing       = TimingSync()

        self._running       = False
        self._frame_count   = 0
        self._start_time    = 0.0

        # FIX (BUG 4): use confidence threshold from config, not hard-coded 0.4.
        cfg = get_config()
        self._confidence_threshold = cfg["pitch"]["confidence_threshold"]

        logger.info("Pipeline initialized, all modules ready!")

    # ------------------------------------------------------------------ #

    def start(self):
        self._running    = True
        self._start_time = time.perf_counter()

        self.timing.start()
        self.capture.start()

        logger.info("Pipeline running, Archer can sing now")

        try:
            self._processing_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping pipeline")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.capture.stop()
        self.timing.stop()

        elapsed = time.perf_counter() - self._start_time
        logger.info(
            f"Pipeline stopped after {elapsed:.1f}s, "
            f"{self._frame_count} frames processed"
        )

    # ------------------------------------------------------------------ #

    def _processing_loop(self):
        self._currently_singing = False

        while self._running:
            try:
                frame = self.capture.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._frame_count += 1
            clean_frame, is_voiced = self.preprocessor.process(frame)
            self.rhythm.push_frame(clean_frame, is_voiced)

            archer_hz       = None
            phoneme_profile = self.cree._neutral_profile

            if is_voiced:
                # FIX (BUG 4): use config-driven confidence threshold.
                archer_hz, confidence = self.pitch.detect(clean_frame)
                if (archer_hz is not None
                        and confidence >= self._confidence_threshold
                        and self.pitch.min_freq < archer_hz < self.pitch.max_freq):
                    note = self.pitch.hz_to_note_name(archer_hz)
                    logger.debug(
                        f"[{self._elapsed:.2f}s] Archer: {archer_hz:.1f} Hz ({note}), "
                        f"conf={confidence:.2f}, BPM={self.rhythm.current_tempo:.1f}"
                    )
                else:
                    archer_hz = None
                phoneme_profile = self.cree.analyze(clean_frame)

            # When Archer stops singing, reset the pitch accumulation buffer
            # so old audio is not mixed with the next phrase.
            if not is_voiced:
                self.pitch.reset()

            self.timing.update_tempo(self.rhythm.current_tempo)

            # Allow singing even if phrase_state lags behind a real detected pitch
            phrase_state = self.rhythm.phrase_state
            if archer_hz is not None and phrase_state in ("silence", "phrase_end"):
                phrase_state = "singing"

            decision = self.harmony.decide(
                archer_hz       = archer_hz,
                phrase_state    = phrase_state,
                tempo_bpm       = self.rhythm.current_tempo,
                phoneme_profile = phoneme_profile,
            )

            if decision.action == "sing":
                audio = self.synth.synthesize(decision)
                self.timing.schedule(audio, decision.action)
                self._currently_singing = True
            elif decision.action == "sustain":
                self._currently_singing = True
            else:   # rest
                if self._currently_singing:
                    self.timing.flush()
                self._currently_singing = False


    @property
    def _elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def set_interval(self, interval: str):
        self.harmony.set_interval(interval)

    def set_noise_threshold(self, threshold: float):
        self.preprocessor.update_threshold(threshold)

    def reset_rhythm(self):
        self.rhythm.reset()

    def status(self) -> dict:
        return {
            "running":               self._running,
            "elapsed_s":             round(self._elapsed, 1),
            "frames_processed":      self._frame_count,
            "current_tempo_bpm":     round(self.rhythm.current_tempo, 1),
            "phrase_state":          self.rhythm.phrase_state,
            "harmony_interval":      self.harmony._current_interval,
            "synthesis_engine":      self.synth.engine,
            "cree_tokenizer_enabled": self.cree.enabled,
        }

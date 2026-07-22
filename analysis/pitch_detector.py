import numpy as np
import aubio
from loguru import logger
from config.config_loader import get_config
import librosa


class PitchDetector:

    HOP_SIZE = 512
    BUF_SIZE = 2048

    def __init__(self):
        cfg = get_config()
        self.sample_rate          = cfg["audio"]["sample_rate"]
        self.confidence_threshold = cfg["pitch"]["confidence_threshold"]
        self.min_freq             = float(cfg["pitch"]["min_frequency"])
        self.max_freq             = float(cfg["pitch"]["max_frequency"])

        # "yin" is the classic time-domain method already in use here.
        # "yinfft" is a spectral variant aubio also ships, and tends to hold
        # up better on real-world, non-isolated audio (background noise,
        # room tone) at a small extra CPU cost. Configurable rather than
        # picked for you, since which one performs better depends on your
        # actual input signal, not something to assume from a synthetic test.
        self._pitch_method = cfg["pitch"].get("method", "yin")
        # Stored so reset() can rebuild an identical aubio object. See the
        # note in reset() for why that rebuild exists.
        self._silence_threshold_db = cfg["preprocessing"]["silence_threshold_db"]
        self._tolerance = cfg["pitch"].get("tolerance", 0.15)

        self._pitch_o = self._build_pitch_object()

        self._leftovers   = np.array([], dtype=np.float32)
        self._history     = []
        self._HISTORY_LEN = 3

        # How far a new pitch can drift from the last stable one, in cents,
        # before it's treated as a genuinely new note rather than smoothed
        # in with recent history. Without this, a real jump between two
        # different notes gets briefly averaged into a pitch that matches
        # neither, right at the moment it matters most.
        self._jump_tolerance_cents = cfg["pitch"].get("jump_tolerance_cents", 80.0)
        self._last_stable_hz: float | None = None

        # A single frame landing far from the last stable pitch could be a
        # real note change, or it could be one bad reading (see reset()'s
        # docstring for where those come from). Rather than trusting it
        # instantly, it's held as "pending" and only promoted to the new
        # stable pitch once a second consecutive frame lands near it too.
        # A lone bad frame in between just gets ignored, with the old
        # stable pitch returned unchanged for that one frame.
        self._pending_jump_hz: float | None = None

        logger.info(
            f"Pitch engine: aubio {self._pitch_method} (hop={self.HOP_SIZE}, buf={self.BUF_SIZE}, "
            f"sr={self.sample_rate}, fmin={self.min_freq:.0f} Hz, fmax={self.max_freq:.0f} Hz)"
        )

    def _build_pitch_object(self):
        pitch_o = aubio.pitch(self._pitch_method, self.BUF_SIZE, self.HOP_SIZE, self.sample_rate)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(self._silence_threshold_db)
        pitch_o.set_tolerance(self._tolerance)
        return pitch_o

    def detect(self, frame: np.ndarray) -> tuple:
        samples = np.concatenate([self._leftovers, frame.astype(np.float32)])

        candidates: list[tuple[float, float]] = []  # (hz, confidence) for every in-range hop this call
        i = 0
        while i + self.HOP_SIZE <= len(samples):
            hop  = samples[i:i + self.HOP_SIZE]
            hz   = float(self._pitch_o(hop)[0])
            conf = float(self._pitch_o.get_confidence())
            i   += self.HOP_SIZE
            if self.min_freq <= hz <= self.max_freq and conf >= self.confidence_threshold:
                candidates.append((hz, conf))
        self._leftovers = samples[i:]

        if not candidates:
            return None, 0.0

        # Actually pick the highest-confidence candidate from this batch,
        # not just whichever one happened to be processed last.
        best_hz, best_conf = max(candidates, key=lambda c: c[1])

        smoothed_hz = self._smooth(best_hz)
        return smoothed_hz, best_conf

    def _smooth(self, hz: float) -> float:
        """
        Median-of-recent smoothing, but aware of real jumps. A new pitch
        more than _jump_tolerance_cents away from the last stable reading
        is treated as a candidate new note rather than blended into recent
        history, which is what caused a couple of frames of wrong,
        in-between output right at the start of every real pitch change.

        That candidate isn't trusted on a single frame, though. It's held
        as pending and only promoted to the new stable pitch once a second
        consecutive frame confirms it by landing near the same value. A
        single stray misread (see reset()'s docstring for where those come
        from) gets ignored instead of instantly yanking the whole tracker
        to a wrong note, which is what was happening before: one bad frame
        was enough to overwrite _last_stable_hz outright, and because nothing
        after that point still agreed with it, the tracker could bounce
        between wrong anchors for several frames in a row before the real
        pitch caught back up on its own.

        Small drift (vibrato, natural pitch wobble) still gets smoothed
        normally, since it never crosses the jump threshold at all.
        """
        if self._last_stable_hz is not None and self._last_stable_hz > 0:
            cents_diff = abs(1200.0 * np.log2(hz / self._last_stable_hz))
            if cents_diff > self._jump_tolerance_cents:
                if self._pending_jump_hz is not None:
                    confirm_diff = abs(1200.0 * np.log2(hz / self._pending_jump_hz))
                    if confirm_diff <= self._jump_tolerance_cents:
                        # Two consecutive frames agree on the new pitch — commit it.
                        self._history = [self._pending_jump_hz, hz]
                        self._last_stable_hz = hz
                        self._pending_jump_hz = None
                        return hz
                # Not confirmed (or contradicts a still-pending candidate).
                # Hold the old stable pitch and wait one more frame.
                self._pending_jump_hz = hz
                return self._last_stable_hz
            else:
                self._pending_jump_hz = None
        else:
            self._pending_jump_hz = None

        self._history.append(hz)
        if len(self._history) > self._HISTORY_LEN:
            self._history.pop(0)

        result = float(np.median(self._history))
        self._last_stable_hz = result
        return result

    def reset(self):
        """
        Called every time the caller sees an unvoiced frame (server.py calls
        this on every single silent frame, not just once per phrase — so
        this runs constantly during any pause between notes).

        Clearing the Python-side smoothing state here was already happening,
        but the underlying aubio pitch object was left untouched. aubio's
        YIN implementation keeps its own internal ring buffer of BUF_SIZE
        (2048) samples spanning several hops of history, maintained inside
        the C object across calls, on the assumption that it's being fed a
        continuous audio stream. Silence still gets fed through detect() up
        until the caller stops calling it, so the aubio object's internal
        buffer doesn't go empty during a pause — it just sits there full of
        stale pre-silence audio.

        When singing resumes, the first several calls mix that stale
        leftover audio with the new audio inside aubio's own window before
        it's fully flushed out (up to BUF_SIZE / HOP_SIZE = 4 hops), which
        can produce a handful of genuinely garbage pitch readings right at
        the start of a new note — not just occasional jitter, but readings
        confident enough to pass the confidence threshold. Since this reset
        happens after every note (any brief gap between notes triggers it),
        the effect compounds over a session: it's not that pitch detection
        degrades on its own, it's that every new note after the first one
        starts from a slightly corrupted analysis window.

        The fix is to rebuild the aubio object itself on a real reset, so
        it starts the next note with a clean internal buffer instead of one
        full of the previous note's tail end. Guarded so it only rebuilds
        once per silence, not on every one of the ~40 silent frames a second
        that server.py's continuous reset() calls would otherwise trigger.
        """
        already_clean = (
            not self._history
            and self._last_stable_hz is None
            and self._pending_jump_hz is None
            and len(self._leftovers) == 0
        )
        if already_clean:
            return

        self._history.clear()
        self._leftovers = np.array([], dtype=np.float32)
        self._last_stable_hz = None
        self._pending_jump_hz = None
        self._pitch_o = self._build_pitch_object()

    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        if freq_hz <= 0:
            return "unknown"
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def hz_to_midi(freq_hz: float) -> float:
        return float(librosa.hz_to_midi(freq_hz))
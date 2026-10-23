"""Turn a performance (audio or symbolic notes) into what the harmonizer needs.

Two paths produce the same object:

``Melody.from_notes``
    Exact symbolic input.  Used for corpus evaluation, for MIDI files and for
    the unit tests.

``analyze_audio``
    A sung take.  Pitch is tracked with probabilistic YIN, the recording's
    tuning is estimated and removed, and every frame contributes a
    Gaussian-wrapped chroma vector.  Unlike a hard pitch quantizer this keeps
    the information in a slightly flat or sharp note instead of throwing the
    note away, which is the same insight MySong's authors used when they
    skipped the dynamic-programming cleanup of their pitch tracker.

The output is a :class:`Melody`: a uniformly sampled 12-bin chroma matrix plus
the beat grid.  The harmonizer only ever asks it for the chroma between two
beat positions, which is a prefix-sum lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mysong.corpus import NoteEvent
from mysong.theory import Key, estimate_key, metric_weight, pc_name


# --------------------------------------------------------------------------- #
# The shared representation
# --------------------------------------------------------------------------- #

class Melody:
    """Metric-weighted chroma on a quarter-note frame grid.

    ``chroma[k]`` covers ``[k * frame_beats, (k + 1) * frame_beats)`` quarter
    notes.  Rows already include the metrical accent of their position, so the
    decoder can simply sum rows over a candidate chord span.
    """

    def __init__(
        self,
        chroma: np.ndarray,
        *,
        frame_beats: float = 0.25,
        beats_per_bar: float = 4.0,
        tempo_bpm: float | None = None,
        total_beats: float | None = None,
        notes: Sequence[NoteEvent] | None = None,
        tuning_cents: float = 0.0,
        voiced_ratio: float = 1.0,
        source: str = "",
    ) -> None:
        chroma = np.asarray(chroma, dtype=np.float64)
        if chroma.ndim != 2 or chroma.shape[1] != 12:
            raise ValueError(f"chroma must be (frames, 12), got {chroma.shape}")
        if frame_beats <= 0:
            raise ValueError("frame_beats must be positive")
        self.chroma = chroma
        self.frame_beats = float(frame_beats)
        self.beats_per_bar = float(beats_per_bar)
        self.tempo_bpm = tempo_bpm
        self.notes = list(notes or [])
        self.tuning_cents = float(tuning_cents)
        self.voiced_ratio = float(voiced_ratio)
        self.source = source
        frames = len(chroma)
        self.total_beats = float(total_beats) if total_beats is not None else frames * self.frame_beats
        self._prefix = np.vstack([np.zeros(12), np.cumsum(chroma, axis=0)])
        self._strong_prefix: np.ndarray | None = None

    # -- geometry -------------------------------------------------------- #
    @property
    def prefix(self) -> np.ndarray:
        """Cumulative chroma, shape ``(frames + 1, 12)``."""
        return self._prefix

    def _index(self, beat: float) -> int:
        return int(round(float(beat) / self.frame_beats))

    @property
    def strong_prefix(self) -> np.ndarray:
        """Cumulative chroma restricted to downbeats (cached)."""
        if self._strong_prefix is None:
            rows = np.zeros_like(self.chroma)
            step = self.frame_beats
            for index in range(len(self.chroma)):
                beat = index * step
                if beat % self.beats_per_bar < step * 0.5:
                    rows[index] = self.chroma[index]
            self._strong_prefix = np.vstack([np.zeros(12), np.cumsum(rows, axis=0)])
        return self._strong_prefix

    def slice_chroma(self, start_beat: float, end_beat: float) -> np.ndarray:
        """Total chroma between two beat positions (edges snap to frames)."""
        start = max(0, self._index(start_beat))
        end = min(len(self.chroma), self._index(end_beat))
        if end <= start:
            return np.zeros(12, dtype=np.float64)
        return self._prefix[end] - self._prefix[start]

    def bar_chroma(self, bar_index: int) -> np.ndarray:
        start = bar_index * self.beats_per_bar
        return self.slice_chroma(start, start + self.beats_per_bar)

    def global_chroma(self, *, normalize: bool = False) -> np.ndarray:
        total = self._prefix[-1].copy()
        if normalize and total.sum() > 0:
            total /= total.sum()
        return total

    def active_beats(self) -> float:
        """Beats that actually contain a voiced pitch."""
        active = float((self.chroma.sum(axis=1) > 1e-9).sum())
        return active * self.frame_beats

    def key_guess(self) -> Key:
        return estimate_key(self.global_chroma())

    # -- construction ---------------------------------------------------- #
    @classmethod
    def from_notes(
        cls,
        notes: Sequence[NoteEvent],
        *,
        beats_per_bar: float = 4.0,
        frame_beats: float = 0.25,
        total_beats: float | None = None,
        soft: bool = True,
        sigma_semitones: float = 0.5,
        tempo_bpm: float | None = None,
        source: str = "symbolic",
    ) -> "Melody":
        if frame_beats <= 0:
            raise ValueError("frame_beats must be positive")
        end = max((n.end for n in notes), default=0.0)
        if total_beats is not None:
            end = max(end, float(total_beats))
        frames = max(1, int(math.ceil(end / frame_beats)))
        chroma = np.zeros((frames, 12), dtype=np.float64)
        for item in notes:
            if item.duration <= 0:
                continue
            accent = metric_weight(item.onset, beats_per_bar)
            start = max(0.0, item.onset)
            stop = min(frames * frame_beats, item.end)
            if stop <= start:
                continue
            first = int(math.floor(start / frame_beats))
            last = int(math.ceil(stop / frame_beats))
            for index in range(first, last):
                frame_start = index * frame_beats
                frame_end = frame_start + frame_beats
                overlap = min(stop, frame_end) - max(start, frame_start)
                if overlap <= 0:
                    continue
                if soft:
                    targets = _soft_pitch_classes(item.pitch, sigma_semitones)
                    for pc, weight in targets:
                        chroma[index, pc] += overlap * accent * weight
                else:
                    chroma[index, item.pitch % 12] += overlap * accent
        return cls(
            chroma,
            frame_beats=frame_beats,
            beats_per_bar=beats_per_bar,
            tempo_bpm=tempo_bpm,
            total_beats=frames * frame_beats,
            notes=list(notes),
            source=source,
        )

    @classmethod
    def from_chroma_frames(
        cls,
        chroma: np.ndarray,
        *,
        frame_beats: float,
        beats_per_bar: float = 4.0,
        tempo_bpm: float | None = None,
        notes: Sequence[NoteEvent] | None = None,
        tuning_cents: float = 0.0,
        voiced_ratio: float = 1.0,
        source: str = "",
    ) -> "Melody":
        return cls(
            chroma,
            frame_beats=frame_beats,
            beats_per_bar=beats_per_bar,
            tempo_bpm=tempo_bpm,
            notes=notes,
            tuning_cents=tuning_cents,
            voiced_ratio=voiced_ratio,
            source=source,
        )

    # -- display --------------------------------------------------------- #
    def describe(self) -> str:
        total = float(self.global_chroma().sum())
        top = np.argsort(self.global_chroma())[::-1][:4]
        names = ", ".join(pc_name(int(pc)) for pc in top if self.global_chroma()[pc] > 0)
        return (
            f"{self.total_beats:.1f} beats ({self.total_beats / self.beats_per_bar:.1f} bars), "
            f"{self.active_beats():.1f} voiced beats, key guess {self.key_guess().name()}, "
            f"top pitch classes {names}"
            + (f", tuning {self.tuning_cents:+.1f} cents" if self.tuning_cents else "")
        )


def _soft_pitch_classes(midi_pitch: float, sigma_semitones: float) -> list[tuple[int, float]]:
    """Wrap a Gaussian over the twelve pitch classes."""
    sigma = max(float(sigma_semitones), 1e-3)
    values: list[tuple[int, float]] = []
    total = 0.0
    for pc in range(12):
        delta = (float(midi_pitch) - pc) % 12
        delta = min(delta, 12.0 - delta)
        weight = math.exp(-0.5 * (delta / sigma) ** 2)
        if weight > 1e-4:
            values.append((pc, weight))
            total += weight
    if total <= 0:
        return [(int(round(midi_pitch)) % 12, 1.0)]
    return [(pc, weight / total) for pc, weight in values]


# --------------------------------------------------------------------------- #
# Audio front-end
# --------------------------------------------------------------------------- #

@dataclass
class TempoEstimate:
    bpm: float
    beat_offset: float      # seconds of the first beat
    confidence: float       # 0..1, correlation of onsets with the pulse
    source: str = "estimated"


@dataclass
class AudioAnalysis:
    melody: Melody
    tempo: TempoEstimate
    tuning_cents: float
    key_guess: Key
    duration: float
    voiced_ratio: float
    notes: list[NoteEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        return self.tempo.bpm


def estimate_tempo(
    y: np.ndarray,
    sr: int,
    *,
    start_bpm: float = 120.0,
    min_bpm: float = 50.0,
    max_bpm: float = 200.0,
    beat_offset: float | None = None,
) -> TempoEstimate:
    """Tempo and beat phase from the onset envelope.

    A solo vocal has fewer percussive onsets than a full mix, so the estimate
    is best when the singer records against a click (the assumption MySong
    makes).  The caller can always supply ``bpm`` explicitly; the app does.
    """
    import librosa

    hop = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if onset.size == 0 or float(onset.max()) <= 1e-9:
        return TempoEstimate(float(start_bpm), 0.0, 0.0, "fallback")
    try:
        from librosa.feature.rhythm import tempo as tempo_function
    except ImportError:  # pragma: no cover - older librosa
        tempo_function = librosa.beat.tempo
    tempo_values = tempo_function(
        onset_envelope=onset,
        sr=sr,
        hop_length=hop,
        start_bpm=float(start_bpm),
        max_tempo=float(max_bpm),
    )
    bpm = float(np.atleast_1d(tempo_values)[0])
    bpm = float(np.clip(bpm, min_bpm, max_bpm))
    period = 60.0 / bpm
    seconds_per_frame = hop / sr

    if beat_offset is None:
        # Search the phase that puts the strongest onsets on beats.
        times = np.arange(len(onset)) * seconds_per_frame
        best_offset, best_score = 0.0, -1.0
        for candidate in np.arange(0.0, period, 0.01):
            positions = np.arange(candidate, times[-1] if len(times) else 0.0, period)
            indices = np.clip(np.round(positions / seconds_per_frame).astype(int), 0, len(onset) - 1)
            if indices.size == 0:
                continue
            score = float(onset[indices].mean())
            if score > best_score:
                best_offset, best_score = float(candidate), score
    else:
        best_offset = float(beat_offset)
        positions = np.arange(best_offset, (len(onset) - 1) * seconds_per_frame, period)
        indices = np.clip(np.round(positions / seconds_per_frame).astype(int), 0, len(onset) - 1)
        best_score = float(onset[indices].mean()) if indices.size else 0.0

    confidence = float(np.clip(best_score / (float(onset.mean()) + 1e-9) / 4.0, 0.0, 1.0))
    return TempoEstimate(bpm, best_offset, confidence)


def estimate_tuning(f0: np.ndarray, voiced_prob: np.ndarray, *, threshold: float = 0.4) -> float:
    """Global tuning offset in cents.

    Real singers sit slightly above or below equal temperament, and a whole
    recording usually shares that offset.  Removing it before quantising is
    what lets a slightly flat take still produce the right chords.
    """
    valid = np.isfinite(f0) & (f0 > 0) & (voiced_prob >= threshold)
    if valid.sum() < 10:
        return 0.0
    midi = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)
    cents = (midi - np.round(midi)) * 100.0
    hist, edges = np.histogram(cents, bins=np.arange(-50.0, 50.5, 1.0))
    if hist.sum() == 0:
        return 0.0
    # Circular smoothing: the distribution is periodic in 100 cents.
    kernel = np.exp(-0.5 * (np.arange(-12, 13) / 3.0) ** 2)
    smooth = np.convolve(np.concatenate([hist, hist, hist]), kernel, mode="same")[len(hist):2 * len(hist)]
    peak = int(np.argmax(smooth))
    offset = float((edges[peak] + edges[peak + 1]) / 2.0)
    # Refine with the mean of the inliers around the peak.
    near = np.abs(cents - offset) < 25.0
    if near.sum() >= 5:
        offset = float(np.mean(cents[near]))
    return float(np.clip(offset, -50.0, 50.0))


def f0_to_notes(
    f0: np.ndarray,
    voiced_prob: np.ndarray,
    times: np.ndarray,
    *,
    tuning_cents: float = 0.0,
    threshold: float = 0.35,
    min_duration: float = 0.09,
    gap_tolerance: float = 0.06,
) -> list[NoteEvent]:
    """Segment a pitch track into notes (for MIDI export and display).

    The harmonizer does not need these notes -- it works from chroma -- but a
    musician does, and the MIDI export is much more useful with the melody in
    it.
    """
    if len(f0) == 0:
        return []
    midi = np.full(len(f0), np.nan)
    valid = np.isfinite(f0) & (f0 > 0) & (voiced_prob >= threshold)
    midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0) - tuning_cents / 100.0
    rounded = np.full(len(f0), -999, dtype=int)
    rounded[valid] = np.round(midi[valid]).astype(int)

    notes: list[NoteEvent] = []
    current_pitch: int | None = None
    current_start = 0.0
    current_end = 0.0
    pending_pitch: int | None = None
    pending_count = 0

    def flush() -> None:
        nonlocal current_pitch, current_start, current_end
        if current_pitch is not None and current_end - current_start >= min_duration:
            velocity = 80
            notes.append(NoteEvent(current_start, current_end - current_start, current_pitch, velocity))
        current_pitch = None

    for index, pitch in enumerate(rounded):
        time = float(times[index])
        if pitch < 0:
            if current_pitch is not None and time - current_end > gap_tolerance:
                flush()
            continue
        if current_pitch is None:
            current_pitch = int(pitch)
            current_start = time
            current_end = time + (times[1] - times[0] if len(times) > 1 else 0.01)
            pending_pitch = None
            pending_count = 0
            continue
        if int(pitch) == current_pitch:
            current_end = time + (times[1] - times[0] if len(times) > 1 else 0.01)
            pending_pitch = None
            pending_count = 0
            continue
        # A new pitch must persist before it interrupts the current note;
        # this is what removes one-frame octave errors.
        if pending_pitch == int(pitch):
            pending_count += 1
        else:
            pending_pitch = int(pitch)
            pending_count = 1
        if pending_count >= 3:
            flush()
            current_pitch = int(pitch)
            current_start = time - (pending_count - 1) * (times[1] - times[0])
            current_end = time + (times[1] - times[0])
            pending_pitch = None
            pending_count = 0
    flush()
    return notes


def analyze_audio(
    audio: np.ndarray | str | Path,
    *,
    sr: int | None = None,
    bpm: float | None = None,
    beat_offset: float | None = None,
    beats_per_bar: float = 4.0,
    frame_beats: float = 0.25,
    fmin: float = 70.0,
    fmax: float = 1200.0,
    tuning: float | None = None,
    backend: str = "pyin",
) -> AudioAnalysis:
    """Analyse a sung take into a :class:`Melody`.

    ``bpm`` and ``beat_offset`` should be supplied when the take was recorded
    against a click (the app records one); otherwise they are estimated.  The
    take is band-limited before pitch tracking so a little room rumble or a
    low accompaniment does not drag the tracker down an octave.
    """
    import librosa
    from scipy.signal import butter, sosfiltfilt

    if isinstance(audio, (str, Path)):
        y, native_sr = librosa.load(str(audio), sr=sr, mono=True)
        sr = native_sr
    else:
        y = np.asarray(audio, dtype=np.float32)
        sr = int(sr or 22050)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        raise ValueError("empty audio")
    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Remove DC and rumble; keep the vocal band.
    sos = butter(4, [max(50.0, fmin * 0.7), min(4000.0, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, y).astype(np.float32)

    if bpm is None:
        tempo = estimate_tempo(
            filtered, sr, start_bpm=120.0, beat_offset=beat_offset
        )
        bpm = tempo.bpm
        beat_offset = tempo.beat_offset if beat_offset is None else beat_offset
    elif beat_offset is None:
        # A supplied tempo means the take was recorded against a click that
        # started with the recording, which is the MySong assumption.  Trying
        # to estimate the phase from a solo voice is less reliable than that
        # assumption and can shift the whole band by a beat.
        beat_offset = 0.0
        tempo = TempoEstimate(float(bpm), 0.0, 1.0, "given")
    else:
        tempo = TempoEstimate(float(bpm), float(beat_offset), 1.0, "given")
    bpm = float(bpm)
    beat_offset = float(beat_offset or 0.0)

    hop = 256
    if backend == "rmvpe":
        try:
            from analysis.rmvpe_detector import RMVPEModel

            model = RMVPEModel("assets/rmvpe/rmvpe.pt", device="cpu")
            f0_hops, conf = model.infer(filtered, voicing_threshold=0.03)
            f0 = np.asarray(f0_hops, dtype=np.float64)
            voiced_prob = np.asarray(conf, dtype=np.float64)
            hop = 160
            times = np.arange(len(f0)) * hop / sr
        except Exception:
            backend = "pyin"
    if backend != "rmvpe":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            filtered,
            fmin=float(fmin),
            fmax=float(fmax),
            sr=sr,
            frame_length=2048,
            hop_length=hop,
            fill_na=np.nan,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        voiced_prob = np.nan_to_num(np.asarray(voiced_prob, dtype=np.float64))

    f0 = np.asarray(f0, dtype=np.float64)
    if tuning is None:
        tuning = estimate_tuning(f0, voiced_prob)
    tuning = float(tuning)

    rms = librosa.feature.rms(y=filtered, frame_length=2048, hop_length=hop, center=True)[0]
    rms = np.resize(rms, len(f0)) if len(rms) != len(f0) else rms
    if rms.max() > 0:
        rms = rms / (np.median(rms[rms > 0]) + 1e-9)
    energy = np.clip(rms, 0.25, 1.6)

    voiced = np.isfinite(f0) & (f0 > 0) & (voiced_prob > 0.3)
    chroma_frames = np.zeros((len(f0), 12), dtype=np.float64)
    for index in np.where(voiced)[0]:
        midi = 69.0 + 12.0 * math.log2(f0[index] / 440.0) - tuning / 100.0
        weight = float(voiced_prob[index]) * float(energy[index])
        for pc, share in _soft_pitch_classes(midi, 0.6):
            chroma_frames[index, pc] += weight * share

    # Aggregate onto the beat frame grid.
    seconds_per_beat = 60.0 / bpm
    seconds_per_frame = seconds_per_beat * frame_beats
    duration = len(y) / sr
    n_out = max(1, int(math.ceil(max(duration - beat_offset, 0.0) / seconds_per_frame)))
    out = np.zeros((n_out, 12), dtype=np.float64)
    frame_index = np.floor((times - beat_offset) / seconds_per_frame).astype(int)
    good = voiced & (frame_index >= 0) & (frame_index < n_out)
    for index in np.where(good)[0]:
        out[frame_index[index]] += chroma_frames[index]

    # Metrical accent, applied once so the decoder can just sum frames.
    for index in range(n_out):
        beat_position = index * frame_beats
        out[index] *= metric_weight(beat_position, beats_per_bar)

    melody = Melody.from_chroma_frames(
        out,
        frame_beats=frame_beats,
        beats_per_bar=beats_per_bar,
        tempo_bpm=bpm,
        notes=[
            NoteEvent(
                onset=(note.onset - beat_offset) / seconds_per_beat,
                duration=note.duration / seconds_per_beat,
                pitch=note.pitch,
                velocity=note.velocity,
            )
            for note in f0_to_notes(f0, voiced_prob, times, tuning_cents=tuning)
        ],
        tuning_cents=tuning,
        voiced_ratio=float(voiced.mean()) if len(voiced) else 0.0,
        source=str(audio) if isinstance(audio, (str, Path)) else "array",
    )
    key = melody.key_guess()
    warnings: list[str] = []
    if melody.voiced_ratio < 0.15:
        warnings.append("very little voiced audio; check the recording level and microphone")
    if tempo.confidence < 0.15 and tempo.source != "given":
        warnings.append("tempo/beat phase is uncertain; pass bpm explicitly for best results")
    return AudioAnalysis(
        melody=melody,
        tempo=tempo,
        tuning_cents=tuning,
        key_guess=key,
        duration=duration,
        voiced_ratio=melody.voiced_ratio,
        notes=melody.notes,
        warnings=warnings,
    )

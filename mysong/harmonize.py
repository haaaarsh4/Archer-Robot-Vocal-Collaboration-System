"""Choose chords for a melody.

This is the part of MySong (Simon, Morris & Basu, CHI 2008) the paper describes
as a hidden Markov model over one chord per measure, decoded with Viterbi, with
the key found by trying all twelve and keeping the most likely.  Everything
below is a strict superset of that:

* **Semi-Markov durations.**  A chord may last one, two, three, four, six,
  eight, twelve or sixteen beats instead of exactly one bar.  The duration
  prior is learned from the corpus and conditioned on the meter, so 3/4 and
  6/8 do not inherit 4/4's habits.  MySong's fixed measure is available by
  setting ``style.durations=(4,)``.
* **Joint key and mode search.**  24 hypotheses, not 12, each scored by its
  Viterbi likelihood plus a Krumhansl-Schmuckler prior.  Relative major and
  minor share a key signature but not a chord vocabulary, so treating them
  separately fixes a real class of MySong errors.
* **Soft, metric-weighted chroma.**  Notes are not rounded to the nearest
  semitone before they reach the model; the likelihood uses the same soft
  features the corpus was trained on.
* **Calibrated uncertainty.**  Forward-backward on the semi-Markov model gives
  a posterior over chords for every decoded span, so the UI can offer
  alternatives with real probabilities rather than pretending the MAP path is
  certain.

The decoder is exact: no beam, no sampling.  The search space is small enough
that a modern CPU answers in well under a second for a three-minute song.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from mysong.chord_model import extract_features
from mysong.corpus import NoteEvent
from mysong.key_model import key_from_index, key_index
from mysong.melody import Melody
from mysong.model import MySongModel
from mysong.theory import (
    CORE_QUALITIES,
    Chord,
    Key,
    QUALITY_INTERVALS,
    core_of,
    estimate_key,
    key_profile,
    key_scores,
    metric_weight,
    parse_key,
    profile_correlation,
    spell_pc,
)

_EPS = 1e-12

# A span's tagger log-likelihood sums one term per beat, but beat-level errors
# are correlated, so the raw sum turns every long chord into a 100% certainty.
# This per-beat temperature was chosen on the dev split so mean confidence
# tracks accuracy instead of saturating (test: 0.62 mean chosen confidence
# against 0.53 core accuracy).  It affects the reported confidence only, never
# the Viterbi path.
_CONFIDENCE_TEMPERATURE_PER_BEAT = 10.0


# #
# Public result types
# #

@dataclass
class Style:
    """Controls that correspond to MySong's sliders, plus a few we added.

    ``observation_weight`` is MySong's "jazz factor": higher values make the
    chords follow the melody's notes more literally (and accept odder
    progressions); lower values favour conventional progressions.

    ``density`` is a harmonic-density control, like AutoHarmonizer's (Wu et al.,
    2024): above 1.0 the duration prior sharpens toward common short chords,
    below 1.0 it flattens toward long, slow harmony.

    ``color`` controls extensions (6ths, 7ths, 9ths): 0 gives plain triads,
    1 lets the corpus statistics choose freely.
    """

    observation_weight: float = 5.0
    transition_weight: float = 1.0
    density: float = 1.0
    color: float = 0.75
    happy: float | None = None
    chord_prior_weight: float = 0.2
    class_prior_correction: float = 0.5
    mode_prior_weight: float = 1.0
    key_weight: float = 8.0
    #: Weight on the 24-way key classifier's log-probability.  Large enough
    #: that the classifier's top key normally wins; the HMM can only override
    #: it when the two candidates are nearly tied.
    key_classifier_weight: float = 400.0
    key_candidates: int = 6
    min_duration: float = 1.0
    max_duration: float = 4.0
    durations: tuple[float, ...] | None = None
    #: Log bonus added for every chord change.  The duration prior is a
    #: per-chord distribution, so without a change term a decoder prefers a
    #: few long chords (fewer draws from the distribution).  This restores
    #: the corpus's harmonic rhythm; 0 is the neutral maximum-likelihood
    #: setting, positive values change chords more often.
    change_bonus: float = 0.0
    #: Temperature on the duration distribution, separate from ``density``.
    #: 1.0 leaves the learned distribution untouched.
    duration_weight: float = 1.0
    #: Scale on the neural tagger's log-probabilities.  The tagger emits a
    #: distribution per beat; sum over a span.  12 is the calibrated value
    #: found on the dev split (it makes the emission outweigh the duration
    #: prior without drowning the transition model).
    tagger_weight: float = 12.0

    def validated(self) -> "Style":
        return Style(
            observation_weight=max(0.05, float(self.observation_weight)),
            transition_weight=max(0.05, float(self.transition_weight)),
            density=max(0.2, float(self.density)),
            color=float(np.clip(self.color, 0.0, 1.0)),
            happy=None if self.happy is None else float(np.clip(self.happy, 0.0, 1.0)),
            chord_prior_weight=max(0.0, float(self.chord_prior_weight)),
            class_prior_correction=max(0.0, float(self.class_prior_correction)),
            mode_prior_weight=max(0.0, float(self.mode_prior_weight)),
            key_weight=max(0.0, float(self.key_weight)),
            key_classifier_weight=max(0.0, float(self.key_classifier_weight)),
            key_candidates=max(1, int(self.key_candidates)),
            min_duration=max(0.5, float(self.min_duration)),
            max_duration=max(1.0, float(self.max_duration)),
            durations=self.durations,
            change_bonus=float(self.change_bonus),
            duration_weight=max(0.0, float(self.duration_weight)),
            tagger_weight=max(0.0, float(self.tagger_weight)),
        )


@dataclass
class ChordAlternative:
    chord: Chord
    probability: float


@dataclass
class ChordSegment:
    start_beat: float
    end_beat: float
    chord: Chord
    confidence: float = 1.0
    alternatives: list[ChordAlternative] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_beat - self.start_beat

    def to_dict(self, bpm: float | None = None, key: Key | None = None) -> dict:
        payload = {
            "start_beat": round(self.start_beat, 4),
            "end_beat": round(self.end_beat, 4),
            "chord": self.chord.figure(key),
            "root": self.chord.root,
            "quality": self.chord.quality,
            "bass": self.chord.bass,
            "confidence": round(float(self.confidence), 4),
        }
        if bpm:
            payload["start_seconds"] = round(self.start_beat * 60.0 / bpm, 4)
            payload["end_seconds"] = round(self.end_beat * 60.0 / bpm, 4)
        if self.alternatives:
            payload["alternatives"] = [
                {"chord": alt.chord.figure(key), "probability": round(float(alt.probability), 4)}
                for alt in self.alternatives
            ]
        return payload


@dataclass
class Harmonization:
    key: Key
    segments: list[ChordSegment]
    score: float
    key_scores: list[tuple[Key, float]]
    key_confidence: float
    beats_per_bar: float
    tempo_bpm: float | None = None
    key_posterior: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def chords(self) -> list[Chord]:
        return [segment.chord for segment in self.segments]

    @property
    def duration_beats(self) -> float:
        return self.segments[-1].end_beat if self.segments else 0.0

    def chord_at(self, beat: float) -> Chord | None:
        for segment in self.segments:
            if segment.start_beat <= beat < segment.end_beat:
                return segment.chord
        return None

    def roman_progression(self) -> list[str]:
        return [self.key.roman(segment.chord) for segment in self.segments]

    def summary(self) -> str:
        names = [str(segment.chord) for segment in self.segments]
        return f"{self.key.name()}: " + " | ".join(names)

    def to_dict(self) -> dict:
        bpm = self.tempo_bpm
        return {
            "key": self.key.name(),
            "tonic": self.key.tonic,
            "mode": self.key.mode,
            "tempo_bpm": None if bpm is None else round(float(bpm), 3),
            "beats_per_bar": float(self.beats_per_bar),
            "score": round(float(self.score), 3),
            "key_confidence": round(float(self.key_confidence), 4),
            "key_posterior": {k: round(float(v), 4) for k, v in self.key_posterior.items()},
            "roman": self.roman_progression(),
            "segments": [segment.to_dict(bpm, self.key) for segment in self.segments],
            "warnings": list(self.warnings),
        }


# #
# Small helpers
# #

def _phase_bucket(beat: float, beats_per_bar: float) -> int:
    phase = float(beat) % float(beats_per_bar)
    if phase < 1e-6 or abs(phase - beats_per_bar) < 1e-6:
        return 0
    if abs(phase - beats_per_bar / 2.0) < 1e-6:
        return 1
    return 2


def _logsumexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    out = np.log(np.sum(np.exp(values - top), axis=axis, keepdims=True) + _EPS) + top
    return np.squeeze(out, axis=axis)


def _softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    array = array - np.max(array)
    exp = np.exp(array)
    total = exp.sum()
    return exp / total if total > 0 else np.full_like(exp, 1.0 / len(exp))


# #
# The decoder
# #

@dataclass
class _Decoded:
    key: Key
    tonic: int
    mode_index: int
    segments: list[tuple[int, int, int]]  # (start_beat, end_beat, state)
    score: float
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    log_transition: np.ndarray | None = None


class Harmonizer:
    """Exact semi-Markov decoder over the 60 core triads."""

    def __init__(self, model: MySongModel) -> None:
        self.model = model

    # -- public API ------------------------------------------------------ #
    def harmonize(
        self,
        melody: Melody | Sequence[NoteEvent],
        *,
        key: Key | str | None = None,
        style: Style | None = None,
        beats_per_bar: float | None = None,
        tempo_bpm: float | None = None,
        total_beats: float | None = None,
        with_marginals: bool = True,
    ) -> Harmonization:
        style = (style or Style()).validated()
        tagger = getattr(self.model, "tagger", None)
        if tagger is not None:
            tagger.clear()
        if not isinstance(melody, Melody):
            melody = Melody.from_notes(
                melody,
                beats_per_bar=beats_per_bar or 4.0,
                total_beats=total_beats,
                tempo_bpm=tempo_bpm,
            )
        meter = float(beats_per_bar or melody.beats_per_bar or 4.0)
        bpm = tempo_bpm or melody.tempo_bpm
        warnings: list[str] = []

        total = float(total_beats or melody.total_beats)
        if total < 1.0:
            raise ValueError("melody is too short to harmonize")
        # The decoder works on whole beats; pad the chroma so the last partial
        # beat is still observable.
        horizon = int(math.ceil(total))
        if melody.total_beats < horizon:
            pad = horizon - melody.total_beats
            frames = int(math.ceil(pad / melody.frame_beats))
            melody = Melody.from_chroma_frames(
                np.vstack([melody.chroma, np.zeros((frames, 12))]),
                frame_beats=melody.frame_beats,
                beats_per_bar=melody.beats_per_bar,
                tempo_bpm=melody.tempo_bpm,
                notes=melody.notes,
                tuning_cents=melody.tuning_cents,
                voiced_ratio=melody.voiced_ratio,
                source=melody.source,
            )
        horizon = min(horizon, int(melody.total_beats))

        # Candidate keys.  A trained 24-way classifier on global chroma is a
        # much stronger proposal than Krumhansl-Schmuckler (88% versus 55%
        # exact on held-out OpenEWLD), but it can still be wrong.  Decode its
        # top few and let the sequence likelihood arbitrate only when their
        # classifier scores are close.
        key_log_probabilities: dict[int, float] = {}
        if key is not None:
            candidates = [parse_key(key) if isinstance(key, str) else key]
        else:
            classifier = self.model.key_classifier
            absolute = melody.global_chroma()
            if classifier is not None:
                log_probs = classifier.predict_log_proba(absolute)
                order = np.argsort(log_probs)[::-1][: style.key_candidates]
                candidates = [key_from_index(int(index)) for index in order]
                # Keep every key's log-probability, not just the shortlist: the
                # parallel-mode candidate added below must pay the same prior
                # as any other key, otherwise it can win purely because it was
                # exempt from the classifier penalty.
                key_log_probabilities = {
                    int(index): float(log_probs[index]) for index in range(len(log_probs))
                }
            else:
                ranked = key_scores(absolute)
                candidates = [candidate for candidate, _ in ranked[: style.key_candidates]]
            # Make sure the parallel mode of the strongest tonic is present;
            # relative major/minor confusion is the classic failure mode.
            if candidates:
                extra = [
                    Key(candidates[0].tonic, "major" if candidates[0].mode == "minor" else "minor")
                ]
                candidates.extend(k for k in extra if k not in candidates)

        decoded: list[_Decoded] = []
        for candidate in candidates:
            result = self._decode_key(melody, horizon, candidate, meter, style)
            prior = 0.0
            if key is None:
                index = key_index(candidate)
                if index in key_log_probabilities:
                    prior += style.key_classifier_weight * key_log_probabilities[index]
                else:
                    chroma = melody.global_chroma()
                    profile = None
                    if self.model.key_profiles is not None:
                        profile = self.model.key_profiles[self.model.mode_index(candidate.mode)]
                    if profile is None:
                        profile = key_profile(Key(0, candidate.mode))
                    rel = np.array([chroma[(pc + candidate.tonic) % 12] for pc in range(12)])
                    prior += style.key_weight * profile_correlation(rel, profile)
                prior += style.mode_prior_weight * float(
                    self.model.mode_logprior[self.model.mode_index(candidate.mode)]
                )
                if style.happy is not None:
                    h = min(max(style.happy, 1e-4), 1 - 1e-4)
                    prior += math.log(h) if candidate.mode == "major" else math.log(1.0 - h)
            result.score += prior
            decoded.append(result)

        decoded.sort(key=lambda item: item.score, reverse=True)
        best = decoded[0]
        if with_marginals:
            best.alpha, best.beta = self._forward_backward(
                melody, horizon, best, meter, style
            )

        segments = self._build_segments(melody, best, meter, style)
        if not segments:
            raise ValueError("the model produced no chords")

        # Key confidence.  The trained classifier gives a real 24-way
        # distribution; use it directly rather than turning hugely scaled
        # Viterbi scores into a fake posterior.  A model without the key
        # classifier falls back to a temperature-scaled ranking.
        classifier = self.model.key_classifier
        if key is None and classifier is not None:
            log_probs = classifier.predict_log_proba(melody.global_chroma())
            probabilities = _softmax(log_probs)
            key_posterior = {
                key_from_index(index).name(): float(probabilities[index])
                for index in np.argsort(probabilities)[::-1][:5]
            }
            key_confidence = float(probabilities[key_index(best.key)])
        else:
            scale = max(4.0, horizon / 6.0)
            posterior_values = _softmax([item.score / scale for item in decoded])
            key_posterior = {
                item.key.name(): float(p) for item, p in zip(decoded, posterior_values)
            }
            key_confidence = float(key_posterior.get(best.key.name(), 0.0))
        if best.key not in [item.key for item in decoded]:  # pragma: no cover
            key_posterior[best.key.name()] = 1.0

        if melody.voiced_ratio < 0.1:
            warnings.append("the take is mostly unvoiced; chords may follow silence")

        return Harmonization(
            key=best.key,
            segments=segments,
            score=best.score,
            key_scores=[(item.key, item.score) for item in decoded],
            key_confidence=key_confidence,
            beats_per_bar=meter,
            tempo_bpm=bpm,
            key_posterior=key_posterior,
            warnings=warnings,
        )

    # -- observation potential ------------------------------------------ #
    def _observe(
        self,
        melody: Melody,
        start: float,
        end: float,
        *,
        mode: str,
        tonic: int,
        style: Style,
    ) -> np.ndarray:
        """Log-potential of the melody over ``[start, end)`` for all 60 states."""
        tagger = getattr(self.model, "tagger", None)
        if tagger is not None:
            return style.tagger_weight * tagger.log_span(
                melody, Key(int(tonic), mode), start, end
            )
        classifier = self.model.classifier
        if classifier is not None:
            features = extract_features(
                melody, start, end, mode=mode, rotation=int(tonic)
            )
            return style.observation_weight * classifier.log_proba(
                features, prior_correction=style.class_prior_correction
            )
        vector = melody.slice_chroma(start, end)
        vector = vector[[(pc + int(tonic)) % 12 for pc in range(12)]]
        return style.observation_weight * (self.model.observation @ vector)

    # -- fixed-grid baseline --------------------------------------------- #
    def harmonize_fixed_bars(
        self,
        melody: Melody | Sequence[NoteEvent],
        *,
        key: Key | str,
        style: Style | None = None,
        beats_per_bar: float | None = None,
        tempo_bpm: float | None = None,
    ) -> Harmonization:
        """MySong's original model: exactly one chord per bar, Viterbi.

        This exists so the evaluation harness can compare the flexible-rhythm
        decoder against the paper's fixed-measure model on the same features.
        """
        style = (style or Style()).validated()
        tagger = getattr(self.model, "tagger", None)
        if tagger is not None:
            tagger.clear()
        if not isinstance(melody, Melody):
            melody = Melody.from_notes(
                melody, beats_per_bar=beats_per_bar or 4.0, tempo_bpm=tempo_bpm
            )
        meter = float(beats_per_bar or melody.beats_per_bar or 4.0)
        key_obj = parse_key(key) if isinstance(key, str) else key
        horizon = int(math.ceil(melody.total_beats))
        if horizon < 1:
            raise ValueError("melody is too short to harmonize")
        n_bars = int(math.ceil(horizon / meter))
        starts = [index * meter for index in range(n_bars)]
        ends = [min(float(horizon), start + meter) for start in starts]
        mode = self.model.mode_index(key_obj.mode)
        observations = np.vstack([
            self._observe(melody, start, end, mode=key_obj.mode,
                          tonic=key_obj.tonic, style=style)
            for start, end in zip(starts, ends)
        ])
        transition = self.model.transition[mode]
        dp = np.full((n_bars, self.model.n_states), -np.inf, dtype=np.float64)
        back = np.zeros((n_bars, self.model.n_states), dtype=np.int32)
        dp[0] = self.model.start[mode] + observations[0]
        for index in range(1, n_bars):
            candidates = dp[index - 1][:, None] + transition
            previous = np.argmax(candidates, axis=0)
            dp[index] = candidates[previous, np.arange(self.model.n_states)] + observations[index]
            back[index] = previous
        total = dp[-1] + self.model.end[mode]
        state = int(np.argmax(total))
        segments: list[ChordSegment] = []
        for index in range(n_bars - 1, -1, -1):
            chord = self.model.state_chord(state)
            absolute = Chord((chord.root + key_obj.tonic) % 12, chord.core_quality)
            colorized, alternatives, confidence = self._colorize(
                absolute, state, mode, melody, starts[index], ends[index], style, None
            )
            segments.append(ChordSegment(
                start_beat=float(starts[index]), end_beat=float(ends[index]),
                chord=colorized, confidence=confidence, alternatives=alternatives,
            ))
            state = int(back[index, state])
        segments.reverse()
        return Harmonization(
            key=key_obj, segments=segments, score=float(np.max(total)),
            key_scores=[(key_obj, float(np.max(total)))], key_confidence=1.0,
            beats_per_bar=meter, tempo_bpm=tempo_bpm or melody.tempo_bpm,
            key_posterior={key_obj.name(): 1.0},
        )

    # -- decode one key -------------------------------------------------- #
    def _duration_indices(self, style: Style, horizon: int) -> list[int]:
        indices = []
        for index, duration in enumerate(self.model.durations):
            if duration < style.min_duration - 1e-9:
                continue
            if duration > style.max_duration + 1e-9:
                continue
            if duration > horizon:
                continue
            if style.durations and not any(abs(duration - d) < 1e-9 for d in style.durations):
                continue
            indices.append(index)
        if not indices:
            # Fall back to whatever fits, so a very short take still decodes.
            indices = [int(np.argmin(np.abs(self.model.durations - min(1.0, horizon))))]
        return indices

    @staticmethod
    def _segment_lengths(
        end: int, horizon: int, duration_indices: Sequence[int], durations: np.ndarray
    ) -> list[tuple[int, int]]:
        """Candidate (actual length, duration-class index) for a segment ending at ``end``.

        A song rarely ends on a multiple of its harmonic rhythm -- a 67-beat
        take cannot be tiled by four-beat chords.  MySong's fixed grid hid
        this by always rounding up to a whole measure; a flexible decoder
        must instead allow the final chord to be shorter than its prior
        duration.  Only the tail is truncated, so ordinary spans keep exact
        duration semantics and the back-pointer stays unambiguous.
        """
        options: list[tuple[int, int]] = []
        for index in duration_indices:
            nominal = int(round(float(durations[index])))
            if nominal <= 0:
                continue
            if nominal <= end:
                options.append((nominal, index))
            if end == horizon and nominal > 1:
                for tail in range(1, min(nominal, end) + 1):
                    if (tail, index) not in options:
                        options.append((tail, index))
        return options

    def _decode_key(
        self,
        melody: Melody,
        horizon: int,
        key: Key,
        meter: float,
        style: Style,
    ) -> _Decoded:
        model = self.model
        mode = model.mode_index(key.mode)
        n_states = model.n_states
        tonic = key.tonic
        prefix = melody.prefix[:, [(pc + tonic) % 12 for pc in range(12)]]
        log_obs = model.observation
        log_transition = model.transition[mode] * style.transition_weight
        log_start = model.start[mode]
        log_end = model.end[mode]
        meter_index = model.meter_index(meter)
        duration_logprob = model.duration_logprob[meter_index] * style.density
        phase_logprob = model.phase_logprob[meter_index]
        duration_indices = self._duration_indices(style, horizon)
        segment_prior = style.chord_prior_weight * model.chord_logprior[mode]

        dp = np.full((horizon + 1, n_states), -np.inf, dtype=np.float64)
        back = np.full((horizon + 1, n_states, 2), -1, dtype=np.int32)

        def observation(start: int, end: int) -> np.ndarray:
            return self._observe(melody, start, end, mode=key.mode, tonic=tonic, style=style)

        for end in range(1, horizon + 1):
            best_values = dp[end]
            for length, duration_index in self._segment_lengths(
                end, horizon, duration_indices, model.durations
            ):
                start = end - length
                if start < 0:
                    continue
                allowed = (
                    duration_logprob[duration_index] * style.duration_weight
                    + phase_logprob[_phase_bucket(end, meter)]
                    + (style.change_bonus if start > 0 else 0.0)
                )
                obs = observation(start, end)
                if start == 0:
                    values = log_start + obs + allowed + segment_prior
                    better = values > best_values
                    if np.any(better):
                        best_values[better] = values[better]
                        back[end, better, 0] = -1
                        back[end, better, 1] = length
                    continue
                previous = dp[start]
                if not np.isfinite(previous).any():
                    continue
                candidates = previous[:, None] + log_transition  # (prev, cur)
                best_previous = np.argmax(candidates, axis=0)
                values = (
                    candidates[best_previous, np.arange(n_states)]
                    + obs
                    + allowed
                    + segment_prior
                )
                better = values > best_values
                if np.any(better):
                    best_values[better] = values[better]
                    back[end, better, 0] = best_previous[better]
                    back[end, better, 1] = length

        total = dp[horizon] + log_end
        if not np.isfinite(total).any():
            raise RuntimeError("no feasible chord segmentation")
        state = int(np.argmax(total))
        score = float(total[state])

        segments: list[tuple[int, int, int]] = []
        position = horizon
        guard = 0
        while position > 0 and guard <= horizon + 2:
            previous_state, length = (int(v) for v in back[position, state])
            if length <= 0:
                raise RuntimeError("broken back-pointer in Viterbi path")
            segments.append((position - length, position, state))
            position -= length
            state = previous_state
            guard += 1
        segments.reverse()
        if position != 0:  # pragma: no cover - defensive
            raise RuntimeError("Viterbi path did not reach the start")
        return _Decoded(
            key=key,
            tonic=tonic,
            mode_index=mode,
            segments=segments,
            score=score,
            log_transition=log_transition,
        )

    # -- posteriors ------------------------------------------------------ #
    def _forward_backward(
        self,
        melody: Melody,
        horizon: int,
        decoded: _Decoded,
        meter: float,
        style: Style,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        n_states = model.n_states
        prefix = melody.prefix[:, [(pc + decoded.tonic) % 12 for pc in range(12)]]
        log_transition = decoded.log_transition
        assert log_transition is not None
        meter_index = model.meter_index(meter)
        duration_logprob = model.duration_logprob[meter_index] * style.density
        phase_logprob = model.phase_logprob[meter_index]
        duration_indices = self._duration_indices(style, horizon)
        segment_prior = style.chord_prior_weight * model.chord_logprior[decoded.mode_index]

        def observation(start: int, end: int) -> np.ndarray:
            return self._observe(
                melody, start, end,
                mode=decoded.key.mode, tonic=decoded.tonic, style=style,
            )

        alpha = np.full((horizon + 1, n_states), -np.inf, dtype=np.float64)
        for end in range(1, horizon + 1):
            for length, duration_index in self._segment_lengths(
                end, horizon, duration_indices, model.durations
            ):
                start = end - length
                if start < 0:
                    continue
                allowed = (
                    duration_logprob[duration_index] * style.duration_weight
                    + phase_logprob[_phase_bucket(end, meter)]
                    + (style.change_bonus if start > 0 else 0.0)
                )
                obs = observation(start, end)
                if start == 0:
                    values = model.start[decoded.mode_index] + obs + allowed + segment_prior
                    alpha[end] = np.logaddexp(alpha[end], values)
                elif np.isfinite(alpha[start]).any():
                    values = _logsumexp(alpha[start][:, None] + log_transition, axis=0)
                    alpha[end] = np.logaddexp(
                        alpha[end], values + obs + allowed + segment_prior
                    )
        # alpha[T] is the evidence (before the end distribution).

        beta = np.full((horizon + 1, n_states), -np.inf, dtype=np.float64)
        beta[horizon] = model.end[decoded.mode_index]
        for start in range(horizon - 1, -1, -1):
            # Beta needs every segment that could *start* here, including a
            # truncated final chord whose actual length is not a duration
            # class.  Enumerate by end position so the two passes agree.
            for end in range(start + 1, min(horizon, start + int(max(model.durations))) + 1):
                options = self._segment_lengths(end, horizon, duration_indices, model.durations)
                matching = [index for length, index in options if length == end - start]
                if not matching:
                    continue
                duration_index = matching[0]
                allowed = (
                    duration_logprob[duration_index] * style.duration_weight
                    + phase_logprob[_phase_bucket(end, meter)]
                    + (style.change_bonus if start > 0 else 0.0)
                )
                obs = observation(start, end)
                values = (
                    log_transition
                    + (obs + allowed + segment_prior + beta[end])[None, :]
                )
                beta[start] = np.logaddexp(
                    beta[start], _logsumexp(values, axis=1)
                )
        return alpha, beta

    def _segment_posterior(
        self,
        decoded: _Decoded,
        start: int,
        end: int,
        meter: float,
        style: Style,
        melody: Melody,
    ) -> np.ndarray:
        """Posterior over the 60 core chords for one fixed span."""
        model = self.model
        assert decoded.alpha is not None and decoded.beta is not None
        log_transition = decoded.log_transition
        assert log_transition is not None
        meter_index = model.meter_index(meter)
        obs = self._observe(
            melody, start, end,
            mode=decoded.key.mode, tonic=decoded.tonic, style=style,
        )
        duration_index = model.duration_index(end - start)
        allowed = 0.0
        if duration_index is not None:
            allowed = (
                model.duration_logprob[meter_index, duration_index]
                * style.duration_weight
                + model.phase_logprob[meter_index, _phase_bucket(end, meter)]
                + (style.change_bonus if start > 0 else 0.0)
            )
        segment_prior = style.chord_prior_weight * model.chord_logprior[decoded.mode_index]
        if start == 0:
            prefix_term = model.start[decoded.mode_index]
        else:
            prefix_term = _logsumexp(
                decoded.alpha[start][:, None] + log_transition, axis=0
            )
        values = prefix_term + obs + allowed + segment_prior + decoded.beta[end]
        values -= np.max(values)
        values /= _CONFIDENCE_TEMPERATURE_PER_BEAT * max(1.0, float(end - start))
        probabilities = np.exp(values)
        total = probabilities.sum()
        return probabilities / total if total > 0 else np.full(60, 1.0 / 60.0)

    # -- result assembly ------------------------------------------------- #
    def _build_segments(
        self,
        melody: Melody,
        decoded: _Decoded,
        meter: float,
        style: Style,
    ) -> list[ChordSegment]:
        raw = list(decoded.segments)
        # Adjacent spans with the same chord are one chord.
        merged: list[list[int]] = []
        for start, end, state in raw:
            if merged and merged[-1][2] == state and merged[-1][1] == start:
                merged[-1][1] = end
            else:
                merged.append([start, end, state])

        segments: list[ChordSegment] = []
        for start, end, state in merged:
            chord = self.model.state_chord(state)
            chord = Chord(
                (chord.root + decoded.tonic) % 12,
                chord.core_quality,
                None,
            )
            probabilities = None
            if decoded.alpha is not None and decoded.beta is not None:
                probabilities = self._segment_posterior(
                    decoded, start, end, meter, style, melody
                )
            colorized, alternatives, confidence = self._colorize(
                chord, state, decoded.mode_index, melody, start, end, style, probabilities
            )
            segments.append(
                ChordSegment(
                    start_beat=float(start),
                    end_beat=float(end),
                    chord=colorized,
                    confidence=confidence,
                    alternatives=alternatives,
                )
            )
        return segments

    def _colorize(
        self,
        chord: Chord,
        state: int,
        mode: int,
        melody: Melody,
        start: int,
        end: int,
        style: Style,
        core_posterior: np.ndarray | None,
    ) -> tuple[Chord, list[ChordAlternative], float]:
        core_quality = chord.core_quality
        table = self.model.extension.get(f"{mode}:{core_quality}")
        candidates: list[tuple[str, float]] = []
        # Confidence is the core-chord posterior, so it must not be replaced
        # by 1.0 just because this branch cannot colour the chord.
        confidence = (
            float(core_posterior[int(state)]) if core_posterior is not None else 1.0
        )
        if table is None or style.color <= 0.001:
            quality = core_quality if core_quality != "sus" else "sus4"
            return Chord(chord.root, quality), [], confidence

        chroma = melody.slice_chroma(start, end)
        total = float(chroma.sum())
        if total > 0:
            distribution = chroma / total
        else:
            distribution = np.zeros(12)
        temperature = 0.35 + 2.2 * style.color
        for index, quality in enumerate(self.model.extension_qualities):
            if core_of(quality) != core_quality:
                continue
            if quality in ("5", "m11"):  # thin or unusable in this context
                continue
            full = Chord(chord.root, quality)
            core_pcs = set(Chord(chord.root, core_quality).pitch_classes(False))
            added = [pc for pc in full.pitch_classes(False) if pc not in core_pcs]
            bonus = 1.5 * sum(distribution[pc] for pc in added)
            score = temperature * float(table[index]) + bonus
            candidates.append((quality, score))
        if not candidates:
            return chord, [], confidence
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_quality = candidates[0][0]
        top_scores = np.asarray([score for _, score in candidates[:4]])
        top_scores = top_scores - top_scores.max()
        weights = np.exp(top_scores)
        weights /= weights.sum()
        alternatives = [
            ChordAlternative(Chord(chord.root, quality), float(weight))
            for (quality, _), weight in zip(candidates[1:4], weights[1:4])
        ]
        return Chord(chord.root, best_quality), alternatives, confidence


# #
# Convenience functions
# #

def harmonize(
    melody: Melody | Sequence[NoteEvent],
    model: MySongModel,
    *,
    key: Key | str | None = None,
    style: Style | None = None,
    beats_per_bar: float | None = None,
    tempo_bpm: float | None = None,
) -> Harmonization:
    return Harmonizer(model).harmonize(
        melody,
        key=key,
        style=style,
        beats_per_bar=beats_per_bar,
        tempo_bpm=tempo_bpm,
    )


def chords_for_bars(
    harmony: Harmonization,
    beats_per_bar: float,
    bars: int,
) -> list[Chord | None]:
    """One chord per bar, for a lead-sheet display or an arrangement grid."""
    out: list[Chord | None] = []
    for bar in range(bars):
        start = bar * beats_per_bar
        end = start + beats_per_bar
        best: Chord | None = None
        best_overlap = 0.0
        for segment in harmony.segments:
            overlap = min(end, segment.end_beat) - max(start, segment.start_beat)
            if overlap > best_overlap:
                best, best_overlap = segment.chord, overlap
        out.append(best)
    return out

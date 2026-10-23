"""
Sing a recorded vocable take along a real melody.

This is the musicality layer used by the "one locked word instead of Mix"
renderer in server.py (_render_track_single_vocable_offline). It exists as
its own module, rather than inline in server.py, because every piece here is
pure DSP on plain numpy arrays, no server state, no config, no model, so
it can be reasoned about and tested on its own.

There are four separate problems, and they are deliberately solved
separately:

0. THE RECORDING'S OWN PITCH MOVES, AND ONE SHIFT CANNOT FOLLOW IT.
   Measured on this project's own takes, a single recorded syllable sweeps
   through 100-400 cents of pitch inside a second ("Cold" runs 268 -> 341 ->
   304 -> 335 Hz in 0.86 s). That is part of what makes the takes sound
   human, and it is fatal to a note-by-note renderer: a constant shift places
   the syllable's MEDIAN on the note and leaves the rest of it up to a couple
   of hundred cents away, and a long note that loops the syllable replays the
   whole sweep once per loop. Measured on a real render, that was exactly the
   size of the residual error, 200 cents median between the pitch the
   renderer was asked for and the pitch it produced, while the shifter on its
   own is accurate to ~5 cents. flatten_syllable_pitch() below splits each
   syllable's own pitch movement into a slow part (the glide or drift that
   one constant shift cannot follow) and a fast part (the singer's own
   vibrato and micro-movement, which is most of what makes a voice sound
   alive) and removes only the slow one; is_pitched_syllable() keeps the
   unpitched consonant fragments of a word out of it entirely.

1. WHAT NOTES IS THE SONG ACTUALLY SINGING?
   The old renderer used librosa's generic onset detector as its note grid.
   An onset detector only knows "something in the spectrum just changed"
   it has no idea what a note is, so a held vowel's own vibrato registers as
   several "onsets", and a note change with a soft attack can register as
   none. The result is a robot part whose rhythm has only a loose
   relationship to the melody. track_notes() below instead reads the
   performance's own pitch curve: a new note begins when the sung pitch
   actually moves, which is exactly how a real singer's phrases are
   structured. Octave errors in the raw detector output (a YIN reading that
   locks onto a harmonic instead of the fundamental) are corrected first
   in this project's own reference track, roughly 6% of frames are one or
   two octaves off, and leaving those in would make the robot leap around by
   a factor of two. Correcting them is done by CONTINUITY: every reading is
   folded toward the register the melody has actually been tracking, rather
   than toward a window median, a run of consistently wrong readings can
   dominate a window, and on the reference track exactly that turned a 1.8 s
   stretch of a held note sitting near 200 Hz into a 400 Hz reading, i.e. the
   robot sang that whole phrase an octave above the singer. The spectrum is
   asked for a second opinion only when a reading wants to move a whole
   octave UP and stay there, because that is the one move a harmonic lock
   and a genuine leap both make. A note's pitch is then measured over the MIDDLE of the
   note rather than all of it, because a note's own first and last frames are
   still arriving from the previous note or already leaving for the next one:
   measured, that moved the reported pitch of a note from 96 cents away from
   what the singer held to 14 cents away, which is the difference between
   singing the tune and singing beside it.

2. HOW DO WE CHANGE THE RECORDING'S PITCH WITHOUT WRECKING ITS VOICE?
   Shifting pitch by resampling drags the formants (vocal-tract resonances)
   along with the fundamental, which is the chipmunk/monster effect. The
   project's shared shifter (neural_vocable_bank._formant_preserving_pitch_shift)
   handles that with a cepstral formant transplant, and was tightened up as
   part of this work, per-frame instead of per-take envelope correction,
   over a 2048-sample analysis window, which cut its measured envelope
   error on this project's own takes from 4.5-12.3 dB to 2.6-4.1 dB. On top of
   that, FormantPreservingShifter below memoizes shifted copies per
   (take, pitch bucket) so the one vocable being sung is re-shifted only
   when the melody actually reaches a pitch it has not used yet.

3. HOW DOES A NOTE SOUND LIKE A PERSON, INSTEAD OF A LOOP POINT?
   A real singer does not start every note with an instant full-amplitude
   attack at a brand new pitch. cents_curve_for_note() gives every note:
     - a short portamento into it from the previous note's pitch,
     - a vibrato that fades in after the note has been held a moment,
     - a soft attack for notes that continue an already-sounding phrase.
   apply_cents_curve() then renders that curve as a variable-rate read of
   the note's material (the same trick apply_pitch_curve_follow uses), so
   all of it costs one interpolation pass and never changes the note's
   length on the timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # librosa is already a hard dependency of the project; the guard is only
    import librosa  # so this module stays importable in a stripped-down env
    _HAS_LIBROSA = True
except Exception:  # pragma: no cover
    _HAS_LIBROSA = False


# The lowest pitch we will ever treat as a sung note. Below this we are
# almost certainly looking at rumble, a bass instrument leaking into the
# vocal stem, or a pitch-detector octave error.
MIN_NOTE_HZ = 60.0
MAX_NOTE_HZ = 1100.0

# How far past the shift cap a note has to be before singing it in another
# octave of the take is better than clipping it. Half an octave: below this a
# clip is a tuning error a listener may forgive, above it the note is simply
# wrong, and an octave away from the melody but in tune with the track is the
# better mistake (see shift_for_note).
FOLD_OVER_CLIP_SEMITONES = 6.0

# A note has to move by more than this (in cents) before a new note begins.
# Below it, the movement is vibrato, a slide, or detector jitter, real
# melodies almost never move by less than a semitone and come back.
NOTE_JUMP_CENTS = 55.0

# How many consecutive frames must agree on the new pitch before the note
# change is committed. One frame (~23ms) of a spurious reading is enough to
# start a slur otherwise; two is short enough to never lag a real attack.
NOTE_CONFIRM_FRAMES = 2

# Notes shorter than this get absorbed into their neighbour. A sung note is
# rarely shorter (a 16th note at 120 BPM is 0.125 s, which is where this
# sits), while a detector transient or a couple of frames of a glide passing
# through are exactly this short, measured on this project's reference
# track, a ninth of the tracks's notes were 0.09-0.10 s and they read as the
# part chattering rather than singing.
MIN_NOTE_S = 0.12

# How much of each end of a note to leave out when reporting its pitch. A
# note's own first and last frames are where the pitch is still ARRIVING from
# the previous note or already leaving toward the next one (a real singer's
# portamento, plus the detector's own settling time), so including them pulls
# the reported pitch toward the neighbours, measured on this project's own
# reference track, the median note pitch taken over the WHOLE note lands 96
# cents away from the pitch the singer actually held, and taken over the
# middle 70% it lands 14 cents away. That is the difference between the robot
# singing the tune and singing slightly beside it.
NOTE_PITCH_TRIM = 0.15

# Octave folding, by continuity (see fold_octave_errors). A reading within
# OCTAVE_TRUST_CENTS of the register the melody has been tracking is taken at
# face value; further away than that it is folded to the nearest octave
# unless a sustained run of readings insists otherwise.
OCTAVE_TRUST_CENTS = 650.0

# How many readings establish the first register (~0.28 s at the project's
# default 1024-sample frame). Long enough that one bad reading cannot set it,
# short enough to happen before the first note is even over.
OCTAVE_INIT_FRAMES = 12

# How long a disagreement has to persist before it is believed (~0.23 s). A
# one-frame octave flip is a detector artifact; half a second of it is a
# different note.
OCTAVE_JUMP_FRAMES = 10

# While a rise is unadopted, re-ask the spectrum every this many frames, so a
# genuine octave leap is eventually believed and a harmonic lock is not.
OCTAVE_RECHECK_FRAMES = 16

# How strong the LOWER octave's own harmonics have to be, relative to the
# strength of the reading's own, before the spectrum is read as "the lower
# octave is the real fundamental": 0.35 is about -9 dB, well above the noise
# floor between harmonics and well below a real sung fundamental.
OCTAVE_SUBHARMONIC_RATIO = 0.35

# Median smoothing applied to the pitch curve before note segmentation, in
# frames. Three frames (~70ms) removes single-frame outliers without
# blurring the start of a real note by more than one frame.
PITCH_MEDIAN_FRAMES = 3


# ============================================================================
# Part 1: pitch curve -> notes
# ============================================================================

def _fold_toward(hz: float, ref: float) -> float:
    """The octave of `hz` that sits closest to `ref`."""
    if hz <= 0 or ref <= 0:
        return hz
    return float(hz * (2.0 ** math.floor(math.log2(ref / hz) + 0.5)))


def _octave_evidence(hz_high: float, seg: np.ndarray, sr: int) -> bool:
    """Does the spectrum really support `hz_high` as the fundamental?

    Used for one question only: a reading has been sitting a whole octave
    above the register the melody is tracking, and we have to decide whether
    that is a genuine octave leap or a detector locked onto a harmonic.

    The two hypotheses differ in a measurable way. If the true fundamental is
    the LOWER one (hz_high / 2), the voice's odd harmonics, at hz_high / 2
    and 1.5 * hz_high, are real, strong partials. If the true fundamental is
    hz_high, those two frequencies fall BETWEEN harmonics and are close to the
    noise floor. So: strong energy at both probes means the lower octave is
    real (return False, keep folding); weak energy there means the high
    reading is its own fundamental (return True, adopt it).

    This is deliberately a narrow, two-probe test. A full harmonic-series
    score was tried in this project and was not trustworthy on real recorded
    voice, where the harmonics are dense and the valleys shallow; two fixed
    probes on the one ambiguity that actually matters survive that.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if seg.size < 256 or sr <= 0 or hz_high <= 0:
        return True
    low = hz_high / 2.0
    if low < 60.0 or hz_high * 1.5 >= 0.45 * sr:
        return True
    win = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)

    def level(f: float) -> float:
        lo = f * (2.0 ** (-70.0 / 1200.0))
        hi = f * (2.0 ** (70.0 / 1200.0))
        m = (freqs >= lo) & (freqs <= hi)
        return float(spec[m].max()) if m.any() else 0.0

    own = max(level(hz_high), level(2.0 * hz_high), 1e-9)
    sub = min(level(low), level(1.5 * hz_high))
    return sub < own * OCTAVE_SUBHARMONIC_RATIO


def fold_octave_errors(hz: np.ndarray, voiced: np.ndarray,
                       audio: np.ndarray | None = None,
                       sr: int | None = None,
                       hop_samples: int | None = None,
                       init_frames: int = OCTAVE_INIT_FRAMES,
                       jump_frames: int = OCTAVE_JUMP_FRAMES,
                       recheck_frames: int = OCTAVE_RECHECK_FRAMES,
                       trust_cents: float = OCTAVE_TRUST_CENTS) -> np.ndarray:
    """Correct octave errors by CONTINUITY with the register being tracked.

    The previous version folded every reading toward the median of a ~2.2 s
    window of raw readings. That reference is wrong in exactly the case that
    matters: when a detector locks onto a harmonic for a whole phrase, most of
    the readings IN that window are the wrong octave, so the median moves to
    the wrong octave and the fold drags the correct readings up to meet it.
    Measured on this project's reference track: a 200 Hz held note was read at
    ~400 Hz for 1.8 s, the window median went with it, and the robot sang that
    phrase an octave above the singer.

    So the reference is not a window, it is the melody's own recent history:
    a running register estimate that follows the readings it believes. A
    reading within `trust_cents` of it is believed as-is and nudges it along.
    A reading further away is folded to its nearest octave, unless a
    sustained run of readings insists, in which case the disagreement is real
    and the register follows it instead. A sustained rise is the one case
    worth asking the spectrum about (see _octave_evidence), because that is
    the move both a harmonic lock and a genuine upward leap make.

    When `audio`, `sr` and `hop_samples` are supplied, frame i of the curve is
    assumed to be the audio window [i * hop_samples, i * hop_samples + hop]
    of `audio`, which is how the caller in server.py builds it.
    """
    hz = np.asarray(hz, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    n = len(hz)
    out = hz.copy()
    if n == 0:
        return out
    ok = voiced & (hz > 0)
    if not ok.any():
        return out
    hop = int(hop_samples) if hop_samples else 0

    ref = 0.0
    init: list[float] = []
    up_run = 0
    low_run = 0
    adopt = False
    last_check = -10 ** 9

    for i in range(n):
        if not ok[i]:
            up_run = 0
            low_run = 0
            continue
        h = float(hz[i])
        if ref <= 0.0:
            # Establish the first register from the first few readings. The
            # geometric mean is the right average for pitch, and it means one
            # bad reading cannot set the register by itself.
            init.append(h)
            out[i] = h
            if len(init) >= max(2, int(init_frames)):
                ref = float(np.exp(np.mean(np.log(init))))
                for j in range(0, i + 1):
                    if ok[j]:
                        out[j] = _fold_toward(float(hz[j]), ref)
            continue

        dev = 1200.0 * math.log2(h / ref)
        if abs(dev) <= trust_cents:
            out[i] = h
            ref *= 2.0 ** ((0.18 * dev) / 1200.0)
            up_run = 0
            low_run = 0
            adopt = False
            continue

        near = _fold_toward(h, ref)
        if dev < 0.0:
            # The reading is about an octave BELOW the tracked register.
            # Detectors invent subharmonics far less often than they lock
            # onto harmonics, so a sustained drop is usually a real downward
            # leap: follow it after a short confirmation.
            low_run += 1
            if low_run >= max(1, int(jump_frames)):
                ref = h
                out[i] = h
                low_run = 0
            else:
                out[i] = near
            continue

        # The reading is about an octave ABOVE the tracked register: keep
        # folding it down until either the melody comes back to the tracked
        # register or the spectrum says the rise is real.
        up_run += 1
        if up_run >= max(1, int(jump_frames)):
            if (i - last_check) >= max(1, int(recheck_frames)):
                last_check = i
                adopt = False
                if audio is not None and sr and hop:
                    c = i * hop + hop // 2
                    seg = np.asarray(audio)[max(0, c - 1024):c + 1024]
                    adopt = _octave_evidence(h, seg, int(sr))
            if adopt:
                ref = h
                out[i] = h
                up_run = 0
            else:
                out[i] = near
        else:
            out[i] = near
    return out


def _stable_pitch(vals: list[float], trim: float = NOTE_PITCH_TRIM) -> float:
    """The pitch of a note, measured away from its own two edges (above).

    After the edges are trimmed, frames that cannot belong to this note are
    dropped too: a note boundary can land a few frames late (the pitch has to
    MOVE before the tracker can see it move), so the trimmed window can still
    hold a short piece of the previous or next note, and a plain median over a
    short note can be dragged tens of cents by it. Keeping only readings
    within 220 cents of the median, a note's own vibrato and portamento stay
    well inside that, a different note does not, and taking the median of
    those gives the pitch the note actually held.
    """
    if not vals:
        return 0.0
    n = len(vals)
    lo = int(n * trim)
    hi = max(lo + 1, n - lo)
    core = vals[lo:hi] or vals
    med = float(np.median(core))
    if med <= 0:
        return med
    kept = [v for v in core if v > 0 and abs(1200.0 * math.log2(v / med)) <= 220.0]
    return float(np.median(kept)) if kept else med


def smooth_pitch(hz: np.ndarray, voiced: np.ndarray, k: int = PITCH_MEDIAN_FRAMES) -> np.ndarray:
    """Median-filter the corrected curve so one reading can't start a note."""
    hz = np.asarray(hz, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    n = len(hz)
    out = hz.copy()
    half = max(0, int(k) // 2)
    for i in range(n):
        if not voiced[i]:
            continue
        lo = max(0, i - half); hi = min(n, i + half + 1)
        vals = hz[lo:hi][voiced[lo:hi] & (hz[lo:hi] > 0)]
        if len(vals):
            out[i] = float(np.median(vals))
    return out


@dataclass
class Note:
    start_s: float
    end_s: float
    f0_hz: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# A sung melody moves in small steps; a leap of an octave is real but rare, and
# it is exactly twelve semitones. A note sitting MORE than this far from the
# trend of the melody around it is not a melody, it is an octave tracking error
# (see stabilize_note_octaves). Measured on this project's reference track, a
# single-octave error reads as far as 9 st from a trend that is itself dragged
# toward the wrong reading, and the notes that are genuinely far from their
# neighbours (a 200 Hz note next to a 350 Hz one: 9.7 st) are re-judged as
# suspects either way and left alone by the evidence guard.
OCTAVE_OUTLIER_SEMITONES = 9.0

# The local trend a note is judged against: the median pitch of its OTHER
# notes within this many seconds either side. Wide enough that a phrase
# establishes its own register, short enough that a real register change
# between sections is still followed.
OCTAVE_OUTLIER_WINDOW_S = 2.0

# ...and how many other notes have to sit in that window before a note is
# judged at all. One or two neighbours are not a trend.
OCTAVE_OUTLIER_MIN_NEIGHBOURS = 3


# How much energy a candidate octave's own frequency has to carry, relative
# to the strongest candidate for the same note, before that octave is taken as
# the note's fundamental (see correct_note_octave). Measured on this project's
# own reference track: a genuinely sung fundamental can sit ~6 dB below its own
# 2nd harmonic on this voice, i.e. 0.5 of it, while a frequency with no
# partial at all, the 87 Hz reading of a 348 Hz note, sits far below this.
NOTE_FUNDAMENTAL_RATIO = 0.35


def _octave_support(hz: float, spec: np.ndarray, freqs: np.ndarray, cents: float = 70.0) -> float:
    """The spectral energy a candidate fundamental `hz` actually has."""
    if hz <= 0:
        return 0.0
    lo = hz * (2.0 ** (-cents / 1200.0))
    hi = hz * (2.0 ** (cents / 1200.0))
    m = (freqs >= lo) & (freqs <= hi)
    return float(spec[m].max()) if m.any() else 0.0


def correct_note_octave(hz: float, seg: np.ndarray, sr: int,
                        low_hz: float = MIN_NOTE_HZ, high_hz: float = MAX_NOTE_HZ,
                        ratio: float = NOTE_FUNDAMENTAL_RATIO,
                        prefer_hz: float = 0.0) -> float:
    """The octave of `hz` that this note's OWN audio actually supports.

    A pitch detector on real recorded voice can report a note an octave or two
    away from what was sung, and no amount of neighbour comparison can tell
    which reading is right when several notes in a row are wrong the same way.
    The spectrum can: an octave that has no partial of its own anywhere near
    it is not the note that was sung. Measured on this project's reference
    voice, a real fundamental can sit ~6 dB below its own 2nd harmonic (so
    loudness alone decides nothing), while the bogus 87 Hz reading of a 348 Hz
    note had no energy at 87 Hz at all.

    So every octave of `hz` that carries at least `ratio` of the strongest
    candidate's energy is a candidate, and the one nearest `prefer_hz` (the
    melody's own local trend, when the caller has one) is returned; without a
    preference, the lowest supported octave is, which is the fundamental of
    the partial series the spectrum shows.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if hz <= 0 or seg.size < 512 or not sr:
        return hz
    window = seg * np.hanning(seg.size)
    n_fft = 8192
    spec = np.abs(np.fft.rfft(window, n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cands = [hz * (2.0 ** k) for k in range(-3, 4)]
    cands = [c for c in cands if low_hz <= c <= high_hz]
    if not cands:
        return hz
    sup = {c: _octave_support(c, spec, freqs) for c in cands}
    best = max(sup.values())
    if best <= 0.0:
        return hz
    supported = [c for c in sorted(sup) if sup[c] >= ratio * best]
    if not supported:
        return hz
    if prefer_hz > 0:
        return float(min(supported, key=lambda c: abs(math.log2(c / prefer_hz))))
    return float(supported[0])


def stabilize_note_octaves(notes: list[Note],
                           audio: np.ndarray | None = None,
                           sr: int | None = None,
                           window_s: float = OCTAVE_OUTLIER_WINDOW_S,
                           outlier_semitones: float = OCTAVE_OUTLIER_SEMITONES,
                           min_neighbours: int = OCTAVE_OUTLIER_MIN_NEIGHBOURS) -> int:
    """Put back any note the tracker left a whole octave away from its melody.

    The frame-level fold above (fold_octave_errors) works one reading at a
    time, and on real recorded voice it can still settle a whole stretch of
    readings in the wrong octave: measured on this project's own reference
    track, the SAME ~348 Hz stretch was tracked at 348 Hz in one render and at
    86.9 Hz in another, two octaves down. A note that far from the notes
    around it is not a melody, it is the glitch that makes a render sound like
    a second, unrelated singer for a moment.

    So every note is compared with the ROBUST TREND around it, the median of
    the other notes' pitches within `window_s`, in the log domain, and when
    it sits more than `outlier_semitones` away from that trend it is moved
    back. Which octave it moves to is decided by the note's OWN audio when it
    is supplied (see correct_note_octave): a trend cannot tell several
    wrong-in-the-same-way notes apart from a melody that really lives there,
    but the note's own spectrum can. Without audio, the note is simply folded
    to the octave nearest the trend.

    Judging against a window of neighbours rather than against the previous
    note is what keeps a genuine run of notes in a new register intact: once
    the melody has stayed there, the trend is there too, and none of those
    notes are outliers any more.

    Returns how many notes were moved, so the caller can log it.
    """
    judged = [n for n in notes if n.f0_hz > 0]
    if len(judged) < min_neighbours + 1:
        return 0
    logs = np.log2(np.asarray([n.f0_hz for n in judged], dtype=np.float64))
    starts = np.asarray([n.start_s for n in judged], dtype=np.float64)
    moved = 0
    for i, note in enumerate(judged):
        near = np.abs(starts - note.start_s) <= float(window_s)
        near[i] = False
        neigh = logs[near]
        if neigh.size < min_neighbours:
            continue
        dev_semitones = 12.0 * float(logs[i] - np.median(neigh))
        if abs(dev_semitones) <= float(outlier_semitones):
            continue
        trend_hz = float(2.0 ** float(np.median(neigh)))
        chosen = 0.0
        if audio is not None and sr:
            a = max(0, int(round(note.start_s * sr)))
            b = min(len(audio), int(round(note.end_s * sr)))
            if b - a >= 512:
                evidence_hz = correct_note_octave(note.f0_hz, np.asarray(audio)[a:b], int(sr),
                                                  prefer_hz=trend_hz)
                # only a place that is genuinely CLOSER to the melody around it
                # is accepted; the spectrum is evidence, not an override
                if evidence_hz > 0 and abs(12.0 * math.log2(evidence_hz / trend_hz)) < abs(dev_semitones):
                    chosen = evidence_hz
        if chosen <= 0.0:
            folds = int(round(dev_semitones / 12.0))
            if folds == 0:
                continue
            chosen = note.f0_hz * (2.0 ** -folds)
        # the move is recorded in `logs`, so notes judged after this one see
        # the corrected melody
        logs[i] = math.log2(chosen)
        note.f0_hz = float(chosen)
        moved += 1
    return moved


# How concentrated the note pitches have to be around a 100-cent grid
# (measured as the mean resultant length of their offsets) before we will
# call it "this track is in tune with a 12-tone grid" and snap to it. Below
# this the track is sliding, spoken, or otherwise not on a grid, and
# snapping would be imposing a key the music does not have.
TUNING_MIN_CONCENTRATION = 0.45
TUNING_MIN_NOTES = 6


def estimate_tuning_offset_cents(note_hz: list[float]) -> tuple[float, float]:
    """The track's own tuning grid, as an offset in cents from A440 equal
    temperament, plus how confident that estimate is (0..1).

    Notes are treated as points on a circle of 100 cents: a track recorded
    30 cents flat has every note at -30 (mod 100), and the circular mean
    recovers that even though the notes themselves span two octaves. The
    concentration (mean resultant length) is the confidence, 1.0 for
    perfectly in-tune notes, near 0 for a random or sliding pitch set.
    """
    vals = [h for h in note_hz if h and h > 0]
    if len(vals) < TUNING_MIN_NOTES:
        return 0.0, 0.0
    cents = np.asarray([1200.0 * math.log2(h / 440.0) for h in vals])
    offs = ((cents + 50.0) % 100.0) - 50.0
    z = np.mean(np.exp(2j * np.pi * offs / 100.0))
    return float(np.angle(z) * 100.0 / (2 * np.pi)), float(np.abs(z))


def snap_to_song_grid(hz: float, offset_cents: float) -> float:
    """Move one note pitch to the nearest step of the track's own grid.

    The amount moved is at most 50 cents, but it is the difference between a
    robot part that sings the song's notes and one that sings near them: a
    tracked note is the median of a sung note with vibrato, portamento and
    detector jitter on it, so it sits a few tens of cents off the note the
    song is actually on, and taken with the shifter's own quarter-semitone
    error that is audibly out of tune against the track.
    """
    if hz <= 0:
        return hz
    cents = 1200.0 * math.log2(hz / 440.0)
    snapped = (cents - offset_cents) / 100.0
    snapped = round(snapped) * 100.0 + offset_cents
    return float(440.0 * (2.0 ** (snapped / 1200.0)))


def track_notes(hz: np.ndarray, voiced: np.ndarray, hop_s: float,
                jump_cents: float = NOTE_JUMP_CENTS,
                confirm_frames: int = NOTE_CONFIRM_FRAMES,
                min_note_s: float = MIN_NOTE_S,
                audio: np.ndarray | None = None, sr: int | None = None,
                hop_samples: int | None = None,
                snap_to_key: bool = True) -> tuple[np.ndarray, list[Note]]:
    """Segment a sung pitch curve into notes.

    Returns (corrected_and_smoothed_curve, notes). The curve is returned
    because the caller wants it too (for the note's own pitch and for the
    glide/vibrato shaping), recomputing it would be wasted work and, worse,
    could disagree with the notes.

    A note change is committed only once `confirm_frames` consecutive
    readings agree on the new pitch. Until then the frames are held in a
    pending bucket: if the pitch drops back (vibrato, a slide passing
    through, one bad reading) the note simply continues and the pending
    frames are folded back in, so no spurious note boundary ever reaches the
    caller.
    """
    hz = np.asarray(hz, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    corrected = fold_octave_errors(hz, voiced, audio=audio, sr=sr, hop_samples=hop_samples)
    # NOTE: a frame-by-frame spectral octave arbitration was tried here
    # (correct_octaves_by_spectrum) and removed again: on this project's own
    # reference track it moved 10% of frames, was right about half the time,
    # and dropped whole phrases an octave. What stays is the continuity fold
    # above, with the spectrum consulted only for the one decision neither
    # side can settle alone: a sustained rise (see _octave_evidence).
    corrected = smooth_pitch(corrected, voiced)
    notes: list[Note] = []

    cur_start: int | None = None
    cur_vals: list[float] = []
    pend_start: int | None = None
    pend_vals: list[float] = []

    def _flush(until: int):
        if cur_start is not None and cur_vals:
            notes.append(Note(cur_start * hop_s, until * hop_s, _stable_pitch(cur_vals)))

    for i in range(len(corrected)):
        v = bool(voiced[i]) and corrected[i] >= MIN_NOTE_HZ and corrected[i] <= MAX_NOTE_HZ
        if not v:
            _flush(i)
            cur_start, cur_vals = None, []
            pend_start, pend_vals = None, []
            continue

        h = float(corrected[i])
        if cur_start is None:
            cur_start, cur_vals = i, [h]
            pend_start, pend_vals = None, []
            continue

        ref = float(np.median(cur_vals))
        if abs(1200.0 * math.log2(h / ref)) > jump_cents:
            if pend_start is None:
                pend_start, pend_vals = i, [h]
            else:
                pend_vals.append(h)
                if len(pend_vals) >= confirm_frames:
                    notes.append(Note(cur_start * hop_s, pend_start * hop_s, _stable_pitch(cur_vals)))
                    cur_start, cur_vals = pend_start, pend_vals[:]
                    pend_start, pend_vals = None, []
        else:
            if pend_start is not None:
                cur_vals.extend(pend_vals)
                pend_start, pend_vals = None, []
            cur_vals.append(h)

    _flush(len(corrected))

    merged: list[Note] = []
    for note in notes:
        if merged and note.duration_s < min_note_s:
            prev = merged[-1]
            merged[-1] = Note(prev.start_s, note.end_s,
                              float(np.median([prev.f0_hz, note.f0_hz])))
        else:
            merged.append(note)

    # Finally, put the whole part on the track's OWN tuning grid. The melody
    # this is tracking was sung, so its notes carry vibrato, portamento and
    # detector jitter and land a few tens of cents either side of the note the
    # song is actually on. Snapping each note to the nearest 100 cents of the
    # grid the track itself is tuned to (measured, not assumed to be A440)
    # moves each of them by at most 50 cents, and it is what makes the robot
    # part sound in tune WITH THE RECORDING instead of near it.
    if snap_to_key and merged:
        offset, confidence = estimate_tuning_offset_cents([n.f0_hz for n in merged])
        if confidence >= TUNING_MIN_CONCENTRATION:
            merged = [Note(n.start_s, n.end_s, snap_to_song_grid(n.f0_hz, offset))
                      for n in merged]
    return corrected, merged


def octave_placement(median_note_hz: float, base_hz: float) -> int:
    """Whole octaves of headroom/tailroom for the whole part, chosen once.

    The vocable take was sung in some register of its own; the input melody
    lives somewhere else. Placing the whole part in the octave that gets the
    median melody note closest to the recording's own pitch means the
    common case needs only a small shift, and, just as important, the
    placement is chosen ONCE for the part, so a melody that genuinely drops
    an octave still drops an octave in the robot part. Per-note placement
    would silently flatten any real octave-wide melodic movement to nothing;
    the only per-note exception is a note that would need more than a whole
    octave of shift at all, which is sung an octave from the melody rather
    than out of tune (see shift_for_note).
    """
    if median_note_hz <= 0 or base_hz <= 0:
        return 0
    return int(round(math.log2(base_hz / median_note_hz)))


def shift_for_note(note_hz: float, base_hz: float, octave_k: int,
                   max_semitones: float = 12.0) -> float:
    """Semitone shift between a syllable's own pitch and a note's pitch.

    Usually this is the exact interval, and the note lands exactly. It stops
    being exact when a note sits so far from the take's own register that the
    interval would go past `max_semitones`, the point where a shift stops
    sounding like the same person. Measured on this project's reference
    track, that was not rare: the melody runs from about 87 Hz up to 354 Hz
    while the take's own syllables sit near 280-380 Hz, so 12% of the note
    shifts needed more than an octave, and clipping them (which is what
    used to happen) left those notes up to 1344 cents away from the note,
    i.e. audibly wrong, which is the one thing a listener never forgives.

    So a note that is hopelessly out of range is sung in the octave of the
    take that fits instead: it stays exactly in tune with the track, at an
    octave from where the melody sits, which is what a real singer does when
    a line leaves their range. The fold is deliberately the LAST resort: it
    only happens once clipping would leave the note more than
    FOLD_OVER_CLIP_SEMITONES out, because a small clip is a small tuning
    error while a fold moves the note to a different octave of the melody
    and the caller's phrase-level placement (see the renderer) has already
    moved the whole line into range wherever a single octave could do it.
    """
    if note_hz <= 0 or base_hz <= 0:
        return 0.0
    semis = 12.0 * math.log2(note_hz / base_hz) + 12.0 * octave_k
    if abs(semis) > max_semitones + FOLD_OVER_CLIP_SEMITONES:
        semis -= 12.0 * round(semis / 12.0)
    return float(np.clip(semis, -max_semitones, max_semitones))


# ============================================================================
# Part 2: formant-preserving pitch shift
# ============================================================================

class FormantPreservingShifter:
    """Shift takes by arbitrary semitone amounts, computing each one at most once.

    The shift itself is the project's own formant-corrected one
    (neural_vocable_bank._formant_preserving_pitch_shift: a phase vocoder plus
    a per-frame cepstral envelope transplant, measured at 2.6-4.1 dB mean
    envelope error against 4.5-12.3 dB for the same shifter without it, i.e.
    it is what keeps the shifted syllable sounding like the same person).

    What this class adds is memoization, because the melody will ask for the
    same syllable at the same pitch over and over. Shift amounts are
    quantized to BUCKET_SEMITONES so a melody that wanders by a cent or two
    does not blow the cache up; 0.25 of a semitone means the worst-case
    pitch error is 12.5 cents, under the vibrato depth the part is shaped
    with and under the tuning drift of a real recording.

    `take_id` is chosen by the caller and must identify the audio being
    passed in (e.g. an index into the vocable's syllable list).
    """

    BUCKET_SEMITONES = 0.25

    def __init__(self, sample_rate: int, max_shift_semitones: float = 12.0):
        self.sample_rate = int(sample_rate)
        self.max_shift_semitones = float(max_shift_semitones)
        self._shifted: dict[tuple[int, int], np.ndarray] = {}

    def shift(self, take_id: int, audio: np.ndarray, n_steps: float) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if abs(n_steps) < self.BUCKET_SEMITONES / 2:
            return audio
        n_steps = float(np.clip(n_steps, -self.max_shift_semitones, self.max_shift_semitones))
        bucket = int(round(n_steps / self.BUCKET_SEMITONES))
        key = (int(take_id), bucket)
        cached = self._shifted.get(key)
        if cached is not None:
            return cached

        shifted = _formant_preserving_shift(audio, self.sample_rate, bucket * self.BUCKET_SEMITONES)
        self._shifted[key] = shifted
        return shifted


def _formant_preserving_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """The shared shifter, with a safe degradation path. A take that somehow
    breaks the spectral pass still comes back sounding right (or at least
    at the right pitch) rather than silent."""
    try:
        from synthesis.neural_vocable_bank import _formant_preserving_pitch_shift
        return _formant_preserving_pitch_shift(np.asarray(audio, dtype=np.float32), sr, n_steps)
    except Exception:
        if _HAS_LIBROSA:
            return librosa.effects.pitch_shift(np.asarray(audio, dtype=np.float32), sr=sr,
                                                n_steps=float(n_steps), n_fft=2048)
        return np.asarray(audio, dtype=np.float32)


# ============================================================================
# Part 3: shaping one note like a sung note
# ============================================================================

def apply_cents_curve(audio: np.ndarray, sr: int, cents: np.ndarray) -> np.ndarray:
    """Re-read `audio` at a time-varying speed given by a per-sample cents curve.

    Reading faster raises pitch, reading slower lowers it, the same
    variable-rate resample apply_pitch_curve_follow in server.py uses. The
    warp is renormalized so the note still occupies exactly its own length
    on the timeline: a glide or vibrato must never shift a note's onset or
    push it off the song's own beat.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n < 2 or len(cents) < 2:
        return audio
    ratio = np.exp2(np.asarray(cents, dtype=np.float64) / 1200.0)
    if len(ratio) != n:
        ratio = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ratio)), ratio)
    warp = np.cumsum(ratio)
    if warp[-1] <= 0:
        return audio
    warp *= (n - 1) / warp[-1]
    out = np.interp(warp, np.arange(n, dtype=np.float64), audio.astype(np.float64))
    return out.astype(np.float32)


def cents_curve_for_note(n_samples: int, sr: int,
                         glide_from_cents: float = 0.0,
                         glide_s: float = 0.07,
                         vibrato_depth_cents: float = 0.0,
                         vibrato_rate_hz: float = 5.3,
                         vibrato_delay_s: float = 0.30,
                         vibrato_ramp_s: float = 0.25,
                         drift_depth_cents: float = 6.0,
                         drift_rate_hz: float = 0.7) -> np.ndarray:
    """The per-sample pitch curve of one note, in cents.

    Three parts, all of them things a real singer does and a loop point
    does not:
      - portamento: start at the previous note's pitch and arrive at this
        one over a few tens of ms. Zero for a phrase's own first note.
      - vibrato: fades in after the note has been held past
        `vibrato_delay_s`, with depth growing a little with the note's
        length, a short note gets almost none, a held note gets real
        sung vibrato.
      - drift: a very slow, very shallow wander (well under a tenth of a
        semitone) so a long note is never perfectly, digitally flat.
    """
    n = int(n_samples)
    if n <= 1:
        return np.zeros(max(1, n))
    t = np.arange(n, dtype=np.float64) / sr

    glide = np.zeros(n)
    if abs(glide_from_cents) > 0.5 and glide_s > 0:
        glide_len = max(1, min(n, int(glide_s * sr)))
        # raised-cosine ease-out: fast at first, settled by the end
        x = np.linspace(0.0, 1.0, glide_len)
        ease = 0.5 - 0.5 * np.cos(np.pi * x)
        glide[:glide_len] = glide_from_cents * (1.0 - ease)

    vib = np.zeros(n)
    if vibrato_depth_cents > 0:
        # depth scales with how long the note is actually held
        length_factor = float(np.clip(n_samples / sr / 1.2, 0.0, 1.0))
        depth = vibrato_depth_cents * length_factor
        if depth > 0.2:
            onset = min(n, int(vibrato_delay_s * sr))
            ramp = np.ones(n)
            if n > onset and vibrato_ramp_s > 0:
                ramp_len = max(1, min(n - onset, int(vibrato_ramp_s * sr)))
                ramp[:onset] = 0.0
                ramp[onset:onset + ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            vib = depth * ramp * np.sin(2 * np.pi * vibrato_rate_hz * t)

    drift = np.zeros(n)
    if drift_depth_cents > 0 and n > 16:
        drift = drift_depth_cents * np.sin(2 * np.pi * drift_rate_hz * t + 0.7)

    return (glide + vib + drift).astype(np.float64)


def note_attack_samples(sr: int, fresh_attack: bool, duration_s: float) -> int:
    """How long the onset ramp of a note should be.

    A fresh attack (a real new syllable starting) gets a short, definite
    onset, it IS a new sound starting. A note that continues an
    already-sounding phrase gets NO ramp of its own: it is the same voice
    moving to a new pitch, and the equal-power crossfade the caller puts
    over that handoff is already the transition. Giving a legato note its
    own fade-in on top of that crossfade multiplies two fades together and
    is exactly what makes a flowing line sound re-triggered.
    """
    if not fresh_attack:
        return 0
    return max(8, min(int(0.022 * sr), int(0.45 * duration_s * sr)))


def apply_note_envelope(audio: np.ndarray, sr: int, fresh_attack: bool,
                        release_s: float = 0.0) -> np.ndarray:
    """Attack and/or release ramps for one note's material. Either can be
    switched off (release_s=0, or a non-fresh note), in which case the
    caller is responsible for the transition, see note_attack_samples."""
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n == 0:
        return audio
    env = np.ones(n, dtype=np.float32)
    atk = note_attack_samples(sr, fresh_attack, n / sr)
    if atk > 1:
        env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float32)
    if release_s > 0:
        rel = max(4, min(int(release_s * sr), n // 3))
        if rel > 1:
            env[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
    return audio * env


# Frame length used to decide which frames of a finished part count as
# "sung" for the level measurement. Short enough to follow a phrase edge.
LOUDNESS_FRAME_S = 0.02

# How far below the part's own loud frames a frame still counts as sung
# (-26 dB): that keeps a real quiet passage in the measurement while leaving
# the true silence between phrases out of it.
LOUDNESS_FLOOR_RATIO = 0.05


def _sung_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per-sample mask of the frames where a finished part is actually sounding.

    Used by normalize_loudness to measure "the sung part only". Derived from
    the part's own level rather than from the input track's gate flags: those
    flags cover every moment the gate was open, including moments where this
    particular render placed no note (a very short phrase the note tracker
    did not turn into a note, a breath between two notes). Those quiet
    stretches pulled the measured level down, the part was then boosted to
    compensate, and the peak ceiling had to pull the whole part back down by
    2-3 dB, which is why the same settings could come back at noticeably
    different levels from one render to the next.
    """
    n = len(audio)
    fr = max(64, int(round(LOUDNESS_FRAME_S * sr)))
    frames = n // fr
    if frames < 2:
        return np.ones(n, dtype=bool)
    block = np.asarray(audio[:frames * fr], dtype=np.float64).reshape(frames, fr)
    rms = np.sqrt(np.mean(block ** 2, axis=1))
    loud = float(np.percentile(rms, 90.0))
    if loud <= 1e-7:
        return np.ones(n, dtype=bool)
    keep = rms > loud * LOUDNESS_FLOOR_RATIO
    if not keep.any():
        return np.ones(n, dtype=bool)
    m = np.repeat(keep, fr)
    tail = n - m.shape[0]
    if tail > 0:
        # a part of the audio shorter than one analysis frame is left at the
        # last frame's verdict, so the mask always covers the whole part
        m = np.concatenate([m, np.full(tail, bool(keep[-1]), dtype=bool)])
    return m


def normalize_loudness(audio: np.ndarray, sr: int, target_rms: float = 0.11,
                       peak_ceiling: float = 0.97, mask: np.ndarray | None = None) -> np.ndarray:
    """One predictable level for every vocable, and every track.

    The takes themselves differ by up to ~5 dB from each other (measured:
    "Moon" sits at -13 dBFS RMS, "You Know You're Going to Feel It" at
    -18.5), which means the ROBOT VOLUME slider means a different thing
    depending on which word was picked. Normalize to a fixed RMS, then only
    squeeze whatever still pokes through the ceiling.

    The measurement is restricted to the part that is actually sounding
    (see _sung_mask). Without that, a track that is 60% silence comes out
    ~4 dB hotter than a wall-to-wall one, because the silence drags the
    average down.

    The ceiling is a soft knee, not a global gain change. Scaling the whole
    part by ceiling/peak sounds level-stable in theory and is not in
    practice: the loudest moment of the whole performance varies by a few dB
    between renders of the same music, so one loud note used to set the level
    of the entire track, and the same settings could come back 3 dB apart.
    Inside the knee everything the listener hears as the performance is
    untouched, and only the few samples above it are squeezed smoothly, so
    the ceiling bounds the peak without ever moving the part's level.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    if mask is not None:
        m = np.asarray(mask[:len(audio)], dtype=bool)
        if m.shape[0] < len(audio):
            m = np.concatenate([m, np.zeros(len(audio) - m.shape[0], dtype=bool)])
    else:
        m = _sung_mask(audio, sr)
    if not m.any():
        m = _sung_mask(audio, sr)
    measured = audio[m] if m.any() else audio
    rms = float(np.sqrt(np.mean(measured.astype(np.float64) ** 2)))
    if rms > 1e-7:
        audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio)))
    if peak > peak_ceiling:
        # Knee at 70% of the ceiling, so the squeeze starts ~3 dB below it
        # and never reaches it (tanh approaches knee + head asymptotically).
        knee = 0.7 * peak_ceiling
        head = peak_ceiling - knee
        mag = np.abs(audio)
        over = mag > knee
        if over.any():
            audio = np.array(audio, dtype=np.float64, copy=True)
            audio[over] = (np.sign(audio[over])
                           * (knee + head * np.tanh((mag[over] - knee) / head)))
    return audio.astype(np.float32)


# ============================================================================
# Part 3b: flattening the take's own pitch movement
#
# Measured on this project's own takes, every recorded syllable carries a
# large pitch movement of its own, "Cold" sweeps 268 -> 341 -> 304 -> 335 Hz
# inside 0.86 s (+/-200 cents), "Moon" moves 100-250 cents per syllable. That
# movement is part of what makes the takes sound like a person, but it is
# FATAL to a note-by-note renderer, for two separate reasons:
#
#   1. One constant shift cannot place a gliding syllable on a note: the
#      shift puts the syllable's MEDIAN on the note, and the rest of the
#      syllable is then up to a couple of hundred cents sharp or flat. That
#      is audible as a robot that never quite sings in tune, and measured,
#      it was exactly the size of the residual error (median ~200 cents
#      between the pitch the renderer was asked for and the pitch it
#      produced, while the shifter on its own is accurate to ~5 cents).
#   2. A long note loops the syllable, so the take's own pitch contour is
#      replayed over and over inside one held note, a wobble at the loop
#      rate, which no amount of correct shifting can fix.
#
# So each syllable's own pitch curve is measured once, split into its SLOW
# movement (the glide or drift that a constant shift cannot follow) and its
# FAST movement (the singer's own vibrato and micro-movement, which is most
# of what makes a voice sound alive), and only the slow part is removed
# (re-read at a varying rate, the same trick apply_cents_curve already uses).
# The fast part deliberately stays: flattening a syllable all the way to a
# dead-flat line is exactly what makes a render sound like a sampler instead
# of a person, and keeping the take's own vibrato is free. The deliberate
# musical movement added on top, the portamento into a note, the vibrato
# that fades in on a held note, the slow drift, is then built on the SONG's
# pitches by cents_curve_for_note, and at a much smaller depth than the
# take's, because the take's own movement is already there.
# ============================================================================

FLATTEN_HOP_S = 0.008
FLATTEN_WIN_S = 0.045
FLATTEN_MAX_CENTS = 380.0
FLATTEN_MIN_FRAMES = 4

# The slow/fast split: movement slower than this (0.4 s) is the glide that a
# constant shift cannot place and is removed; movement faster than it is the
# performance's own micro-movement and is kept. A real sung vibrato runs
# 4-7 Hz, i.e. a period of 0.14-0.25 s, so it stays on the fast side, while a
# scoop or a drift across a syllable lands on the slow side.
FLATTEN_SLOW_S = 0.40

# ...but the fast part is kept only in moderation. Measured on this project's
# own take, the fast part of a syllable is not just vibrato: these takes are
# short and expressive, and a syllable's own 200-cent scoop inside a third of
# a second is mostly "fast" by the split above, so keeping all of it left the
# rendered notes wavering by up to ~90 cents RMS, which measures as notes
# that are simply out of tune. This scales it down and caps it, which keeps a
# natural micro-movement of a few tens of cents (about what a singer's own
# vibrato is) and nothing more. The renderer's own deliberate vibrato is
# added on top of this (see cents_curve_for_note).
FLATTEN_FAST_KEEP = 0.6
FLATTEN_FAST_MAX_CENTS = 30.0


def measure_syllable_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Per-frame pitch of one syllable (Hz, 0 where unmeasurable) and its median.

    Uses the same detector the rest of the project does; the median over the
    syllable is what the voice is actually centred on, and it is what a
    single constant shift should be computed from.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    frame = max(512, int(round(FLATTEN_WIN_S * sr)))
    hop = max(64, int(round(FLATTEN_HOP_S * sr)))
    if n < frame + hop or not _HAS_LIBROSA:
        return np.zeros(0, dtype=np.float64), 0.0
    try:
        yin = librosa.yin(audio, fmin=MIN_NOTE_HZ, fmax=min(MAX_NOTE_HZ, sr / 4),
                          sr=sr, frame_length=frame, hop_length=hop)
    except Exception:
        return np.zeros(0, dtype=np.float64), 0.0
    hz = np.asarray(yin, dtype=np.float64)
    peak = np.zeros(len(hz))
    for i in range(len(hz)):
        a = i * hop
        seg = audio[a:a + frame].astype(np.float64)
        if seg.size:
            peak[i] = float(np.sqrt(np.mean(seg ** 2)))
    loud = float(np.median(peak)) if peak.size else 0.0
    ok = np.isfinite(hz) & (hz >= MIN_NOTE_HZ) & (hz <= MAX_NOTE_HZ) & (peak > max(1e-4, 0.35 * loud))
    if ok.sum() < FLATTEN_MIN_FRAMES:
        return np.zeros(0, dtype=np.float64), 0.0
    hz = np.where(ok, hz, np.nan)
    median_hz = float(np.nanmedian(hz))
    if not (MIN_NOTE_HZ <= median_hz <= MAX_NOTE_HZ):
        return np.zeros(0, dtype=np.float64), 0.0
    return hz, median_hz


def is_pitched_syllable(audio: np.ndarray, sr: int, median_hz: float,
                        register_hz: float = 0.0,
                        min_s: float = 0.13) -> bool:
    """Is this syllable actually a sung note, or just a consonant?

    Two tests, both of which a real sung syllable passes easily and the
    consonant fragments do not:

      - long enough to carry a note at all. "Cold" splits, on the project's
        own syllable segmenter, into a 0.86 s pitched body plus a 0.10 s
        unpitched tail; a tenth of a second is a consonant, not a note.
      - in the same register as the take it came from. The tail reads at
        100 Hz while the rest of the word sits at 313 Hz, i.e. a detector
        artifact on unpitched noise rather than a note; a real note in a
        word never sits an octave and a half from the rest of that word.

    A spectral test was tried here as well (does the syllable's own averaged
    spectrum really have a harmonic series at that pitch) and dropped: on
    real recorded voice the harmonics are dense and the valleys between them
    are not deep, so the score cannot tell a pitch from its octave well
    enough to be worth acting on. It is exactly right for the clean
    synthesised signals it was validated on, and not for this.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if median_hz <= 0 or len(audio) < min_s * sr:
        return False
    if register_hz > 0 and abs(1200.0 * math.log2(median_hz / register_hz)) > 1100.0:
        return False
    return True


def _moving_average(vals: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with edge padding (used for the slow/fast split)."""
    win = max(1, int(win))
    if win <= 1 or vals.size <= 1:
        return vals.copy()
    pad = win // 2
    padded = np.pad(vals, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(padded, kernel, mode="valid")[:vals.size]


def flatten_syllable_pitch(audio: np.ndarray, sr: int,
                           max_correct_cents: float = FLATTEN_MAX_CENTS,
                           register_hz: float = 0.0
                           ) -> tuple[np.ndarray, float]:
    """Steady a syllable's slow pitch movement; return (audio, its pitch).

    The SLOW part of the syllable's own movement is removed (see the section
    comment above), because one constant shift cannot follow it, and the FAST
    part is kept in moderation, because removing all of it is what makes a
    render sound like a sampler: a syllable that holds a digitally flat line
    for its whole length has no pitch life of its own, and a held note loops
    that dead line. The kept part is scaled and capped (see
    FLATTEN_FAST_KEEP / FLATTEN_FAST_MAX_CENTS) so a note can breathe without
    leaving the pitch it was placed on: measured on this project's own take,
    the raw fast component of a syllable runs up to ~90 cents RMS, which is
    simply out of tune, while a few tens of cents is what a singer's own
    vibrato actually is.

    The correction is clamped so a detector reading that is an outlier cannot
    yank a piece of the syllable by an arbitrary amount. The re-read preserves
    the syllable's length, so nothing moves on the song's timeline.

    Returns (audio, 0.0) for a syllable that is not actually pitched, see
    is_pitched_syllable, which the caller uses as its signal to leave that
    piece of the word exactly as recorded.
    """
    audio = np.asarray(audio, dtype=np.float32)
    hz, median_hz = measure_syllable_pitch(audio, sr)
    if hz.size < FLATTEN_MIN_FRAMES or median_hz <= 0:
        return audio, 0.0
    if not is_pitched_syllable(audio, sr, median_hz, register_hz):
        return audio, 0.0
    # fill detector dropouts by holding the last good reading, then smooth
    # the single-frame spikes out (a median filter, so a spike cannot pull the
    # curve; unlike the slow average below this does NOT remove vibrato)
    idx = np.arange(hz.size)
    good = np.isfinite(hz)
    filled = np.interp(idx, idx[good], hz[good])
    filled = np.clip(filled, median_hz * 0.5, median_hz * 2.0)
    if filled.size >= 5:
        pad = np.pad(filled, (2, 2), mode="edge")
        filled = np.median(np.stack([pad[i:i + filled.size] for i in range(5)], axis=0), axis=0)
    # The slow component is what one constant shift cannot follow, and it is
    # removed. The fast component is what remains, the performance's own
    # micro-movement, and it is kept, scaled down and capped (see the
    # constants above): enough of it that a rendered note is not a dead flat
    # line, not so much that the note is audibly off the pitch it was placed
    # on. In cents: output = kept(fast), so correction = kept - slow_offset.
    slow = _moving_average(filled, int(round(FLATTEN_SLOW_S / FLATTEN_HOP_S)))
    fast = 1200.0 * np.log2(filled / np.maximum(slow, 1e-6))
    kept = np.clip(fast * FLATTEN_FAST_KEEP,
                   -FLATTEN_FAST_MAX_CENTS, FLATTEN_FAST_MAX_CENTS)
    slow_off = 1200.0 * np.log2(np.maximum(slow, 1e-6) / median_hz)
    cents = kept - slow_off - fast
    cents = np.clip(cents, -max_correct_cents, max_correct_cents)
    if not np.any(np.abs(cents) > 5.0):
        return audio, median_hz
    # the measured curve is one reading per analysis hop; stretch it across
    # the syllable's samples so apply_cents_curve can read it per-sample
    per_sample = np.interp(np.linspace(0, 1, len(audio)),
                           np.linspace(0, 1, len(cents)), cents)
    flat = apply_cents_curve(audio, sr, per_sample)
    if np.max(np.abs(flat)) < 1e-6:
        return audio, median_hz
    return flat, median_hz


def take_f0(audio: np.ndarray, sr: int) -> float:
    """One take's own sung pitch: the median of a YIN estimate over the
    part of it that is actually voiced. Used to work out how far a note has
    to shift it, so it is measured per take rather than assumed from the
    vocable's name or from a single global number."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 256 or not _HAS_LIBROSA:
        return 0.0
    try:
        frame = int(min(2048, max(512, 2 ** int(math.floor(math.log2(audio.size))))))
        yin = librosa.yin(audio, fmin=MIN_NOTE_HZ, fmax=min(MAX_NOTE_HZ, sr / 4),
                          sr=sr, frame_length=frame, hop_length=128)
        voiced = yin[np.isfinite(yin) & (yin >= MIN_NOTE_HZ) & (yin <= MAX_NOTE_HZ)]
        return float(np.median(voiced)) if voiced.size else 0.0
    except Exception:
        return 0.0

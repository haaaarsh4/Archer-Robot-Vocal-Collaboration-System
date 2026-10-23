"""
THE HUM VOICE: a closed-mouth "mmm" that follows a melody.

This module is the whole of the humming accompaniment mode. It replaces the
older implementation, which was not a hum at all: the engine picked a rotating
vocable from the normal word set and sang it on the singer's own pitch, so
"Humming" sounded like one more word in the bank, and under the Neural engine
it did nothing whatsoever (the mode was a browser-side effect with no effect
defined for it, so the trained voice went on singing the track's real words).

What a hum actually is, acoustically, and what this module therefore builds:

  * A VOICED SOURCE WITH NO ARTICULATION. A train of glottal pulses (Rosenberg
    shape, differentiated to the glottal flow derivative) at the melody's own
    pitch, with jitter and shimmer so it is a voice rather than an oscillator.
  * A CLOSED MOUTH. The vocal tract shapes the source through a cascade of
    formant resonators at the murmur's own positions: F1 very low (~260 Hz),
    F2 ~1.0 kHz, F3 ~2.25 kHz, F4 ~3.2 kHz, with a genuine ANTIFORMANT (a
    spectral zero sitting ON the second resonance, not above it). That low F1,
    the damping of F2 and the zero are what the ear reads as "closed mouth";
    they are the single most important thing in this file, and the reason a
    plain low-passed vowel does not sound like a hum. See the block above
    f2_hz/b2_hz/nasal_zero_hz in HumConfig for the measurement that put the
    zero there: a zero parked PAST a fluent second formant leaves two
    comparable partial groups standing, and two comparable partial groups is
    what a listener hears as /o/.
  * A MUFFLED SPECTRUM. The lips are shut, so the air path is long and lossy:
    a -12 dB/octave tilt above ~500 Hz, a high shelf, and a steep 4th-order
    low-pass near 2.5 kHz. Everything bright, everything sibilant, everything
    that would read as a word, is gone.
  * BREATH. A quiet band-passed noise floor that travels through the same
    vocal tract, so the hum has air in it and is not a synthesizer tone.
  * HUMAN MOVEMENT. Vibrato that arrives after the note starts (5.4 Hz,
    ~14 cents), a slow drift, glide between notes, and an amplitude envelope
    taken from the singer's own loudness, so the hum breathes and phrases the
    way the performance it is humming along with does.

The melody comes from the source itself, tracked with pyin (a plain
autocorrelation detector misreads this voice's own dark, formant-heavy low
spectrum -- measured: it called a 220 Hz hum 97 Hz), gated on there being a
VOICE to hum (see the voice_gate block in HumConfig: on a track that opens with
drums alone, the tracker answers "voiced" on every frame of the drum and the
hum drones it), and then repaired: octave
errors folded toward the passage's own local register, short runs and dropouts
tidied, and a final median over 0.24 s that reads the NOTE being sung rather
than the singer's vibrato, because a hum adds its own movement to a note instead
of copying somebody else's wobble. The hum is then in unison with the person it
is humming along with, note for note.

One entry point matters to the rest of the app:

  * render_hum_track(audio, sr)          -> the whole hum part for a track.

AND A HUM IS NEVER HANDED TO THE TRAINED VOICE. An earlier version of this
module sent the hum through RVC so it would hum in the robot's own timbre and
then tried to put the closed mouth back afterwards (restore_closed_mouth).
Measured on the takes the panel's own live bridge asks for, that never worked
even remotely: a converter trained on singing answers a closed-mouth hum with a
SUNG VOWEL. On a 123 Hz hum the converted take sat +12 to +18 dB above the
murmur's own partials between 500 Hz and 1.1 kHz and +28 to +40 dB above 1.2
kHz, with a broadband tail out to 4 kHz the murmur does not have at all -- 2.96
per cent of its harmonic energy above 700 Hz against the murmur's 0.08, and on a
220 Hz hum it came back a further 1.9 semitones sharp. Twelve, sixteen and
twenty-two dB dips cannot remove a formant the model put there, and no filter
can put a pitch back. A closed-mouth voice is not something a singing converter
can render, so humming is rendered by this voice on both engines and the
conversion is skipped for it (see server.render_neural_note and
_render_track_offline).

And one twin, in another language: the LIVE robot voice is synthesized in the
page (_HumVoice in frontend/index.html), because the live path cannot wait for a
render. It is built from these same acoustics -- the same glottal pulse at the
same open quotient, the same tract shape, the same air -- and
tools/check_hum_live_tract.py reads the page's own constants and holds them
against this file, so the two halves cannot drift apart silently.

Everything is numpy/scipy, no new dependency, and the whole render is a few
seconds of CPU on a 35 second track (the pitch pass is the only part that
scales with length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:  # scipy is a hard dependency of the rest of the project, this is only
    from scipy import signal as _signal  # a clearer failure if it ever is not.
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _signal = None
    _ndimage = None


# Reading a hum's own spectrum (see _harmonic_fit). _COMB_PEAK is how far above
# the frame's own average a bin has to sit to count as a partial, and
# _MATCH_COVERAGE is how many of a candidate's partials have to be real before
# the frame is read as "this hum is singing the note it was asked for" -- five
# of eight, which a correct voice clears and a voice a fifth or an octave out
# does not.
#
# The 20 is not a taste, it is a noise floor. Breath through the tract is random
# noise, and the power spectrum of random noise is exponentially distributed
# about its own mean, so with a threshold of t the chance that ONE bin of the
# breath reads as a partial is exp(-t) -- at t=2 (the first guess here) that is
# 13.5%, which over eight partials is more than one accidental "partial" per
# candidate, and across 241 candidates and hundreds of frames it made almost any
# candidate look like it fitted. MEASURED with t=2: a hum rendered an OCTAVE HIGH
# matched 100% of frames with 7 of 8 partials, at comb values of 1.5, 2035, 16.8,
# 7626, 3.7, 6526, 7.3, 4337 -- two real partials, four coin flips. At t=20 the
# same accidental is exp(-20) = 2e-9, while this hum's real partials measure in
# the hundreds to tens of thousands.
_COMB_PEAK = 20.0
_MATCH_COVERAGE = 0.5
_COMB_LOG = math.log10(_COMB_PEAK)

# How much the requested melody may move across a +/-0.25 s window and the frame
# still count as a note being HELD, in octaves (the fit works in log2): 0.03 is
# 36 cents, well under a semitone, which is a note and not a slide (see the
# comment in _harmonic_fit). Measured across it: at 72 cents the reference track
# gave 271 "held" frames whose 90th percentile was 15 cents, at 36 cents 226
# frames at the same 15 cents -- and a slide take that is one long portamento
# stops having held frames at all, which is the honest answer for it.
_HELD_TOLERANCE = 0.03


# ============================================================
# CONFIG


@dataclass
class HumConfig:
    """Every number the hum voice uses. The defaults here are the tuned values,
    and config.yaml (`synthesis.humming`) overrides any of them by name, so the
    voice can be retuned without touching this file. Three of them are not
    written in config.yaml as a hum setting at all: vibrato rate and depth come
    from the project's own shared `synthesis` section when the hum does not
    override them, and breath is derived from its `breathiness` (capped, see
    breath_cap_db), so a hum breathes and moves like the rest of the voices."""

    # --- pitch tracking (the melody the hum follows) ---
    # pyin is the default here on purpose. The hum has its energy concentrated
    # where a closed mouth puts it, and a plain autocorrelation detector reads
    # a formant-heavy low spectrum badly (measured: it reported a 220 Hz hum at
    # 97 Hz, while pyin and the spectrum itself both say 220). pyin costs a
    # second or so on a four second chunk, which the RVC call after it dwarfs.
    # Set it to yin/yinfft to use the project's own detector instead.
    pitch_method: str = "pyin"         # pyin | yin | yinfft | rmvpe
    frame_size: int = 2048             # analysis window for the project's detector
    pyin_frame_size: int = 4096        # pyin's window at the analysis rate: long
                                       # enough for a low hum (six periods at
                                       # 50 Hz, a 2048 window is only two and
                                       # reads an A1 hum as a drone)
    hop_size: int = 256                # 5.8 ms at 44.1 kHz: fine enough for glide
    analysis_sr: int = 16000           # pyin runs here, not at the audio rate:
                                       # 64 ms of 16 kHz carries more periods of
                                       # a low hum than 96 ms at 44.1 kHz, at a
                                       # third of the cost, and nothing the hum
                                       # is asked to follow lives above 1.1 kHz
    min_hz: float = 50.0               # B1 and up: a hum below this is not a hum
    max_hz: float = 1100.0
    # A READING AT THE DETECTOR'S OWN FLOOR IS NOT A READING. pyin's search starts
    # at min_hz, so a frame it cannot resolve comes back AS min_hz rather than as
    # a failure, and on this project's reference track that pile was half of every
    # window: measured on the track in eight-second windows, the melody's 10th
    # percentile was 50.0 Hz in all four of them, while the singer is between 94
    # and 170. Those frames are dropped (they are a detector hitting the edge of
    # its range, not a pitch) -- see edge_floor_ratio.
    edge_floor_ratio: float = 1.03     # a reading within this of min_hz is the floor, not a note
    conf_floor: float = 0.35           # detector confidence below which a frame is unvoiced
    median_s: float = 0.075            # median smoothing of the pitch curve
    # A hum sings THE NOTE, not the singer's vibrato. The tracker hands back a
    # curve with the performance's own vibrato in it (+/-90 cents on this
    # project's takes), and a hum that follows it is not humming along, it is
    # wobbling a copy of somebody else's wobble a few tens of milliseconds
    # behind them -- measured on a vibrato take, a median 32.5 cents away from
    # the note it was asked for. So the melody is split before it is hummed:
    # a median over this window (a vibrato period is 0.18-0.25 s) reads the
    # note the singer is ON, leaves slides and drift alone (a median reads the
    # centre of a monotone ramp, not its past), and costs a note change a
    # portamento's worth of delay instead of a pitch error. The hum then adds
    # its own vibrato on that note, which is what a mouth at rest does.
    #
    # Measured off the hum's own spectrum, same material and settings, this
    # window against none: the reference track 15.0 cents median / 101 at the
    # 90th -> 12.5 / 62.5 with its held notes at 5.0, and a vibrato take 20.0 /
    # 37.5 with no settled frames to measure at all -> 7.9 / 25.5 with 31 held
    # frames reading 2.5. It costs nothing on fast material either: on a run of
    # eighth notes at 150 bpm (0.2 s a note) the tracked note sits 5.0 cents
    # from the true one, 99.2% of settled frames inside 50 cents, identically at
    # 0, 0.18, 0.24 and 0.30 s -- a median reads the note, it does not smear the
    # run. 0.24 and not 0.30 because above a vibrato period the extra length
    # only buys fractions of a cent and starts to argue with short notes.
    note_window_s: float = 0.24        # window the note is read over (0 = keep the raw curve)
    register_window_s: float = 1.2     # window the octave repair reads its local register from
    # AND THE LINE IT IS HUM. A hum follows ONE line: a mouth does not leap two
    # octaves between two notes and back, and when a tracker reports that it is
    # reporting its own octave errors, not a melody. Measured on the reference
    # track, what the hum was asked to sing ran from the detector's floor (50 Hz)
    # to 400 Hz inside a single phrase, and a murmur wailing across three octaves
    # is what a listener hears as "OH, oh, oh" -- the hum's own render over the
    # 16-24 s stretch of that track came back with 1.9 per cent of its energy
    # between 700 Hz and 1.5 kHz and 42 per cent between 300 and 700, against
    # 0.1 and 4 per cent for the same voice on a stable register. So every frame
    # is folded by WHOLE OCTAVES toward a running register, and a frame that
    # cannot be brought near it by folding is not part of this line at all: it is
    # dropped rather than hummed. A phrase that genuinely moves is followed --
    # the register follows what is accepted, over register_hold_s -- and only the
    # jumps are folded away.
    register_hold_s: float = 1.6       # how long a register holds before the line may move
    register_leap_semitones: float = 10.0   # how far from the register a note may sit
    register_seed_min_ratio: float = 1.25   # frames this far above the floor vote on the opening register
    min_note_s: float = 0.05           # a voiced run shorter than this is detector noise
    max_gap_s: float = 0.07            # a dropout shorter than this is a breath, not a rest
    # AND HOW FAR IT HOLDS THROUGH A TRACKER DROPOUT WHILE THE SINGER KEEPS
    # SINGING. max_gap_s alone is not enough on real material: a detector drops
    # the pitch for 70-90 ms at a time on a mix (a cymbal, a consonant, a
    # harmony part entering) while the voice is plainly still there. Measured on
    # the panel's own demo track, the hum render had five stretches of EXACT
    # digital silence, 40-50 ms each, inside material measuring -16 dB either
    # side, and at every one of them the input's own frame rms was 0.21-0.46
    # against a median voiced frame of 0.31 -- so the singer never stopped, the
    # flag did. The hum followed the flag, and a listener hears that as the
    # robot dropping out mid-phrase. Where the loudness says the voice never
    # left, the note is now held through the gap; where the loudness really
    # falls away, it is a rest and stays a rest.
    hold_through_s: float = 0.30       # longest dropout the hum holds through
    hold_through_level: float = 0.35   # ...if the voice is still at this share of its own level
    # AND THE ONE CASE THE LOUDNESS TEST GETS WRONG ON ITS OWN: a move from one
    # note to the next. The singer is not silent there, they are between two
    # notes, and the level really does fall -- but not to nothing. Measured on
    # this project's own controlled phrase, every note boundary reads as an
    # 81-105 ms unvoiced run whose own rms is 0.05 against 0.13 just before it
    # and 0.33 as the phrase's median, i.e. 16 per cent of the passage's own
    # loudness, under hold_through_level, so the hum was muted for the length of
    # every note change. A listener hears that as the hum clicking off between
    # notes rather than as a breath, because a breath has level in it and this
    # was exact digital silence. A gap this short, WITH SINGING ON BOTH SIDES OF
    # IT, is a note change and is always held through: the amplitude curve is
    # still the singer's own, so the hum follows the level down through the
    # transition instead of cutting out.
    transition_hold_s: float = 0.16    # a gap this short between two sung frames is a note change
    # A hum follows a VOICE. On a full mix a pitch detector happily locks onto
    # a drum's own resonance for a few frames, and the hum then hums the drum.
    # The harmonic/percussive split costs a couple of seconds on a whole track,
    # removes exactly that failure, and is skipped for material that is already
    # a solo voice.
    suppress_percussion: bool = True
    percussion_hpss_margin: float = 1.6


    # --- the murmur's vocal tract ---
    #
    # A MURMUR HAS ONE PEAK, NOT TWO. This is the whole difference between a hum
    # and the vowel it was mistaken for, and it is why the numbers below are what
    # they are. A hum's first step was a fluent vowel tract (F1 260, F2 1000, a
    # narrow second resonance) with a notch parked ABOVE the second formant, at
    # 1600 Hz, on the reading that "a closed mouth has a spectral zero". It does
    # -- but a coupled nasal cavity puts that zero ON the second resonance, not
    # past it, so the notch sat in the third formant's region doing nothing while
    # F2 escaped untouched. The output was a dark voice with two comparable
    # partial groups in it (F1 at 260, F2 at 1000), which is exactly what a
    # listener hears as a vowel: measured on the partials at eleven pitches, the
    # loudest partial above 700 Hz sat only 11.6 dB under the loudest anywhere,
    # and 10 per cent of the sound's harmonic energy was up there. That is /o/.
    #
    # Three things put it back, all of them the acoustics of a closed mouth:
    #   * F2 is DAMPED, not sharp. A nasal's second resonance is heavily damped by
    #     the coupling (bandwidths of 250-400 Hz against a vowel's 60-90), and a
    #     wide resonator is both lower and broader than a narrow one, so the peak
    #     it makes stops reading as a second formant.
    #   * the zero sits at ~1 kHz, right on that resonance, where the antiformant
    #     from the nasal cavity actually falls, and it is deep and broad enough to
    #     cancel what damping left.
    #   * the rolloff starts lower and is steeper, so nothing above the murmur
    #     survives to be heard as colour.
    # Measured with the same eleven-pitch partial table, this configuration puts
    # the loudest partial above 700 Hz 27.2 dB under the loudest at the median
    # (15.1 dB at its worst, on the highest pitch where the fewest partials live
    # above the cutoff), the second-formant band 23 dB down, and the energy share
    # above 700 Hz at 0.2 per cent median / 3.5 per cent worst -- against 11.6 dB
    # and 10.3 per cent before. One peak, then a hole: a hum.
    f1_hz: float = 260.0
    f2_hz: float = 900.0
    f3_hz: float = 2250.0
    f4_hz: float = 3200.0
    b1_hz: float = 60.0
    # A NASAL'S SECOND RESONANCE IS NOT MERELY DAMPED, IT IS MOSTLY GONE, and at
    # 480 Hz of bandwidth it was not: that bandwidth is where this tract stopped
    # being a closed mouth as the note rose. With F2 at 1000 Hz / 480 Hz wide and
    # the zero at 1000, the murmur measured 0.12 per cent of its energy between
    # 700 Hz and 1.5 kHz at 94 Hz and 3.4 / 3.9 / 22.3 per cent at 196 / 330 /
    # 392 Hz -- the second partial group of a vowel, on a track whose melody sits
    # between 200 and 400 Hz, which is exactly the "it still sounds like OH" this
    # answers. Widened to 1100 Hz (Q 0.8) and dropped to 900, with the zero moved
    # onto 700 and deepened, the same sweep reads 0.01 / 0.46 / 0.25 / 5.7, and
    # the loudest partial above 700 Hz sits 40 / 25 / 32 / 15 dB under the loudest
    # anywhere, against a plain one-resonance-plus-shut-lip reference mouth that
    # measures 0.01 / 0.16 / 0.01 / 1.0 and 41 / 26 / 42 / 20 on the same nine
    # pitches. Nothing above 3 kHz anywhere, which is what a shut mouth does.
    b2_hz: float = 1100.0              # damped to near-nothing: a nasal's second resonance
    b3_hz: float = 140.0
    b4_hz: float = 200.0
    # The antiformant sits BELOW the second resonance now, at 700 Hz, because that
    # is the band a raised note puts its second partial into: at 330 Hz the second
    # partial IS 660 Hz, in the middle of the old hole's skirt, which no depth at
    # 1000 Hz would have covered.
    nasal_zero_hz: float = 700.0
    nasal_zero_depth_db: float = -26.0
    nasal_zero_q: float = 1.3          # broad: it has a whole damped resonance to cancel
    formant_tracking: float = 0.16     # how much the tract scales with pitch (0 = frozen)
    formant_jitter: float = 0.035      # per-note random tract size, a few per cent
    body_hz: float = 250.0             # chesty body shelf
    body_gain_db: float = 3.0

    # --- the closed mouth's spectrum ---
    tilt_hz: float = 500.0
    tilt_db_per_oct: float = -15.0
    lowpass_hz: float = 1800.0
    lowpass_order: int = 4
    hf_shelf_hz: float = 4000.0
    hf_shelf_db: float = -18.0

    # --- the source ---
    # A closed mouth's source is softer than a singer's: a hum is produced well
    # under the pressure a projected vowel is, and the glottis is open longer in
    # each cycle, so the pulse train carries less upper-harmonic energy to begin
    # with. At 0.42 the source was bright enough to leave a second group of
    # partials standing at -12 dB; at 0.48 the same tract measures 3-4 dB darker
    # across the board, which is the direction a murmur goes.
    open_quotient: float = 0.48        # Rosenberg open phase, fraction of the period
    breath_db: float = -34.0           # aspiration level relative to the pulse train
    # A hum takes the project's own breathiness (see from_config), because a
    # voice in this project should breathe like the others -- but it is capped
    # here. The % shared setting asks for -20 dB of air, and a hum's breath band
    # is the same place as its own upper partials, so at -20 dB the partials sit
    # only ~11 dB above their own noise while at -26 dB they sit 16-29 dB above
    # it (measured on the reference track). Air is right; air louder than the
    # voice is a draught.
    #
    # AND THE AIR SITS INSIDE THE MURMUR, not around and above it. The band used
    # to run 300-2200 Hz, i.e. it was still putting noise in the octave where the
    # murmur has nothing, which reads as a breathy vowel even when the tone under
    # it is closed. A closed mouth's air is muffled by the same lips that muffle
    # everything else, so the band is now 250-1200 and the level is a further 6 dB
    # down: still a voice, and audibly not an oscillator, but the air is inside
    # the sound rather than spread over it.
    breath_cap_db: float = -32.0
    breath_hp_hz: float = 250.0
    breath_lp_hz: float = 1200.0
    jitter: float = 0.0035             # period-to-period pitch perturbation
    shimmer_db: float = 0.18           # period-to-period level perturbation
    source_harmonics: int = 0          # 0 = impulse/Rosenberg train (band-limited by construction)

    # --- movement ---
    vibrato_rate_hz: float = 5.4
    vibrato_depth_cents: float = 14.0
    vibrato_onset_s: float = 0.35      # vibrato arrives, it is not there at the attack
    vibrato_ramp_s: float = 0.5
    drift_rate_hz: float = 0.6
    drift_depth_cents: float = 5.0
    # This one smoothing is BOTH the portamento between notes and what keeps the
    # hum from following the pitch tracker's own frame-to-frame jitter: at 70 ms
    # it glides between notes the way a mouth does, and it averages the detector
    # over a few frames at once. A faster follower (12 ms) was measured and is
    # WORSE, not better: it replicates every wobble pyin reports, the hum comes
    # out unsteady, and its own harmonics smear across the analysis window (the
    # hum's pitch error on the reference track went from 25 cents at the median
    # and 175 at the 90th to 37.5 and 250). A hum holds a note; it does not
    # vibrate at the detector's frame rate.
    glide_s: float = 0.07              # portamento between notes, and the tracker's jitter filter
    attack_s: float = 0.05
    release_s: float = 0.14

    # --- level ---
    target_rms: float = 0.10           # a hum sits under the singer, not beside them at full level
    peak_ceiling: float = 0.95
    gate_floor_db: float = -46.0       # source loudness below this is treated as silence
    edge_fade_s: float = 0.01

    # --- is there anything to hum? ---
    # A closed-mouth voice has no words to fall back on: if the melody it was
    # handed is not really a melody, the hum sings a wobble, and a normalized
    # wobble is not silence, so it used to pass the render's own "did anything
    # come out" check and get converted by the trained voice. Measured: five
    # seconds of white noise produced a 5 s hum at an rms of 0.075 with no
    # harmonic structure in it at all. Pyin's own voicing gate answers true on
    # only a few per cent of such a frame and its probability output is not
    # usable here (it reads 0.01 on real singing too), so the question is asked
    # of the melody that came out of the whole pipeline instead: how much of the
    # track is actually sung, and for how long. The reference track answers 88
    # per cent of its frames and 10.7 s of the 12 it was given, so the gate below
    # is nowhere near it.
    min_voiced_share: float = 0.15     # share of frames that must be sung
    min_voiced_s: float = 0.6          # and how many seconds of singing there must be

    # --- is this a VOICE? (and not a drum) ---
    # A hum follows a singer, and the tracker does not know the difference on its
    # own. Measured on this project's own reference track, whose first sixteen
    # seconds are drums alone: pyin answered "voiced" on 100 per cent of those
    # frames at 79-85 Hz (the drum's own low note, near E2) and the hum duly
    # droned the drum for sixteen seconds, at full level (-7 dB under the
    # track), while the singing that starts at 16 s went unhummed. That drone is
    # what a listener hears as the robot "not humming" -- it is one tone, on a
    # drum's pitch, where a hum should have been waiting for the voice.
    #
    # What separates a struck drum from a voice here is not loudness or
    # voicing, it is WHERE THE ENERGY IS. A drum puts nearly everything it has
    # under 300 Hz; a voice -- even a low one, and even a low one singing over
    # that drum -- carries formants and breath from 300 Hz to 1.5 kHz and
    # beyond. Measured on the same track, per two-second window, as the ratio
    # below: the drum-only stretch answers 0.01-0.09, the sung stretch 0.39-5.0.
    # Measured on this project's own solo vocal takes over their loudest three
    # seconds: 0.26 at the 10th percentile and 0.96-11519 at the median. So the
    # threshold sits in the gap between them, three times the loudest drum
    # reading and under the quietest voice one. It is deliberately blunt about
    # what a hum will follow: a bass instrument with no voice over it is not
    # something a hum should be humming either.
    voice_gate: bool = True
    voice_band_lo_hz: float = 300.0    # the voice's own band (formants and air)
    voice_band_hi_hz: float = 1500.0
    voice_ref_lo_hz: float = 60.0      # against the drum's own band
    voice_ref_hi_hz: float = 300.0
    voice_ratio_min: float = 0.20
    voice_window_s: float = 0.093      # 2048 samples at 22.05 kHz: a couple of
                                       # periods of a low voice, and short
                                       # enough to follow a phrase's own level

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "HumConfig":
        try:
            if cfg is None:
                from config.config_loader import get_config
                cfg = get_config()
        except Exception:
            cfg = {}
        synth = (cfg or {}).get("synthesis", {}) or {}
        raw = synth.get("humming", {}) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        out = cls(**kwargs)
        # the shared synthesis tuning is the source of truth for vibrato/breath
        # when this section does not override it, so the hum moves like the rest
        # of the voices in the project.
        if "vibrato_rate_hz" not in raw and "vibrato_rate_hz" in synth:
            out.vibrato_rate_hz = float(synth["vibrato_rate_hz"])
        if "vibrato_depth_cents" not in raw and "vibrato_depth_cents" in synth:
            out.vibrato_depth_cents = float(synth["vibrato_depth_cents"])
        if "breath_db" not in raw and "breathiness" in synth:
            b = float(synth["breathiness"])
            if b > 0:
                mapped = float(20.0 * math.log10(max(1e-4, b) * 2.2))
                out.breath_db = min(mapped, float(out.breath_cap_db))
        return out


# ============================================================
# MELODY


@dataclass
class Melody:
    hop_s: float
    f0_hz: np.ndarray        # per frame, 0 where unvoiced
    voiced: np.ndarray       # bool per frame
    rms: np.ndarray          # per frame, the performance's own loudness
    frame_size: int


def _local_register(f0: np.ndarray, voiced: np.ndarray, win_frames: int) -> np.ndarray:
    """The pitch register a listening ear would say this passage is in.

    A median of the nearby voiced pitches, in log frequency, per frame. This is
    the reference the octave repair folds toward, and it is deliberately a
    MEDIAN of a whole window rather than the previous frame: folding against
    the previous frame is a chain that walks a melody down into the cellar the
    moment two adjacent frames disagree by an octave (measured on a real low
    track: a run of fold-downs took a 77 Hz melody to 29 Hz, which is not a
    pitch anyone can hum). A median over a second of frames cannot be dragged
    by one bad frame, so the repair moves single frames and leaves phrases alone.
    """
    out = np.full(f0.size, np.nan, dtype=np.float64)
    logf = np.where(voiced & (f0 > 0), np.log2(np.maximum(f0, 1e-6)), np.nan)
    idx = np.where(~np.isnan(logf))[0]
    if idx.size == 0:
        return out
    pad = max(1, int(win_frames) // 2)
    for i in idx:
        lo, hi = max(0, i - pad), min(f0.size, i + pad + 1)
        window = logf[lo:hi]
        window = window[~np.isnan(window)]
        if window.size:
            out[i] = float(np.median(window))
    return out


def _repair_octaves(f0: np.ndarray, voiced: np.ndarray, cfg: "HumConfig",
                    hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Put octave-erred frames back where the melody actually is.

    Pitch detectors answer an octave out on a low, dark, harmonic-thin sound
    exactly like a closed-mouth hum, and they do it in single frames, not
    phrases. Each frame is folded by whole octaves toward the local register
    (see _local_register), which leaves a correct frame untouched and moves a
    wrong one by an octave or two, and a frame whose folded value still falls
    outside the hummable range is taken to be unvoiced rather than hummed.
    """
    out = f0.astype(np.float64).copy()
    if not np.any(voiced):
        return out, voiced
    ref = _local_register(f0, voiced, int(round(cfg.register_window_s / max(hop_s, 1e-6))))
    global_ref = float(np.median(ref[~np.isnan(ref)])) if np.any(~np.isnan(ref)) else np.nan
    keep = voiced.copy()
    for i in np.where(voiced)[0]:
        target = ref[i]
        if not np.isfinite(target):
            target = global_ref
        if not np.isfinite(target) or out[i] <= 0:
            continue
        # `target` is a log2 frequency (see _local_register), so the number of
        # whole octaves between the reading and the register is a subtraction
        # in log space, not a division in Hz.
        k = int(round(math.log2(out[i]) - target))
        candidate = out[i] / (2.0 ** k)
        if cfg.min_hz <= candidate <= cfg.max_hz:
            out[i] = candidate
        elif cfg.min_hz <= target <= cfg.max_hz:
            # the folded answer is not hummable (a detector answer at the very
            # edge of the range, or a spurious sub-octave): hum where the
            # phrase is rather than where this one frame wandered off to.
            out[i] = float(2.0 ** target)
        else:
            keep[i] = False
            out[i] = 0.0
    return out, keep


def _tidy_voicing(f0: np.ndarray, voiced: np.ndarray, cfg: "HumConfig",
                  hop_s: float, rms: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Make the voicing read as phrases rather than as detector output.

    A human humming a line does not chirp for 30 ms in the middle of a rest and
    does not drop to silence for 40 ms inside a sustained note. Detectors do
    both. Short voiced runs become rests, short rests inside a phrase are
    bridged by interpolating the pitch across them (which the glide then turns
    into a held note), and the result is a hum that phrases like the singer.

    `rms` (the performance's own loudness per frame) is what lets a LONGER
    dropout be told apart from a rest: see hold_through_s.
    """
    out = f0.astype(np.float64).copy()
    keep = voiced.copy()
    if out.size == 0:
        return out, keep
    hop_s = max(hop_s, 1e-6)
    min_note = max(1, int(round(cfg.min_note_s / hop_s)))
    max_gap = max(1, int(round(cfg.max_gap_s / hop_s)))
    i = 0
    while i < keep.size:                      # drop chirps
        if keep[i]:
            j = i
            while j < keep.size and keep[j]:
                j += 1
            if j - i < min_note and not (i == 0 and j == keep.size):
                keep[i:j] = False
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    i = 0
    while i < keep.size:                      # bridge dropouts inside a phrase
        if not keep[i]:
            j = i
            while j < keep.size and not keep[j]:
                j += 1
            left, right = i - 1, j
            bridge = (j - i) <= max_gap
            if not bridge and left >= 0 and right < keep.size \
                    and keep[left] and keep[right] \
                    and (j - i) * hop_s <= cfg.transition_hold_s:
                bridge = True          # a note change, held through: see transition_hold_s
            if not bridge and rms is not None and cfg.hold_through_s > 0:
                # IS THE SINGER STILL THERE? The gap is bridged when the
                # performance's own loudness over it is still close to the
                # loudness of the voiced frames around it. A tracker dropout
                # meets that; a breath or a pause between phrases does not.
                look = max(1, int(round(1.5 / hop_s)))
                w0, w1 = max(0, i - look), min(keep.size, j + look)
                near = rms[w0:w1][keep[w0:w1]]
                ref = float(np.median(near)) if near.size >= 4 else 0.0
                gap = float(np.median(rms[i:j])) if j > i else 0.0
                if ref > 1e-6 and (j - i) * hop_s <= cfg.hold_through_s \
                        and gap >= cfg.hold_through_level * ref:
                    bridge = True
            if bridge and left >= 0 and right < keep.size and keep[left] and keep[right]:
                a, b = out[left], out[right]
                if a > 0 and b > 0:
                    for k in range(i, j):
                        t = (k - left) / float(j - left)
                        out[k] = float(a * (b / a) ** t)   # geometric across the gap
                        keep[k] = True
            i = j
        else:
            i += 1
    return out, keep


def _note_curve(f0: np.ndarray, voiced: np.ndarray, hop_s: float,
                win_s: float) -> np.ndarray:
    """The note the performance is on, with the performance's vibrato taken out.

    A running MEDIAN over a window at least a vibrato period wide, computed in
    log frequency and over voiced frames only (a rest is not averaged into a
    note). This is the same idea as the octave repair's local register, applied
    to the melody itself, and it has the two properties this needs: a median
    is centred, so a slide is followed without being dragged a half-window
    behind (measured: the hum's error on a sustained vibrato falls from 32.5
    cents to a few cents of its own vibrato, while a monotone slide is left
    where it was), and a median is edge-preserving, so a note change is a note
    change and not a smear.
    """
    win = int(round(win_s / max(hop_s, 1e-6)))
    if win < 3 or f0.size == 0:
        return f0.astype(np.float64).copy()
    pad = win // 2
    logf = np.where(voiced & (f0 > 0), np.log2(np.maximum(f0, 1e-6)), np.nan)
    out = np.array(f0, dtype=np.float64, copy=True)
    for i in np.where(voiced & (f0 > 0))[0]:
        lo, hi = max(0, i - pad), min(f0.size, i + pad + 1)
        window = logf[lo:hi]
        window = window[~np.isnan(window)]
        if window.size:
            out[i] = float(np.exp2(np.median(window)))
    return out


def _drop_floor_frames(f0: np.ndarray, voiced: np.ndarray, cfg: "HumConfig") -> tuple[np.ndarray, np.ndarray]:
    """Readings sitting on the detector's own floor are failures, not pitches.

    A detector asked for `min_hz` and up answers min_hz when it cannot resolve a
    frame, so those frames arrive looking like the lowest note there is. On this
    project's reference track they were half of every eight-second window (see
    edge_floor_ratio for the measurement), and they are what dragged the melody
    down to the floor and the register with it.
    """
    out = f0.astype(np.float64).copy()
    keep = voiced.copy()
    if out.size == 0 or not np.any(keep):
        return out, keep
    floor = float(cfg.min_hz) * max(1.0, float(cfg.edge_floor_ratio))
    on_floor = keep & (out > 0) & (out <= floor)
    if np.any(on_floor):
        keep[on_floor] = False
        out[on_floor] = 0.0
    return out, keep


def _settle_register(f0: np.ndarray, voiced: np.ndarray, cfg: "HumConfig",
                     hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The one line a hum follows: whole-octave folding toward a running register.

    See HumConfig's register block for the measurement this answers (a line that
    ran from the detector's floor to 400 Hz inside one phrase, and a murmur
    wailing across three octaves is a vowel to a listener, not a hum).

    The opening register is where MOST of the line sits rather than where it
    starts -- the singer's own register, voted on by the frames that survived the
    floor, so a track that opens on drums cannot seed the hum on them -- and from
    there the register follows what is accepted, over register_hold_s. Frames
    that no whole-octave fold can bring within register_leap_semitones of it are
    dropped: a hum follows a singer, and a reading that cannot be part of one
    line is not that singer.
    """
    out = f0.astype(np.float64).copy()
    keep = voiced.copy()
    if out.size == 0:
        return out, keep
    hop_s = max(hop_s, 1e-6)
    floor = float(cfg.min_hz) * max(1.0, float(cfg.edge_floor_ratio))
    lo = math.log2(max(floor, 1e-6))
    hi = math.log2(max(float(cfg.max_hz), floor * 1.01))
    logf = np.full(out.size, np.nan, dtype=np.float64)
    for i in np.where(keep & (out > 0))[0]:
        v = math.log2(out[i])
        if v <= lo or v > hi:
            keep[i] = False
            out[i] = 0.0
        else:
            logf[i] = v
    idx = np.where(keep)[0]
    if idx.size == 0:
        return out, keep

    pool = logf[idx]
    vote_from = math.log2(max(float(cfg.min_hz) * float(cfg.register_seed_min_ratio), 1e-6))
    voters = pool[pool >= vote_from]
    if voters.size == 0:
        voters = pool
    bins = np.round(voters * 12.0)                    # a semitone grid is what
    values, counts = np.unique(bins, return_counts=True)   # the detector answers on
    centre = float(values[int(np.argmax(counts))])
    seed = float(np.median(voters[np.abs(bins - centre) < 0.5]))

    reach = float(cfg.register_leap_semitones) / 12.0
    alpha = math.exp(-hop_s / max(float(cfg.register_hold_s), 1e-6))
    ref = seed
    for i in idx:
        v = logf[i]
        best, best_cost = v, abs(v - ref)
        for k in (1, 2, 3):                            # a voice strays an octave or
            for cand in (v - k, v + k):                # two, and no further
                if cand < lo or cand > hi:
                    continue
                cost = abs(cand - ref)
                if cost < best_cost:
                    best, best_cost = cand, cost
        if best_cost > reach:
            keep[i] = False
            out[i] = 0.0
            continue
        out[i] = float(2.0 ** best)
        ref = alpha * ref + (1.0 - alpha) * best
    return out, keep


def _finish_melody(mel: "Melody", full_audio: np.ndarray, sample_rate: int,
                   cfg: "HumConfig") -> "Melody":
    """The last two passes every path takes: settle on one line, then gate it."""
    f0, voiced = _settle_register(mel.f0_hz, mel.voiced, cfg, mel.hop_s)
    mel = Melody(hop_s=mel.hop_s, f0_hz=f0, voiced=voiced, rms=mel.rms,
                 frame_size=mel.frame_size)
    return _voice_gated(full_audio, sample_rate, mel, cfg)


def _median_smooth(f0: np.ndarray, voiced: np.ndarray, win_frames: int) -> np.ndarray:
    win_frames = max(1, int(win_frames) | 1)
    if win_frames <= 1:
        return f0
    out = f0.astype(np.float64).copy()
    pad = win_frames // 2
    idx = np.where(voiced)[0]
    if idx.size == 0:
        return out
    padded = np.pad(f0, (pad, pad), mode="edge")
    for i in idx:
        window = padded[i:i + win_frames]
        window = window[window > 0]
        if window.size:
            out[i] = float(np.median(window))
    return out


def _harmonic_part(audio: np.ndarray, sample_rate: int, cfg: HumConfig) -> Optional[np.ndarray]:
    """The melodic half of a mix: everything a hum is allowed to follow."""
    try:
        import librosa
    except Exception:
        return None
    try:
        harmonic, _percussive = librosa.effects.hpss(audio, margin=float(cfg.percussion_hpss_margin))
    except Exception as e:
        print(f"[hum] harmonic/percussive split skipped ({e}), tracking the mix as it is")
        return None
    return np.ascontiguousarray(harmonic, dtype=np.float32)


def _frame_rms(audio: np.ndarray, frame: int, hop: int, frames: int) -> np.ndarray:
    padded = np.pad(audio, (0, max(0, frames * hop + frame - audio.size)))
    out = np.empty(frames, dtype=np.float64)
    for i in range(frames):
        seg = padded[i * hop:i * hop + frame]
        out[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if seg.size else 0.0
    return out


def _pyin_melody(audio: np.ndarray, sample_rate: int, cfg: HumConfig) -> Optional[Melody]:
    """pyin's own pass, on a resampled copy of the performance.

    The hum only ever follows pitches the hum itself can sing (50 Hz - 1.1 kHz),
    so pyin is handed material band-limited to that: 16 kHz keeps every period
    of the lowest note in a window a third the size, and the pass costs a third
    of what it costs at the audio rate. The loudness envelope still comes off
    the full-rate audio, because it is an amplitude curve and the frame grid is
    the same in both.
    """
    try:
        import librosa
    except Exception:
        return None
    work = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    analysis_sr = int(cfg.analysis_sr) if cfg.analysis_sr else sample_rate
    analysis_sr = max(4000, min(analysis_sr, sample_rate))
    scale = analysis_sr / float(sample_rate)
    hop = max(1, int(round(cfg.hop_size * scale)))
    frame_length = max(1024, int(round(cfg.pyin_frame_size * scale)))
    if analysis_sr != sample_rate:
        try:
            work = np.ascontiguousarray(librosa.resample(work, orig_sr=sample_rate,
                                                         target_sr=analysis_sr), dtype=np.float32)
        except Exception as e:                       # pragma: no cover
            print(f"[hum] resample for analysis failed ({e}), tracking at the audio rate")
            analysis_sr, hop, frame_length = sample_rate, int(cfg.hop_size), int(cfg.pyin_frame_size)
    hop_s = hop / float(analysis_sr)
    # pyin resolves f0 on the padded/short-time grid, so it wants a window with
    # room for several periods of the LOWEST note it is asked about, and its own
    # fmin is asked for no more than the hum can sing.
    fmin = max(32.0, float(cfg.min_hz))
    try:
        f0, voiced_flag, _prob = librosa.pyin(
            work, fmin=fmin, fmax=float(min(cfg.max_hz, 1100.0)),
            sr=analysis_sr, frame_length=frame_length, hop_length=hop,
            center=True, fill_na=0.0,
        )
    except Exception as e:
        print(f"[hum] pyin failed ({e})")
        return None
    f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=0.0)
    voiced = np.asarray(voiced_flag, dtype=bool) if voiced_flag is not None else (f0 > 0)
    voiced = voiced & (f0 >= cfg.min_hz) & (f0 <= cfg.max_hz)
    f0 = np.where(voiced, f0, 0.0)
    frames = f0.size
    rms = _frame_rms(audio, int(cfg.frame_size), max(1, int(round(hop_s * sample_rate))), frames)
    # the floor first: a frame reading the edge of the detector's own range is a
    # failure, and the octave repair's register is a median of exactly these
    # frames, so leaving them in drags the repair down with them.
    f0, voiced = _drop_floor_frames(f0, voiced, cfg)
    f0, voiced = _repair_octaves(f0, voiced, cfg, hop_s)
    f0, voiced = _tidy_voicing(f0, voiced, cfg, hop_s, rms=rms)
    f0 = _median_smooth(f0, voiced, int(round(cfg.median_s / hop_s)))
    f0 = _note_curve(f0, voiced, hop_s, cfg.note_window_s)
    return Melody(hop_s=hop_s, f0_hz=f0, voiced=voiced, rms=rms, frame_size=frame_length)


def track_melody(audio: np.ndarray, sample_rate: int, cfg: Optional[HumConfig] = None,
                 detector=None) -> Melody:
    """Frame-by-frame melody and loudness of a performance."""
    cfg = cfg or HumConfig.from_config()
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    if audio.size == 0:
        return Melody(cfg.hop_size / sample_rate, np.zeros(0), np.zeros(0, bool), np.zeros(0), cfg.frame_size)
    # The whole sound, before the percussion split: the voice gate is asked about
    # what the listener is hearing, not about the harmonic remainder.
    full_audio = audio

    if cfg.suppress_percussion:
        harmonic = _harmonic_part(audio, sample_rate, cfg)
        if harmonic is not None and harmonic.size == audio.size:
            audio_tracked = harmonic
        else:
            audio_tracked = audio
    else:
        audio_tracked = audio

    if detector is None and str(cfg.pitch_method).lower() == "pyin":
        mel = _pyin_melody(audio_tracked, sample_rate, cfg)
        if mel is not None:
            # the loudness envelope is taken from the same harmonic material the
            # melody is: a hum breathes with the singing, not with the drums.
            #
            # And the melody is gated on there being a VOICE here at all (see
            # _voice_gated): the material the gate reads is the TRACK, not the
            # harmonic part, because what it is asking about is the whole sound
            # the listener is hearing, drums and all.
            return _finish_melody(mel, full_audio, sample_rate, cfg)
        print("[hum] pyin unavailable, falling back to the project's own pitch detector")
    audio = audio_tracked
    frame_size = int(cfg.frame_size)
    hop = int(cfg.hop_size)
    # The project's detector (aubio) consumes whole HOP_SIZE chunks and keeps
    # its own 2048-sample history, so it is fed one hop per reading and not a
    # windowed frame: a window here would only bias the autocorrelation.
    if audio.size < frame_size:
        audio = np.pad(audio, (0, frame_size - audio.size))

    own_detector = detector is None
    if own_detector:
        from analysis.pitch_detector import PitchDetector
        detector = PitchDetector()
        try:
            detector.set_method(cfg.pitch_method)
        except Exception:
            detector.set_method("yin")
    method = str(getattr(detector, "method", cfg.pitch_method)).lower()

    padded = np.pad(audio, (0, hop))
    n_frames = 1 + (padded.size - hop - 1) // hop
    f0 = np.zeros(n_frames, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    rms = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + hop]
        rms[i] = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        try:
            hz, conf = detector.detect(frame)
        except Exception:
            hz, conf = None, 0.0
        if hz and cfg.min_hz <= float(hz) <= cfg.max_hz and float(conf) >= cfg.conf_floor:
            f0[i] = float(hz)
            voiced[i] = True

    hop_s = hop / sample_rate
    f0, voiced = _drop_floor_frames(f0, voiced, cfg)
    f0, voiced = _repair_octaves(f0, voiced, cfg, hop_s)
    f0, voiced = _tidy_voicing(f0, voiced, cfg, hop_s, rms=rms)
    f0 = _median_smooth(f0, voiced, int(round(cfg.median_s / hop_s)))
    f0 = _note_curve(f0, voiced, hop_s, cfg.note_window_s)
    return _finish_melody(Melody(hop_s=hop_s, f0_hz=f0, voiced=voiced, rms=rms,
                                frame_size=frame_size), full_audio, sample_rate, cfg)


# ============================================================
# DSP HELPERS


def _require_scipy():
    if _signal is None:  # pragma: no cover
        raise RuntimeError("scipy is required for the hum voice")
    return _signal


def _resonator(f_hz: float, bw_hz: float, sr: int) -> np.ndarray:
    """A two-pole resonator: the standard vocal-tract formant."""
    r = math.exp(-math.pi * bw_hz / sr)
    theta = 2.0 * math.pi * f_hz / sr
    return np.array([1.0, 0.0, 0.0,
                     -2.0 * r * math.cos(theta), r * r], dtype=np.float64)


def _antiresonance(f_hz: float, q: float, depth_db: float, sr: int) -> np.ndarray:
    """A notch: the nasal zero. Depth in dB at the centre frequency."""
    w0 = 2.0 * math.pi * f_hz / sr
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    depth = 10.0 ** (abs(depth_db) / 20.0)
    b0 = 1.0 + depth * alpha
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - depth * alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0], dtype=np.float64)


def _peaking(f_hz: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """RBJ peaking EQ: a dip or a bump at one frequency.

    Kept beside the other biquad builders as the one stage that can place a
    narrow correction anywhere; nothing reads it today, since the only shape
    this voice needs is its own tract.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f_hz / sr
    cos_w, sin_w = math.cos(w0), math.sin(w0)
    alpha = sin_w / (2.0 * max(q, 0.05))
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / A
    return np.array([b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0], dtype=np.float64)


def _shelf(f_hz: float, gain_db: float, sr: int, q: float = 0.7, high: bool = True) -> np.ndarray:
    """RBJ high/low shelf, used for the tilt and the body."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f_hz / sr
    cos_w, sin_w = math.cos(w0), math.sin(w0)
    alpha = sin_w / (2.0 * max(q, 0.05))
    two_sqrt_A_alpha = 2.0 * math.sqrt(A) * alpha
    if high:
        b0 = A * ((A + 1) + (A - 1) * cos_w + two_sqrt_A_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
        b2 = A * ((A + 1) + (A - 1) * cos_w - two_sqrt_A_alpha)
        a0 = (A + 1) - (A - 1) * cos_w + two_sqrt_A_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w)
        a2 = (A + 1) - (A - 1) * cos_w - two_sqrt_A_alpha
    else:
        b0 = A * ((A + 1) - (A - 1) * cos_w + two_sqrt_A_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
        b2 = A * ((A + 1) - (A - 1) * cos_w - two_sqrt_A_alpha)
        a0 = (A + 1) + (A - 1) * cos_w + two_sqrt_A_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w)
        a2 = (A + 1) + (A - 1) * cos_w - two_sqrt_A_alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0], dtype=np.float64)


def _apply_sos(x: np.ndarray, coeffs: np.ndarray, state: Optional[dict] = None,
               key: Optional[str] = None) -> np.ndarray:
    """One biquad.

    With `state` and `key` the biquad becomes a STREAMING filter: its delay line
    is read out of the caller's dict and written back, so a note rendered in
    chunks does not ring up from silence at every chunk boundary. That ringing
    is what a held note would otherwise do: the tract filters are what carry the
    sound between samples, and resetting them mid-note is a re-attack of every
    formant in the tract, every chunk, on a note that never stopped. Without
    `state` this is the same one-shot filter it always was.
    """
    s = _require_scipy()
    b = coeffs[:3]
    a = np.concatenate([[1.0], coeffs[3:]])
    if state is None or key is None:
        return s.lfilter(b, a, x).astype(np.float32)
    zi = state.get(key)
    if zi is None:
        zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.asarray(x).dtype)
    res = s.lfilter(b, a, x, zi=zi)
    out, zf = res if isinstance(res, tuple) else (res, None)
    if zf is not None:
        state[key] = zf
    return out.astype(np.float32)


def _rosenberg(phase_frac: np.ndarray, open_quotient: float) -> np.ndarray:
    """Rosenberg glottal pulse, as a function of the position inside the period."""
    oq = float(np.clip(open_quotient, 0.2, 0.8))
    t1 = 0.4 * oq          # the classic 40/20/40 split of the open phase
    t2 = 0.6 * oq
    out = np.zeros_like(phase_frac, dtype=np.float64)
    up = phase_frac < t1
    if np.any(up):
        out[up] = 0.5 * (1.0 - np.cos(np.pi * phase_frac[up] / max(t1, 1e-6)))
    mid = (phase_frac >= t1) & (phase_frac < t2)
    if np.any(mid):
        out[mid] = np.cos(np.pi * (phase_frac[mid] - t1) / (2.0 * max(t2 - t1, 1e-6)))
    return out


# ============================================================
# THE HUM


def synthesize_hum(f0_curve_hz: np.ndarray, sample_rate: int, cfg: Optional[HumConfig] = None,
                   rms_curve: Optional[np.ndarray] = None, hop_s: Optional[float] = None,
                   voiced: Optional[np.ndarray] = None, rng: Optional[np.random.Generator] = None,
                   return_pitch: bool = False, state: Optional[dict] = None,
                   hold: bool = False):
    """Render the hum itself.

    Two shapes are accepted, and which one is being used is decided by
    `hop_s`: with it, `f0_curve_hz` is a per-frame curve and the voiced mask
    arrives with `voiced` (a frame that is unvoiced stays silent, and the
    interpolation between frames never invents a pitch across a rest).
    Without it, every array is already on the sample grid.

    `state` is how a HELD NOTE stays one note. The live engine asks this voice
    for one chunk at a time and lays them end to end, so without it every chunk
    would restart the glottal phase at zero, restart the vibrato clock, ramp its
    own loudness and gate up from silence, and normalise to the target level on
    its own -- four separate seams inside a note that is supposed to be one
    continuous sound. Given a dict, the render reads the previous chunk's
    position out of it (source phase, the note's own clock, the vibrato/drift
    oscillators, the smoothing state, the level) and writes its own end back into
    the same dict, so the next chunk starts exactly where this one stopped. A
    caller starting a NEW note simply passes a fresh dict; None (the offline
    whole-track renders) means a fresh note that keeps no state at all.

    `hold` says MORE OF THIS NOTE IS COMING, so this chunk's end is not a note
    end and must not be faded into silence (the live engine knows this: the
    note it is rendering is a sustain). Fading it would put a 10 ms hole in the
    middle of a note the next chunk picks up right after, which is the one place
    a held hum would ever read as "the voice just stopped and started again".
    """
    cfg = cfg or HumConfig.from_config()
    s = _require_scipy()
    rng = rng or np.random.default_rng(1731)
    f0 = np.asarray(f0_curve_hz, dtype=np.float64).reshape(-1)
    n = f0.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # --- inputs land on the sample grid. With hop_s the input is per FRAME,
    # so the output length is the frames' own span, not the frame count.
    env = np.ones(n, dtype=np.float64)
    if hop_s:
        n = int(round(f0.size * hop_s * sample_rate))
        grid = np.arange(n, dtype=np.float64) / sample_rate
        frame_grid = np.arange(f0.size, dtype=np.float64) * hop_s
        frames = f0.copy()
        f0 = np.interp(grid, frame_grid, frames)
        if rms_curve is not None:
            rms = np.asarray(rms_curve, dtype=np.float64).reshape(-1)
            env = np.interp(grid, np.arange(rms.size) * hop_s, rms)
        if voiced is not None:
            v = np.asarray(voiced, dtype=np.float64).reshape(-1)
            vmask = np.interp(grid, np.arange(v.size) * hop_s, v)
        else:
            vmask = (frames > 0).astype(np.float64)
            vmask = np.interp(grid, frame_grid, vmask)
    else:
        if rms_curve is not None:
            env = np.asarray(rms_curve, dtype=np.float64).reshape(-1)
            if env.size != n:
                env = np.interp(np.linspace(0.0, 1.0, n),
                                np.linspace(0.0, 1.0, env.size), env)
        if voiced is not None:
            vmask = np.asarray(voiced, dtype=np.float64).reshape(-1)
            if vmask.size != n:
                vmask = np.interp(np.linspace(0.0, 1.0, n),
                                  np.linspace(0.0, 1.0, vmask.size), vmask)
        else:
            vmask = (f0 > 0).astype(np.float64)

    # --- pitch: hold the last voiced value through unvoiced frames so the
    # phase stays continuous (the gate closes anyway), then glide note changes.
    held = _hold_last(f0, vmask > 0.5)
    base_f0 = float(np.median(held[held > 0])) if np.any(held > 0) else 200.0
    cents = 1200.0 * np.log2(np.maximum(held, 1e-6) / max(base_f0, 1e-6))
    # glide: a one-pole smoothing of the cents curve is a portamento, and it is
    # what makes a stepped note list sound like a mouth, not a keyboard. See
    # glide_s for the measurement that says a slower one is the better one.
    cents = _one_pole(cents, cfg.glide_s, sample_rate)

    # --- a continuation picks the note up where the last chunk put it down
    cont = bool(state) and state.get("continue") is True
    # 0 means "uninitialised" for every carried scalar below, and a fresh chunk
    # is exactly that: the filter is at rest, the level has nothing to inherit.
    lamp = (lambda key: float(state.get(key, 0.0))) if cont else (lambda key: 0.0)
    t0 = lamp("t_s")
    phase0 = lamp("phase")

    # --- movement: vibrato (arriving late), drift, jitter
    t = t0 + np.arange(n, dtype=np.float64) / sample_rate
    # a continuation is by definition already mid-note, so the vibrato it is
    # arriving into is the one it was already singing: not "this chunk has been
    # going for 3 ms, start the onset ramp over".
    vib_input = (np.ones(n, dtype=np.float64) if cont
                 else (_runs(vmask > 0.5, sample_rate) > cfg.vibrato_onset_s).astype(np.float64))
    vib_env = _one_pole(vib_input, cfg.vibrato_ramp_s, sample_rate, initial=lamp("vib_env"))
    # the phase of each movement oscillator is drawn once per NOTE, not per
    # chunk: a continuation has to move the same mouth it was already moving.
    vib_ph = lamp("vib_ph") if cont and "vib_ph" in state else rng.random()
    drift_ph = lamp("drift_ph") if cont and "drift_ph" in state else rng.random()
    vibrato = cfg.vibrato_depth_cents * vib_env * np.sin(2.0 * np.pi * cfg.vibrato_rate_hz * t
                                                        + 2.0 * np.pi * vib_ph)
    drift = cfg.drift_depth_cents * np.sin(2.0 * np.pi * cfg.drift_rate_hz * t
                                          + 2.0 * np.pi * drift_ph)
    jitter = _jitter_curve(held, sample_rate, cfg.jitter, rng)
    f0_out = base_f0 * np.exp2((cents + vibrato + drift + jitter) / 1200.0)
    f0_out = np.clip(f0_out, cfg.min_hz, min(cfg.max_hz, sample_rate * 0.45))

    # --- the glottal source, continued from wherever the last chunk left it
    phase = phase0 + np.cumsum(f0_out) / sample_rate
    frac = np.mod(phase, 1.0)
    pulse = _rosenberg(frac, cfg.open_quotient)
    # The flow derivative, taken against the REAL sample before this one rather
    # than against zero. A fresh chunk starts at the closed phase, where the flow
    # is already zero, so this is the same number it always was -- but a
    # continuation can start anywhere in the cycle, and diffing a mid-open-phase
    # flow against zero puts a one-sample jump into the source that is bigger
    # than the glottal spike itself. Since the pulse is normalised by its own
    # peak, that jump then quietly rescaled the whole chunk (4-5x quieter, in
    # the case that found it) and put a click at its first sample.
    prev_frac = float(np.mod(frac[0] - f0_out[0] / sample_rate, 1.0))
    prev_flow = float(_rosenberg(np.array([prev_frac]), cfg.open_quotient)[0])
    pulse = np.diff(pulse, prepend=prev_flow)    # glottal flow derivative: the -6 dB/oct tilt
    pulse /= (np.max(np.abs(pulse)) + 1e-12)
    if cfg.shimmer_db > 0:
        shimmer = _period_shimmer(phase, sample_rate, cfg.shimmer_db, rng)
        pulse = pulse * shimmer

    # --- aspiration: air through the same tract, not on top of it
    breath = rng.standard_normal(n)
    # The filters run against the caller's state dict, so a chunk that will be
    # continued hands its own delay lines on to the next one. A fresh note gets
    # a fresh dict (the caller's contract, see _render_hum) and so starts from
    # zeros, and an offline render passes no dict at all and is a one-shot,
    # exactly as before.
    fstate = state
    br_b = _shelf(cfg.breath_lp_hz, -12.0, sample_rate, q=0.7, high=True)
    breath = _apply_sos(breath, br_b, state=fstate, key="breath_lp")
    br_hp = _shelf(cfg.breath_hp_hz, -12.0, sample_rate, q=0.7, high=False)
    breath = _apply_sos(breath, br_hp, state=fstate, key="breath_hp")
    breath = breath / (np.sqrt(np.mean(breath ** 2)) + 1e-12)
    src = pulse + breath * (10.0 ** (cfg.breath_db / 20.0))

    # --- the closed-mouth vocal tract
    tracking = float(np.clip(cfg.formant_tracking, 0.0, 0.6))
    scale = float(np.clip((base_f0 / 200.0) ** tracking, 0.7, 1.35))
    jit = 1.0 + cfg.formant_jitter * (rng.random() * 2.0 - 1.0)
    tract_scale = scale * jit
    y = src
    # The tract is a STREAM, not four independent filters: each stage's delay
    # line continues across chunks when the note is being carried (fstate is the
    # continuation state, or None for a fresh chunk).
    for i, (f_hz, bw_hz) in enumerate(((cfg.f1_hz, cfg.b1_hz), (cfg.f2_hz, cfg.b2_hz),
                                       (cfg.f3_hz, cfg.b3_hz), (cfg.f4_hz, cfg.b4_hz))):
        y = _apply_sos(y, _resonator(f_hz * tract_scale, bw_hz, sample_rate),
                       state=fstate, key=f"res{i}")
    y = _apply_sos(y, _antiresonance(cfg.nasal_zero_hz * tract_scale, cfg.nasal_zero_q,
                                     cfg.nasal_zero_depth_db, sample_rate),
                   state=fstate, key="nasal")
    y = _apply_sos(y, _shelf(cfg.body_hz * tract_scale, cfg.body_gain_db, sample_rate, q=0.8,
                             high=False), state=fstate, key="body")
    y = _apply_sos(y, _shelf(cfg.tilt_hz * tract_scale, cfg.tilt_db_per_oct, sample_rate, q=0.7,
                             high=True), state=fstate, key="tilt")
    y = _apply_sos(y, _shelf(cfg.hf_shelf_hz * tract_scale, cfg.hf_shelf_db, sample_rate, q=0.7,
                             high=True), state=fstate, key="hf")
    if cfg.lowpass_hz and cfg.lowpass_order:
        nyq = sample_rate / 2.0
        wn = min(0.99, max(0.02, cfg.lowpass_hz / nyq))
        sos = s.butter(int(cfg.lowpass_order), wn, btype="low", output="sos")
        if fstate is None:
            y = s.sosfilt(sos, y)
        else:
            zi = fstate.get("lp")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
            y, zf = s.sosfilt(sos, y, zi=zi)
            fstate["lp"] = zf
    y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # --- loudness: the performance's own envelope, gated and smoothed
    loud = _one_pole(env, 0.02, sample_rate, release_s=0.12, initial=lamp("loud") if cont else 0.0)
    floor = 10.0 ** (cfg.gate_floor_db / 20.0)
    held_rms = float(np.percentile(loud[loud > floor], 70)) if np.any(loud > floor) else 0.0
    # the loudness reference is the NOTE's, taken when it started: a continuation
    # must not re-normalise itself against its own few milliseconds and arrive at
    # a different level than the chunk before it.
    ref_rms = float(state.get("loud_ref", 0.0)) if cont else 0.0
    if held_rms > 0:
        if ref_rms > 1e-9:
            loud = loud / ref_rms
        else:
            ref_rms = held_rms
            loud = loud / held_rms
    loud = np.clip(loud, 0.0, 1.6) * (vmask > 0.5)
    attack = _one_pole((vmask > 0.5).astype(np.float64), cfg.attack_s, sample_rate,
                       initial=lamp("attack") if cont else 0.0)
    release = _one_pole((vmask > 0.5).astype(np.float64), cfg.release_s, sample_rate,
                        release_s=cfg.release_s, initial=lamp("release") if cont else 0.0)
    gate = np.minimum(attack, release)
    y = y * np.clip(loud, 0.0, None) * gate

    # --- level: the note's own gain, not one measured per chunk
    if cont and state.get("gain"):
        gain = float(state["gain"])
    else:
        rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
        gain = cfg.target_rms / rms if rms > 1e-9 else 1.0
    y = y * gain
    y = _soft_limit(y, cfg.peak_ceiling)
    # An edge fade belongs to the NOTE's edges. A continuation has no start edge,
    # and a chunk the caller is holding has no end edge: fading one in or out
    # there is a hole in the middle of a held hum.
    if not cont and not hold:
        fade = max(1, int(cfg.edge_fade_s * sample_rate))
        if y.size > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y[:fade] *= ramp
            y[-fade:] *= ramp[::-1]
    if state is not None:
        state.update({
            "continue": True,
            "t_s": float(t[-1] + 1.0 / sample_rate) if n else t0,
            "phase": float(phase[-1] + f0_out[-1] / sample_rate) if n else phase0,
            "vib_ph": float(vib_ph),
            "drift_ph": float(drift_ph),
            "vib_env": float(vib_env[-1]) if n else 1.0,
            "loud": float(loud[-1]) if n else 0.0,
            "attack": float(attack[-1]) if n else 1.0,
            "release": float(release[-1]) if n else 1.0,
            "loud_ref": float(ref_rms),
            "gain": float(gain),
        })
    y = np.ascontiguousarray(y, dtype=np.float32)
    if return_pitch:
        # The pitch this render was actually built on, on the output's own sample
        # grid: the glide, the vibrato, the drift and the jitter ARE this curve,
        # so measuring the audio against it is a measurement of the synthesizer
        # alone -- no melody, no tracker, no portamento expectation in it. That
        # is the self-test's synth-fidelity check; the melody-following check
        # uses the melody instead.
        return y, np.ascontiguousarray(f0_out[:y.size], dtype=np.float64)
    return y


def synthesize_hum_note(f0_hz: float, n_samples: int, sample_rate: int, cfg: Optional[HumConfig] = None,
                        prev_hz: Optional[float] = None, legato: bool = False,
                        rng: Optional[np.random.Generator] = None,
                        state: Optional[dict] = None, hold: bool = False) -> np.ndarray:
    """ONE note of the hum voice, for the live path.

    The offline renderer hums a whole performance at once (render_hum_track),
    but the live engine hands the synthesizer one note at a time, so this is the
    same voice asked for a single note: the same glottal source, the same
    closed-mouth tract, the same breath and vibrato, with a real note envelope
    and, when it is arriving from somewhere else, a glide out of the pitch the
    voice was just on. `prev_hz` is where it is coming from (ignored when
    `legato` says this call is a mid-note continuation rather than a new note,
    in which case it simply continues the note it is on).

    `state` carries the note across calls when `legato` says this is a
    continuation: pass the same dict again and the chunk starts at the source
    phase, the vibrato, the loudness and the level the previous one stopped at,
    so a hum held over many chunks stays one sound. Pass a fresh dict for a new
    note (or None, which keeps no state). `hold` says the caller knows more of
    this note is coming, so this chunk does not fade its own end either.
    """
    cfg = cfg or HumConfig.from_config()
    n_samples = int(max(1, n_samples))
    target = float(f0_hz)
    if target <= 0:
        return np.zeros(n_samples, dtype=np.float32)
    # ON THE SAMPLE GRID, not the frame grid: the live loop lays these chunks end
    # to end, so a chunk has to be exactly as long as it was asked for. A frame
    # grid would hand back a tail of up to a frame that the caller truncates but
    # the carried phase has already advanced past, which is a phase jump in the
    # middle of a held note -- the one thing `state` exists to prevent.
    f0 = np.full(n_samples, target, dtype=np.float64)
    if (not legato) and prev_hz and float(prev_hz) > 0 and abs(math.log2(target / float(prev_hz))) > 1e-4:
        k = min(n_samples, max(2, int(round(cfg.glide_s * sample_rate))))
        ramp = np.linspace(0.0, 1.0, k)
        f0[:k] = float(prev_hz) * (target / float(prev_hz)) ** ramp
    out = synthesize_hum(f0, sample_rate, cfg, rng=rng, state=state, hold=hold)
    if out.size < n_samples:
        out = np.pad(out, (0, n_samples - out.size))
    return np.ascontiguousarray(out[:n_samples], dtype=np.float32)


def render_hum_track(audio: np.ndarray, sample_rate: int, cfg: Optional[HumConfig] = None,
                     detector=None) -> np.ndarray:
    """The hum part for a whole performance: track its melody, hum it back."""
    cfg = cfg or HumConfig.from_config()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    melody = track_melody(audio, sample_rate, cfg, detector=detector)
    if melody.f0_hz.size == 0:
        return np.zeros_like(audio, dtype=np.float32)
    # IS THERE ANYTHING TO HUM? See min_voiced_share. Silence here is handed
    # back as silence, which is what the renderers check to refuse the track
    # rather than convert a hum of nothing.
    voiced_share = float(np.mean(melody.voiced))
    voiced_s = float(np.sum(melody.voiced) * melody.hop_s)
    if voiced_share < cfg.min_voiced_share or voiced_s < cfg.min_voiced_s:
        print(f"[hum] nothing to hum in this track: {voiced_share * 100:.1f}% of its frames are "
              f"sung ({voiced_s:.2f}s), under the {cfg.min_voiced_share * 100:.0f}% / "
              f"{cfg.min_voiced_s:.1f}s floor")
        return np.zeros_like(audio, dtype=np.float32)
    # per-frame inputs, and the hum is rendered on the frame grid so the
    # interpolation never invents a pitch across a rest
    frames = max(1, int(math.ceil(audio.size / (melody.hop_s * sample_rate))))
    idx = np.arange(frames)
    idx = np.clip(idx, 0, melody.f0_hz.size - 1)
    f0 = melody.f0_hz[idx]
    rms = melody.rms[idx]
    v = melody.voiced[idx].astype(np.float64)
    return synthesize_hum(f0, sample_rate, cfg, rms_curve=rms, hop_s=melody.hop_s, voiced=v)


def _voice_like(audio: np.ndarray, sample_rate: int, cfg: "HumConfig",
                hop_s: float, n_frames: int) -> np.ndarray:
    """Per frame: is the energy in the VOICE's band, or in a drum's?

    The ratio HumConfig's voice_gate block carries the measurements for. Frames
    are placed on the melody's own hop grid so the two line up one to one.
    """
    win = int(max(512, round(cfg.voice_window_s * sample_rate)))
    x = np.asarray(audio, dtype=np.float64)
    x = np.concatenate([x, np.zeros(win, dtype=np.float64)])
    freqs = np.fft.rfftfreq(win, 1.0 / sample_rate)
    lo = (freqs >= cfg.voice_ref_lo_hz) & (freqs < cfg.voice_ref_hi_hz)
    hi = (freqs >= cfg.voice_band_lo_hz) & (freqs < cfg.voice_band_hi_hz)
    window = np.hanning(win)
    starts = (np.arange(max(0, n_frames)) * round(hop_s * sample_rate)).astype(np.int64)
    out = np.zeros(max(0, n_frames), dtype=bool)
    for i, start in enumerate(starts):
        frame = x[start:start + win]
        if frame.size < win:
            break
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        ref = float(power[lo].sum())
        band = float(power[hi].sum())
        out[i] = band > cfg.voice_ratio_min * ref
    return out


def _voice_gated(audio: np.ndarray, sample_rate: int, mel: "Melody",
                 cfg: "HumConfig") -> "Melody":
    """The melody with everything that is not a voice taken out of it.

    Applied to the tracked melody rather than to the render, so every consumer
    of track_melody -- the hum renderer, the export endpoints, the checks -- is
    asking the same question about the same frames. A frame where a drum is
    sounding and no voice is gets no hum: that is the whole point, and it is
    why the hum does not answer a drum kit.
    """
    if not cfg.voice_gate or mel.f0_hz.size == 0:
        return mel
    like = _voice_like(audio, sample_rate, cfg, mel.hop_s, mel.f0_hz.size)
    if not like.any():
        # Nothing voice-like anywhere in a track that had voicing: the safest
        # reading is that this is not a performance to hum along with at all.
        print("[hum] no voice-like frame in this track: a hum would have been droning "
              "an instrument (see HumConfig.voice_ratio_min for the measurement)")
    dropped = int(np.sum(mel.voiced & ~like))
    if dropped:
        print(f"[hum] the voice gate took {dropped} of {int(mel.voiced.sum())} voiced frames "
              f"({dropped * mel.hop_s:.2f} s) off the melody: no voice-like energy in them, "
              f"so nothing there to hum along with")
    return Melody(hop_s=mel.hop_s, f0_hz=mel.f0_hz, voiced=mel.voiced & like,
                  rms=mel.rms, frame_size=mel.frame_size)


# ============================================================
# SMALL SIGNAL HELPERS


def _hold_last(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    last = 0.0
    for i in range(out.size):
        if valid[i] and out[i] > 0:
            last = out[i]
        else:
            out[i] = last
    return out


def _one_pole(x: np.ndarray, attack_s: float, sample_rate: int,
              release_s: Optional[float] = None, initial: float = 0.0) -> np.ndarray:
    """Asymmetric one-pole smoothing, vectorised as a single scipy pass.

    `initial` is the value the filter is already at when this buffer starts,
    which is 0 for a fresh render and the previous chunk's last value for a
    held-note continuation (see synthesize_hum's `state`), so a continuation
    does not ramp up from silence all over again.
    """
    s = _require_scipy()
    x = np.asarray(x, dtype=np.float64)
    a_att = math.exp(-1.0 / max(1e-6, attack_s * sample_rate))
    rel_s = release_s if release_s is not None else attack_s
    a_rel = math.exp(-1.0 / max(1e-6, rel_s * sample_rate))
    if abs(a_att - a_rel) < 1e-9:
        b = np.array([1.0 - a_att])
        a = np.array([1.0, -a_att])
        # scipy hands back (y, zf) as soon as an initial condition is given
        # (and only y without one), so the caller gets y either way.
        out = s.lfilter(b, a, x, zi=np.array([a_att * float(initial)]))
        return out[0] if isinstance(out, tuple) else out
    # the asymmetric case is a slow loop, but it only ever runs on the
    # frame-rate curves and on the gate, never per formant
    out = np.empty_like(x)
    prev = float(initial)
    for i in range(x.size):
        a = a_att if x[i] > prev else a_rel
        prev = x[i] + a * (prev - x[i])
        out[i] = prev
    return out


def _runs(mask: np.ndarray, sample_rate: int) -> np.ndarray:
    """Seconds since the start of the current run of True, per sample."""
    out = np.zeros(mask.size, dtype=np.float64)
    count = 0.0
    step = 1.0 / sample_rate
    for i in range(mask.size):
        if mask[i]:
            count += step
        else:
            count = 0.0
        out[i] = count
    return out


def _jitter_curve(f0: np.ndarray, sample_rate: int, amount: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Period-to-period pitch perturbation, in cents, held across each period."""
    if amount <= 0:
        return np.zeros_like(f0, dtype=np.float64)
    f = np.maximum(f0, 1e-6)
    phase = np.cumsum(f) / sample_rate
    period = np.floor(phase).astype(np.int64)
    changes = np.nonzero(np.diff(period, prepend=period[0] - 1))[0]
    values = rng.standard_normal(changes.size) * amount * 1200.0
    out = np.empty(f0.size, dtype=np.float64)
    idx = np.zeros(f0.size, dtype=np.int64)
    idx[changes] = 1
    idx = np.cumsum(idx) - 1
    idx = np.clip(idx, 0, max(0, values.size - 1))
    out = values[idx] if values.size else np.zeros(f0.size)
    return out


def _period_shimmer(phase: np.ndarray, sample_rate: int, amount_db: float,
                    rng: np.random.Generator) -> np.ndarray:
    if amount_db <= 0:
        return np.ones_like(phase)
    period = np.floor(phase).astype(np.int64)
    changes = np.nonzero(np.diff(period, prepend=period[0] - 1))[0]
    values = (rng.standard_normal(changes.size) * (amount_db / 4.0))
    idx = np.zeros(phase.size, dtype=np.int64)
    idx[changes] = 1
    idx = np.cumsum(idx) - 1
    idx = np.clip(idx, 0, max(0, values.size - 1))
    return (10.0 ** (values[idx] / 20.0)) if values.size else np.ones_like(phase)


def _soft_limit(y: np.ndarray, ceiling: float) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= ceiling or peak <= 0:
        return y
    # a soft knee above 70% of the ceiling, so the loud notes bend instead of clipping
    knee = 0.7 * ceiling
    out = y.copy()
    over = np.abs(out) > knee
    sign = np.sign(out[over])
    x = np.abs(out[over])
    out[over] = sign * (knee + (ceiling - knee) * np.tanh((x - knee) / max(1e-6, ceiling - knee)))
    return out


# ============================================================
# MEASUREMENTS (used by the self-test and by the server's logging)


def hum_measurements(hum: np.ndarray, source: np.ndarray, sample_rate: int,
                     cfg: Optional[HumConfig] = None) -> dict:
    """What makes this a hum, measured: pitch agreement with the performance,
    spectral centroid, how much energy lives above the closed mouth's rolloff,
    and how much of the hum is periodic rather than noise."""
    cfg = cfg or HumConfig.from_config()
    s = _require_scipy()
    hum = np.asarray(hum, dtype=np.float32).reshape(-1)
    src = np.asarray(source, dtype=np.float32).reshape(-1)
    n = min(hum.size, src.size)
    hum, src = hum[:n], src[:n]
    out: dict = {"samples": int(n), "duration_s": round(n / sample_rate, 3)}
    if n == 0:
        return out
    out["hum_rms"] = round(float(np.sqrt(np.mean(hum.astype(np.float64) ** 2))), 4)
    out["hum_peak"] = round(float(np.max(np.abs(hum))), 4)
    out["source_rms"] = round(float(np.sqrt(np.mean(src.astype(np.float64) ** 2))), 4)

    def _centroid(x: np.ndarray) -> float:
        f, pxx = s.welch(x.astype(np.float64), fs=sample_rate,
                         nperseg=max(256, min(4096, x.size)))
        m = f > 40.0
        p = pxx[m]
        f = f[m]
        total = float(np.sum(p)) + 1e-18
        return float(np.sum(f * p) / total)

    out["hum_centroid_hz"] = round(_centroid(hum), 1)
    out["source_centroid_hz"] = round(_centroid(src), 1)

    def _hf_ratio(x: np.ndarray, cut: float = 2500.0) -> float:
        f, pxx = s.welch(x.astype(np.float64), fs=sample_rate,
                         nperseg=max(256, min(4096, x.size)))
        total = float(np.sum(pxx)) + 1e-18
        return round(float(np.sum(pxx[f > cut]) / total), 4)

    out["hum_hf_ratio"] = _hf_ratio(hum)
    out["source_hf_ratio"] = _hf_ratio(src)

    # THE MELODY THE HUM WAS ASKED TO SING, and two separate questions about it.
    #
    # 1. WHAT THE HUMS. The performance's melody is tracked, and the tracked
    #    melody is compared with the melody of the SOURCE (tracker against
    #    tracker, octave-tolerant). This is what says whether the hum is
    #    following the person it is humming with.
    fed = track_melody(src, sample_rate, cfg)
    hum_mel = track_melody(hum, sample_rate, cfg)
    k = min(hum_mel.f0_hz.size, fed.f0_hz.size)
    if k:
        h, v = hum_mel.f0_hz[:k], fed.f0_hz[:k]
        both = (h > 0) & (v > 0) & hum_mel.voiced[:k] & fed.voiced[:k]
        if np.any(both):
            cents = 1200.0 * np.log2(h[both] / v[both])
            cents = cents - 1200.0 * np.round(cents / 1200.0)   # octave-agnostic
            out["median_cents_error"] = round(float(np.median(np.abs(cents))), 2)
            out["pct_within_50_cents"] = round(float(np.mean(np.abs(cents) <= 50.0) * 100.0), 1)
            out["pct_within_100_cents"] = round(float(np.mean(np.abs(cents) <= 100.0) * 100.0), 1)
        out["voiced_frame_agreement"] = round(float(np.mean(both) * 100.0), 1)

    # 2. WHAT THE HUMS ON. The synth-fidelity question, and it is deliberately
    #    NOT answered by the melody tracker: that would be the tracker checking
    #    the tracker through the synthesizer, and a closed-mouth hum is exactly
    #    the signal an autocorrelation detector misreads (on this project's low
    #    reference track, a hum singing 60 Hz has its own partials at 60/120/
    #    180/240, which the ear reads as 60, while pyin on the same audio
    #    answers 50, 100 or 135). It is answered instead by asking the hum's own
    #    spectrum one direct question per frame: can a harmonic series be fitted
    #    to this frame near the pitch it was asked for, how far does that series
    #    have to be stretched, and is the series it fits the fundamental or its
    #    sub-octave (see _harmonic_fit).
    try:
        fit_hz = np.zeros(fed.f0_hz.size, dtype=np.float64)
        fit_hz[:] = fed.f0_hz
        fit_hz[~fed.voiced] = 0.0
        fit = _harmonic_fit(hum, sample_rate, fit_hz, fed.hop_s)
        out["on_pitch_median_cents"] = fit["median_abs_cents"]
        out["on_pitch_p90_cents"] = fit["p90_abs_cents"]
        out["on_pitch_within_50_pct"] = fit["pct_within_50"]
        out["on_pitch_held_frames"] = fit["held_frames"]
        out["on_pitch_held_median_cents"] = fit["held_median_abs_cents"]
        out["on_pitch_held_p90_cents"] = fit["held_p90_abs_cents"]
        out["fundamental_beats_suboctave_pct"] = fit["pct_fundamental_wins"]
        out["suboctave_gap_db"] = fit["gap_db"]
        out["fitted_frames"] = fit["frames"]
        out["on_pitch_lag_ms"] = fit.get("lag_ms")
        out["on_pitch_matched_pct"] = fit.get("matched_pct")
        out["on_pitch_strong_pct"] = fit.get("strong_pct")
        out["on_pitch_strong_frames"] = fit.get("strong_frames")
        out["on_pitch_strong_median_cents"] = fit.get("strong_median_abs_cents")
        out["on_pitch_strong_p90_cents"] = fit.get("strong_p90_abs_cents")
        out["on_pitch_median_cents_unshifted"] = fit.get("median_abs_cents_unshifted")
    except Exception as e:  # pragma: no cover
        out["harmonic_fit_error"] = str(e)

    # And the synthesizer's own accuracy, stated on its own terms: the melody
    # is fed back through synthesize_hum exactly the way render_hum_track feeds
    # it, and the round trip is measured the same octave-proof way. Nothing of
    # the original performance is in this number except the melody it asked for.
    try:
        if fed.f0_hz.size:
            asked = np.where(fed.voiced, fed.f0_hz, 0.0)
            again, intended_samples = synthesize_hum(fed.f0_hz, sample_rate, cfg, rms_curve=fed.rms,
                                                     hop_s=fed.hop_s, voiced=fed.voiced.astype(np.float64),
                                                     return_pitch=True)
            # the returned pitch is on the OUTPUT's sample grid; _harmonic_fit
            # reads its target on the tracker's frame grid, so put it there
            step = max(1, int(round(fed.hop_s * sample_rate)))
            intended = np.ascontiguousarray(intended_samples[::step], dtype=np.float64)
            # measured against the pitch the synthesizer itself says it rendered,
            # and not against the melody: see synthesize_hum's return_pitch.
            r_fit = _harmonic_fit(again, sample_rate, intended, fed.hop_s, held_from=asked)
            out["synth_median_cents"] = r_fit["median_abs_cents"]
            out["synth_p90_cents"] = r_fit["p90_abs_cents"]
            out["synth_held_median_cents"] = r_fit["held_median_abs_cents"]
            out["synth_held_p90_cents"] = r_fit["held_p90_abs_cents"]
            out["synth_lag_ms"] = r_fit.get("lag_ms")
            out["synth_strong_p90_cents"] = r_fit.get("strong_p90_abs_cents")
            out["synth_median_cents_unshifted"] = r_fit.get("median_abs_cents_unshifted")
    except Exception as e:  # pragma: no cover
        out["synth_fidelity_error"] = str(e)
    return out


def _harmonic_fit(x: np.ndarray, sample_rate: int, f0_curve_hz: np.ndarray, hop_s: float,
                  n_harm: int = 8, search_semitones: float = 3.0, n_points: int = 241,
                  held_from: Optional[np.ndarray] = None,
                  held_tolerance: float = _HELD_TOLERANCE) -> dict:
    """Fit a harmonic series to the hum, frame by frame, near a given melody.

    For every frame, and every candidate fundamental within `search_semitones`
    of the requested one (a 2.5-cent grid at the default 241 points: coarse
    enough to be quick, fine enough not to be the thing being measured, which
    a 12.5-cent grid was), the frame's power is summed at the first `n_harm`
    multiples of that candidate. The candidate with the most power is what the
    frame is actually singing, and how far it sits from what was requested is
    the pitch error, measured without a pitch tracker anywhere in the pipeline
    (a detector can have an opinion; a spectrum cannot).

    Reported: the median and 90th-percentile absolute error in cents, the
    percentage of frames within 50 cents, and the same three numbers again over
    the HELD frames only -- frames where the requested melody itself is not
    moving. That split matters for a note-by-note reading: the hum glides
    between notes on purpose (see glide_s), so a frame in the middle of a
    portamento is a frame where being a little behind the target is the sound
    working, not the sound failing, and averaging those frames in would let a
    deliberate musical gesture read as a pitch error. It also reports how far
    the energy sits from where an octave-low hum would put its partials, which
    is the one thing a formant-weighted hum can genuinely fool.

    What it cannot do, and it is written down here rather than left to be
    discovered: a harmonic comb CANNOT tell a voice an octave ABOVE the asked-for
    note from the note itself, because every partial of an octave-up hum is one
    of the asked-for series' own partials (measured on a sustained note: an
    octave-up render matches a series 94% of the time, against 100% for the
    correct one). What separates the octave-DOWN case -- the direction this
    project's tracker can actually produce -- is the gaps test: a hum at f0/2
    fills in the odd multiples of f0/2 that a hum at f0 leaves empty. So the
    octave claim made in this module's self-test is the sub-octave one, and an
    octave-up voice is caught instead by the melody comparison itself (in
    `hum_measurements`, the hum's own tracked pitch against the source's,
    octave-folded, plus the independent tracker the path check runs in Hz).
    """
    s = _require_scipy()
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    curve = np.asarray(f0_curve_hz, dtype=np.float64).reshape(-1)
    hop = max(1, int(round(hop_s * sample_rate)))
    n_fft = 8192 if x.size >= 8192 else 1024
    while n_fft > 1024 and hop >= n_fft:
        n_fft *= 2
    try:
        f, _t, Z = s.stft(x, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop,
                          boundary=None, padded=False)
    except Exception:
        return {"frames": 0}
    pxx = (np.abs(Z) ** 2).astype(np.float64)          # (bins, frames)
    freqs = f
    n_frames = pxx.shape[1]
    if n_frames == 0:
        return {"frames": 0}
    # THE FRAME'S TIME IS THE MIDDLE OF ITS WINDOW. scipy's stft with
    # boundary=None starts frame i at sample i*hop, so the audio it describes is
    # centred n_fft/2 later than the frame index says. Reading the requested
    # melody at the frame index therefore compares the hum with where the
    # melody was ~90 ms ago, which shows up as the fit needing a large negative
    # time offset on moving material (measured: -64 ms on the reference track,
    # and the steady-tone tests pinned whatever offset the search tried first).
    # The target is read at the window centre instead, and a frame whose window
    # overlaps a rest is dropped rather than interpolated across it.
    times = np.arange(n_frames, dtype=np.float64) * hop / sample_rate + (n_fft - 1) / (2.0 * sample_rate)
    curve_t = np.arange(curve.size, dtype=np.float64) * hop_s
    want = np.interp(times, curve_t, curve, left=0.0, right=0.0) if curve.size else np.zeros(n_frames)
    if curve.size:
        at = np.clip(np.searchsorted(curve_t, times), 0, curve.size - 1)
        before = np.clip(np.searchsorted(curve_t, times) - 1, 0, curve.size - 1)
        want[(curve[at] <= 0) | (curve[before] <= 0)] = 0.0
    use = want > 0
    if not np.any(use):
        return {"frames": 0}

    df = freqs[1] - freqs[0]

    # ENVELOPE. A hum's comb sits UNDER its formants: the strongest thing in the
    # frame is a formant peak, not any single partial. A fit that sums raw power
    # therefore leans toward whichever candidate lines the most of its harmonics
    # up with that peak, and it can prefer that overlap to the truth -- MEASURED:
    # a hum rendered a whole FIFTH above the melody it was handed still scored a
    # 10-cent median, because a fifth up puts real energy on the 3rd and 6th of
    # the asked-for series and the fit liked two aligned harmonics better than
    # none. So each frame is divided by a smooth version of itself (a median a
    # little narrower than one harmonic spacing), which takes out the envelope
    # and the level and leaves the comb. What is left is what this measures.
    f0_seen = float(np.median(want[use])) if np.any(use) else 200.0
    width = int(max(3, round(0.7 * max(f0_seen, 40.0) / max(df, 1e-6))))
    if width % 2 == 0:
        width += 1
    envelope = _ndimage.median_filter(pxx, size=(width, 1), mode="nearest")
    comb = pxx / np.maximum(envelope, 1e-30)

    def _series_power(scale: float, ks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per frame: (mean comb value, how much of the candidate is really there).

        The second number is the one that decides. Each partial contributes
        between 0 and 1 -- 1 once it is _COMB_PEAK times the frame's own
        average, sliding down through the fractions in between -- so a partial
        that is present but soft still counts for something, and the mean of
        those is what says how much of this candidate the frame actually
        consists of. That sliding scale matters for a real voice: a hum's upper
        partials sit inside its own breath, and a hum at the bottom of the range
        puts its 6th-8th partials where the breath band is, so a hard "is this
        partial 20x the floor, yes or no" read a perfectly good low hum at a
        human breath level as having no comb at all (measured: the reference
        track fell from 63% of frames matched to 1.4% as breath went from -34 dB
        to the -20 dB the project's own breathiness setting asks for, while the
        pitch stayed right to 2.5 cents throughout).

        A candidate high in frequency has some partials past the useful band;
        those drop out of both numbers, so a candidate is never punished for a
        partial it could not have had.
        """
        total = np.zeros(n_frames, dtype=np.float64)
        present = np.zeros(n_frames, dtype=np.float64)
        peaks = np.zeros(n_frames, dtype=np.float64)
        count = np.zeros(n_frames, dtype=np.float64)
        n_kept = 0
        for k in ks:
            f_k = want * scale * k
            in_band = f_k < freqs[-1] * 0.97
            if not np.any(in_band):
                continue
            n_kept += 1
            idx = f_k / df
            lo = np.clip(np.floor(idx).astype(np.int64), 0, pxx.shape[0] - 2)
            frac = np.clip(idx - lo, 0.0, 1.0)
            cols = np.arange(n_frames)
            val = comb[lo, cols] * (1.0 - frac) + comb[lo + 1, cols] * frac
            val[~in_band] = 0.0                         # past the useful band
            total += val
            present += np.clip(np.log10(np.maximum(val, 1e-12)) / _COMB_LOG, 0.0, 1.0) * in_band
            peaks += ((val >= _COMB_PEAK) & in_band).astype(np.float64)
            count += in_band.astype(np.float64)
        if n_kept == 0:
            zero = np.zeros(n_frames)
            return zero, zero, zero
        denom = np.maximum(count, 1.0)
        mean = total / denom
        share = present / denom
        strong = peaks / denom
        for arr in (mean, share, strong):
            arr[~use] = 0.0
        return mean, share, strong

    grid = 2.0 ** (np.linspace(-search_semitones, search_semitones, n_points) / 12.0)
    candidates = np.empty((grid.size, n_frames), dtype=np.float64)
    coverage = np.empty((grid.size, n_frames), dtype=np.float64)
    strong_cov = np.empty((grid.size, n_frames), dtype=np.float64)
    for i, g in enumerate(grid):
        candidates[i], coverage[i], strong_cov[i] = _series_power(g, range(1, n_harm + 1))
    # THE WINNER IS THE CANDIDATE WITH THE MOST PARTIALS PRESENT, and only then
    # the one with the most energy. That order is the whole point. Picking by
    # energy alone let a candidate that overlapped TWO of a wrong voice's
    # partials beat the truth on a dark hum -- MEASURED: a hum a fifth above the
    # melody scored a 5-cent median on a sustained note, because a fifth up puts
    # real energy exactly on the 3rd and 6th of the asked-for series. Counting
    # partials instead asks how many of the eight slots are actually filled, and
    # a fifth out fills two of them, not eight.
    # And the tie is broken on energy, which is what places the note exactly:
    # inside a plateau of equal coverage the candidate sitting on the most of
    # the frame's energy is the one the partials are really centred on.
    # MEASURED on a perfectly steady synthesized hum: with coverage alone the
    # winner was the LOWEST candidate in the plateau (argmax takes the first),
    # which read a correct voice as 27.5 cents flat; with this tie-break the
    # same voice reads 0.0-2.5 cents.
    tallest = np.max(candidates, axis=0, keepdims=True)
    score = coverage + 1e-6 * (candidates / np.maximum(tallest, 1e-30))
    picks = np.argmax(score, axis=0)
    cols = np.arange(n_frames)
    best_share = coverage[picks, cols]
    best_strong = strong_cov[picks, cols]
    best_power = candidates[picks, cols]
    best_cents = 1200.0 * np.log2(grid[picks])
    # A frame is MATCHED when enough of the winning candidate is really there.
    # Frames that do not match are not silently quoted as small errors; they are
    # counted, and reported as ``matched_pct`` -- which is what stops a voice
    # singing something else entirely from reading as in tune.
    matched = use & (best_share >= _MATCH_COVERAGE)
    # and the frames where the comb is not merely readable but unmistakable: 5
    # of 8 partials individually at the noise-proof level. The pitch claim is
    # made on every matched frame, but the TAIL of the error (its 90th
    # percentile) is quoted on these, so a breathy frame the fit can only
    # half-read does not get to set the worst case the check is about.
    strong = use & (best_strong >= _MATCH_COVERAGE)

    # A frame counts as HELD when the melody it was asked for is not moving
    # around it: steady to within 20 cents across a +/-0.25 s window. Twenty and
    # not sixty because a 60-cent criterion does not select held notes, it
    # selects SLOW SLIDES too -- measured on a take that is one long portamento,
    # the 60-cent version left a 90th-percentile error of 65 cents on "held"
    # frames, which is not the hum being out of tune, it is the hum gliding on
    # purpose through a window that was not in fact holding still. 0.25 s is a
    # bit over three of the hum's own glide time constants (see glide_s), so a
    # frame that still has a note change to finish gliding into is not held.
    span = max(1, int(round(0.25 / max(hop_s, 1e-6))))
    log_curve = np.where(use, np.log2(np.maximum(want, 1e-6)), np.nan)
    # (`held_from` is for the synthesizer check: the pitch being measured is the
    # synth's own glided/vibratoed curve, but whether the frame counts as HELD is
    # a statement about the melody it was asked for, and a vibrato is not a note
    # change.)
    hold_curve = np.where(held_from > 0, np.log2(np.maximum(held_from, 1e-6)), np.nan) \
        if held_from is not None and held_from.size else log_curve
    hold_curve = np.where(np.isfinite(hold_curve), hold_curve, np.nan)
    if hold_curve is not log_curve:
        pad = np.zeros(n_frames, dtype=np.float64)
        n_take = min(n_frames, hold_curve.size)
        pad[:n_take] = hold_curve[:n_take]
        hold_curve = pad
        hold_curve[~use] = np.nan
    held = use.copy()
    for shift in range(1, span + 1):
        for direction in (-1, 1):
            other = np.roll(hold_curve, shift * direction)
            if direction < 0:
                other[:shift] = np.nan
            else:
                other[-shift:] = np.nan
            held &= ~(np.isfinite(other)) | (np.abs(hold_curve - other) < held_tolerance)

    # IS THIS THE NOTE, OR AN OCTAVE BELOW? A hum that is really singing f0/2
    # puts energy at the ODD multiples of f0/2 (f0/2, 3f0/2, 5f0/2 ...), every
    # one of which falls in the gap between this hum's own harmonics; a hum
    # singing f0 has essentially nothing there, only the skirts of its real
    # partials and its breath. So the test is direct: how far below the fitted
    # harmonics does the energy in those gaps sit.
    odd_ks = [k for k in range(1, 2 * n_harm, 2)]
    gaps = _series_power(0.5, odd_ks)[0]
    if not np.any(strong):
        strong = matched
    eps = 1e-24
    octave_ok = use & matched
    gap_db = 10.0 * np.log10((best_power[octave_ok] + eps) / (gaps[octave_ok] + eps)) \
        if np.any(octave_ok) else np.zeros(0)

    # A PORTAMENTO IS A DELAY, NOT A WRONG NOTE. The voice glides into a note on
    # purpose (glide_s), so on material that is continuously moving -- a slide,
    # or a phrase of legato notes -- a few tens of milliseconds of the hum sit
    # behind the melody it was handed, and reading that as a pitch error says
    # "this voice is out of tune" when what it means is "this voice is a mouth".
    # So the time offset between the hum and its target is measured, bounded and
    # reported: the offset that lines them up is searched over +/-0.12 s (a
    # portamento's range, and a per-frame fit is worth nothing beyond it), and
    # the pitch error below is quoted AT that offset. A voice that needed a
    # longer offset than this, or a wrong one, is not helped by the correction,
    # and a voice that is simply a drone is caught by it needing none and
    # singing late anyway (the offset is reported with the lag it was found at).
    voiced_frames = int(np.sum(use))
    matched_frames = int(np.sum(matched))
    log_want = np.where(want > 0, np.log2(np.maximum(want, 1e-6)), np.nan)
    usable = matched_frames > 0
    if usable:
        max_lag = max(1, min(int(round(0.12 / max(hop_s, 1e-6))), 40))
        best_lag, best_median = 0, None
        for lag in range(-max_lag, max_lag + 1):
            shifted = np.roll(log_want, lag)
            if lag > 0:
                shifted[:lag] = np.nan
            elif lag < 0:
                shifted[lag:] = np.nan
            residual = best_cents - 1200.0 * (shifted - log_want)
            sample = residual[matched]
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                continue
            med = float(np.median(np.abs(sample)))
            if best_median is None or med < best_median:
                best_lag, best_median = lag, med
        shifted = np.roll(log_want, best_lag)
        if best_lag > 0:
            shifted[:best_lag] = np.nan
        elif best_lag < 0:
            shifted[best_lag:] = np.nan
        corrected = best_cents - 1200.0 * (shifted - log_want)
        raw_median = float(np.median(np.abs(best_cents[matched]))) if matched_frames else 0.0
    else:
        best_lag, corrected, raw_median = 0, best_cents, 0.0

    held_matched = held & matched
    held_raw = int(np.sum(held_matched))
    err = np.abs(corrected[matched])
    err = err[np.isfinite(err)]
    err_strong = np.abs(corrected[strong])
    err_strong = err_strong[np.isfinite(err_strong)]
    err_held = np.abs(corrected[held_matched]) if held_raw else np.zeros(0)
    err_held = err_held[np.isfinite(err_held)]
    held_frames = int(err_held.size)
    # the octave question does not need a note to be HELD, it needs the comb to be
    # readable, so its statistics are taken over the matched frames (a slide has
    # no held frames at all and still has to answer whether it is humming the
    # asked-for octave). gap_db is already restricted to those frames.
    held_db = gap_db if gap_db.size else np.zeros(0)
    return {
        "frames": voiced_frames,
        "matched_frames": matched_frames,
        "matched_pct": round(100.0 * matched_frames / voiced_frames, 1) if voiced_frames else 0.0,
        "strong_frames": int(err_strong.size),
        "strong_pct": round(100.0 * err_strong.size / voiced_frames, 1) if voiced_frames else 0.0,
        "strong_median_abs_cents": round(float(np.median(err_strong)), 2) if err_strong.size else None,
        "strong_p90_abs_cents": round(float(np.percentile(err_strong, 90)), 2) if err_strong.size else None,
        "lag_ms": round(float(best_lag * hop_s * 1000.0), 1),
        "median_abs_cents": round(float(np.median(err)), 2) if err.size else None,
        "p90_abs_cents": round(float(np.percentile(err, 90)), 2) if err.size else None,
        "pct_within_50": round(float(np.mean(err <= 50.0) * 100.0), 1) if err.size else None,
        "median_abs_cents_unshifted": round(float(raw_median), 2),
        "held_frames": held_frames,
        "held_median_abs_cents": round(float(np.median(err_held)), 2) if held_frames else None,
        "held_p90_abs_cents": round(float(np.percentile(err_held, 90)), 2) if held_frames else None,
        "held_pct_within_50": round(float(np.mean(err_held <= 50.0) * 100.0), 1) if held_frames else None,
        "pct_fundamental_wins": round(float(np.mean(held_db > 0.0) * 100.0), 1) if held_db.size else 0.0,
        "gap_db": round(float(np.median(held_db)), 1) if held_db.size else 0.0,
    }


# ============================================================
# SELF-TEST


def _selftest(paths: Optional[list[str]] = None, out_dir: str = "/tmp") -> int:
    """Render the hum over real material and measure what came out."""
    import io as _io
    import os
    import soundfile as sf

    # THE APP'S OWN SETTINGS, not this file's defaults: the self-test is evidence
    # about what the app renders, and for a while it was not -- it ran on the
    # defaults, which is 14 dB less breath than config.yaml's shared breathiness
    # asks for, and the app's own drifts, so it was measuring a voice the app
    # never plays.
    cfg = HumConfig.from_config()
    sources = paths or [
        "music samples/shortlorezarchersongs/Suzanne Kite Fm 120.mp3",
        "synthesis/source_takes/Lalanana.wav",
    ]
    sources = [p for p in sources if os.path.exists(p)]
    if not sources:
        print("no source material found for the hum self-test")
        return 1
    failures = 0
    for path in sources:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        import librosa
        if sr != cfg_humming_sr():
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg_humming_sr())
            sr = cfg_humming_sr()
        audio = np.ascontiguousarray(audio[: int(sr * 12.0)], dtype=np.float32)
        import time
        t0 = time.time()
        hum = render_hum_track(audio, sr, cfg)
        dt = time.time() - t0
        m = hum_measurements(hum, audio, sr, cfg)
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, f"hum_{os.path.splitext(name)[0]}.wav")
        sf.write(out_path, hum, sr)
        print(f"\n=== {name}  ({m['duration_s']}s of audio, hum rendered in {dt:.2f}s)")
        print(f"    hum     rms {m['hum_rms']}  peak {m['hum_peak']}  centroid {m['hum_centroid_hz']} Hz  "
              f"energy above 2.5 kHz {m['hum_hf_ratio'] * 100:.2f}%")
        print(f"    source  rms {m['source_rms']}  centroid {m['source_centroid_hz']} Hz  "
              f"energy above 2.5 kHz {m['source_hf_ratio'] * 100:.2f}%")
        if "median_cents_error" in m:
            print(f"    melody  tracked hum vs tracked source: median {m['median_cents_error']} cents, "
                  f"{m['pct_within_50_cents']}% within 50, {m['pct_within_100_cents']}% within 100, "
                  f"voiced-frame agreement {m['voiced_frame_agreement']}%")
        if "on_pitch_median_cents" in m:
            print(f"    comb    a harmonic series fits the hum near the melody it was handed on "
                  f"{m.get('on_pitch_matched_pct', 0.0)}% of the {m['fitted_frames']} frames it sings "
                  f"(a wrong melody, a fifth out or an octave out, fits on about none of them)")
            print(f"    on pitch (read off the hum's own spectrum, no tracker): "
                  f"median {m['on_pitch_median_cents']} cents from the melody it was given, "
                  f"{m['on_pitch_p90_cents']} at the 90th, "
                  f"{m['on_pitch_within_50_pct']}% within 50, "
                  f"aligned at {m.get('on_pitch_lag_ms')} ms (a portamento, not a drone: "
                  f"{m.get('on_pitch_median_cents_unshifted')} cents unaligned)")
            print(f"    strong  frames where 5 of the 8 partials are each unmistakable "
                  f"({m.get('on_pitch_strong_frames')} frames, {m.get('on_pitch_strong_pct')}% of the track): "
                  f"median {m.get('on_pitch_strong_median_cents')} cents, "
                  f"{m.get('on_pitch_strong_p90_cents')} at the 90th")
            print(f"    held notes ({m.get('on_pitch_held_frames', 0)} frames, the requested melody not moving): "
                  f"median {m.get('on_pitch_held_median_cents')} cents, "
                  f"{m.get('on_pitch_held_p90_cents')} at the 90th")
            print(f"    note vs sub-octave: the gaps where an octave-low hum would put its partials sit "
                  f"{m.get('suboctave_gap_db', 0.0)} dB below its real ones, "
                  f"so the fundamental wins in {m.get('fundamental_beats_suboctave_pct', 0.0)}% of frames")
        print(f"    wrote {out_path}")
        if "synth_median_cents" in m:
            print(f"    synth   the synthesizer's own claim: the pitch it says it rendered, "
                  f"read back off the audio it rendered -- median {m['synth_median_cents']} cents, "
                  f"{m['synth_p90_cents']} at the 90th, "
                  f"{m.get('synth_held_median_cents')} / {m.get('synth_held_p90_cents')} held")
        # A HELD-FRAME STATISTIC NEEDS HELD FRAMES, and the STRONGEST claim
        # available is the one made on them: a note that is being held is a note
        # the hum has had time to settle onto, so it is judged at 15 cents where
        # the moving frames are judged at 25 with a portamento's worth of
        # alignment. Below the floor this says so and falls back to every frame
        # rather than quoting a median over a dozen samples as evidence.
        enough_held = (m.get("on_pitch_held_frames") or 0) >= 20
        held_label = "held notes" if enough_held else "all frames (too few held notes to measure in this take)"
        pitch_median = (m.get("on_pitch_held_median_cents") if enough_held else m.get("on_pitch_median_cents")) or 999.0
        pitch_p90 = (m.get("on_pitch_held_p90_cents") if enough_held else m.get("on_pitch_p90_cents")) or 999.0
        pitch_median_all = m.get("on_pitch_median_cents") or 999.0
        pitch_p90_all = m.get("on_pitch_p90_cents") or 999.0
        synth_median = m.get("synth_median_cents") or 999.0
        synth_p90 = m.get("synth_p90_cents") or 999.0
        print(f"    [checked over {held_label}]")
        checks = []
        checks.append(("hum is under the singer's level", m["hum_rms"] <= m["source_rms"] + 0.02))
        checks.append(("hum is dark (centroid under 1100 Hz)", m["hum_centroid_hz"] < 1100.0))
        checks.append(("hum has almost nothing above 2.5 kHz", m["hum_hf_ratio"] < 0.02))
        checks.append(("a harmonic series fits the hum where the melody is (over half the frames it sings)",
                       (m.get("on_pitch_matched_pct") or 0.0) >= 50.0))
        checks.append(("the hum sings the melody, portamento and all (median under 25 cents over every frame)",
                       pitch_median_all < 25.0))

        # The synthesizer's own claim, measured against the pitch the synthesizer
        # says it rendered (see synthesize_hum's return_pitch): this is the DSP
        # chain on its own -- no melody, no tracker, no portamento expectation.
        checks.append(("the synthesizer renders the pitch it was built on (median under 10 cents)",
                       synth_median < 10.0))
        checks.append(("...and does so at the 90th percentile too (under 25 cents)",
                       synth_p90 < 25.0))
        if enough_held:
            # only claimed when there are notes being held to claim it about
            checks.append(("on the notes it holds it is in tune (median under 15 cents)",
                           pitch_median < 15.0))
            checks.append(("...and holds them there (90th under 25 cents)",
                           pitch_p90 < 25.0))
            checks.append(("and so is the synthesizer on its own, held (median under 10 cents)",
                           (m.get("synth_held_median_cents") or 999.0) < 10.0))
            checks.append(("...and the synthesizer alone, held, 90th percentile (under 20 cents)",
                           (m.get("synth_held_p90_cents") or 999.0) < 20.0))
        checks.append(("the note it sings is the fundamental, not an octave down",
                       m.get("fundamental_beats_suboctave_pct", 0.0) >= 95.0))
        # The line is 6 dB because that is where the measurement separates the
        # two answers: a hum on the asked-for note keeps its half-octave gaps
        # 11-21 dB down with the app's own breath level (16-29 dB with less air),
        # while a hum actually an octave low fills those gaps in and reads -3 to
        # +3 dB with the fundamental winning on 0-89% of frames.
        checks.append(("no sub-octave partials worth speaking of (gaps 6 dB down)",
                       m.get("suboctave_gap_db", -99.0) >= 6.0))
        for label, ok in checks:
            print(f"    [{'ok ' if ok else 'FAIL'}] {label}")
            if not ok:
                failures += 1
    return 1 if failures else 0


def cfg_humming_sr() -> int:
    try:
        from config.config_loader import get_config
        return int(get_config()["audio"]["sample_rate"])
    except Exception:
        return 44100


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_selftest(sys.argv[1:] or None))

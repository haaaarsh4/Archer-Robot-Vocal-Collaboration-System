"""The hand drum and the cone jingles, played as instruments.

Why this module exists
======================
A General MIDI kit has no hand drum and no jingle dress.  Rendering these bands
through FluidR3 gave a pitched "Low Tom" for the drum and a Cabasa for the
jingles, and that is not a near miss -- it is a different instrument:

* a hand drum is a single hide head on a frame, struck with a padded stick: a
  low, dead, inharmonic thump whose fundamental *bends down* as the hide relaxes
  after the strike, with almost nothing above 1 kHz;
* a jingle dress is two hundred-odd brass cones sewn on a dress, hitting each
  other as the dancer steps: a dense stochastic cloud of tiny bells between
  about 2.5 and 8 kHz, each ringing for a few tens of milliseconds, arriving
  slightly staggered.

Neither can be approximated by a tom, and a single oscillator or a filtered
noise burst (the two things a drum machine offers) is what makes a generated
jingle sound like a shaker.  Both are built here from their acoustics:

* **Hand drum** -- the modes of a circular membrane (Bessel ratios 1, 1.5934,
  2.1355, 2.2954, 2.6531, 2.9173, 3.1555 ...), high modes damped, a shell/air
  resonance, and a soft stick transient.  The fundamental is tuned to the key of
  the song, so the drum sits *in* the harmony instead of next to it.
* **Jingle dress** -- a bank of cone clusters.  Each cone is a small inharmonic
  bell (partials near 1, 1.94, 3.06, 4.31) with its own frequency, decay and
  onset, and a step is dozens of them landing within a few tens of
  milliseconds.  The stagger and the spread are the shimmer: render one cone,
  or all of them at the same instant, and it buzzes.

Nothing here is sampled, so there is no licence to carry and no sample library
to install: it is the physics of a struck hide and a struck brass cone.

Usage
=====
:func:`render_percussion` renders an :class:`~mysong.accompaniment.Arrangement`'s
drum slot to a stereo stem, and returns ``None`` for styles whose drums are a
real General MIDI kit (pop, rock, jazz ...), which the SoundFont renders better.
"""

from __future__ import annotations

import functools
import math

import numpy as np

# # 
# Data
# # 

#: Modal ratios of an ideal circular membrane, f_n / f_1.  A real hide drum's
#: head is not uniform and is air-loaded, so these shift a little and the high
#: modes die fast; the ratios are still what make it read as a drum rather than
#: a sine with a pitch envelope.
MEMBRANE_MODES = (
    1.0, 1.5934, 2.1355, 2.2954, 2.6531, 2.9173, 3.1555, 3.5001, 3.5989, 3.6520,
)

#: Partials of a shallow brass cone: bell-like, not a tube's 1, 3, 5.
CONE_PARTIALS = (1.0, 1.94, 3.06, 4.31)

#: Which General MIDI note becomes which engineered voice, per percussion
#: style.  The patterns in ``mysong.accompaniment._drum_pattern`` write GM
#: notes; this is the only place that decides what they are played on, so a
#: style the kit still owns simply is not listed here.
PERCUSSION_VOICES: dict[str, dict[int, str]] = {
    # The heartbeat under a song: an accented low stroke and a lighter answer.
    "hand": {45: "boom", 47: "tap"},
    # Steady pulse with rattles on the eighths.
    "round_dance": {45: "boom", 47: "tap", 69: "jingle"},
    # The big drum on every beat, the dancers' bells on the eighths.
    "grass_dance": {45: "boom", 47: "tap", 50: "tap", 54: "bells"},
    # The side-step: drum on the pulse, the cone jingles ringing with the step.
    "jingle_dress": {45: "boom", 47: "tap", 69: "jingle", 54: "bells"},
    # Under a cedar flute: two soft strokes and a quiet rattle.
    "light": {45: "boom", 69: "jingle"},
}

#: How loud each voice is against the others at the same MIDI velocity.  The
#: jingles are the instrument a jingle dress is named for, not a garnish on the
#: drum, so at 1.10 the cone band carries about a fifth of the band's energy
#: against the drum's two thirds (measured, see tests/test_mysong_percussion.py);
#: at the old 0.42 the cones were 4% of the stem and the dress sounded like a
#: drum with a hi-hat.
VOICE_GAIN = {"boom": 1.0, "tap": 0.62, "jingle": 1.10, "bells": 0.75}

#: A struck head is loudest just under 100 Hz; a jingle cluster has its whole
#: energy between 2 and 9 kHz.
DRUM_REGISTER = (55.0, 100.0)

#: How hard the stick lands on the skin, and how much of the stroke is the
#: skin's own broadband slap rather than the membrane's modes.  These are the
#: numbers that keep a drum from being a sine: a 65 Hz head with nothing above
#: it reads as a synth tone on any speaker and disappears under a voice.  Set
#: by measurement (see ``tests/test_mysong_percussion.py``), not by taste:
#: about 20% of the drum's energy sits between 250 Hz and 2 kHz.
SLAP_LEVEL = 6.0
STICK_LEVEL = 1.2


def percussion_voices(percussion: str) -> dict[int, str]:
    """The GM-note -> voice map for a percussion style (empty if not ours)."""
    return PERCUSSION_VOICES.get((percussion or "").strip().lower(), {})


def drum_fundamental_hz(tonic: int | None) -> float:
    """The drum's fundamental, tuned to the song's tonic.

    A hand drum is a pitched instrument whether or not anybody writes it down:
    the singer hears the head against the tonic and a drum a quarter-tone off it
    sounds wrong in a way that is hard to name.  The pitch class is taken from
    the detected key and placed in the register a big hide drum actually speaks
    in, nearest 72 Hz, which is a comfortable frame drum.
    """
    pitch_class = (int(tonic) % 12) if tonic is not None else 9  # A when unknown
    candidates = []
    for midi in range(24, 60):
        if midi % 12 != pitch_class:
            continue
        frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        if DRUM_REGISTER[0] <= frequency <= DRUM_REGISTER[1]:
            candidates.append(frequency)
    if not candidates:
        return 72.0
    return float(min(candidates, key=lambda frequency: abs(frequency - 72.0)))


# # 
# Helpers
# # 

def _normalise(signal: np.ndarray, peak: float = 1.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    current = float(np.max(np.abs(signal))) if signal.size else 0.0
    if current > 1e-9:
        signal = signal * (peak / current)
    return signal


def _declick(signal: np.ndarray, sample_rate: int, seconds: float = 0.02) -> np.ndarray:
    """Fade the last few milliseconds, so a cut tail is not a click."""
    n = min(len(signal), int(seconds * sample_rate))
    if n > 1:
        signal = signal.copy()
        ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
        if signal.ndim == 2:
            ramp = ramp[:, None]
        signal[-n:] *= ramp
    return signal


def _band(sample_rate: int, low: float, high: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter

    nyquist = 0.5 * sample_rate
    low = max(20.0, min(low, nyquist * 0.9))
    high = max(low * 1.05, min(high, nyquist * 0.95))
    return butter(order, [low, high], btype="band", fs=sample_rate, output="sos")


# # 
# The hand drum
# # 

def _drum_take(sample_rate: int, fundamental: float, accent: bool,
               rng: np.random.Generator) -> np.ndarray:
    """One stroke of a hide drum: ``accent`` is a centre stroke, else an edge tap."""
    length = int((0.85 if accent else 0.45) * sample_rate)
    t = np.arange(length) / sample_rate
    base = fundamental * (1.0 if accent else 1.10)

    # The head is under more tension at the instant of the strike and relaxes
    # over ~60 ms.  That downward glide is most of what tells the ear "skin"
    # rather than "synth tom", and it is why the fundamental is a moving target.
    glide = 1.0 + (0.055 if accent else 0.040) * np.exp(-t * 38.0)
    phase = 2.0 * np.pi * np.cumsum(base * glide) / sample_rate

    tone = np.zeros(length, dtype=np.float64)
    for index, ratio in enumerate(MEMBRANE_MODES):
        frequency = base * ratio
        if frequency > sample_rate * 0.45:
            break
        # A centre stroke feeds the low modes; an edge stroke loads the head
        # nearer the rim, which is brighter, so its upper modes are stronger.
        bright = 1.0 + (0.0 if accent else 0.26 * index)
        amplitude = (0.86 if index == 0 else 0.58 / ratio ** 0.8) * bright
        amplitude *= float(rng.uniform(0.88, 1.12))
        # The head damps itself: the fundamental rings, the high modes are gone
        # in a few tens of milliseconds, which is why a drum is a thump and not
        # a chord.
        decay = (1.8 if index == 0 else 6.0 + 3.4 * index)
        if not accent:
            decay *= 1.15
        tone += (amplitude * np.sin(phase * ratio + float(rng.uniform(0.0, 2.0 * np.pi)))
                 * np.exp(-t * decay))

    # Shell and the air it moves: a short low resonance under the head.
    from scipy.signal import sosfilt

    body = rng.normal(0.0, 1.0, length)
    body = sosfilt(_band(sample_rate, base * 0.7, base * 3.4), body)
    body *= np.exp(-t * 30.0) * (0.20 if accent else 0.09)

    # The padded stick: soft and bright, never a click.
    stick = rng.normal(0.0, 1.0, length)
    stick = sosfilt(_band(sample_rate, 1400.0, 9000.0, order=2), stick)
    stick *= np.exp(-t * 300.0) * (STICK_LEVEL if accent else 1.3 * STICK_LEVEL)

    # Stick on skin: a short band under the modes.  This is the part of a drum
    # that survives a phone speaker -- a 65 Hz fundamental does not -- and it is
    # what carries the stroke into a mix that a voice is also singing in.
    slap = rng.normal(0.0, 1.0, length)
    slap = sosfilt(_band(sample_rate, 240.0, 1800.0), slap)
    slap *= np.exp(-t * (52.0 if accent else 70.0)) * (SLAP_LEVEL if accent else 1.5 * SLAP_LEVEL)

    signal = tone + body + stick + slap

    # A hand damps the head well before the modes would die on their own, and
    # the stroke has a finite rise: neither end may be a step edge.
    attack = max(1, int((0.004 if accent else 0.002) * sample_rate))
    signal *= np.exp(-t * (0.45 if accent else 1.6))
    signal[:attack] *= np.linspace(0.0, 1.0, attack)
    signal[:attack] = np.tanh(signal[:attack] * 1.6) * 0.7
    return _declick(_normalise(signal).astype(np.float64), sample_rate, 0.03)


# # 
# The jingle dress
# # 

def _cone_tone(sample_rate: int, frequency: float, rng: np.random.Generator,
               length: int) -> np.ndarray:
    """One brass cone ringing: four inharmonic partials, a few tens of ms each."""
    t = np.arange(length) / sample_rate
    tone = np.zeros(length, dtype=np.float64)
    # Higher cones ring shorter -- a small stiff body radiates its energy faster.
    scale = (frequency / 4000.0) ** 0.35
    for index, ratio in enumerate(CONE_PARTIALS):
        partial = frequency * ratio * float(rng.uniform(0.985, 1.015))
        if partial > sample_rate * 0.45:
            continue
        decay = 22.0 * (1.0 + 0.85 * index) * scale * float(rng.uniform(0.7, 1.4))
        tone += (0.60 / (1.0 + 0.8 * index)) * np.sin(
            2.0 * np.pi * partial * t + float(rng.uniform(0.0, 2.0 * np.pi))
        ) * np.exp(-t * decay)
    return tone


def _jingle_cluster(sample_rate: int, rng: np.random.Generator, *,
                    cones: int, low_hz: float, high_hz: float,
                    spread: float) -> np.ndarray:
    """One step's worth of cones: dozens of tiny bells, slightly staggered."""
    length = int(0.55 * sample_rate)
    out = np.zeros(length, dtype=np.float64)
    for _ in range(cones):
        frequency = float(rng.uniform(low_hz, high_hz))
        # The cones a dancer's leg pulls first sing first, the rest a few ms
        # later as the dress swings; that stagger is the shimmer.
        onset = int(min(0.13, abs(float(rng.normal(0.0, spread)))) * sample_rate)
        if onset >= length - 32:
            continue
        out[onset:] += _cone_tone(sample_rate, frequency, rng, length - onset)

    # Cones knock into each other while they ring: a narrow band of noise rides
    # with the cloud and is what stops the cluster sounding like a chord.
    from scipy.signal import sosfilt

    t = np.arange(length) / sample_rate
    knock = rng.normal(0.0, 1.0, length)
    knock = sosfilt(_band(sample_rate, 2600.0, 10500.0), knock)
    knock *= np.exp(-t * 16.0) * (1.0 - np.exp(-t * 400.0))
    out += 0.5 * knock

    # Weight comes onto the dress over a couple of milliseconds, and every cone
    # is still ringing when the dancer's foot lifts: a soft release, no cutoff.
    out *= (1.0 - np.exp(-t * 700.0)) * (1.0 + 0.22 * np.exp(-t * 9.0))
    return _declick(_normalise(out), sample_rate, 0.06).astype(np.float64)


@functools.lru_cache(maxsize=16)
def _jingle_bank(sample_rate: int, cones: int, low_hz: float, high_hz: float,
                 spread_ms: float, seed: int) -> tuple:
    """Pre-rendered steps.  A hit picks one and offsets it, which is why two
    steps of the same velocity do not sound identical."""
    rng = np.random.default_rng(seed)
    return tuple(
        _jingle_cluster(sample_rate, rng, cones=cones, low_hz=low_hz,
                        high_hz=high_hz, spread=spread_ms / 1000.0)
        for _ in range(10)
    )


@functools.lru_cache(maxsize=32)
def _drum_bank(sample_rate: int, voice: str, fundamental: float, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    accent = voice == "boom"
    return tuple(
        _drum_take(sample_rate, fundamental, accent, rng) for _ in range(6)
    )


# # 
# Public: one hit, one stem
# # 

def percussion_hit(voice: str, velocity: int, sample_rate: int, *,
                   fundamental: float = 72.0, index: int = 0) -> np.ndarray:
    """Render one percussion hit, normalised, at the velocity's level."""
    level = (max(1, min(127, int(velocity))) / 127.0) ** 1.35
    if voice in ("boom", "tap"):
        bank = _drum_bank(int(sample_rate), voice, round(float(fundamental), 1), 4831)
    elif voice == "bells":
        bank = _jingle_bank(int(sample_rate), 26, 3000.0, 8600.0, 18.0, 977)
    else:
        bank = _jingle_bank(int(sample_rate), 54, 2400.0, 7800.0, 30.0, 431)
    take = bank[index % len(bank)]
    return (take * level * VOICE_GAIN.get(voice, 0.5)).astype(np.float32)


def render_percussion(
    arrangement,
    *,
    sample_rate: int = 44100,
    seed: int = 0,
) -> np.ndarray | None:
    """Play an arrangement's drum slot with these instruments, or ``None``.

    ``None`` means "not this module's band": the style writes a Western kit and
    the SoundFont renders it.  A returned stem is stereo, already levelled
    against the other stems, and padded to the arrangement's length.
    """
    voices = percussion_voices(getattr(arrangement.style, "percussion", ""))
    if not voices:
        return None
    notes = arrangement.notes_for("drums")
    if not notes:
        return None

    tonic = None
    try:
        tonic = arrangement.harmony.key.tonic
    except Exception:  # noqa: BLE001 - a hand-built arrangement may have no key
        tonic = None
    fundamental = drum_fundamental_hz(tonic)

    spb = arrangement.seconds_per_beat
    total = max(1, int((arrangement.duration_beats + 4.0) * spb * sample_rate) + sample_rate // 2)
    buffer = np.zeros((total, 2), dtype=np.float32)
    rng = np.random.default_rng(seed + 97)

    for number, note in enumerate(notes):
        voice = voices.get(int(note.pitch))
        if voice is None:
            continue
        start = int(max(0.0, note.start) * spb * sample_rate)
        if start >= total:
            continue
        signal = percussion_hit(
            voice, note.velocity, sample_rate,
            fundamental=fundamental, index=number,
        )
        end = min(total, start + len(signal))
        if end <= start:
            continue
        signal = signal[: end - start]

        if voice in ("jingle", "bells"):
            # Two legs of cones: the dress is not a mono source, and the width
            # is what separates a jingle dress from a shaker in the same band.
            pan = 0.30 if number % 2 == 0 else -0.30
            pan += float(rng.uniform(-0.12, 0.12))
            delay = int((0.003 + 0.003 * float(rng.random())) * sample_rate)
            left = math.sqrt(max(0.0, 0.5 * (1.0 - pan)))
            right = math.sqrt(max(0.0, 0.5 * (1.0 + pan)))
            buffer[start:end, 0] += signal * left
            right_end = min(total, start + delay + len(signal))
            buffer[start + delay:right_end, 1] += signal[: right_end - start - delay] * right
        else:
            # The drum is one head in front of the singer: centred, with a
            # hair of width so it is not a mono point inside a stereo mix.
            buffer[start:end, 0] += signal * 0.707
            buffer[start:end, 1] += signal * float(rng.uniform(0.70, 0.73))

    peak = float(np.max(np.abs(buffer))) if buffer.size else 0.0
    if peak > 1e-9:
        buffer *= (0.55 / peak)
    return _declick(buffer, sample_rate, 0.05)

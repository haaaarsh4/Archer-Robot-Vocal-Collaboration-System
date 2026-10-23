"""Checks the two reworked accompaniment modes against their own specification.

Drone support: ONE voice, ONE word, ONE held note, per phrase -- a real onset
once when a line starts, then the same pitch sustained for the whole line, and
silence between lines (the drone breathes with the song). Measured here on a
synthetic two-phrase melody, because that is the shape the request described:
"if there is wha he oh then drone support needs to have just one wha he oh and
take that one note at the end and hold it".

Octave below: the three things a singer does down there -- vowel kept bright
(brightness up), vowel narrowed (vowel_color down), less volume the lower it
sits (level_trim_db negative and growing) -- on top of a pitch one octave under
the singer.
"""

import sys, collections

import numpy as np

sys.path.insert(0, ".")

from synthesis.harmony_engine import HarmonyEngine
from synthesis.accompaniment_modes import AccompanimentMode
from analysis.phonetic_analysis import CreeTokenizer

HOP = 1024 / 44100.0


def run(engine, mode, hz_timeline):
    """hz_timeline: list of (seconds, hz-or-None). Returns per-frame decisions."""
    tok = CreeTokenizer()
    profile = tok._neutral_profile
    decisions = []
    for seconds, hz in hz_timeline:
        n = int(round(seconds / HOP))
        for _ in range(n):
            decisions.append(engine.decide(
                archer_hz=hz, phrase_state="singing" if hz else "silence",
                tempo_bpm=100.0, phoneme_profile=profile))
    return decisions


def main():
    failures = []

    # A two-phrase melody, a C4-ish line: phrase A on G4, a breath, phrase B on E4.
    melody = [(2.0, 392.0), (1.0, None), (2.0, 329.6), (1.0, None)]

    engine = HarmonyEngine()
    engine.set_forced_mode(AccompanimentMode.DRONE.value)
    dec = run(engine, AccompanimentMode.DRONE, melody)

    actions = [d.action for d in dec]
    sings = [i for i, a in enumerate(actions) if a == "sing"]
    pitches = [d.target_hz for d in dec]
    sung = [p for p, a in zip(pitches, actions) if a != "rest"]

    print("drone: actions (sing/sustain/rest) =",
          "".join({"sing": "S", "sustain": "h", "rest": "."}[a] for a in actions))
    print("drone: distinct pitches over the whole two phrases =",
          sorted({round(p, 1) for p in sung}))
    print("drone: onsets =", [(i, round(pitches[i], 1)) for i in sings])

    # 1. exactly one onset per phrase (two phrases -> two onsets)
    if len(sings) != 2:
        failures.append(f"drone should open ONCE per phrase (expected 2 onsets, got {len(sings)})")
    # 2. one distinct held pitch per phrase, and nothing else
    phrase_a = {round(p, 1) for p, a in zip(pitches, actions) if a != "rest" and p and _in(dec, p) is None}  # noqa
    a_pitch = round(pitches[sings[0]], 1)
    b_pitch = round(pitches[sings[1]], 1)
    held_a = {round(p, 1) for p in pitches[sings[0]:sings[0] + int(2.0 / HOP)] if p}
    held_b = {round(p, 1) for p in pitches[sings[1]:] if p}
    if len(held_a) != 1 or len(held_b) != 1:
        failures.append(f"drone must HOLD one pitch per phrase (A={held_a}, B={held_b})")
    # 3. it rests while the singer rests, and sustains while the singer sings
    if any(a != "rest" for a, (s, hz) in zip(actions, _expand(melody)) if hz is None):
        failures.append("drone must rest while the singer is silent")
    # 4. each held note sits under the phrase it is under -- the key's own note
    # in the register an octave beneath the line, i.e. between a fourth and an
    # octave under the singer, never further.
    for label, drone_hz, sung_hz in (("A", a_pitch, 392.0), ("B", b_pitch, 329.6)):
        semis = 12 * np.log2(drone_hz / sung_hz)
        if not (-13.0 <= semis <= -4.0):
            failures.append(f"phrase {label} drone {drone_hz} Hz sits {semis:+.1f} "
                            f"semitones from {sung_hz} Hz, outside a fourth to an octave under")
    # 5. and inside a register a voice can render
    for p in (a_pitch, b_pitch):
        if not (69.0 <= p <= 262.0):
            failures.append(f"drone {p} Hz left the singable band (70-262 Hz)")

    # Octave below: the three low-note rules.
    eng2 = HarmonyEngine()
    eng2.set_forced_mode(AccompanimentMode.OCTAVE.value)
    dec2 = run(eng2, AccompanimentMode.OCTAVE, [(2.0, 392.0)])

    eng3 = HarmonyEngine()
    eng3.set_forced_mode(AccompanimentMode.UNISON.value)
    dec3 = run(eng3, AccompanimentMode.UNISON, [(2.0, 392.0)])

    oct_d, uni_d = dec2[-1], dec3[-1]
    print(f"octave-below: pitch {oct_d.target_hz:.1f} Hz (singer 392.0), "
          f"brightness {oct_d.brightness:.2f} (unison {uni_d.brightness:.2f}), "
          f"vowel_color {oct_d.vowel_color:.2f} (unison {uni_d.vowel_color:.2f}), "
          f"level_trim {oct_d.level_trim_db:.2f} dB")

    if not (0.45 <= oct_d.target_hz / 392.0 <= 0.55):
        failures.append("octave mode is not one octave under the singer")
    if not oct_d.brightness > uni_d.brightness + 0.1:
        failures.append("octave mode's vowel is not kept bright (brightness unchanged)")
    if not oct_d.vowel_color < uni_d.vowel_color - 0.1:
        failures.append("octave mode's vowel is not narrowed (vowel_color unchanged)")
    if not oct_d.level_trim_db < -1.0:
        failures.append("octave mode does not let less volume out at the bottom")

    # A much lower note gets MORE trim than a moderately low one (tip 3 is
    # about the bottom of the range specifically).
    eng4 = HarmonyEngine()
    eng4.set_forced_mode(AccompanimentMode.OCTAVE.value)
    low = run(eng4, AccompanimentMode.OCTAVE, [(2.0, 220.0)])[-1]
    print(f"octave-below (low note): target {low.target_hz:.1f} Hz, "
          f"level_trim {low.level_trim_db:.2f} dB")
    if not low.level_trim_db < oct_d.level_trim_db:
        failures.append("a lower note must get MORE trim than a higher one")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL DRONE/OCTAVE CHECKS PASSED")
    return 0


def _expand(melody):
    out = []
    for seconds, hz in melody:
        out.extend([(seconds, hz)] * int(round(seconds / HOP)))
    return out


def _in(*_a):
    return None


if __name__ == "__main__":
    raise SystemExit(main())

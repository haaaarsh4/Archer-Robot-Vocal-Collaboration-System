"""Check the LIVE HUMMING VOICE in the page against the server's.

The humming accompaniment mode is synthesized twice: once in the browser, for
the live robot voice, and once on the server, for the renders. They are supposed
to be the same voice -- a closed-mouth murmur with one low peak and nothing
above it -- and nothing in either language can enforce that. This check does:
it reads the constants the page actually builds its voice from
(frontend/index.html), recomputes the source and the tract from them the same
way the browser does, and holds them against synthesis/hum_voice.py's own.

Checked, and what each one is for:

  * the page's source harmonics against the server's glottal pulse. The browser
    cannot read the server's code, so the pulse is re-derived from the same
    Rosenberg shape at the same open quotient, one period, differentiated, and
    transformed. The two must agree harmonic by harmonic -- a source that is
    merely "dark" is not the same voice, and the shape of the first few
    harmonics (a murmur's 2nd and 3rd sit ABOVE the fundamental) is most of what
    separates it from a dark sawtooth;
  * the page's tract response against the server's cascade (four resonators, the
    nasal zero, the body shelf, the tilt, the high shelf, the 4th-order
    Butterworth low-pass). Web Audio has no pole-pair resonator, so the page
    builds the same SHAPE out of biquads it has; how close that gets is the
    number this prints;
  * the air: the same band, the same level against the pulse;
  * the level trim the page applies, against what the note it replaced delivered,
    so a murmur that is right is not three times too loud as well.

    .venv/bin/python tools/check_hum_live_tract.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps

sys.path.insert(0, ".")
from synthesis.hum_voice import (  # noqa: E402
    HumConfig, _antiresonance, _resonator, _rosenberg, _shelf,
)

PAGE = Path("frontend/index.html")
FAILURES: list[str] = []
PROBE = [110, 130, 150, 175, 196, 240, 262, 300, 330, 392, 470, 550, 650, 780, 900, 1000,
         1100, 1300, 1500, 1700, 2000, 2400, 3000]
SR = 44100


def check(label: str, ok: bool, detail: str) -> None:
    print(f"    [{'ok' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


# what the page says it does

def page_constants() -> dict:
    html = PAGE.read_text(encoding="utf-8")
    out: dict = {}
    m = re.search(r"const HUM_OPEN_QUOTIENT\s*=\s*([0-9.]+)", html)
    out["open_quotient"] = float(m.group(1)) if m else None
    m = re.search(r"const HUM_SOURCE_HARMONICS\s*=\s*(\d+)", html)
    out["harmonics"] = int(m.group(1)) if m else None
    m = re.search(r"const HUM_VOICE_TRIM\s*=\s*([0-9.]+)", html)
    out["trim"] = float(m.group(1)) if m else None
    block = re.search(r"const HUM_TRACT\s*=\s*\[(.*?)\n\];", html, re.S)
    stages = []
    if block:
        for row in re.findall(r"\[(.*?)\]", block.group(1)):
            parts = [p.strip() for p in row.split(",")]
            stages.append((parts[0].strip("'\""), float(parts[1]), float(parts[2]),
                           float(parts[3]), parts[4].strip() == "true"))
    out["tract"] = stages
    m = re.search(r"breathHp\.frequency\.value\s*=\s*([0-9.]+)", html)
    out["air_hp"] = float(m.group(1)) if m else None
    m = re.search(r"breathLp\.frequency\.value\s*=\s*([0-9.]+)", html)
    out["air_lp"] = float(m.group(1)) if m else None
    m = re.search(r"breathGain\.gain\.value\s*=\s*([0-9.]+)\s*/\s*([0-9.]+)", html)
    out["air_gain"] = (float(m.group(1)), float(m.group(2))) if m else None
    m = re.search(r"const UP\s*=\s*(\d+)", html)
    out["up"] = int(m.group(1)) if m else 2048
    return out


def page_source_harmonics(open_quotient: float, harmonics: int, up: int) -> np.ndarray:
    """The browser's humSourceWave, in numpy: one period, differentiated, transformed."""
    frac = (np.arange(up) + 0.5) / up
    flow = _rosenberg(frac, open_quotient)
    deriv = np.diff(flow, prepend=flow[-1]) * up
    amp = np.empty(harmonics + 1)
    idx = np.arange(up)
    for k in range(1, harmonics + 1):
        w = 2.0 * np.pi * k * (idx + 0.5) / up
        re = float(np.sum(deriv * np.cos(w)))
        im = float(-np.sum(deriv * np.sin(w)))
        amp[k] = 2.0 * math.sqrt(re * re + im * im) / up
    return amp / amp[1]


def server_source_harmonics(cfg: HumConfig, harmonics: int, up: int = 8192) -> np.ndarray:
    frac = (np.arange(up) + 0.5) / up
    flow = _rosenberg(frac, cfg.open_quotient)
    deriv = np.diff(flow, prepend=flow[-1]) * up
    spec = np.abs(np.fft.rfft(deriv)) / up * 2.0
    amp = spec[: harmonics + 1].copy()
    amp[0] = 1e-12
    return amp / amp[1]


# the page's tract, in numpy. A peaking filter and a low-pass use the Q the page
# sets; Web Audio's shelves do not use Q at all (the slope is fixed at 1), which
# is what this uses too.

def biquad_db(freqs, b, a) -> np.ndarray:
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / SR
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    return 20.0 * np.log10(np.abs((b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)) + 1e-30)


def page_stage_db(freqs, kind: str, f0: float, q: float, gain_db: float) -> np.ndarray:
    w0 = 2.0 * math.pi * f0 / SR
    cw, sw = math.cos(w0), math.sin(w0)
    if kind == "peaking":
        A = 10.0 ** (gain_db / 40.0)
        al = sw / (2.0 * max(q, 0.05))
        return biquad_db(freqs, [1 + al * A, -2 * cw, 1 - al * A], [1 + al / A, -2 * cw, 1 - al / A])
    if kind == "lowpass":
        al = sw / (2.0 * max(q, 0.05))
        return biquad_db(freqs, [(1 - cw) / 2, 1 - cw, (1 - cw) / 2], [1 + al, -2 * cw, 1 - al])
    if kind in ("lowshelf", "highshelf"):
        A = 10.0 ** (gain_db / 40.0)
        al = sw / 2.0 * math.sqrt(2.0)          # Web Audio: slope 1, Q unused
        t = 2.0 * math.sqrt(A) * al
        if kind == "highshelf":
            b = [A * ((A + 1) + (A - 1) * cw + t), -2 * A * ((A - 1) + (A + 1) * cw),
                 A * ((A + 1) + (A - 1) * cw - t)]
            a = [(A + 1) - (A - 1) * cw + t, 2 * ((A - 1) - (A + 1) * cw),
                 (A + 1) - (A - 1) * cw - t]
        else:
            b = [A * ((A + 1) - (A - 1) * cw + t), 2 * A * ((A - 1) - (A + 1) * cw),
                 A * ((A + 1) - (A - 1) * cw - t)]
            a = [(A + 1) + (A - 1) * cw + t, -2 * ((A - 1) + (A + 1) * cw),
                 (A + 1) + (A - 1) * cw - t]
        return biquad_db(freqs, b, a)
    raise ValueError(kind)


def server_tract_db(freqs, cfg: HumConfig) -> np.ndarray:
    stages = [_resonator(cfg.f1_hz, cfg.b1_hz, SR), _resonator(cfg.f2_hz, cfg.b2_hz, SR),
              _resonator(cfg.f3_hz, cfg.b3_hz, SR), _resonator(cfg.f4_hz, cfg.b4_hz, SR),
              _antiresonance(cfg.nasal_zero_hz, cfg.nasal_zero_q, cfg.nasal_zero_depth_db, SR),
              _shelf(cfg.body_hz, cfg.body_gain_db, SR, q=0.8, high=False),
              _shelf(cfg.tilt_hz, cfg.tilt_db_per_oct, SR, q=0.7, high=True),
              _shelf(cfg.hf_shelf_hz, cfg.hf_shelf_db, SR, q=0.7, high=True)]
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / SR
    total = np.ones(len(freqs), dtype=complex)
    for c in stages:
        _, h = sps.freqz(c[:3], np.concatenate([[1.0], c[3:]]), worN=w)
        total *= h
    lp = sps.butter(cfg.lowpass_order, min(0.99, cfg.lowpass_hz / (SR / 2.0)), btype="low", output="sos")
    _, h = sps.sosfreqz(lp, worN=w)
    total *= h
    return 20.0 * np.log10(np.maximum(np.abs(total), 1e-12))


def main() -> int:
    const = page_constants()
    cfg = HumConfig.from_config()
    print("=== the page's live humming voice vs the server's")
    print(f"    page: open quotient {const['open_quotient']}, {const['harmonics']} harmonics, "
          f"trim {const['trim']}, tract stages {len(const['tract'])}")
    print(f"    server: open quotient {cfg.open_quotient}, tract "
          f"(F1 {cfg.f1_hz}/{cfg.b1_hz}, F2 {cfg.f2_hz}/{cfg.b2_hz}, zero "
          f"{cfg.nasal_zero_hz} at {cfg.nasal_zero_depth_db} dB, tilt {cfg.tilt_db_per_oct} "
          f"over {cfg.tilt_hz}, hf {cfg.hf_shelf_db} at {cfg.hf_shelf_hz}, "
          f"low-pass {cfg.lowpass_hz} order {cfg.lowpass_order})")

    if const["open_quotient"] is None or not const["tract"]:
        print("FATAL: the page's humming constants could not be read")
        return 2

    # 1. the source
    print("\n--- the glottal source")
    check("the page sings the server's pulse shape",
          abs(const["open_quotient"] - cfg.open_quotient) < 1e-6,
          f"page open quotient {const['open_quotient']} against the server's {cfg.open_quotient}")
    page_src = page_source_harmonics(const["open_quotient"], 20, const["up"])
    srv_src = server_source_harmonics(cfg, 20)
    page_db = 20.0 * np.log10(np.maximum(page_src[1:21], 1e-9))
    srv_db = 20.0 * np.log10(np.maximum(srv_src[1:21], 1e-9))
    worst = float(np.max(np.abs(page_db - srv_db)))
    print("    harmonics 1-10, dB rel. the fundamental  page: "
          + " ".join(f"{v:+.1f}" for v in page_db[:10]))
    print("                                             server: "
          + " ".join(f"{v:+.1f}" for v in srv_db[:10]))
    check("the two sources are the same source", worst <= 1.5,
          f"worst harmonic difference {worst:.2f} dB over the first 20")
    check("the source is a glottal pulse, not a sawtooth",
          page_db[1] > -1.0 and page_db[2] > -1.0,
          f"2nd harmonic {page_db[1]:+.1f} dB, 3rd {page_db[2]:+.1f} dB against the fundamental "
          f"(a sawtooth's would be -6 and -9.5)")

    # 2. the tract
    print("\n--- the murmur's tract")
    page_tract = np.zeros(len(PROBE))
    for kind, f0, q, g, _tracks in const["tract"]:
        page_tract += page_stage_db(PROBE, kind, f0, q, g)
    srv_tract = server_tract_db(PROBE, cfg)
    ref_p = float(np.interp(196.0, PROBE, page_tract))
    ref_s = float(np.interp(196.0, PROBE, srv_tract))
    page_rel, srv_rel = page_tract - ref_p, srv_tract - ref_s
    diff = page_rel - srv_rel
    body = [i for i, f in enumerate(PROBE) if 130.0 <= f <= 2000.0]
    worst_body = float(np.max(np.abs(diff[body])))
    key = [i for i, f in enumerate(PROBE) if abs(f - 392.0) < 1 or abs(f - 588.0) < 1
           or abs(f - 784.0) < 1 or abs(f - 980.0) < 1]
    print("    Hz      " + " ".join(f"{PROBE[i]:>6.0f}" for i in key))
    print("    page    " + " ".join(f"{page_rel[i]:>6.1f}" for i in key))
    print("    server  " + " ".join(f"{srv_rel[i]:>6.1f}" for i in key))
    print("    diff    " + " ".join(f"{diff[i]:>6.1f}" for i in key))
    body_hz = [PROBE[i] for i in body]
    check("the page's tract has the server's shape", worst_body <= 6.0,
          f"worst difference {worst_body:.1f} dB between {min(body_hz):.0f} and "
          f"{max(body_hz):.0f} Hz")
    # the one thing this whole mode is for: one peak, low, then a hole
    peak_hz = float(PROBE[int(np.argmax(page_rel))])
    high = [i for i, f in enumerate(PROBE) if f >= 1000.0]
    best_high = float(np.max(page_rel[high]))
    check("one low peak, then a hole", peak_hz <= 420.0 and best_high < -8.0,
          f"loudest response at {peak_hz:.0f} Hz, and the best above 1 kHz is {best_high:+.1f} dB")

    # 3. the air
    print("\n--- the air")
    check("the air sits inside the murmur, not over it",
          const["air_hp"] == cfg.breath_hp_hz and const["air_lp"] == cfg.breath_lp_hz,
          f"page bands it {const['air_hp']:.0f}-{const['air_lp']:.0f} Hz against the server's "
          f"{cfg.breath_hp_hz:.0f}-{cfg.breath_lp_hz:.0f}")
    air_db = 20.0 * math.log10(const["air_gain"][0] * const["air_gain"][1] / 0.577)
    check("the air is as far under the voice as the server's", abs(air_db - cfg.breath_cap_db) <= 2.0,
          f"page {air_db:.1f} dB against the server's cap of {cfg.breath_cap_db} dB")

    # 4. the level
    print("\n--- the level")
    check("the trim matches the voice it replaced", 0.05 <= const["trim"] <= 0.12,
          f"trim {const['trim']} (the voice this one replaced delivered an rms of 0.503 at 0.30 "
          f"of a note gain; this chain needs about 0.088 of one)")

    print()
    if FAILURES:
        print(f"LIVE HUM CHECKS FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("LIVE HUM TRACT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

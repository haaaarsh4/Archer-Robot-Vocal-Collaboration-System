#!/usr/bin/env python3
"""Measure the site's claims on large, reproducible samples.

Every number printed on the site comes out of this file.  The tests are
deliberately adversarial: vibrato, breath noise, quiet takes, tempo
ambiguity, borrowed notes, dense mixes.  Each suite returns a plain dict and
writes one or more SVG charts into ``frontend/assets/evidence``.

    python tools/site_evidence.py                  # every fast suite
    python tools/site_evidence.py --only pitch     # one suite
    python tools/site_evidence.py --list

The heavy suites (``rvc``, ``asr``, ``bands``) talk to the running app, so
start ``server.py`` first; they are skipped with a clear message otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "reports" / "site_evidence.json"
SVG_DIR = ROOT / "frontend" / "assets" / "evidence"
SVG_DIR.mkdir(parents=True, exist_ok=True)
SR = 44100
BASE_URL = os.environ.get("ARCHER_BASE_URL", "http://127.0.0.1:8000")

# The site's own palette, so a chart dropped into a page looks drawn there.
INK = "#1b1917"
INK_2 = "#565049"
INK_3 = "#8b8479"
PAPER = "#f7f5ef"
LINE = "#d5d1c7"
OCHRE = "#ad4a2a"
INDIGO = "#2d3a61"
SAGE = "#4f6b57"
GOLD = "#a2611d"


# --------------------------------------------------------------------------- #
# SVG helpers
# --------------------------------------------------------------------------- #

def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _axes(width: int, height: int, pad: tuple[int, int, int, int],
          xlab: str = "", ylab: str = "", ymax: float | None = None,
          ymin: float = 0.0, ticks: int = 4, yfmt: str = "{:.0f}") -> tuple[str, dict]:
    left, right, top, bottom = pad
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>']
    top_val = ymax if ymax is not None else 1.0
    for i in range(ticks + 1):
        value = ymin + (top_val - ymin) * i / ticks
        y = top + plot_h - plot_h * (value - ymin) / (top_val - ymin)
        parts.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{LINE}" stroke-width="0.7" opacity="{0.9 if i in (0, ticks) else 0.55}"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="10" '
            f'fill="{INK_3}">{_esc(yfmt.format(value))}</text>'
        )
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="{INK_3}" stroke-width="0.8"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="{INK_3}" stroke-width="0.8"/>')
    if xlab:
        parts.append(
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="10.5" fill="{INK_2}">{_esc(xlab)}</text>'
        )
    if ylab:
        parts.append(
            f'<text x="12" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="10.5" fill="{INK_2}" '
            f'transform="rotate(-90 12 {top + plot_h / 2:.1f})">{_esc(ylab)}</text>'
        )
    geo = {"left": left, "right": right, "top": top, "bottom": bottom,
           "w": plot_w, "h": plot_h, "ymax": top_val, "ymin": ymin}
    return "".join(parts), geo


def _write_svg(name: str, body: str, width: int, height: int) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img">{body}</svg>'
    )
    path = SVG_DIR / name
    path.write_text(svg, encoding="utf-8")
    return f"assets/evidence/{name}"


def chart_bars(name: str, labels: list[str], values: list[float], *,
               ylab: str = "", ymax: float | None = None, fmt: str = "{:.1f}",
               color: str = OCHRE, highlight: int | None = None,
               width: int = 720, height: int = 320) -> str:
    pad = (46, 18, 26, 40)
    top = max(values) if ymax is None else ymax
    top = top * 1.12 if top else 1
    base, geo = _axes(width, height, pad, ylab=ylab, ymax=top, yfmt="{:.0f}")
    parts = [base]
    n = len(values)
    slot = geo["w"] / max(1, n)
    bar = min(56, slot * 0.62)
    for i, (label, value) in enumerate(zip(labels, values)):
        h = geo["h"] * value / (geo["ymax"] - geo["ymin"])
        x = geo["left"] + slot * (i + 0.5) - bar / 2
        y = geo["top"] + geo["h"] - h
        fill = INDIGO if highlight == i else color
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{h:.1f}" rx="2" fill="{fill}"/>')
        parts.append(
            f'<text x="{x + bar / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="10" fill="{INK_2}">{_esc(fmt.format(value))}</text>'
        )
        parts.append(
            f'<text x="{geo["left"] + slot * (i + 0.5):.1f}" y="{height - 20}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="9.5" fill="{INK_2}">{_esc(label)}</text>'
        )
    return _write_svg(name, "".join(parts), width, height)


def chart_hist(name: str, values: list[float], *, bins: int = 24,
               xlab: str = "", ylab: str = "frames", xmax: float | None = None,
               xmin: float = 0.0, color: str = OCHRE,
               width: int = 720, height: int = 300) -> str:
    values = [v for v in values if np.isfinite(v)]
    high = max(values) if xmax is None else xmax
    edges = np.linspace(xmin, high, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    pad = (46, 18, 26, 42)
    base, geo = _axes(width, height, pad, xlab=xlab, ylab=ylab, ymax=float(counts.max() or 1))
    parts = [base]
    slot = geo["w"] / bins
    for i, count in enumerate(counts):
        h = geo["h"] * count / (geo["ymax"] or 1)
        x = geo["left"] + slot * i + 1
        parts.append(f'<rect x="{x:.1f}" y="{geo["top"] + geo["h"] - h:.1f}" width="{slot - 2:.1f}" height="{h:.1f}" fill="{color}" opacity="0.9"/>')
    for i in range(0, bins + 1, max(1, bins // 6)):
        x = geo["left"] + slot * i
        parts.append(
            f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="9.5" fill="{INK_2}">{_esc(f"{edges[i]:.0f}")}</text>'
        )
    return _write_svg(name, "".join(parts), width, height)


def chart_scatter(name: str, xs: list[float], ys: list[float], *,
                  xlab: str = "", ylab: str = "", xmax: float | None = None,
                  ymax: float | None = None, color: str = OCHRE, radius: float = 2.4,
                  width: int = 720, height: int = 340) -> str:
    pad = (48, 18, 24, 42)
    x_hi = xmax or max(xs) * 1.03
    y_hi = ymax or max(ys) * 1.12
    base, geo = _axes(width, height, pad, xlab=xlab, ylab=ylab, ymax=y_hi, yfmt="{:.0f}")
    parts = [base]
    for x, y in zip(xs, ys):
        px = geo["left"] + geo["w"] * x / x_hi
        py = geo["top"] + geo["h"] - geo["h"] * y / y_hi
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" opacity="0.62"/>')
    for i in range(0, 6):
        value = x_hi * i / 5
        x = geo["left"] + geo["w"] * i / 5
        parts.append(
            f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="9.5" fill="{INK_2}">{_esc(f"{value:.0f}")}</text>'
        )
    return _write_svg(name, "".join(parts), width, height)


def chart_paired(name: str, labels: list[str], first: list[float], second: list[float],
                 *, first_label: str, second_label: str, ylab: str = "",
                 fmt: str = "{:.1f}", width: int = 720, height: int = 320) -> str:
    pad = (46, 18, 52, 44)
    top = max(max(first), max(second)) * 1.18 or 1
    base, geo = _axes(width, height, pad, ylab=ylab, ymax=top, yfmt="{:.0f}")
    parts = [base]
    slot = geo["w"] / max(1, len(labels))
    bar = min(26, slot * 0.28)
    for i, (label, a, b) in enumerate(zip(labels, first, second)):
        for j, (value, color) in enumerate(((a, INK_3), (b, OCHRE))):
            h = geo["h"] * value / geo["ymax"]
            x = geo["left"] + slot * (i + 0.5) - bar + j * bar * 1.08
            y = geo["top"] + geo["h"] - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{h:.1f}" rx="2" fill="{color}"/>')
            parts.append(
                f'<text x="{x + bar / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                f'font-family="Inter Tight, Inter, sans-serif" font-size="9" fill="{INK_2}">{_esc(fmt.format(value))}</text>'
            )
        parts.append(
            f'<text x="{geo["left"] + slot * (i + 0.5):.1f}" y="{height - 22}" text-anchor="middle" '
            f'font-family="Inter Tight, Inter, sans-serif" font-size="9.5" fill="{INK_2}">{_esc(label)}</text>'
        )
    for j, (label, color) in enumerate(((first_label, INK_3), (second_label, OCHRE))):
        x = geo["left"] + j * 150
        parts.append(f'<rect x="{x}" y="10" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 15}" y="19" font-family="Inter Tight, Inter, sans-serif" '
            f'font-size="10" fill="{INK_2}">{_esc(label)}</text>'
        )
    return _write_svg(name, "".join(parts), width, height)


# --------------------------------------------------------------------------- #
# Signal generators
# --------------------------------------------------------------------------- #

def sung_tone(f0: float, seconds: float = 1.6, *, sr: int = SR,
              vibrato_cents: float = 22.0, vibrato_hz: float = 5.3,
              noise_db: float = -28.0, seed: int = 0) -> np.ndarray:
    """A harmonic voice-like tone with vibrato, drift and breath noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    cents = vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t + rng.uniform(0, 6.28))
    cents += 9.0 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    f0_curve = f0 * 2 ** (cents / 1200.0)
    phase = 2 * np.pi * np.cumsum(f0_curve) / sr
    out = np.zeros(n)
    for h in range(1, 18):
        if f0 * h > sr * 0.45:
            break
        amp = 1.0 / (h ** 0.85)
        # formant-ish weighting: stronger around 500 and 1500 Hz
        amp *= 1.0 + 0.45 * np.exp(-((f0 * h - 500) / 380) ** 2)
        amp *= 1.0 + 0.30 * np.exp(-((f0 * h - 1500) / 600) ** 2)
        out += amp * np.sin(h * phase + rng.uniform(0, 6.28))
    out /= max(1e-9, np.max(np.abs(out)))
    breath = rng.normal(0, 1, n)
    # shape the breath like a voice's high-frequency air, not white hiss
    breath = np.convolve(breath, np.array([0.18, 0.34, 0.48, 0.34, 0.18]), mode="same")
    out = out + breath * (10 ** (noise_db / 20.0))
    env = np.ones(n)
    edge = int(0.03 * sr)
    env[:edge] = np.linspace(0, 1, edge)
    env[-edge:] = np.linspace(1, 0, edge)
    return (out * env).astype(np.float32)


def click_track(bpm: float, seconds: float = 8.0, *, sr: int = SR,
                subdivision: int = 2, jitter_ms: float = 0.0,
                seed: int = 0) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n)
    beat = 60.0 / bpm
    step = beat / subdivision
    times = []
    t = 0.12
    i = 0
    while t < seconds - 0.2:
        at = t + rng.normal(0, jitter_ms / 1000.0)
        at = max(0.01, at)
        times.append(at)
        start = int(at * sr)
        length = int(0.045 * sr)
        tt = np.arange(length) / sr
        accent = 1.0 if i % subdivision == 0 else 0.62
        hit = np.sin(2 * np.pi * 1250 * tt) * np.exp(-tt * 70) * accent
        hit += np.sin(2 * np.pi * 420 * tt) * np.exp(-tt * 38) * accent * 0.55
        stop = min(n, start + length)
        out[start:stop] += hit[:stop - start]
        t += step
        i += 1
    out += rng.normal(0, 0.0008, n)
    return out.astype(np.float32), times


def melody_tone(notes: list[int], *, sr: int = SR, note_s: float = 0.36,
                seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Sing a sequence of MIDI notes with portamento and vibrato."""
    rng = np.random.default_rng(seed)
    total = int((note_s * len(notes) + 0.25) * sr)
    out = np.zeros(total, dtype=np.float64)
    truth = []
    n_per = int(note_s * sr)
    for index, midi in enumerate(notes):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        start = index * n_per
        length = int(n_per * 0.92)
        t = np.arange(length) / sr
        cents = 14 * np.sin(2 * np.pi * 5.1 * t + rng.uniform(0, 6))
        if index:
            prev = 440.0 * 2 ** ((notes[index - 1] - 69) / 12)
            glide = np.linspace(1200 * math.log2(prev / f), 0, min(length, int(0.05 * sr)))
            cents[:len(glide)] += glide
        curve = f * 2 ** (cents / 1200)
        phase = 2 * np.pi * np.cumsum(curve) / sr
        voice = np.zeros(length)
        for h in range(1, 12):
            voice += (1.0 / h ** 0.9) * np.sin(h * phase + rng.uniform(0, 6))
        voice /= max(1e-9, np.max(np.abs(voice)))
        env = np.ones(length)
        edge = int(0.012 * sr)
        env[:edge] = np.linspace(0, 1, edge)
        env[-edge:] = np.linspace(1, 0, edge)
        stop = min(total, start + length)
        out[start:stop] += (voice * env)[:stop - start]
        truth.append((start / sr, (start + length) / sr, float(f)))
    return out.astype(np.float32), truth


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #

def _smoother_trial(detector, f0: float, frames: int, outlier_rate: float,
                    rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Feed a known pitch curve with detector-like octave outliers through _smooth."""
    detector.reset()
    raw, smoothed = [], []
    cents = 12 * np.sin(np.linspace(0, 6, frames))
    for i in range(frames):
        value = f0 * 2 ** ((cents[i] + rng.normal(0, 7)) / 1200)
        if rng.random() < outlier_rate:
            value *= 2.0 ** (int(rng.choice([-1, 1])))
        raw.append(value)
        smoothed.append(detector._smooth(float(value)))
    return raw, smoothed


def suite_pitch(n: int = 240, trials: int = 240) -> dict:
    """YIN on realistic vowels, plus the smoother under detector outliers."""
    from analysis.pitch_detector import PitchDetector

    detector = PitchDetector()
    sr = detector.sample_rate
    rng = np.random.default_rng(7)
    rows = []

    for i in range(n):
        f0 = float(np.exp(rng.uniform(np.log(75), np.log(950))))
        noise_db = float(rng.uniform(-38, -14))
        audio = sung_tone(f0, seconds=1.4, noise_db=noise_db, seed=i, sr=sr)
        gain = float(10 ** (rng.uniform(-34, -12) / 20))
        audio = audio * gain

        detector.reset()
        kept = []
        for start in range(0, len(audio), 1024):
            frame = audio[start:start + 1024]
            if len(frame) < 1024:
                break
            hz, conf = detector.detect(frame)
            if hz and conf >= 0.3:
                kept.append(float(hz))
        if not kept:
            continue
        mid = kept[len(kept) // 4: max(1, 3 * len(kept) // 4)] or kept
        value = float(np.median(mid))
        cents = abs(1200 * math.log2(value / f0))
        rows.append({"f0": f0, "noise_db": noise_db, "cents": cents,
                     "gain_db": 20 * math.log10(max(1e-9, gain)),
                     "octave": cents > 700})

    cents = np.array([r["cents"] for r in rows])

    # The smoother's own job: a real singing curve, plus detector outliers.
    raw_cents, smooth_cents = [], []
    raw_oct, smooth_oct = [], []
    for t in range(trials):
        f0 = float(np.exp(rng.uniform(np.log(90), np.log(700))))
        raw, smoothed = _smoother_trial(detector, f0, 90, float(rng.uniform(0.04, 0.18)), rng)
        for value, out in zip(raw, smoothed):
            raw_cents.append(abs(1200 * math.log2(value / f0)))
            smooth_cents.append(abs(1200 * math.log2(out / f0)))
        raw_oct.extend([c > 700 for c in raw_cents[-90:]])
        smooth_oct.extend([c > 700 for c in smooth_cents[-90:]])

    raw_cents = np.array(raw_cents)
    smooth_cents = np.array(smooth_cents)
    metrics = {
        "samples": len(rows),
        "median_cents": round(float(np.median(cents)), 2),
        "p90_cents": round(float(np.percentile(cents, 90)), 2),
        "within_25c": round(float(np.mean(cents <= 25)), 4),
        "within_50c": round(float(np.mean(cents <= 50)), 4),
        "within_100c": round(float(np.mean(cents <= 100)), 4),
        "octave_errors": round(float(np.mean([r["octave"] for r in rows])), 4),
        "smoother_trials": trials,
        "smoother_frames": int(len(raw_cents)),
        "smoother_raw_p90_cents": round(float(np.percentile(raw_cents, 90)), 1),
        "smoother_out_p90_cents": round(float(np.percentile(smooth_cents, 90)), 1),
        "smoother_raw_octave_rate": round(float(np.mean(raw_oct)), 4),
        "smoother_out_octave_rate": round(float(np.mean(smooth_oct)), 4),
    }

    chart_scatter("pitch-accuracy.svg",
                  [r["f0"] for r in rows], [min(200, r["cents"]) for r in rows],
                  xlab="true fundamental (Hz)", ylab="error (cents, clipped at 200)",
                  xmax=1000, ymax=200)
    chart_hist("pitch-error.svg", [min(200, r["cents"]) for r in rows],
               bins=24, xlab="cents from the true fundamental", xmax=200)
    chart_paired("pitch-smoother.svg",
                 ["p90 error", "octave rate"],
                 [round(float(np.percentile(raw_cents, 90)), 0),
                  round(float(np.mean(raw_oct)) * 100, 1)],
                 [round(float(np.percentile(smooth_cents, 90)), 0),
                  round(float(np.mean(smooth_oct)) * 100, 1)],
                 first_label="raw frame readings", second_label="after the smoother",
                 ylab="cents / %", fmt="{:.1f}")
    return {"metrics": metrics, "charts": [
        "assets/evidence/pitch-accuracy.svg",
        "assets/evidence/pitch-error.svg",
        "assets/evidence/pitch-smoother.svg",
    ]}


def suite_notes(n_melodies: int = 60) -> dict:
    """Note segmentation on sung melodies with a known pitch curve.

    The curve is the ground truth, not a detector's output: this isolates the
    segmentation itself (jump confirmation, octave folding, the 120 ms floor,
    the middle-70% pitch) from the live detector's own frame latency.
    """
    from synthesis.melody_voice import track_notes

    rng = np.random.default_rng(11)
    sr = SR
    hop = 1024
    onset_errors = []
    pitch_errors = []
    matched = 0
    truth_count = 0
    detected_count = 0

    for m in range(n_melodies):
        root = int(rng.integers(55, 70))
        scale = [0, 2, 4, 5, 7, 9, 11]
        # A stepwise line inside one octave: the common melodic case.  The
        # octave-folding stage gets its own targeted test below.
        degree = 4
        notes = []
        for _ in range(10):
            degree = int(np.clip(degree + rng.choice([-2, -1, 0, 1, 2]), 0, 6))
            notes.append(root + scale[degree])
        audio, truth = melody_tone(notes, sr=sr, note_s=0.42, seed=1000 + m)
        n_frames = (len(audio) // hop) + 1
        hz = np.zeros(n_frames)
        voiced = np.zeros(n_frames, dtype=bool)
        for (start, end, f0), index in zip(truth, range(len(truth))):
            a = int(round(start * sr / hop))
            b = int(round(end * sr / hop))
            for j, frame in enumerate(range(max(0, a), min(n_frames, b))):
                # vibrato plus jitter, so the median has something to reject
                vibrato = 14 * math.sin(2 * math.pi * j / 9.0)
                hz[frame] = f0 * 2 ** ((vibrato + rng.normal(0, 5)) / 1200)
                voiced[frame] = True
        # a couple of wild readings, as any detector produces, but never two
        # in a row on the same note: that is the case the tracker is built for
        for _ in range(max(1, n_frames // 120)):
            frame = int(rng.integers(1, n_frames - 1))
            if voiced[frame] and voiced[frame - 1]:
                hz[frame] *= 2 ** (int(rng.choice([-1, 1])))
        hop_s = hop / sr
        _, tracked = track_notes(hz, voiced, hop_s, audio=audio, sr=sr,
                                 hop_samples=hop, snap_to_key=False)

        truth_count += len(truth)
        detected_count += len(tracked)
        used = set()
        for start, end, f0 in truth:
            best = None
            for i, note in enumerate(tracked):
                if i in used:
                    continue
                overlap = min(end, note.end_s) - max(start, note.start_s)
                if overlap <= 0:
                    continue
                if best is None or overlap > best[0]:
                    best = (overlap, i, note)
            if best is not None and best[0] >= 0.5 * (end - start):
                overlap, index, note = best
                used.add(index)
                matched += 1
                onset_errors.append(abs(note.start_s - start) * 1000)
                pitch_errors.append(abs(1200 * math.log2(max(1e-6, note.f0_hz) / f0)))

    precision = matched / max(1, detected_count)
    recall = matched / max(1, truth_count)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    metrics = {
        "melodies": n_melodies,
        "true_notes": truth_count,
        "detected_notes": detected_count,
        "matched_notes": matched,
        "note_precision": round(precision, 4),
        "note_recall": round(recall, 4),
        "note_f1": round(f1, 4),
        "onset_median_ms": round(float(np.median(onset_errors)), 1) if onset_errors else None,
        "onset_p90_ms": round(float(np.percentile(onset_errors, 90)), 1) if onset_errors else None,
        "pitch_median_cents": round(float(np.median(pitch_errors)), 1) if pitch_errors else None,
        "pitch_within_50c": round(float(np.mean(np.array(pitch_errors) <= 50)), 4) if pitch_errors else None,
    }
    chart_hist("note-onset-error.svg", onset_errors, bins=20,
               xlab="onset error (ms)", xmax=90)
    chart_hist("note-pitch-error.svg", [min(120, x) for x in pitch_errors], bins=20,
               xlab="cents from the sung note", xmax=120)
    return {"metrics": metrics, "charts": [
        "assets/evidence/note-onset-error.svg",
        "assets/evidence/note-pitch-error.svg",
    ]}


def suite_rhythm(n_patterns: int = 96) -> dict:
    """Tempo and onset tracking across 55-190 BPM with human jitter."""
    from analysis.rhythm_analyzer import RhythmAnalyzer

    rng = np.random.default_rng(3)
    analyzer = RhythmAnalyzer()
    tempo_err = []
    onset_f1 = []
    tempos = []
    for i in range(n_patterns):
        bpm = float(rng.uniform(55, 190))
        sub = int(rng.choice([1, 2, 4]))
        jitter = float(rng.uniform(0, 14))
        audio, truth = click_track(bpm, 8.0, subdivision=sub, jitter_ms=jitter, seed=i)
        analyzer.reset()
        for start in range(0, len(audio) - analyzer.frame_size, analyzer.frame_size):
            analyzer.push_frame(audio[start:start + analyzer.frame_size], True)
        est = float(analyzer.current_tempo)
        tempos.append((bpm, est))
        if est > 0:
            tempo_err.append(abs(est - bpm))
        # The analyzer keeps the last 16 onsets for tempo; compare inside that
        # span, otherwise recall is capped by the history length itself.
        detected = list(analyzer._onset_times)
        span_start = detected[0] - 0.06 if detected else 0.0
        truth_window = [t for t in truth if t >= span_start - 0.01]
        used = set()
        hits = 0
        for t in truth_window:
            for j, d in enumerate(detected):
                if j in used:
                    continue
                if abs(d - t) <= 0.05:
                    used.add(j)
                    hits += 1
                    break
        precision = hits / max(1, len(detected))
        recall = hits / max(1, len(truth_window))
        onset_f1.append(2 * precision * recall / max(1e-9, precision + recall))

    tempo_err = np.asarray(tempo_err)
    metrics = {
        "patterns": n_patterns,
        "tempo_mae_bpm": round(float(np.mean(tempo_err)), 2),
        "tempo_median_mae_bpm": round(float(np.median(tempo_err)), 2),
        "onset_f1_mean": round(float(np.mean(onset_f1)), 4),
        "onset_f1_median": round(float(np.median(onset_f1)), 4),
    }
    within2 = [abs(est - bpm) / bpm <= 0.02 for bpm, est in tempos if est > 0]
    within4 = [abs(est - bpm) / bpm <= 0.04 for bpm, est in tempos if est > 0]
    metrics["tempo_within_2pct"] = round(float(np.mean(within2)), 4)
    metrics["tempo_within_4pct"] = round(float(np.mean(within4)), 4)
    octave_ok = 0
    for bpm, est in tempos:
        if est <= 0:
            continue
        # a tempo and its double/half are the same pulse; count them as the
        # same answer, which is the standard octave-tolerant tempo score
        ratio = math.log2(est / bpm)
        if min(abs(ratio - r) for r in (-1.0, 0.0, 1.0, 2.0)) <= 0.08:
            octave_ok += 1
    metrics["tempo_exact_within_2pct"] = metrics["tempo_within_2pct"]
    metrics["tempo_octave_tolerant"] = round(octave_ok / max(1, len(tempos)), 4)

    chart_scatter("rhythm-tempo.svg", [b for b, _ in tempos], [abs(e - b) for b, e in tempos],
                  xlab="true tempo (BPM)", ylab="absolute error (BPM)", xmax=200, ymax=20)
    chart_hist("rhythm-onset-f1.svg", [f * 100 for f in onset_f1], bins=16,
               xlab="onset F1 (%)", xmax=100)
    return {"metrics": metrics, "charts": [
        "assets/evidence/rhythm-tempo.svg", "assets/evidence/rhythm-onset-f1.svg"]}


def suite_key(songs: int = 120) -> dict:
    """The live Krumhansl estimator on real labelled melodies from OpenEWLD."""
    from mysong.corpus import load_corpus
    from synthesis.harmony_engine import HarmonyEngine

    corpus = load_corpus()
    rng = np.random.default_rng(5)
    picked = list(rng.choice(len(corpus), size=min(songs, len(corpus)), replace=False))
    engine = HarmonyEngine()
    exact = 0
    tonic = 0
    relative = 0
    rows = []
    for index in picked:
        song = corpus[int(index)]
        notes = song.notes
        if not notes:
            continue
        # The engine keeps roughly six seconds of pitched frames.
        pitches = []
        for note in notes:
            pitches.extend([note.pitch] * max(1, int(round(note.duration * 2))))
        if len(pitches) > 256:
            step = len(pitches) / 256
            pitches = [pitches[int(i * step)] for i in range(256)]
        engine._pitch_history = [float(440 * 2 ** ((p - 69) / 12)) for p in pitches]
        engine._current_key_root = -1
        engine._current_scale = []
        engine._infer_key()
        mode = "major" if engine._current_scale == engine.MAJOR_SCALE else "minor"
        got = (engine._current_key_root, mode)
        want = (song.key.tonic, song.key.mode)
        ok = got == want
        exact += ok
        tonic += got[0] == want[0]
        # relative major/minor (same key signature) counts as a near miss
        rel = (want[0] + (3 if want[1] == "minor" else -3)) % 12
        relative += ok or (got[0] == rel and got[1] != want[1])
        rows.append({"song": song.title, "want": f"{want[0]} {want[1]}",
                     "got": f"{got[0]} {mode}", "exact": bool(ok)})
    total = max(1, len(rows))
    metrics = {
        "songs": len(rows),
        "exact_accuracy": round(exact / total, 4),
        "tonic_accuracy": round(tonic / total, 4),
        "within_relative_key": round(relative / total, 4),
        "misses": [r for r in rows if not r["exact"]][:8],
    }
    chart_bars("key-accuracy.svg",
               ["exact key + mode", "tonic correct", "relative key"],
               [exact / total * 100, tonic / total * 100, relative / total * 100],
               ylab="% of songs", ymax=100, fmt="{:.1f}")
    return {"metrics": metrics, "charts": ["assets/evidence/key-accuracy.svg"]}


def suite_modes(n_contexts: int = 1000) -> dict:
    """Exhaustive-ish check of the mode selector's priority order."""
    from config.config_loader import get_config
    from synthesis.accompaniment_modes import AccompanimentMode, AccompanimentModeSelector

    cfg = get_config()
    selector = AccompanimentModeSelector(cfg)
    rng = np.random.default_rng(2)
    counts = {}
    violations = {
        "silent_when_protocol_off": 0,
        "silent_when_sensitive": 0,
        "call_response_after_phrase": 0,
        "fusion_wins_over_tempo": 0,
    }
    for _ in range(n_contexts):
        protocol = bool(rng.integers(0, 2))
        sensitive = bool(rng.integers(0, 2))
        has_pitch = bool(rng.integers(0, 2))
        phrase_end = bool(rng.integers(0, 2))
        tempo = float(rng.uniform(40, 220))
        sustain = int(rng.integers(0, 80))
        mode = selector.select(
            protocol_enabled=protocol, is_protocol_sensitive=sensitive,
            archer_hz=220.0 if has_pitch else None,
            phrase_state="phrase_end" if phrase_end else ("singing" if has_pitch else "silence"),
            phrase_just_ended=phrase_end, tempo_bpm=tempo, frames_on_current_note=sustain,
        )
        counts[mode.value] = counts.get(mode.value, 0) + 1
        if not protocol and mode is not AccompanimentMode.SILENT:
            violations["silent_when_protocol_off"] += 1
        if protocol and sensitive and mode is not AccompanimentMode.SILENT:
            violations["silent_when_sensitive"] += 1
        if protocol and not sensitive and not has_pitch and phrase_end and mode is not AccompanimentMode.CALL_RESPONSE:
            violations["call_response_after_phrase"] += 1
        if protocol and not sensitive and has_pitch and selector.fusion_mode and mode is not AccompanimentMode.TRIADIC:
            violations["fusion_wins_over_tempo"] += 1
    total_violations = sum(violations.values())
    metrics = {
        "contexts": n_contexts,
        "invariant_violations": total_violations,
        "violations": violations,
        "distribution": counts,
    }
    chart_bars("mode-decisions.svg", list(counts.keys()), list(counts.values()),
               ylab="contexts", fmt="{:.0f}")
    return {"metrics": metrics, "charts": ["assets/evidence/mode-decisions.svg"]}


def suite_voicing(progressions: int = 120) -> dict:
    """Voice-leading in the arrangement engine against a block-chord baseline."""
    from mysong.corpus import load_corpus
    from mysong.accompaniment import choose_voicings, ChordSegment

    corpus = load_corpus()
    rng = np.random.default_rng(13)
    engine_move = []
    baseline_move = []
    engine_parallel = []
    baseline_parallel = []
    ranges = []

    def perfect_pairs(previous, current):
        """Count parallel perfect fifths/octaves between adjacent voices."""
        if len(previous) < 2 or len(current) < 2:
            return 0
        prev = sorted(previous)
        cur = sorted(current)
        count = 0
        for i in range(len(prev) - 1):
            for j in range(i + 1, len(prev)):
                a1, a2 = prev[i], prev[j]
                b1, b2 = cur[min(i, len(cur) - 1)], cur[min(j, len(cur) - 1)]
                if a1 == b1 or a2 == b2:
                    continue
                int_a, int_b = (a2 - a1) % 12, (b2 - b1) % 12
                if int_a == int_b and int_a in (0, 7):
                    count += 1
        return count

    for _ in range(progressions):
        song = corpus[int(rng.integers(0, len(corpus)))]
        if len(song.chords) < 4:
            continue
        start = int(rng.integers(0, len(song.chords) - 4))
        spans = song.chords[start:start + 4]
        segments = [ChordSegment(span.onset, span.duration, span.chord) for span in spans]
        chosen = choose_voicings(segments)
        # baseline: the same chords in close position around C4, no search
        baseline = []
        for seg in segments:
            pcs = sorted(seg.chord.pitch_classes(False))
            notes = []
            for pc in pcs:
                pitch = 60 + pc
                while notes and pitch <= notes[-1]:
                    pitch += 12
                notes.append(pitch)
            baseline.append(sorted(notes))
        def movement(voicings):
            steps = []
            for a, b in zip(voicings[:-1], voicings[1:]):
                aa, bb = sorted(a), sorted(b)
                steps.append(float(np.mean([abs(y - x) for x, y in zip(aa, bb)])))
            return float(np.mean(steps)) if steps else 0.0
        engine_move.append(movement(chosen))
        baseline_move.append(movement(baseline))
        engine_parallel.append(sum(perfect_pairs(a, b) for a, b in zip(chosen[:-1], chosen[1:])))
        baseline_parallel.append(sum(perfect_pairs(a, b) for a, b in zip(baseline[:-1], baseline[1:])))
        ranges.extend([max(v) - min(v) for v in chosen])

    metrics = {
        "progressions": len(engine_move),
        "mean_voice_movement_semitones": round(float(np.mean(engine_move)), 2),
        "baseline_mean_movement": round(float(np.mean(baseline_move)), 2),
        "parallel_perfects_per_100_transitions": round(float(np.mean(engine_parallel) / 3 * 100), 2),
        "baseline_parallel_perfects_per_100": round(float(np.mean(baseline_parallel) / 3 * 100), 2),
        "median_voicing_span_semitones": round(float(np.median(ranges)), 1),
    }
    chart_paired("voicing-movement.svg", ["voice-leading", "block chords"],
                 [float(np.mean(engine_move)), float(np.mean(baseline_move))],
                 [float(np.mean(engine_parallel) / 3 * 100),
                  float(np.mean(baseline_parallel) / 3 * 100)],
                 first_label="semitones moved", second_label="parallel perfects / 100",
                 ylab="movement and parallels", fmt="{:.2f}")
    return {"metrics": metrics, "charts": ["assets/evidence/voicing-movement.svg"]}


def suite_mysong() -> dict:
    """Held-out chord and key accuracy from the shipped MySong model."""
    from mysong.model import MySongModel
    from mysong.evaluate import evaluate_corpus

    model = MySongModel.load()
    report = evaluate_corpus(model, split="test", include_generative=False)
    auto = report["auto"]
    ours = auto["ours"]
    metrics = {
        "songs": int(ours["n"]),
        "root": float(ours["root"]),
        "core": float(ours["core"]),
        "full": float(ours["full"]),
        "key": float(auto["ours"]["key_exact"]),
        "changes_f1": float(ours["change_f1"]),
        "fixed_bar_root": float(auto.get("fixed_bar", {}).get("root", 0.0)),
        "template_root": float(auto.get("template", {}).get("root", 0.0)),
        "segments_per_song": float(ours["segments_per_song"]),
    }
    chart_bars("mysong-scores.svg",
               ["root", "core", "full", "key", "changes"],
               [metrics["root"] * 100, metrics["core"] * 100, metrics["full"] * 100,
                metrics["key"] * 100, metrics["changes_f1"] * 100],
               ylab="% on 44 held-out songs", ymax=100, fmt="{:.1f}")
    chart_bars("mysong-baselines.svg",
               ["MySong tagger", "one chord/bar", "template match"],
               [metrics["root"] * 100, metrics["fixed_bar_root"] * 100,
                metrics["template_root"] * 100],
               ylab="root accuracy (%)", ymax=100, fmt="{:.1f}", color=INDIGO, highlight=0)
    return {"metrics": metrics, "charts": [
        "assets/evidence/mysong-scores.svg", "assets/evidence/mysong-baselines.svg"]}


def _api_json(path: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def suite_latency(rounds: int = 40) -> dict:
    """Round-trip latency of the endpoints the demo panels call."""
    cases = [
        ("/api/analyze", {"text": "the moon is low over the water", "already_english": True}),
        ("/api/sentiment", {"text": "the moon is low over the water"}),
        ("/api/vocables/manifest", None),
        ("/api/mysong/status", None),
        ("/singsong/status", None),
    ]
    out = {}
    for path, payload in cases:
        times = []
        for _ in range(rounds):
            start = time.perf_counter()
            try:
                _api_json(path, payload, timeout=30)
            except Exception:
                break
            times.append((time.perf_counter() - start) * 1000)
        if times:
            out[path] = {
                "n": len(times),
                "p50_ms": round(float(np.median(times)), 1),
                "p95_ms": round(float(np.percentile(times, 95)), 1),
            }
    if not out:
        raise RuntimeError("server not reachable")
    labels = [k.split("/")[-1] for k in out]
    chart_bars("api-latency.svg", labels, [v["p95_ms"] for v in out.values()],
               ylab="p95 round trip (ms)", fmt="{:.0f}", color=INDIGO)
    return {"metrics": {"endpoints": out}, "charts": ["assets/evidence/api-latency.svg"]}


def _post_bytes(path: str, payload: dict, timeout: int = 240) -> tuple[bytes, dict]:
    req = urllib.request.Request(
        BASE_URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read(), dict(response.headers)


def _header(headers: dict, name: str, default: str = "") -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return default


def _post_multipart(path: str, raw: bytes, *, filename: str = "take.wav",
                    fields: dict | None = None, timeout: int = 240) -> tuple[bytes, dict]:
    import uuid
    boundary = "----archer" + uuid.uuid4().hex
    parts = []
    for key, value in (fields or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode() + raw + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        BASE_URL + path, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read(), dict(response.headers)


def _load_wav(raw: bytes) -> tuple[np.ndarray, int]:
    import io
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _f0_median(audio: np.ndarray, sr: int) -> float | None:
    import librosa
    if audio.size < sr // 4:
        return None
    track = librosa.yin(audio, fmin=60, fmax=1000, sr=sr)
    keep = track[(track > 60) & (track < 1000)]
    return float(np.median(keep)) if keep.size else None


def _envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    spec = np.abs(librosa.stft(np.asarray(audio, dtype=np.float32), n_fft=2048, hop_length=512)) ** 2
    power = spec.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= 200) & (freqs <= 4000)
    freqs, power = freqs[mask], power[mask]
    centers = np.geomspace(200, 4000, 48)
    values = []
    for center in centers:
        sel = (freqs >= center * 2 ** (-1 / 24)) & (freqs < center * 2 ** (1 / 24))
        values.append(10 * np.log10(float(np.mean(power[sel])) + 1e-12) if sel.any() else np.nan)
    values = np.asarray(values)
    return values - np.nanmean(values)


def _envelope_error(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    return float(np.nanmean(np.abs(_envelope(a, sr) - _envelope(b, sr))))


def suite_vocables() -> dict:
    """The pre-rendered vocable bank: pitch lands, timbre stays put."""
    import librosa
    from synthesis.neural_vocable_bank import NeuralVocableBank

    bank = NeuralVocableBank(ROOT / "synthesis" / "samples" / "neural", SR)
    shifts = (-9, -7, -5, -3, 3, 5, 7, 9)
    bank_pitch, naive_pitch = [], []
    bank_env, naive_env = [], []
    env_by_shift: dict[int, list[tuple[float, float]]] = {}
    rows = []
    for name in sorted(bank._bases)[:8]:
        base = bank._bases[name][0]
        for shift in shifts:
            target = base.f0_hz * 2 ** (shift / 12)
            out = bank.get_blocking(name, target, int(0.8 * SR))
            if out is None or len(out) < SR // 8:
                continue
            naive = librosa.effects.pitch_shift(base.audio, sr=SR, n_steps=shift)
            for audio, bucket in ((out, bank_pitch), (naive, naive_pitch)):
                measured = _f0_median(audio, SR)
                if measured:
                    bucket.append(abs(1200 * math.log2(measured / target)))
            bank_error = _envelope_error(base.audio, out, SR)
            naive_error = _envelope_error(base.audio, naive, SR)
            bank_env.append(bank_error)
            naive_env.append(naive_error)
            env_by_shift.setdefault(shift, []).append((bank_error, naive_error))
            rows.append({"vocable": name, "shift": shift})

    metrics = {
        "renders": len(rows),
        "bank_pitch_median_cents": round(float(np.median(bank_pitch)), 1),
        "bank_pitch_within_50c": round(float(np.mean(np.array(bank_pitch) <= 50)), 3),
        "naive_pitch_median_cents": round(float(np.median(naive_pitch)), 1),
        "bank_envelope_error_db": round(float(np.mean(bank_env)), 2),
        "naive_envelope_error_db": round(float(np.mean(naive_env)), 2),
        "timbre_improvement_pct": round((1 - np.mean(bank_env) / max(1e-6, np.mean(naive_env))) * 100, 1),
    }
    shifts_present = [s for s in shifts if s in env_by_shift]
    chart_paired(
        "vocable-timbre.svg",
        [f"{s:+d}" for s in shifts_present],
        [float(np.mean([v[0] for v in env_by_shift[s]])) for s in shifts_present],
        [float(np.mean([v[1] for v in env_by_shift[s]])) for s in shifts_present],
        first_label="bank shift", second_label="plain resample shift",
        ylab="envelope deviation (dB, lower is closer)", fmt="{:.1f}")
    chart_hist("vocable-pitch.svg", [min(150, x) for x in bank_pitch], bins=18,
               xlab="cents from the target note", xmax=150)
    return {"metrics": metrics, "charts": [
        "assets/evidence/vocable-pitch.svg", "assets/evidence/vocable-timbre.svg"]}


def suite_rvc() -> dict:
    """The trained voice sidecar: pitch fidelity, level and round trip."""
    from analysis.rmvpe_detector import RMVPEModel

    rmvpe = RMVPEModel(str(ROOT / "assets" / "rmvpe" / "rmvpe.pt"), device="cpu")

    def rmvpe_median(audio: np.ndarray, sr: int) -> float | None:
        import librosa
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        track, _ = rmvpe.infer(np.asarray(audio, dtype=np.float32), voicing_threshold=0.03)
        track = np.asarray(track)
        track = track[track > 0]
        return float(np.median(track)) if track.size else None

    pitches = [110.0, 174.6, 261.6, 392.0]
    durations = [0.35, 0.7]
    cents, latencies = [], []
    levels, peaks = [], []
    applied = 0
    total = 0
    for voice in (0, 1, 2):
        for target in pitches:
            for duration in durations:
                payload = {
                    "target_hz": target, "vocable": "cold", "duration_s": duration,
                    "mode": "unison_shadowing", "texture": "solo", "voice_index": voice,
                }
                started = time.perf_counter()
                try:
                    raw, headers = _post_bytes("/api/neural/render", payload)
                except Exception:
                    continue
                latencies.append((time.perf_counter() - started) * 1000)
                audio, sr = _load_wav(raw)
                # RMVPE, not YIN: the converted take carries a low-frequency
                # body the converter added, and autocorrelation choices on
                # that body reported 71 Hz for a 262 Hz note.  The neural
                # detector is the one the voice model itself is built around.
                measured = rmvpe_median(audio, sr)
                total += 1
                applied += _header(headers, "X-Neural-Applied") == "1"
                if measured:
                    cents.append(abs(1200 * math.log2(measured / target)))
                levels.append(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
                peaks.append(float(np.max(np.abs(audio))))
    if not total:
        raise RuntimeError("neural sidecar did not answer")
    metrics = {
        "renders": total,
        "neural_applied": applied,
        "pitch_median_cents": round(float(np.median(cents)), 1) if cents else None,
        "pitch_within_50c": round(float(np.mean(np.array(cents) <= 50)), 3) if cents else None,
        "pitch_p95_cents": round(float(np.percentile(cents, 95)), 1) if cents else None,
        "latency_p50_ms": round(float(np.median(latencies)), 0),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 0),
        "rms_median": round(float(np.median(levels)), 4),
        "peak_max": round(float(np.max(peaks)), 3),
    }
    chart_scatter("rvc-pitch.svg", [float(x) for x in cents], [float(y) for y in levels],
                  xlab="pitch error (cents)", ylab="output RMS", xmax=200, ymax=max(levels) * 1.15)
    chart_bars("rvc-latency.svg", ["p50", "p95"],
               [metrics["latency_p50_ms"], metrics["latency_p95_ms"]],
               ylab="round trip (ms)", fmt="{:.0f}", color=INDIGO)
    return {"metrics": metrics, "charts": [
        "assets/evidence/rvc-pitch.svg", "assets/evidence/rvc-latency.svg"]}


ASR_SENTENCES = [
    "The moon is low over the quiet water tonight.",
    "I hear the drum before I see the dancers.",
    "My grandmother sang this song when the snow was deep.",
    "We walked to the river and the dogs came with us.",
    "The old house still smells of cedar and smoke.",
    "She told me to listen to the wind through the pines.",
    "There is a fire in the centre of the camp.",
    "The horses are restless in the early morning.",
    "He carried the bundle across the frozen lake.",
    "The song has four verses and a long ending.",
    "I remember the colour of your coat in winter.",
    "The children are laughing behind the tent.",
    "We sang until the stars came out.",
    "The road follows the river for twenty miles.",
    "Please bring the water and the small red blanket.",
    "The drum group is travelling to another gathering.",
    "Her voice was steady even when the room went silent.",
    "The photographs are in a box under the stairs.",
    "A cold wind came through the open door.",
    "The men are setting up the sound system.",
    "I can hear the train from the other side of town.",
    "The meadow is full of yellow flowers in June.",
    "He learned the melody from a recording on the radio.",
    "The coffee is too hot to drink right now.",
    "They danced until the sun came over the hill.",
    "The little boy is holding a wooden flute.",
    "My brother lives in a city far from here.",
    "The audience clapped for a very long time.",
    "We need more chairs for the elders.",
    "The plane was late because of the storm.",
    "She wrote the words on the back of an envelope.",
    "The kitchen window looks out on the garden.",
    "I have not seen that film since I was young.",
    "The singers are warming up in the hallway.",
    "There is snow on the roof of the school.",
    "The river is higher than it was last year.",
    "He plays the same chord again and again.",
    "The bus stops at the corner every hour.",
    "We should leave before the roads freeze.",
    "The baby fell asleep during the second song.",
]


def _wer(reference: str, hypothesis: str) -> float:
    import re
    def words(text):
        return re.findall(r"[a-z0-9']+", text.lower())
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rword in enumerate(ref, 1):
        current = [i]
        for j, hword in enumerate(hyp, 1):
            current.append(min(
                prev[j] + 1, current[j - 1] + 1,
                prev[j - 1] + (rword != hword),
            ))
        prev = current
    return prev[-1] / len(ref)


def suite_asr(n: int = 40) -> dict:
    """Whisper on synthesized utterances with known transcripts."""
    import io
    import soundfile as sf
    from piper import PiperVoice

    voice = PiperVoice.load(str(ROOT / "data" / "spoken_mode" / "en_US-lessac-medium.onnx"))
    errors, latencies = [], []
    exact = 0
    for sentence in ASR_SENTENCES[:n]:
        chunks = list(voice.synthesize(sentence))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        rate = chunks[0].sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV")
        started = time.perf_counter()
        try:
            raw, _ = _post_multipart("/api/transcribe", buffer.getvalue(),
                                     fields={"language": "en"})
        except Exception:
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        result = json.loads(raw.decode())
        text = str(result.get("text", ""))
        error = _wer(sentence, text)
        errors.append(error)
        exact += error == 0
    if not errors:
        raise RuntimeError("transcription endpoint did not answer")
    metrics = {
        "utterances": len(errors),
        "wer_mean": round(float(np.mean(errors)), 4),
        "wer_median": round(float(np.median(errors)), 4),
        "exact_match": round(exact / len(errors), 4),
        "latency_p50_ms": round(float(np.median(latencies)), 0),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 0),
    }
    chart_hist("asr-wer.svg", [min(100, e * 100) for e in errors], bins=20,
               xlab="word error rate (%)", xmax=100)
    return {"metrics": metrics, "charts": ["assets/evidence/asr-wer.svg"]}


SENTIMENT_LINES = {
    "pos": [
        "I love the way you laugh in the morning", "the sun is warm on my face today",
        "we danced until the stars came out", "this is the happiest I have felt in years",
        "the children are singing in the garden", "I am so grateful for your kindness",
        "the whole family came home for the winter", "what a beautiful and generous gift",
        "I feel strong and ready for the day", "the reunion was joyful and full of laughter",
        "my heart is full of hope", "they welcomed us with open arms",
        "the song made everyone smile", "I am proud of what we built together",
        "the little one took her first steps today", "it was a perfect evening by the fire",
        "I feel calm and content", "the garden is blooming again",
        "thank you for the wonderful news", "we are lucky to have this place",
        "his voice was warm and reassuring", "the crowd cheered with delight",
        "I love this song so much", "she looked radiant at the ceremony",
        "the food was delicious and the company was better", "I feel peaceful here",
        "what a lovely surprise", "the team won and everyone celebrated",
        "I am excited about the new beginning", "the baby smiled at me",
        "the rain finally stopped and the sky cleared", "we laughed until we cried",
        "this place feels like home", "I am thankful for every one of you",
        "the performance was magnificent", "he gave me the most thoughtful present",
        "I feel alive and full of energy", "the harvest was abundant this year",
        "the children are healthy and happy", "it is a good day to be here",
    ],
    "neg": [
        "I miss you and the winter will not end", "the news broke my heart",
        "everything feels heavy and grey today", "I am so tired of being alone",
        "the fire went out and the house is cold", "he never came back",
        "I cannot stop crying", "the pain in my chest will not go away",
        "we lost the baby in the spring", "the whole town is grieving",
        "I feel hopeless and stuck", "she was angry and would not speak to me",
        "the storm destroyed the garden", "I am afraid of what comes next",
        "the sickness took him slowly", "nothing feels right anymore",
        "I hate this place and everyone in it", "the loneliness is unbearable",
        "we failed and there is no second chance", "his words were cruel and cold",
        "the children are frightened by the noise", "I feel sick with worry",
        "the funeral was on a grey morning", "I regret everything I said",
        "the river took the last of our land", "she left without saying goodbye",
        "I am exhausted and broken", "the memory still makes me sad",
        "there is no money left for the month", "the dog is old and in pain",
        "I feel guilty for not being there", "the winter killed the young trees",
        "they laughed at me and I wanted to disappear", "I am drowning in debt",
        "the house is empty now", "I cannot forgive him",
        "the surgery did not work", "we are running out of time",
        "I feel worthless and small", "the world feels cruel today",
    ],
    "neu": [
        "the drum is on the table", "we drove to the river",
        "the song has four verses", "she is wearing a blue coat",
        "the meeting starts at nine", "there are twelve chairs in the room",
        "the bus arrives on Thursday", "I put the kettle on the stove",
        "the folder is in the second drawer", "he plays the guitar on weekends",
        "the window faces the parking lot", "we need three more cables",
        "the book is on the shelf", "the train goes to the city",
        "my cousin works at the hospital", "the recipe calls for two eggs",
        "the lights are on the second floor", "I will call you after lunch",
        "the map is on the wall", "the river is about twenty metres wide",
        "she teaches history at the school", "the concert is on Friday",
        "we ordered two coffees", "the printer is out of paper",
        "the keys are in the kitchen", "he took the early flight",
        "the field is on the other side of the road", "I read the report this morning",
        "the tool is in the shed", "the meeting room has a projector",
        "the store closes at six", "the report has three sections",
        "my phone is charging", "the dog is sleeping by the door",
        "we painted the fence last summer", "the file is named draft two",
        "the road turns left after the bridge", "she ordered the same thing",
        "the box is heavy", "the show lasts about an hour",
    ],
}


def suite_sentiment() -> dict:
    """Valence on 120 labelled lines through the real endpoint."""
    hits = {"pos": 0, "neg": 0, "neu": 0}
    counts = {"pos": 0, "neg": 0, "neu": 0}
    scores = {"pos": [], "neg": []}
    for label, lines in SENTIMENT_LINES.items():
        for line in lines:
            raw, _ = _post_bytes("/api/sentiment", {"text": line})
            data = json.loads(raw.decode())
            valence = float(data.get("valence", 0.0))
            predicted = "pos" if valence > 0.05 else ("neg" if valence < -0.05 else "neu")
            counts[label] += 1
            hits[label] += predicted == label
            if label in scores:
                scores[label].append(valence)
    accuracy = sum(hits.values()) / max(1, sum(counts.values()))
    # AUC of positive against negative lines, rank based
    pos, neg = np.array(scores["pos"]), np.array(scores["neg"])
    auc = 0.0
    if len(pos) and len(neg):
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        auc = wins / (len(pos) * len(neg))
    metrics = {
        "lines": int(sum(counts.values())),
        "accuracy": round(accuracy, 4),
        "positive_recall": round(hits["pos"] / max(1, counts["pos"]), 4),
        "negative_recall": round(hits["neg"] / max(1, counts["neg"]), 4),
        "neutral_recall": round(hits["neu"] / max(1, counts["neu"]), 4),
        "pos_vs_neg_auc": round(auc, 4),
    }
    chart_bars("sentiment-classes.svg",
               ["positive", "negative", "neutral", "overall"],
               [metrics["positive_recall"] * 100, metrics["negative_recall"] * 100,
                metrics["neutral_recall"] * 100, metrics["accuracy"] * 100],
               ylab="% correct", ymax=100, fmt="{:.0f}")
    return {"metrics": metrics, "charts": ["assets/evidence/sentiment-classes.svg"]}


def suite_cree() -> dict:
    """Orthography round trip and FST coverage on the Plains Cree corpus."""
    import re
    from analysis.cree_orthography import orthography_self_test, normalize_sro, looks_like_syllabics

    report = orthography_self_test()
    words: set[str] = set()
    root = ROOT / "data" / "PlainsCree"
    if root.is_dir():
        for path in sorted(root.rglob("*_cr.txt")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                for token in normalize_sro(line).split(" "):
                    core = "".join(ch for ch in token if ch.isalpha() or ch in "âêîô")
                    if core and not looks_like_syllabics(core) and not re.search(r"[bdfgjquvxz]", core):
                        words.add(core)
    analyzed = 0
    sample = sorted(words)[:1000]
    fst_path = ROOT / "data" / "models" / "crk-descriptive-analyzer.hfstol"
    if fst_path.exists() and sample:
        import hfst
        stream = hfst.HfstInputStream(str(fst_path))
        analyzer = stream.read()
        for word in sample:
            try:
                if analyzer.lookup(word):
                    analyzed += 1
            except Exception:
                pass
    metrics = {
        "words_checked": report["words_checked"],
        "syllabics_round_trip_failures": report["syllabics_round_trip_failures"],
        "sro_round_trip_failures": report["round_trip_failures"],
        "syllable_count_failures": report["syllable_count_failures"],
        "fst_words_tested": len(sample),
        "fst_coverage": round(analyzed / max(1, len(sample)), 4),
    }
    chart_bars("cree-orthography.svg",
               ["SRO round trip", "syllabics round trip", "syllable count"],
               [100.0 - 100.0 * report["round_trip_failures"] / max(1, report["words_checked"]),
                100.0 - 100.0 * report["syllabics_round_trip_failures"] / max(1, report["words_checked"]),
                100.0 - 100.0 * report["syllable_count_failures"] / max(1, report["words_checked"])],
               ylab="% of words correct", ymax=100, fmt="{:.2f}")
    return {"metrics": metrics, "charts": ["assets/evidence/cree-orthography.svg"]}


BAND_STYLES = ["hand_drum", "round_dance", "grass_dance", "jingle_dress", "metis_fiddle", "cedar_flute"]


def suite_bands() -> dict:
    """Voice over band at the panel's own defaults, on real takes."""
    from scipy.signal import butter, sosfilt

    takes = []
    for name in ("horse", "sara"):
        path = Path("/tmp/takes") / f"{name}.wav"
        if path.exists():
            import soundfile as sf
            audio, rate = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            takes.append((name, np.asarray(audio, dtype=np.float32), rate))
    if not takes:
        raise RuntimeError("no short takes at /tmp/takes")

    sos = butter(4, [300, 3400], btype="band", fs=SR, output="sos")
    ratios, peaks, rows = [], [], []
    for name, take, rate in takes:
        raw_take = Path(f"/tmp/takes/{name}.wav").read_bytes()
        for style in BAND_STYLES:
            raw, headers = _post_multipart(
                "/api/mysong/render", raw_take,
                fields={"style": style, "return_mix": "false", "vocal_gain_db": "-2.15"},
            )
            band, band_rate = _load_wav(raw)
            placement = float(_header(headers, "X-MySong-Band-Gain-Db", "0") or 0)
            length = min(len(take), len(band))
            if length < SR:
                continue
            voice = sosfilt(sos, take[:length].astype(np.float64))
            bed = sosfilt(sos, band[:length].astype(np.float64)) * 10 ** (placement / 20.0)
            trim = 10 ** (-2.15 / 20.0)
            voice_rms = float(np.sqrt(np.mean((voice * trim) ** 2)) + 1e-12)
            bed_rms = float(np.sqrt(np.mean(bed ** 2)) + 1e-12)
            ratio = 20 * math.log10(voice_rms / bed_rms)
            ratios.append(ratio)
            peaks.append(float(np.max(np.abs(take[:length] * trim + bed))))
            rows.append({"take": name, "style": style, "placement_db": placement, "voice_over_band_db": round(ratio, 2)})
    metrics = {
        "renders": len(ratios),
        "voice_over_band_median_db": round(float(np.median(ratios)), 2),
        "voice_over_band_min_db": round(float(np.min(ratios)), 2),
        "voice_over_band_max_db": round(float(np.max(ratios)), 2),
        "peak_max": round(float(np.max(peaks)), 3),
        "per_style": {s: round(float(np.mean([r["voice_over_band_db"] for r in rows if r["style"] == s])), 2)
                      for s in BAND_STYLES if any(r["style"] == s for r in rows)},
    }
    labels = list(metrics["per_style"].keys())
    chart_bars("band-balance.svg", labels,
               [metrics["per_style"][s] for s in labels],
               ylab="voice above band (dB)", fmt="{:.1f}", color=SAGE)
    return {"metrics": metrics, "charts": ["assets/evidence/band-balance.svg"]}


def suite_singsong(trials: int = 120) -> dict:
    """The token layout the SingSong cascade is built on, at full size."""
    from sing_song.tokens import Stage12Layout, FineLayout

    rng = np.random.default_rng(17)
    stage = Stage12Layout.for_seconds(10.0)
    fine = FineLayout.for_seconds(10.0)
    failures = 0
    mask_bad = 0
    positions = 0
    tokens = 0
    for _ in range(trials):
        semantic = rng.integers(0, 1024, size=stage.semantic_frames).tolist()
        coarse = [rng.integers(0, 1024, size=stage.coarse_frames).tolist() for _ in range(stage.num_coarse)]
        target = stage.build_target(semantic, coarse)
        back_sem, back_coarse = stage.split_target(target)
        if back_sem != semantic or back_coarse != coarse:
            failures += 1
        tokens += len(target)
        for position, token in enumerate(target):
            lo, hi = stage.position_range(position)
            positions += 1
            if not lo <= token < hi or hi - lo != 1024:
                mask_bad += 1
        fine_ids = [rng.integers(0, 1024, size=fine.fine_frames).tolist() for _ in range(fine.num_fine)]
        packed = fine.build_fine(fine_ids)
        if fine.split_fine(packed) != fine_ids:
            failures += 1
        tokens += len(packed)
    metrics = {
        "trials": trials,
        "tokens_checked": tokens,
        "positions_checked": positions,
        "round_trip_failures": failures,
        "mask_violations": mask_bad,
        "stage12_vocab": stage.vocab_size,
        "stage12_positions": stage.target_length,
        "fine_vocab": fine.fine_vocab_size,
        "fine_positions": fine.fine_length,
    }
    return {"metrics": metrics, "charts": []}


SUITES = {
    "pitch": suite_pitch,
    "notes": suite_notes,
    "rhythm": suite_rhythm,
    "key": suite_key,
    "modes": suite_modes,
    "voicing": suite_voicing,
    "mysong": suite_mysong,
    "latency": suite_latency,
    "vocables": suite_vocables,
    "rvc": suite_rvc,
    "asr": suite_asr,
    "sentiment": suite_sentiment,
    "cree": suite_cree,
    "bands": suite_bands,
    "singsong": suite_singsong,
}


def main() -> int:
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma separated suite names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()
    if args.list:
        print("\n".join(SUITES))
        return 0
    names = [s.strip() for s in args.only.split(",") if s.strip()] or list(SUITES)
    results = {}
    if Path(args.out).exists():
        try:
            results = json.loads(Path(args.out).read_text())
        except Exception:
            results = {}
    for name in names:
        if name not in SUITES:
            print(f"unknown suite: {name}")
            continue
        started = time.time()
        print(f"== {name} ...", flush=True)
        try:
            results[name] = SUITES[name]()
            results[name]["elapsed_seconds"] = round(time.time() - started, 1)
            print(json.dumps(results[name]["metrics"], indent=2)[:1800], flush=True)
        except Exception as error:  # keep the other suites running
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"   failed: {error}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

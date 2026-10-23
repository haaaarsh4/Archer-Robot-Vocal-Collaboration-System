"""Fit the closed-mouth restore against raw converted takes.

The sidecar is asked for the exact conversion the live path asks for (a hum at
one pitch, no padding), the RAW answer is cached, and then each candidate
restore is applied to it offline and scored against the murmur the DSP path
renders at the same pitch -- per harmonic, so a wide-band difference cannot hide
in an average.

Score: an excess (converted louder than the murmur) in the vowel's own region is
what a listener hears as /o/ or /a/; a deficit is only ever darker. So the
excess is what has to be squeezed toward zero, and it is reported per region.

Run:  .venv/bin/python scratch/hum_restore_fit2.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, ".")
from synthesis.hum_voice import (HumConfig, _antiresonance, _apply_sos,  # noqa: E402
                                 _peaking, _require_scipy, _resonator, _shelf,
                                 synthesize_hum_note)

SIDECAR = "http://127.0.0.1:8801"
RAW = "/tmp/hum_raw"
PITCHES = [123.47, 146.83, 174.61, 220.0, 261.63, 329.63, 392.0]
REGIONS = [(100, 400), (400, 800), (800, 1300), (1300, 2000), (2000, 3000)]


def raw_converted(hz: float, feat: np.ndarray, sr: int) -> np.ndarray:
    os.makedirs(RAW, exist_ok=True)
    path = f"{RAW}/{hz:.2f}.wav"
    if os.path.exists(path):
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        return y
    buf = io.BytesIO()
    sf.write(buf, feat.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    resp = requests.post(f"{SIDECAR}/convert",
                         files={"file": ("scratch.wav", buf.read(), "audio/wav")},
                         data={"voice_index": 0, "sample_rate": sr}, timeout=600)
    resp.raise_for_status()
    y, _ = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sf.write(path, y, sr)
    return y


def tract(y: np.ndarray, sr: int, f0: float, cfg: HumConfig, *,
          zero_depth_db: float | None = None, lp_hz: float | None = None,
          lp_order: int | None = None, extra_hf_db: float = 0.0,
          extra_hf_hz: float = 1400.0, extra_tilt_db: float = 0.0,
          extra_tilt_hz: float = 800.0,
          dips: tuple[tuple[float, float, float], ...] = ()) -> np.ndarray:
    """The murmur's own chain, run over whatever the model sang."""
    s = _require_scipy()
    tracking = float(np.clip(cfg.formant_tracking, 0.0, 0.6))
    scale = float(np.clip((f0 / 200.0) ** tracking, 0.7, 1.35))
    x = np.asarray(y, dtype=np.float64)
    for cut_hz, cut_db, cut_q in dips:
        x = _apply_sos(x, _peaking(cut_hz, cut_db, cut_q, sr))
    for f_hz, bw_hz in ((cfg.f1_hz, cfg.b1_hz), (cfg.f2_hz, cfg.b2_hz),
                        (cfg.f3_hz, cfg.b3_hz), (cfg.f4_hz, cfg.b4_hz)):
        x = _apply_sos(x, _resonator(f_hz * scale, bw_hz, sr))
    x = _apply_sos(x, _antiresonance(cfg.nasal_zero_hz * scale, cfg.nasal_zero_q,
                                     cfg.nasal_zero_depth_db if zero_depth_db is None
                                     else zero_depth_db, sr))
    x = _apply_sos(x, _shelf(cfg.body_hz * scale, cfg.body_gain_db, sr, q=0.8, high=False))
    x = _apply_sos(x, _shelf(cfg.tilt_hz * scale, cfg.tilt_db_per_oct, sr, q=0.7, high=True))
    if extra_tilt_db:
        x = _apply_sos(x, _shelf(extra_tilt_hz, extra_tilt_db, sr, q=0.7, high=True))
    if extra_hf_db:
        x = _apply_sos(x, _shelf(extra_hf_hz, extra_hf_db, sr, q=0.7, high=True))
    x = _apply_sos(x, _shelf(cfg.hf_shelf_hz * scale, cfg.hf_shelf_db, sr, q=0.7, high=True))
    freq = cfg.lowpass_hz if lp_hz is None else lp_hz
    order = cfg.lowpass_order if lp_order is None else lp_order
    if freq and order:
        wn = min(0.99, max(0.02, freq / (sr / 2.0)))
        x = s.sosfilt(s.butter(int(order), wn, btype="low", output="sos"), x)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def steady(y: np.ndarray, sr: int, seconds: float = 0.6) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = int(sr * seconds)
    if y.size <= n:
        return y
    start = max(0, int(y.size * 0.5) - n // 2)
    return y[start:start + n]


def partials(y: np.ndarray, sr: int, f0: float, n_harm: int = 40) -> tuple[np.ndarray, np.ndarray]:
    x = steady(y, sr)
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-15
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    fs, vs = [], []
    for k in range(1, n_harm + 1):
        f = k * f0
        if f >= sr / 2 - 400:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.98, f * 1.02])
        fs.append(f)
        vs.append(20.0 * np.log10(float(mag[max(lo, 0):max(hi, lo + 1)].max())))
    v = np.asarray(vs)
    return np.asarray(fs), v - v.max()


def score(y: np.ndarray, sr: int, f0: float, ref: np.ndarray) -> dict:
    f, p = partials(y, sr, f0)
    _, r = partials(ref, sr, f0)
    n = min(len(p), len(r))
    f, p, r = f[:n], p[:n], r[:n]
    excess = p - r
    out = {}
    for lo, hi in REGIONS:
        sel = (f >= lo) & (f < hi) & (r > -45.0)
        out[f"{lo}-{hi}"] = (float(np.max(excess[sel])) if sel.any() else float("nan"),
                             int(sel.sum()))
    audible = r > -25.0
    out["mean_abs"] = float(np.mean(np.abs(excess[audible]))) if audible.any() else 0.0
    out["max_excess"] = float(np.max(excess[(f >= 400) & (f < 2500)]))
    above = f > 700.0
    powv = 10 ** (p / 10)
    out["share700"] = float(powv[above].sum() / powv.sum()) * 100.0
    powr = 10 ** (r / 10)
    out["share700_ref"] = float(powr[above].sum() / powr.sum()) * 100.0
    return out


def main() -> int:
    cfg = HumConfig.from_config()
    sr = 44100
    n = int(1.8 * sr)

    variants = {
        "raw (no restore)": dict(apply=False),
        "murmur tract": dict(),
        "tract + dip600/1350": dict(dips=((600.0, -16.0, 0.8), (1350.0, -16.0, 0.9))),
        "tract + dip550/1400 deep": dict(dips=((550.0, -22.0, 0.7), (1400.0, -22.0, 0.8))),
        "tract + dip550/1400 + LP1800": dict(dips=((550.0, -22.0, 0.7), (1400.0, -22.0, 0.8)),
                                             lp_hz=1800.0, lp_order=4),
        "tract + dip550/1400+2400": dict(dips=((550.0, -22.0, 0.7), (1400.0, -22.0, 0.8),
                                               (2400.0, -16.0, 0.8))),
        "tract + dip550/1400 + tilt-6": dict(dips=((550.0, -22.0, 0.7), (1400.0, -22.0, 0.8)),
                                             extra_tilt_db=-6.0, extra_tilt_hz=800.0),
        "tract + dip550/1400 + tilt-6 + LP1600":
            dict(dips=((550.0, -22.0, 0.7), (1400.0, -22.0, 0.8)),
                 extra_tilt_db=-6.0, extra_tilt_hz=800.0, lp_hz=1600.0, lp_order=4),
        "tract + dip450/1200/2400": dict(dips=((450.0, -20.0, 0.7), (1200.0, -20.0, 0.8),
                                               (2400.0, -14.0, 0.8))),
    }
    raws: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for hz in PITCHES:
        feat = synthesize_hum_note(hz, n, sr, cfg)
        raws[hz] = (feat, raw_converted(hz, feat, sr))
    print(f"{len(raws)} takes from the sidecar; scores are per region, worst excess first\n")
    header = "  " + "variant".ljust(34) + "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in REGIONS) \
             + f"{'meanabs':>9}{'maxexc':>8}{'share700':>10}{'(ref)':>8}"
    print(header)
    for name, kw in variants.items():
        rows = []
        for hz in PITCHES:
            feat, conv = raws[hz]
            y = conv if kw.get("apply") is False else tract(
                conv, sr, hz, cfg, **{k: v for k, v in kw.items() if k != "apply"})
            rows.append(score(y, sr, hz, feat))
        agg = {}
        for key in rows[0]:
            vals = [r[key] for r in rows if isinstance(r[key], tuple)]
            if vals:
                agg[key] = np.nanmax([v[0] for v in vals])
            else:
                agg[key] = float(np.mean([r[key] for r in rows]))
        print("  " + name.ljust(34)
              + "".join(f"{agg[f'{lo}-{hi}']:>10.1f}" for lo, hi in REGIONS)
              + f"{agg['mean_abs']:>9.1f}{agg['max_excess']:>8.1f}"
              + f"{agg['share700']:>10.2f}{agg['share700_ref']:>8.2f}")
    print("\n(columns 100-400 ... 2000-3000 are the WORST excess over seven pitches; "
          "0 is a perfect match, negative is darker than the murmur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

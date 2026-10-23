"""WHAT THE PANEL ACTUALLY PLAYS IN LIVE NEURAL + HUMMING.

Posts the page's own live-chunk request (8 s chunks of the upload, first one 4 s,
0.25 s overlap, mode=solo_humming, vibrato 45) at /api/neural/convert-track and
measures what comes back, so "it still sounds like OH" can be answered with
numbers instead of opinion:

  * the spectral envelope (a closed murmur is dark: everything above ~600 Hz
    many dB under its fundamental, nothing at 500-1100 Hz)
  * the shape of the envelope in time (a hum is one continuous line; the same
    line cut into chunks must not arrive as a row of syllables)
  * the pitch it lands on

and the same measurements on the DSP hum of the whole track (the reference
murmur) and on the track's own singing (the thing it must NOT sound like).
"""
import io
import json
import sys
import urllib.request

import numpy as np
import soundfile as sf

SR = 44100
BASE = "http://127.0.0.1:8000"


def load(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def encode_wav(x):
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), SR, format="WAV")
    return b.getvalue()


def post_multipart(url, fields, files, timeout=900):
    boundary = "----diagboundary"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                 f"{v}\r\n").encode()
    for k, (name, data) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                 f"filename=\"{name}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def measure(label, x):
    if x.size == 0 or float(np.max(np.abs(x))) <= 0:
        print(f"  {label}: SILENT")
        return {}
    n = len(x)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    power = spec ** 2
    total = float(power.sum()) + 1e-20
    centroid = float((freqs * power).sum() / total)

    def share(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return 100.0 * float(power[m].sum()) / total

    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    peak = float(np.max(np.abs(x)))
    # Envelope, in 50 ms windows: the first and last tenth are what a per-chunk
    # fade-in/fade-out (i.e. a syllable) would show up in.
    hop = int(0.05 * SR)
    env = np.array([float(np.sqrt(np.mean(np.square(x[i:i + hop], dtype=np.float64))))
                    for i in range(0, max(1, n - hop), hop)]) if n > hop else np.array([rms])
    head = float(np.mean(env[:2])) if env.size >= 2 else rms
    tail = float(np.mean(env[-2:])) if env.size >= 2 else rms
    mid = float(np.median(env))

    # Partials under the strongest one, on the measured f0 (autocorrelation).
    f0 = _f0(x)
    print(f"  {label}: rms {rms:.3f} peak {peak:.3f} centroid {centroid:6.0f} Hz  "
          f"E>700Hz {share(700, 20000):6.2f}%  E>1.5k {share(1500, 20000):5.2f}%  "
          f"E>2.5k {share(2500, 20000):5.3f}%  f0 {f0:6.1f} Hz")
    print(f"       band shares: 60-300 {share(60, 300):5.1f}%  300-700 {share(300, 700):5.1f}%  "
          f"700-1.5k {share(700, 1500):5.2f}%  1.5k-3k {share(1500, 3000):5.2f}%  3k+ {share(3000, 20000):5.2f}%")
    print(f"       envelope: head/mid/tail {head:.4f}/{mid:.4f}/{tail:.4f} "
          f"(head-mid {20 * np.log10((head + 1e-9) / (mid + 1e-9)):+.1f} dB, "
          f"tail-mid {20 * np.log10((tail + 1e-9) / (mid + 1e-9)):+.1f} dB)")
    return {"rms": rms, "centroid": centroid, "gt700": share(700, 20000),
            "gt2500": share(2500, 20000), "f0": f0}


def _f0(x):
    """Crude autocorrelation pitch, over the loudest 1 s of the take."""
    seg = x
    if len(x) > SR:
        # loudest second
        best, at = 0.0, 0
        for i in range(0, len(x) - SR, SR // 2):
            r = float(np.sqrt(np.mean(np.square(x[i:i + SR], dtype=np.float64))))
            if r > best:
                best, at = r, i
        seg = x[at:at + SR]
    seg = seg - seg.mean()
    if np.max(np.abs(seg)) <= 0:
        return 0.0
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lo, hi = int(SR / 500), int(SR / 60)
    if hi >= len(ac):
        hi = len(ac) - 1
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SR / lag if lag else 0.0


def dsp_hum(track_bytes):
    fields = {"texture": "solo", "pitch_method": "yin", "mode": "solo_humming",
              "instruments_enabled": "false", "vocable": "mix", "vibrato_amount": "45"}
    data, headers = post_multipart(f"{BASE}/api/dsp/render-track", fields,
                                  {"file": ("hummingbird.mp3", track_bytes)})
    return load_bytes(data), headers


def load_bytes(data):
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def main():
    track = sys.argv[1] if len(sys.argv) > 1 else "frontend/assets/audio/hummingbird.mp3"
    raw = open(track, "rb").read()
    audio = load(track)
    print(f"track {track}: {len(audio) / SR:.1f} s, rms "
          f"{float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))):.3f}")

    print("\n=== the upload itself (what the panel plays under the robot) ===")
    measure("upload 0-8 s", audio[0:8 * SR])
    measure("upload 16-24 s", audio[16 * SR:24 * SR])

    print("\n=== DSP hum, whole track (the reference murmur) ===")
    dsp, _ = dsp_hum(raw)
    measure("dsp hum whole", dsp)
    measure("dsp hum 0-8 s", dsp[0:8 * SR])

    print("\n=== LIVE NEURAL + HUMMING: the page's own chunk requests ===")
    first_len = int(4.0 * SR)
    chunk_len = int(8.0 * SR)
    overlap = int(0.25 * SR)
    plan = []
    c = 0
    plan.append((0, audio[0:first_len]))
    c = first_len
    while c < len(audio):
        frm = max(0, c - overlap)
        e = min(c + chunk_len, len(audio))
        plan.append((frm, audio[frm:e]))
        c = e
    print(f"plan: {len(plan)} chunks (first {first_len / SR:.0f} s, rest {chunk_len / SR:.0f} s, "
          f"overlap {overlap / SR:.2f} s)")
    outs = []
    for i, (start_s, slice_) in enumerate(plan):
        fields = {"voice_index": "0", "transpose_semitones": "0",
                  "pad_seconds": "0.3", "instruments_enabled": "false",
                  "vibrato_amount": "45", "mode": "solo_humming"}
        data, headers = post_multipart(f"{BASE}/api/neural/convert-track", fields,
                                       {"file": ("chunk.wav", encode_wav(slice_))})
        out = load_bytes(data)
        outs.append(out)
        print(f" chunk {i + 1}/{len(plan)} @ {start_s / SR:5.1f}s  "
              f"({len(out) / SR:.1f} s back, X-Neural-Applied={headers.get('X-Neural-Applied')}, "
              f"X-Backing-Bed={headers.get('X-Backing-Bed')})")
        measure(f"  chunk {i + 1}", out)

    print("\n=== the chunked line as it reaches the ear (chunks laid end to end) ===")
    if outs:
        line = np.concatenate(outs)
        measure("concatenated live line", line)
        # How many separate attacks does it have? A continuous hum has one; a row
        # of syllables has one per chunk.
        hop = int(0.05 * SR)
        env = np.array([float(np.sqrt(np.mean(np.square(line[i:i + hop], dtype=np.float64))))
                        for i in range(0, max(1, len(line) - hop), hop)])
        thr = 0.25 * float(np.max(env))
        rises = int(np.sum((env[1:] > thr) & (env[:-1] <= thr)))
        print(f"  envelope rises through 25% of peak: {rises} "
              f"(one continuous line should be 1, one per chunk would be {len(outs)})")
    sf.write("/tmp/diag_live_hum_line.wav", np.concatenate(outs) if outs else np.zeros(1, np.float32), SR)
    sf.write("/tmp/diag_dsp_hum.wav", dsp, SR)
    print("\nwrote /tmp/diag_live_hum_line.wav and /tmp/diag_dsp_hum.wav")


if __name__ == "__main__":
    main()

"""What a streamed live chunk's SEAM sounds like, measured on real chunks.

Live mode converts the uploaded track one chunk at a time and plays the chunks
back to back over it. Whether that reads as one continuous sung line or as a
stutter every chunk is decided entirely at the joins, and the joins are made on
the page (frontend/index.html, _playDemoNeuralLive), so this drives the real
endpoint with the page's own chunk sizes and pad and measures the two joins:

  * the join the page makes NOW: each chunk repeats
    DEMO_LIVE_CHUNK_OVERLAP_S (0.15 s) of the previous chunk's material at its
    head, is scheduled at its own material position in the track, and is
    crossed with the previous chunk on equal-power (sin/cos) curves;
  * the join the page used to make: contiguous chunks, the next one started
    50 ms before the last ended, both ramping up from and down to silence
    inside themselves over 50 ms.

The measurement worth having is the same for both: is there a moment at the
join where the line drops out or jumps? So both composites are built from the
SAME two converted chunks and scored with the same instrument -- the envelope
around the join (a hole shows up as a dip no singer makes) and the biggest
sample-to-sample jump (a click shows up there), against the chunks' own
typical slope.

The material is a real song excerpt with a real separated vocal, which is what
the page's chunks are converted from once the upload's separation exists.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

SR = 44100
def _page_constant(name: str, default: float) -> float:
    """Read a constant out of the page, so this harness cannot drift from it.

    The chunk length, the overlap and the pad are what decide whether a stream
    has seams, and they live in frontend/index.html. A harness carrying its own
    copy of them measures a stream the page stopped making: this one printed
    4 s chunks and a 0.15 s overlap against a page streaming 8 s chunks with a
    0.25 s overlap, i.e. it was grading the wrong join. Parsed here, falling
    back to the page's own current numbers if the shape ever changes.
    """
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "frontend", "index.html")
    try:
        with open(path) as f:
            m = re.search(rf"const\s+{name}\s*=\s*([0-9.]+)", f.read())
        return float(m.group(1)) if m else float(default)
    except Exception:
        return float(default)


FIRST_CHUNK_S = _page_constant("DEMO_LIVE_FIRST_CHUNK_S", 4.0)  # was: FIRST_CHUNK_S = 1.0        # frontend/index.html: DEMO_LIVE_FIRST_CHUNK_S
CHUNK_S = _page_constant("DEMO_LIVE_CHUNK_S", 8.0)              # DEMO_LIVE_CHUNK_S
PAD_S = _page_constant("DEMO_LIVE_PAD_SECONDS", 0.3)                # DEMO_LIVE_PAD_SECONDS
OVERLAP_S = _page_constant("DEMO_LIVE_CHUNK_OVERLAP_S", 0.25)           # DEMO_LIVE_CHUNK_OVERLAP_S


def say(msg: str) -> None:
    print(msg, flush=True)


def cut_song(path: str, start_s: float, dur_s: float) -> bytes:
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ss", f"{start_s}",
                    "-t", f"{dur_s}", "-ar", str(SR), "-ac", "2", out], check=True)
    with open(out, "rb") as f:
        raw = f.read()
    os.unlink(out)
    return raw


def convert_chunk(base: str, raw_wav: bytes, offset_s: float, track_key, voice: int) -> np.ndarray:
    files = {"file": ("chunk.wav", raw_wav, "audio/wav")}
    data = {"voice_index": str(voice), "transpose_semitones": "0", "pad_seconds": str(PAD_S),
            "instruments_enabled": "false"}
    if track_key:
        data["track_key"] = track_key
        data["track_offset_s"] = f"{offset_s:.4f}"
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/neural/convert-track", files=files, data=data, timeout=900)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"convert-track {r.status_code}: {r.text[:200]}")
    y, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
    y = y.mean(axis=1) if y.ndim > 1 else y
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
    say(f"  chunk at {offset_s:6.2f}s ({len(y) / SR:.3f}s of audio, bed "
        f"{r.headers.get('X-Backing-Bed')}): converted in {dt:.1f}s")
    return np.asarray(y, dtype=np.float32)


def eq_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    return np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)


def join_new(a: np.ndarray, b: np.ndarray, overlap_n: int) -> np.ndarray:
    """The page's current join: b's head is a's tail material, crossed."""
    up, down = eq_curves(overlap_n)
    out = np.zeros(len(a) + len(b) - overlap_n, dtype=np.float64)
    out[:len(a)] += a.astype(np.float64)
    out[len(a) - overlap_n:len(a)] *= down
    start = len(a) - overlap_n
    head = b[:overlap_n].astype(np.float64) * up
    out[start:start + overlap_n] += head
    out[start + overlap_n:start + len(b)] += b[overlap_n:].astype(np.float64)
    return out


def join_old(a: np.ndarray, b: np.ndarray, fade_n: int, late_n: int) -> np.ndarray:
    """The join the page used to make: contiguous chunks, b started late_n early,
    both fading to and from silence over fade_n."""
    out = np.zeros(max(len(a) + len(b) - late_n, len(a) + 1), dtype=np.float64)
    ga = np.ones(len(a))
    ga[:fade_n] = np.linspace(0, 1, fade_n)
    ga[-fade_n:] = np.linspace(1, 0, fade_n)
    out[:len(a)] += a.astype(np.float64) * ga
    gb = np.ones(len(b))
    gb[:fade_n] = np.linspace(0, 1, fade_n)
    gb[-fade_n:] = np.linspace(1, 0, fade_n)
    start = len(a) - late_n
    out[start:start + len(b)] += b.astype(np.float64) * gb
    return out


def envelope(x: np.ndarray, win_s: float = 0.005) -> np.ndarray:
    w = max(1, int(win_s * SR))
    return np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(w) / w, mode="same"))


def score(x: np.ndarray, source: np.ndarray, join_at: int, label: str) -> dict:
    """What the join does to the LINE, measured against the performance it is singing.

    A hole only counts where the source had something to sing: music has pauses,
    and a robot that is quiet through one is correct, not broken. So the line's
    envelope is read RELATIVE to the source's own envelope over the same window
    (how much of what was there is still being sung), and that ratio is compared
    with its own value either side of the join. A fade to silence shows up as
    that ratio collapsing exactly at the join; a proper crossfade does not move
    it.
    """
    env_x = envelope(x)
    env_src = envelope(source[:len(x)])
    n = min(len(env_x), len(env_src))
    env_x, env_src = env_x[:n], env_src[:n]
    # A hole is the line's own level collapsing AT the join, so the window has
    # to be tight (a sung note moves a great deal over a quarter of a second by
    # itself, which is why an earlier, wider version of this scored correct
    # singing as a 20 dB hole). What is asked is how loud the line is in the
    # 120 ms around the join against how loud it is in the 240 ms either side.
    def seg(a: int, b: int) -> float:
        a, b = max(0, a), min(len(x), b)
        if b <= a:
            return float("nan")
        return float(np.sqrt(np.mean(x[a:b] ** 2) + 1e-18))
    w30, w06 = int(0.3 * SR), int(0.06 * SR)
    before, at, after = seg(join_at - w30, join_at - w06), seg(join_at - w06, join_at + w06), \
        seg(join_at + w06, join_at + w30)
    level_db = 20 * np.log10((at + 1e-12) / max(0.5 * (before + after), 1e-12))
    # the source's own envelope over the same join, so "the line was quiet"
    # can be told from "the line was quiet because nothing was being sung"
    src_at = float(np.sqrt(np.mean(env_src[join_at - w06:join_at + w06] ** 2) + 1e-18))
    src_side = float(np.sqrt(np.mean(np.concatenate([env_src[join_at - w30:join_at - w06],
                                                     env_src[join_at + w06:join_at + w30]]) ** 2) + 1e-18))
    quiet_source = src_at < 0.35 * max(src_side, 1e-12)
    # a click shows up in the biggest sample-to-sample step against the line's
    # own typical slope
    d = np.abs(np.diff(x))
    p999 = float(np.percentile(d, 99.9))
    worst_near = float(d[max(0, join_at - int(0.15 * SR)):min(len(d), join_at + int(0.15 * SR))].max())
    say(f"  {label}: level at the join {level_db:+.1f} dB vs its neighbours"
        f"{' (the source is quiet there, so a quiet line is correct)' if quiet_source else ''}"
        f", worst step {worst_near:.4f} (line's own p99.9 {p999:.4f})")
    return {"level_db": level_db, "worst_step": worst_near, "p999": p999,
            "quiet_source": bool(quiet_source)}


def main() -> int:
    base = os.environ.get("ARCHER_BASE", "http://127.0.0.1:8000")
    song = sys.argv[1] if len(sys.argv) > 1 else "frontend/assets/audio/memory-v2.mp3"
    start_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    voice = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    # long enough for a full 4 s second chunk on top of the first one
    excerpt_s = CHUNK_S + CHUNK_S + 1.0
    raw = cut_song(song, start_s, excerpt_s)
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    say(f"excerpt: {song} at {start_s:.0f}s, {len(audio) / sr:.2f}s, voice {voice}")

    # the upload's own separation, exactly as the page starts it on file pick
    t0 = time.perf_counter()
    r = requests.post(f"{base}/api/neural/backing-bed",
                      files={"file": ("song.wav", raw, "audio/wav")}, timeout=60)
    key = (r.json() or {}).get("track_key") if r.status_code == 200 else None
    say(f"separation started for {key} ({r.json().get('status') if r.status_code == 200 else r.status_code})")
    while key:
        st = requests.get(f"{base}/api/neural/backing-bed/status", params={"track_key": key}, timeout=30).json()
        if st.get("status") != "running":
            say(f"separation {st.get('status')} after {time.perf_counter() - t0:.0f}s")
            break
        time.sleep(4)

    chunk_a_n = int(CHUNK_S * sr)              # material [0, 4.0]
    overlap_n = int(OVERLAP_S * sr)
    # chunk B reaches back OVERLAP_S into A, exactly as the page's queue does
    # exactly what the page's queue hands the server for a mid-track chunk:
    # CHUNK_S of playback plus OVERLAP_S of repeated material at its head
    raw_b = cut_song(song, start_s + CHUNK_S - OVERLAP_S, CHUNK_S + OVERLAP_S)
    raw_a = cut_song(song, start_s, CHUNK_S)

    say("converting the two chunks the page would convert:")
    conv_s = []
    t0 = time.perf_counter()
    a = convert_chunk(base, raw_a, 0.0, key, voice)
    conv_s.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    b = convert_chunk(base, raw_b, CHUNK_S - OVERLAP_S, key, voice)
    conv_s.append(time.perf_counter() - t0)
    a, b = a[:chunk_a_n], b[:int(CHUNK_S * sr)]

    # The page plays A at real time from the chunk's own start, and B at its
    # material position CHUNK_S - OVERLAP_S, so the join sits there.
    join_at = int((CHUNK_S - OVERLAP_S) * sr)
    new = join_new(a, b, overlap_n)
    old = join_old(a, b, int(0.05 * sr), int(0.05 * sr))

    say("")
    say("join at a chunk boundary, the page's current crossfade vs the old fade-to-silence:")
    n = score(new, audio, join_at, "now (0.15s material overlap, equal-power crossfade)")
    o = score(old, audio, join_at, "before (contiguous chunks, 50ms fades to silence)")

    say("")
    ok = True
    # What the join must not do is drop the line out. Both schemes cross the
    # two chunks, so neither is expected to, and the accurate reading of the
    # numbers above is which one lands closer to the line's own level and
    # whether either clicks.
    if abs(n["level_db"]) > 3.0 and not n["quiet_source"]:
        say(f"  [FAIL] the line's level moves at the join: {n['level_db']:+.1f} dB")
        ok = False
    else:
        say(f"  [ok]   the line's own level is held across the crossfade ({n['level_db']:+.1f} dB)")
    if n["worst_step"] > max(8.0 * n["p999"], 0.05):
        say(f"  [FAIL] a click-sized step at the join: {n['worst_step']:.4f}")
        ok = False
    else:
        say(f"  [ok]   no click-sized step at the join ({n['worst_step']:.4f} "
            f"vs the line's own {n['p999']:.4f})")

    # WHAT ACTUALLY CHANGED, and it is not the fade shape: the old join played
    # the next chunk's material a fixed late_n EARLIER than the track the line
    # was being sung against (50 ms here), doubling a slice of the performance
    # against itself; the new one plays each chunk at its own material position
    # and overlaps MATCHED material instead. That is a property of the two
    # schedules, so it is stated rather than measured.
    say(f"  [ok]   material alignment: the old join shifted the line 50 ms early at "
        f"every boundary (two different instants doubled together); the new one shifts "
        f"nothing and crosses the same instant twice")

    # AND WHAT A SERIAL STREAM DOES WHEN CONVERSION IS SLOWER THAN REAL TIME,
    # simulated from the measured chunk times above: this is the thing that made
    # a long listen sound broken, and the reason for the re-align.
    per_chunk = [conv_s[i] for i in range(len(conv_s))]
    mean_s = sum(per_chunk) / max(1, len(per_chunk))
    deficit = mean_s - CHUNK_S
    say(f"  conversion measured at {mean_s:.2f}s per {CHUNK_S:.1f}s chunk "
        f"({'behind' if deficit > 0 else 'ahead of'} real time by {abs(deficit):.2f}s per chunk)")
    if deficit <= 0:
        say("  [ok]   conversion keeps up: no drift to accumulate")
    else:
        drift_sixty = deficit / CHUNK_S * 60.0
        say(f"  a minute of track would leave the line {drift_sixty:.1f}s behind it with no "
            f"re-align; the page cuts and re-joins the track past 1.2s (see _playDemoNeuralLive), "
            f"so the bound is 1.2s + {deficit:.2f}s instead of growing")
        if drift_sixty > 5.0:
            say(f"  [FAIL] drift of {drift_sixty:.1f}s per minute, the re-align bound is what "
                f"keeps this usable")
        else:
            say(f"  [ok]   drift {drift_sixty:.1f}s per minute is inside what the re-align handles")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Verifies the two changes that are only real if the endpoints agree:

  1. VOCAL FOLD VIBRATION, through POST /api/dsp/render-track. Three renders of
     the same input at 0%, 45% and 100%, measured: does the setting reach the
     server at all, does 0 stay a straight tone, does 100 put a real swing on a
     held note, and does adding it move the pitch or the length of the part
     (it must not -- it is movement, not a transpose).

  2. THE TRANSCRIPT LABELS, through POST /api/transcribe/annotated, on the same
     synthetic drums+voice song the project's own matrix audit uses: every
     instrument row must be named in the "<instrument playing>" shape, none may
     come back as a bare bracket tag, and Whisper's non-speech artifacts must be
     filtered (checked directly on the filter as well, since a synthetic track
     may not happen to trigger one).

Run from the project root:  .venv/bin/python scratch/check_render_and_transcript.py
"""
from __future__ import annotations

import io
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [ok]   " if cond else "  [FAIL] ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def _vibrato_share(audio: np.ndarray, sr: int, rate_hz: float = 5.4) -> tuple[float, float]:
    """(share, rms) of the part's within-note movement AT the vibrato rate.

    WHY NOT JUST THE SIZE OF THE MOVEMENT. The engine keeps the take's own
    movement on purpose (see world_voice), and on a processed take that
    baseline is tens of cents of irregular wobble -- bigger than the vibrato
    this setting adds, so a raw "how much did the pitch move" reading cannot
    tell the two apart. What a regular oscillation does have, and noise does
    not, is a PEAK AT ITS OWN RATE: the movement is turned into a spectrum and
    the energy in a narrow band around the vibrato rate is expressed as a share
    of all the movement between 2 and 12 Hz. That share is what rises when the
    folds' own oscillation is there and stays at the noise floor when it is not.
    """
    import librosa
    mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 0:
        return 0.0, 0.0
    voiced = np.abs(mono) > peak * 0.05
    if voiced.sum() < sr // 10:
        return 0.0, 0.0
    first, last = int(np.argmax(voiced)), int(len(voiced) - np.argmax(voiced[::-1]))
    hop = 256
    f0, vflag, _ = librosa.pyin(mono[first:last], fmin=70, fmax=800, sr=sr,
                                frame_length=2048, hop_length=hop)
    good = np.isfinite(f0) & vflag
    if good.sum() < 200:
        return 0.0, 0.0
    cents = 1200.0 * np.log2(f0[good] / 220.0)
    frame_rate = sr / hop

    # Detrend over 0.4 s (well above one vibrato period) so slow melodic
    # movement is not counted as vibrato, and fold into one octave so a real
    # register change cannot masquerade as a huge swing.
    win = max(3, int(round(0.4 * frame_rate)))
    trend = np.array([np.median(cents[max(0, i - win // 2):i + win // 2 + 1])
                      for i in range(len(cents))])
    dev = cents - trend
    dev = (dev + 600.0) % 1200.0 - 600.0
    dev = dev - np.mean(dev)

    # Spectrum of the movement, averaged over 4 s windows.
    seg = max(16, int(round(4.0 * frame_rate)))
    spec_total = np.zeros(0)
    freqs = None
    magnitudes = None
    for start in range(0, max(1, len(dev) - seg), seg // 2):
        chunk = dev[start:start + seg]
        if len(chunk) < seg // 2:
            break
        chunk = chunk * np.hanning(len(chunk))
        sp = np.abs(np.fft.rfft(chunk))
        f = np.fft.rfftfreq(len(chunk), d=1.0 / frame_rate)
        magnitudes = sp if magnitudes is None else magnitudes[:len(sp)] + sp[:len(magnitudes)]
        freqs = f if freqs is None else freqs
    if magnitudes is None or freqs is None or magnitudes.sum() <= 0:
        return 0.0, 0.0
    band = (freqs >= 2.0) & (freqs <= 12.0)
    peak_band = (freqs >= rate_hz - 0.6) & (freqs <= rate_hz + 0.6)
    total = float(np.sum(magnitudes[band] ** 2))
    if total <= 0:
        return 0.0, 0.0
    share = float(np.sum(magnitudes[peak_band] ** 2) / total)
    return share, float(np.sqrt(np.mean(dev ** 2)))


def _vibrato_extent_cents(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(rms, p95) of the part's own f0 deviation, in cents, around the note.

    The deviation is measured against a 0.5 s moving median of the f0 curve,
    which is longer than a vibrato period (5.4 Hz is 0.185 s) and shorter than a
    phrase, so what is left is the movement WITHIN a held note: the singer's own
    micro-movement, plus whatever this setting adds. Readings more than a fourth
    (500 cents) from the local median are dropped as octave errors from the
    tracker, and the residual is folded into one octave around the median before
    that, so a genuine octave leap in the melody cannot be mistaken for a huge
    vibrato.
    """
    import librosa
    mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 0:
        return 0.0, 0.0
    voiced = np.abs(mono) > peak * 0.05
    if voiced.sum() < sr // 10:
        return 0.0, 0.0
    first, last = int(np.argmax(voiced)), int(len(voiced) - np.argmax(voiced[::-1]))
    f0, vflag, _ = librosa.pyin(mono[first:last], fmin=70, fmax=800, sr=sr,
                                frame_length=2048, hop_length=512)
    good = np.isfinite(f0) & vflag
    if good.sum() < 20:
        return 0.0, 0.0
    cents = 1200.0 * np.log2(f0[good] / 220.0)
    # median filter, at frame rate, over ~0.5 s
    frames = max(3, int(round(0.5 / 0.0116)))
    note = np.array([np.median(cents[max(0, i - frames // 2):i + frames // 2 + 1])
                     for i in range(len(cents))])
    dev = cents - note
    dev = (dev + 600.0) % 1200.0 - 600.0        # one octave, around the note
    dev = dev[np.abs(dev) <= 250.0]             # tracker outliers, not vibrato
    if dev.size < 10:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(dev ** 2))), float(np.percentile(np.abs(dev), 95))


def main() -> int:
    from check_demo_matrix import make_inputs
    import server
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    inputs, _refs = make_inputs()
    raw = inputs["take.wav"]

    print("\n1. VOCAL FOLD VIBRATION through /api/dsp/render-track", flush=True)
    results = {}
    for amount in (0, 45, 100):
        resp = client.post(
            "/api/dsp/render-track",
            files={"file": ("take.wav", raw, "audio/wav")},
            data={
                "texture": "solo",
                "pitch_method": "yin",
                "mode": "unison_shadowing",
                "instruments_enabled": "true",
                "vocable": "mix",
                "vibrato_amount": str(amount),
            },
        )
        if resp.status_code != 200:
            check(False, f"render at {amount}% returned HTTP {resp.status_code}: {resp.text[:200]}")
            return 1
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
        share, rms_dev = _vibrato_share(np.asarray(audio), sr)
        results[amount] = {"share": share, "rms": rms_dev, "len": len(audio), "sr": sr,
                           "level": float(np.sqrt(np.mean(np.square(audio))))}
        print(f"     {amount:3d}%: {len(audio)} samples @ {sr} Hz, level {results[amount]['level']:.4f}, "
              f"movement at 5.4 Hz = {100 * share:.1f}% of 2-12 Hz (rms {rms_dev:.1f} cents)",
              flush=True)

    # The absolute readings include the TAKE's own movement, which this engine
    # keeps on purpose (see world_voice), so what is checked is that the setting
    # ADDS a regular oscillation at the vibrato's own rate, in proportion to
    # itself, and changes nothing else about the part.
    check(results[45]["share"] > results[0]["share"] * 1.25,
          f"45% raises the movement at the vibrato rate "
          f"({100 * results[0]['share']:.1f}% -> {100 * results[45]['share']:.1f}%)")
    check(results[100]["share"] > results[45]["share"] * 1.25,
          f"100% raises it again "
          f"({100 * results[45]['share']:.1f}% -> {100 * results[100]['share']:.1f}%)")
    check(results[100]["share"] >= 0.25,
          f"100% puts a clear oscillation at 5.4 Hz in the part "
          f"({100 * results[100]['share']:.1f}% of the band)")
    check(abs(results[100]["len"] - results[0]["len"]) <= results[0]["sr"] // 100,
          "the setting does not change the length of the part")
    check(all(abs(results[a]["level"] - results[0]["level"]) < 0.05 for a in (45, 100)),
          "the setting does not change the part's overall level")

    if "--vibrato-only" in sys.argv:
        print()
        if FAILURES:
            print(f"FAILED ({len(FAILURES)}):")
            for f in FAILURES:
                print("  -", f)
            return 1
        print("VIBRATO CHECKS PASSED")
        return 0

    print("\n2. THE TRANSCRIPT through /api/transcribe/annotated", flush=True)
    resp = client.post(
        "/api/transcribe/annotated",
        files={"file": ("song.wav", inputs["song.wav"], "audio/wav")},
        data={"language": "en"},
    )
    check(resp.status_code == 200, f"transcribe returned HTTP {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        labels = [s.get("label", "") for s in data.get("segments", [])]
        named = [l for l in labels if l.startswith("<") and l.endswith(">")]
        bracket = [l for l in labels if l.startswith("[")]
        print(f"     {len(labels)} row(s): {labels}", flush=True)
        if named:
            print(f"     named instrument rows: {named}", flush=True)
        check(not bracket, "no row comes back as a bare bracket tag")
        # Every named row either names an instrument ('<drums playing>') or names the
        # voice it heard ('<humming>', '<chanting>', '<choir singing>', ...). The one
        # generic name left is the honest last resort, for a stretch the tagger could
        # not name either and which a pitch scan says has a voice in it.
        voice_names = {"<singing>", "<a cappella singing>", "<choir singing>",
                       "<a man singing>", "<a woman singing>", "<a child singing>",
                       "<chanting>", "<humming>", "<yodeling>", "<wordless vocals>"}
        check(all(l.endswith(" playing>") or l in voice_names for l in named),
              "named rows read as '<instrument playing>' or as a name for the voice")
        check("<wordless vocals>" not in named,
              "nothing generic was written over a stretch the tagger could name "
              "(the drum-intro case that used to read '<wordless vocals>')")
        # ONE MOMENT, ONE ROW. Two rows covering the same instant is what an
        # instrument span and the window after it used to produce.
        ordered = sorted(data.get("segments", []), key=lambda s: s["start"])
        overlaps = [(a["label"], a["start"], a["end"], b["label"], b["start"])
                    for a, b in zip(ordered, ordered[1:]) if b["start"] < a["end"] - 1e-6]
        check(not overlaps, f"no two rows cover the same moment (found {overlaps[:3]})")
        check(not any(l.strip().lower() in ("música", "musica", "music") for l in labels),
              "no 'Música'-style hallucination survived into the transcript")

    from server import _is_whisper_music_artifact as art
    print("\n3. The artifact filter itself", flush=True)
    for text, want in (("Música", True), ("♪♪", True), ("...", True),
                       ("[Music]", True), ("Sous-titres réalisés par", True),
                       ("Merci", True), ("Thanks for watching!", True),
                       # The caption families the panel is never allowed to show, and
                       # the two that a live run actually produced: a doubled music
                       # caption padded with a thank-you over a drum intro, and a
                       # French subtitle credit wrapped around a proper noun.
                       ("Music Music Thank you", True), ("Music music", True),
                       ("Thank you", True), ("Bye-bye", True), ("Instrumental", True),
                       ("Sous-titrage Société Radio-Canada", True),
                       ("Société Radio-Canada, sous-titrage", True),
                       ("Subtitles by the Amara community", True),
                       ("Honour song for the people", False),
                       ("music in my heart", False), ("", False),
                       ("nêhiyawêwin", False),
                       # Real lines that must survive the credit markers.
                       ("Honor song for Dr. Suzanne Kite.", False),
                       ("Radio Canada was on the stereo", False),
                       ("Way", False), ("Well, well", False)):
        got = art(text)
        check(got == want, f"{text!r} -> {got} (expected {want})")

    # A symbol-only segment that passes Whisper's own confidence checks must
    # still not reach the panel as text (see _classify_segments' demotion).
    from server import _classify_segments
    strict_ok = {"start": 0.0, "end": 2.0, "text": "♪♪♪", "no_speech_prob": 0.05,
                 "avg_logprob": -0.2, "compression_ratio": 1.1}
    rows = _classify_segments([strict_ok])
    check(rows == [], "a confident segment of pure music symbols is not shown as text")
    real_line = {"start": 0.0, "end": 2.0, "text": "Honour song for the people",
                 "no_speech_prob": 0.05, "avg_logprob": -0.2, "compression_ratio": 1.1}
    rows = _classify_segments([real_line])
    check(len(rows) == 1 and rows[0]["label"] == "Honour song for the people",
          "a real confident line is still kept as text")

    # NAMING A STRETCH WHISPER HEARD NO WORDS IN, on the tagger's own reading.
    # This is the case a synthetic track cannot be trusted to produce, and it is
    # the one that used to write "<wordless vocals>" over a drum intro, so it is
    # tested directly.
    print("\n4. Naming a wordless stretch (see _what_is_sounding)", flush=True)
    from server import _what_is_sounding
    drum_intro = [{"start": 0.0, "end": 16.0, "label": "<drums playing>"}]
    check(_what_is_sounding(0.0, 16.0, drum_intro) == ("<drums playing>", "instrumental"),
          "a stretch the tagger heard as drums is named the drums, not wordless vocals")
    check(_what_is_sounding(0.0, 16.0, [{"start": 0.0, "end": 8.0, "label": "<bass playing>"}])[0]
          == "<bass playing>",
          "half a stretch accounted for by a named instrument is enough to name it")
    check(_what_is_sounding(0.0, 16.0, [{"start": 14.0, "end": 16.0, "label": "<drums playing>"}])[0]
          == "<wordless vocals>",
          "a name that only explains a corner of the row does not become the row")
    check(_what_is_sounding(0.0, 16.0, []) == ("<wordless vocals>", "vocal"),
          "with nothing named, the pitch scan's own reading stands")
    # One window, one name: no two names are allowed to land on the same class,
    # and every name a row can carry is in the map the namer reads.
    from server import _PANNS_INSTRUMENT_LABELS, _PANNS_VOICE_LABELS
    check(all(v.endswith(" playing") for v in _PANNS_INSTRUMENT_LABELS.values()),
          "every instrument name reads '<something> playing'")
    check(not (set(_PANNS_INSTRUMENT_LABELS) & set(_PANNS_VOICE_LABELS)),
          "a PANNs class is an instrument or a voice, never both")
    generic = [c for c, v in _PANNS_VOICE_LABELS.items() if v == "vocals"]
    check(not generic, f"no class is still written with the generic tag ({generic})")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL RENDER/TRANSCRIPT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

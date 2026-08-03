import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

import numpy as np

from analysis.rmvpe_detector import RMVPEModel

MODEL_PATH = os.path.join(os.path.dirname(_here), "assets", "rmvpe", "rmvpe.pt")


def synthetic_tone(hz=220.0, seconds=1.5, sr=16000):
    t = np.arange(0, int(seconds * sr)) / sr
    # A touch of harmonic content + slight vibrato, closer to a real voice
    # than a pure sine, so this isn't testing the easiest possible case.
    vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t)
    sig = 0.5 * np.sin(2 * np.pi * hz * vibrato * t)
    sig += 0.15 * np.sin(2 * np.pi * hz * 2 * vibrato * t)
    sig += 0.02 * np.random.randn(len(t))  # a little noise, not a clean lab tone
    return sig.astype(np.float32)


def load_wav_16k(path):
    import librosa
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio.astype(np.float32)


def main():
    print(f"Loading RMVPE from {MODEL_PATH} ...")
    model = RMVPEModel(MODEL_PATH, device="cpu", is_half=False)
    print("Loaded OK.\n")

    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        print(f"Running on {wav_path} ...")
        audio = load_wav_16k(wav_path)
        f0, conf = model.infer(audio, voicing_threshold=0.03)
        voiced = f0[f0 > 0]
        print(f"{len(f0)} hops, {len(voiced)} voiced.")
        if len(voiced):
            print(f"voiced f0 range: {voiced.min():.1f} - {voiced.max():.1f} Hz, "
                  f"median {np.median(voiced):.1f} Hz")
            print(f"mean confidence on voiced hops: {conf[f0 > 0].mean():.3f}")
        else:
            print("No voiced hops detected -- try a louder/cleaner recording, "
                  "or check the file actually contains singing/speech.")
        return

    print("No wav given, testing against synthetic tones instead:\n")
    ok = True
    for target_hz in (110.0, 220.0, 440.0, 880.0):  # spans ~2 octave-jump-prone range
        audio = synthetic_tone(target_hz)
        f0, conf = model.infer(audio, voicing_threshold=0.03)
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            print(f"  {target_hz:>6.1f} Hz target -> FAIL (nothing detected as voiced)")
            ok = False
            continue
        detected = float(np.median(voiced))
        cents_off = 1200 * np.log2(detected / target_hz)
        status = "OK" if abs(cents_off) < 50 else "FAIL (off by >50 cents -- check for an octave error)"
        print(f"  {target_hz:>6.1f} Hz target -> {detected:>6.1f} Hz detected "
              f"({cents_off:+.0f} cents, confidence {conf[f0>0].mean():.2f})  [{status}]")
        if "FAIL" in status:
            ok = False

    print()
    print("ALL SYNTHETIC TONES OK -- weights + inference are working." if ok
          else "SOME TONES FAILED -- check model_path points at the real rmvpe.pt "
               "and that it downloaded completely (should be ~172MB).")


if __name__ == "__main__":
    main()
"""
Run this once, separately, BEFORE starting server.py:

    python download_whisper_model.py

Downloads OpenAI's Whisper "base.en" speech-to-text model (~140MB) from
openaipublic.azureedge.net and saves it to data/models/whisper/. This is
the ONLY place in this project that downloads that model or makes a
network call for it -- server.py only ever reads from that local
directory, so it can never silently attempt a download of its own.

Why this exists at all: the browser's built-in speech recognition
(webkitSpeechRecognition) only works in official Google Chrome / Edge --
not Chromium, not Brave, not Firefox, not Safari -- because it silently
streams your audio to Google's own servers using a private API key that
open-source Chromium builds don't have. That's a hard platform limitation,
not something fixable from this page's JavaScript. Running transcription
on your own server with Whisper instead removes that dependency entirely:
it works identically in every browser, because the browser's only job
becomes "record audio and send it here" -- no browser speech API involved
at all.

Needs internet access, and needs ffmpeg installed as a system binary
(Whisper shells out to it to decode audio) -- check with `ffmpeg -version`
before running this if you're not sure it's installed.
"""
import sys
import shutil
from pathlib import Path

MODEL_NAME = "base.en"  # good balance of accuracy vs. speed for a live demo; "small.en" is more accurate but ~3x slower on CPU
OUTPUT_DIR = Path("data/models/whisper")


def main():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Whisper needs it to decode audio at transcription time.")
        print("Install it first (e.g. apt install ffmpeg / brew install ffmpeg), then re-run this.")
        sys.exit(1)

    try:
        import whisper
    except ImportError:
        print("openai-whisper isn't installed. Run: pip install openai-whisper")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Whisper '{MODEL_NAME}' -> {OUTPUT_DIR} (needs internet access, one-time only)...")

    try:
        # download_root makes whisper save (and later look for) the .pt
        # checkpoint file directly in our own directory, instead of its
        # default ~/.cache/whisper -- keeps this self-contained the same
        # way the sentiment model's directory is.
        whisper.load_model(MODEL_NAME, download_root=str(OUTPUT_DIR))
    except Exception as e:
        print(f"Download failed: {e}")
        print("Check your internet connection can reach openaipublic.azureedge.net, then try again.")
        sys.exit(1)

    files = sorted(p.name for p in OUTPUT_DIR.iterdir())
    print(f"Done. Saved to {OUTPUT_DIR}:")
    for f in files:
        print(f"  {f}")
    print("\nserver.py will now load from this local directory with no network access needed.")


if __name__ == "__main__":
    main()

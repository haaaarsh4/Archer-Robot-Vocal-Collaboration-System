"""
One-time setup script for real-time, fully local live captions (Vosk).

Run once (needs internet access):

    pip install vosk
    python download_vosk_model.py

Downloads and unzips a small English streaming model (~40MB) to
data/models/vosk-model-small-en-us-0.15/. After this, server.py loads
it from local files and never touches the network for it again -- same
"download once, run offline" pattern as download_faster_whisper_model.py.

Why this exists, and why it's a genuinely different thing from Whisper:
Whisper (either backend already in this project) is a batch model -- it
can only transcribe a complete audio clip you hand it, which is why the
mic pipeline has always had to wait for a pause before showing anything.
An earlier attempt at instant captions used the browser's built-in
SpeechRecognition API instead, which genuinely can stream results as you
talk -- but it works by sending your microphone audio to Google's speech
servers over the network, and Brave (and some other privacy-focused
browsers) blocks that specific traffic by default. If live captions
never appeared at all in your setup, that's almost certainly why.

Vosk (built on Kaldi) is a real local streaming engine: no network call,
runs entirely on your machine, and emits partial results within roughly
100-300ms of audio arriving. It's noticeably less accurate than Whisper
-- it's a much smaller model built for speed, not correctness -- so this
project uses it only for the instant "something is happening" preview
while you're still talking. The accurate, hallucination-filtered,
sentiment-integrated transcript still comes from the existing
Whisper-based pipeline once your phrase ends.
"""

import io
import os
import zipfile
import urllib.request

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_URL_MIRROR = "https://huggingface.co/grimso/vosk-models/resolve/main/vosk-model-small-en-us-0.15.zip"
OUT_DIR = "data/models"
MODEL_DIR_NAME = "vosk-model-small-en-us-0.15"

if __name__ == "__main__":
    target = os.path.join(OUT_DIR, MODEL_DIR_NAME)
    if os.path.isdir(target):
        print(f"Already present at {target}, nothing to do.")
    else:
        try:
            print(f"Downloading {MODEL_URL} ...")
            with urllib.request.urlopen(MODEL_URL) as resp:
                data = resp.read()
        except Exception as e:
            print(f"Primary host failed ({e}); trying mirror: {MODEL_URL_MIRROR}")
            with urllib.request.urlopen(MODEL_URL_MIRROR) as resp:
                data = resp.read()
        print("Unzipping...")
        os.makedirs(OUT_DIR, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(OUT_DIR)
        print(f"Done. server.py will now load this from local files at {target}.")

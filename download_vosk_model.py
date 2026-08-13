"""
One-time setup script for real-time, fully local live captions (Vosk).

Run once (needs internet access):

    pip install vosk
    python download_vosk_model.py

Downloads and unzips the larger, more accurate English streaming model
(~1.8GB) to data/models/vosk-model-en-us-0.22/. After this, server.py
loads it from local files and never touches the network for it again --
same "download once, run offline" pattern as download_faster_whisper_model.py.

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
100-300ms of audio arriving.

MODEL_DIR_NAME = "vosk-model-en-us-0.22": this is the larger of Vosk's
two mainline English models (~1.8GB vs ~40MB for the small one used
previously). Independent evaluation puts its word error rate roughly
20% lower than the small model's, and it's the model Vosk's own
real-time streaming examples use directly -- it's still built for live
use, just with a bigger acoustic/language model underneath, not a
batch-only model like Whisper. It will take noticeably longer to load
at server startup than the small model did (bigger file to read into
memory), but per-chunk streaming latency stays in the same real-time
ballpark. If you're on constrained hardware and startup time or RAM
becomes a problem, set MODEL_URL/MODEL_DIR_NAME back to
vosk-model-small-en-us-0.15 (https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
and update VOSK_MODEL_DIR in server.py to match.

It's still noticeably less accurate than Whisper on this project's
finalized (post-pause) transcript -- that's expected and fine. This
model is only ever used for the instant "something is happening"
preview while you're still talking; the accurate, hallucination-
filtered, sentiment-integrated entry always comes from the
large-v3-turbo Whisper pipeline once your phrase actually ends.
"""

import io
import os
import zipfile
import urllib.request

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
OUT_DIR = "data/models"
MODEL_DIR_NAME = "vosk-model-en-us-0.22"

if __name__ == "__main__":
    target = os.path.join(OUT_DIR, MODEL_DIR_NAME)
    if os.path.isdir(target):
        print(f"Already present at {target}, nothing to do.")
    else:
        print(f"Downloading {MODEL_URL} ...")
        print("This is a much larger download than the old small model (~1.8GB) -- may take a while.")
        with urllib.request.urlopen(MODEL_URL) as resp:
            data = resp.read()
        print("Unzipping...")
        os.makedirs(OUT_DIR, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(OUT_DIR)
        print(f"Done. server.py will now load this from local files at {target}.")
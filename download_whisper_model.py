"""
One-time setup script for the faster-whisper transcription backends.

Run this once (needs internet access) before starting server.py:

    python download_faster_whisper_model.py

Downloads BOTH models this project uses and caches them locally. After
this completes, server.py loads both with local_files_only=True and
never touches the network again -- same "download once, run offline"
pattern as download_whisper_model.py (legacy fallback) and
download_vosk_model.py (live-caption layer).

Why two separate models, not one: a live mic phrase and an uploaded full
track have very different latency budgets, so server.py uses two
separate faster-whisper configs (see MIC_WHISPER_MODEL_SIZE /
TRACK_WHISPER_MODEL_SIZE in server.py):

  - small.en (~75M params) -- fast enough for the live Sentiment-tab
    mic pipeline, where every single phrase pays this model's latency.
  - large-v3-turbo (~809M params, ~99% of full large-v3's accuracy) --
    used only for the Live Demo's "Upload an MP3" whole-track analysis,
    where a single slower request per upload is an acceptable trade for
    much better accuracy.

An earlier version of this script only downloaded large-v3-turbo. That
left the mic model directory empty, so server.py's mic path silently
fell back to the older, slower, PyTorch-only openai-whisper backend
(with word-timestamp alignment that's noticeably more crash-prone on
short/quiet clips than faster-whisper's). If your server startup log
ever shows "faster-whisper (small.en) failed to load ... trying legacy
openai-whisper", that's this gap -- re-run this script to fix it.
"""

from faster_whisper import WhisperModel

MODELS = [
    # (model_size, download_root) -- keep these in sync with
    # MIC_WHISPER_MODEL_SIZE/MIC_WHISPER_MODEL_DIR and
    # TRACK_WHISPER_MODEL_SIZE/TRACK_WHISPER_MODEL_DIR in server.py.
    ("small.en", "data/models/faster-whisper-small.en"),
    ("large-v3-turbo", "data/models/faster-whisper-large-v3-turbo"),
]

if __name__ == "__main__":
    for model_size, out_dir in MODELS:
        print(f"Downloading faster-whisper model '{model_size}' to {out_dir} ...")
        if model_size == "large-v3-turbo":
            print("This one's a larger download (~1.6GB) -- may take a while.")
        # Instantiating it is what triggers the download+cache; device/compute_type
        # here don't matter for the download itself, only for the server's later load.
        WhisperModel(model_size, device="cpu", compute_type="int8", download_root=out_dir)
        print(f"Done -- server.py will load this from local files at {out_dir}.")
    print("\nBoth models are set up. Restart server.py and check the startup log for "
          "two 'faster-whisper (..., int8, CPU) loaded' lines (not 'legacy backend').")
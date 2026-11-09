from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    ("small.en", ROOT / "data" / "models" / "faster-whisper-small.en"),
    ("large-v3-turbo", ROOT / "data" / "models" / "faster-whisper-large-v3-turbo"),
]

if __name__ == "__main__":
    for model_size, out_dir in MODELS:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading faster-whisper model '{model_size}' to {out_dir} ...")
        if model_size == "large-v3-turbo":
            print("This one is a larger download (~1.6GB), may take a while.")
        WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(out_dir))
        print(f"Done, server.py will load this from local files at {out_dir}.")
    print("\nBoth models are set up. Restart server.py and check the startup log for "
          "two 'faster-whisper (..., int8, CPU) loaded' lines.")

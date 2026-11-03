import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
OUT_DIR = ROOT / "data" / "models"
MODEL_DIR_NAME = "vosk-model-en-us-0.22"

if __name__ == "__main__":
    target = OUT_DIR / MODEL_DIR_NAME
    if target.is_dir():
        print(f"Already present at {target}, nothing to do.")
    else:
        print(f"Downloading {MODEL_URL} ...")
        print("This is a larger download (~1.8GB), may take a while.")
        with urllib.request.urlopen(MODEL_URL) as resp:
            data = resp.read()
        print("Unzipping ...")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(OUT_DIR)
        print(f"Done. server.py will now load this from local files at {target}.")

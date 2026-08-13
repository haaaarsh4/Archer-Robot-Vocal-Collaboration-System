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
"""
Downloads and caches cardiffnlp/twitter-roberta-base-sentiment-latest
locally, so server.py can load it fully offline afterwards (it enforces
local_files_only=True and never hits the network itself -- see the
comments above SENTIMENT_MODEL_DIR in server.py).

Needs internet access to huggingface.co. One-time only -- after this
finishes, server.py will find the model on disk on every future run.

Usage:
    python download_sentiment_model.py
"""

import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUT_DIR = "data/models/sentiment-roberta"

if __name__ == "__main__":
    if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
        print(f"Already present at {OUT_DIR}, nothing to do.")
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Downloading {MODEL_NAME} (~500MB) to {OUT_DIR} ...")
        os.makedirs(OUT_DIR, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        tokenizer.save_pretrained(OUT_DIR)
        model.save_pretrained(OUT_DIR)

        print(f"Done -- saved to {OUT_DIR}.")

    print(
        "\nRestart server.py and check the startup log for "
        "'Sentiment analyzer (RoBERTa) loaded from local files at "
        f"{OUT_DIR}.' If that model directory is missing or incomplete, "
        "server.py falls back to VADER automatically -- which one actually "
        "ran is reported as \"engine\" in every /api/sentiment response."
    )

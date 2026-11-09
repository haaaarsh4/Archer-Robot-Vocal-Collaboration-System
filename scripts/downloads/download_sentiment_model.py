from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUT_DIR = ROOT / "data" / "models" / "sentiment-roberta"

if __name__ == "__main__":
    if OUT_DIR.is_dir() and any(OUT_DIR.iterdir()):
        print(f"Already present at {OUT_DIR}, nothing to do.")
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Downloading {MODEL_NAME} (~500MB) to {OUT_DIR} ...")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        tokenizer.save_pretrained(OUT_DIR)
        model.save_pretrained(OUT_DIR)

        print(f"Done, saved to {OUT_DIR}.")

    print(
        "\nRestart server.py and check the startup log for "
        "'Sentiment analyzer (RoBERTa) loaded from local files at "
        f"{OUT_DIR}.' If that model directory is missing or incomplete, "
        "server.py falls back to VADER automatically. The response's "
        "\"engine\" field reports which one ran."
    )

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import morfessor
from loguru import logger

MODEL_PATH = "data/models/morfessor_cree.bin"


def count_cree_word_frequencies(corpus_path: Path) -> Counter:
    counts = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts.update(row["cree"].strip().lower().split())
    return counts


def train_morfessor(corpus_path: Path, model_path: str = MODEL_PATH) -> morfessor.BaselineModel:
    counts = count_cree_word_frequencies(corpus_path)
    logger.info(f"Training Morfessor on {len(counts)} distinct Cree word forms "
                f"({sum(counts.values())} total tokens)")

    io = morfessor.MorfessorIO()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for word, count in counts.items():
            f.write(f"{count} {word}\n")
        tmp_path = f.name

    data = list(io.read_corpus_file(tmp_path))
    model = morfessor.BaselineModel()
    model.load_data(data)
    model.train_batch()

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    io.write_binary_model_file(model_path, model)
    logger.info(f"Saved Morfessor model -> {model_path}")
    Path(tmp_path).unlink(missing_ok=True)
    return model


_model_cache: Optional[morfessor.BaselineModel] = None


def get_model(model_path: str = MODEL_PATH) -> morfessor.BaselineModel:
    global _model_cache
    if _model_cache is None:
        io = morfessor.MorfessorIO()
        _model_cache = io.read_binary_model_file(model_path)
    return _model_cache


def segment_cree_text(text: str, model: Optional[morfessor.BaselineModel] = None) -> str:
    model = model or get_model()
    morphs = []
    for word in text.strip().lower().split():
        segmented, _ = model.viterbi_segment(word)
        morphs.extend(segmented)
    return " ".join(morphs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/parallel_cr_en_combined.jsonl"))
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    train_morfessor(args.corpus, args.model_path)

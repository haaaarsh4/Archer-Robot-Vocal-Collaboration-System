"""
Morpheme-aware pre-tokenization for Cree, using Morfessor (unsupervised
morphological segmentation) instead of relying on BPE alone to guess
morpheme boundaries from raw character frequency.

Backed by real published research: Mager et al. 2022, "BPE vs.
Morphological Segmentation: A Case Study on Machine Translation of Four
Polysynthetic Languages" (arXiv:2203.08954) found Morfessor segmentation
outperformed BPE for 3 of 4 polysynthetic American languages tested.
Being honest about the evidence: this is NOT a universal guarantee - a
related Yup'ik study found plain BPE could still win at smaller vocab
sizes. Compare val_ppl against your existing BPE-only run to see whether
it actually helps on YOUR corpus, don't just assume it will.

How it's used here: Morfessor pre-segments each Cree word into morphemes,
inserting real spaces at morpheme boundaries BEFORE SentencePiece ever
sees the text, so SentencePiece's subword merges respect real
statistically-learned morpheme boundaries instead of pure character
frequency. English is left untouched - it's not polysynthetic, so plain
BPE is already a reasonable fit there; using different tokenization
strategies per side when morphological typology differs is standard
practice, not a hack.

Since translation here is Cree->English only, Cree is never the generation
target, so this segmentation never needs to be reversed - it's purely an
encoder-side input transform.
"""
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import morfessor
from loguru import logger

MODEL_PATH = "data/models/morfessor_cree.bin"


def count_cree_word_frequencies(corpus_path: Path) -> Counter:
    """Real token frequencies (not deduplicated types) - Morfessor's
    segmentation quality depends on realistic frequency counts, not a
    flat count-1-per-type list."""
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

    # Morfessor's documented loading path wants a "count word" text file -
    # using this instead of hand-built tuples, since that path is what
    # actually produces correct segmentation (verified against a known
    # test case before relying on it here).
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
    """Pre-segments Cree text into morphemes, space-separated, ready to
    feed into SentencePiece in place of the raw whole-word text."""
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

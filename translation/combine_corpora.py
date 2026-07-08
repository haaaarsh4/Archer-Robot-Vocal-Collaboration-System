"""
Combines the sentence-level EdTeKLA corpus with the word-level Watkins
dictionary pairs into one training file for the neural transformer -
capped at a ratio so single-word "sentences" don't dominate training and
bias the model toward short, single-word outputs.

Usage:
    python -m translation.combine_corpora \
        --sentences data/processed/parallel_cr_en.jsonl \
        --dictionary data/processed/watkins_dictionary_pairs.jsonl \
        --out data/processed/parallel_cr_en_combined.jsonl \
        --max-dictionary-ratio 0.3
"""
import argparse
import json
import random
from pathlib import Path

from loguru import logger


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, default=Path("data/processed/parallel_cr_en.jsonl"))
    parser.add_argument("--dictionary", type=Path, default=Path("data/processed/watkins_dictionary_pairs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/parallel_cr_en_combined.jsonl"))
    parser.add_argument(
        "--max-dictionary-ratio", type=float, default=0.3,
        help="Cap dictionary pairs at this fraction of sentence-pair count. "
             "0 excludes dictionary data entirely. 1.0 allows up to 1:1. "
             "Recommended: stay at or below 0.3-0.5 - these are single "
             "words, not sentences, so let real sentence data stay dominant."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sentence_pairs = load_jsonl(args.sentences)
    for p in sentence_pairs:
        p.setdefault("source", "edtekla")
    dictionary_pairs = load_jsonl(args.dictionary)

    if not sentence_pairs:
        raise FileNotFoundError(f"No sentence pairs found at {args.sentences}")

    max_dict = int(len(sentence_pairs) * args.max_dictionary_ratio)
    if len(dictionary_pairs) > max_dict:
        rng = random.Random(args.seed)
        rng.shuffle(dictionary_pairs)
        dictionary_pairs = dictionary_pairs[:max_dict]

    combined = sentence_pairs + dictionary_pairs
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for pair in combined:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        f"Combined corpus: {len(sentence_pairs)} sentence pairs (EdTeKLA) + "
        f"{len(dictionary_pairs)} dictionary pairs (Watkins 1865, capped at "
        f"ratio={args.max_dictionary_ratio}) = {len(combined)} total -> {args.out}"
    )


if __name__ == "__main__":
    main()

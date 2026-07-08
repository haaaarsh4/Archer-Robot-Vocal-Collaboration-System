"""
Reports how much of your PlainsCree dataset actually made it into the
training corpus, broken down by source subfolder - so you can verify
build_translation_corpus.py used everything (Bible, Dictionary,
ChildrenBooks, etc), not just one folder.

Usage:
    python -m translation.corpus_stats --data-dir data/PlainsCree \
        --corpus data/processed/parallel_cr_en.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from translation.utils import tokenize
from translation.build_translation_corpus import CR_SUFFIX, EN_SUFFIX


def raw_file_coverage(data_dir: Path) -> Counter:
    """Count *_cr.txt files found per top-level subfolder (Bible, Dictionary, ...)."""
    counts = Counter()
    for cr_path in data_dir.rglob(f"*{CR_SUFFIX}"):
        try:
            top_level = cr_path.relative_to(data_dir).parts[0]
        except (ValueError, IndexError):
            top_level = "(root)"
        counts[top_level] += 1
    return counts


def corpus_stats(corpus_path: Path):
    n_pairs = 0
    cr_tokens, en_tokens = 0, 0
    cr_vocab, en_vocab = set(), set()
    lengths = []
    source_counts = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cr_t = tokenize(row["cree"])
            en_t = tokenize(row["english"])
            n_pairs += 1
            cr_tokens += len(cr_t)
            en_tokens += len(en_t)
            cr_vocab.update(cr_t)
            en_vocab.update(en_t)
            lengths.append(len(cr_t))
            source_counts[row.get("source", "unlabeled")] += 1

    return {
        "pairs": n_pairs,
        "cree_vocab_size": len(cr_vocab),
        "english_vocab_size": len(en_vocab),
        "avg_cree_sentence_len": round(cr_tokens / n_pairs, 1) if n_pairs else 0,
        "avg_english_sentence_len": round(en_tokens / n_pairs, 1) if n_pairs else 0,
        "source_breakdown": dict(source_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/PlainsCree"))
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/parallel_cr_en.jsonl"))
    args = parser.parse_args()

    print("\n=== Raw file coverage by source folder ===")
    counts = raw_file_coverage(args.data_dir)
    if not counts:
        print(f"No *{CR_SUFFIX} files found under {args.data_dir} - check the path.")
    for folder, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {folder:<25} {n} file pairs")
    print(f"  {'TOTAL':<25} {sum(counts.values())} file pairs")

    print("\n=== Processed training corpus ===")
    if not args.corpus.exists():
        print(f"{args.corpus} not found - run build_translation_corpus.py first.")
        return
    stats = corpus_stats(args.corpus)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(
        "\nRule of thumb: a from-scratch transformer wants low tens of "
        "thousands of sentence pairs minimum to generalize reasonably. "
        "Below that, expect frequent hallucinated/wrong output on anything "
        "not close to a training example."
    )


if __name__ == "__main__":
    main()

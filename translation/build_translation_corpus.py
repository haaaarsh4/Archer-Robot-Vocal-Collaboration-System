import argparse
import json
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from translation.sentence_aligner import align_sequences

CR_SUFFIX = "_cr.txt"
EN_SUFFIX = "_en.txt"


def find_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Find every *_cr.txt that has a matching *_en.txt in the same folder."""
    pairs = []
    cr_files = sorted(data_dir.rglob(f"*{CR_SUFFIX}"))
    logger.info(f"Found {len(cr_files)} candidate Cree files under {data_dir}")

    for cr_path in cr_files:
        base = cr_path.name[: -len(CR_SUFFIX)]
        en_path = cr_path.with_name(base + EN_SUFFIX)
        if en_path.exists():
            pairs.append((cr_path, en_path))
        else:
            logger.warning(f"No matching English file for {cr_path}, skipping")

    logger.info(f"Matched {len(pairs)} Cree/English file pairs")
    return pairs


def read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_sentence_pairs(cr_path: Path, en_path: Path) -> List[Tuple[str, str]]:
    cr_lines = read_lines(cr_path)
    en_lines = read_lines(en_path)

    if not cr_lines or not en_lines:
        return []

    if len(cr_lines) == len(en_lines):
        return list(zip(cr_lines, en_lines))

    logger.warning(
        f"Line count mismatch in {cr_path.name} ({len(cr_lines)} cr / "
        f"{len(en_lines)} en) — running sentence alignment instead of "
        f"line-by-line zip"
    )
    return align_sequences(cr_lines, en_lines, band=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/PlainsCree"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/parallel_cr_en.jsonl")
    )
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)

    all_sentence_pairs: List[Tuple[str, str]] = []
    for cr_path, en_path in pairs:
        sentence_pairs = extract_sentence_pairs(cr_path, en_path)
        all_sentence_pairs.extend(sentence_pairs)
        logger.info(f"{cr_path.relative_to(args.data_dir)}: {len(sentence_pairs)} pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cree, english in all_sentence_pairs:
            f.write(json.dumps({"cree": cree, "english": english}, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(all_sentence_pairs)} sentence pairs to {args.out}")


if __name__ == "__main__":
    main()

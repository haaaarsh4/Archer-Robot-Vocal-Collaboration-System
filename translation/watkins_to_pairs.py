import argparse
import json
from pathlib import Path

from loguru import logger

from translation.utils import tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watkins", type=Path, default=Path("data/processed/watkins_entries.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/watkins_dictionary_pairs.jsonl"))
    args = parser.parse_args()

    if not args.watkins.exists():
        raise FileNotFoundError(f"{args.watkins} not found - run parse_watkins1865.py first.")

    pairs = []
    skipped = 0
    with open(args.watkins, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            english = " ".join(tokenize(entry["english"]))
            if not english:
                continue
            for cree_form in entry.get("cree_forms", []):
                cree = " ".join(tokenize(cree_form))
                if not cree:
                    skipped += 1
                    continue
                pairs.append({"cree": cree, "english": english, "source": "watkins1865"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} Watkins pairs -> {args.out} ({skipped} empty forms skipped)")


if __name__ == "__main__":
    main()

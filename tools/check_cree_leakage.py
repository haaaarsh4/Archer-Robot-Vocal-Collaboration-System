#!/usr/bin/env python3
"""Check that the held-out split never entered training.

`eval` reports chrF on `parallel_test.jsonl`. That number only means
generalisation if those sentences are absent from everything the model trained
on, and here that is a real risk rather than a formality: the Cree corpus, the
parallel corpus and the dictionary all come from overlapping sources, so a
verse held out of one book can reappear through another.

    python tools/check_cree_leakage.py --work ~/cree-work
    python tools/check_cree_leakage.py --work ~/cree-work --limit 200

Exit status is 1 if a held-out sentence appears verbatim in a training file, so
a contaminated evaluation cannot be reported as a clean one. Near-duplicates
(the first eight words match, the rest differs) are reported as a warning and do
not fail the run: they say the split is by sentence, not by text, which is a
weaker guarantee worth knowing about.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAINING_FILES = ("lm_train.jsonl", "sft_train.jsonl", "parallel_train.jsonl")
# A one-word answer is not a sentence: many held-out rows answer "yes", and
# counting that as a leak says nothing about generalisation while making every
# abstention example in the training data look like a leak.
_MIN_WORDS = 2
_HELD_OUT_DEFAULT = "parallel_test.jsonl"
# Anything that is not a letter, apostrophe or hyphen is a separator, so
# "kîsikâw." and "kîsikâw" compare equal.
_SEPARATOR = re.compile(r"[^a-z'\-\u00c0-\u024f]+")
_SENTENCE_END = re.compile(r"[.!?;]+")


def normalise(text: str) -> str:
    text = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(_SEPARATOR.sub(" ", text).split())


def _strings(value):
    """Every string inside a row, whatever shape the row is."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def sentences(path: Path) -> set[str]:
    """Normalised sentences from every string field of a jsonl file."""
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for value in _strings(row):
                for piece in _SENTENCE_END.split(value):
                    text = normalise(piece)
                    if len(text.split()) >= _MIN_WORDS:
                        out.add(text)
    return out


def prefixes(items: set[str], n: int = 8) -> set[str]:
    """The first n words of each sentence, for near-duplicate detection."""
    out: set[str] = set()
    for text in items:
        words = text.split()
        if len(words) >= n:
            out.add(" ".join(words[:n]))
    return out


def held_out_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (row.get("cree") or row.get("english")):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True, help="training work folder, for example ~/cree-work")
    ap.add_argument("--held-out", type=Path, default=None,
                    help=f"held-out file (default: <work>/processed/{_HELD_OUT_DEFAULT})")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N held-out rows")
    args = ap.parse_args()

    processed = args.work / "processed"
    held_path = args.held_out or processed / _HELD_OUT_DEFAULT
    if not held_path.exists():
        raise SystemExit(f"{held_path} does not exist; run the clean stage first")
    rows = held_out_rows(held_path, args.limit)
    if not rows:
        raise SystemExit(f"{held_path} has no usable rows")

    training: dict[str, tuple[set[str], set[str]]] = {}
    for name in TRAINING_FILES:
        path = processed / name
        found = sentences(path)
        if found:
            training[name] = (found, prefixes(found))

    print(f"held out     {len(rows)} rows from {held_path}")
    for name, (found, _) in training.items():
        print(f"training     {name}: {len(found)} sentences")
    if not training:
        raise SystemExit(f"none of {', '.join(TRAINING_FILES)} exist under {processed}")

    leaks: dict[str, list[tuple[str, str]]] = {"cree": [], "english": []}
    near: dict[str, list[tuple[str, str]]] = {"cree": [], "english": []}
    for row in rows:
        for side in ("cree", "english"):
            text = normalise(_SENTENCE_END.split(str(row.get(side) or ""))[0])
            if len(text.split()) < _MIN_WORDS:
                continue
            words = text.split()
            head = " ".join(words[:8]) if len(words) >= 8 else ""
            for name, (found, heads) in training.items():
                if text in found:
                    leaks[side].append((text, name))
                elif head and head in heads:
                    near[side].append((text, name))

    for side in ("cree", "english"):
        total = sum(1 for row in rows if normalise(str(row.get(side) or "")))
        print(f"\n{side:8s} exact in training: {len(leaks[side])}/{total}"
              f"   near-duplicate: {len(near[side])}/{total}")
        for text, name in leaks[side][:5]:
            print(f"    LEAK  {text[:80]!r}  ({name})")
        for text, name in near[side][:5]:
            print(f"    near  {text[:80]!r}  ({name})")

    leaked = len(leaks["cree"]) + len(leaks["english"])
    if leaked:
        print(f"\nverdict: {leaked} held-out sentence(s) are in the training data, so the chrF "
              f"number overstates what the model can do")
        return 1
    print("\nverdict: clean, no held-out sentence appears verbatim in training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

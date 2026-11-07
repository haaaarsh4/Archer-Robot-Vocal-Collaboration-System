#!/usr/bin/env python3
"""Replay every correction against the installed model.

A correction becomes a regression case the moment it is saved. This tool sends
the original question through the real chat path and checks that the answer now
resembles the correction. Exit status 1 means at least one mistake came back,
so a candidate model can be refused before it is installed.

    python tools/run_regressions.py                 # all cases, model answers
    python tools/run_regressions.py --dry-run       # list cases, no model calls
    python tools/run_regressions.py --limit 5 --min-overlap 0.5

``--min-overlap`` is the share of the corrected answer's content words that
must appear in the reply. Exact string equality is the wrong test here: a
faithful answer can phrase the same fact differently.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat.learning import load_regressions                 # noqa: E402

_STOP = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
         "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
         "of", "on", "or", "she", "that", "the", "their", "them", "there",
         "they", "this", "to", "was", "were", "with", "you", "your"}


def _words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[^\W\d_]+", text.lower())
        if len(word) > 1 and word not in _STOP
    }


def overlap(expected: str, reply: str) -> float:
    want = _words(expected)
    if not want:
        return 0.0
    return len(want & _words(reply)) / len(want)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="regressions.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_regressions(args.path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("no regressions yet; correct an answer in the chat first")
        return 0

    if args.dry_run:
        for row in rows:
            print(f"Q {row['user']}\nA {row['expected']}\n")
        print(f"{len(rows)} case(s); no model calls made")
        return 0

    from chat.chat_api import ChatRequest, chat  # slow imports, only when needed

    failures = 0
    for row in rows:
        result = chat(ChatRequest(message=str(row["user"]), session_id="regression"))
        reply = str(result.get("reply") or "")
        score = overlap(str(row["expected"]), reply)
        ok = score >= args.min_overlap
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {score:.2f}  {row['user']!r}")
        print(f"      want {row['expected'][:140]!r}")
        print(f"      got  {reply[:140]!r}")
    print(f"\n{len(rows) - failures}/{len(rows)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

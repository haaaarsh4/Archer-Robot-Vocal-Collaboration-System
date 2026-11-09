#!/usr/bin/env python3
"""Review what the assistant learned and export it for the next training run.

    python tools/consolidate_feedback.py --stats
    python tools/consolidate_feedback.py --export
    python tools/consolidate_feedback.py --db /path/to/chat_memory.sqlite3 --export

``--export`` writes three files:

    data/training/feedback_sft.jsonl   chat-shaped SFT examples
    data/training/feedback_dpo.jsonl   {prompt, chosen, rejected} pairs
    data/eval/regressions.jsonl        cases tools/run_regressions.py replays

The SFT and DPO shapes match what training/cree_llm/cells/08_sft_data.py and
cells/10_dpo.py already read, so no conversion step is needed on the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat.learning import export                       # noqa: E402
from chat.memory import Memory                         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="memory database path")
    parser.add_argument("--stats", action="store_true", help="counts only")
    parser.add_argument("--export", action="store_true", help="write the three files")
    parser.add_argument("--clear", action="store_true", help="delete all memory")
    args = parser.parse_args()

    memory = Memory(path=args.db)
    if args.clear:
        memory.clear()
        print(json.dumps(memory.stats(), indent=2))
        return 0
    if args.export:
        print(json.dumps(export(memory), indent=2))
        return 0
    print(json.dumps(memory.stats(), indent=2))
    if not args.stats:
        print("\npass --export to write the training and regression files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

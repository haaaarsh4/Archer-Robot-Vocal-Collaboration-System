#!/usr/bin/env python3
"""Run the notebook's code cells in order, outside Colab.

This is the smoke-test harness. It executes exactly the same source the notebook
contains, in one namespace, so a bug in the notebook is a bug here. Colab magic
lines are skipped.

    python training/cree_llm/run_cells.py --stop 5
    CREE_SMOKE=1 python training/cree_llm/run_cells.py
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells"


def code_cells() -> list[Path]:
    return [p for p in sorted(CELLS.iterdir()) if p.suffix == ".py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, default=99, help="run cells up to and including this index")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    files = code_cells()
    ns: dict = {"__name__": "__main__"}
    for i, path in enumerate(files):
        if i < args.start or i > args.stop:
            continue
        print(f"\n{'=' * 78}\n[{i}] {path.name}\n{'=' * 78}", flush=True)
        source = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        try:
            exec(compile(source, str(path), "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"\nFAILED in {path.name}", file=sys.stderr)
            return 1
    print("\nall selected cells completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

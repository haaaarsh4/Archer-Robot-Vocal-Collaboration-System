#!/usr/bin/env python3
"""Assemble Cree_LLM_Training.ipynb from the readable cell sources in cells/.

Each cell is either ``NN_name.md`` (markdown) or ``NN_name.py`` (code).  The
notebook is generated, never hand-edited, so the sources and the .ipynb cannot
drift apart:

    python training/cree_llm/build_notebook.py           # write the notebook
    python training/cree_llm/build_notebook.py --check   # fail if out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "cells"
OUT = HERE.parent / "notebooks" / "Cree_LLM_Training.ipynb"


def load_cells() -> list[dict]:
    files = sorted(p for p in CELLS.iterdir() if p.suffix in {".md", ".py"})
    if not files:
        raise SystemExit(f"no cells in {CELLS}")
    out: list[dict] = []
    for path in files:
        source = path.read_text(encoding="utf-8").rstrip("\n").splitlines(keepends=True)
        if path.suffix == ".md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            out.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            })
    return out


def validate(nb: dict) -> None:
    """Fail on a malformed or uncompilable notebook before it reaches Colab."""
    required = {"cells", "metadata", "nbformat", "nbformat_minor"}
    missing = required - set(nb)
    if missing:
        raise SystemExit(f"notebook is missing {sorted(missing)}")
    if nb["nbformat"] != 4:
        raise SystemExit("only nbformat 4 is supported")
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") not in {"markdown", "code"}:
            raise SystemExit(f"cell {i}: bad cell_type")
        if not cell.get("source"):
            raise SystemExit(f"cell {i}: empty")
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                compile(source, f"cell-{i}", "exec")
            except SyntaxError as exc:
                raise SystemExit(f"cell {i}: {exc}") from exc


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "A100",
                "toc_visible": True,
                "name": "Cree_LLM_Training.ipynb",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    nb = notebook(load_cells())
    validate(nb)
    rendered = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not OUT.exists():
            print(f"missing {OUT}", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != rendered:
            print(f"{OUT} is out of date; rerun build_notebook.py", file=sys.stderr)
            return 1
        print(f"{OUT.name} matches its cell sources")
        return 0

    OUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT} ({len(load_cells())} cells, {len(rendered) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

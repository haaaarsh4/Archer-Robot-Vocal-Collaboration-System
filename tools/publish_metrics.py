#!/usr/bin/env python3
"""Copy the real run metrics and charts into the frontend assets.

No number is invented here. Every value is read from a file a run wrote:

    <work>/out/creebench_report.json       training and held-out scores
    <work>/out/web/cree-llm.json           the site bundle from the language tests
    data/eval/model_benchmark_report.json  the 1,000-row post-install benchmark
    data/processed/language_id_metrics.json learned language-ID accuracy
    data/eval/audit_report.json            optional installed-model audit

If a file is missing, the publisher says so and omits that section. It never
fills the gap with a placeholder number.

    python tools/publish_metrics.py --work /data/cree_llm
    python tools/publish_metrics.py --work /data/cree_llm --frontend frontend

The frontend can read the result at
`/static/assets/evidence/cree-llm.json` and the images beside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _copy(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target.relative_to(ROOT)).replace("\\", "/")


def _slim_training(report: dict) -> dict:
    tuned = report.get("tuned") or {}
    base = report.get("base") or {}
    return {
        "meta": report.get("meta") or {},
        "base": base,
        "tuned": tuned,
        "head_to_head": report.get("head_to_head") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=ROOT / "cree_llm_work")
    ap.add_argument("--frontend", type=Path, default=ROOT / "frontend")
    args = ap.parse_args()

    evidence = args.frontend / "assets" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    bundle: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "sources": {},
        "figures": {},
    }

    creebench = _load(args.work / "out" / "creebench_report.json")
    if creebench:
        bundle["sources"]["training"] = str(args.work / "out" / "creebench_report.json")
        bundle["training"] = _slim_training(creebench)
        for name in ("creebench.png", "sft_mix.png", "tokenizer_fertility.png",
                     "training_curves.png"):
            copied = _copy(args.work / "out" / "plots" / name, evidence / name)
            if copied:
                bundle["figures"][name.removesuffix(".png")] = f"/static/assets/evidence/{name}"
    else:
        print(f"training metrics not found: {args.work / 'out' / 'creebench_report.json'}")

    web = _load(args.work / "out" / "web" / "cree-llm.json")
    if web:
        bundle["sources"]["language_tests"] = str(args.work / "out" / "web" / "cree-llm.json")
        bundle["language_tests"] = web
    else:
        print(f"language-test bundle not found: {args.work / 'out' / 'web' / 'cree-llm.json'}")

    language_id = _load(ROOT / "data" / "processed" / "language_id_metrics.json")
    if language_id:
        bundle["sources"]["language_id"] = "data/processed/language_id_metrics.json"
        bundle["language_id"] = language_id

    benchmark = _load(ROOT / "data" / "eval" / "model_benchmark_report.json")
    if benchmark:
        bundle["sources"]["benchmark"] = "data/eval/model_benchmark_report.json"
        bundle["benchmark"] = {
            "meta": benchmark.get("meta") or {},
            "summary": benchmark.get("summary") or {},
            "categories": benchmark.get("categories") or {},
        }
        copied = _copy(ROOT / "data" / "eval" / "model_benchmark.png",
                       evidence / "model_benchmark.png")
        if copied:
            bundle["figures"]["model_benchmark"] = "/static/assets/evidence/model_benchmark.png"
    else:
        print("benchmark metrics not found: data/eval/model_benchmark_report.json")

    audit = _load(ROOT / "data" / "eval" / "audit_report.json")
    if audit:
        bundle["sources"]["audit"] = "data/eval/audit_report.json"
        bundle["audit"] = audit

    target = evidence / "cree-llm.json"
    target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    if not bundle.get("training") and not bundle.get("benchmark"):
        print("no training or benchmark metrics were available; the bundle has no invented numbers")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

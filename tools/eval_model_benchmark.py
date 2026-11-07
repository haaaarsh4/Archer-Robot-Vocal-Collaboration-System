#!/usr/bin/env python3
"""Score the 1,000-row benchmark against a model served by Ollama.

This is the post-install check. It sends the same prompts the training data
uses, scores each category with the metric written in the row, and writes a
report and a chart that the frontend publisher can copy.

    python tools/eval_model_benchmark.py --model archer:14b
    python tools/eval_model_benchmark.py --model archer:14b --limit 50
    python tools/eval_model_benchmark.py --model archer:14b --strict

The score is not a claim of perfection. It is the number this model actually
produced on this test set, with the failures kept in the report.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "data" / "eval" / "model_benchmark_1000.jsonl"
DEFAULT_OUT = ROOT / "data" / "eval" / "model_benchmark_report.json"
DEFAULT_PLOT = ROOT / "data" / "eval" / "model_benchmark.png"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(text).lower())).strip()


def _token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return 0.0
    overlap = 0
    left = list(g)
    for token in p:
        if token in left:
            left.remove(token)
            overlap += 1
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _chrf(pred: str, gold: str) -> float:
    return float(sacrebleu.corpus_chrf([pred], [[gold]], word_order=2).score)


def _score_row(row: dict, reply: str) -> tuple[float, bool, str]:
    metric = row.get("metric", "chrf")
    expected = str(row.get("expected") or "")
    if metric == "chrf":
        score = _chrf(reply, expected) / 100.0
        return score, score > 0.0, f"chrF++ {score:.3f}"
    if metric == "exact":
        ok = _norm(reply) == _norm(expected)
        return 1.0 if ok else 0.0, ok, "exact"
    if metric == "contains":
        needle = str(row.get("must_contain") or expected)
        ok = _norm(needle) in _norm(reply)
        return 1.0 if ok else 0.0, ok, f"contains {needle!r}"
    if metric == "not_echo":
        prompt = re.sub(r"<\|[^|]*\|>", " ", row["prompt"])
        ok = _norm(reply) != _norm(prompt) and _norm(reply) != _norm(expected)
        return 1.0 if ok else 0.0, ok, "not the prompt repeated"
    return 0.0, False, f"unknown metric {metric}"


def _ask(url: str, model: str, system: str, prompt: str, timeout: float) -> str:
    response = requests.post(
        f"{url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 160},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("message", {}).get("content", "")).strip()


def _plot(categories: dict[str, dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: {exc}")
        return
    labels = sorted(categories)
    values = [categories[name]["score"] for name in labels]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color="#2f5d50")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (1.0 is best)")
    ax.set_title("Archer 1,000-line benchmark")
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    ap.add_argument("--model", default="archer:14b")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any category scores 0")
    args = ap.parse_args()

    if not args.rows.exists():
        raise SystemExit(f"{args.rows} does not exist; run tools/make_model_benchmark.py first")
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no benchmark rows")

    try:
        from chat.archer_prompt import ARCHER_SYSTEM_PROMPT as system
    except Exception:
        system = "You are Archer's assistant. Answer briefly and honestly. Never invent a fact."

    predictions: list[dict] = []
    started = time.time()
    for n, row in enumerate(rows, 1):
        try:
            reply = _ask(args.url, args.model, system, row["prompt"], args.timeout)
            error = ""
        except Exception as exc:  # noqa: BLE001 - keep the failure in the report
            reply, error = "", f"{type(exc).__name__}: {exc}"
        score, passed, note = _score_row(row, reply) if reply else (0.0, False, "no reply")
        predictions.append({
            "id": row["id"], "category": row["category"], "prompt": row["prompt"],
            "expected": row["expected"], "reply": reply, "score": round(score, 4),
            "passed": passed, "note": note, "error": error,
        })
        if n % 25 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)
    categories: dict[str, dict] = {}
    for name, group in sorted(by_category.items()):
        scores = [row["score"] for row in group]
        categories[name] = {
            "n": len(group),
            "score": round(statistics.mean(scores), 4),
            "passed": sum(row["passed"] for row in group),
            "mean_token_f1": round(statistics.mean(
                _token_f1(row["reply"], row["expected"]) for row in group
            ), 4),
            "errors": sum(bool(row["error"]) for row in group),
        }

    overall = round(statistics.mean(row["score"] for row in predictions), 4)
    report = {
        "meta": {
            "model": args.model,
            "rows": len(predictions),
            "seconds": round(time.time() - started, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "source": str(args.rows),
        },
        "summary": {
            "score": overall,
            "passed": sum(row["passed"] for row in predictions),
            "n": len(predictions),
            "categories": len(categories),
        },
        "categories": categories,
        "rows": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(categories, args.plot)

    print(f"\n{'category':22s} {'n':>5s} {'score':>7s} {'passed':>7s} {'F1':>7s}")
    for name, item in categories.items():
        print(f"{name:22s} {item['n']:5d} {item['score']:7.3f} "
              f"{item['passed']:7d} {item['mean_token_f1']:7.3f}")
    print(f"\noverall {overall:.3f} on {len(predictions)} rows, written to {args.out}")
    if args.strict and any(item["score"] == 0 for item in categories.values()):
        print("strict mode: a category scored zero")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

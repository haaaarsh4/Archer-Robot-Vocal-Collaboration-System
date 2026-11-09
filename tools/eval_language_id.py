"""Score the language handling on thousands of labelled samples.

Two things are measured on the set tools/make_language_tests.py writes:

    language   did the identifier get the message's language right
    routing    did `route()` pick the right prompt shape and the right answer
               language for that message

It also reports the accuracy of the diacritic rule this replaced on the same
rows, so the improvement is a number rather than a claim.

    .venv/bin/python tools/eval_language_id.py
    .venv/bin/python tools/eval_language_id.py --show 10
    .venv/bin/python tools/eval_language_id.py --model 20   # also ask the model

``--model N`` sends N samples to the installed model and checks that the reply
comes back in the language the row expects. That is the end-to-end test: the
identifier, the router and the model together. On this machine it is slow
because every call is a CPU generation, so it is opt-in and off by default.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat.archer_prompt import route                                 # noqa: E402
from chat.language_id import language_of                             # noqa: E402

TESTS = ROOT / "data" / "eval" / "language_tests.jsonl"
NOTE = [("Horse", "Horse is an imagined story about memory and responsibility.")]
GATES = {"language": 0.90, "task": 0.90, "answer_language": 0.90}


def _load() -> list[dict]:
    if not TESTS.exists():
        raise SystemExit(f"{TESTS} does not exist, run tools/make_language_tests.py first")
    with TESTS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _diacritic_baseline(text: str) -> str:
    """The rule this replaces: any long vowel means Cree, otherwise English."""
    return "cr" if re.search(r"[âêîô]", text) else "en"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=5, help="failures to print per family")
    ap.add_argument("--model", type=int, default=0, help="samples to send to the model")
    args = ap.parse_args()

    rows = _load()
    print(f"{len(rows)} labelled samples from {TESTS.name}\n")

    per_family: dict[str, dict[str, int]] = {}
    language_ok = task_ok = answer_ok = baseline_ok = 0
    failures: list[tuple[str, str, str]] = []

    for row in rows:
        family = row["family"]
        text = row["text"]
        expect = row["expect"]
        notes = NOTE if expect.get("notes") else []
        verdict = language_of(text)
        plan = route(text, notes, [])

        gold = {
            "language": expect["lang"],
            "task": expect["task"],
            "answer_language": expect["answer_language"],
        }
        got = {
            "language": verdict.lang,
            "task": plan.task,
            "answer_language": plan.answer_language,
        }
        stats = per_family.setdefault(family, {key: 0 for key in (*GATES, "n")})
        stats["n"] += 1
        for key, _gate in GATES.items():
            if got[key] == gold[key]:
                stats[key] += 1
                if key == "language":
                    language_ok += 1
                elif key == "task":
                    task_ok += 1
                else:
                    answer_ok += 1
            elif key == "language":
                failures.append((family, text, f"language {got[key]} != {gold[key]}"))
        if got["task"] != gold["task"]:
            failures.append((family, text, f"task {got['task']} != {gold['task']}"))
        if got["answer_language"] != gold["answer_language"]:
            failures.append(
                (family, text,
                 f"answer {got['answer_language']} != {gold['answer_language']}")
            )
        if _diacritic_baseline(text) == gold["language"]:
            baseline_ok += 1

    total = len(rows)
    print(f"{'family':20s} {'n':>5s} {'language':>9s} {'task':>7s} {'answer':>7s}")
    for family, stats in per_family.items():
        n = max(stats["n"], 1)
        print(f"{family:20s} {stats['n']:5d} {stats['language'] / n:9.2f} "
              f"{stats['task'] / n:7.2f} {stats['answer_language'] / n:7.2f}")

    scores = {
        "language": language_ok / total,
        "task": task_ok / total,
        "answer_language": answer_ok / total,
        "diacritic_baseline": baseline_ok / total,
    }
    print(f"\noverall      language {scores['language']:.3f}   task {scores['task']:.3f}   "
          f"answer {scores['answer_language']:.3f}")
    print(f"old diacritic rule, same rows: {scores['diacritic_baseline']:.3f}")

    if failures and args.show:
        print(f"\nfirst {args.show} failures per family")
        shown: dict[str, int] = {}
        for family, text, why in failures:
            if shown.get(family, 0) >= args.show:
                continue
            shown[family] = shown.get(family, 0) + 1
            print(f"  [{family}] {why}\n      {text[:110]!r}")

    model_score = None
    if args.model > 0:
        from chat.chat_api import ChatRequest, chat

        print(f"\nsending {args.model} samples to the installed model")
        checked = right = 0
        times: list[float] = []
        for row in rows[:: max(total // args.model, 1)][: args.model]:
            started = time.perf_counter()
            try:
                reply = chat(ChatRequest(message=row["text"]))["reply"]
            except Exception as exc:  # noqa: BLE001 - reported, not hidden
                print(f"  model call failed: {exc}")
                break
            times.append(time.perf_counter() - started)
            want = row["expect"]["answer_language"]
            got = language_of(reply).lang
            ok = got == want or (want == "en" and got == "en") or (want == "cr" and got == "cr")
            checked += 1
            right += ok
            if not ok:
                print(f"  MISS want {want} got {got}: {row['text'][:60]!r} -> {reply[:70]!r}")
        if checked:
            model_score = right / checked
            print(f"  reply language {model_score:.2f} on {checked} samples, "
                  f"mean {statistics.mean(times):.1f}s")

    print("\nchecks")
    failed = False
    for key in ("language", "task", "answer_language"):
        ok = scores[key] >= GATES[key]
        failed |= not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {key} {scores[key]:.3f} (gate {GATES[key]:.2f})")
    if model_score is not None:
        ok = model_score >= 0.80
        failed |= not ok
        print(f"  {'ok  ' if ok else 'FAIL'} model reply language {model_score:.2f} (gate 0.80)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

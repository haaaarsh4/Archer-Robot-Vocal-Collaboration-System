"""Does the decoding change actually change anything? Measure both samplers.

The research record claims min-p and DRY are applied. "Applied" is not the same
as "helped", so this runs the same prompts through the installed model twice, in
the same order, on the same warm weights:

    old   only temperature. This is what the app sent before the change, so
          Ollama's own defaults (top_p 0.9, top_k 40) do the truncation and
          there is no loop penalty.
    new   min_p 0.05 with top_p 1.0 and top_k 0 (min-p is the only cut) plus
          DRY 0.8. This is what config.yaml now asks for.

Two things are scored, because they are the two failures a visitor notices:

    echo        a greeting handed back instead of answered (tools/audit_chat_model)
    repeat      the share of three-token spans that occur more than once in the
                reply. This is what a stuck clause is, before it becomes an echo,
                and it is the thing DRY is meant to charge for.
    overlap     translation, as token overlap with the reference sentence. It is
                here to catch the opposite failure: a sampler that "fixes" loops
                by making the model stop writing Cree.

    .venv/bin/python tools/decoding_ab.py
    .venv/bin/python tools/decoding_ab.py --greetings 6 --pairs 4   # quicker
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat.archer_prompt import ARCHER_SYSTEM_PROMPT, route           # noqa: E402
from chat.llm_client import LLMConfig, LocalLLMClient                # noqa: E402


def _audit():
    """The audit module owns the definitions of echo and overlap already."""
    spec = importlib.util.spec_from_file_location(
        "audit_chat_model", ROOT / "tools" / "audit_chat_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repeated_span_rate(text: str, n: int = 3) -> float:
    """Share of n-token spans in the reply that are duplicates of another span.

    A reply of one repeated clause scores high; a reply with an ordinary repeated
    particle does not, because a single token is shorter than the span. Returns
    0.0 for anything too short to have a span, which is not a pass, it is "no
    evidence".
    """
    words = text.lower().split()
    if len(words) < n * 2:
        return 0.0
    spans = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen: dict[tuple, int] = {}
    for span in spans:
        seen[span] = seen.get(span, 0) + 1
    duplicated = sum(count - 1 for count in seen.values() if count > 1)
    return duplicated / len(spans)


def old_sampler(model: str, temperature: float, num_thread: int, max_tokens: int) -> LLMConfig:
    return LLMConfig(
        model=model,
        temperature=temperature,
        num_thread=num_thread,
        max_tokens=max_tokens,
        min_p=0.0,           # not sent
        dry_multiplier=0.0,  # not sent
    )


def new_sampler(ccfg: dict, model: str) -> LLMConfig:
    return LLMConfig(
        model=ccfg.get("model", model),
        temperature=float(ccfg.get("temperature", 0.3)),
        num_thread=int(ccfg.get("num_thread", 0) or 0),
        max_tokens=int(ccfg.get("max_tokens", 260)),
        keep_alive=str(ccfg.get("keep_alive", "1h")),
        min_p=float(ccfg.get("min_p", 0.05)),
        top_p=float(ccfg.get("top_p", 1.0)),
        top_k=int(ccfg.get("top_k", 0) or 0),
        dry_multiplier=float(ccfg.get("dry_multiplier", 0.8)),
        dry_base=float(ccfg.get("dry_base", 1.75)),
        dry_allowed_length=int(ccfg.get("dry_allowed_length", 2)),
    )


def ask(client: LocalLLMClient, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    reply = client.chat(prompt, [], ARCHER_SYSTEM_PROMPT)
    return reply, time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greetings", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()

    audit = _audit()
    from config.config_loader import get_config

    ccfg = (get_config() or {}).get("chat", {}) or {}
    model = str(ccfg.get("model", "archer:7b"))

    old = LocalLLMClient(old_sampler(
        model, float(ccfg.get("temperature", 0.3)),
        int(ccfg.get("num_thread", 0) or 0), int(ccfg.get("max_tokens", 260)),
    ))
    new = LocalLLMClient(new_sampler(ccfg, model))

    greetings = audit.GREETINGS[: args.greetings]
    pairs = audit.short_pairs(args.pairs)

    print(f"model {model}  old = temperature only  new = min_p {new.cfg.min_p} + DRY {new.cfg.dry_multiplier}\n")

    print("echo and repetition, same prompt through both samplers")
    tally = {"old": {"echo": 0, "rep": [], "t": []}, "new": {"echo": 0, "rep": [], "t": []}}
    for greeting in greetings:
        prompt = route(greeting).prompt
        row = {}
        for label, client in (("old", old), ("new", new)):
            reply, dt = ask(client, prompt)
            echo = audit.is_echo(greeting, reply)
            rep = repeated_span_rate(reply)
            tally[label]["echo"] += echo
            tally[label]["rep"].append(rep)
            tally[label]["t"].append(dt)
            row[label] = (reply, echo, rep)
        print(f"  {greeting!r}")
        for label in ("old", "new"):
            reply, echo, rep = row[label]
            print(f"    {label} {'ECHO' if echo else 'ok  '} rep {rep:.2f}  {reply[:66]!r}")

    print("\ntranslation, same held-out sentences through both samplers")
    scores = {"old": [], "new": []}
    for pair in pairs:
        prompt = route(f"translate into Plains Cree: {pair['english']}").prompt
        line = f"  {pair['english']!r}\n    want {pair['cree'][:70]!r}"
        for label, client in (("old", old), ("new", new)):
            reply, dt = ask(client, prompt)
            score = audit.overlap(pair["cree"], reply)
            scores[label].append(score)
            line += f"\n    {label} {score:.2f}  {reply[:70]!r}"
        print(line)

    print("\nsummary")
    for label in ("old", "new"):
        e = tally[label]
        print(
            f"  {label}: echo {e['echo']}/{len(greetings)}  "
            f"repeat {statistics.mean(e['rep']):.2f}  "
            f"mean {statistics.mean(e['t']):.1f}s  "
            f"translation overlap {statistics.mean(scores[label]) if scores[label] else 0:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

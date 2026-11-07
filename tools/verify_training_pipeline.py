#!/usr/bin/env python3
"""Verify that the 14B pipeline still contains every required improvement.

This is a wiring check, not a training run. It reads the cell sources that the
notebook and train_cree_llm.py are generated from, and fails if a required stage
or data path has been removed. Run it before a GPU run:

    python tools/verify_training_pipeline.py

It exits 1 on any missing piece, so "the pipeline is complete" is a check and
not a claim.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "training" / "cree_llm" / "cells"
TRAINER = ROOT / "train_cree_llm.py"
NOTEBOOK = ROOT / "training" / "notebooks" / "Cree_LLM_Training.ipynb"

FAIL: list[str] = []


def text(name: str) -> str:
    path = CELLS / name
    if not path.is_file():
        FAIL.append(f"missing {path}")
        return ""
    return path.read_text(encoding="utf-8")


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"{mark} {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAIL.append(name)


setup = text("01_setup.py")
corpus = text("03_corpus.py")
clean = text("04_clean.py")
tokenizer = text("05_tokenizer.py")
sft = text("08_sft_data.py")
dpo = text("10_dpo.py")
evaluation = text("11_eval.py")
export = text("13_export.py")
runbook = (ROOT / "docs" / "server-training-runbook.md").read_text(encoding="utf-8")

check("14B profile 'big' uses Qwen2.5-14B",
      'base="Qwen/Qwen2.5-14B-Instruct"' in setup and '"big": dict' in setup)
check("24 GB 14B profile 'big24' exists",
      '"big24": dict' in setup and setup.count('Qwen/Qwen2.5-14B-Instruct') >= 2)
check("export name follows the trained base, not a fixed 7B",
      'MODEL_TAG = next' in setup and 'f"archer-{MODEL_TAG}"' in setup)
check("corpus has a completeness gate",
      "corpus incomplete" in corpus and "allow_missing_sources" in corpus)
check("cleaning has language filtering and MinHash deduplication",
      "lang_guess" in clean and "MinHash" in clean)
check("cleaning fails on held-out 12-gram contamination",
      "contamination_12gram_overlap" in clean and "held-out 12-gram contamination" in clean)
check("tokenizer reports a boundary and fertility audit",
      "tokenizer_audit.json" in tokenizer)
check("SFT has varied casual conversation and persona examples",
      '"casual"' in sft and '"persona"' in sft and "_CASUAL_MORE" in sft)
check("SFT trains language identification and answer-language policy",
      "language_id" in sft and "language_policy" in sft and "Which language is this?" in sft)
check("14B profiles warn below 20M tokens and still train on all data",
      "min_tokens=20_000_000" in setup and "training continues with all data" in corpus
      and "corpus_size.json" in corpus)
check("Little Cree Books and extra public-domain Cree volumes are wired in",
      "def load_lcb" in corpus and "rosettaproject_crk_vertxt-2" in corpus)
check("cleaning audits duplicate words and duplicate rows",
      "duplicate_audit.json" in clean and "spelling_variants" in clean
      and "no_duplicate_sentences" in clean and "_folded" in clean)
check("the duplicate audit fails the stage on a surviving duplicate",
      "duplicate audit failed" in clean)
check("runbook documents durable execution",
      all(word in runbook for word in ("tmux", "nohup", "systemd", "Restart=on-failure")))
check("resumable server driver exists and refuses a second copy",
      (ROOT / "scripts" / "train_archer_14b.sh").is_file()
      and "flock" in (ROOT / "scripts" / "train_archer_14b.sh").read_text(encoding="utf-8"))
check("SFT reads runtime corrections",
      "feedback_sft.jsonl" in sft and "load_feedback_sft" in sft)
check("SFT weights corrections above ordinary examples",
      "feedback_correction" in sft and '"feedback_correction": 6' in sft)
check("SFT names the actual base model in project answers",
      "Qwen2.5-7B-Instruct" not in sft and "{CFG.base_model}" in sft)
check("DPO reads runtime correction pairs",
      "feedback_dpo.jsonl" in dpo and "feedback_prefs" in dpo)
check("DPO has echo and verbatim negatives",
      "_echo_prefs" in dpo and "echo negatives" in dpo)
check("DPO fingerprint invalidates an older adapter",
      "substitute-echo-feedback-v2" in dpo)
check("evaluation gates attested Cree, abstention and echo",
      "lexicon_validity" in evaluation and "abstention" in evaluation
      and "echo rate" in evaluation and "Greeting echo" not in evaluation)
check("export derives the Ollama tag from the trained size",
      "ollama create archer:{MODEL_TAG}" in export and "archer:7b" not in export)
check("14B runbook installs archer:14b",
      "ollama create archer:14b" in (ROOT / "docs" / "pipeline-best-chatbot.md").read_text(encoding="utf-8"))
check("feedback is documented in the runbook",
      "feedback_sft.jsonl" in (ROOT / "docs" / "pipeline-best-chatbot.md").read_text(encoding="utf-8"))
check("1,000-row benchmark generator exists",
      (ROOT / "tools" / "make_model_benchmark.py").is_file()
      and (ROOT / "data" / "eval" / "model_benchmark_1000.jsonl").is_file())
_bench_path = ROOT / "data" / "eval" / "model_benchmark_1000.jsonl"
_bench_rows = [json.loads(line) for line in _bench_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if _bench_path.is_file() else []
check("benchmark is exactly 1,000 unique prompts",
      len(_bench_rows) == 1000 and len({row.get("prompt") for row in _bench_rows}) == 1000,
      f"{len(_bench_rows)} rows")
check("benchmark evaluator and frontend metrics publisher exist",
      (ROOT / "tools" / "eval_model_benchmark.py").is_file()
      and (ROOT / "tools" / "publish_metrics.py").is_file())

# The generated copies must match the cells. A cell edit that was not rebuilt is
# exactly how the notebook and the command line trainer drift apart.
for label, script in (("trainer", ROOT / "training" / "cree_llm" / "build_trainer.py"),
                      ("notebook", ROOT / "training" / "cree_llm" / "build_notebook.py")):
    result = subprocess.run([sys.executable, str(script), "--check"],
                            capture_output=True, text=True)
    check(f"generated {label} matches its cell sources", result.returncode == 0,
          (result.stdout + result.stderr).strip().splitlines()[-1] if (result.stdout + result.stderr).strip() else "")

for path in (TRAINER, NOTEBOOK):
    check(f"{path.name} exists", path.is_file())

print()
if FAIL:
    print(f"{len(FAIL)} pipeline check(s) failed")
    sys.exit(1)
print("all pipeline checks passed")

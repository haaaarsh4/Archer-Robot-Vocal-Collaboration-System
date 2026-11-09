"""Turn real mistakes into future behaviour, and into data for the next run.

There are two loops, and they happen at different speeds.

The fast loop is memory: a correction is stored against the answer it corrects,
and the next similar question is answered with that correction in front of the
model. That changes behaviour immediately, without touching the weights.

The slow loop is training: the same correction is written as a DPO pair
(``chosen`` is what the visitor said, ``rejected`` is what the model said) and
as an SFT example. Those files are written in the exact shapes
``training/cree_llm/cells/08_sft_data.py`` and ``cell/10_dpo.py`` consume, so a
later GPU run can bake the fix into the model instead of retrieving it every
time.

The third output is a regression file. A correction that is never tested is a
correction that comes back, so every one becomes a case that
``tools/run_regressions.py`` replays against the installed model.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from chat.memory import Memory

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "training"
EVAL_DIR = ROOT / "data" / "eval"

# "No, that is wrong", "actually it was recorded in 1974", "you said X but it is
# Y". Deliberately narrow: a message beginning with "no" can be a new question,
# and turning every one of those into a training example would poison the data.
_CORRECTION = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect|that(?:'s| is) (?:wrong|not right)|"
    r"not quite|actually)\b"
    r"|\byou (?:said|wrote|answered)\b[^.!?]{0,80}\bbut\b"
    r"|\bit(?:'s| is) actually\b",
    re.IGNORECASE,
)

_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember(?:\s+that)?|keep\s+in\s+mind(?:\s+that)?)"
    r"\s*[:,-]?\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_NOTE_TO_SELF = re.compile(
    r"^\s*note\s+to\s+self\s*[:,-]\s*(.+)$", re.IGNORECASE | re.DOTALL
)


def looks_like_correction(text: str) -> bool:
    """Is this message correcting the previous answer, rather than asking anew?"""
    text = " ".join(str(text).split())
    if not text or len(text) > 600:
        return False
    return bool(_CORRECTION.search(text))


# The answer inside a correction, not the complaint around it. These patterns
# are applied only after looks_like_correction has said the message is one.
_EXACT_ANSWER = re.compile(
    r"\b(?:reply|answer|respond|say)\s+(?:exactly\s*)?(?:with\s+)?[:,-]?\s*"
    r"[\"'“”]?(.+?)[\"'“”]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SHOULD_BE = re.compile(
    r"\b(?:the correct answer is|it should (?:be|say)|it is actually|"
    r"it's actually)\b\s*[:,-]?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_BUT_ANSWER = re.compile(
    r"\byou (?:said|wrote|answered)\b.*?\bbut\b\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_COMPLAINT = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect|that(?:'s| is) (?:wrong|not right)|"
    r"not quite|actually)\b[\s,.;:!-]*",
    re.IGNORECASE,
)


def corrected_answer(text: str) -> str:
    """The correction without the complaint that introduced it.

    "No, that was wrong. Reply exactly: Good day." becomes "Good day." so the
    stored correction, the DPO ``chosen`` answer and the regression expectation
    are all the answer the visitor wanted, not a transcript of the argument.
    """
    text = " ".join(str(text).split())
    if not text:
        return ""
    for pattern in (_EXACT_ANSWER, _SHOULD_BE, _BUT_ANSWER):
        match = pattern.search(text)
        if match:
            text = match.group(1)
            break
    else:
        text = _COMPLAINT.sub("", text, count=1)
    return text.strip().strip('"\'“”').strip()[:1000]


def remember_command(text: str) -> str | None:
    """The fact in "remember that X", or None.

    Only an explicit command is stored. Inferring facts from ordinary sentences
    is how a memory system becomes confidently wrong about its user.
    """
    for pattern in (_REMEMBER, _NOTE_TO_SELF):
        match = pattern.match(" ".join(str(text).split()))
        if match:
            fact = match.group(1).strip().strip('"“”')
            return fact[:1000] or None
    return None


def export(memory: Memory | None = None, train_dir: str | Path | None = None,
           eval_dir: str | Path | None = None) -> dict:
    """Write the training and regression files from the feedback store.

    Returns a small report rather than printing, so the endpoint and the CLI
    share one implementation.
    """
    memory = memory or Memory()
    train_dir = Path(train_dir or TRAIN_DIR)
    eval_dir = Path(eval_dir or EVAL_DIR)
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    from chat.archer_prompt import ARCHER_SYSTEM_PROMPT

    sft: list[dict] = []
    dpo: list[dict] = []
    regressions: list[dict] = []
    for row in memory.feedback_rows():
        user = str(row.get("user") or "").strip()
        assistant = str(row.get("assistant") or "").strip()
        correction = str(row.get("correction") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        if not user or not assistant:
            continue
        if correction:
            sft.append({
                "messages": [
                    {"role": "system", "content": ARCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": correction},
                ],
                "task": "feedback_correction",
            })
            dpo.append({
                "prompt": prompt or f"<|song|>\n{user}",
                "chosen": correction,
                "rejected": assistant,
            })
            regressions.append({
                "user": user,
                "expected": correction,
                "rejected": assistant,
                "created_at": row.get("created_at") or time.time(),
            })
        elif row.get("rating") == "up":
            sft.append({
                "messages": [
                    {"role": "system", "content": ARCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "task": "feedback_up",
            })

    sft_path = train_dir / "feedback_sft.jsonl"
    dpo_path = train_dir / "feedback_dpo.jsonl"
    _write_jsonl(sft_path, sft)
    _write_jsonl(dpo_path, dpo)

    regression_path = eval_dir / "regressions.jsonl"
    merged: dict[str, dict] = {}
    for row in load_regressions(regression_path):
        if row.get("user"):
            merged[str(row["user"])] = row
    for row in regressions:
        merged[str(row["user"])] = row
    _write_jsonl(regression_path, list(merged.values()))

    return {
        "sft": len(sft),
        "dpo": len(dpo),
        "regressions": len(merged),
        "sft_path": str(sft_path),
        "dpo_path": str(dpo_path),
        "regression_path": str(regression_path),
    }


def load_regressions(path: str | Path | None = None) -> list[dict]:
    path = Path(path or EVAL_DIR / "regressions.jsonl")
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("user"):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

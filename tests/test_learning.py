"""The learning loop: detect a correction, store it, export it, replay it.

    PYTHONPATH=. .venv/bin/python tests/test_learning.py
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_learning.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chat import learning
from chat.memory import Memory


def test_corrections_are_detected_without_catching_new_questions():
    assert learning.looks_like_correction("no, it was recorded in D major")
    assert learning.looks_like_correction("that's wrong, the key is D")
    assert learning.looks_like_correction("actually it is maskisin")
    assert not learning.looks_like_correction("what key was it recorded in?")
    assert not learning.looks_like_correction("tell me about the song")


def test_a_correction_is_stored_without_the_complaint():
    assert learning.corrected_answer(
        "no, that was wrong. Reply exactly: Good day.") == "Good day."
    assert learning.corrected_answer(
        "no, it was recorded in D major") == "it was recorded in D major"
    assert learning.corrected_answer(
        "you said it was D but it is E") == "it is E"
    assert learning.corrected_answer("actually it is maskisin") == "it is maskisin"
    assert learning.corrected_answer("no.") == ""


def test_remember_only_takes_an_explicit_command():
    assert learning.remember_command("remember that Archer's dog is Niska") == \
        "Archer's dog is Niska"
    assert learning.remember_command("remember: the key is D major") == \
        "the key is D major"
    assert learning.remember_command("I have a dog") is None


def test_export_writes_dpo_sft_and_a_regression():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        store = Memory(root / "memory.sqlite3")
        turn_id = store.record_turn(
            "s1", "what key was Horse recorded in?",
            "The notes do not say.", "<|song|>\nwhat key", "song_notes", "en",
        )
        store.record_feedback(
            turn_id=turn_id, session_id="s1", rating="down",
            correction="The notes say Horse was recorded in D major.",
        )
        report = learning.export(
            store, train_dir=root / "train", eval_dir=root / "eval",
        )
        assert report["dpo"] == 1 and report["sft"] == 1 and report["regressions"] == 1

        dpo = [json.loads(line) for line in
               (root / "train" / "feedback_dpo.jsonl").read_text().splitlines()]
        assert dpo[0]["chosen"].startswith("The notes say Horse")
        assert dpo[0]["rejected"] == "The notes do not say."
        assert dpo[0]["prompt"].startswith("<|song|>")

        sft = [json.loads(line) for line in
               (root / "train" / "feedback_sft.jsonl").read_text().splitlines()]
        assert sft[0]["task"] == "feedback_correction"
        assert sft[0]["messages"][-1]["content"].startswith("The notes say Horse")

        rows = learning.load_regressions(root / "eval" / "regressions.jsonl")
        assert rows[0]["user"] == "what key was Horse recorded in?"
        assert "D major" in rows[0]["expected"]


def _main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

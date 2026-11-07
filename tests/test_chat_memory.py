"""The memory store: every turn, every correction, and recall that finds them.

    PYTHONPATH=. .venv/bin/python tests/test_chat_memory.py
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_chat_memory.py -q
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from chat.memory import Memory


def test_a_turn_is_stored_and_recalled():
    with tempfile.TemporaryDirectory() as tmp:
        store = Memory(pathlib.Path(tmp) / "memory.sqlite3")
        turn_id = store.record_turn(
            "s1", "what key was Horse recorded in?",
            "The notes do not say.", "<|song|>\nwhat key", "song_notes", "en",
        )
        assert turn_id > 0
        hits = store.recall("what key was Horse recorded in?")
        assert hits and hits[0].kind == "turn"
        assert "recorded" in hits[0].text


def test_a_correction_outranks_the_answer_it_corrects():
    with tempfile.TemporaryDirectory() as tmp:
        store = Memory(pathlib.Path(tmp) / "memory.sqlite3")
        turn_id = store.record_turn(
            "s1", "what key was Horse recorded in?",
            "The notes do not say.", "", "song_notes", "en",
        )
        assert store.record_feedback(
            "s1", turn_id, "down",
            "The notes say Horse was recorded in D major.",
        ) > 0
        hits = store.recall("what key was Horse recorded in?")
        assert hits[0].kind == "correction"
        assert "D major" in hits[0].text


def test_a_remembered_fact_is_recalled_by_content():
    with tempfile.TemporaryDirectory() as tmp:
        store = Memory(pathlib.Path(tmp) / "memory.sqlite3")
        assert store.remember("Archer's dog is called Niska.") > 0
        hits = store.recall("what is the dog called?")
        assert any(hit.kind == "fact" and "Niska" in hit.text for hit in hits)


def test_last_turn_recent_stats_and_clear():
    with tempfile.TemporaryDirectory() as tmp:
        store = Memory(pathlib.Path(tmp) / "memory.sqlite3")
        store.record_turn("s1", "one", "first")
        store.record_turn("s1", "two", "second")
        assert store.last_turn("s1").user == "two"
        assert [turn.user for turn in store.recent("s1", 2)] == ["one", "two"]
        stats = store.stats()
        assert stats["turns"] == 2
        store.clear()
        assert store.stats()["turns"] == 0
        assert store.recall("one") == []


def test_a_locked_or_bad_path_never_raises():
    with tempfile.TemporaryDirectory() as tmp:
        store = Memory(pathlib.Path(tmp) / "no" / "such" / "memory.sqlite3")
        assert store.record_turn("s1", "hello", "hi") > 0
        assert isinstance(store.recall("hello"), list)


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

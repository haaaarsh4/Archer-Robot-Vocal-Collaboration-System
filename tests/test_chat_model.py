"""The chat is one model. These tests pin that down.

They check the prompt shapes, the training prompt staying in step with the app,
and the config having no second model to fall back to.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from chat.archer_prompt import (
    ARCHER_SYSTEM_PROMPT,
    format_prompt,
    looks_like_cree,
    script_of,
    task_of,
)

ROOT = Path(__file__).resolve().parents[1]


def test_system_prompt_describes_one_assistant():
    assert "four jobs" in ARCHER_SYSTEM_PROMPT
    assert "never speak as Archer" in ARCHER_SYSTEM_PROMPT
    assert "the notes do not say" in ARCHER_SYSTEM_PROMPT


def test_training_system_prompt_is_the_app_system_prompt():
    cell = (ROOT / "training/cree_llm/cells/08_sft_data.py").read_text(encoding="utf-8")
    tree = ast.parse(cell)
    trained = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SYSTEM" for t in node.targets
        ):
            trained = ast.literal_eval(node.value)
    assert trained == ARCHER_SYSTEM_PROMPT, "the SFT system prompt drifted from the app"


def test_script_detection():
    assert script_of("tânisi") == "Latn"
    assert script_of("ᑖᓂᓯ") == "Cans"
    assert script_of("hello there") is None
    assert looks_like_cree("tânisi")
    assert not looks_like_cree("What is this project?")


@pytest.mark.parametrize(
    "message, has_notes, expected",
    [
        ("What is the song Horse about?", True, "song_notes"),
        ("What is the song Horse about?", False, "chat"),
        ("How do you say river in Cree?", False, "translate"),
        ("Translate into Cree: the river is cold", False, "translate"),
        ("tânisi", False, "translate"),
        ("What does nêhiyawêwin mean?", False, "chat"),
        # A retrieved note must not turn a word question into a song question.
        ("What does nêhiyawêwin mean?", True, "chat"),
        ("What is the Cree word for river?", False, "translate"),
        ("What does the Cree word maskwa mean?", False, "chat"),
        ("Explain the Cree word maskwa.", False, "chat"),
        ("Write nêhiyawêwin in syllabics", False, "orthography"),
        ("Write ᑖᓂᓯ in SRO", False, "orthography"),
    ],
)
def test_task_of_is_a_prompt_shape(message, has_notes, expected):
    assert task_of(message, has_notes) == expected


def test_notes_question_uses_the_training_shape():
    prompt = format_prompt(
        "What is Horse about?",
        [("Horse", "The song began as a vocable and became a round.")],
    )
    assert prompt.startswith("<|song|><|note|>\nNotes on Horse:\n")
    assert prompt.endswith("Question: What is Horse about?")


def test_a_question_with_no_retrieved_notes_takes_the_chat_path():
    # Retrieval, not a language guess, decides whether notes are present. Below
    # the 0.30 floor the app passes an empty list and the prompt is this shape.
    prompt = format_prompt("What is this project?")
    assert prompt == "<|song|>\nWhat is this project?"


def test_translation_is_never_padded_with_song_prose():
    prompt = format_prompt(
        "How do you say river in Cree?", [("Horse", "a song about a horse")]
    )
    assert prompt.startswith("<|translate|><|english|><|sro|>\n")
    assert "Notes on" not in prompt


def test_cree_to_english_prompt():
    prompt = format_prompt("tânisi")
    assert prompt.startswith("<|translate|><|cree|><|english|>\n")
    assert prompt.endswith("\ntânisi")


def test_syllabics_input_keeps_its_script_marker():
    prompt = format_prompt("ᑖᓂᓯ")
    assert prompt.startswith("<|translate|><|cree|><|english|>\n<|cans|>\n")


def test_syllabics_conversion_in_both_directions():
    to_syllabics = format_prompt("Write nêhiyawêwin in syllabics")
    assert to_syllabics.startswith("<|sro|><|cans|>\n")
    to_sro = format_prompt("Write ᑖᓂᓯ in SRO")
    assert to_sro.startswith("<|cans|><|sro|>\n")


def test_config_has_one_chat_model():
    import yaml

    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    chat = cfg["chat"]
    assert chat["model"] == "archer:7b"
    assert "cree_model" not in chat
    assert "cree_gguf_path" not in chat
    assert "ollama_model" not in chat


def test_client_defaults_to_the_one_model():
    from chat.llm_client import LLMConfig

    cfg = LLMConfig()
    assert cfg.model == "archer:7b"
    assert not hasattr(cfg, "cree_model")


def test_missing_model_fails_instead_of_falling_back(monkeypatch):
    import chat.llm_client as llm_client

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "something-else:latest"}]}

    monkeypatch.setattr(llm_client.requests, "get", lambda *a, **k: _Resp())
    client = llm_client.LocalLLMClient(llm_client.LLMConfig())
    with pytest.raises(RuntimeError, match="ollama create"):
        client.chat("hello", [])


def test_the_chat_endpoint_has_no_route_branch():
    source = (ROOT / "chat/chat_api.py").read_text(encoding="utf-8")
    assert "/route" not in source
    assert "/prompt" in source
    assert "choose_route" not in source
    assert re.search(r"MIN_NOTE_SCORE\s*=\s*0\.30", source)

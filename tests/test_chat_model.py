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
    route,
    script_of,
    task_of,
)

ROOT = Path(__file__).resolve().parents[1]


def test_system_prompt_describes_one_assistant():
    assert "four jobs" in ARCHER_SYSTEM_PROMPT
    assert "never speak as Archer" in ARCHER_SYSTEM_PROMPT
    assert "the notes do not say" in ARCHER_SYSTEM_PROMPT
    # A greeting is not a project status report. This sentence is in the
    # training system prompt too, so the next SFT run learns it.
    assert "If someone greets you" in ARCHER_SYSTEM_PROMPT
    assert "Do not describe the project" in ARCHER_SYSTEM_PROMPT


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


HORSE = [("Horse", "Horse is an imagined story set in 1868, about memory and responsibility.")]


def test_a_song_question_that_asks_for_cree_keeps_its_notes():
    # "in Cree" is an output preference here, not a request to translate the
    # question. The old router dropped the notes and translated the request.
    plan = route("explain Horse in Cree", HORSE)
    assert plan.task == "song_notes"
    assert plan.answer_language == "cr"
    assert plan.prompt.startswith("<|song|><|note|>\nNotes on Horse:\n")
    assert "Question: explain Horse\nAnswer in English." in plan.prompt
    assert "Translate into Plains Cree (SRO):\nexplain Horse in Cree" not in plan.prompt


def test_a_note_question_that_uses_mean_keeps_its_notes():
    plan = route("What does the warning in Horse mean?", HORSE)
    assert plan.task == "song_notes"
    assert "Notes on Horse:" in plan.prompt
    assert plan.prompt.endswith("Question: What does the warning in Horse mean?")


def test_a_vocabulary_question_does_not_turn_into_a_song_question():
    plan = route("What is the Cree word for horse?", HORSE)
    assert plan.task == "translate"
    assert plan.prompt == (
        "<|translate|><|english|><|sro|>\n"
        "Translate into Plains Cree (SRO):\nhorse"
    )
    assert "Notes on" not in plan.prompt


def test_a_follow_up_translation_uses_the_previous_answer():
    history = [
        ("user", "What is Horse about?"),
        ("assistant", "The notes say Horse is an imagined story set in 1868."),
    ]
    plan = route("can you answer that in cree pls", None, history)
    assert plan.task == "translate"
    assert plan.used_history is True
    assert plan.payload == "The notes say Horse is an imagined story set in 1868."
    assert plan.prompt.endswith(
        "Translate into Plains Cree (SRO):\n"
        "The notes say Horse is an imagined story set in 1868."
    )
    assert "can you answer" not in plan.prompt


def test_pls_and_eng_are_part_of_the_request_not_the_payload():
    plan = route("PLS TRANSLATE THIS TO ENG PLS: kîspin ê îki nihiyawi îkâyihton cî ?")
    assert plan.task == "translate"
    assert plan.payload == "kîspin ê îki nihiyawi îkâyihton cî ?"
    assert plan.prompt.startswith("<|translate|><|cree|><|english|>\n")
    assert "Translate into English:\nkîspin" in plan.prompt
    assert "PLS" not in plan.prompt and "ENG" not in plan.prompt


def test_a_cree_instruction_before_a_song_question_is_not_the_question():
    plan = route("Answer in Cree: what is Horse about?", HORSE)
    assert plan.task == "song_notes"
    assert plan.answer_language == "cr"
    assert "Question: what is Horse about?" in plan.prompt
    assert "Answer in Cree: what is Horse about?" not in plan.prompt


def test_a_script_conversion_can_refer_to_the_previous_answer():
    history = [("user", "Say something in Cree"), ("assistant", "tânisi")]
    plan = route("convert that to syllabics", None, history)
    assert plan.task == "orthography"
    assert plan.used_history is True
    assert plan.prompt == "<|sro|><|cans|>\nWrite this in Cree syllabics:\ntânisi"


def test_a_greeting_takes_the_chat_path():
    assert task_of("hi, how is it going?") == "chat"
    assert route("hi, how is it going?").prompt == "<|song|>\nhi, how is it going?"


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


@pytest.mark.parametrize(
    "message, expected",
    [
        ("How do you say river in Cree?", "river"),
        ("What is the Cree word for river?", "river"),
        ("what is river in cree", "river"),
        ("Translate into Cree: the river is cold", "the river is cold"),
        ("Translate the river is cold into Cree", "the river is cold"),
        ("Write nêhiyawêwin in syllabics", "nêhiyawêwin"),
        ("Write tânisi in Cree syllabics.", "tânisi"),
        ("Write ᑖᓂᓯ in SRO", "ᑖᓂᓯ"),
        ("Write this in Cree syllabics:\ntânisi", "tânisi"),
        ("Say 'the river is cold' in Cree", "the river is cold"),
        ("Translate into Cree: it's John's hat", "it's John's hat"),
        ('Say "the river is cold" in Cree', "the river is cold"),
        ("tânisi", "tânisi"),
    ],
)
def test_the_prompt_carries_the_phrase_not_the_request(message, expected):
    # The model was trained on the header followed by the sentence. Handing it
    # the question that asked for the sentence produces a translation of the
    # question, which is what happened before the payload was unwrapped.
    prompt = format_prompt(message)
    assert prompt.rsplit("\n", 1)[-1] == expected


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


def test_cree_guardrail_rejects_copied_english():
    from chat.chat_api import _reliable_cree

    copied = (
        'From the notes on Horse: The story also uses "Garry Owen", an Irish '
        "marching tune that was historically associated with the U.S. ê têtipêk "
        "mostos kisisapimôw."
    )
    assert not _reliable_cree(copied)
    assert not _reliable_cree("The river is cold.")


def test_cree_guardrail_rejects_a_repeating_loop():
    from chat.chat_api import _reliable_cree

    loop = (
        "ê itâcimostâk âmaciwêw kîhasêk îkhasêk kîki masinawason îkhasêk îkhasêk "
        "îkhasêk ê ispîhci mâmaskâc pîwâwi manâ kîhasêk îkhasêw mîhcêt îwastawâwâ "
        "ê pîmikinêhk ôhi misatimwa."
    )
    assert not _reliable_cree(loop)


def test_cree_guardrail_accepts_short_cree_and_syllabics():
    from chat.chat_api import _reliable_cree

    assert _reliable_cree("sîpiy kîsow.")
    assert _reliable_cree("êkêyêwpâi masinôw awa misatim ê mîskawikiyit.")
    assert _reliable_cree("ᑖᓂᓯ")

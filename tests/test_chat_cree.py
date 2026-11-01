"""The Cree chat route: detection, routing and configuration.

No model is loaded here. These tests protect the decision that sends a question
to the trained Cree model, because a wrong route is invisible in a transcript:
the general model will still answer, just without the language training.
"""

import yaml

from chat.cree_prompt import (
    CREE_SYSTEM_PROMPT,
    choose_route,
    format_cree_prompt,
    looks_like_cree,
    script_of,
    wants_cree,
)


def test_sro_is_cree():
    assert looks_like_cree("tânisi kiya?")
    assert script_of("tânisi kiya?") == "Latn"


def test_syllabics_is_cree():
    text = "ᑖᓂᓯ ᑮᐢᐱᐣ"
    assert looks_like_cree(text)
    assert script_of(text) == "Cans"


def test_english_about_cree_routes_to_the_cree_model():
    # The sentence is English, but the subject is Cree, and the Cree model is
    # the one with the lexicon and the orthography training.
    text = "What does the Cree word maskwa mean?"
    assert not looks_like_cree(text)
    assert wants_cree(text) is True
    assert choose_route(text, "auto", "archer-cree:7b") == "cree"


def test_explicit_translation_request_routes_to_cree():
    assert wants_cree("Translate this into Cree: the river is cold")
    assert wants_cree("write it in syllabics")
    assert choose_route("Translate this into Cree: the river is cold", "auto", "archer-cree:7b") == "cree"


def test_english_stays_english():
    msg = "Tell me about the song Horse and how it was performed."
    assert not wants_cree(msg)
    assert choose_route(msg, "auto", "archer-cree:7b") == "english"


def test_missing_model_never_fails_a_request():
    assert choose_route("tânisi", "auto", "") == "english"
    assert choose_route("tânisi", "cree", "") == "english"


def test_explicit_choice_wins():
    assert choose_route("hello", "cree", "archer-cree:7b") == "cree"
    assert choose_route("tânisi", "english", "archer-cree:7b") == "english"


def test_mentioning_cree_is_not_asking_for_it():
    msg = "Tell me about Cree music and the bands in this scene."
    assert choose_route(msg, "auto", "archer-cree:7b") == "english"


def test_notes_prompt_matches_the_training_shape():
    prompt = format_cree_prompt(
        "What does Horse mean?",
        [("Horse", "It is a song about a horse.")],
    )
    assert prompt.startswith("<|song|><|note|>\nNotes on Horse:\n")
    assert prompt.endswith("\n\nQuestion: What does Horse mean?")


def test_translation_request_is_not_padded_with_notes():
    prompt = format_cree_prompt(
        "Translate this into Cree: the river is cold",
        [("Horse", "It is a song about a horse.")],
    )
    assert prompt.startswith("<|translate|><|english|><|sro|>\n")
    assert "Notes on" not in prompt


def test_cree_text_is_translated_towards_english():
    prompt = format_cree_prompt("What does nêhiyawêwin mean?")
    assert prompt.startswith("<|translate|><|cree|><|english|>\n")


def test_sro_to_syllabics_uses_the_right_direction():
    prompt = format_cree_prompt("Write nêhiyawêwin in syllabics")
    assert prompt.startswith("<|sro|><|cans|>\n")


def test_syllabics_to_sro_uses_the_right_direction():
    prompt = format_cree_prompt("Write ᑖᓂᓯ in SRO")
    assert prompt.startswith("<|cans|><|sro|>\n")


def test_prompt_is_the_training_prompt():
    # The notebook's SYSTEM string and this one must agree; a different prompt
    # at inference time is a distribution shift.
    assert CREE_SYSTEM_PROMPT.startswith("You are the nêhiyawêwin (Plains Cree) language guide")
    assert "never speak as Archer" in CREE_SYSTEM_PROMPT


def test_config_exposes_the_cree_model():
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    chat = cfg["chat"]
    assert chat["cree_model"] == ""
    assert chat["cree_temperature"] <= chat["temperature"]
    assert chat["cree_max_tokens"] > 0


def test_chat_client_reads_the_config():
    from chat import chat_api

    chat_api._llm_client = None
    cfg = chat_api._get_llm().cfg
    assert cfg.cree_model == ""
    assert cfg.cree_temperature == 0.2
    assert cfg.cree_max_tokens == 400


def test_ollama_request_uses_the_requested_model(monkeypatch):
    from chat.llm_client import LLMConfig, LocalLLMClient

    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "tânisi"}}

    def fake_post(url, json, timeout):
        seen.update(json)
        return Response()

    monkeypatch.setattr("chat.llm_client.requests.post", fake_post)
    client = LocalLLMClient(LLMConfig(cree_model="archer-cree:7b"))
    client._checked_models.add("archer-cree:7b")
    reply = client.chat(
        "tânisi", [], [], system_prompt="cree", model="archer-cree:7b",
        temperature=0.2, max_tokens=64,
    )
    assert reply == "tânisi"
    assert seen["model"] == "archer-cree:7b"
    assert seen["options"]["temperature"] == 0.2
    assert seen["options"]["num_predict"] == 64

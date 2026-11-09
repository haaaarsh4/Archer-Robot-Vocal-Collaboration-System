"""The chat router answers in the language the visitor wrote in.

Two failures are covered here. The router appended "Answer in English." to any
chat prompt whose message contained Cree, including when it had just decided to
answer in Cree, so a Cree visitor was told to answer in English. And the output
language had to be asked for, so a Cree greeting was read as a sentence to
translate and came back as an English gloss of itself.

The last test is the guard in the other direction: a bare Cree sentence is still
a translation request, which is what the app is for, so the greeting rule must
not swallow it.
"""

from __future__ import annotations

from chat.archer_prompt import route

HORSE = [("Horse", "Horse is an imagined story set in 1868, about memory and responsibility.")]


def test_a_cree_greeting_is_answered_in_cree_without_being_asked():
    plan = route("tânisi, kîya mîna?", None)
    assert plan.task == "chat"
    assert plan.answer_language == "cr"
    assert "Answer in English" not in plan.prompt


def test_a_cree_question_to_the_assistant_is_not_a_translation_request():
    plan = route("kîkway kî-itôtaman?", None)
    assert plan.task == "chat"
    assert plan.answer_language == "cr"


def test_the_decision_to_answer_in_cree_is_not_contradicted_in_the_prompt():
    # This is the bug: the branch that chose Cree appended "Answer in English."
    plan = route("Answer in Cree: how are you?", None)
    assert plan.answer_language == "cr"
    assert "Answer in English" not in plan.prompt


def test_an_english_question_about_a_cree_phrase_is_still_answered_in_english():
    plan = route("what does kîsikâw mîna sîpiy mean?", None)
    assert plan.task == "chat"
    assert plan.answer_language == "en"
    assert "Answer in English" in plan.prompt


def test_a_single_cree_word_stays_a_lookup_rather_than_a_conversation():
    # One word is a dictionary lookup, so the greeting rule must not claim it.
    from chat.archer_prompt import _is_cree_greeting

    assert not _is_cree_greeting("tânisi")


def test_a_bare_cree_sentence_is_still_a_translation_request():
    plan = route("kîsikâw mîna sîpiy kîsow", None)
    assert plan.task == "translate"

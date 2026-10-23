"""The demo's model numbers are positions in config.yaml, held to that.

`frontend/index.html` builds one button per entry in
`synthesis.neural.model_paths` and sends the clicked position as
`voice_index`; the sidecar loads that same list by position. There is no
second mapping anywhere, which is what makes the order safe -- and also what
makes it invisible: moving an entry changes which audio a number plays with no
code change at all. These assertions are the thing that notices if the order
drifts, and they run with no model weights on the machine.

The intended order, as of the change that added the third voice:

    Model 1 -> female2
    Model 2 -> female3   (the newest voice; this is the one that moved)
    Model 3 -> mi-test   (the previous Model 2)
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _neural_config() -> dict:
    with open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["synthesis"]["neural"]


def _stems(paths: list[str]) -> list[str]:
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def test_the_three_voices_are_in_the_intended_order():
    cfg = _neural_config()
    assert _stems(cfg["model_paths"]) == ["female2", "female3", "mi-test"]


def test_every_model_slot_is_present_and_unique():
    cfg = _neural_config()
    paths = [p.strip() for p in cfg["model_paths"]]
    assert len(paths) == 3
    assert all(paths), "an empty model path would silently leave a button dead"
    assert len({p.lower() for p in paths}) == len(paths)


def test_index_and_transpose_slots_line_up_with_the_models():
    cfg = _neural_config()
    models = cfg["model_paths"]
    assert len(cfg["index_paths"]) == len(models), (
        "index_paths must be positional with model_paths; a short list would "
        "give a model someone else's index or none at all"
    )
    assert len(cfg["transpose_semitones"]) == len(models), (
        "transpose_semitones must be positional with model_paths, or the "
        "missing voices all fall back to 0 semitones"
    )
    # female3 has no retrieval index today, so its slot is deliberately empty
    # and must be read as 'no index', never as a path. If a matching index is
    # trained later, that is an improvement rather than a regression -- but it
    # has to belong to the voice in that slot, not be copied from another one.
    female3_index = cfg["index_paths"][1]
    assert female3_index == "" or "female3" in os.path.basename(female3_index).lower()


def test_the_shared_clarity_stage_is_configured_for_every_voice():
    cfg = _neural_config()
    clarity = cfg.get("clarity")
    assert isinstance(clarity, dict)
    assert clarity.get("presence_match") is True
    assert 0.0 < float(clarity.get("presence_max_lift_db", 0)) <= 6.0
    assert 0.9 <= float(clarity.get("peak_ceiling", 1.0)) <= 1.0

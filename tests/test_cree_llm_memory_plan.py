"""Checks for the Cree notebook's memory plan and vocabulary graft.

These run the notebook's own cell sources, not a copy of them, so a regression in
cells/01_setup.py or cells/06_model.py fails here. peft is stubbed because the
logic under test is ours: choosing a precision, resizing the vocabulary and
freezing the table.

    python -m pytest tests/test_cree_llm_memory_plan.py -q
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "training" / "cree_llm" / "cells"


def _source(name: str) -> str:
    text = (CELLS / name).read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("!", "%")))


def _stub_peft() -> None:
    """Stand in for peft only when it is not installed.

    A stub in sys.modules would be picked up by every later test in the same
    session, including ones that train a real adapter.
    """
    try:
        import peft  # noqa: F401

        return
    except ImportError:
        pass

    import importlib.machinery

    peft = types.ModuleType("peft")
    # transformers probes `importlib.util.find_spec("peft")`, which raises when a
    # module in sys.modules carries no spec. A stub needs one to look importable.
    peft.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    peft.__version__ = "0.17.0"

    class LoraConfig:  # noqa: D401 - a stand-in with the same attributes
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    peft.LoraConfig = LoraConfig
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model, **kwargs: model
    # transformers.trainer_utils imports both names, so both have to exist.
    peft.PeftModel = type("PeftModel", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    peft.PeftMixedModel = type("PeftMixedModel", (), {})
    sys.modules["peft"] = peft


@pytest.fixture()
def nb(tmp_path, monkeypatch):
    """The namespace cells 01 and 06 produce, with the work dir under tmp."""
    monkeypatch.setenv("CREE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("CREE_SMOKE", "1")
    monkeypatch.delenv("CREE_PROFILE", raising=False)
    _stub_peft()

    ns: dict = {"__name__": "__main__"}
    exec(compile(_source("01_setup.py"), "01_setup.py", "exec"), ns)

    proc = ns["PROC"]
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "new_tokens.json").write_text(
        json.dumps({"tokens": ["nêhiyaw", "maskwa"], "specials": [], "new_token_start": 32, "old_size": 32}),
        encoding="utf-8",
    )
    exec(compile(_source("06_model.py"), "06_model.py", "exec"), ns)
    return ns


def _tiny_model(ns, vocab: int = 32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


class _Tok:
    def __init__(self, size: int):
        self.size = size
        self.pad_token = "<pad>"
        self.pad_token_id = 0

    def __len__(self) -> int:
        return self.size


def test_precision_follows_the_card(nb):
    plan = nb["plan_weights"]
    # The T4 that ran out of memory: 7.6B at 16 bits is 15.2 GB of weights.
    assert plan(14.56, 7.6, False) is True
    assert plan(15.0, 7.6, False) is True
    # An A100 40 GB holds the same model unquantised, which trains better.
    assert plan(40.0, 7.6, False) is False
    assert plan(80.0, 14.7, False) is False
    assert plan(80.0, 32.8, False) is True
    # Smoke runs and CPU never quantise.
    assert plan(15.0, 7.6, True) is False
    assert plan(0.0, 7.6, False) is False
    # A 0.5B smoke model is small everywhere.
    assert plan(15.0, 0.5, False) is False


def _save_graft(nb, rows):
    """A graft file the way the current cell writes it: rows plus their key."""
    torch.save({"key": nb["_GRAFT_KEY"], "start": 32, "rows": rows}, nb["GRAFTED"])


def test_graft_survives_a_reload(nb):
    model = _tiny_model(nb)
    tok = _Tok(34)
    rows = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 16) / 100
    _save_graft(nb, rows)

    nb["graft_rows"](model, tok)
    assert model.get_input_embeddings().weight.shape[0] == 34
    got = model.get_input_embeddings().weight[32:34].detach()
    assert torch.allclose(got, rows.to(got.dtype))
    assert torch.allclose(model.get_output_embeddings().weight[32:34].detach(), rows.to(got.dtype))


def test_graft_is_what_makes_the_resize_meaningful(nb):
    """A second load must land on the saved rows, not on random ones."""
    model = _tiny_model(nb)
    tok = _Tok(34)
    rows = torch.full((2, 16), 0.25)
    _save_graft(nb, rows)

    nb["graft_rows"](model, tok)
    first = model.get_input_embeddings().weight[32:34].detach().clone()

    # A fresh model, resized without the graft, is random. This is the state the
    # notebook would be in on every reload if the rows were not saved.
    fresh = _tiny_model(nb)
    fresh.resize_token_embeddings(34)
    unset = fresh.get_input_embeddings().weight[32:34].detach()
    assert not torch.allclose(unset, rows.to(unset.dtype))

    nb["graft_rows"](fresh, tok)
    assert torch.allclose(fresh.get_input_embeddings().weight[32:34].detach(), first)


def test_a_graft_without_a_key_is_rebuilt_not_trusted(nb):
    """A file from an older build has no key. It may belong to another vocabulary."""
    model = _tiny_model(nb)
    torch.save({"start": 32, "rows": torch.zeros(2, 16)}, nb["GRAFTED"])

    class Decomposer:
        unk_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            return {"nêhiyaw": [3, 4], "maskwa": [5, 6]}[text]

    class Factory:
        @staticmethod
        def from_pretrained(path):
            return Decomposer()

    nb["AutoTokenizer"] = Factory
    nb["graft_rows"](model, _Tok(34))

    saved = torch.load(nb["GRAFTED"], map_location="cpu", weights_only=True)
    assert saved.get("key") == nb["_GRAFT_KEY"]
    assert not torch.allclose(saved["rows"], torch.zeros(2, 16))
    emb = model.get_input_embeddings().weight
    expected = torch.stack([emb[[3, 4]].mean(0), emb[[5, 6]].mean(0)])
    assert torch.allclose(saved["rows"], expected, atol=1e-6)


def test_first_load_builds_the_graft_from_pieces(nb):
    nb["GRAFTED"].unlink(missing_ok=True)
    model = _tiny_model(nb)
    emb = model.get_input_embeddings().weight.detach().clone()

    class Decomposer:
        unk_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens=False):
            return {"nêhiyaw": [3, 4], "maskwa": [5, 6]}[text]

    class Factory:
        @staticmethod
        def from_pretrained(path):
            return Decomposer()

    nb["AutoTokenizer"] = Factory
    nb["graft_rows"](model, _Tok(34))

    assert nb["GRAFTED"].exists()
    saved = torch.load(nb["GRAFTED"], map_location="cpu", weights_only=True)
    expected = emb[[3, 4]].mean(dim=0).unsqueeze(0)
    assert torch.allclose(saved["rows"][0], expected.squeeze(0), atol=1e-6)
    assert int(saved["start"]) == 32


def test_freeze_takes_the_vocabulary_out_of_the_optimizer(nb):
    model = _tiny_model(nb)
    model.get_input_embeddings().weight.requires_grad_(True)
    model.get_output_embeddings().weight.requires_grad_(True)

    nb["freeze_grafted"](model)

    assert not model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    # The transformer blocks are still trainable underneath the adapter.
    assert any(p.requires_grad for p in model.model.layers[0].parameters())


def test_stale_checkpoints_are_moved_not_deleted(nb):
    ckpt = nb["CKPT"]
    (ckpt / "cpt").mkdir(parents=True, exist_ok=True)
    (ckpt / "cpt" / "adapter_config.json").write_text("{}", encoding="utf-8")
    nb["jdump"](nb["PLAN_FILE"], {"base": "old/model", "four_bit": True, "trainable_embeddings": True, "lora_r": 8})

    nb["reset_stale_checkpoints"]()
    moved = list(ckpt.glob("cpt.stale-*"))
    assert len(moved) == 1 and (moved[0] / "adapter_config.json").exists()
    plan = json.loads(nb["PLAN_FILE"].read_text(encoding="utf-8"))
    assert plan["trainable_embeddings"] is False

    # Running it again with the same plan leaves everything alone.
    nb["reset_stale_checkpoints"]()
    assert len(list(ckpt.glob("cpt.stale-*"))) == 1

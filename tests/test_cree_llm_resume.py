"""Checks for the Cree notebook's crash-safe resume.

Colab ends a runtime without warning, and the cost of getting this wrong is
hours, so the behaviour is tested rather than assumed: a stage must not be
skipped without its artifact, a torn checkpoint must not be resumed from, and a
re-run of a training stage must continue at the step it reached.

These run the notebook's own cell sources, not a copy of them.

    python -m pytest tests/test_cree_llm_resume.py -q
"""

from __future__ import annotations

import json
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

    The resume test below trains a real adapter, so a stub here would silently
    turn the model into a plain one and the checkpoint would hold the wrong
    files. On a machine without peft that test skips instead.
    """
    try:
        import peft  # noqa: F401

        return
    except ImportError:
        pass

    import importlib.machinery

    peft = types.ModuleType("peft")
    peft.__spec__ = importlib.machinery.ModuleSpec("peft", None)
    peft.__version__ = "0.17.0"

    class LoraConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    peft.LoraConfig = LoraConfig
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model, **kwargs: model
    peft.PeftModel = type("PeftModel", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    peft.PeftMixedModel = type("PeftMixedModel", (), {})
    sys.modules["peft"] = peft


@pytest.fixture()
def nb(tmp_path, monkeypatch):
    monkeypatch.setenv("CREE_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("CREE_SMOKE", "1")
    monkeypatch.delenv("CREE_PROFILE", raising=False)
    monkeypatch.delenv("CREE_FORCE", raising=False)
    _stub_peft()

    ns: dict = {"__name__": "__main__"}
    exec(compile(_source("01_setup.py"), "01_setup.py", "exec"), ns)
    proc = ns["PROC"]
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "new_tokens.json").write_text(
        json.dumps({"tokens": ["nêhiyaw"], "specials": [], "new_token_start": 32, "old_size": 32}),
        encoding="utf-8",
    )
    exec(compile(_source("06_model.py"), "06_model.py", "exec"), ns)
    return ns


def _write_adapter(folder: Path, *, weights: bytes = b"") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(weights)
    return folder


def _valid_safetensors(path: Path) -> Path:
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(2, dtype=torch.float32)}, str(path))
    return path


def test_a_stage_is_not_skipped_without_its_artifact(nb):
    assert nb["stage_done"]("cpt", "key1", quiet=True) is False

    nb["mark_stage"]("cpt", "key1", blocks=3)
    # The marker alone is not enough: the adapter it promises must exist.
    assert nb["stage_done"]("cpt", "key1", requires=nb["CKPT"] / "cpt" / "adapter_config.json", quiet=True) is False

    _write_adapter(nb["CKPT"] / "cpt", weights=b"x")
    assert nb["stage_done"]("cpt", "key1", requires=nb["CKPT"] / "cpt" / "adapter_config.json") is True


def test_changed_inputs_invalidate_a_stage(nb):
    nb["mark_stage"]("sft", "old")
    assert nb["stage_done"]("sft", "old", quiet=True) is True
    # A different corpus, model or rank produces a different fingerprint, and a
    # result computed from other inputs must not be reused.
    assert nb["stage_done"]("sft", "new", quiet=True) is False


def test_force_overrides_a_finished_stage(nb, monkeypatch):
    monkeypatch.setenv("CREE_FORCE", "cpt")
    # CREE_FORCE is read once at setup, so the namespace is rebuilt with it set.
    monkeypatch.setenv("CREE_WORK", str(nb["ROOT"]))
    ns = dict(nb)
    ns.pop("FORCE")
    exec(compile(_source("01_setup.py"), "01_setup.py", "exec"), ns)
    ns["mark_stage"]("cpt", "k")
    assert ns["stage_done"]("cpt", "k", quiet=True) is False


def test_ledger_survives_a_half_written_file(nb):
    nb["mark_stage"]("clean", "k")
    nb["STATE_FILE"].write_text('{"stages": {"clean": ', encoding="utf-8")
    # A torn ledger reads as empty rather than crashing the notebook on startup.
    assert nb["load_run_state"]()["stages"] == {}
    assert nb["stage_done"]("clean", "k", quiet=True) is False


def test_metrics_are_not_duplicated_by_a_resume(nb):
    nb["append_metrics"]("CPT", [{"step": 5, "loss": 1.0}, {"step": 10, "loss": 0.5}])
    # A resumed trainer logs the step it resumed from again.
    nb["append_metrics"]("CPT", [{"step": 10, "loss": 0.5}, {"step": 15, "loss": 0.4}])
    rows = nb["read_metrics"]("CPT")
    assert [r["step"] for r in rows] == [5, 10, 15]


def test_a_torn_checkpoint_is_skipped_for_the_previous_one(nb):
    out = nb["CKPT"] / "cpt"
    good = _write_adapter(out / "checkpoint-10")
    _valid_safetensors(good / "adapter_model.safetensors")
    (good / "trainer_state.json").write_text('{"global_step": 10}', encoding="utf-8")

    torn = _write_adapter(out / "checkpoint-20", weights=b"\xff" * 64)
    (torn / "trainer_state.json").write_text('{"global_step": 20}', encoding="utf-8")

    # The newest folder is unreadable, so the previous one is the resume point.
    assert nb["pick_resume_checkpoint"](out) == good


def test_a_checkpoint_cut_off_before_its_state_is_skipped(nb):
    out = nb["CKPT"] / "sft"
    older = _write_adapter(out / "checkpoint-5")
    _valid_safetensors(older / "adapter_model.safetensors")
    (older / "trainer_state.json").write_text('{"global_step": 5}', encoding="utf-8")

    newer = _write_adapter(out / "checkpoint-9")
    _valid_safetensors(newer / "adapter_model.safetensors")
    # No trainer_state.json: the process died between saving the adapter and
    # the state, so this folder cannot restore a step count.

    assert nb["pick_resume_checkpoint"](out) == older


def test_forcing_a_stage_archives_its_checkpoints(nb, monkeypatch):
    folder = nb["CKPT"] / "cpt"
    _write_adapter(folder)

    nb["fresh_if_forced"]("cpt")
    assert folder.exists(), "nothing is moved unless the stage was forced"

    monkeypatch.setitem(nb, "FORCE", {"cpt"})
    nb["fresh_if_forced"]("cpt")
    assert not folder.exists()
    archived = list(nb["CKPT"].glob("cpt.forced-*"))
    assert len(archived) == 1 and (archived[0] / "adapter_config.json").exists()


def test_an_interrupted_stage_is_not_mistaken_for_a_finished_one(nb):
    _write_adapter(nb["CKPT"] / "cpt")
    # Only checkpoint folders exist for SFT: the root adapter is written after
    # training returns, so this stage is still in progress.
    partial = _write_adapter(nb["CKPT"] / "sft" / "checkpoint-40")
    _valid_safetensors(partial / "adapter_model.safetensors")

    assert nb["best_adapter"]() == nb["CKPT"] / "cpt"
    assert nb["best_adapter"](upto="cpt") == nb["CKPT"] / "cpt"
    assert nb["best_adapter"](upto="sft") == nb["CKPT"] / "cpt"


def test_best_adapter_takes_the_newest_finished_stage(nb):
    for name in ("cpt", "sft", "dpo"):
        _write_adapter(nb["CKPT"] / name)
    assert nb["best_adapter"]() == nb["CKPT"] / "dpo"
    assert nb["best_adapter"](upto="sft") == nb["CKPT"] / "sft"
    assert nb["best_adapter"](upto="cpt") == nb["CKPT"] / "cpt"


def test_training_resumes_at_the_step_it_reached(nb, tmp_path, capsys):
    """The real thing: a Trainer run that is interrupted and continued."""
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=False,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    base = Qwen2ForCausalLM(cfg)
    model = peft.get_peft_model(base, peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ))

    blocks = [[(i * 7 + j) % 64 for j in range(8)] for i in range(64)]
    data = nb["BlockDataset"](blocks)
    out = tmp_path / "run"

    first = nb["run_training"](model, data, None, out, epochs=1, lr=1e-3,
                               micro=1, accum=1, tag="RESUME")
    assert max(r["step"] for r in first) == 64

    # Simulate the disconnect: the stage root was never written, so the stage is
    # unfinished, and only the checkpoints survive.
    (out / "adapter_config.json").unlink(missing_ok=True)
    (out / "adapter_model.safetensors").unlink(missing_ok=True)
    resume_point = nb["pick_resume_checkpoint"](out)
    assert resume_point is not None and resume_point.name == "checkpoint-64"

    # A fresh kernel rebuilds the adapter from scratch, exactly like cell 07.
    torch.manual_seed(1)
    fresh = peft.get_peft_model(Qwen2ForCausalLM(cfg), peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ))
    capsys.readouterr()
    second = nb["run_training"](fresh, data, None, out, epochs=2, lr=1e-3,
                                micro=1, accum=1, tag="RESUME")

    # The cell said so, and the schedule ran to the two-epoch total rather than
    # restarting: 64 steps were already done, 64 more were trained.
    assert "resuming from checkpoint-64" in capsys.readouterr().out
    assert max(r["step"] for r in second) == 128

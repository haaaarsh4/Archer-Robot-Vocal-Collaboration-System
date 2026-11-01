# ─────────────────────────────────────────────────────────────────────────────
# Configuration. One place for every knob; every later cell reads from here.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import unicodedata
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

IN_COLAB = "google.colab" in sys.modules

# Colab ships an older stack; install the small number of packages this
# notebook actually needs. Torch is left exactly as Colab provides it.
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.46,<6", "peft>=0.13", "datasets>=3", "accelerate>=1.0",
         "bitsandbytes>=0.44", "sentencepiece>=0.2", "sacrebleu>=2.4",
         "matplotlib>=3.8", "pandas>=2.0", "psutil>=5.9"],
        check=True,
    )

import torch
import torch.nn.functional as F

SMOKE = os.environ.get("CREE_SMOKE", "0") == "1"
PROFILE = os.environ.get("CREE_PROFILE", "smoke" if SMOKE else "auto")


def _gpu_line() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"{name} ({gb:.0f} GB)"


_PRESETS = {
    # A tiny model and a handful of steps: proves every cell runs, changes nothing.
    "smoke": dict(base="Qwen/Qwen2.5-0.5B-Instruct", seq=384, lora_r=8,
                  cpt_epochs=1, sft_epochs=1, dpo_epochs=1, eval_n=2,
                  max_cpt_packs=6, max_sft=48),
    # Free / pay-as-you-go T4 (16 GB). Shorter context, two passes.
    "t4": dict(base="Qwen/Qwen2.5-7B-Instruct", seq=1024, lora_r=32,
               cpt_epochs=2, sft_epochs=2, dpo_epochs=1, eval_n=120,
               max_cpt_packs=100_000, max_sft=100_000),
    # A100 40 GB: the intended run.
    "a100": dict(base="Qwen/Qwen2.5-7B-Instruct", seq=2048, lora_r=64,
                 cpt_epochs=3, sft_epochs=2, dpo_epochs=1, eval_n=250,
                 max_cpt_packs=100_000, max_sft=100_000),
    # A100 80 GB / H100: the 14B model. Set CREE_PROFILE=big.
    "big": dict(base="Qwen/Qwen2.5-14B-Instruct", seq=2048, lora_r=64,
                cpt_epochs=3, sft_epochs=2, dpo_epochs=1, eval_n=250,
                max_cpt_packs=100_000, max_sft=100_000),
    # The largest preset that fits one 80 GB card with QLoRA. Merging and
    # quantising a 32B needs a separate high-memory step; the notebook says so
    # rather than pretending it fits everywhere.
    "huge": dict(base="Qwen/Qwen2.5-32B-Instruct", seq=2048, lora_r=64,
                 cpt_epochs=3, sft_epochs=2, dpo_epochs=1, eval_n=250,
                 max_cpt_packs=100_000, max_sft=100_000),
}


def choose_profile() -> str:
    if PROFILE != "auto":
        return PROFILE
    if not torch.cuda.is_available():
        return "smoke"
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return "a100" if gb >= 38 else "t4"


ACTIVE_PROFILE = choose_profile()
P = _PRESETS[ACTIVE_PROFILE]


@dataclass
class Config:
    base_model: str = P["base"]
    seq_len: int = P["seq"]
    lora_r: int = P["lora_r"]
    lora_alpha: int = P["lora_r"] * 2
    lora_dropout: float = 0.05
    cpt_epochs: float = P["cpt_epochs"]
    sft_epochs: float = P["sft_epochs"]
    dpo_epochs: float = P["dpo_epochs"]
    eval_n: int = P["eval_n"]
    max_cpt_packs: int = P["max_cpt_packs"]
    max_sft: int = P["max_sft"]
    cpt_lr: float = 2e-4
    sft_lr: float = 1.5e-4
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.1
    micro_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    new_token_budget: int = 5200
    spm_vocab: int = 6000
    seed: int = 17
    # Licence policy: NoDerivatives material is never loaded. NonCommercial
    # ShareAlike material is loaded because this model is released under the
    # same licence; set to False for a commercially licensable build.
    include_noncommercial: bool = True
    dpo_enabled: bool = True
    external_baseline: str = os.environ.get("CREE_EXTERNAL_MODEL", "")  # ollama tag


CFG = Config()

# The exported artifact is named for the model actually trained, not for the
# default. A 14B run must not produce a file called archer-cree-7b.
_BASE_LOWER = CFG.base_model.lower()
MODEL_TAG = next(
    (tag for tag in ("32b", "14b", "7b", "3b", "1.5b", "0.5b") if tag in _BASE_LOWER),
    "custom",
)
MODEL_STEM = f"archer-cree-{MODEL_TAG}"

# ── where everything lives ────────────────────────────────────────────────────
if IN_COLAB and os.environ.get("CREE_USE_DRIVE", "1") == "1":
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/archer_cree_llm")
    except Exception as exc:  # drive auth declined: still run, just not persistent
        print(f"[warning] Drive is not mounted ({exc}).")
        print("[warning] Everything goes to /content, which is deleted with the runtime, so a")
        print("[warning] disconnect will lose the run. Re-run this cell and allow the Drive prompt.")
        ROOT = Path("/content/cree_llm")
elif IN_COLAB:
    ROOT = Path("/content/cree_llm")
else:
    ROOT = Path(os.environ.get("CREE_WORK", Path.cwd() / "cree_llm_work")).resolve()

RAW = ROOT / "raw"
PROC = ROOT / "processed"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "out"
PLOTS = OUT / "plots"
for d in (RAW, PROC, CKPT, OUT, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# ── where the Hugging Face cache lives ────────────────────────────────────────
# Model weights are never cached on Drive. Drive's FUSE mount does not reliably
# finish writing multi-gigabyte binary files, and a shard that reads back with a
# damaged header is exactly the `SafetensorError: header too large` that stops
# training before the first step. Weights are a cache, they can always be fetched
# again, so they live on the runtime's local disk. Checkpoints and exports, which
# cannot be recreated, still go to ROOT on Drive.
HF_CACHE = Path(
    os.environ.get("CREE_HF_HOME")
    or (Path("/content/hf_cache") if IN_COLAB else ROOT / "hf")
)
if str(HF_CACHE).startswith("/content/drive"):
    print("[note] the Hugging Face cache was pointed at Drive; using /content/hf_cache instead, "
          "because the Drive mount corrupts large model shards")
    HF_CACHE = Path("/content/hf_cache")
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Read timeouts, not transfer limits. A keep-alive socket that the network stack
# has already dropped otherwise parks a download until the process is killed,
# which looks exactly like a hung corpus stage. Sixty seconds without a byte is
# already a broken connection, so failing and retrying is the faster path.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16 else (torch.float16 if DEVICE == "cuda" else torch.float32)

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if DEVICE == "cuda" else 0.0

# Parameter counts of the bases this notebook knows about, in billions. Used only
# to plan memory, never to describe the model.
_BASE_PARAMS_B = {"0.5b": 0.5, "1.5b": 1.5, "3b": 3.0, "7b": 7.6, "14b": 14.7, "32b": 32.8}
BASE_PARAMS_B = next((v for k, v in _BASE_PARAMS_B.items() if k in _BASE_LOWER), 7.6)


def plan_weights(gpu_gb: float, params_b: float, smoke: bool) -> bool:
    """True when the base weights must be quantised to 4 bits.

    16-bit weights cost 2 bytes per parameter, before any gradient, optimizer
    state or activation. A 7.6B base is about 15 GB of weights, which on a
    free Colab T4 (14.56 GiB usable) leaves nothing at all: the run dies on the
    first cross-entropy with a 608 MiB allocation it cannot make. Quantised to
    nf4 the same weights are about 4.4 GB.

    The 0.55 factor is the share of the card that weights are allowed to take,
    leaving the rest for activations, the logits and the CUDA context.
    """
    if gpu_gb <= 0 or smoke:
        return False
    return 2 * params_b > gpu_gb * 0.55


USE_4BIT = plan_weights(GPU_GB, BASE_PARAMS_B, SMOKE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def jload(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path | str, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def download(url: str, dest: Path, timeout: int = 120) -> Path:
    """Fetch once, reuse forever. A half-written file is never trusted."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "archer-cree-llm/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    return dest


# ── crash-safe run state ─────────────────────────────────────────────────────
# Colab ends a runtime without asking: a closed tab, an idle timeout, a GPU
# reset. So every step that takes longer than a minute leaves two traces on
# Drive. A stage marker records that the step finished, and periodic
# checkpoints record how far an unfinished step got. Re-running the notebook
# after a disconnect walks past the finished work and continues the interrupted
# stage from its last checkpoint instead of beginning again.
#
# A marker carries a fingerprint of the inputs that produced it. Change the
# corpus, the model, the context length or the LoRA rank and the affected stage
# runs again on its own: a result computed from different inputs is not a
# result for this run.

STATE_FILE = ROOT / "run_state.json"
FORCE = {name.strip().lower() for name in os.environ.get("CREE_FORCE", "").split(",") if name.strip()}
SAVE_MINUTES = float(os.environ.get("CREE_SAVE_MINUTES", "10"))
STAGE_ORDER = ("corpus", "clean", "tokenizer", "cpt", "sft", "dpo", "eval", "llmtests")


def fingerprint(**parts) -> str:
    """A short, stable hash of the inputs a stage's output depends on."""
    return hashlib.sha1(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:12]


def load_run_state() -> dict:
    if not STATE_FILE.exists():
        return {"stages": {}}
    try:
        state = jload(STATE_FILE)
    except (OSError, json.JSONDecodeError):
        # A half-written ledger is treated as an empty one. The next mark_stage
        # rewrites it, and the worst case is one stage running twice.
        return {"stages": {}}
    state.setdefault("stages", {})
    return state


def save_run_state(state: dict) -> None:
    """Replace the ledger atomically, so a disconnect cannot halve it."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def stage_entry(name: str) -> dict | None:
    return load_run_state()["stages"].get(name)


def stage_done(name: str, key: str | None = None, requires: Path | str | None = None,
               quiet: bool = False) -> bool:
    """True when this stage already finished with these inputs.

    `requires` is the artifact the marker promises. If it is missing, the stage
    runs again: a marker without its file would otherwise skip work and leave
    every later cell reading nothing.
    """
    if name in FORCE:
        if not quiet:
            log(f"{name}: forced to run again (CREE_FORCE={name})")
        return False
    entry = stage_entry(name)
    if not entry:
        return False
    if requires is not None and not Path(requires).exists():
        log(f"{name}: marked finished but {Path(requires).name} is missing, so it runs again")
        return False
    if key is not None and entry.get("key") != key:
        log(f"{name}: the inputs changed since it last ran, so it runs again")
        return False
    if not quiet:
        log(f"{name}: already finished {entry.get('at', '')}, skipping")
    return True


def mark_stage(name: str, key: str | None = None, **extra) -> None:
    state = load_run_state()
    state["stages"][name] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "key": key, **extra,
    }
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_run_state(state)


def clear_stage(*names: str) -> None:
    state = load_run_state()
    for name in names:
        state["stages"].pop(name, None)
    save_run_state(state)


def metrics_path(tag: str) -> Path:
    folder = OUT / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{tag.lower()}.jsonl"


def append_metrics(tag: str, rows: list[dict]) -> None:
    """Append to the loss log on Drive the moment a row exists."""
    if not rows:
        return
    with metrics_path(tag).open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metrics(tag: str) -> list[dict]:
    """Every logged row for this stage, including the ones from earlier sessions.

    A resumed run logs its last step again, so rows are de-duplicated on step.
    Without that, the loss curve after three sessions would show a spike at
    every reconnect.
    """
    path = metrics_path(tag)
    if not path.exists():
        return []
    out, seen = [], set()
    for row in read_jsonl(path):
        key = (row.get("step"), row.get("event"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def checkpoints(folder: Path | str) -> list[Path]:
    """Trainer checkpoints in a stage folder, oldest first."""
    root = Path(folder)
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    return sorted(found, key=lambda p: int(p.name.split("-")[-1]))


def stage_overview() -> None:
    """Where this run stands, printed at setup so a reconnect is not a mystery."""
    stages = load_run_state()["stages"]
    if not stages:
        print("fresh run: no stage has finished yet")
        return
    print(f"{'stage':10s} {'state':8s} {'finished':19s} detail")
    for name in STAGE_ORDER:
        entry = stages.get(name)
        if entry:
            detail = ", ".join(f"{k}={v}" for k, v in entry.items() if k not in {"at", "key"})
            print(f"{name:10s} {'done':8s} {entry.get('at', ''):19s} {detail}")
        else:
            found = checkpoints(CKPT / name)
            detail = f"interrupted, last checkpoint {found[-1].name}" if found else ""
            print(f"{name:10s} {'pending':8s} {'':19s} {detail}")


def plot_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#f7f4ec",
        "axes.facecolor": "#fffdf7",
        "axes.edgecolor": "#c9c0ad",
        "axes.labelcolor": "#2b2620",
        "text.color": "#2b2620",
        "xtick.color": "#6b6254",
        "ytick.color": "#6b6254",
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "#e4ddcf",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


plot_style()

# A 2048-token window on a 15 GB card costs four times the activation memory of
# 1024 and buys nothing here: the corpus is sentences, not long documents.
if DEVICE == "cuda" and 0 < GPU_GB < 24 and CFG.seq_len > 1024:
    log(f"context {CFG.seq_len} does not fit {GPU_GB:.0f} GB; using 1024")
    CFG.seq_len = 1024

log(f"profile={ACTIVE_PROFILE}  device={_gpu_line()}  autocast={'bf16' if BF16 else 'fp16'}")
log(f"base={CFG.base_model}  seq={CFG.seq_len}  LoRA r={CFG.lora_r}  work={ROOT}")
log(f"weights={'4-bit nf4' if USE_4BIT else str(DTYPE).split('.')[-1]}"
    f" ({BASE_PARAMS_B:.1f}B parameters, {GPU_GB:.1f} GB card)" if DEVICE == "cuda"
    else f"weights={str(DTYPE).split('.')[-1]} (CPU)")
log(f"hf cache={HF_CACHE}  free={shutil.disk_usage(str(HF_CACHE)).free / 1024 ** 3:.1f} GB")
log(f"work dir={ROOT}  free={shutil.disk_usage(str(ROOT)).free / 1024 ** 3:.1f} GB")
log(f"checkpoints: every {SAVE_MINUTES:.0f} min to {CKPT}")
if FORCE:
    log(f"forcing these stages to run again: {', '.join(sorted(FORCE))}")
stage_overview()
if SMOKE:
    log("SMOKE mode: tiny model, tiny schedules. Scores are meaningless; it only proves the code runs.")

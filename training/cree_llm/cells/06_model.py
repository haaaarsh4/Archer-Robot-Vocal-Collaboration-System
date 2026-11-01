# ─────────────────────────────────────────────────────────────────────────────
# The model: quantised base, resized vocabulary, statistically initialised new
# embeddings, and LoRA adapters.
#
# New tokens start as the average of the base model's embeddings for the pieces
# it would have used to spell the same string. That is not decoration: a token
# initialised at random has to be learned from scratch, and with a corpus this
# size it never fully is. Starting from the pieces it replaces means the model
# already reads the word, and training only has to refine it.
# ─────────────────────────────────────────────────────────────────────────────

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TARGET_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "W_pack", "c_attn")
_TOKEN_INFO = jload(PROC / "new_tokens.json")
NEW_TOKENS = _TOKEN_INFO["tokens"]
NEW_TOKEN_START = _TOKEN_INFO["new_token_start"]


# ─────────────────────────────────────────────────────────────────────────────
# Weights that are actually readable.
#
# `SafetensorError: Error while deserializing header: header too large` is not a
# transformers bug and not a bug in this notebook. It means the first eight bytes
# of a shard do not decode to a valid header length, which is what a
# multi-gigabyte file looks like when the filesystem it was written to never
# finished syncing it. In Colab that filesystem is almost always the Google
# Drive mount, which is why the weights are cached on local disk instead.
#
# So every shard is checked here, before transformers ever mmaps it, and a
# damaged shard is fetched again on its own rather than re-downloading 15 GB.
# ─────────────────────────────────────────────────────────────────────────────
SAFETENSORS_HEADER_LIMIT = 100_000_000  # the ceiling the safetensors format itself enforces
_WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "tokenizer*"]
_SKIP_PATTERNS = ["*.bin", "*.h5", "*.msgpack", "*.gguf", "*.onnx", "*.onnx_data", "*.pth", "*.pt"]


def free_gb(path: Path | str) -> float:
    """Free space on the filesystem that will hold `path`, creating nothing."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free / 1024 ** 3
    except OSError:
        return float("nan")


def safetensors_problem(path: Path) -> str | None:
    """Why this shard cannot be read, or None when it is intact.

    The check is the same one the loader performs, done in plain file reads so it
    cannot itself fail with the error it is trying to explain: an 8-byte
    little-endian header length, a JSON header of that many bytes, and tensor
    offsets that fit inside the file.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"the file cannot be opened ({exc})"
    if size < 9:
        return f"the file is only {size} bytes"
    try:
        with path.open("rb") as fh:
            header_len = int.from_bytes(fh.read(8), "little")
            if not 0 < header_len <= SAFETENSORS_HEADER_LIMIT:
                return (f"its first 8 bytes decode to a {header_len:,}-byte header, outside the valid "
                        f"1 to {SAFETENSORS_HEADER_LIMIT:,} range")
            if 8 + header_len > size:
                return f"its header claims {header_len:,} bytes but the whole file is {size:,}"
            header = json.loads(fh.read(header_len).decode("utf-8"))
            if not isinstance(header, dict):
                return "its header is not a JSON object"
            end = 8 + header_len
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                offsets = (info or {}).get("data_offsets")
                if not (isinstance(offsets, list) and len(offsets) == 2):
                    return f"tensor {name!r} has no usable data_offsets"
                end = max(end, 8 + header_len + int(offsets[1]))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return f"its header is unreadable ({exc})"
    if end > size:
        return f"it declares {end:,} bytes of tensors but the file is only {size:,}"
    return None


def verify_weights(folder: Path, quiet: bool = False) -> list[tuple[Path, str]]:
    shards = sorted(Path(folder).glob("*.safetensors"))
    if not shards:
        raise RuntimeError(
            f"no .safetensors weights in {folder}. Point CFG.base_model at the folder that holds the shards."
        )
    bad = [(p, why) for p in shards if (why := safetensors_problem(p)) is not None]
    if not bad and not quiet:
        total = sum(p.stat().st_size for p in shards) / 1024 ** 3
        log(f"base weights verified: {len(shards)} shard(s), {total:.1f} GB in {folder}")
    return bad


def _damage_message(folder: Path, bad: list[tuple[Path, str]]) -> str:
    rows = []
    for path, why in bad:
        gb = path.stat().st_size / 1024 ** 3 if path.exists() else float("nan")
        rows.append(f"  {path.name}  {gb:.2f} GB  ({why})")
    return (
        "A checkpoint shard is damaged, so the weights cannot be loaded:\n"
        + "\n".join(rows)
        + f"\n\nFolder: {folder}\nFree space there: {free_gb(folder):.1f} GB"
        + f"\nHugging Face cache: {HF_CACHE}\n\n"
        + "Run this cell again to fetch the damaged file fresh, or delete it and re-copy it if "
        + "this is a local folder. Keep the cache on the local runtime disk: a multi-gigabyte shard "
        + "written through a Google Drive mount can come back half-synced, and that is what "
        + "produces `SafetensorError: header too large`."
    )


def ensure_base_weights(repair: bool = True, force_check: bool = False) -> str:
    """A base-model folder whose every shard passed the header check."""
    local = Path(str(CFG.base_model)).expanduser()
    from_hub = not local.is_dir()
    if from_hub:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        try:
            siblings = HfApi().model_info(CFG.base_model, files_metadata=True).siblings or []
            need_bytes = sum((s.size or 0) for s in siblings if s.rfilename.endswith((".safetensors", ".json")))
            need_gb = need_bytes / 1024 ** 3
            room = free_gb(HF_CACHE)
            log(f"base model {CFG.base_model}: about {need_gb:.1f} GB of weights, "
                f"{room:.1f} GB free at {HF_CACHE}")
            if need_gb and not math.isnan(room) and need_gb > room * 0.95:
                raise RuntimeError(
                    f"Downloading {CFG.base_model} needs about {need_gb:.1f} GB and only "
                    f"{room:.1f} GB is free at {HF_CACHE}. Free space, or set "
                    f"CREE_HF_HOME to a larger disk. A download that runs out of room is one of "
                    f"the ways a shard ends up damaged."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            log(f"could not read the file sizes ahead of time ({exc}); continuing")

        folder = Path(snapshot_download(
            CFG.base_model,
            allow_patterns=_WEIGHT_PATTERNS,
            ignore_patterns=_SKIP_PATTERNS,
        ))
    else:
        folder = local

    bad = verify_weights(folder, quiet=not force_check)
    if not bad:
        return str(folder)
    for path, why in bad:
        log(f"damaged shard {path.name}: {why}")
    if not repair or not from_hub:
        raise RuntimeError(_damage_message(folder, bad))

    for path, _ in bad:
        log(f"re-downloading {path.name} into {HF_CACHE}")
        hf_hub_download(CFG.base_model, path.relative_to(folder).as_posix(), force_download=True)
    bad = verify_weights(folder, quiet=True)
    if bad:
        for path, why in bad:
            log(f"still damaged: {path.name}: {why}")
        raise RuntimeError(_damage_message(folder, bad))
    log("damaged shards replaced; base weights verified")
    return str(folder)


def find_target_modules(model) -> list[str]:
    names = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in TARGET_HINTS and hasattr(module, "weight"):
            names.add(leaf)
    if not names:
        raise RuntimeError("no LoRA target modules found; check the base model architecture")
    return sorted(names)


def init_new_embeddings(model, decomposer, tokens: list[str], start: int) -> None:
    """Give each new token the average embedding of the pieces it replaces."""
    emb = model.get_input_embeddings()
    weight = emb.weight
    device = weight.device
    unk = decomposer.unk_token_id or 0
    with torch.no_grad():
        rows = []
        for token in tokens:
            ids = decomposer.encode(token, add_special_tokens=False) or [unk]
            rows.append(weight[torch.tensor(ids, device=device)].mean(dim=0))
        weight[start: start + len(rows)] = torch.stack(rows).to(weight.dtype)

    out = model.get_output_embeddings()
    if out is not None and not model.config.tie_word_embeddings and out.weight.shape[0] >= start + len(tokens):
        with torch.no_grad():
            out.weight[start: start + len(tokens)] = weight[start: start + len(tokens)].to(out.weight.dtype)


def _dtype_kwargs(dtype=None) -> dict:
    """The keyword this transformers install calls the weight dtype.

    `torch_dtype` was renamed to `dtype` in transformers 4.56 and the old name
    now prints a deprecation warning on every load. Passing whichever name the
    installed version understands keeps the notebook clean on Colab, which pins
    a range rather than one version.
    """
    import transformers

    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        major, minor = (0, 0)
    return {"dtype": dtype or DTYPE} if (major, minor) >= (4, 56) else {"torch_dtype": dtype or DTYPE}


# The grafted rows live in one small file instead of inside the adapter.
# 3,949 rows of 3,584 values is 28 MB; keeping them separate is what lets the
# embedding table stay frozen, and that is the whole memory fix.
GRAFTED = PROC / "grafted_embeddings.pt"


def graft_rows(model, tok) -> None:
    """Resize the vocabulary and restore the grafted rows.

    `resize_token_embeddings` fills new rows with random values, so resizing
    alone would throw away the statistical start every time a cell reloads the
    model. The graft is written once, on the first load, and copied back on every
    later one, which also means the export cell sees exactly the embeddings that
    training saw.
    """
    n = len(tok)
    if GRAFTED.exists():
        saved = torch.load(GRAFTED, map_location="cpu", weights_only=True)
    elif n > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(n)
        decomposer = AutoTokenizer.from_pretrained(PROC / "base_tokenizer")
        init_new_embeddings(model, decomposer, NEW_TOKENS, NEW_TOKEN_START)
        rows = model.get_input_embeddings().weight[NEW_TOKEN_START: NEW_TOKEN_START + len(NEW_TOKENS)]
        rows = rows.detach().cpu().clone()
        saved = {"start": NEW_TOKEN_START, "rows": rows}
        torch.save(saved, GRAFTED)
        log(f"grafted {len(NEW_TOKENS)} new rows from base subword pieces -> {GRAFTED.name}")
    else:
        return

    # Applied whenever the file exists, even if the sizes already agree: a model
    # that was resized elsewhere carries random rows, and matching a length is
    # not the same as holding the graft.
    start = int(saved["start"])
    rows = saved["rows"]
    need = start + rows.shape[0]
    if model.get_input_embeddings().weight.shape[0] < need:
        model.resize_token_embeddings(n)
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is None or module.weight.shape[0] < need:
            continue
        with torch.no_grad():
            module.weight[start:need] = rows.to(module.weight.dtype)


def freeze_grafted(model) -> None:
    """Keep the 7B vocabulary table out of the optimizer.

    Marking `embed_tokens` and `lm_head` for saving means 1.1 billion extra
    trainable parameters. In fp32 with gradients and Adam state that is roughly
    11 GB, and on a 14.56 GiB T4 it is what ended the first run at 14.28 GiB
    allocated with a failed 608 MiB allocation at cross-entropy. The grafted
    rows already start at the average of the pieces they replace, so the adapter
    is what trains and the table stays a fixed, readable input.
    """
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            module.weight.requires_grad_(False)


PLAN_FILE = CKPT / "plan.json"
TRAIN_STAGES = ("cpt", "sft", "dpo")


def reset_stale_checkpoints() -> None:
    """Move adapters trained under a different plan out of the way.

    An adapter saved while the vocabulary table was trainable carries extra
    tensors and a different adapter_config. Reusing that folder for a run with a
    frozen table mixes two incompatible checkpoints, and the mismatch surfaces
    hours later. Renaming keeps the old run instead of deleting it.
    """
    plan = {
        "base": CFG.base_model,
        "four_bit": USE_4BIT,
        "trainable_embeddings": False,
        "lora_r": CFG.lora_r,
    }
    previous = jload(PLAN_FILE) if PLAN_FILE.exists() else None
    if previous and previous != plan:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in TRAIN_STAGES:
            folder = CKPT / name
            if folder.exists():
                folder.rename(CKPT / f"{name}.stale-{stamp}")
        # The adapters are gone, so their finished markers are lies. Clearing
        # them is what makes the next run retrain instead of skipping to an
        # adapter that no longer exists.
        clear_stage(*TRAIN_STAGES)
        log(f"previous adapters used a different plan {previous}; moved them to *.stale-{stamp}")
    jdump(PLAN_FILE, plan)


def load_model(adapter: Path | None = None, trainable: bool = True):
    tok = AutoTokenizer.from_pretrained(PROC / "tokenizer")
    kwargs: dict = {**_dtype_kwargs(), "low_cpu_mem_usage": True}
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
        kwargs["device_map"] = {"": 0}
    elif DEVICE == "cuda":
        kwargs["device_map"] = {"": 0}
    # No device_map on CPU: a single-device map buys nothing there and would
    # make the CPU smoke run depend on `accelerate` being installed.

    model = AutoModelForCausalLM.from_pretrained(ensure_base_weights(), **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    graft_rows(model, tok)

    if trainable and USE_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif trainable and DEVICE == "cuda":
        # bf16 on a card large enough to skip quantisation still cannot hold the
        # activations of a 2048-token sequence without checkpointing, and an
        # explicit use_reentrant silences the warning torch 2.9 turns into an
        # error.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if trainable:
        model.enable_input_require_grads()

    if adapter is not None:
        from peft import PeftModel

        # Resume an existing run: load its adapters, do not wrap them in a second
        # set of fresh ones.
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
        if trainable:
            freeze_grafted(model)
            model.print_trainable_parameters()
        log(f"loaded adapter from {adapter}")
    elif trainable:
        lora = LoraConfig(
            r=CFG.lora_r,
            lora_alpha=CFG.lora_alpha,
            lora_dropout=CFG.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_target_modules(model),
        )
        model = get_peft_model(model, lora)
        freeze_grafted(model)
        model.print_trainable_parameters()

    model.config.use_cache = False
    return model, tok


def fresh_if_forced(name: str) -> None:
    """A forced stage starts over instead of continuing its old checkpoints.

    `CREE_FORCE=cpt` is documented as "run this stage again", and resuming the
    checkpoints of the run being replaced would make it a no-op that quietly
    trained a few more steps. The old folder is renamed rather than deleted, so
    a force is still reversible.
    """
    if name not in FORCE:
        return
    folder = CKPT / name
    if folder.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder.rename(CKPT / f"{name}.forced-{stamp}")
        log(f"{name}: forced, previous folder moved to {name}.forced-{stamp}")


def adapter_ready(name: str) -> Path | None:
    """The adapter folder for a finished stage, or None.

    The root of a stage folder is written once, after training returns. Partial
    saves go into `checkpoint-N/` subfolders, so a stage that was interrupted
    half way is not mistaken for a finished one.
    """
    folder = CKPT / name
    weights = (folder / "adapter_model.safetensors").exists() or (folder / "adapter_model.bin").exists()
    if (folder / "adapter_config.json").exists() and weights:
        return folder
    return None


_LOADED_ADAPTER = ""


def best_adapter(upto: str | None = None) -> Path | None:
    """The newest finished adapter at or before `upto`.

    This is what keeps a fresh kernel honest. Re-running the SFT cell after a
    disconnect used to load the bare base model and quietly train instruction
    tuning without the Cree pretraining underneath it, which is worse than not
    training at all because the scores still look plausible.
    """
    names = TRAIN_STAGES if upto is None else TRAIN_STAGES[: TRAIN_STAGES.index(upto) + 1]
    for name in reversed(names):
        folder = adapter_ready(name)
        if folder is not None:
            return folder
    return None


def ensure_model(upto: str | None = None, force: bool = False):
    """`model` and `tok` in the kernel, continued from the best saved adapter."""
    global model, tok, _LOADED_ADAPTER
    target = best_adapter(upto)
    wanted = str(target) if target is not None else ""
    if not force and "model" in globals() and "tok" in globals() and _LOADED_ADAPTER == wanted:
        return model, tok
    model, tok = load_model(adapter=target)
    _LOADED_ADAPTER = wanted
    return model, tok


class CompletionOnlyCollator:
    """Masks everything except the assistant's reply from the loss.

    Training on the whole conversation teaches the model to generate the user's
    question too, which shows up later as a model that answers by asking. Only
    the reply is scored here.
    """

    def __init__(self, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len

    def _one(self, messages: list[dict]) -> dict:
        prompt = self.tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        if f_ids[: len(p_ids)] != p_ids:
            # Template edge case: fall back to scoring the whole example rather
            # than silently misaligning the mask.
            p_ids = []
        f_ids = f_ids[: self.max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[min(len(p_ids), len(f_ids)):]
        return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}

    def __call__(self, features: list[dict]) -> dict:
        rows = [self._one(f["messages"]) for f in features]
        max_len = max(len(r["input_ids"]) for r in rows)
        pad = self.tok.pad_token_id
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for r in rows:
            n = max_len - len(r["input_ids"])
            out["input_ids"].append(r["input_ids"] + [pad] * n)
            out["labels"].append(r["labels"] + [-100] * n)
            out["attention_mask"].append(r["attention_mask"] + [0] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class PackedCausalDataset(torch.utils.data.Dataset):
    """Concatenated token streams cut into full-length windows.

    Padding is waste. Packing joins documents with an EOS and slices the stream
    into exact windows, so every position in a batch carries a real token.
    """

    def __init__(self, texts: list[str], tok, seq_len: int, limit: int | None = None):
        ids: list[int] = []
        eos = tok.eos_token_id
        for text in texts:
            ids.extend(tok(text, add_special_tokens=False)["input_ids"])
            ids.append(eos)
            if limit and len(ids) > limit * seq_len:
                break
        n = len(ids) // seq_len
        self.blocks = [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict:
        block = self.blocks[i]
        return {"input_ids": block, "labels": list(block), "attention_mask": [1] * len(block)}


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict:
        block = self.blocks[i]
        return {"input_ids": block, "labels": list(block), "attention_mask": [1] * len(block)}


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return {"messages": self.rows[i]["messages"]}


def train_args(out_dir: Path, epochs: float, lr: float, micro: int, accum: int,
               warmup_steps: int = 0, save_steps: int = 0) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=CFG.weight_decay,
        max_grad_norm=1.0,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps" if save_steps else "no",
        save_steps=max(1, save_steps),
        # Two checkpoints, so a write torn by a disconnect falls back to the
        # previous one instead of losing the stage. The optimizer and scheduler
        # are saved too: without the scheduler a resumed run restarts the cosine
        # curve from step zero and jumps the learning rate back to its peak,
        # which is a bigger change to the run than the extra 160 MB per save.
        save_total_limit=2,
        save_only_model=False,
        report_to="none",
        remove_unused_columns=False,
        fp16=(DEVICE == "cuda" and not BF16),
        bf16=BF16,
        optim="paged_adamw_8bit" if (USE_4BIT and DEVICE == "cuda") else "adamw_torch",
        dataloader_num_workers=0 if DEVICE == "cpu" else 2,
        dataloader_pin_memory=True,
        seed=CFG.seed,
        data_seed=CFG.seed,
    )


class MetricsWriter(TrainerCallback):
    """Writes every logged row to Drive as it is produced.

    `trainer.state.log_history` lives in the kernel, so a disconnect deleted the
    loss curve along with the run and the report card had nothing to plot. A row
    written here survives, and a resumed run appends to the same file, so the
    curve after three sessions reads as the curve of one run.
    """

    def __init__(self, tag: str):
        self.tag = tag

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.global_step:
            return
        row = {"step": int(state.global_step), "epoch": round(float(state.epoch or 0), 3)}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        append_metrics(self.tag, [row])


def pick_resume_checkpoint(out_dir: Path) -> Path | None:
    """The newest checkpoint that is complete enough to continue from.

    A disconnect can land in the middle of a checkpoint write, and Drive happily
    returns the half file afterwards. So the newest folder is checked rather
    than trusted: it needs a trainer state and an adapter whose safetensors
    header is intact. A torn checkpoint is skipped and the previous one is used,
    which costs at most one save interval.
    """
    for folder in reversed(checkpoints(out_dir)):
        state = folder / "trainer_state.json"
        adapter = folder / "adapter_model.safetensors"
        if not state.exists():
            log(f"ignoring {folder.name}: no trainer_state.json, it was cut off mid-write")
            continue
        if not adapter.exists() and not (folder / "adapter_model.bin").exists():
            log(f"ignoring {folder.name}: no adapter weights")
            continue
        if adapter.exists():
            problem = safetensors_problem(adapter)
            if problem:
                log(f"ignoring {folder.name}: {problem}")
                continue
        return folder
    return None


def prune_checkpoints(folder: Path | str, keep: int = 2) -> None:
    for old in checkpoints(folder)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def run_training(model, dataset, collator, out_dir: Path, epochs: float, lr: float,
                 micro: int, accum: int, tag: str,
                 save_minutes: float | None = None, key: str = "") -> list[dict]:
    """Train, checkpointing to Drive and continuing from the last checkpoint.

    The save interval is derived from the schedule rather than fixed, because a
    200-step stage and a 2,000-step stage need different granularity: eight
    checkpoints per stage keeps a disconnect cheap without letting Drive writes
    dominate the run.
    """
    if len(dataset) == 0:
        raise RuntimeError(f"{tag}: empty training set")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints belong to the data and schedule that produced them. A changed
    # instruction set changes the stage fingerprint, and resuming the old
    # checkpoints would train the new examples on top of the old run without
    # anyone noticing. A mismatch moves the old folder aside instead.
    key_path = out_dir / "stage_key.json"
    same_run = bool(key) and key_path.exists() and jload(key_path).get("key") == key
    if key and not same_run:
        if any(out_dir.glob("checkpoint-*")) or (out_dir / "adapter_config.json").exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            stale = out_dir.with_name(f"{out_dir.name}.stale-{stamp}")
            out_dir.rename(stale)
            out_dir.mkdir(parents=True, exist_ok=True)
            log(f"{tag}: inputs changed, previous checkpoints moved to {stale.name}")
        jdump(key_path, {"key": key})

    steps = math.ceil(len(dataset) / (micro * accum) * epochs)
    warmup = max(1, int(steps * CFG.warmup_ratio))
    save_steps = max(10, steps // 8)
    args = train_args(out_dir, epochs, lr, micro, accum, warmup_steps=warmup, save_steps=save_steps)

    resume = pick_resume_checkpoint(out_dir) if same_run or not key else None
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, data_collator=collator,
        callbacks=[MetricsWriter(tag)],
    )
    log(f"{tag}: {len(dataset)} examples, {epochs} epochs -> about {steps} optimizer steps")
    log(f"{tag}: a checkpoint is written to {out_dir} every {save_steps} steps")
    log(f"{tag}: resuming from {resume.name}" if resume else f"{tag}: starting from step 0")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    minutes = (time.time() - t0) / 60
    log(f"{tag}: done in {minutes:.1f} min")

    # The stage root is written last and is what adapter_ready() looks for, so a
    # run that dies before this line is never reported as finished.
    model.save_pretrained(str(out_dir))
    # The kernel now holds exactly these weights, so the next cell does not pay
    # for reloading them from disk.
    global _LOADED_ADAPTER
    _LOADED_ADAPTER = str(out_dir)
    prune_checkpoints(out_dir)
    append_metrics(tag, [{"step": int(trainer.state.global_step), "event": "finished",
                          "minutes": round(minutes, 1)}])
    return read_metrics(tag)


def loss_curve(history: list[dict], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    points = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    if not points:
        return
    steps, losses = zip(*points)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps, losses, marker="o", ms=3, color="#2f5d50")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.show()
    print(f"saved {out_path}")


# Adapters trained under a different plan are moved aside before any cell can
# load them. Done here rather than in the CPT cell because a run that restarts
# at SFT must hit the same guard.
reset_stale_checkpoints()

print("model helpers ready")

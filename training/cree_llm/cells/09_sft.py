# ─────────────────────────────────────────────────────────────────────────────
# Supervised instruction tuning.
#
# Only the assistant's tokens contribute to the loss. The schedule is short and
# the learning rate is lower than CPT: this stage is teaching a format and a
# behaviour, not a language, and over-training it flattens the variety of the
# replies into one template.
#
# SFT is built on CPT, never on the bare base model. After a disconnect the
# kernel has no memory of CPT, so the adapter is loaded from Drive explicitly;
# training instruction-following on an untuned base would produce a model that
# answers neatly in English and cannot write Cree at all.
# ─────────────────────────────────────────────────────────────────────────────
train_rows = read_jsonl(PROC / "sft_train.jsonl")
val_rows = read_jsonl(PROC / "sft_val.jsonl")

def _file_digest(path: Path) -> str:
    """Cheap identity for an instruction set.

    The row count is not enough: changing which passages are used as retrieval
    distractors leaves the count identical and the examples different, and the
    stage would then be skipped while its adapter no longer matches the data.
    """
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


_SFT_KEY = fingerprint(
    base=CFG.base_model, seq=CFG.seq_len, lora=CFG.lora_r,
    epochs=CFG.sft_epochs, examples=len(train_rows),
    data=_file_digest(PROC / "sft_train.jsonl"),
)

if SCORING_ONLY or stage_done("sft", _SFT_KEY, requires=CKPT / "sft" / "adapter_config.json"):
    model, tok = ensure_model()
    sft_history = read_metrics("SFT")
else:
    fresh_if_forced("sft")
    model, tok = ensure_model(upto="cpt")

    def _fits(row: dict) -> bool:
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        return len(tok(text, add_special_tokens=False)["input_ids"]) <= CFG.seq_len

    before = len(train_rows)
    train_rows = [r for r in train_rows if _fits(r)]
    val_rows = [r for r in val_rows if _fits(r)]
    log(f"SFT: kept {len(train_rows)} of {before} examples within {CFG.seq_len} tokens")

    collator = CompletionOnlyCollator(tok, CFG.seq_len)
    sft_data = ChatDataset(train_rows)
    model.train()
    run_training(
        model, sft_data, collator, CKPT / "sft",
        epochs=CFG.sft_epochs, lr=CFG.sft_lr,
        micro=CFG.micro_batch, accum=max(4, CFG.grad_accum // 2), tag="SFT", key=_SFT_KEY,
    )
    mark_stage("sft", _SFT_KEY, examples=len(train_rows))
    log(f"SFT adapter saved to {CKPT / 'sft'}")
    sft_history = read_metrics("SFT")

loss_curve(sft_history, PLOTS / "sft_loss.png", "Instruction tuning loss")


@torch.no_grad()
def reply(prompt: str, max_new_tokens: int = 120, system: str = SYSTEM) -> str:
    model.eval()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    model.train()
    return tok.decode(new, skip_special_tokens=True).strip()


print("\n── held-out prompts, after tuning ──")
for row in val_rows[: (1 if SMOKE else 4)]:
    question = row["messages"][1]["content"]
    print(f"\nQ: {question[:180]}")
    print(f"A: {reply(question, 40 if SMOKE else 80)[:400]}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

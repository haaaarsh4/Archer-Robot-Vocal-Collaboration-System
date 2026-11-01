# ─────────────────────────────────────────────────────────────────────────────
# Continued pretraining (CPT).
#
# Instruction tuning alone teaches a model what to say, not how the language
# works. CPT is the stage that lets the model absorb Cree spelling, morphology
# and sentence rhythm from running text. Two details keep it from doing damage:
#
#   * **Replay.** One block in ten is English. Without it a small LoRA can drift
#     far enough that the model forgets how to follow an English instruction --
#     the exact failure that makes a specialist model useless as an assistant.
#   * **Packing.** Documents are joined with an EOS and sliced into full
#     windows, so no compute is spent on padding.
#
# This is the longest stage, so it is also the one a disconnect hurts most. It
# checks the run ledger first: if CPT already finished, the packed dataset is
# not rebuilt and the English replay is not downloaded again, the trained
# adapter is loaded, and the cell moves on. If it was interrupted, the trainer
# continues from its last checkpoint on Drive.
# ─────────────────────────────────────────────────────────────────────────────
_CPT_KEY = fingerprint(
    base=CFG.base_model, seq=CFG.seq_len, lora=CFG.lora_r,
    epochs=CFG.cpt_epochs, sentences=(PROC / "lm_train.jsonl").stat().st_size,
)

if stage_done("cpt", _CPT_KEY, requires=CKPT / "cpt" / "adapter_config.json"):
    model, tok = ensure_model()
    cpt_history = read_metrics("CPT")
else:
    fresh_if_forced("cpt")
    cree_texts = [r["text"] for r in read_jsonl(PROC / "lm_train.jsonl")]

    # The tokenizer is loaded first so the packed dataset can use it.
    from transformers import AutoTokenizer as _AutoTokenizer

    _tok = _AutoTokenizer.from_pretrained(PROC / "tokenizer")
    cree_ds = PackedCausalDataset(cree_texts, _tok, CFG.seq_len, limit=CFG.max_cpt_packs)
    log(f"CPT: {len(cree_texts)} Cree sentences -> {len(cree_ds)} packed blocks of {CFG.seq_len} tokens")

    english_blocks: list[list[int]] = []
    try:
        from datasets import load_dataset

        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        en_texts = [t for t in wt["text"] if len(t.strip()) > 200]
        if SMOKE:
            en_texts = en_texts[:40]
        en_ds = PackedCausalDataset(en_texts, _tok, CFG.seq_len, limit=max(8, len(cree_ds) // 9))
        english_blocks = en_ds.blocks
        log(f"CPT: {len(english_blocks)} English replay blocks (10% of the stream)")
    except Exception as exc:
        log(f"CPT: English replay unavailable ({exc}); training on Cree only")

    mixed_blocks: list[list[int]] = []
    for i, block in enumerate(cree_ds.blocks):
        if english_blocks and i % 9 == 8:
            mixed_blocks.append(english_blocks[(i // 9) % len(english_blocks)])
        else:
            mixed_blocks.append(block)

    cpt_data = BlockDataset(mixed_blocks)
    del _tok

    # CPT always begins at the base model. A resumed CPT does not need the
    # adapter loaded here: the trainer restores it from its checkpoint.
    model, tok = load_model()
    model.train()

    cpt_history = run_training(
        model, cpt_data, None, CKPT / "cpt",
        epochs=CFG.cpt_epochs, lr=CFG.cpt_lr,
        micro=CFG.micro_batch, accum=CFG.grad_accum, tag="CPT",
    )
    mark_stage("cpt", _CPT_KEY, sentences=len(cree_texts), blocks=len(cpt_data))
    log(f"CPT adapter saved to {CKPT / 'cpt'}")

loss_curve(cpt_history, PLOTS / "cpt_loss.png", "Continued pretraining loss")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# Preference tuning for faithfulness.
#
# A translation model can be fluent and still be wrong: a dropped negation, a
# changed number, an English word left in the Cree, a long vowel flattened. Each
# of those is a real failure, so each becomes a rejected answer while the gold
# translation is the chosen one. The objective is DPO, with the reference being
# the same model with its adapters switched off, so no second copy is needed.
# ─────────────────────────────────────────────────────────────────────────────
CFG.dpo_enabled = CFG.dpo_enabled and CFG.dpo_epochs > 0

pairs_train = read_jsonl(PROC / "parallel_train.jsonl")
pair_rng = random.Random(CFG.seed + 1)
prefs: list[dict] = []


def corrupt(cree: str, english: str) -> str | None:
    kind = pair_rng.randrange(5)
    words = cree.split()
    if kind == 0 and len(words) > 3:                      # drop the final clause
        cut = max(2, int(len(words) * 0.66))
        return " ".join(words[:cut])
    if kind == 1 and re.search(r"[âêîô]", cree):           # flatten a long vowel
        return re.sub(r"[âêîô]", lambda m: {"â": "a", "ê": "e", "î": "i", "ô": "o"}[m.group(0)], cree, count=1)
    if kind == 2 and english.split():                     # English leaks in
        donor = pair_rng.choice([w for w in english.split() if len(w) > 3] or english.split())
        pos = pair_rng.randrange(len(words) + 1)
        return " ".join(words[:pos] + [donor] + words[pos:])
    if kind == 3 and re.search(r"\b(nîso|nisto|nêwo|niyânan|nikotwâsik|pêyak)\b", cree):
        return re.sub(r"\b(nîso|nisto|nêwo|niyânan|nikotwâsik|pêyak)\b", "pêyak", cree, count=1)
    if len(words) > 3:                                     # reverse the word order
        return " ".join(list(reversed(words)))
    return None


for row in pairs_train:
    if row.get("source") == "watkins1865":
        continue
    rejected = corrupt(row["cree"], row["english"])
    if not rejected or rejected == row["cree"]:
        continue
    instruction = pair_rng.choice(EN_INSTRUCTIONS)
    prompt = f"<|translate|><|english|><|sro|>\n{instruction}\n{row['english']}"
    prefs.append({"prompt": prompt, "chosen": row["cree"], "rejected": rejected})

pair_rng.shuffle(prefs)
prefs = prefs[: 2000 if not SMOKE else 40]
log(f"DPO: {len(prefs)} preference pairs")

print("\nexample preference:")
print("  prompt  ", prefs[0]["prompt"].replace("\n", " | ")[:160])
print("  chosen  ", prefs[0]["chosen"])
print("  rejected", prefs[0]["rejected"])


_DPO_DIR = CKPT / "dpo"
_DPO_KEY = fingerprint(
    base=CFG.base_model, seq=CFG.seq_len, epochs=CFG.dpo_epochs,
    lr=CFG.dpo_lr, pairs=len(prefs),
)

if not CFG.dpo_enabled or not prefs:
    if not stage_done("dpo", _DPO_KEY, requires=_DPO_DIR / "adapter_config.json"):
        shutil.copytree(CKPT / "sft", _DPO_DIR, dirs_exist_ok=True)
        mark_stage("dpo", _DPO_KEY, skipped=True)
        log("DPO skipped; SFT adapter carried forward unchanged")
elif stage_done("dpo", _DPO_KEY, requires=_DPO_DIR / "adapter_config.json"):
    model, tok = ensure_model()


class PreferenceDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return len(prefs)

    def __getitem__(self, i: int) -> dict:
        return prefs[i]


class PreferenceCollator:
    def __init__(self, tok, max_len: int):
        self.tok = tok
        self.max_len = max_len

    def _encode(self, prompt: str, answer: str):
        p = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        p_ids = self.tok(p, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        ids = (p_ids + a_ids)[-self.max_len:]
        labels = ([-100] * len(p_ids) + a_ids)[-self.max_len:]
        return ids, labels

    def __call__(self, features: list[dict]) -> dict:
        out = {}
        for side in ("chosen", "rejected"):
            rows = [self._encode(f["prompt"], f[side]) for f in features]
            width = max(len(r[0]) for r in rows)
            out[side] = {
                "input_ids": torch.tensor([r[0] + [self.tok.pad_token_id] * (width - len(r[0])) for r in rows]),
                "labels": torch.tensor([r[1] + [-100] * (width - len(r[1])) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r[0]) + [0] * (width - len(r[0])) for r in rows]),
            }
        return out


def sequence_logprob(model, input_ids, labels, attention_mask) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets = labels[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = targets != -100
    return (logp * mask).sum(dim=1)


if CFG.dpo_enabled and prefs and not stage_done("dpo", _DPO_KEY, quiet=True):
    # A disconnect during DPO used to lose the whole stage, because this loop is
    # hand-written and had no saving of its own. It now writes the adapter and a
    # `trainer_state.json` every few dozen steps, which is enough for the same
    # checker the CPT and SFT stages use to accept it as a resume point.
    fresh_if_forced("dpo")
    partial = pick_resume_checkpoint(_DPO_DIR)
    done = int(jload(partial / "trainer_state.json")["global_step"]) if partial is not None else 0

    if partial is not None:
        log(f"DPO: resuming from {partial.name} ({done} steps already done)")
        model, tok = load_model(adapter=partial)
        _LOADED_ADAPTER = str(partial)
    else:
        model, tok = ensure_model(upto="sft")

    dpo_collator = PreferenceCollator(tok, CFG.seq_len)
    dpo_data = PreferenceDataset()
    # The generator makes the shuffle identical after a restart, so skipping the
    # first `done` batches lands on the same pairs the interrupted run had.
    loader = torch.utils.data.DataLoader(
        dpo_data, batch_size=1, shuffle=True, collate_fn=dpo_collator,
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    total_steps = max(1, math.ceil(len(loader) * CFG.dpo_epochs))
    done = min(done, max(0, total_steps - 1))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG.dpo_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.dpo_lr, total_steps=total_steps, pct_start=0.1)
    for _ in range(done):
        sched.step()          # replay the schedule to the resumed point
    model.train()
    beta = CFG.dpo_beta
    losses: list[float] = []
    step = seen = 0
    save_every = max(10, total_steps // 8)
    t0 = time.time()
    for epoch in range(int(CFG.dpo_epochs)):
        for batch in loader:
            if seen < done:
                seen += 1
                continue
            seen += 1
            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen = {k: v.to(model.device) for k, v in chosen.items()}
            rejected = {k: v.to(model.device) for k, v in rejected.items()}

            pol_c = sequence_logprob(model, chosen["input_ids"], chosen["labels"], chosen["attention_mask"])
            pol_r = sequence_logprob(model, rejected["input_ids"], rejected["labels"], rejected["attention_mask"])
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                ref_c = sequence_logprob(model, chosen["input_ids"], chosen["labels"], chosen["attention_mask"])
                ref_r = sequence_logprob(model, rejected["input_ids"], rejected["labels"], rejected["attention_mask"])
            model.train()

            margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 10 == 0:
                mean = float(np.mean(losses[-10:]))
                log(f"DPO step {step}/{total_steps}  loss {mean:.4f}")
                append_metrics("DPO", [{"step": step, "loss": round(mean, 5)}])
            if step % save_every == 0:
                folder = _DPO_DIR / f"checkpoint-{step}"
                model.save_pretrained(str(folder))
                jdump(folder / "trainer_state.json", {"global_step": step, "total": total_steps})
                prune_checkpoints(_DPO_DIR)
                log(f"DPO checkpoint {step}/{total_steps} written to {folder.name}")
    minutes = (time.time() - t0) / 60
    log(f"DPO: done in {minutes:.1f} min")

    model.save_pretrained(str(_DPO_DIR))
    _LOADED_ADAPTER = str(_DPO_DIR)
    prune_checkpoints(_DPO_DIR)
    mark_stage("dpo", _DPO_KEY, pairs=len(prefs), steps=total_steps)
    append_metrics("DPO", [{"step": total_steps, "event": "finished", "minutes": round(minutes, 1)}])

    import matplotlib.pyplot as plt

    rows = [r for r in read_metrics("DPO") if "loss" in r]
    xs = [r["step"] for r in rows] or list(range(len(losses)))
    ys = [r["loss"] for r in rows] or losses
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, ys, color="#8c3b2e")
    ax.set_title("Preference tuning loss (faithfulness)")
    ax.set_xlabel("step")
    ax.set_ylabel("DPO loss")
    fig.tight_layout()
    fig.savefig(PLOTS / "dpo_loss.png", dpi=160)
    plt.show()

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
log(f"final adapter: {CKPT / 'dpo'}")

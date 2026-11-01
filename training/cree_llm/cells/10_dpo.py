# ─────────────────────────────────────────────────────────────────────────────
# Preference tuning for faithfulness.
#
# The one model can be fluent and still be wrong: a dropped negation, a changed
# number, an English word left in the Cree, a long vowel flattened, or an
# invented detail in a song answer. Each of those is a real failure, so each
# becomes a rejected answer while the faithful one is chosen. The objective is
# DPO, with the reference being the same model with its adapters switched off,
# so no second copy is needed.
# ─────────────────────────────────────────────────────────────────────────────
CFG.dpo_enabled = CFG.dpo_enabled and CFG.dpo_epochs > 0

pairs_train = read_jsonl(PROC / "parallel_train.jsonl")
pair_rng = random.Random(CFG.seed + 1)
prefs: list[dict] = []

# Content words drawn from the corpus, for the substitution corruption below.
# Reversing the word order would not be a faithful negative: Plains Cree word
# order is freer than English, and a reordered sentence is often grammatical,
# so that pair would teach the model to downvote valid Cree.
_DPO_VOCAB = sorted({w for row in pairs_train[:3000] for w in row["cree"].split() if len(w) > 2})


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
    if len(words) > 2 and _DPO_VOCAB:                      # a different word: wrong meaning
        pos = pair_rng.randrange(len(words))
        choices = [w for w in _DPO_VOCAB if w != words[pos]]
        if choices:
            words = list(words)
            words[pos] = pair_rng.choice(choices)
            return " ".join(words)
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

# Abstention is part of the same skill. A note answer that fills a gap with a
# plausible invention is the failure this project cares about most, so the
# honest reply is chosen and an invented one is rejected.
# The title that belongs to `_horse_context`, which is not necessarily the
# first note: the first section of the first note can be the held-out one.
_abstain_title = _horse_title
for question, honest, invented in [
    (f"What key was {_abstain_title} recorded in?",
     "The notes do not say what key it was recorded in.",
     f"It was recorded in D major, in {_abstain_title}, at 92 BPM."),
    ("Who played bass on these recordings?",
     "The notes do not say who played on the recordings.",
     "Aaron Twoyoungmen played bass on every track."),
    ("What is the backstory of the song The Unwritten Song?",
     "I do not have notes on that song yet, so I cannot tell you.",
     "It was written in 2019 after a trip to Batoche, and it began as a round."),
]:
    prompt = f"<|song|><|note|>\nNotes on {_abstain_title}:\n{_horse_context}\n\nQuestion: {question}"
    prefs.append({"prompt": prompt, "chosen": honest, "rejected": invented})

pair_rng.shuffle(prefs)
prefs = prefs[: 2000 if not SMOKE else 40]
log(f"DPO: {len(prefs)} preference pairs")

print("\nexample preference:")
print("  prompt  ", prefs[0]["prompt"].replace("\n", " | ")[:160])
print("  chosen  ", prefs[0]["chosen"])
print("  rejected", prefs[0]["rejected"])


_DPO_DIR = CKPT / "dpo"
# `ref` versions the reference policy. The first version of this stage used the
# adapter-disabled base model, which is a different objective; a run that holds
# one of those adapters must not be told its DPO stage is still current.
_DPO_KEY = fingerprint(
    base=CFG.base_model, seq=CFG.seq_len, epochs=CFG.dpo_epochs,
    lr=CFG.dpo_lr, pairs=len(prefs), ref="sft-adapter-v1",
    corrupt="substitute-v1",
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
    # The preference set changed with the unified instruction data, so the old
    # DPO checkpoints belong to a different objective. The same rule as SFT:
    # continue only when the stage fingerprint matches.
    _dpo_key_path = _DPO_DIR / "stage_key.json"
    if not _dpo_key_path.exists() or jload(_dpo_key_path).get("key") != _DPO_KEY:
        if any(_DPO_DIR.glob("checkpoint-*")):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            stale = _DPO_DIR.with_name(f"dpo.stale-{stamp}")
            _DPO_DIR.rename(stale)
            log(f"DPO: inputs changed, previous checkpoints moved to {stale.name}")
        _DPO_DIR.mkdir(parents=True, exist_ok=True)
        jdump(_dpo_key_path, {"key": _DPO_KEY})
        partial = None
    else:
        partial = pick_resume_checkpoint(_DPO_DIR)
    done = int(jload(partial / "trainer_state.json")["global_step"]) if partial is not None else 0

    if partial is not None:
        log(f"DPO: resuming from {partial.name} ({done} steps already done)")
        model, tok = load_model(adapter=partial)
        _LOADED_ADAPTER = str(partial)
    else:
        model, tok = ensure_model(upto="sft")

    # The reference policy is the instruction-tuned adapter, frozen and loaded
    # beside the policy adapter (LoRA weights only, so no second base model).
    # `disable_adapter()` would make the *base* model the reference instead, and
    # DPO is KL-regularised: regularising towards the base pulls the policy back
    # and undoes the instruction tuning this stage exists to sharpen. That is a
    # quiet way to end up worse than SFT, so the reference is stated here.
    if "ref" not in getattr(model, "peft_config", {}):
        model.load_adapter(str(CKPT / "sft"), adapter_name="ref", is_trainable=False)
        log(f"DPO: reference policy = the frozen SFT adapter at {CKPT / 'sft'}")
    model.set_adapter("default")

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
            model.set_adapter("ref")
            with torch.no_grad():
                ref_c = sequence_logprob(model, chosen["input_ids"], chosen["labels"], chosen["attention_mask"])
                ref_r = sequence_logprob(model, rejected["input_ids"], rejected["labels"], rejected["attention_mask"])
            model.set_adapter("default")
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
                # Only the policy adapter: the frozen reference is a copy of the
                # SFT adapter and saving it would double the export folder.
                model.save_pretrained(str(folder), selected_adapters=["default"])
                jdump(folder / "trainer_state.json", {"global_step": step, "total": total_steps})
                prune_checkpoints(_DPO_DIR)
                log(f"DPO checkpoint {step}/{total_steps} written to {folder.name}")
    minutes = (time.time() - t0) / 60
    log(f"DPO: done in {minutes:.1f} min")

    model.save_pretrained(str(_DPO_DIR), selected_adapters=["default"])
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

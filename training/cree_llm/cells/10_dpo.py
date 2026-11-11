# Preference tuning for faithfulness.
#
# The one model can be fluent and still be wrong: a dropped negation, a changed
# number, an English word left in the Cree, a long vowel flattened, or an
# invented detail in a song answer. Each of those is a real failure, so each
# becomes a rejected answer while the faithful one is chosen. The objective is
# DPO, with the reference being the same model with its adapters switched off,
# so no second copy is needed.
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
# accurate reply is chosen and an invented one is rejected.
# The title that belongs to `_horse_context`, which is not necessarily the
# first note: the first section of the first note can be the held-out one.
_abstain_title = _horse_title
for question, accurate, invented in [
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
    prefs.append({"prompt": prompt, "chosen": accurate, "rejected": invented})

# Echo negatives. The installed model answers "hi" with "hi", "howdy" with
# "howdy" and "good morning" with "good morning". That is a copy of the prompt,
# not a reply, and preference training is the right place to penalise it: the
# same prompt gets the real answer as chosen and the visitor's own words as
# rejected. The General Preference Optimization paper and ParaPO both treat
# verbatim reproduction as a preference failure rather than a sampling one.
_echo_prefs: list[dict] = []
_sft_path = PROC / "sft_train.jsonl"
if _sft_path.exists():
    for row in read_jsonl(_sft_path):
        if row.get("task") not in {"casual", "persona"}:
            continue
        messages = row.get("messages") or []
        users = [m for m in messages if m.get("role") == "user"]
        answers = [m for m in messages if m.get("role") == "assistant"]
        if not users or not answers:
            continue
        user = str(users[-1].get("content", ""))
        assistant = str(answers[-1].get("content", ""))
        clean_user = re.sub(r"<\|[^|]*\|>", " ", user)
        clean_user = re.sub(r"\s+", " ", clean_user).strip()
        if not clean_user or len(clean_user) > 120:
            continue
        if clean_user.lower().strip(" .,!?:;") == assistant.lower().strip(" .,!?:;"):
            continue
        _echo_prefs.append({"prompt": user, "chosen": assistant, "rejected": clean_user})
        if len(_echo_prefs) >= 240:
            break
    log(f"DPO: {len(_echo_prefs)} echo negatives from the instruction set")

# Language identification, as a preference rather than a rule. The identifier
# trained on the parallel corpus inherits two biases from it: the Cree side
# often carries English proper names, and the English side is often a two-word
# gloss. The result is a model that answers "Plains Cree." for English
# sentences that talk about Cree, and for short glosses that look like the
# dictionary rows. Preference training shifts a boundary the model already has,
# which is what these pairs do: the same prompt, the right label chosen and the
# plausible wrong label rejected. English prose that names Cree is taken first,
# because that is the class it gets wrong.
_LANG_Q = "<|song|>\nWhich language is this?\n"
_lang_prefs: list[dict] = []
_lang_seen: set[str] = set()
_CREE_MENTION = re.compile(r"\b(cree|n[êe]hiyaw|indian|syllabic)", re.IGNORECASE)


def _lang_pref(text: str, chosen: str, rejected: str) -> None:
    key = re.sub(r"\s+", " ", text).strip().lower()
    if not key or key in _lang_seen or len(text.split()) < 3:
        return
    _lang_seen.add(key)
    _lang_prefs.append({"prompt": _LANG_Q + text, "chosen": chosen, "rejected": rejected})


_mention_rows = [r for r in pairs_train
                 if _CREE_MENTION.search(str(r.get("english", "")))]
for _row in _mention_rows + pairs_train:
    if len(_lang_prefs) >= 200:
        break
    _cree = str(_row.get("cree", "")).strip()
    _english = str(_row.get("english", "")).strip()
    if _row.get("source") == "watkins1865" or re.search(r"\d", _cree + _english):
        continue
    if len(_english.split()) >= 4:
        _lang_pref(_english, "English.", "Plains Cree.")
    if len(_cree.split()) >= 3 and _looks_cree(_cree):
        _lang_pref(_cree, "Plains Cree.", "English.")
    if _cree and _english and _looks_cree(_cree):
        _lang_pref(f"{_cree} {_english}", "Mixed.", "English.")

log(f"DPO: {len(_lang_prefs)} language-identification pairs "
    f"({len(_mention_rows)} English rows name Cree)")

# Corrections the visitor supplied at runtime. The same rows the app exports to
# feedback_dpo.jsonl were previously invisible to training, so a correction
# fixed one conversation and was lost before the next run. They are appended
# after the synthetic cap so a crowd of generated pairs cannot push them out.
_feedback_prefs: list[dict] = []
_feedback_path = _feedback_path("feedback_dpo.jsonl")
if _feedback_path is None:
    log("DPO: no feedback_dpo.jsonl found (set ARCHER_FEEDBACK_DIR to include corrections)")
else:
    for n, row in enumerate(read_jsonl(_feedback_path), 1):
        prompt, chosen, rejected = row.get("prompt"), row.get("chosen"), row.get("rejected")
        if not all(isinstance(v, str) and v.strip() for v in (prompt, chosen, rejected)):
            raise ValueError(f"{_feedback_path}:{n}: prompt, chosen and rejected must be non-empty strings")
        if chosen.strip() == rejected.strip():
            raise ValueError(f"{_feedback_path}:{n}: chosen and rejected are identical")
        _feedback_prefs.append({"prompt": prompt.strip(), "chosen": chosen.strip(),
                                "rejected": rejected.strip()})
    log(f"DPO: {len(_feedback_prefs)} correction pairs from {_feedback_path}")

pair_rng.shuffle(prefs)
prefs = prefs[: 2000 if not SMOKE else 40]
prefs.extend(_feedback_prefs)
prefs.extend(_echo_prefs[: 10 if SMOKE else len(_echo_prefs)])
prefs.extend(_lang_prefs[: 60 if SMOKE else len(_lang_prefs)])
log(f"DPO: {len(prefs)} preference pairs ({len(_feedback_prefs)} feedback, "
    f"{len(_echo_prefs)} echo, {len(_lang_prefs)} language)")

print("\nexample preference:")
print("  prompt  ", prefs[0]["prompt"].replace("\n", " | ")[:160])
print("  chosen  ", prefs[0]["chosen"])
print("  rejected", prefs[0]["rejected"])


_DPO_DIR = CKPT / "dpo"
# `ref` versions the reference policy. The first version of this stage used the
# adapter-disabled base model, which is a different objective; a run that holds
# one of those adapters must not be told its DPO stage is still current.
def _sft_stamp() -> str:
    """Identity of the adapter DPO starts from.

    DPO trains on top of SFT. Re-running SFT changes the policy, so a DPO
    adapter trained on the previous one is not this stage's output -- without
    this the new SFT adapter would be skipped over and the old DPO adapter, with
    the old behaviour baked in, would be exported.
    """
    folder = adapter_ready("sft")
    if folder is None:
        return "none"
    weights = folder / "adapter_model.safetensors"
    stamp = weights.stat() if weights.exists() else folder.stat()
    return f"{folder.name}:{stamp.st_size}:{int(stamp.st_mtime)}"


_DPO_KEY = fingerprint(
    base=CFG.base_model, seq=CFG.seq_len, epochs=CFG.dpo_epochs,
    lr=CFG.dpo_lr, pairs=len(prefs), ref="sft-adapter-v1",
    sft=_sft_stamp(), corrupt="substitute-echo-langid-v3",
    clean=CLEAN_VERSION,
)

# `SCORING_ONLY` is set by the model cell: a scoring run is asking what an
# existing adapter scores, so it must not train a new one on the way there.

if not CFG.dpo_enabled or not prefs:
    if not stage_done("dpo", _DPO_KEY, requires=_DPO_DIR / "adapter_config.json"):
        # copy over the SFT adapter as a *new* folder: a merge into an existing
        # dpo/ leaves the previous adapter's shards beside the copied ones.
        if _DPO_DIR.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            _DPO_DIR.rename(_DPO_DIR.with_name(f"dpo.stale-{stamp}"))
        shutil.copytree(CKPT / "sft", _DPO_DIR)
        mark_stage("dpo", _DPO_KEY, skipped=True)
        log("DPO skipped; SFT adapter carried forward unchanged")
elif SCORING_ONLY or stage_done("dpo", _DPO_KEY, requires=_DPO_DIR / "adapter_config.json"):
    model, tok = ensure_model()
    if SCORING_ONLY:
        log("DPO: scoring run, training skipped")


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


if CFG.dpo_enabled and prefs and not SCORING_ONLY and not stage_done("dpo", _DPO_KEY, quiet=True):
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

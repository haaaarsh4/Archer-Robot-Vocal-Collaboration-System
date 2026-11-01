# ─────────────────────────────────────────────────────────────────────────────
# The report card.
#
# A model without a data card is a rumour. This writes down what went in, what
# was excluded and why, how it was trained, what it scores, and what it must not
# be used for. It is generated from the run's own files, so it cannot drift from
# the numbers in the notebook.
# ─────────────────────────────────────────────────────────────────────────────
report = jload(OUT / "creebench_report.json")
stats = jload(PROC / "stats.json")
tok_audit = jload(PROC / "tokenizer_audit.json")
ledger = jload(RAW / "ledger.json")
status = jload(RAW / "source_status.json")

licences = " ".join(item["licence"].lower() for item in ledger)
if "noncommercial" in licences or "non-commercial" in licences or "nc-" in licences:
    model_licence = "CC BY-NC-SA 4.0 (inherited from NonCommercial corpus material)"
elif "sharealike" in licences or "by-sa" in licences:
    model_licence = "CC BY-SA 4.0 (inherited from ShareAlike corpus material)"
else:
    model_licence = "CC BY 4.0 for the adapter; base model licence applies to the weights"

b, t = report["base"], report["tuned"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


table_rows = [
    ("EN → CRK", "chrF++", b["en_cr"]["chrf++"], t["en_cr"]["chrf++"]),
    ("EN → CRK", "BLEU", b["en_cr"]["bleu"], t["en_cr"]["bleu"]),
    ("EN → CRK", "token F1", b["en_cr"]["token_f1"], t["en_cr"]["token_f1"]),
    ("EN → CRK", "Cree word validity", b["en_cr"].get("validity", 0), t["en_cr"].get("validity", 0)),
    ("EN → CRK", "English code-switch (lower)", b["en_cr"].get("code_switch", 0), t["en_cr"].get("code_switch", 0)),
    ("CRK → EN", "chrF++", b["cr_en"]["chrf++"], t["cr_en"]["chrf++"]),
    ("CRK → EN", "BLEU", b["cr_en"]["bleu"], t["cr_en"]["bleu"]),
    ("CRK → EN", "token F1", b["cr_en"]["token_f1"], t["cr_en"]["token_f1"]),
]

lines = [
    "---",
    "language: crk",
    "language_name: nêhiyawêwin (Plains Cree)",
    "base_model: " + report["meta"]["base_model"],
    f"licence: {model_licence}",
    f"trained: {report['meta']['timestamp']}",
    "tags: [cree, nehiyawewin, low-resource, lora, gguf]",
    "---",
    "",
    "# Cree LLM for the Archer project",
    "",
    "A Plains Cree language model for the Archer project. It translates between English and nêhiyawêwin in both Standard Roman Orthography and syllabics, explains Cree words and sentences, and answers questions about Archer's songs from their notes, with the source listed beside the reply.",
    "",
    "It is not Archer, it does not sing in Archer's voice, and it is not a substitute for a fluent speaker or an Elder.",
    "",
]
if SMOKE:
    lines += [
        "> **Smoke run.** This report came from the tiny CPU profile, whose job is to",
        "> prove the pipeline runs. It deliberately uses a small slice of the corpus",
        "> and a 0.5B model, so the word count and scores below are not the model.",
        "> The `t4`, `a100`, `big` and `huge` profiles use the complete corpus and the",
        "> 7B, 14B or 32B base model.",
        "",
    ]
lines += [
    "## Results on held-out data",
    "",
    "| Task | Measure | Base | Tuned |",
    "|---|---|---:|---:|",
]
for task, measure, bv, tv in table_rows:
    lines.append(f"| {task} | {measure} | {bv:.3f} | {tv:.3f} |")
if report.get("external") and "en_cr" in report["external"]:
    ext = report["external"]
    lines += [
        "",
        f"### Head-to-head: `{ext['model']}`",
        "",
        "The same held-out sentences and metrics, sent through Ollama without this model's control tokens:",
        "",
        "| Task | chrF++ | BLEU | token F1 | Cree validity |",
        "|---|---:|---:|---:|---:|",
        f"| EN → CRK | {ext['en_cr']['chrf++']:.3f} | {ext['en_cr']['bleu']:.3f} | {ext['en_cr']['token_f1']:.3f} | {ext['en_cr'].get('validity', 0):.3f} |",
        f"| CRK → EN | {ext['cr_en']['chrf++']:.3f} | {ext['cr_en']['bleu']:.3f} | {ext['cr_en']['token_f1']:.3f} | - |",
    ]
lines += [
    "",
    f"- Held-out perplexity: **{b['perplexity']:.2f} → {t['perplexity']:.2f}**",
    f"- Syllabics output: **{pct(b['syllabics']['script_share'])} → {pct(t['syllabics']['script_share'])}** of letters in syllabics",
    f"- Abstention: **{pct(t['abstention']['accuracy'])}** of unanswerable questions correctly declined",
]
if "note_reading" in t:
    lines.append(f"- Reading an unseen note section: **token F1 {t['note_reading']['token_f1']:.3f}**")

# The language-model checks measure the failures a translation score cannot
# see: instability, invented confidence, forgotten English, and the cost of a
# Cree word. They are reported here so the card is still the whole story.
_lt = report.get("tests") or {}
_checks: list[tuple[str, str, str]] = []
if _lt.get("determinism", {}).get("n"):
    d = _lt["determinism"]
    _checks.append(("Same answer twice", f"{d['identical']}/{d['n']}", "greedy decoding is stable"))
if _lt.get("abstention", {}).get("unanswerable"):
    a = _lt["abstention"]
    _checks.append(("Declines the unknown", pct(a["rate"]),
                    f"false abstentions {a['false_abstain']}/{a['answerable']}"))
if _lt.get("fertility", {}).get("reduction_percent") is not None:
    f = _lt["fertility"]
    _checks.append(("Tokens per Cree word",
                    f"{f['base_tokens_per_word']:.2f} to {f['cree_tokens_per_word']:.2f}",
                    f"{f['reduction_percent']:.1f}% fewer than the base tokenizer"))
if _lt.get("english_retention", {}).get("change_percent") is not None:
    r = _lt["english_retention"]
    _checks.append(("English perplexity",
                    f"{r['base_perplexity']:.2f} to {r['tuned_perplexity']:.2f}",
                    f"{r['change_percent']:+.1f}% change, the replay working"))
if _lt.get("prompt_ablation", {}).get("en_cr"):
    p = _lt["prompt_ablation"]["en_cr"]
    _checks.append(("Control tokens, EN to CRK",
                    f"{p['with_prefix']:.2f} vs {p['without_prefix']:.2f} chrF++",
                    "what the prefixes are worth"))
if _lt.get("syllabics", {}).get("n"):
    s = _lt["syllabics"]
    _checks.append(("Syllabics, written and read back", pct(s["script_share"]),
                    f"chrF++ {s['roundtrip_chrf']:.1f} after conversion back to SRO"))
if _lt.get("calibration", {}).get("n"):
    _checks.append(("Confidence against quality",
                    f"r = {_lt['calibration']['pearson']:+.2f}",
                    "a model that is confidently wrong is worse than one that is unsure"))
if _checks:
    lines += ["", "### Language model checks", "", "| Check | Result | Note |", "|---|---|---|"]
    for _name, _result, _note in _checks:
        lines.append(f"| {_name} | {_result} | {_note} |")
    lines.append("")
    lines.append("The charts behind these numbers are in `plots/`; the copies sized for the "
                 "project site are in `out/web/` along with `cree-llm.json`.")
lines += [
    "",
    f"All scores are on {report['meta']['n_pairs']} sentence pairs the model never saw. Full per-task output is in `creebench_report.json`.",
    "",
    "## Data",
    "",
    "| Source | Licence | Dialect | Used for |",
    "|---|---|---|---|",
]
for item in ledger:
    lines.append(f"| {item['source']} | {item['licence'][:90]} | {item['dialect']} | {item['note']} |")

lines += [
    "",
    f"Training sentences: **{stats['sentence_words']['train']:,} words** across {stats['sentences']['train']:,} sentences; "
    f"**{stats['pairs']['train']:,} aligned pairs**; **{stats['lexicon_forms']:,} lexicon forms** used for validity checks.",
    "",
    "NoDerivatives material in the EdTeKLA collection is excluded at the loader. Bag-of-words files are used for vocabulary counts only, never as prose. Near-duplicate sentences are removed and the test split is document-disjoint, with a second leakage pass.",
    "",
    "## Tokenizer",
    "",
    f"- Tokens per Cree word: **{tok_audit['base']['fertility']:.2f} → {tok_audit['extended']['fertility']:.2f}** ({tok_audit['reduction_percent']:.1f}% fewer)",
    f"- Characters per token: **{tok_audit['base']['chars_per_token']:.2f} → {tok_audit['extended']['chars_per_token']:.2f}**",
    "- New tokens start from the average of the base model's embeddings for the pieces they replace. Those rows are kept out of the optimizer, so the 1.1 billion row vocabulary table never has to hold gradients or Adam state; the adapter is what learns to read and emit them.",
    "",
    "## Training recipe",
    "",
    f"| Stage | Data | Schedule |",
    "|---|---|---|",
    f"| Continued pretraining | Cree prose + 10% English replay | {CFG.cpt_epochs} epoch(s), lr {CFG.cpt_lr}, LoRA r={CFG.lora_r} |",
    f"| Instruction tuning | translation, orthography, notes, abstention, persona | {CFG.sft_epochs} epoch(s), lr {CFG.sft_lr} |",
    f"| Preference tuning | faithfulness pairs (dropped clauses, English leaks, flattened vowels) | {CFG.dpo_epochs} epoch(s), beta {CFG.dpo_beta} |",
    "",
    "Packing removes padding from pretraining. Instruction tuning masks the prompt and scores only the reply. DPO uses the adapter-disabled model as its reference, so no second model is loaded.",
    "",
    "## Intended use",
    "",
    "- Translating short sentences and phrases between English and Plains Cree.",
    "- Reading and writing SRO and syllabics.",
    "- Answering questions about Archer's songs from the supplied notes.",
    "- Drafting Cree text that a fluent speaker then reviews.",
    "",
    "## Out of scope",
    "",
    "- Ceremonial, legal, medical or treaty language.",
    "- Any claim that a translation is authoritative without review by a fluent speaker.",
    "- Generating new songs or lyrics in Cree.",
    "- Speaking as Archer or imitating his voice.",
    "",
    "## Attribution",
    "",
    "The Cree corpus is built on the EdTeKLA Indigenous Languages Corpora (University of Alberta), the Plains Cree Figurative Language corpus (KonradBRG, CC BY 4.0), Cree Wikipedia (CC BY-SA 4.0), FineWeb-2, FinePDFs, FineTranslations, GlotCC and GlotStoryBook (ODC-BY 1.0 and CC BY-NC-SA where stated), Tatoeba (CC BY 2.0 FR), the Internet Archive's public-domain Cree books, and the GiellaLT lang-crk lexicon and public texts (ALTLab, University of Alberta, with the Maskwacîs Education Schools Commission and First Nations University). The orthography tables follow GiellaLT's Latn-to-Cans and Cans-to-Latn transducers.",
    "",
    "The song notes are Archer Pechawis's own. They are read from the project's `song_notes/` directory to build grounded question and answer examples and are not included as a text corpus in the release.",
    "",
    "## Files",
    "",
    "| File | What it is |",
    "|---|---|",
    f"| `{MODEL_STEM}.Q4_K_M.gguf` | quantised model for Ollama / llama.cpp |",
    "| `Modelfile` | system prompt, context length and sampling |",
    "| `creebench_report.json` | every score and the generated examples |",
    "| `training_curves.png`, `creebench.png`, `tokenizer_fertility.png` | figures |",
    "",
]

(OUT / "report_card.md").write_text("\n".join(lines), encoding="utf-8")
print((OUT / "report_card.md").read_text(encoding="utf-8")[:2800])
print(f"\nwrote {OUT / 'report_card.md'}")

# ── the run is only finished when these are true ─────────────────────────────
assert (OUT / "creebench_report.json").exists()
assert t["en_cr"]["chrf++"] == t["en_cr"]["chrf++"], "NaN translation score"
assert t["perplexity"] > 0
assert stats["sentences"]["test"] > 0
print("\nreport checks passed")

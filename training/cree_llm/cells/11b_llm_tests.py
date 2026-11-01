# ─────────────────────────────────────────────────────────────────────────────
# The language-model test battery.
#
# CreeBench answers one question: is the translation better than the base
# model's? A translation score cannot tell you whether the model is stable,
# whether it keeps English, whether it knows when it does not know, what a Cree
# word costs in tokens, or whether its confidence means anything. Those are the
# failures a speaker notices first, so they are measured separately here, with
# a fixed probe set so two runs of the same adapter produce the same numbers.
#
# Everything is written twice: a full record for the run (llm_tests.json, also
# merged into creebench_report.json) and a small bundle under out/web/ that the
# project site can read without parsing the whole report. The charts are written
# as SVG in the site's own palette, so a page can use them without restyling.
# ─────────────────────────────────────────────────────────────────────────────
import torch

model, tok = ensure_model()
model.eval()

_report_path = OUT / "creebench_report.json"
_report = jload(_report_path)
_pairs = sorted(read_jsonl(PROC / "parallel_test.jsonl"), key=lambda r: sha1(r["cree"]))
_eval_pairs = _pairs[: min(CFG.eval_n, len(_pairs))]

# ── the site's palette, copied so the figures look drawn on the page ─────────
_WEB_INK, _WEB_INK_2, _WEB_INK_3 = "#1b1917", "#565049", "#8b8479"
_WEB_PAPER, _WEB_LINE = "#f7f5ef", "#d5d1c7"
_WEB_OCHRE, _WEB_INDIGO, _WEB_SAGE, _WEB_GOLD = "#ad4a2a", "#2d3a61", "#4f6b57", "#a2611d"
_WEB_SANS = "Inter Tight, Inter, sans-serif"
WEB_DIR = OUT / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)


def _web_esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _web_frame(width: int, height: int, pad: tuple[int, int, int, int],
               ymax: float, ylab: str = "", fmt: str = "{:.0f}", ticks: int = 4) -> tuple[str, dict]:
    left, right, top, bottom = pad
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="{_WEB_PAPER}"/>']
    for i in range(ticks + 1):
        value = ymax * i / ticks
        y = top + plot_h - plot_h * i / ticks
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="{_WEB_LINE}" stroke-width="0.7" opacity="{0.9 if i in (0, ticks) else 0.55}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 3.5:.1f}" text-anchor="end" '
                     f'font-family="{_WEB_SANS}" font-size="10" fill="{_WEB_INK_3}">{_web_esc(fmt.format(value))}</text>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="{_WEB_INK_3}" stroke-width="0.8"/>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" stroke="{_WEB_INK_3}" stroke-width="0.8"/>')
    if ylab:
        parts.append(f'<text x="12" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
                     f'font-family="{_WEB_SANS}" font-size="10.5" fill="{_WEB_INK_2}" '
                     f'transform="rotate(-90 12 {top + plot_h / 2:.1f})">{_web_esc(ylab)}</text>')
    return "".join(parts), {"left": left, "top": top, "w": plot_w, "h": plot_h, "ymax": ymax}


def _web_save(name: str, body: str, width: int, height: int) -> None:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="100%" role="img">{body}</svg>')
    for folder in (WEB_DIR, PLOTS):
        (folder / name).write_text(svg, encoding="utf-8")


def _web_bars(name: str, labels: list[str], values: list[float], *,
              ylab: str = "", ymax: float = 1.0, fmt: str = "{:.2f}",
              colors: list[str] | None = None, width: int = 720, height: int = 320) -> None:
    base, geo = _web_frame(width, height, (46, 18, 26, 44), ymax, ylab, "{:.0f}")
    parts = [base]
    slot = geo["w"] / max(1, len(values))
    bar = min(56, slot * 0.62)
    for i, (label, value) in enumerate(zip(labels, values)):
        h = geo["h"] * max(0.0, min(1.0, value / geo["ymax"]))
        x = geo["left"] + slot * (i + 0.5) - bar / 2
        y = geo["top"] + geo["h"] - h
        fill = (colors[i] if colors else _WEB_OCHRE)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{h:.1f}" rx="2" fill="{fill}"/>')
        parts.append(f'<text x="{x + bar / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                     f'font-family="{_WEB_SANS}" font-size="10" fill="{_WEB_INK_2}">{_web_esc(fmt.format(value))}</text>')
        parts.append(f'<text x="{geo["left"] + slot * (i + 0.5):.1f}" y="{height - 20}" text-anchor="middle" '
                     f'font-family="{_WEB_SANS}" font-size="9.5" fill="{_WEB_INK_2}">{_web_esc(label)}</text>')
    _web_save(name, "".join(parts), width, height)


def _web_paired(name: str, labels: list[str], first: list[float], second: list[float], *,
                first_label: str, second_label: str, ylab: str = "", fmt: str = "{:.1f}",
                width: int = 720, height: int = 330) -> None:
    top = max(max(first), max(second)) * 1.18 or 1
    base, geo = _web_frame(width, height, (46, 18, 52, 44), top, ylab, "{:.0f}")
    parts = [base]
    slot = geo["w"] / max(1, len(labels))
    bar = min(26, slot * 0.28)
    for i, (label, a, b) in enumerate(zip(labels, first, second)):
        for j, (value, color) in enumerate(((a, _WEB_INK_3), (b, _WEB_OCHRE))):
            h = geo["h"] * value / geo["ymax"]
            x = geo["left"] + slot * (i + 0.5) - bar + j * bar * 1.08
            y = geo["top"] + geo["h"] - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{h:.1f}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{x + bar / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                         f'font-family="{_WEB_SANS}" font-size="9" fill="{_WEB_INK_2}">{_web_esc(fmt.format(value))}</text>')
        parts.append(f'<text x="{geo["left"] + slot * (i + 0.5):.1f}" y="{height - 22}" text-anchor="middle" '
                     f'font-family="{_WEB_SANS}" font-size="9.5" fill="{_WEB_INK_2}">{_web_esc(label)}</text>')
    for j, (label, color) in enumerate(((first_label, _WEB_INK_3), (second_label, _WEB_OCHRE))):
        x = geo["left"] + j * 160
        parts.append(f'<rect x="{x}" y="10" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 15}" y="19" font-family="{_WEB_SANS}" font-size="10" '
                     f'fill="{_WEB_INK_2}">{_web_esc(label)}</text>')
    _web_save(name, "".join(parts), width, height)


def _web_scatter(name: str, xs: list[float], ys: list[float], *, xlab: str, ylab: str,
                 width: int = 720, height: int = 340) -> None:
    # Confidence is a negative log-probability, so the axis has to work with a
    # range that does not start at zero. Map the observed interval onto the plot
    # and print the real values under it.
    x_lo = min(xs) - 0.1 * (max(xs) - min(xs) or 1.0) if xs else -1.0
    x_hi = max(xs) + 0.1 * (max(xs) - min(xs) or 1.0) if xs else 0.0
    y_hi = max(ys) * 1.12 if ys else 1.0
    base, geo = _web_frame(width, height, (48, 18, 24, 46), y_hi, ylab, "{:.0f}")
    parts = [base]
    for x, y in zip(xs, ys):
        px = geo["left"] + geo["w"] * (x - x_lo) / max(1e-9, x_hi - x_lo)
        py = geo["top"] + geo["h"] - geo["h"] * y / y_hi
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{_WEB_INDIGO}" opacity="0.6"/>')
    for i in range(6):
        parts.append(f'<text x="{geo["left"] + geo["w"] * i / 5:.1f}" y="{height - 22}" text-anchor="middle" '
                     f'font-family="{_WEB_SANS}" font-size="9.5" fill="{_WEB_INK_2}">'
                     f'{(x_lo + (x_hi - x_lo) * i / 5):.1f}</text>')
    parts.append(f'<text x="{geo["left"] + geo["w"] / 2:.1f}" y="{height - 4}" text-anchor="middle" '
                 f'font-family="{_WEB_SANS}" font-size="10.5" fill="{_WEB_INK_2}">{_web_esc(xlab)}</text>')
    _web_save(name, "".join(parts), width, height)


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return num / den if den else float("nan")


_ABSENT = re.compile(
    r"\b(no notes|do not have|don't have|does not say|does not mention|do not say|"
    r"cannot say|can't say|not recorded|not in the notes|no information)\b", re.I)


def _translate_prompts(rows: list[dict], direction: str, prefix: bool = True) -> list[str]:
    if direction == "en_cr":
        tag = "<|translate|><|english|><|sro|>\n" if prefix else ""
        return [f"{tag}Translate into Plains Cree (SRO):\n{r['english']}" for r in rows]
    tag = "<|translate|><|cree|><|english|>\n" if prefix else ""
    return [f"{tag}Translate into English:\n{r['cree']}" for r in rows]


@torch.no_grad()
def _peaked(texts: list[str], use_adapters: bool) -> float:
    losses = []
    ctx = nullcontext() if use_adapters else model.disable_adapter()
    with ctx:
        for text in texts:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if ids.shape[1] < 8:
                continue
            ids = ids[:, : CFG.seq_len]
            losses.append(float(model(input_ids=ids, labels=ids).loss))
    return float(math.exp(np.mean(losses))) if losses else float("nan")


@torch.no_grad()
def _scored(prompts: list[str], max_new_tokens: int = 44) -> tuple[list[str], list[float]]:
    """Generate greedily and keep the mean token log-probability of each answer.

    A model that is right for the wrong reason is worse than one that is wrong
    loudly, so confidence is measured against the sentence's own chrF score
    rather than assumed to track it.
    """
    model.eval()
    tok.padding_side = "left"
    texts_out: list[str] = []
    scores: list[float] = []
    for i in range(0, len(prompts), 4):
        chunk = prompts[i:i + 4]
        chats = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(chats, return_tensors="pt", padding=True, truncation=True,
                  max_length=CFG.seq_len - max_new_tokens).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             output_scores=True, return_dict_in_generate=True,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        width = enc["input_ids"].shape[1]
        transition = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        for j in range(len(chunk)):
            texts_out.append(tok.decode(gen.sequences[j][width:], skip_special_tokens=True).strip())
            row = transition[j]
            row = row[torch.isfinite(row)]
            scores.append(float(row.mean()) if row.numel() else float("nan"))
    tok.padding_side = "right"
    return texts_out, scores


_TESTS_KEY = fingerprint(base=CFG.base_model, adapter=_adapter_stamp(), n=len(_eval_pairs),
                         probes="v1")

if stage_done("llmtests", _TESTS_KEY, requires=OUT / "llm_tests.json"):
    llm_tests = jload(OUT / "llm_tests.json")
    print(f"LLM checks already run at {llm_tests['meta']['timestamp']}; "
          f"re-run with CREE_FORCE=llmtests")
else:
    llm_tests: dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "base_model": CFG.base_model,
            "adapter": _adapter_stamp(),
            "profile": ACTIVE_PROFILE,
            "eval_pairs": len(_eval_pairs),
            "config": {
                "seq_len": CFG.seq_len,
                "lora_r": CFG.lora_r,
                "lora_alpha": CFG.lora_alpha,
                "cpt_epochs": CFG.cpt_epochs,
                "sft_epochs": CFG.sft_epochs,
                "dpo_epochs": CFG.dpo_epochs,
                "four_bit": USE_4BIT,
            },
        },
    }
    probes = _eval_pairs[: min(6, len(_eval_pairs))]
    log(f"LLM checks: {len(probes)} probe pairs, determinism, ablation, consistency, abstention")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── 1. the same question twice ───────────────────────────────────────────
    try:
        prompts = _translate_prompts(probes, "en_cr")
        first = generate(prompts, 48, batch_size=4)
        second = generate(prompts, 48, batch_size=4)
        same = sum(a.strip() == b.strip() for a, b in zip(first, second))
        llm_tests["determinism"] = {
            "n": len(prompts), "identical": same,
            "rate": round(same / max(1, len(prompts)), 4),
        }
    except Exception as exc:
        llm_tests["determinism"] = {"error": str(exc)}

    # ── 2. do the control tokens earn their place? ──────────────────────────
    # Same sentences, same model, prefix removed. If the score does not move,
    # the prefixes are decoration and the app could drop them.
    try:
        ablation = {}
        for direction, gold_key in (("en_cr", "cree"), ("cr_en", "english")):
            gold = [r[gold_key] for r in probes]
            with_tag = generate(_translate_prompts(probes, direction, prefix=True), 48, batch_size=4)
            plain = generate(_translate_prompts(probes, direction, prefix=False), 48, batch_size=4)
            tagged_chrf = sacrebleu.corpus_chrf(with_tag, [gold], word_order=2).score
            plain_chrf = sacrebleu.corpus_chrf(plain, [gold], word_order=2).score
            ablation[direction] = {
                "with_prefix": round(tagged_chrf, 3),
                "without_prefix": round(plain_chrf, 3),
                "delta": round(tagged_chrf - plain_chrf, 3),
            }
        llm_tests["prompt_ablation"] = ablation
    except Exception as exc:
        llm_tests["prompt_ablation"] = {"error": str(exc)}

    # ── 3. three ways to ask the same thing ─────────────────────────────────
    try:
        phrasings = (
            "Translate into Plains Cree (SRO):\n{en}",
            "How would you say this in Cree?\n{en}",
            "Write this sentence in nêhiyawêwin:\n{en}",
        )
        consistency, by_phrase = [], {str(i): [] for i in range(len(phrasings))}
        for row in probes[: min(5, len(probes))]:
            outs = [generate([p.format(en=row["english"])], 48, batch_size=1)[0] for p in phrasings]
            for i, out in enumerate(outs):
                by_phrase[str(i)].append(token_f1(out, row["cree"]))
            consistency.append(float(np.mean([token_f1(outs[0], other) for other in outs[1:]])))
        llm_tests["paraphrase"] = {
            "n": len(consistency),
            "agreement": round(float(np.mean(consistency)), 4) if consistency else 0.0,
            "by_phrasing": {k: round(float(np.mean(v)), 4) for k, v in by_phrase.items() if v},
        }
    except Exception as exc:
        llm_tests["paraphrase"] = {"error": str(exc)}

    # ── 4. does it know when the notes are silent? ──────────────────────────
    try:
        context = notes_eval[0]["context"] if notes_eval else "Notes on Horse: a song about a horse."
        unanswerable = [
            f"<|song|><|note|>\nNotes:\n{context}\n\nQuestion: Who played bass on these recordings?",
            f"<|song|><|note|>\nNotes:\n{context}\n\nQuestion: What year was it recorded?",
            "<|song|><|note|>\nNotes: (no matching notes found)\n\nQuestion: What is the backstory of The Unwritten Song?",
            f"<|song|><|note|>\nNotes:\n{context}\n\nQuestion: How many copies did it sell?",
            f"<|song|><|note|>\nNotes:\n{context}\n\nQuestion: Which studio was it mixed in?",
            f"<|song|><|note|>\nNotes:\n{context}\n\nQuestion: Who produced the album?",
        ]
        answerable = [
            f"<|song|><|note|>\nNotes:\n{r['context']}\n\nQuestion: {r['question']}"
            for r in notes_eval[:4]
        ]
        declined = generate(unanswerable, 70, batch_size=3)
        answered = generate(answerable, 90, batch_size=2) if answerable else []
        abstained = sum(bool(_ABSENT.search(text)) for text in declined)
        false_abstain = sum(bool(_ABSENT.search(text)) for text in answered)
        llm_tests["abstention"] = {
            "unanswerable": len(unanswerable),
            "abstained": abstained,
            "rate": round(abstained / len(unanswerable), 4),
            "answerable": len(answerable),
            "false_abstain": false_abstain,
            "false_rate": round(false_abstain / len(answerable), 4) if answerable else 0.0,
            "examples": [{"q": q.rsplit("Question:", 1)[-1].strip(), "a": a[:220]}
                         for q, a in zip(unanswerable, declined)],
        }
    except Exception as exc:
        llm_tests["abstention"] = {"error": str(exc)}

    # ── 5. English did not get worse ────────────────────────────────────────
    # CPT replays one block in ten in English for exactly this reason. The test
    # is the same text through the same model with the adapters on and off.
    try:
        english = []
        for row in read_jsonl(PROC / "sft_val.jsonl")[:60]:
            answer = row["messages"][-1]["content"]
            if len(answer.split()) >= 12:
                english.append(answer)
        english = english[:30] or ["The archive holds fourteen books printed before 1935."]
        base_ppl = _peaked(english, False)
        tuned_ppl = _peaked(english, True)
        llm_tests["english_retention"] = {
            "n": len(english),
            "base_perplexity": round(base_ppl, 3),
            "tuned_perplexity": round(tuned_ppl, 3),
            "change_percent": round(100 * (tuned_ppl - base_ppl) / base_ppl, 2) if base_ppl else None,
        }
    except Exception as exc:
        llm_tests["english_retention"] = {"error": str(exc)}

    # ── 6. what a Cree word costs in tokens ─────────────────────────────────
    try:
        from transformers import AutoTokenizer as _BaseTokenizer

        base_tok = _BaseTokenizer.from_pretrained(CFG.base_model)
        texts = [r["text"] for r in test_sents][:200]
        words = sum(len(t.split()) for t in texts) or 1
        base_tokens = sum(len(base_tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
        cree_tokens = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
        llm_tests["fertility"] = {
            "sentences": len(texts),
            "words": words,
            "base_tokens_per_word": round(base_tokens / words, 3),
            "cree_tokens_per_word": round(cree_tokens / words, 3),
            "reduction_percent": round(100 * (1 - cree_tokens / max(1, base_tokens)), 2),
        }
    except Exception as exc:
        llm_tests["fertility"] = {"error": str(exc)}

    # ── 7. syllabics all the way back ───────────────────────────────────────
    try:
        rows = [r for r in _eval_pairs if len(r["cree"].split()) <= 20][:12]
        prompts = [f"<|translate|><|english|><|cans|>\nWrite this in Cree syllabics:\n{r['english']}"
                   for r in rows]
        written = generate(prompts, 72, batch_size=4)
        back = [to_sro(h) for h in written]
        llm_tests["syllabics"] = {
            "n": len(rows),
            "script_share": round(float(np.mean([script_share(h) for h in written])), 4) if written else 0.0,
            "roundtrip_chrf": round(sacrebleu.corpus_chrf(
                back, [[r["cree"] for r in rows]], word_order=2).score, 3) if rows else 0.0,
            "examples": [{"syllabics": h[:120], "back": b[:160]} for h, b in zip(written[:3], back[:3])],
        }
    except Exception as exc:
        llm_tests["syllabics"] = {"error": str(exc)}

    # ── 8. is its confidence worth anything? ────────────────────────────────
    try:
        rows = _eval_pairs[: min(10, len(_eval_pairs))]
        answers, confidence = _scored(_translate_prompts(rows, "en_cr"))
        sentence_chrf = [sacrebleu.sentence_chrf(a, [r["cree"]], word_order=2).score
                         for a, r in zip(answers, rows)]
        llm_tests["calibration"] = {
            "n": len(rows),
            "pearson": round(_pearson(confidence, sentence_chrf), 4),
            "confidence": [round(c, 4) for c in confidence],
            "sentence_chrf": [round(s, 2) for s in sentence_chrf],
        }
    except Exception as exc:
        llm_tests["calibration"] = {"error": str(exc)}

    # ── 9. what a reply costs ───────────────────────────────────────────────
    try:
        start = time.time()
        generate([_translate_prompts(probes[:1], "en_cr")[0]], 32, batch_size=1)
        elapsed = time.time() - start
        llm_tests["cost"] = {
            "seconds": round(elapsed, 2),
            "tokens_per_second": round(32 / elapsed, 2) if elapsed else None,
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                             if torch.cuda.is_available() else None),
            "holdout_pairs": len(_eval_pairs),
        }
    except Exception as exc:
        llm_tests["cost"] = {"error": str(exc)}

    # ── figures and the site bundle ─────────────────────────────────────────
    def _value(section: str, key: str, default: float = 0.0) -> float:
        block = llm_tests.get(section, {})
        value = block.get(key, default) if isinstance(block, dict) else default
        return float(value) if isinstance(value, (int, float)) else default

    checks: list[tuple[str, float, str]] = []
    if "error" not in llm_tests.get("determinism", {}):
        checks.append(("same answer twice", _value("determinism", "rate"), _WEB_SAGE))
    if "error" not in llm_tests.get("paraphrase", {}):
        checks.append(("paraphrase agreement", _value("paraphrase", "agreement"), _WEB_INDIGO))
    if "error" not in llm_tests.get("abstention", {}):
        checks.append(("declines the unknown", _value("abstention", "rate"), _WEB_OCHRE))
        checks.append(("answers the known", 1 - _value("abstention", "false_rate"), _WEB_GOLD))
    if "error" not in llm_tests.get("syllabics", {}):
        checks.append(("syllabics in script", _value("syllabics", "script_share"), _WEB_INDIGO))
    if checks:
        _web_bars("llm-checks.svg", [c[0] for c in checks], [c[1] for c in checks],
                  ylab="share of probes", ymax=1.0, fmt="{:.2f}",
                  colors=[c[2] for c in checks])

    if "fertility" in llm_tests and "error" not in llm_tests["fertility"]:
        f = llm_tests["fertility"]
        _web_bars("llm-fertility.svg", ["base tokenizer", "Cree tokenizer"],
                  [f["base_tokens_per_word"], f["cree_tokens_per_word"]],
                  ylab="tokens per word", ymax=max(f["base_tokens_per_word"], 1) * 1.25,
                  fmt="{:.2f}", colors=[_WEB_INK_3, _WEB_OCHRE])

    if "prompt_ablation" in llm_tests and "error" not in llm_tests["prompt_ablation"]:
        a = llm_tests["prompt_ablation"]
        _web_paired("llm-prompt-ablation.svg", ["EN to CRK", "CRK to EN"],
                    [a["en_cr"]["with_prefix"], a["cr_en"]["with_prefix"]],
                    [a["en_cr"]["without_prefix"], a["cr_en"]["without_prefix"]],
                    first_label="with control tokens", second_label="plain instruction",
                    ylab="chrF++", fmt="{:.1f}")

    if "english_retention" in llm_tests and "error" not in llm_tests["english_retention"]:
        r = llm_tests["english_retention"]
        _web_bars("llm-retention.svg", ["base", "tuned"],
                  [r["base_perplexity"], r["tuned_perplexity"]],
                  ylab="held-out English perplexity", ymax=max(r["tuned_perplexity"], 1) * 1.25,
                  fmt="{:.2f}", colors=[_WEB_INK_3, _WEB_OCHRE])

    if "calibration" in llm_tests and "error" not in llm_tests["calibration"]:
        c = llm_tests["calibration"]
        _web_scatter("llm-calibration.svg", c["confidence"], c["sentence_chrf"],
                     xlab="mean token log-probability", ylab="sentence chrF++")

    figures = {}
    for name, caption in (
        ("llm-checks.svg", "behaviour checks"),
        ("llm-fertility.svg", "tokens per Cree word"),
        ("llm-prompt-ablation.svg", "chrF++ with and without control tokens"),
        ("llm-retention.svg", "held-out English perplexity"),
        ("llm-calibration.svg", "confidence against quality"),
    ):
        if (WEB_DIR / name).exists():
            figures[name] = caption

    headline = {
        "chrf_en_cr": _report["tuned"]["en_cr"]["chrf++"],
        "chrf_cr_en": _report["tuned"]["cr_en"]["chrf++"],
        "chrf_en_cr_base": _report["base"]["en_cr"]["chrf++"],
        "chrf_cr_en_base": _report["base"]["cr_en"]["chrf++"],
        "validity": _report["tuned"]["en_cr"].get("validity"),
        "code_switch": _report["tuned"]["en_cr"].get("code_switch"),
        "perplexity": _report["tuned"].get("perplexity"),
        "perplexity_base": _report["base"].get("perplexity"),
        "time_to_first_token_note": "measured on the training GPU, not a laptop",
    }

    jdump(OUT / "llm_tests.json", llm_tests)
    _report["tests"] = llm_tests
    jdump(_report_path, _report)
    jdump(WEB_DIR / "cree-llm.json", {
        "meta": llm_tests["meta"],
        "headline": headline,
        "tests": {k: v for k, v in llm_tests.items() if k != "meta"},
        "figures": figures,
    })

    # The site's data section is generated from the run's own licence ledger,
    # not retyped into a page. A source that was skipped, or a licence that
    # changed, is reflected the next time this cell runs.
    try:
        _ledger = jload(RAW / "ledger.json")
        _stats = jload(PROC / "stats.json")
        jdump(WEB_DIR / "corpus.json", {
            "sources": _ledger,
            "stats": _stats,
            "note": "Generated from the run's licence ledger. NoDerivatives material is "
                    "excluded at the loader, so allowed is false for those entries.",
        })
    except Exception as exc:
        log(f"site corpus bundle unavailable: {exc}")

    mark_stage("llmtests", _TESTS_KEY,
               deterministic=_value("determinism", "rate"),
               abstention=_value("abstention", "rate"))

    print("\n" + "=" * 78)
    print(f"{'LLM checks':22s}  result")
    print("=" * 78)
    if "determinism" in llm_tests and "error" not in llm_tests["determinism"]:
        d = llm_tests["determinism"]
        print(f"{'same answer twice':22s}  {d['identical']}/{d['n']} ({100 * d['rate']:.0f}%)")
    if "abstention" in llm_tests and "error" not in llm_tests["abstention"]:
        a = llm_tests["abstention"]
        print(f"{'declines the unknown':22s}  {a['abstained']}/{a['unanswerable']} "
              f"({100 * a['rate']:.0f}%), false abstentions {a['false_abstain']}/{a['answerable']}")
    if "prompt_ablation" in llm_tests and "error" not in llm_tests["prompt_ablation"]:
        for direction, block in llm_tests["prompt_ablation"].items():
            print(f"{'control tokens ' + direction:22s}  chrF++ {block['with_prefix']:.2f} vs "
                  f"{block['without_prefix']:.2f} plain ({block['delta']:+.2f})")
    if "fertility" in llm_tests and "error" not in llm_tests["fertility"]:
        f = llm_tests["fertility"]
        print(f"{'tokens per Cree word':22s}  {f['base_tokens_per_word']:.2f} -> "
              f"{f['cree_tokens_per_word']:.2f} ({f['reduction_percent']:.1f}% fewer)")
    if "english_retention" in llm_tests and "error" not in llm_tests["english_retention"]:
        r = llm_tests["english_retention"]
        print(f"{'English perplexity':22s}  {r['base_perplexity']:.2f} -> {r['tuned_perplexity']:.2f} "
              f"({r['change_percent']:+.1f}%)")
    if "calibration" in llm_tests and "error" not in llm_tests["calibration"]:
        print(f"{'confidence vs quality':22s}  r = {llm_tests['calibration']['pearson']:+.2f}")
    print("=" * 78)
    print(f"wrote {OUT / 'llm_tests.json'} and {WEB_DIR}")

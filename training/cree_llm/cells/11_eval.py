# CreeBench: the same test, before and after.
#
# Every number below is computed on data the model never trained on, with the
# base model (adapters off) and the tuned model (adapters on) answering exactly
# the same prompts. Translation is scored with chrF++ and BLEU, but also with
# three measures that matter more for a low-resource language:
#
#   validity     how many generated Cree words exist in the lexicon or obey Cree
#                phonotactics -- invented words are the main failure
#   lexicon_validity  the same count against the lexicon alone. The phonotactic
#                fallback accepts any plausible-looking string, so it cannot tell
#                a real word from an invention; this is the number to gate on.
#   code-switch  how often English leaks into a Cree answer
#   abstention   whether the model says "the notes do not say" instead of
#                inventing a fact
import sacrebleu
from collections import Counter
from contextlib import nullcontext

test_pairs = read_jsonl(PROC / "parallel_test.jsonl")
test_sents = read_jsonl(PROC / "lm_test.jsonl")
notes_eval = read_jsonl(PROC / "sft_notes_eval.jsonl")
lexicon = jload(PROC / "lexicon.json")

VOCAB = set(lexicon)
LEXICON_FORMS = frozenset(lexicon)
for row in read_jsonl(PROC / "lm_train.jsonl"):
    VOCAB.update(re.findall(r"[a-zâêîô-]+", normalize_sro(row["text"])))
_vowel = re.compile(r"[aeiouâêîô]")


def is_cree_word(word: str) -> bool:
    word = word.strip(".,!?;:\"'()[]")
    if not word:
        return True
    if looks_like_syllabics(word):
        return True
    if word in VOCAB:
        return True
    return bool(re.fullmatch(r"[ptkcsmnhwy lr]?[aeiouâêîô][a-zâêîô'-]*", word)) and not re.search(r"[bdfgjqvxz]", word)


def validity(text: str) -> float:
    words = [w for w in re.findall(r"[^\W\d_]+", text) if w]
    if not words:
        return 0.0
    return sum(is_cree_word(w) for w in words) / len(words)


def lexicon_validity(text: str) -> float:
    """Share of words that are attested, with no phonotactic guesswork."""
    words = [w for w in re.findall(r"[^\W\d_]+", text) if w]
    if not words:
        return 0.0
    return sum(w in LEXICON_FORMS for w in words) / len(words)


_EN_LEAK = {"the", "and", "of", "to", "is", "was", "a", "in", "that", "it", "for", "with", "this", "you", "we"}


def code_switch(text: str) -> float:
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text)]
    if not words:
        return 0.0
    return sum(w in _EN_LEAK or bool(re.search(r"[bdfgjqvxz]", w)) for w in words) / len(words)


def script_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(is_syllabics(c) for c in letters) / max(1, len(letters))


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(re.findall(r"[^\W\d_]+", pred.lower())), Counter(re.findall(r"[^\W\d_]+", gold.lower()))
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / max(1, sum(p.values())), overlap / max(1, sum(g.values()))
    return 2 * precision * recall / (precision + recall)


def score_translation(hyps: list[str], refs: list[str], cree_side: bool) -> dict:
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    exact = sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / max(1, len(hyps))
    out = {
        "n": len(hyps),
        "chrf++": round(chrf, 3),
        "bleu": round(bleu, 3),
        "exact_match": round(exact, 4),
        "token_f1": round(float(np.mean([token_f1(h, r) for h, r in zip(hyps, refs)])), 4),
    }
    if cree_side:
        out["validity"] = round(float(np.mean([validity(h) for h in hyps])), 4)
        out["lexicon_validity"] = round(float(np.mean([lexicon_validity(h) for h in hyps])), 4)
        out["code_switch"] = round(float(np.mean([code_switch(h) for h in hyps])), 4)
    return out


@torch.no_grad()
def generate(prompts: list[str], max_new_tokens: int = 64, batch_size: int = 6,
             use_adapters: bool = True) -> list[str]:
    model.eval()
    tok.padding_side = "left"
    outs: list[str] = []
    ctx = nullcontext() if use_adapters else model.disable_adapter()
    with ctx:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=CFG.seq_len - max_new_tokens).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            width = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                outs.append(tok.decode(out[j][width:], skip_special_tokens=True).strip())
    tok.padding_side = "right"
    return outs


def _adapter_stamp() -> str:
    """Identity of the adapter these scores belong to.

    Without it, retraining and re-running the notebook would skip evaluation and
    keep showing the previous model's numbers, which is worse than a slow run.
    """
    best = best_adapter()
    if best is None:
        return "none"
    weights = best / "adapter_model.safetensors"
    stamp = weights.stat() if weights.exists() else best.stat()
    return f"{best.name}:{stamp.st_size}:{int(stamp.st_mtime)}"


# `bench` versions the benchmark itself. Adding the notes and conversation
# baselines changed what a CreeBench row means, and without this an existing
# work folder would keep the old scores and never compute the new ones.
_EVAL_KEY = fingerprint(base=CFG.base_model, n=CFG.eval_n, adapter=_adapter_stamp(),
                        speaker=CFG.external_baseline, bench="v4")

if stage_done("eval", _EVAL_KEY, requires=OUT / "creebench_report.json"):
    results = jload(OUT / "creebench_report.json")
    print(f"CreeBench already run at {results['meta'].get('timestamp', '?')} on "
          f"{results['meta'].get('n_pairs', '?')} held-out pairs; figures are on Drive.")
    print(f"  EN->CRK chrF++ {results['tuned']['en_cr']['chrf++']:.2f} "
          f"(base {results['base']['en_cr']['chrf++']:.2f})")
    print(f"  CRK->EN chrF++ {results['tuned']['cr_en']['chrf++']:.2f} "
          f"(base {results['base']['cr_en']['chrf++']:.2f})")
    print(f"  held-out perplexity {results['tuned']['perplexity']:.2f} "
          f"(base {results['base']['perplexity']:.2f})")
    print("  set CREE_FORCE=eval to score it again")
else:
    model, tok = ensure_model()
    N = min(CFG.eval_n, len(test_pairs))
    eval_pairs = sorted(test_pairs, key=lambda r: sha1(r["cree"]))[:N]
    en_prompts = [f"<|translate|><|english|><|sro|>\nTranslate into Plains Cree (SRO):\n{r['english']}" for r in eval_pairs]
    cr_prompts = [f"<|translate|><|cree|><|english|>\nTranslate into English:\n{r['cree']}" for r in eval_pairs]

    log(f"CreeBench: adapter {best_adapter() or 'none (base only)'}")
    log(f"CreeBench: {len(eval_pairs)} held-out pairs, both directions, base and tuned")
    t0 = time.time()
    tuned_en = generate(en_prompts, 64, use_adapters=True)
    tuned_cr = generate(cr_prompts, 48, use_adapters=True)
    base_en = generate(en_prompts, 64, use_adapters=False)
    base_cr = generate(cr_prompts, 48, use_adapters=False)
    log(f"translation generations done in {(time.time() - t0) / 60:.1f} min")

    # Optional head-to-head: any model already installed in Ollama, including the
    # 70B Cree model this project is measured against. Same prompts, same metrics.
    # Set CREE_EXTERNAL_MODEL=the-ollama-tag before running the notebook.
    results = {
        "meta": {
            "base_model": CFG.base_model,
            "profile": ACTIVE_PROFILE,
            "n_pairs": len(eval_pairs),
            "tokenizer": jload(PROC / "tokenizer_audit.json"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        },
        "base": {
            "en_cr": score_translation(base_en, [r["cree"] for r in eval_pairs], True),
            "cr_en": score_translation(base_cr, [r["english"] for r in eval_pairs], False),
        },
        "tuned": {
            "en_cr": score_translation(tuned_en, [r["cree"] for r in eval_pairs], True),
            "cr_en": score_translation(tuned_cr, [r["english"] for r in eval_pairs], False),
        },
    }

    # Which script did it answer in? A model that writes syllabics when SRO was
    # asked scores near zero on chrF++ for a reason that has nothing to do with
    # translation, and without this number a script failure reads as "it cannot
    # translate". The SRO-converted score separates the two, and five of the
    # actual replies are kept so a run can be inspected without reloading it.
    _en_cr_script = round(float(np.mean([script_share(h) for h in tuned_en])), 4)
    results["tuned"]["en_cr"]["script_share"] = _en_cr_script
    results["base"]["en_cr"]["script_share"] = round(float(np.mean([script_share(h) for h in base_en])), 4)
    results["tuned"]["en_cr"]["chrf++_as_sro"] = round(
        sacrebleu.corpus_chrf(
            [to_sro(h) if script_share(h) > 0.5 else h for h in tuned_en],
            [[r["cree"] for r in eval_pairs]], word_order=2,
        ).score, 3)
    results["tuned"]["en_cr"]["examples"] = [
        {"english": r["english"], "pred": h[:200], "gold": r["cree"][:200]}
        for r, h in list(zip(eval_pairs, tuned_en))[:5]
    ]

    if CFG.external_baseline:
        import requests

        OLLAMA_URL = os.environ.get("CREE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

        def ask_ollama(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
            out = []
            for prompt in prompts:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": max_new_tokens},
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                out.append(resp.json().get("message", {}).get("content", "").strip())
            return out

        def plain(prompt: str) -> str:
            """Drop this model's control tokens; another model must not receive them.

            The prompts above start with <|translate|><|english|><|sro|> because this
            model was trained on them. Handing those bytes to a different model
            measures its reaction to a strange prefix, not its translation.
            """
            return prompt.split("\n", 1)[1] if "\n" in prompt else prompt

        try:
            log(f"CreeBench: external baseline {CFG.external_baseline}")
            ext_en = ask_ollama(CFG.external_baseline, [plain(p) for p in en_prompts], 64)
            ext_cr = ask_ollama(CFG.external_baseline, [plain(p) for p in cr_prompts], 48)
            results["external"] = {
                "model": CFG.external_baseline,
                "en_cr": score_translation(ext_en, [r["cree"] for r in eval_pairs], True),
                "cr_en": score_translation(ext_cr, [r["english"] for r in eval_pairs], False),
            }
        except Exception as exc:
            results["external"] = {"model": CFG.external_baseline, "error": str(exc)}
            log(f"external baseline failed: {exc}")
    else:
        results["external"] = None

    # syllabics: can it write the script, not just name it?
    syl_rows = [r for r in eval_pairs if len(r["cree"].split()) <= 20][: min(60, len(eval_pairs))]
    if syl_rows:
        syl_prompts = [f"<|translate|><|english|><|cans|>\nWrite this in Cree syllabics:\n{r['english']}" for r in syl_rows]
        syl_hyp = generate(syl_prompts, 64, use_adapters=True)
        syl_base = generate(syl_prompts, 64, use_adapters=False)
        gold_cans = [to_syllabics(r["cree"]) for r in syl_rows]
        results["tuned"]["syllabics"] = {
            "n": len(syl_rows),
            "script_share": round(float(np.mean([script_share(h) for h in syl_hyp])), 4),
            "chrf++_vs_gold_syllabics": round(sacrebleu.corpus_chrf(syl_hyp, [gold_cans], word_order=2).score, 3),
        }
        results["base"]["syllabics"] = {
            "n": len(syl_rows),
            "script_share": round(float(np.mean([script_share(h) for h in syl_base])), 4),
            "chrf++_vs_gold_syllabics": round(sacrebleu.corpus_chrf(syl_base, [gold_cans], word_order=2).score, 3),
        }
    else:
        results["tuned"]["syllabics"] = {"n": 0, "script_share": 0.0}
        results["base"]["syllabics"] = {"n": 0, "script_share": 0.0}

    # grounded note reading and abstention
    # Both are measured with the adapters on *and off*. The old system answered
    # song questions with the general model reading retrieved notes, so the
    # adapter-free run is that system, on the same notes and the same questions.
    # Without it "the new model is better at notes" would be an assertion.
    note_prompts = [r["messages"][1]["content"] for r in notes_eval]
    if note_prompts:
        note_hyp = generate(note_prompts, 110, use_adapters=True)
        note_base = generate(note_prompts, 110, use_adapters=False)
        for split, hyps in (("tuned", note_hyp), ("base", note_base)):
            results[split]["note_reading"] = {
                "n": len(note_prompts),
                "token_f1": round(float(np.mean([token_f1(h, r["answer"]) for h, r in zip(hyps, notes_eval)])), 4),
            }
        results["tuned"]["note_reading"]["answers"] = [
            {"q": r["question"], "pred": h[:220], "gold": r["answer"][:220]}
            for r, h in zip(notes_eval, note_hyp)
        ]

    _abs_context = notes_eval[0]["context"] if notes_eval else ""
    abs_prompts = [
        "<|song|><|note|>\nNotes: (no matching notes found)\n\nQuestion: What is the backstory of the song The Unwritten Song?",
        f"<|song|><|note|>\nNotes:\n{_abs_context}\n\nQuestion: What key was it recorded in?",
        f"<|song|><|note|>\nNotes:\n{_abs_context}\n\nQuestion: Who played bass on these recordings?",
    ]
    _ABSENT = r"\b(no notes|do not have|does not say|do not say|cannot say|not recorded|not in the notes)\b"
    abs_hyp = generate(abs_prompts, 60, use_adapters=True)
    abs_base = generate(abs_prompts, 60, use_adapters=False)
    for split, hyps in (("tuned", abs_hyp), ("base", abs_base)):
        abstained = sum(bool(re.search(_ABSENT, h, re.I)) for h in hyps)
        results[split]["abstention"] = {
            "n": len(abs_prompts),
            "correct": abstained,
            "accuracy": round(abstained / len(abs_prompts), 3),
        }
    results["tuned"]["abstention"]["answers"] = abs_hyp

    # conversation about the project
    # Asked three ways the training set never used. A model that only answers the
    # exact wording it memorised is not holding a conversation.
    project_eval = read_jsonl(PROC / "sft_project_eval.jsonl")
    if SMOKE:
        project_eval = project_eval[:6]
    if project_eval:
        proj_prompts = [r["messages"][1]["content"] for r in project_eval]
        proj_hyp = generate(proj_prompts, 110, use_adapters=True)
        proj_base = generate(proj_prompts, 110, use_adapters=False)
        for split, hyps in (("tuned", proj_hyp), ("base", proj_base)):
            results[split]["project"] = {
                "n": len(project_eval),
                "token_f1": round(float(np.mean([token_f1(h, r["answer"]) for h, r in zip(hyps, project_eval)])), 4),
            }
        results["tuned"]["project"]["answers"] = [
            {"q": r["question"], "pred": h[:220], "gold": r["answer"][:220]}
            for r, h in zip(project_eval, proj_hyp)
        ]

    # The echo check. A greeting must be answered, not repeated. This is the
    # failure the installed 7B shows on "hi", "howdy" and "good morning", and
    # it is measured here so a run cannot pass on translation alone.
    _ECHO_PROMPTS = [
        "hi", "hey", "hello", "hi there", "howdy", "good morning",
        "yo", "hey there", "morning", "hello there", "how's it going?",
        "what's up?",
    ]
    echo_prompts = _ECHO_PROMPTS[:4] if SMOKE else _ECHO_PROMPTS
    echo_tuned = generate([f"<|song|>\n{p}" for p in echo_prompts], 64, use_adapters=True)
    echo_base = generate([f"<|song|>\n{p}" for p in echo_prompts], 64, use_adapters=False)

    def _echo_norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()

    for split, hyps in (("tuned", echo_tuned), ("base", echo_base)):
        echoed = sum(_echo_norm(h) == _echo_norm(p) for h, p in zip(hyps, echo_prompts))
        results[split]["echo"] = {
            "n": len(echo_prompts),
            "echoed": echoed,
            "rate": round(echoed / max(1, len(echo_prompts)), 3),
            "answers": [{"q": p, "a": h[:160]} for p, h in zip(echo_prompts, hyps)],
        }

    # does it deny being Archer?
    persona_probes = jload(PROC / "persona_probes.json") if (PROC / "persona_probes.json").exists() else []
    if persona_probes:
        _denial = re.compile(r"\b(no|not|never|neither)\b", re.I)
        persona_hyp = generate([f"<|song|>\n{q}" for q in persona_probes], 70, use_adapters=True)
        persona_base = generate([f"<|song|>\n{q}" for q in persona_probes], 70, use_adapters=False)
        for split, hyps in (("tuned", persona_hyp), ("base", persona_base)):
            results[split]["persona"] = {
                "n": len(persona_probes),
                "denials": sum(bool(_denial.search(h)) for h in hyps),
                "answers": [{"q": q, "a": h[:200]} for q, h in zip(persona_probes, hyps)],
            }

    # perplexity on held-out Cree
    @torch.no_grad()
    def perplexity(texts: list[str], use_adapters: bool) -> float:
        model.eval()
        losses = []
        ctx = nullcontext() if use_adapters else model.disable_adapter()
        with ctx:
            for text in texts[:40]:
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                if ids.shape[1] < 8:
                    continue
                ids = ids[:, : CFG.seq_len]
                out = model(input_ids=ids, labels=ids)
                losses.append(float(out.loss))
        return float(math.exp(np.mean(losses))) if losses else float("nan")


    ppl_texts = [r["text"] for r in test_sents]
    results["tuned"]["perplexity"] = round(perplexity(ppl_texts, True), 3)
    results["base"]["perplexity"] = round(perplexity(ppl_texts, False), 3)

    # the bar: the best score measured before this model
    # The old app was not one model. It was a general model, a separate Cree
    # model and a retrieval index, with a router choosing between them. A single
    # model that replaces all three has to beat the best of those pieces on each
    # job, so the bar is computed here instead of assumed: for every measure, the
    # strongest score this run actually measured from an older part, and the gap
    # to this model. Only the base and, when `--external-model` is given, that
    # model enter the comparison. The separate Cree model is not in the bar
    # unless it was scored too, and `bar_from` says which one set it, so the
    # table cannot pass an unmeasured model off as beaten.
    _ext = results.get("external") if isinstance(results.get("external"), dict) else None
    _ext_ok = bool(_ext and "en_cr" in _ext)

    def bar(task: str, measure: str, tuned: float, base: float,
            external: float | None = None, higher: bool = True) -> dict:
        candidates = [("base model", base)] if base == base else []
        if external is not None and external == external:
            candidates.append((f"{CFG.external_baseline}", external))
        best_name, best = (max(candidates, key=lambda kv: kv[1]) if higher
                           else min(candidates, key=lambda kv: kv[1]))
        meets = tuned >= best - 1e-9 if higher else tuned <= best + 1e-9
        return {"task": task, "measure": measure, "tuned": round(tuned, 4),
                "bar": round(best, 4), "bar_from": best_name, "meets": bool(meets)}

    bars = [
        bar("EN -> CRK", "chrF++", results["tuned"]["en_cr"]["chrf++"],
            results["base"]["en_cr"]["chrf++"],
            _ext["en_cr"]["chrf++"] if _ext_ok else None),
        bar("CRK -> EN", "chrF++", results["tuned"]["cr_en"]["chrf++"],
            results["base"]["cr_en"]["chrf++"],
            _ext["cr_en"]["chrf++"] if _ext_ok else None),
    ]
    if results["tuned"]["en_cr"].get("script_share") is not None:
        # Measured as SRO share, so higher is better and the bar is the base
        # model's own tendency to answer in the script it was asked for.
        bars.append(bar("EN -> CRK", "SRO share",
                        1 - results["tuned"]["en_cr"]["script_share"],
                        1 - results["base"]["en_cr"].get("script_share", 0)))
    if "note_reading" in results["base"] and "note_reading" in results["tuned"]:
        bars.append(bar("Song notes", "token F1", results["tuned"]["note_reading"]["token_f1"],
                        results["base"]["note_reading"]["token_f1"]))
    if "project" in results["base"] and "project" in results["tuned"]:
        bars.append(bar("Conversation", "token F1", results["tuned"]["project"]["token_f1"],
                        results["base"]["project"]["token_f1"]))
    if "abstention" in results["base"]:
        bars.append(bar("Abstention", "accuracy", results["tuned"]["abstention"]["accuracy"],
                        results["base"]["abstention"]["accuracy"]))
    if results["tuned"]["syllabics"].get("n"):
        bars.append(bar("Syllabics", "script share", results["tuned"]["syllabics"]["script_share"],
                        results["base"]["syllabics"]["script_share"]))
    bars.append(bar("Held-out Cree", "perplexity", results["tuned"]["perplexity"],
                    results["base"]["perplexity"], higher=False))

    if "echo" in results["tuned"]:
        # A greeting is answered, never copied. This is an absolute gate, not a
        # comparison with the base model, because a base model that also echoes
        # is not a bar worth clearing.
        _echo_rate = results["tuned"]["echo"]["rate"]
        bars.append({"task": "Conversation", "measure": "echo rate",
                     "tuned": _echo_rate, "bar": 0.0, "bar_from": "gate",
                     "meets": bool(_echo_rate == 0.0)})

    met = sum(item["meets"] for item in bars)
    results["head_to_head"] = {
        "rows": bars,
        "met": met,
        "total": len(bars),
        "verdict": f"meets or beats the best measured score on {met} of {len(bars)} measures",
    }

    # latency, measured rather than guessed
    def latency(prompt: str, n: int = 32) -> float:
        start = time.time()
        generate([prompt], n, batch_size=1, use_adapters=True)
        return round((time.time() - start) / n, 4)


    results["tuned"]["seconds_per_token"] = latency(en_prompts[0])

    jdump(OUT / "creebench_report.json", results)

    # print the table
    def row(name: str, base: dict, tuned: dict, keys: list[str]) -> str:
        parts = [f"{name:22s}"]
        for key in keys:
            b, t = base.get(key, float("nan")), tuned.get(key, float("nan"))
            parts.append(f"{b:8.3f} -> {t:8.3f}")
        return "  ".join(parts)


    print("\n" + "=" * 78)
    print(f"{'CreeBench':22s}  {'base':>8s}    {'tuned':>8s}   (higher is better, except PPL)")
    print("=" * 78)
    for task, keys in [("EN -> CRK", ["chrf++", "bleu", "token_f1", "validity", "code_switch"]),
                       ("CRK -> EN", ["chrf++", "bleu", "token_f1"])]:
        key = "en_cr" if task.startswith("EN") else "cr_en"
        print(row(task, results["base"][key], results["tuned"][key], keys))
    _script = results["tuned"]["en_cr"].get("script_share")
    if _script is not None:
        print(f"{'EN->CRK script':22s}  {100 * _script:.1f}% of letters in syllabics "
              f"(SRO was asked); after conversion chrF++ {results['tuned']['en_cr'].get('chrf++_as_sro', 0):.3f}")
    ext = results.get("external")
    if ext and "en_cr" in ext:
        tag = ext["model"][:22]
        print(f"\n{'external model':22s}  {tag}")
        print(f"  EN->CRK  chrF++ {ext['en_cr']['chrf++']:.3f}   BLEU {ext['en_cr']['bleu']:.3f}   "
              f"token F1 {ext['en_cr']['token_f1']:.3f}   validity {ext['en_cr']['validity']:.3f}")
        print(f"  CRK->EN  chrF++ {ext['cr_en']['chrf++']:.3f}   BLEU {ext['cr_en']['bleu']:.3f}   "
              f"token F1 {ext['cr_en']['token_f1']:.3f}")
        print(f"  (this model's tuned rows above: EN->CRK chrF++ {results['tuned']['en_cr']['chrf++']:.3f}, "
              f"CRK->EN chrF++ {results['tuned']['cr_en']['chrf++']:.3f})")
    print(row("perplexity", {"perplexity": results["base"]["perplexity"]},
              {"perplexity": results["tuned"]["perplexity"]}, ["perplexity"]))
    print(f"{'syllabics script':22s}  {results['base']['syllabics']['script_share']:.3f} -> {results['tuned']['syllabics']['script_share']:.3f} of letters")
    print(f"{'abstention':22s}  {results['tuned']['abstention']['accuracy']:.3f} correct")
    if "note_reading" in results["tuned"]:
        print(f"{'note reading F1':22s}  {results['tuned']['note_reading']['token_f1']:.3f}"
              + (f"  (base {results['base']['note_reading']['token_f1']:.3f})"
                 if "note_reading" in results["base"] else ""))
    if "project" in results["tuned"]:
        print(f"{'conversation F1':22s}  {results['tuned']['project']['token_f1']:.3f}"
              + (f"  (base {results['base']['project']['token_f1']:.3f})"
                 if "project" in results["base"] else ""))
    if "echo" in results["tuned"]:
        print(f"{'greeting echo':22s}  {results['tuned']['echo']['echoed']}/{results['tuned']['echo']['n']} "
              f"repeated verbatim (base {results['base']['echo']['echoed']}/{results['base']['echo']['n']})")

    _h2h = results.get("head_to_head") or {}
    if _h2h.get("rows"):
        print("\nthe bar: the best score measured before this model on each job")
        for item in _h2h["rows"]:
            mark = "ok  " if item["meets"] else "MISS"
            print(f"  {mark} {item['task']:14s} {item['measure']:12s} "
                  f"{item['tuned']:8.3f}  bar {item['bar']:8.3f}  ({item['bar_from']})")
        print(f"  {_h2h['verdict']}")
    print("=" * 78)

    # figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    labels = ["EN→CRK", "CRK→EN"]
    x = np.arange(2)
    w = 0.36
    base_chrf = [results["base"]["en_cr"]["chrf++"], results["base"]["cr_en"]["chrf++"]]
    tuned_chrf = [results["tuned"]["en_cr"]["chrf++"], results["tuned"]["cr_en"]["chrf++"]]
    ax[0].bar(x - w / 2, base_chrf, w, label="base", color="#8c3b2e")
    ax[0].bar(x + w / 2, tuned_chrf, w, label="tuned", color="#2f5d50")
    ax[0].set_xticks(x, labels)
    ax[0].set_title("chrF++ (higher is better)")
    ax[0].legend(frameon=False)

    ax[1].bar(["validity", "no code-switch"], [results["tuned"]["en_cr"]["validity"],
                                               1 - results["tuned"]["en_cr"]["code_switch"]],
              color=["#2f5d50", "#c98b3a"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Generated Cree quality")

    ax[2].bar(["base", "tuned"], [results["base"]["perplexity"], results["tuned"]["perplexity"]],
              color=["#8c3b2e", "#2f5d50"])
    ax[2].set_title("Held-out perplexity (lower is better)")
    fig.tight_layout()
    fig.savefig(PLOTS / "creebench.png", dpi=160)
    plt.show()
    print(f"\nsaved {OUT / 'creebench_report.json'}")

    mark_stage("eval", _EVAL_KEY, pairs=len(eval_pairs),
               chrf_en_cr=results["tuned"]["en_cr"]["chrf++"],
               chrf_cr_en=results["tuned"]["cr_en"]["chrf++"])

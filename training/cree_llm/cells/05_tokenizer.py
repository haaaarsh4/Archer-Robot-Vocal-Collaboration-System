# ─────────────────────────────────────────────────────────────────────────────
# A tokenizer that knows Cree.
#
# The base model's tokenizer was trained mostly on English. For a language with
# long polysynthetic words and marked vowels, that means every word becomes a
# pile of fragments: more tokens per word, more chances to drop a diacritic, and
# less text per context window.
#
# The fix is not "add some words". A SentencePiece BPE model is trained on the
# Cree corpus itself, its pieces are filtered so they can live inside the base
# model's byte-level vocabulary, and the surviving pieces are added as ordinary
# tokens. The next cell initialises their embeddings from the pieces the base
# model already knows, so a new token starts life already meaning something.
# ─────────────────────────────────────────────────────────────────────────────

from transformers import AutoTokenizer
from sentencepiece import SentencePieceTrainer

SPECIAL_TOKENS = [
    "<|cree|>", "<|english|>", "<|sro|>", "<|cans|>",
    "<|translate|>", "<|define|>", "<|song|>", "<|note|>",
]

# The tokenizer is the one artifact whose absence is not obvious: a model with a
# vocabulary of the wrong size loads happily and produces garbage. So the guard
# checks the tokenizer folder, the token map and the audit, and rebuilds all of
# them together if any one is missing.
_TOKENIZER_KEY = fingerprint(
    sentences=(PROC / "lm_train.jsonl").stat().st_size,
    spm_vocab=CFG.spm_vocab,
    budget=CFG.new_token_budget,
    specials=len(SPECIAL_TOKENS),
)

if ((PROC / "tokenizer" / "tokenizer_config.json").exists()
        and (PROC / "new_tokens.json").exists()
        and stage_done("tokenizer", _TOKENIZER_KEY, requires=PROC / "tokenizer_audit.json")):
    comparison = jload(PROC / "tokenizer_audit.json")
    print(json.dumps(comparison, indent=2))
    log(f"tokenizer loaded from Drive: fertility {comparison['reduction_percent']:.1f}% lower than the base")
else:
    lm_train = read_jsonl(PROC / "lm_train.jsonl")
    lm_test = read_jsonl(PROC / "lm_test.jsonl")
    lexicon = jload(PROC / "lexicon.json")

    sp_input = RAW / "spm_input.txt"
    with sp_input.open("w", encoding="utf-8") as fh:
        for row in lm_train:
            fh.write(row["text"].replace("\n", " ") + "\n")
        # Word lists give the tokenizer the shapes of inflected forms that running
        # text mentions only once. They are appended once, not repeated, so they
        # cannot outvote real sentences.
        for word in sorted(lexicon, key=lambda w: -lexicon[w])[:30_000]:
            fh.write(word + "\n")

    sp_prefix = RAW / "cree_spm"
    SentencePieceTrainer.train(
        input=str(sp_input),
        model_prefix=str(sp_prefix),
        vocab_size=CFG.spm_vocab,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        split_digits=True,
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=False,
        shuffle_input_sentence=True,
        max_sentence_length=8192,
        num_threads=max(1, min(8, os.cpu_count() or 1)),
        seed_sentencepiece_size=1_000_000,
        train_extremely_large_corpus=False,
    )

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
    log(f"tokenizer: SentencePiece trained ({sp.get_piece_size()} pieces)")
    log("tokenizer: loading the base tokenizer")
    base_tok = AutoTokenizer.from_pretrained(CFG.base_model)
    log("tokenizer: base tokenizer loaded")

    # Pieces that are safe to graft onto a byte-level BPE vocabulary: no space
    # marker, no control symbol, no bare punctuation, at least two characters.
    seen = set(base_tok.get_vocab())
    candidates: list[str] = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if not piece or piece.startswith("<"):
            continue
        piece = piece.replace("▁", "")
        if not piece or "▁" in piece:
            continue
        piece = normalize_sro(piece)
        if len(piece) < 2 or len(piece) > 32:
            continue
        if piece in seen or not re.fullmatch(r"[a-zâêîô'-]+", piece):
            continue
        seen.add(piece)
        candidates.append(piece)

    # Longest useful pieces first. A three-letter Cree stem taught as one token
    # saves more than a two-letter fragment that the base model already splits well.
    candidates.sort(key=lambda p: (-len(p), p))
    log(f"tokenizer: {len(candidates)} safe Cree pieces after filtering")
    new_tokens = candidates[: CFG.new_token_budget]

    # Syllabics are a script, not a decoration. The base tokenizer treats each
    # glyph as raw bytes, which is both wasteful and fragile. Every glyph the
    # converter knows is added as one token, so a syllable costs one token.
    syllabics_tokens = sorted({
        glyph for glyph in _SRO_TO_CANS.values()
        if len(glyph) == 1 and all(is_syllabics(ch) for ch in glyph)
    })
    for glyph in syllabics_tokens:
        if glyph not in base_tok.get_vocab() and glyph not in new_tokens:
            new_tokens.append(glyph)

    old_size = len(base_tok)
    log(f"tokenizer: adding {len(new_tokens)} pieces to a {old_size}-token vocabulary")
    _t = time.time()
    base_tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    log(f"tokenizer: specials added in {time.time() - _t:.2f}s")
    new_token_start = len(base_tok)          # first ordinary token after the specials
    _t = time.time()
    vocab = base_tok.get_vocab()
    log(f"tokenizer: vocabulary read in {time.time() - _t:.2f}s")
    new_tokens = [t for t in new_tokens if t not in vocab]
    _t = time.time()
    n_new = base_tok.add_tokens(new_tokens, special_tokens=False)
    log(f"tokenizer: added {n_new} tokens in {time.time() - _t:.2f}s")
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_tok.padding_side = "right"

    # Every new token must sit in one contiguous block, because the next cell
    # initialises their embeddings by position.
    assert len(base_tok) == new_token_start + n_new, "new tokens are not contiguous"
    assert all(base_tok.convert_tokens_to_ids(t) >= new_token_start for t in new_tokens)

    log(f"SentencePiece vocab {sp.get_piece_size()}; grafted {n_new} tokens + {len(SPECIAL_TOKENS)} specials")
    log(f"tokenizer size {len(base_tok)} (base was {old_size})")


    def fertility(tok, texts: list[str]) -> dict:
        words = sum(len(t.split()) for t in texts)
        tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
        chars = sum(len(t) for t in texts)
        per_word = [len(tok.encode(t, add_special_tokens=False)) / max(1, len(t.split())) for t in texts]
        return {
            "words": words,
            "tokens": tokens,
            "fertility": tokens / max(1, words),
            "chars_per_token": chars / max(1, tokens),
            "p50_tokens_per_word": float(np.percentile(per_word, 50)),
            "p95_tokens_per_word": float(np.percentile(per_word, 95)),
        }


    # Measure on held-out text only. Training text would flatter any tokenizer.
    eval_texts = [r["text"] for r in lm_test]
    log("tokenizer: measuring fertility on held-out text")
    base_only = AutoTokenizer.from_pretrained(CFG.base_model)
    base_only_metrics = fertility(base_only, eval_texts)
    new_metrics = fertility(base_tok, eval_texts)

    comparison = {
        "base": base_only_metrics,
        "extended": new_metrics,
        "reduction_percent": 100 * (1 - new_metrics["fertility"] / base_only_metrics["fertility"]),
    }
    jdump(PROC / "tokenizer_audit.json", comparison)
    print(json.dumps(comparison, indent=2))

    for probe in ["nêhiyawêwin", "tânisi", "kîkway", "maskwa", "êkwa mîna", to_syllabics("tânisi nêhiyawêwin")]:
        print(f"  {probe:16s} base={base_only.encode(probe, add_special_tokens=False)}")
        print(f"  {'':16s} new ={base_tok.encode(probe, add_special_tokens=False)} -> {base_tok.convert_ids_to_tokens(base_tok.encode(probe, add_special_tokens=False))}")

    base_tok.save_pretrained(PROC / "tokenizer")
    base_only.save_pretrained(PROC / "base_tokenizer")
    shutil.copyfile(str(sp_prefix) + ".model", str(PROC / "cree_spm.model"))
    jdump(PROC / "new_tokens.json", {
        "tokens": new_tokens,
        "specials": SPECIAL_TOKENS,
        "new_token_start": new_token_start,
        "old_size": old_size,
    })

    # ── figure ───────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = ["base\n(Qwen)", "extended\n(+Cree)"]
    vals = [base_only_metrics["fertility"], new_metrics["fertility"]]
    bars = ax[0].bar(labels, vals, color=["#8c3b2e", "#2f5d50"])
    ax[0].set_title("Tokens per Cree word (lower is better)")
    ax[0].set_ylabel("fertility")
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")

    bins = np.linspace(0.5, 6, 34)
    ax[1].hist([len(base_only.encode(t, add_special_tokens=False)) / max(1, len(t.split())) for t in eval_texts],
               bins=bins, alpha=0.65, label="base", color="#8c3b2e", density=True)
    ax[1].hist([len(base_tok.encode(t, add_special_tokens=False)) / max(1, len(t.split())) for t in eval_texts],
               bins=bins, alpha=0.65, label="extended", color="#2f5d50", density=True)
    ax[1].set_title("Tokens per word on held-out Cree")
    ax[1].set_xlabel("tokens / word")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "tokenizer_fertility.png", dpi=160)
    plt.show()

    assert new_metrics["fertility"] < base_only_metrics["fertility"], "tokenizer did not improve fertility"
    assert len(base_tok) > len(base_only), "no tokens were added"
    print(f"\nsaved tokenizer to {PROC / 'tokenizer'}")

    mark_stage("tokenizer", _TOKENIZER_KEY, tokens=len(new_tokens),
               fertility_drop_percent=round(100 * (1 - new_metrics["fertility"] / base_only_metrics["fertility"]), 1))

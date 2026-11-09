# The pipeline for the best Cree chat model this project can build

> For the complete server sequence, including the 20M corpus readiness check,
> staged training, export, install, the 1,000-row benchmark and frontend metrics
> publishing, see [`server-training-runbook.md`](server-training-runbook.md).

`docs/research/indigenous-llm-survey.md` records every paper this pipeline rests
on, what it found, and the change made here because of it, including the ones
that were read and rejected.

This document is the whole pipeline, end to end, with the numbers it has to hit
and the reason for each choice. It replaces "train a model and hope" with a
sequence of stages that each fail loudly. It is written against the repository as
it exists: every stage names the file that implements it, and the stages marked
**missing** are the work that has not been written yet.

Two things are stated up front because everything below depends on them.

**No model answers literally anything perfectly.** Not in English, not in Cree,
not at any parameter count. A pipeline can make a model right on ordinary input,
able to say "the notes do not say", and measurably better after every run. It
cannot make it infallible, and a model that claims infallibility is the failure
mode this project exists to avoid.

**The corpus is the ceiling.** Parameters, layers and schedules move quality by a
few points. Ten times more clean Cree moves it by tens. The current full profile
is 1,124,146 words (about 1.5M tokens); the target for this pipeline is 20M
tokens minimum, 50M to be comfortable, and the only honest way to get there is in
section 3.

---

## 0. What "best" is allowed to mean: the acceptance criteria

The build is done when all of these pass on a held-out set that no stage ever
trains on, measured by `tools/audit_chat_model.py` and the evaluation cells:

| gate | measure | target |
| --- | --- | --- |
| echo | greetings and one-line questions repeated back | 0 of 12 |
| identity | project facts, boundary answers | 3 of 3 exact |
| glossary | words with a dictionary entry | 100% answered from the dictionary, not generated |
| translation | chrF++ on held-out `parallel_test.jsonl` | at least +8 over the untuned base, and 35 or better |
| abstention | questions whose answer is not in the notes | says it does not know, does not invent |
| persona | does it speak as Archer's assistant learning nêhiyawêwin | passes on 8 of 10 probe prompts |
| latency | first token, warm, on the deployment machine | under 2 s on GPU, under 8 s on CPU |
| human | fluent speaker rubric, 4-point fluency, 100 sentences | mean 3.0 or better, no meaning reversals |

"Best in the world" is not a thing I can measure or promise. These eight numbers
are. A run that misses any of them does not ship, which is the only definition of
"best" that survives contact with a language that has 66,000 sentences of open
text.

---

## 1. Architecture: one model, plus tools that cannot hallucinate

Three components, one Ollama tag.

1. **The model** answers conversation, song-note questions from retrieved
   passages, and translation. It is the only thing that writes prose.
2. **Deterministic tools** answer what a model should never be asked to
   remember: dictionary lookups (`chat/cree_glossary.py`), SRO to syllabics
   conversion, and the morphological analyser. These are exact, instant, and
   free. `itwêwina` and GiellaLT's `lang-crk` are the reference implementations
   for Plains Cree morphology ([itwêwina](https://itwewina.altlab.app/),
   [GiellaLT](https://github.com/giellalt/lang-crk)).
3. **Retrieval** puts Archer's notes in front of the model with a header that
   says what they are. The app already does this; the model is instructed to
   answer from them and to say when they do not contain the answer.

The reason for this split is measurable: glossary answers are 100% right because
they never touch the model, and every hour spent trying to make a 7B memorise a
dictionary is an hour not spent on the data that makes it write Cree.

---

## 2. Base model: Qwen3-14B-Instruct, with Hunyuan-MT-7B as the translation specialist

The current pipeline trains Qwen2.5-7B/14B/32B (`training/cree_llm/cells/01_setup.py`,
`_PRESETS`). The 2025-2026 evidence says two things worth acting on.

* **Qwen3 is the strongest open multilingual family for this job.** The
  NiuTrans.LMT study chose Qwen3 explicitly for multilingual capability and
  reports that continued pretraining contributes 5 to 9 COMET points in
  low-resource directions, against 1 to 3 in high-resource ones
  ([ACL 2026](https://aclanthology.org/2026.acl-long.1153.pdf)). That is the
  regime Cree is in: CPT is worth more here than another ten billion parameters.
* **Translation-specific bases beat general ones at the same size on MT.**
  Hunyuan-MT-7B is an open 7B translation model with explicit low-resource work
  ([technical report](https://arxiv.org/pdf/2509.05209v1)) and it ranks with
  Tower-Plus-9B and Aya Expanse 8B among open systems in WMT25/26 findings.

So: **one base for chat, one for translation, both from the same recipe.**

| role | base | licence | why |
| --- | --- | --- | --- |
| chat, notes, persona | `Qwen/Qwen3-14B-Instruct` | Apache-2.0 | strongest open multilingual at 14B, permissive licence, fits 24 GB at 4-bit |
| translation adapter | `Qwen/Qwen3-14B-Instruct` first, `Hunyuan-MT-7B` as a second adapter if chrF++ stalls | Apache-2.0 / Tencent licence | same tokenizer work, comparable routing, one deploy path |
| not chosen | Aya Expanse 8B/32B | CC-BY-NC | excellent MT scores but non-commercial; this project skips NC sources and should not depend on an NC base |

**Missing:** a `chat`/`translate` profile pair in `01_setup.py`, so the
translation adapter can be trained and exported separately while the app keeps
one tag. The app's router already distinguishes the two paths
(`chat/archer_prompt.py` `route`), so this is a training change, not an app one.

Hardware, stated plainly: 14B at QLoRA needs one 24 GB card, merging needs about
64 GB of system RAM, and 32B needs an 80 GB card. This laptop has no GPU at all
(`torch.cuda.is_available()` is False), 3 GB of free RAM and 5.9 GB of free disk,
so training happens elsewhere and only inference happens here.

---

## 3. Data: the only stage that decides whether this works

### 3.1 What exists

`training/cree_llm/cells/03_corpus.py` downloads 13 sources at run time: EdTeKLA,
Plains Cree Figurative, Cree Wikipedia, FineWeb-2, FinePDFs, FineTranslations,
GlotCC, Tatoeba, GlotStoryBook, the phrase recordings, Internet Archive books,
GiellaLT `lang-crk`, and this project's own pairs. `tools/check_cree_sources.py`
proves all 13 endpoints answer; `03_corpus.py` now refuses to continue if any
source contributed nothing.

After cleaning the full profile is 1,124,146 words, 66,071 sentences, 10,934
pairs, 57,579 lexicon forms. That is the number on the site, and it is about 8%
of the 20M-token readiness target.

### 3.2 What to add, in order of yield

1. **Morphological synthesis from the FST (highest yield, no new permissions).**
   GiellaLT `lang-crk` and `itwêwina` are finite-state models: they generate every
   inflected form of a lemma, with the analysis tags that say what it is. For each
   of the 57,579 lexicon forms, generate the paradigm, tag it, and emit two
   training signals: `(analysis -> word form)` for the morphology task, and
   `(English gloss + features -> Cree word form)` for translation of short
   phrases. This turns a dictionary into hundreds of thousands of exact,
   licence-clean examples of the language's hardest part, the verb.
2. **Back-translation of English into Cree.** Train the Cree-to-English
   direction first, translate a large English corpus, keep only pairs where a
   round trip returns the original meaning (round-trip filter), and add them as
   synthetic pairs marked `origin="synthetic"`. This is the standard lever for
   low-resource MT and the reason the Tibetan study could move chrF at all.
3. **Community and archival text, with permission.** EdTeKLA's texts, Cree
   Literacy Network material, and the elders' recordings that are already
   transcribed. Every addition goes into the licence ledger; NC and ND sources
   stay out unless the project is released under the same terms.
4. **Speech transcripts.** 64 phrase transcriptions now; the Maskwacîs speech
   database and any consented recordings are the register learners actually need.

### 3.3 Cleaning and splits (mostly built)

`04_clean.py` already does the language check per sentence, the bag-of-words
rejection, exact and MinHash deduplication, and document-disjoint splits with a
second leakage pass. Keep it. Add three things:

* **Contamination audit:** no held-out sentence, and no 12-gram from a held-out
  sentence, may appear in any training split. This is implemented in
  `04_clean.py`: it builds the 12-gram sets for train and test, raises on any
  overlap, and writes `contamination_12gram_overlap` into `stats.json`.
* **Provenance-weighted sampling:** human text at weight 1.0, synthetic at 0.5,
  dictionary-derived at 0.7. A model trained on a flood of synthetic pairs starts
  writing plausible Cree that no speaker wrote.
* **A held-out set that is never touched:** 500 sentences, modern orthography,
  from sources the model never sees, plus 200 project questions. Every stage
  reports against it.

### 3.4 The gate

```bash
python tools/build_cree_corpus.py --min-tokens 20000000
```

The standalone command fails today at about 1.5M tokens. Training records the
same number as a warning in `raw/corpus_size.json`, logs it, and continues with
every verified token. A run below 20M is a diagnostic model, not a high-end
assistant, so its results must not be presented as one.

---

## 4. Tokenizer: morphology-aware, measured before it is trusted

`05_tokenizer.py` extends the base tokenizer with about 4,000 Cree pieces and
grafts their embeddings. The literature says that is not enough on its own:

* MorphBPE (Findings of ACL 2026) constrains BPE merges so they cannot cross a
  morpheme boundary, leaves inference unchanged, and reports better
  morphological consistency and lower cross-entropy, with the gains concentrated
  in morphologically rich languages. It also states that fertility alone is a bad
  proxy for tokenizer quality.
* The Mapudungun work (AmericasNLP 2026) applies BPE inside morphemes for a
  polysynthetic American language and finds that morphology-aware tokenization
  can substitute for a 5x increase in model parameters in one direction.
* The COLING 2025 study of why models do worse on complex morphology rules out
  fertility and word length as the explanation and points at morphological
  alignment of the tokens.

Those three are why `tools/tokenizer_lab.py` exists and why it is a gate, not an
experiment. It trains standard BPE and morpheme-constrained BPE on this
repository's own Cree and measures fertility, long-word fertility, boundary
precision and boundary recall. Measured on the 6,815 distinct words available in
this checkout, vocabulary 2,000:

| tokenizer | fertility | long-word fertility | boundary precision | boundary recall |
| --- | --- | --- | --- | --- |
| standard BPE | 5.16 | 6.74 | 0.47 | 0.57 |
| morpheme-constrained BPE | 4.81 | 6.25 | 0.35 | 1.00 |

What that says, without dressing it up: constraining the merges guarantees the
tokenizer never splits inside a morpheme (recall 1.00), lowers the token count
per word and lowers it more on the long words that polysynthetic Cree is made of
(6.25 against 6.74), and pays for it with more boundaries inside morphemes
(precision 0.35 against 0.47). That is the same trade the MorphBPE paper reports:
better morphological coherence at a fertility cost that does not track quality.

What it does not say: that the Cree inventory in this lab is the real one. The
segmenter here is a documented prefix and suffix list, not the finite-state
analyser, because `lang-crk` is not installed on this machine. The decision this
buys is a test to run, not a settled answer: the stage gate is boundary recall at
or above 0.95 with precision not below the standard tokenizer's, measured again
with the real FST, and only then does the tokenizer change ship.

Three audits are required before any training: **fertility**, **boundary
alignment** (the table above), and **round-trip**, where SRO to syllabics and back
through the tokenizer must return the same string. A tokenizer that shreds
syllabics makes every downstream number meaningless, and the corpus contains
mixed-script lines that a Latin-only round-trip test would miss.

**Missing:** install the FST and rerun the lab with real analyses; cover syllabics
and mixed-script lines in the round-trip test.

---

## 5. Training: four stages, each with a gate

| stage | what it does | cell | gate to pass before the next stage |
| --- | --- | --- | --- |
| 1. CPT | Cree prose, 10% English replay, 3 epochs, seq 2048, LoRA r=64 on all linear layers | `07_cpt.py` | held-out perplexity falls; English instruction-following does not collapse (probe 20 English prompts) |
| 2. SFT | translation, orthography, notes, abstention, persona, conversation, 2 epochs, only assistant tokens in the loss | `09_sft.py` | echo 0/12, identity 3/3, persona 8/10 |
| 3. DPO | preference pairs: good answer vs. echo, invented answer, wrong dialect, English fallback | `10_dpo.py` | chrF++ up, echo still 0, abstention precision up |
| 4. Export | merge, GGUF, Q4_K_M or Q5_K_M, Modelfile, `ollama create` | `13_export.py` | the whole audit table in section 0 |

Notes that matter, from the literature and from this repository's own failures:

* **LoRA is not the safe option, it is the cheap one.** A 2025 study of
  catastrophic forgetting found LoRA and full fine-tuning produce comparable
  forgetting when the adapters cover the same parameters
  ([MRL 2025](https://aclanthology.org/2025.mrl-main.23.pdf)). Forgetting is
  controlled by replay and data composition, not by freezing weights.
* **Echoing is memorisation plus a copying mechanism, and temperature does not
  fix it.** The 2026 analysis of prompt echoing in small instruct models shows it
  occurs across models and survives high-temperature decoding
  ([EMNLP findings](https://aclanthology.org/2026.findings-emnlp.1633.pdf)). This
  matches what was measured here: 3 of 4 greetings echoed at 0.4, 0.6 and 0.7.
  The fixes are data diversity, negative preference pairs against repetition
  (the ParaPO idea, [arXiv 2504.14452](https://arxiv.org/pdf/2504.14452)), and an
  explicit abstention target. That is why stage 3 exists and why its negatives
  include the echo itself.
* **Data purity beats volume.** The LoResMT 2026 Tatar result is blunt: for
  low-resource languages, "quality and purity of the training data are more
  important than data volume"
  ([LoResMT 2026](https://aclanthology.org/2026.loresmt-1.16.pdf)). A corpus full
  of OCR noise and English lines trains a model to produce OCR noise and English.
* **Instruction diversity is not optional.** `08_sft_data.py` now generates 42
  casual-conversation rows from parts, 14 openings crossed with 6 replies, so no
  single phrasing dominates the model's voice. This is the same lesson as the
  data-diversity work in instruction tuning
  ([EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.195.pdf)).

---

## 6. Evaluation: chrF++, COMET where it is trustworthy, and people

Low-resource translation is where BLEU lies. WMT's own findings use chrF++ alone
for the lowest-resource pairs because learned metrics degrade there
([WMT25 ranking](https://arxiv.org/html/2508.14909v2)). So:

* **chrF++ is the primary automatic metric** (`sacrebleu`), reported per
  direction, on the 500-sentence held-out set. Target: +8 over base, 35 or better.
* **COMET/XCOMET only as a secondary signal**, and never as the sole gate below
  about 5M tokens of training data, where it is noisy.
* **Echo rate, abstention and glossary accuracy** from
  `tools/audit_chat_model.py`, which runs against the installed tag through the
  app's own prompt builder.
* **Human evaluation by fluent speakers**, 100 sentences, 4-point fluency plus a
  meaning-reversal flag. This is the only metric that catches a model writing
  grammatical Cree that says the wrong thing, which is the failure that matters
  for a living artist's work.

---

## 7. Serving on this machine

* 14B at Q4_K_M is about 9 GB, so it needs a machine with 16 GB of free RAM; this
  laptop currently has 3 GB free with the app running, so either the 7B stays or
  the machine gets more memory.
* Decoding: min-p 0.05 instead of top-p, with `top_p 1.0` and `top_k 0` sent to
  disable the other two cuts, plus DRY 0.8 against repeated spans, temperature
  0.3 to 0.7, `repeat_penalty` 1.08, `num_ctx` 4096, `num_thread` 5 (measured: 16
  threads gave 0.3 to 3.5 tokens a second, 5 gave about 5.3). The min-p change
  was A/B measured on the installed tag: echo 6/12 to 4/12, translation overlap
  0.10 to 0.25, mean reply 3.8s to 2.0s (`tools/decoding_ab.py`).
* `keep_alive` 1h and a warm-up at startup, so the first message is not a 4.7 GB
  load.
* Streaming with the same token-level filters the app already applies, including
  the dash rule and the space-preserving stream filter.

---

## 8. Runbook

```bash
# 1. prove every source is reachable
python tools/check_cree_sources.py

# 2. build the corpus and record readiness; training will still use every token
python tools/build_cree_corpus.py --min-tokens 20000000 || true
python -c "import json; print(json.load(open('data/processed/cree_corpus.jsonl')))" 2>/dev/null || true

# 2b. bring the corrections with the run. The app writes these when a visitor
#     says an answer was wrong. The SFT stage reads feedback_sft.jsonl and the
#     DPO stage reads feedback_dpo.jsonl; without this copy they are absent and
#     the run prints that they were not found.
export ARCHER_FEEDBACK_DIR=/path/to/the/repository/data/training
python tools/verify_training_pipeline.py

# 3. train on the GPU box, in order, each stage gated
python train_cree_llm.py --profile big --work /data/cree_llm     # 14B

# 4. install the result and audit it against the numbers in section 0
cd assets/archer_llm && ollama create archer:14b -f Modelfile

# 5. switch the app to the new tag only after the audit passes
#    chat:
#      model: archer:14b
python tools/audit_chat_model.py
```

Every stage is fingerprinted in the run ledger, so an interrupted or partially
downloaded run cannot be mistaken for a finished one; `03_corpus.py` now stops on
a missing source rather than training on the remainder.

---

## 9. What this will and will not produce

With 20M tokens, FST-derived paradigms, back-translation and the four stages
above, the realistic outcome is a model that: holds a short conversation without
echoing, answers project and note questions from evidence, translates ordinary
sentences of the kind in the corpus at chrF++ in the 30s to 40s, says it does not
know when it is out of its depth, and still makes morphological mistakes on
uncommon verb forms.

Without the corpus work, it will be the model that is installed today: fluent in
English, uncertain in Cree, and able to repeat a greeting back to you.

That is the whole pipeline. The parts that exist are marked with their files. The
remaining research stages are the profile split, FST morphology synthesis and the
back-translation loop; the contamination audit is now built into `04_clean.py`.
Those three need resources this machine does not have: a GPU, the `lang-crk` FST,
and a translation model for the round-trip filter.

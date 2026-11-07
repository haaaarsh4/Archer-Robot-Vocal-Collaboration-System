# Every stage of this LLM, what the literature says, and what changed here

This is the second research record. `docs/research/indigenous-llm-survey.md`
covers Indigenous-language work and tokenization. This one covers the rest of the
pipeline, stage by stage: data, continued pretraining, SFT, preference tuning,
retrieval, decoding, quantisation and evaluation. Each entry says what the paper
found, whether it applies to this project, and what was changed because of it.

Where a recommendation was rejected, the reason is recorded. A pipeline that
claims to follow every paper is lying; a pipeline that says which ones it follows
and why can be checked.

---

## 1. Data curation and ordering

| paper | finding | applied here |
| --- | --- | --- |
| **D4: Improving LLM Pretraining via Document De-Duplication and Diversification**, NeurIPS 2023 ([link](https://arxiv.org/abs/2308.12284)) | Deduplication plus embedding-based selection gave about 20% training efficiency and up to 2% downstream accuracy at 6.7B. In the data-limited regime, repeating a *selected* subset beats one pass over everything, while repeating random data does worse than baseline. | This project is data-limited by definition. The corpus gate and provenance weighting are the selection step; `04_clean.py` already does exact and MinHash dedup. Recorded as the justification for the 3 CPT epochs over a curated set instead of 1 epoch over everything. |
| **Data, Data Everywhere: A Guide for Pretraining Dataset Construction**, EMNLP 2024 ([link](https://aclanthology.org/2024.emnlp-main.596.pdf)) | Deduplicated, quality-filtered data improves accuracy; 95% fuzzy-dedup selection is optimal; older sources are worth prioritising when deduplicating. | Cleaning already exists; the survey is the reason the synthetic and dictionary-derived sources are weighted below human text rather than counted equally. |
| **Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics**, arXiv 2601.21698 ([link](https://arxiv.org/abs/2601.21698)) | Curricula change time spent in latent phases; random ordering gives noisier gradients in small models; reversing the order loses the gain. | Not applied: this stage needs enough data for phases to matter, and 20M tokens is not that. Noted so it is not retried from scratch every run. |
| **How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining**, arXiv 2511.18903 ([link](https://arxiv.org/abs/2511.18903)) | Curriculum sorting only helps with a moderate decay schedule; under standard decay the advantage disappears, and a moderate decay plus curriculum gave 1.64% over random shuffling. | Applied as a rule for the run book: if quality-sorted data is ever used, the LR must not decay to near zero. Documented in the pipeline doc. |
| **Seamless Data Packing**, Findings of ACL 2025 ([link](https://aclanthology.org/2025.findings-acl.777.pdf)) | Seamless packing keeps documents contiguous and improves continual pretraining over naive concatenation. | `07_cpt.py` already packs with an EOS between documents rather than splitting mid-sentence; kept, with the paper cited. |
| **Emergent Abilities under Continued Pretraining for Language Adaptation**, ACL 2025 ([link](https://aclanthology.org/2025.acl-long.1547.pdf)) | English replay at 20% does not help target-language modelling itself; curriculum and EMA can replace it. | Rejected the 20% figure and the removal of English. Replay is kept at about 10% because the goal here is an assistant that still follows English instructions, not a pure Cree LM. The trade is stated in the pipeline doc. |

## 2. Supervised fine-tuning

| paper | finding | applied here |
| --- | --- | --- |
| **Fine-tuning LLMs with Limited Data: A Survey and Practical Guide**, TACL 2026 ([link](https://aclanthology.org/2026.tacl-1.17.pdf)) | Masking prompt tokens is the dominant practice. A small loss weight of 0.01 to 0.1 on prompt tokens helps short outputs. 2 to 3 epochs usually suffice, up to 20 to 25 in extreme low-data with early stopping. Diminishing returns past a few thousand samples; selection and coverage beat volume. | Epochs stay at 2 (`CFG.sft_epochs`), and the "a few thousand samples" ceiling is why the SFT set is built for coverage rather than size. Prompt-token weighting was **rejected**: in this dataset the prompt carries long note passages, so scoring them would train the model to write the notes rather than the answer. |
| **Packing Analysis**, Findings of ACL 2025 ([link](https://aclanthology.org/2025.findings-acl.256.pdf)) | Packing is not automatically faster or better for small models on small datasets; padding can be more time-efficient at 20k to 30k examples. | Not changed: `ChatDataset` uses dynamic padding, not packing, which the paper says is the right choice at this size. The decision is now recorded instead of accidental. |
| **Data Diversity Matters for Robust Instruction Tuning**, Findings of EMNLP 2024 ([link](https://aclanthology.org/2024.findings-emnlp.195.pdf)) | Diversity and quality trade off; quality-diversity selection beats quality alone. | `08_sft_data.py` generates 42 casual rows from parts rather than 13 fixed greetings, so no single phrasing dominates. |
| **The Best Instruction-Tuning Data are Those That Fit**, arXiv 2502.04194 ([link](https://arxiv.org/abs/2502.04194)) | Selecting responses the base model already finds likely, then fine-tuning on them, beats training on several times more data. | Applied as a rule for the next SFT run: score candidate answers under the CPT adapter and keep the ones it finds likely, rather than adding more rows. Not automated yet; recorded as the next data step. |
| **Data Repetition Beats Data Scaling in Long-CoT SFT**, arXiv 2602.11149 ([link](https://arxiv.org/abs/2602.11149)) | Under a fixed update budget, more epochs on fewer samples beat one pass over more, without extra forgetting in the reasoning setting. | Read, not applied: the effect is demonstrated for long chain-of-thought data, which this project does not train. Noted so the 2-epoch setting is a choice, not ignorance. |

## 3. Preference tuning

| paper | finding | applied here |
| --- | --- | --- |
| **ORPO**, EMNLP 2024 ([link](https://aclanthology.org/2024.emnlp-main.626.pdf)) | A fused SFT plus preference objective, no reference model, preferred over SFT and PPO in their evaluation. | Rejected: this pipeline already has a clean SFT checkpoint and a frozen reference is cheap with adapters disabled, which is what `10_dpo.py` does. ORPO would merge two stages that are usefully separable when data changes. |
| **DPO length bias** ([link](https://aclanthology.org/2024.findings-acl.297.pdf)) and **SamPO** ([link](https://aclanthology.org/2024.emnlp-main.60.pdf)) | DPO can learn to prefer longer answers; smaller beta produces longer responses; length-normalised or down-sampled variants reduce it. | Checked and already safe: `CFG.dpo_beta = 0.1` and `CFG.dpo_lr = 1e-6` are the values the literature recommends, and the negatives in `10_dpo.py` are content corruptions of equal length (dropped clause, flattened vowel, leaked English word, wrong numeral), not shorter answers. A length bias cannot form from pairs built that way. |
| **Small-model preference comparison (DPO/IPO/KTO)**, 2026 ([link](https://github.com/zlznzyq/dpo-small-models)) | DPO with beta 0.05 to 0.1 beat the untuned base on a 0.6B model; no method beat an already well-aligned 1.7B base, and gains shrank with scale. | Read as a warning: preference tuning fixes specific behaviours (echo, invention), it does not add capability. That is exactly how `10_dpo.py` uses it. |
| **Prompt echoing in small instruct models**, Findings EMNLP 2026 ([link](https://aclanthology.org/2026.findings-emnlp.1633.pdf)); **ParaPO**, arXiv 2504.14452 ([link](https://arxiv.org/pdf/2504.14452)) | Echoing is memorisation plus a copy mechanism and survives temperature; preference training against verbatim reproduction reduces it. | The echo class is now a required negative in the DPO stage, and the audit fails a run that echoes. |

## 4. Retrieval

| paper | finding | applied here |
| --- | --- | --- |
| **A Systematic Analysis of Chunking Strategies for Reliable QA**, arXiv 2601.14123 ([link](https://arxiv.org/pdf/2601.14123)) | Chunking choices matter; overlap gave no measurable benefit; smaller context concentrates evidence, larger context raises recall. | The notes are already small documents retrieved whole with `top_k=4` and a score floor, which is the "small context, concentrated evidence" end of that finding. No overlap is introduced. |
| **Hybrid sparse-dense retrieval** at SemEval 2026 ([link](https://aclanthology.org/2026.semeval-1.320.pdf), [link](https://aclanthology.org/2026.semeval-1.345.pdf)) | BM25 plus dense, fused and reranked, beats either alone; unanswerable turns need an explicit answerability decision. | Applied in the smallest useful form: the existing score floor is the answerability gate, and the prompt requires the model to say the notes do not say. A cross-encoder reranker is **rejected for now**: it is another model in the latency path for four short notes. |
| **Prompt-Based Abstention Fails Under Misleading Context**, arXiv 2608.22228 ([link](https://arxiv.org/html/2608.22228)) | Small frozen RAG models abstain when context is missing but not when it is misleading and locally plausible. | This is why abstention is trained (DPO pairs with invented note answers) and audited, not just prompted. |
| **On the Influence of Context Size and Model Choice in RAG**, Findings NAACL 2025 ([link](https://aclanthology.org/2025.findings-naacl.375.pdf)) | More retrieved context saturates and can distract; model choice dominates. | `top_k=4` stays. Raising it was considered and rejected on this evidence. |

## 5. Decoding and quantisation

| paper | finding | applied here |
| --- | --- | --- |
| **Min-p Sampling**, ICLR 2025 ([link](https://arxiv.org/abs/2407.01082)) | Dynamic truncation scaled by the top token's probability improves quality and diversity over top-p, especially at higher temperature, from 1B to 123B, and humans preferred it. | **Applied and measured**: `min_p 0.05` is sent on every request, with `top_p 1.0` and `top_k 0`, the values that disable the other two. That detail is the whole point: Ollama applies all three filters in sequence, so leaving its `top_p 0.9` / `top_k 40` defaults in place would let them cut the tail first and min-p would look like it did nothing. A/B on the installed tag, same 12 greetings and 8 held-out translation pairs through both samplers: echoes 6/12 to 4/12, mean translation overlap 0.10 to 0.25, mean reply time 3.8s to 2.0s (`tools/decoding_ab.py`). |
| **DRY: stopping verbatim loops at sampling time**, 2026 ([link](https://arxiv.org/html/2608.22761v1)) | Penalising continuation of an already-seen suffix cuts repetition loops by about half without losing benchmark quality, and composes with temperature and repetition penalty. | **Applied, effect not yet observed**: `dry_multiplier 0.8`, `dry_base 1.75` and `dry_allowed_length 2` are sent on every request and the installed Ollama accepts them. But no reply in the A/B sample looped, so the repeated-span metric was 0.00 under both samplers and the penalty had nothing to charge for. It stays because it costs nothing at this length and it targets the failure this project actually produced (`_reliable_cree` exists because one reached the panel), but the honest status is expected gain, not measured gain. |
| **The Good, The Bad, and The Greedy**, NAACL 2025 ([link](https://aclanthology.org/2025.naacl-long.211.pdf)) | Greedy and sampled decoding can differ by more than 10 points on some benchmarks, and repetition penalty interacts strongly with the task. | The audit now reports the temperature it used and reads it from `config.yaml`, so a result is never quoted without the decoding it came from. |
| **Which Quantization Should I Use?**, 2026 ([link](https://arxiv.org/pdf/2601.14277)) and GGUF K-quant guidance ([link](https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i)) | Q4_K_M is the general 4-bit default; Q5_K_M is closer to F16 at a modest size cost; Q8_0 is near-lossless. | Recorded as the export rule: Q5_K_M for the 7B when memory allows, Q4_K_M when it does not. The export cell's choice is documented rather than implicit. |

## 6. Evaluation

| paper | finding | applied here |
| --- | --- | --- |
| **WMT25 ranking** ([link](https://arxiv.org/html/2508.14909v2)) | chrF++ alone is used for the lowest-resource pairs. | chrF++ is the primary gate; COMET secondary. |
| **chrF++ vs BLEU for extremely low-resource MT** ([link](https://arxiv.org/html/2602.17425v1)) | BLEU misrepresents quality at this scale; chrF++ is more stable. | `tools/audit_chat_model.py` scores token overlap and the pipeline gate is chrF++; BLEU is not used. |
| **IndigiEval**, AmericasNLP 2026 ([link](https://aclanthology.org/2026.americasnlp-6.8.pdf)) | No commercial LLM performs well across categories; they hallucinate orthography, grammar and culture. | Human and deterministic checks stay in the acceptance table; a single automatic score cannot pass a run. |

---

## What was changed in code this round

| change | file | evidence |
| --- | --- | --- |
| min-p sampling on every request | `chat/llm_client.py`, `config/config.yaml` | Min-p, ICLR 2025 |
| DRY loop suppression on every request | `chat/llm_client.py`, `config/config.yaml` | DRY, 2026 |
| decoding settings exposed in config with the papers in the comments | `config/config.yaml` | Min-p, DRY, NAACL 2025 non-determinism |
| audit reads temperature from config and reports it | `tools/audit_chat_model.py` | NAACL 2025 non-determinism |
| A/B harness for the sampler, so a decoding claim is a number | `tools/decoding_ab.py` | Min-p, DRY |
| morphology-aware tokenizer measured and gated | `tools/tokenizer_lab.py` | MorphBPE, Mapudungun |
| corpus completeness and size gates | `03_corpus.py`, `train_cree_llm.py`, `tools/build_cree_corpus.py` | D4, Data Data Everywhere |
| 42 varied conversation rows instead of 13 greetings | `training/cree_llm/cells/08_sft_data.py` | Data Diversity, EMNLP 2024 |

## What the decoding change actually measured

Run on the installed `archer:7b`, the same prompts in the same order, on the
same warm weights, so the only difference is the sampler. `old` is what the app
sent before this round: temperature only, Ollama's `top_p 0.9` and `top_k 40`
doing the truncation, no loop penalty. `new` is min-p 0.05 with those two
disabled, plus DRY 0.8.

| measure | old | new |
| --- | --- | --- |
| greetings echoed verbatim | 6 / 12 | 4 / 12 |
| mean repeated-span rate | 0.00 | 0.00 |
| mean translation token overlap (8 held-out pairs) | 0.10 | 0.25 |
| exact-match translations | 0 / 8 | 2 / 8 |
| mean reply time | 3.8 s | 2.0 s |

The wins are real and small: `hi` stopped coming back as `hi`, and two of the
eight held-out sentences became exact (`go and eat !` to `nitawi mîcisok !`,
`did you buy shoes ?` to `kikîhatâwân cî maskisina ?`) where the old sampler got
them partly or not at all. Replies also got shorter and faster, which is what a
min-p cut at 0.05 does: it removes the low-probability tail that was being spent
on trailing clauses.

Two honest caveats, because a single run is not a measurement:

* A second run of `tools/audit_chat_model.py` on the new sampler alone gave
  4/12 echoes again but overlap 0.15, not 0.25. At temperature 0.3 the sampler is
  not deterministic, so the claim is the direction (0.10 before, 0.15 to 0.25
  after), not the exact point estimate. Any future comparison should average
  three runs before quoting a number.
* That audit still **fails its own gate**, because it requires translation
  overlap of 0.5 and this tag scores 0.15 to 0.25. That failure predates this
  change and is the data problem (a 7B model with the corpus the export used),
  not a sampler regression: the old sampler scored lower on the same pairs. It is
  in the record as a failure so the passing checks cannot be read as a pass for
  the model as a whole.

The remaining 4 of 12 are `hey`, `howdy`, `hey there` and `good morning`, and
that is the point worth keeping in the record. Decoding cannot fix them, because
the model is not confused about these greetings, it is confident that the answer
to `hey` is `hey`. That is memorisation of an SFT target, it is the same cause
the audit already identified, and the fix is the retrain, not another sampler
setting. A pipeline that reported this change as a fix for the echo would be
reporting the 2 greetings it repaired and hiding the 4 it did not.

## What was read and deliberately not applied

* Curriculum pretraining: needs more data than this project has, and its benefit
  depends on a moderate LR decay that the run does not use.
* Prompt-token loss weighting: the prompts here carry long retrieved passages,
  so scoring them trains note writing, not answering.
* ORPO as a replacement for SFT plus DPO: loses the ability to retrain one stage
  when only one kind of data changes.
* Packing for SFT: the paper finds padding more efficient at this dataset size,
  which is what the code already does.
* A cross-encoder reranker: a second model in the latency path for four short
  notes, with no measured retrieval failure to justify it.
* 20% English replay and its removal: both ends were considered; 10% is kept
  deliberately to protect instruction following.

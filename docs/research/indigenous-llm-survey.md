# What the literature says, and what changed in this pipeline because of it

Every entry here is a paper that was read for this pipeline, what it found, and
the specific change made to this repository. Where a paper's recommendation was
rejected, the reason is written down. Numbers are the papers' own.

## 1. Tokenization for polysynthetic and morphologically rich languages

| paper | finding | change made here |
| --- | --- | --- |
| **MorphBPE: Morphology-Aware Tokenization for Efficient LLM Training**, Findings of ACL 2026 ([link](https://aclanthology.org/2026.findings-acl.2068.pdf)) | Constraining BPE merges so they cannot cross morpheme boundaries keeps inference unchanged, improves morphological consistency F1, lowers cross-entropy, and gains are largest in morphologically rich languages. Fertility alone is an insufficient proxy. | `tools/tokenizer_lab.py` measures boundary precision and recall, not just fertility; the tokenizer stage is screened with it before any run. |
| **Bringing Mapudungun into the Modern MT Ecosystem: Morphology-Aware Tokenization for NLLB-200 Fine-Tuning**, AmericasNLP 2026 ([link](https://aclanthology.org/2026.americasnlp-6.16.pdf)) | For a polysynthetic American language, morphology-aware tokenization substituted for a 5x increase in model parameters in one direction; standard BPE was the weakest condition at 600M. | The pipeline doc now requires a morphology-aware tokenizer as a stage gate, not an optional experiment, and names the FST as the boundary source. |
| **Why do language models perform worse for morphologically complex languages?**, COLING 2025 ([link](https://aclanthology.org/2025.coling-main.441.pdf)) | The performance gap for agglutinative languages is not explained by fertility or word length; morphological alignment of tokens matters. | Confirms the choice above; the acceptance table now includes a morphological alignment measure. |
| **Exploring morphology-aware tokenization: a case study on Spanish language modeling**, EMNLP 2025 ([link](https://aclanthology.org/2025.emnlp-main.1552.pdf)) | A learned segmentation model generalises imperfectly and can raise fertility while improving morphological quality. | Reason to use the FST rather than a learned segmenter for Cree: the FST is exact, licensed, and already exists. |
| **Predictive Text for Agglutinative and Polysynthetic Languages**, Field Matters 2022 ([link](https://aclanthology.org/2022.fieldmatters-1.9/)) | For K'iche' and Chukchi, models trained on morphologically segmented data had the best word and character perplexities. | Supports segmenting the CPT corpus as well as the tokenizer training text. |
| **Surface-Level Morphological Segmentation of Low-resource Inuktitut Using Pre-trained LLMs**, NoDaLiDa 2025 ([link](https://aclanthology.org/2025.nodalida-1.69.pdf)) | A Glot500-m model fine-tuned for boundary classification beats earlier segmentation for a polysynthetic Canadian language in syllabics. | Rejected for now: it needs a trained model before it can help train a model. Listed as the fallback if the FST cannot be installed. |

### Measured on this repository, not taken on faith

`tools/tokenizer_lab.py` trains both tokenizers on the Cree in this checkout and
writes `data/processed/tokenizer_lab.json`. Vocabulary 2,000, 6,815 distinct
words:

| tokenizer | fertility | long-word fertility | boundary precision | boundary recall |
| --- | --- | --- | --- | --- |
| standard BPE | 5.16 | 6.74 | 0.47 | 0.57 |
| morpheme-constrained BPE | 4.81 | 6.25 | 0.35 | 1.00 |

The constraint buys full boundary recall and fewer tokens on the long words, and
costs precision inside morphemes. That matches MorphBPE's reported trade, and it
is not enough on its own to switch the pipeline: the segmenter used here is a
prefix and suffix inventory, not the FST, and the corpus is 6,815 distinct words.
The gate is stated in the pipeline doc: recall at or above 0.95 with precision
not below the standard tokenizer's, measured with `lang-crk` installed.

## 2. Indigenous language models and what actually moved the numbers

| paper | finding | change made here |
| --- | --- | --- |
| **IndigiEval**, AmericasNLP 2026 ([link](https://aclanthology.org/2026.americasnlp-6.8.pdf)) | No commercial LLM performs well across five North American Indigenous languages; they hallucinate orthography, grammar and cultural knowledge. | Justifies the deterministic dictionary and morphology path and the abstention gate as first-class parts of the architecture, not fallbacks. |
| **A hybrid approach to low-resource machine translation for Ojibwe verbs**, AmericasNLP 2025 ([link](https://aclanthology.org/2025.americasnlp-1.3.pdf)) | An LLM builds templates from dictionary entries, an FST analyses the verb, a rule-based module translates, at about 70 words per minute, and it beats pure neural for complex verbs. | The pipeline now names a hybrid path for verbs: FST analysis first, model only for what the rules cannot cover. This is the strongest evidence for the split architecture. |
| **Interactive Word Completion for Plains Cree**, ACL 2022 ([link](https://aclanthology.org/2022.acl-long.232.pdf)) | The Cree FST covers 21,232 stems, 13,860 of them verbs, with corpus-validated prefixes. | The morphology synthesis stage is sized against these numbers; it is the source of the paradigm expansion. |
| **Word-level prediction in Plains Cree**, AmericasNLP 2024 ([link](https://aclanthology.org/2024.americasnlp-1.3/)) | A BiLSTM on a small Cree corpus handles word suggestion, but prefixing polysynthesis makes the vocabulary explode. | Reason to constrain the tokenizer rather than grow the vocabulary: the vocabulary is already the problem. |
| **Analyzing and generating English phrases with finite-state methods for Plains Cree**, AmericasNLP 2025 ([link](https://aclanthology.org/2025.americasnlp-1.6.pdf)) | Bidirectional English phrase to Cree word-form transducers exist and are integrated into itwêwina; the generator is about 30 MB. | This is the exact tool to ship as the deterministic translation path for short phrases, which is now in the architecture section. |
| **Low-Resource Methods for Hawaiian Machine Translation**, ComputEL 2026 ([link](https://aclanthology.org/2026.computel-1.11.pdf)) | mBART with added language tokens reaches chrF++ 43.6; BLEU and chrF disagree strongly at this scale. | chrF++ stays the primary metric; COMET is secondary. |
| **AmericasNLP 2025 shared task systems** ([link](https://preview.aclanthology.org/stats/events/americasnlp-2025)) | NLLB-600M and Llama 3.1 8B fine-tunes reached chrF++ 35.16 (Awajun) and 31.01 (Quechua), and failed on the most polysynthetic languages, Nahuatl and Otomí. | Sets the realistic band for Cree: 30s chrF++ is a good result, not a failure, and the gate in the pipeline is 35. |
| **The Indigenous Languages Technology project at NRC Canada**, COLING 2020 ([link](https://aclanthology.org/2020.coling-main.516.pdf)) | 1.3M aligned Inuktut pairs under CC-BY-4.0 made MT possible for a polysynthetic language; community-led data collection is what produced it. | Data acquisition stays the top of the pipeline, and the community-upload path already in `03_corpus.py` is promoted to a first-class source. |
| **Challenges of language technologies for the indigenous languages of the Americas**, ACL 2018 ([link](https://aclanthology.org/C18-1006.pdf)) | FST morphology is the dependable technology for these languages when parallel data is scarce. | Keeps the FST at the centre instead of treating it as preprocessing. |

## 3. Low-resource adaptation and forgetting

| paper | finding | change made here |
| --- | --- | --- |
| **Adapting LLMs to Low-Resource Tibetan: two-stage CPT and SFT**, arXiv 2512.03976 ([link](https://arxiv.org/abs/2512.03976)) | Qwen2.5-3B, CPT then SFT: perplexity 2.98 to 1.54, chrF 2.2 to 6.6 on Chinese to Tibetan. Adaptation concentrates in embeddings and output heads. | Confirms CPT then SFT for Cree, and why the tokenizer and embedding grafting stage matters more than depth. |
| **Conditions for Catastrophic Forgetting in Multilingual Translation**, MRL 2025 ([link](https://aclanthology.org/2025.mrl-main.23.pdf)) | LoRA and full fine-tuning forget comparably when adapters cover the same parameters; forgetting is driven by data composition and scale. | Replay stays at 10% and the gate checks English instruction-following, instead of assuming LoRA is safe. |
| **NiuTrans.LMT**, ACL 2026 ([link](https://aclanthology.org/2026.acl-long.1153.pdf)) | Qwen3 chosen for multilingual strength; CPT contributes 5 to 9 COMET in low-resource directions against 1 to 3 in high-resource. | Base model switched to Qwen3-14B in the pipeline doc, with the reason recorded. |
| **LoResMT 2026 English-Tatar** ([link](https://aclanthology.org/2026.loresmt-1.16.pdf)) | For low-resource languages, data quality and purity matter more than volume. | Provenance weighting and the purity filters are required, and synthetic data is down-weighted to 0.5. |
| **Prompt echoing in small instruct models**, Findings of EMNLP 2026 ([link](https://aclanthology.org/2026.findings-emnlp.1633.pdf)) | Echoing is memorisation plus an intrinsic copy mechanism, occurs across models, and survives high-temperature decoding. | Explains the measured 3 of 4 echoed greetings at 0.4, 0.6 and 0.7 here; echo becomes a DPO negative class and a hard gate. |
| **ParaPO**, arXiv 2504.14452 ([link](https://arxiv.org/pdf/2504.14452)) | Preference optimisation against verbatim reproduction reduces regurgitation without losing the ability to quote when asked. | The DPO stage's negative set now includes the prompt echoed back as the rejected answer. |
| **Data Diversity Matters for Robust Instruction Tuning**, Findings of EMNLP 2024 ([link](https://aclanthology.org/2024.findings-emnlp.195.pdf)) | Diversity and quality trade off; diversity at fixed quality improves robustness. | `08_sft_data.py` generates 42 casual rows from parts instead of 13 fixed greetings. |

## 4. Evaluation

| paper | finding | change made here |
| --- | --- | --- |
| **WMT25 preliminary ranking** ([link](https://arxiv.org/html/2508.14909v2)) | For the lowest-resource pairs the ranking uses chrF++ alone, because learned metrics degrade there. | chrF++ is the primary gate; COMET secondary and not a sole gate under 5M tokens. |
| **Comparing ChrF++ and BLEU for extremely low-resource MT**, arXiv 2602.17425 ([link](https://arxiv.org/html/2602.17425v1)) | BLEU misrepresents quality at this scale; chrF++ is more stable but can flatter closely related languages. | Cree is unrelated to English, so the flattering case does not apply; chrF++ stays. |
| **Measuring Linguistic Competence of LLMs on Indigenous Languages of the Americas**, EACL 2026 short ([link](https://aclanthology.org/2026.eacl-short.21/)) | Many model-language pairs are near chance on Indigenous languages; zero and few-shot prompting does not fix it. | Prompt-only improvements are not counted as progress; the gates are measured on the fine-tuned tag. |

## 5. What was rejected, and why

* **A Morfessor-based segmenter.** It needs a trained unsupervised model, and the
  FST is exact and already licensed. Kept as a fallback only if `lang-crk` cannot
  be installed.
* **A learned boundary classifier (the Inuktitut route).** It is the right tool
  when no FST exists. Cree has one.
* **Aya Expanse as the base.** Strong MT scores, non-commercial licence, and this
  project already refuses NC sources.
* **BLEU as a gate.** It disagrees with chrF++ at this data scale and WMT itself
  drops it for the lowest-resource pairs.
* **Bigger models as a substitute for data.** The Mapudungun result is explicit:
  the right tokenizer matched a 5x parameter increase. Parameters are the
  expensive way to buy what morphology-awareness gives for free.

## 6. What the papers changed, in file order

| change | file | paper that caused it |
| --- | --- | --- |
| morphology-aware tokenizer becomes a gated stage, plus a lab that measures boundary precision and recall | `tools/tokenizer_lab.py`, `docs/pipeline-best-chatbot.md` | MorphBPE (ACL 2026), Mapudungun (AmericasNLP 2026), COLING 2025 |
| base model moved to Qwen3-14B, CPT valued above parameters | `docs/pipeline-best-chatbot.md` section 2 | NiuTrans.LMT (ACL 2026), Tibetan CPT study (arXiv 2512.03976) |
| FST morphology synthesis promoted to the top data source, and a hybrid rule path for verbs | `docs/pipeline-best-chatbot.md` sections 1 and 3 | Ojibwe hybrid MT (AmericasNLP 2025), Word Completion (ACL 2022), FST phrase transducers (AmericasNLP 2025) |
| echo becomes a DPO negative class and a hard gate, with the measured echo rates recorded | `docs/pipeline-best-chatbot.md` section 5, `tools/audit_chat_model.py` | Prompt echoing (EMNLP 2026), ParaPO (arXiv 2504.14452) |
| chrF++ is the primary gate and COMET is secondary | `tools/audit_chat_model.py`, pipeline doc section 6 | WMT25 ranking, chrF++ vs BLEU study |
| provenance weighting, purity over volume, synthetic data down-weighted | pipeline doc section 3 | LoResMT 2026 Tatar |
| abstention and deterministic dictionary path are first-class | `chat/cree_glossary.py`, pipeline doc section 1 | IndigiEval (AmericasNLP 2026) |
| data diversity in the conversation set | `training/cree_llm/cells/08_sft_data.py` | Data Diversity Matters (EMNLP 2024) |

## 7. The one thing no paper promises

No paper reviewed here reports a model that answers arbitrary questions in a
low-resource Indigenous language without error. The best published results on
comparable languages sit in the 30s on chrF++ and rely on hybrid rule-based or
retrieval components for correctness. That is what this pipeline targets, and it
is why the acceptance table in `docs/pipeline-best-chatbot.md` has eight numbers
instead of the word "perfect".

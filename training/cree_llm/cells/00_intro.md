# The Archer model

This notebook trains the one assistant model the app runs. It is not a Cree model sitting beside a general one: the same weights hold the conversation, read the retrieved song notes, translate between English and nêhiyawêwin, explain words, and convert between SRO and syllabics. The corpus is what makes that possible, and the instruction set is what teaches the model which job it is doing.

It is written to run on a Colab GPU from top to bottom. Nothing is left as an exercise: it downloads every corpus, cleans and deduplicates it, builds a Cree tokenizer, continues pretraining, tunes on instruction data, teaches faithfulness with preference pairs, scores everything on held-out data, and exports a quantised GGUF that the app can load.

## What is different from a generic fine-tune

A general model sees Cree as a long tail of rare byte sequences. Three things change that here.

1. **The tokenizer learns Cree first.** A tokenizer trained on English splits `nêhiyawêwin` into many pieces; this one learns the actual syllables and morphemes of the language, then grafts them onto the base model with statistically initialised embeddings. Fewer tokens per word means more real text per context window, faster generation, and less room for spelling drift.
2. **The data is curated, not scraped blindly.** Every source carries its licence, dialect and script in a provenance ledger. Restricted material in the EdTeKLA collection is excluded at the loader, not quietly copied. Near-duplicate sentences are removed, translation pairs are checked for alignment, and the test split is document-disjoint.
3. **The model is taught to be honest.** The instruction set contains unanswerable questions whose correct answer is "I do not have notes on that", plus preference pairs that punish invented names, dropped clauses and English leaking into a Cree sentence. That is the failure mode that matters for a cultural project.

## What it produces

| Artifact | Where | What it is |
|---|---|---|
| `archer-<size>.Q4_K_M.gguf` | `out/` | the only chat model the app loads, about 4.5 GB for the 7B default |
| `creebench_report.json` | `out/` | every score, per task, per split |
| `report_card.md` | `out/` | a model card with data provenance, licence and results |
| `training_curves.png`, `creebench.png` | `out/` | the figures the report cites |
| `Modelfile` | `out/` | sampling settings and context length for Ollama |
| `run_state.json` | `archer_cree_llm/` | which stages are finished, and from which inputs |
| `checkpoints/<stage>/checkpoint-N` | `archer_cree_llm/` | the resume points, written every few minutes |

## Before you run

Use **Runtime → Change runtime type → A100** for the full recipe. A free T4 also works, and needs no flag: the notebook reads the card, loads the base in nf4 when 16-bit weights would not leave room to train, and keeps the vocabulary table out of the optimizer, which is what lets a 7B model fit in 15 GB. `CREE_PROFILE="t4"` shortens the schedules to match. A free CPU runtime is enough for the data, tokenizer and evaluation cells but not for training.

Model weights are cached on the local runtime disk instead of Drive, because a multi-gigabyte shard written through the Drive mount can come back damaged; each shard is checked before it is loaded and a broken one is fetched again on its own. Everything that cannot be recreated, meaning checkpoints, the run ledger, plots and exports, is written to Drive. Set `CREE_USE_DRIVE=0` to keep everything in `/content` instead.

## If Colab disconnects

Free Colab ends a session without warning, and a run this long will lose one eventually. The notebook is built for that. Each finished stage writes a marker to `run_state.json` on Drive, and each training stage writes a checkpoint to Drive every few minutes. The recovery is: reconnect, **Runtime → Run all**, and leave it. The data cells find their outputs already on Drive and skip the downloads, a finished stage is not trained again, and the stage that was interrupted continues from its last checkpoint with its step count, learning rate schedule and loss curve intact. `Runtime → Restart and run all` behaves the same way; nothing is repeated that already has a marker.

Two settings control it. `CREE_FORCE=cpt,sft` re-runs those stages from the beginning even though they are marked finished, and `CREE_SAVE_MINUTES=5` shortens the checkpoint interval. The stage names are `corpus`, `clean`, `tokenizer`, `cpt`, `sft`, `dpo`, `eval`. The setup cell prints which of them are done and where an interrupted one had reached.

Every cell prints what it did. If a source is unreachable the cell records that fact and continues with the rest; it never substitutes invented text.

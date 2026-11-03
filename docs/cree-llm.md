# The Archer model

The chat tab runs one model, trained for this project by
`training/notebooks/Cree_LLM_Training.ipynb` and served as `archer:7b`. It does every job the
assistant has:

- conversation about Archer's work, the songs and this system
- answers from Archer's song notes, retrieved per question, with the source
- translation between English and nêhiyawêwin in both directions
- SRO and syllabics conversion, and explanations of Cree words

There is no second chat model and no router. Fetching notes, translating and
talking are the same weights under different task prefixes, listed under
[how the app calls it](#how-the-app-calls-it). Retrieval decides whether notes
are in the prompt; it does not decide which model answers.

Retrieval uses a small sentence-embedding model to rank passages. It is a search
index, not a second chat model: it returns text and never writes an answer.

That is a training decision, not only a wiring one. The instruction set mixes
translation, orthography, word glosses, note reading, abstention and project
conversation in one adapter, so the model learns the behaviour of the whole
assistant instead of one narrow skill. English replay during continued
pretraining is what keeps the base model's conversation intact while the Cree
tokenizer and corpus are added.

## What the notebook trains

`training/notebooks/Cree_LLM_Training.ipynb` is a complete run, not a fine-tuning snippet. It
downloads the corpus, cleans it, builds a tokenizer, trains in three stages,
scores the result against its own base model and against the old three-part
setup, and exports a GGUF.

| Stage | What happens | Why it is there |
|---|---|---|
| Corpus | EdTeKLA, the Plains Cree Figurative corpus, Cree Wikipedia, FineWeb-2, FinePDFs, FineTranslations, GlotCC, GlotStoryBook, Tatoeba, the Plains Cree phrase recordings, the project's aligned pairs, Watkins 1865 and 14 public-domain Cree books from the Internet Archive | every source is ledgered with licence, dialect and script; historical books are tagged as historical and never used as modern translation targets |
| Cleaning | language check per sentence, bag-of-words rejection, exact and MinHash deduplication, document-disjoint split, second leakage pass | a score earned on leaked text is not a score |
| Tokenizer | SentencePiece BPE trained on Cree, filtered, grafted onto the base vocabulary, syllabics glyphs added | the base tokenizer spends over four tokens on a Cree word; the Cree one spends about a third fewer, and the notebook reports the exact held-out figure |
| Embeddings | each new token initialised from the average of the base model's pieces for the same string | a token that starts with meaning needs far less data to learn |
| Continued pretraining | packed Cree text with 10% English replay, rank-64 LoRA | teaches the language without erasing the base model's instruction ability |
| Instruction tuning | translation both ways, SRO and syllabics, word glosses, grounded note answers, abstention, project conversation, persona | one adapter learns every job the assistant does, not just the language |
| Preference tuning | DPO on faithfulness pairs: dropped clauses, changed numbers, English leaks, flattened vowels | fluency is not fidelity |
| CreeBench | chrF++, BLEU, token F1, lexicon validity, code-switch, abstention, perplexity, latency | one BLEU number hides the failures a speaker hears first |
| Export | merge, convert to GGUF, Q4_K_M, Modelfile | the app runs on a laptop |

The notebook's report card records every source and its licence. NoDerivatives
material in the EdTeKLA collection is excluded at the loader; NonCommercial
ShareAlike material is included only while the model itself carries that
licence.

## Running it

Open `training/notebooks/Cree_LLM_Training.ipynb` in Colab. Select **Runtime → Change runtime type
→ A100** for the full recipe, or set `CREE_PROFILE=t4` for a 16 GB T4. The
notebook writes checkpoints to Google Drive when it is mounted, so a dropped
session does not lose the run.

### On your own GPU

`train_cree_llm.py` in the repository root is the same pipeline as one command.
It is generated from the notebook's cells, so the two cannot disagree, and it is
standalone: copy the file to the machine, install the requirements, run it.
Nothing is read from the repository and nothing needs Colab.

```bash
python -m pip install -r training/requirements-cree.txt
python train_cree_llm.py --check
python train_cree_llm.py --profile a100 --work /data/cree_llm
```

`--profile` picks the same presets the notebook uses. `--base-model`,
`--seq-len`, `--lora-r` and the three epoch counts override them, `--to STAGE`
stops the run after a stage you want to inspect, and every run writes a
timestamped log to `<work>/logs/` as well as the terminal.

Base weights are cached under `<work>/hf` by default, which is 15 GB for the 7B.
When the work folder is a network volume, point `--hf-home` at local disk so the
weights stay fast while the checkpoints and the report sit on the big volume.

Resuming is the same single action: run the command again. `--force cpt` redoes
a stage and everything trained on top of it, because a fresh CPT with an old SFT
adapter underneath is not a result.

### Interruptions

Colab ends a session without asking, so every stage records itself.
`run_state.json` in the work folder maps each stage (`corpus`, `clean`,
`tokenizer`, `cpt`, `sft`, `dpo`, `eval`) to a fingerprint of the inputs that
produced it. A matching marker means the stage is skipped and its files are read
back from Drive; a changed base model, corpus, context length or LoRA rank
invalidates the markers that depend on it, so a new run cannot inherit numbers
from an old one.

Training stages also write a checkpoint every few minutes and resume from the
newest one whose adapter header and trainer state are intact, so an interrupted
CPT continues at the step it reached with its schedule and loss history rather
than starting again. The stage's root adapter is written only after training
returns, which is what stops an interrupted stage from looking finished.

Recovery is one action: reconnect and **Runtime → Run all**. `CREE_FORCE=cpt,sft`
re-runs those stages anyway, and `CREE_SAVE_MINUTES=5` shortens the checkpoint
interval.

Model weights are cached on the runtime's local disk (`/content/hf_cache`), not
on Drive. A 3.9 GB shard written through the Drive mount can come back
half-synced, which appears as `SafetensorError: header too large` before the
first training step. Every shard is header-checked before it is handed to
transformers, a damaged one is downloaded again on its own instead of the whole
15 GB model, and the error names the exact file if the repair does not take.
Checkpoints and exports, which cannot be recreated, still go to Drive.

On a 15 GB card the run fits because of one decision: the vocabulary table never
trains. `embed_tokens` and `lm_head` for a 7B model are 1.1 billion parameters,
and in fp32 with gradients and Adam state that is about 11 GB, which is the whole
reason a free T4 dies at 14.28 GiB allocated on the first cross-entropy. The
grafted rows are saved to `processed/grafted_embeddings.pt` (28 MB) and copied
back on every load, the base is loaded in nf4 when 16-bit weights would not leave
room to train, and the adapter is what learns. Precision is chosen from the card
rather than from a flag, so the same notebook takes an A100 at full 16-bit and a
T4 quantised without being told which it is on.

The full run is roughly:

| Profile | Base | Context | Time on the target GPU |
|---|---|---|---|
| `t4` | Qwen2.5-7B-Instruct | 1024 | 4 to 7 hours |
| `big24` | Qwen2.5-14B-Instruct | 1024 | 14B on one 24 GB card; slower than the 7B, merge needs about 64 GB of RAM |
| `a100` | Qwen2.5-7B-Instruct | 2048 | 2 to 4 hours |
| `big` | Qwen2.5-14B-Instruct | 2048 | 6 to 10 hours on an 80 GB A100 |
| `huge` | Qwen2.5-32B-Instruct | 2048 | about a day on an 80 GB A100; merging needs a separate machine with roughly 160 GB of RAM |
| `smoke` | Qwen2.5-0.5B-Instruct | 384 | minutes, for checking the pipeline |

The model is a 7B by default. That is the largest size whose
Q4_K_M export runs comfortably on the laptop that runs the app, and the
tokenizer and data work matter more here than another ten billion parameters.
`CREE_PROFILE=big` and `CREE_PROFILE=huge` are there when you want the larger
model and have the GPU.

The corpus is the real constraint, not the parameter count. The open web has
very little Plains Cree: FineWeb-2 carries 192 documents and 108,000 words, and
Cree Wikipedia has 187 articles, of which 37 survive the language filter. That
is why the loader reaches past the usual web crawl. FinePDFs adds 733 Cree
PDFs and 1.19 million words of extracted text. FineTranslations adds 582
aligned Cree and English chunks. The Internet Archive adds 14 public-domain
books, 612,000 words, including Bloomfield's Plains Cree Texts and Sacred
Stories of the Sweet Grass Cree, both checked window by window so a bad
syllabic scan cannot pass as Cree. The smaller sources (GlotCC, GlotStoryBook,
Tatoeba, the phrase recordings, GiellaLT's public texts) contribute the
remaining modern and reference material.

After cleaning, the full profile has **1,124,146 words across 66,071
sentences**, 10,934 training pairs and 203 held-out pairs, plus 57,579 lexicon
forms. That is roughly eight times the text the previous notebook used. The
smoke profile uses a small slice, which is why its report shows a
smaller number; the full profile uses every accepted document. The tokenizer,
the embedding initialisation, the English replay and the instruction mix are
how the notebook gets the most out of that, and the upload folder is how it
grows. More consented contemporary Cree text will still do more for the model
than another ten billion parameters.

## Installing the result

```bash
mkdir -p assets/archer_llm
cp archer-7b.Q4_K_M.gguf assets/archer_llm/
cp Modelfile assets/archer_llm/
cd assets/archer_llm && ollama create archer:7b -f Modelfile
```

`config/config.yaml` already points at that tag:

```yaml
chat:
  model: archer:7b
  temperature: 0.3
  max_tokens: 500
```

For `chat.backend: llama_cpp`, set `chat.gguf_path` to the same
`archer-7b.Q4_K_M.gguf` file. It is the same model, not a smaller fallback.

If the tag is missing, the chat returns a 503 with the `ollama create` command.
It does not answer from another model, because another model would not be this
assistant. Every reply reports the model and the task it used:

```json
{"reply": "...", "task": "song_notes", "language": "english", "model": "archer:7b", "sources": ["Horse.md"]}
```

`task` is `song_notes`, `chat`, `translate` or `orthography`. It describes the
prompt shape, not a model choice. `language` reports whether the question was
written in Cree; it does not change the backend.

## How the app calls it

The model was trained with control prefixes, so the app reproduces them rather
than sending a bare chat message:

| Request | Prompt shape |
|---|---|
| question with retrieved notes | `<|song|><|note|>` · `Notes on <song>:` · `Question:` |
| question with no notes | `<|song|>` · the question |
| English to Cree | `<|translate|><|english|><|sro|>` (or `<|cans|>` for syllabics) |
| Cree to English | `<|translate|><|cree|><|english|>` |
| SRO to syllabics | `<|sro|><|cans|>` |
| syllabics to SRO | `<|cans|><|sro|>` |

The prompt builder is [`chat/archer_prompt.py`](../chat/archer_prompt.py). It
classifies the request into one of those shapes with `task_of`, and
`format_prompt` emits the shape. Retrieval runs on every question, but notes are
attached only when the request is a notes question and the passage clears the
0.30 relevance floor. A translation is never padded with song prose, and a
general question with no matching notes is answered on the `<|song|>` path.

`POST /api/chat/prompt` exposes the same call for the site's demo: it returns
the task, the retrieved passages, the notes that were actually used and the
exact string sent to the model.

## Comparing against another Cree model

The benchmark is useful only if it can score something else. Install the other
model in the same Ollama daemon the notebook can reach, set
`CREE_EXTERNAL_MODEL` to its tag before the setup cell runs, and the harness
sends the identical held-out sentences to that model, scores them with the same
metrics, and prints them under the tuned model's rows. The control tokens this
model was trained on are stripped first, so the other model is measured on the
instruction, not on a prefix it has never seen. Comparing
`mickymultani/Cree-Llama-3.3-70B` this way takes one tag and one run, and the
same setting with `llama3.1:8b-instruct-q4_K_M` measures the general instruction
model the app used before this project trained its own.

The report also computes the bar this model has to clear. The app used to be a
general model, a separate Cree model and a retrieval index, with a router
choosing between them, so "better than all three" means beating the best of
those pieces on each job. Note reading and abstention are therefore measured
twice, with the adapter and without it: the adapter-free run is the old system,
reading the same retrieved notes with the same general model. The report prints
`meets or beats` or `below` for every row, and the count of rows cleared, so the
result is measured instead of claimed.

In Colab, put `os.environ["CREE_EXTERNAL_MODEL"] = "llama3.1:8b-instruct-q4_K_M"`
at the top of the setup cell, before `CFG` is created. Point the notebook at an
Ollama endpoint it can reach with `CREE_OLLAMA_URL` if Ollama is not on the
runtime itself; the comparison is skipped, with a recorded reason, when the
daemon is unreachable.

## What the model is not

It is not a fluent speaker and it is not a substitute for an Elder or a fluent
reviewer. It is a careful reader with a good memory for this corpus, a tokenizer
that respects the spelling, and an instruction set that rewards saying "the
notes do not say". Ceremonial, legal, medical and treaty language is out of
scope, and the report card says so.

## Adding data

The highest-value additions are consented contemporary Cree text, more song
notes, and corrections from fluent speakers. The public-domain books are already
in the ledger; modern community text is the piece that cannot be scraped and
must be contributed. Put `.txt` or `.md` files in the
folder the notebook calls `uploads` (by default `cree_llm_work/uploads`, or
`/content/drive/MyDrive/cree_corpus` on Colab) and rerun from the corpus cell.
Every file is recorded as `community` with an `unspecified` dialect, so it is
visible in the ledger until someone labels it.

The evaluation is the point of rerunning: it will say whether the new text
helped, rather than assuming that more data always does.

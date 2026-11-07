# Making the chat model good at conversation and translation

`docs/pipeline-best-chatbot.md` is the full end to end pipeline: base model
choice, the data programme, the four training stages, the gates each one has to
pass, evaluation, and what the result can and cannot do. This file is the short
version of why the current model fails and what was changed in the data.

This is the honest account of what the installed `archer:7b` can and cannot do,
and the concrete change that fixes it. No part of the fix is a reply table in the
app: the app sends a system prompt and the model answers. Everything below is
training data, a training run, and a measurement.

## What the model does properly today

Measured with `python tools/audit_chat_model.py` against the installed tag:

- Project questions it was trained on, answered from memory, word for word:
  "What is this project?" returns the project description that is in the SFT set,
  "Where do your song answers come from?" returns the `song_notes` explanation,
  "Are you Archer?" returns the boundary answer. These are correct and stable.
- Song-note questions: the retrieval layer hands the model passages from
  `song_notes/`, and it answers from them with the filenames as sources.
- A Cree word with a dictionary entry: `maskisin`, `tânisi`, `maskwa` are answered
  from `chat/cree_glossary.py` over the project dictionaries. That path does not
  use the model at all, which is why it is reliable.
- Translation of sentences that are close to its training pairs, and abstention
  when a word has no entry.

## What it does badly, and why

1. **It repeats the visitor's sentence back.** "hi" returns "hi"; "howdy, hows
   your day been big bro" returns "Howdy, how is your day been, big bro?". Cause:
   the SFT set had thirteen fixed greetings and nothing else conversational, so
   for an unseen phrasing the most likely continuation of the prompt is the
   prompt. Measured at temperature 0.0, 0.4, 0.6 and 0.7: three of four greetings
   echoed at every setting, so this is data, not decoding.
2. **It recites a capability list when asked who it is.** Cause: the SFT target
   for "Who are you?" literally was that list.
3. **Translation is only as good as the pairs it saw.** About 18,700 aligned
   sentences exist (`parallel_cr_en_combined.jsonl`) plus 10,300 dictionary
   pairs. That is enough for a LoRA to translate short sentences of the kind in
   the corpus, and not enough for arbitrary prose. Nobody can promise "no
   mistakes at all" on a low-resource language with this data; what can be
   promised is a number, printed by the audit, that goes up or down.

## What changed

`training/cree_llm/cells/08_sft_data.py` (mirrored in `train_cree_llm.py`):

- The two capability-list identity answers were deleted and replaced by three
  persona examples that describe a stance: Archer's assistant, learning
  nêhiyawêwin, answering from his notes.
- A generated casual-conversation block: 14 openings x 6 replies, 12 ways of
  asking how you are x 5 replies, and 16 short exchanges (thanks, bye, "you
  there?", "what's up?"). Both sides vary, and no answer is a copy of the prompt,
  so "repeat the input" stops being the highest-probability continuation.
- "What can you do?" is no longer a four-item feature list.
- The 13 greeting examples were rewritten earlier so the old "What can I help you
  with?" phrase is not the model's whole voice.

## How much Cree the repository actually holds

`python tools/build_cree_corpus.py` counts every Cree source on disk and writes
`data/processed/cree_corpus.jsonl` and `data/processed/parallel_clean.jsonl`. On
this checkout:

| measure | value |
| --- | --- |
| distinct entries in the corpus | 6,276 |
| words | 9,364 |
| estimated tokens (3.2 chars/token) | about 26,600 |
| clean translation pairs | 6,999 |
| pairs with 3 or more words on both sides | 1,391 |
| pair rows that were dictionary entries with sense numbers | 15,914 of 27,123 |

Most of what looked like a parallel corpus is a dictionary: single forms like
`wapinio` with an English gloss, and rows carrying sense numbers (`niyânan 2`,
`okay 6`) that are not sentences. The running-text total is a few thousand words.
That is the reason translation is weak, and it is not a reason a different
training schedule can fix: 26,600 tokens is roughly what a model sees in a few
minutes of ordinary text.

Landmarks, stated as order of magnitude and not as promises: below about 2M
tokens a model learns spelling and common sentences; 10M covers frequent grammar;
beyond that it starts writing its own sentences. Fluency in Cree needs a real
corpus on the order of tens of millions of tokens of modern running text, with a
held-out split, licensed and credited like the sources already in the report.

So before any run, either add that corpus or accept the ceiling. The gate is
there so nobody has to argue about it later:

```bash
python tools/build_cree_corpus.py --min-tokens 20000000   # fails until the data is there
```

## The run

Training needs a GPU. This laptop has none (`torch.cuda.is_available()` is
False), 3 GB of free RAM and 5.9 GB of free disk, so the run happens on the
machine the notebook was built for.

```bash
# 14B, one 80 GB card, the smallest base worth calling a rewrite of the voice
python train_cree_llm.py --profile big --work /data/cree_llm

# 32B, still one 80 GB card with QLoRA; merging and quantising needs a
# high-memory step of its own (the notebook's export cell says so)
python train_cree_llm.py --profile huge --work /data/cree_llm
```

Stages: CPT on Cree prose, SFT on the data above, DPO on the preference pairs,
then `export` merges the adapter, converts to GGUF, quantises to Q4_K_M and
writes the Modelfile. Copy the GGUF and the Modelfile to this machine and:

```bash
cd assets/archer_llm && ollama create archer:14b -f Modelfile
```

Then point the app at that tag in `config/config.yaml`:

```yaml
chat:
  model: archer:14b
```

The run also reads the corrections the app exported, `feedback_sft.jsonl` for
instruction tuning and `feedback_dpo.jsonl` for preference tuning. Set
`ARCHER_FEEDBACK_DIR` to the folder that contains them, or copy that folder to
`data/training` in the repository the run starts from. The stage prints how many
rows it loaded; a missing folder is reported rather than silently skipped.

`python tools/verify_training_pipeline.py` fails if the 14B profiles, the corpus
completeness gate, correction ingestion, echo negatives, the attested-word gate,
the dynamic export tag, or the generated trainer and notebook have drifted.

## The check that decides whether it worked

```bash
python tools/audit_chat_model.py          # full, about two minutes on CPU
python tools/audit_chat_model.py --quick  # greetings and identity only
```

It fails the run on any echoed greeting, any identity answer missing its facts,
a word with a dictionary entry that is not glossed, or a mean token overlap below
0.5 on six short held-out translation pairs. Run it before and after installing
a new tag: the same script prints both numbers, so "better" is a comparison, not
an opinion.

## What is not achievable

"There are no mistakes at all" is not a reachable state for translation into a
low-resource language, and a model that claims it would be lying confidently,
which is the failure this project was built to avoid. What is reachable: no
echoed prompts, a persona in the model's own voice, correct answers on the
project facts, and a translation score that rises with the data and is printed
honestly. If a word or a sentence is beyond it, it says so; that behaviour is
trained and tested too.

# Archer server training runbook

This is the full sequence for training the Archer assistant on your own GPU
server. It is written for the 14B model, because that is the model you asked to
train. The same commands work for 7B by changing `--profile`.

Read this before renting a GPU: the repository currently contains about 1.6M
tokens of usable Cree/English material. The 14B profiles now warn below 20M
tokens but continue, because you asked that every verified token be used. The
warning is written to `raw/corpus_size.json` and the source status. Training a
14B on 1.6M tokens still gives you a weak model with more parameters, so the
runbook does not call that result high-end. The standalone corpus command still
exits 1 below 20M so a release check cannot quietly pass.

No training process can honestly promise "perfect" or "best in the world".
This pipeline can promise that every score comes from a real held-out run, that
the model is trained rather than prompt-patched, and that the run stops when the
data is not good enough.

## 0. What the finished system must do

The target behaviour is:

1. Read a message and identify Cree, English, or mixed language from learned
   character patterns, without a hardcoded Cree word list.
2. Answer a Cree message in English by default.
3. Answer in Cree only when the visitor explicitly asks for Cree.
4. Know its identity and purpose from its weights, not from a line pasted into
   every request.
5. Translate both directions, explain dictionary words, and use Archer's song
   notes for project questions.
6. Refuse to invent facts, and never repeat the prompt as the answer.

Items 1 and 3 are trained in the SFT data now. The app's learned
`chat/language_id.py` is a second, independent check at inference time; it is
not a word list.

## 1. Hardware and disk

Use one GPU. The training wrapper is not a multi-GPU distributed trainer.

| Profile | GPU | Base | Context | LoRA | Use |
|---|---|---|---|---|---|
| `big` | 80 GB A100/H100 | Qwen2.5-14B-Instruct | 2048 | 64 | the intended 14B run |
| `big24` | 24 GB 4090/A10 | Qwen2.5-14B-Instruct | 1024 | 32 | 14B on one smaller card |
| `a100` | 40 GB A100 | Qwen2.5-7B-Instruct | 2048 | 64 | 7B comparison |
| `t4` | 16 GB T4 | Qwen2.5-7B-Instruct | 1024 | 32 | small diagnostic |
| `smoke` | CPU is fine | Qwen2.5-0.5B-Instruct | 384 | 8 | pipeline test only |

The 24 GB profile merges the adapter on CPU. Budget at least 64 GB of system
RAM for that merge. The 80 GB profile still needs a lot of disk, not just VRAM.

Recommended server:

- one 80 GB A100 or H100 for the real 14B run
- 128 GB system RAM
- 8 or more modern CPU cores
- 250 GB free disk for the work folder
- a second disk or object store for the Hugging Face cache is useful but not
  required

Check the machine before doing anything else:

```bash
nvidia-smi
free -h
df -h /data
python3 --version
```

You need Python 3.10 or newer. The project's local environment is Python 3.12.

Disk budget: the base weights, the merged fp16 model, the adapter checkpoints,
the GGUF export, and the tokenizer all coexist during export. A 14B merge can
need roughly 30 GB for the fp16 weights and another 30 GB while writing the
merged copy, before the quantised GGUF. Leave 100 GB free even after the corpus.

## 2. Put the repository on the server

From your computer:

```bash
rsync -av --exclude .venv --exclude .git \
  "/path/to/Archer-Robot-Vocal-Collaboration-System/" \
  user@server:/opt/archer/
```

On the server:

```bash
cd /opt/archer
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install the PyTorch wheel that matches the server's CUDA driver. Check
`nvidia-smi` for the CUDA version, then use the matching index from
`https://pytorch.org/get-started/locally/`. A typical CUDA 12.4 machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install the project's training requirements:

```bash
pip install -r training/requirements-cree.txt
```

If `training/requirements-cree.txt` is not present on your checkout, install the
packages the trainer imports:

```bash
pip install transformers datasets peft trl accelerate bitsandbytes \
  sentencepiece protobuf huggingface_hub safetensors \
  pandas matplotlib sacrebleu requests
```

For GGUF export you also need a compiler and CMake:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

Do not use `sudo` for the Python environment; keep the virtual environment
owned by your user.

Check the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import transformers, peft, trl, bitsandbytes; print(transformers.__version__, peft.__version__)"
python train_cree_llm.py --check
```

`--check` must show the GPU and the installed package versions. If it says no
GPU, the PyTorch wheel is a CPU wheel. Reinstall the CUDA wheel before going
further. If it warns that bitsandbytes has no CUDA kernel, the bitsandbytes
build does not match the installed torch/CUDA combination; fix that before a
long run.

## 3. Create the work and cache folders

Use a real data disk, not `/tmp`:

```bash
export CREE_WORK=/data/cree_llm
export HF_HOME=/data/hf
mkdir -p "$CREE_WORK" "$HF_HOME"
```

Set these in the same shell for every command, or put them in a small
`env.sh` file that you source. The trainer also accepts `--work`.

Log in to Hugging Face if the base model needs an authenticated download:

```bash
huggingface-cli login
```

The default Qwen models are public, so a token is not normally required.

## 4. Check every corpus source is reachable

Run this before the download, not after:

```bash
python tools/check_cree_sources.py
python tools/check_cree_sources.py --json > /data/cree_llm/source_check.json
```

It checks the same endpoints the corpus cell uses: EdTeKLA, GiellaLT,
Tatoeba, FineWeb-2, FinePDFs, FineTranslations, GlotCC, GlotStoryBook,
Plains Cree Figurative, Cree Wikipedia, Internet Archive, and the project
pairs file. It exits non-zero if an endpoint is down.

If one source is down, fix the network or wait. Do not use
`allow_missing_sources` for the 14B run: a partial corpus is exactly how the
model ends up unable to write the language while the report still looks green.

## 5. Download and build the corpus

The corpus stage is the slowest network part. Run it separately so a failure is
easy to diagnose:

```bash
cd /opt/archer
. .venv/bin/activate
export CREE_WORK=/data/cree_llm
export HF_HOME=/data/hf

python train_cree_llm.py \
  --profile big \
  --work "$CREE_WORK" \
  --to corpus \
  --log-file "$CREE_WORK/corpus.log"
```

The corpus cell writes:

```text
$CREE_WORK/raw/docs.jsonl
$CREE_WORK/raw/pairs.jsonl
$CREE_WORK/raw/lexicon.json
$CREE_WORK/raw/ledger.json
$CREE_WORK/raw/source_status.json
```

Inspect them:

```bash
ls -lh "$CREE_WORK/raw"
python - <<'PY'
import json, os
root = os.environ.get("CREE_WORK", "/data/cree_llm")
for name in ("docs", "pairs"):
    path = f"{root}/raw/{name}.jsonl"
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    print(name, n)
status = json.load(open(f"{root}/raw/source_status.json", encoding="utf-8"))
for key, value in sorted(status.items()):
    print(f"{key:20s} {value}")
PY
```

Every expected source must say `ok:` and contribute something. `ok: 0 docs` is
not a contribution.

### 5.1 Build the training text and check its size

The training cell's source check is not enough. Build the processed corpus and
run the release gate:

```bash
python tools/build_cree_corpus.py --min-tokens 20000000
```

This command exits 1 when the corpus is below 20M tokens. That is a readiness
result, not a training stop. `train_cree_llm.py` records the same number as a
warning in `raw/corpus_size.json`, logs it, and continues with every verified
token. Do not treat a warning as proof of quality. A 14B run below 20M is a
diagnostic model, not a high-end claim.

Add text in `.txt` or `.md` files here:

```bash
export CREE_UPLOAD_DIR=/data/cree_uploads
mkdir -p "$CREE_UPLOAD_DIR"
```

Then put your files in that directory and rerun the corpus stage. The corpus
cell reads `CREE_UPLOAD_DIR` and the work folder's `uploads` directory. Keep
the files visible in the run ledger; do not silently mix in unlicensed or
machine-generated text.

Good additions are modern Plains Cree prose, community writing you have
permission to use, and aligned Cree/English pairs. Do not pad the corpus with
English-only text to make the number pass. The gate counts tokens, but a paid
GPU will still learn the wrong distribution if the added tokens are not Cree or
aligned translation.

### 5.2 What public data actually exists

This matters more than any training command:

- EdTeKLA Cree Corpus: about 49,038 words and 3,727 lines, with per-collection
  licences. The loader uses the permissive parts. Its `CreeLiteracyOrg` folder
  is CC BY-NC-ND and is correctly skipped because NoDerivatives forbids a
  trained adaptation.
- Little Cree Books: four Plains Cree stories in SRO with English siblings,
  CC BY-NC-SA. They are now downloaded and aligned by `load_lcb`.
- itwêwina: a dictionary, not running prose. Its CC BY 4.0 site can supply
  lexical pairs, but not millions of sentences.
- Public-domain historical books: Bloomfield, hymns, primers and Gospels are
  now searched across the full public-domain shelf, not a fixed sixty-five
  books. OCR filters reject bad scans.
- ALTLab Korp: about 186,921 words of Plains Cree text, but the Ahenakew-Wolfart
  collections require an access code. Request it from Antti Arppe at the
  University of Alberta before counting that text.
- Cree Literacy Network: CC BY-NC-ND, not usable for model training without
  explicit permission.
- Cree: Language of the Plains: CC BY-NC-ND, same restriction.
- The 2018 Canadian Bible Society Plains Cree Bible: personal use only. The old
  1862 Mason text is public domain, but the modern digitized edition is not.

There is no honest way to reach 20M Plains Cree tokens from the public sources
listed above alone. The 20M target needs community permission, the Korp access
grant, or a licensed text collection. Do not replace the missing Cree with
English or generated text and call it data.

## 6. Confirm the training data

Before the GPU is busy:

```bash
python tools/verify_training_pipeline.py
python train_cree_llm.py --check
```

The verifier checks that the SFT cell contains language identification and
answer-language policy examples, that the 14B profiles warn below 20M and keep
training, that the feedback loaders are present, that the duplicate audit is
wired into the clean stage, that the runbook documents durable execution, and
that the benchmark and publisher tools exist.

The SFT mix now includes:

- 1,200 identification examples from the labelled pairs: Cree, English, mixed
- 1,000 answer-language examples: Cree message to English answer, and an
  explicit "Answer in Cree" request to a Cree answer
- project, persona, translation, dictionary, notes, casual, and feedback rows

The language labels come from the corpus columns, not from a hand-written word
list. The answer policy is a trained pattern, not an app-only `if` statement.

## 7. Copy real user feedback, if any exists

The app writes feedback to its memory database. Export it on the app machine:

```bash
python tools/consolidate_feedback.py --export
```

Copy the exported `data/training/feedback_sft.jsonl` and
`data/training/feedback_dpo.jsonl` to the server:

```bash
scp data/training/feedback_sft.jsonl user@server:/data/archer_feedback/
scp data/training/feedback_dpo.jsonl user@server:/data/archer_feedback/
```

On the server:

```bash
export ARCHER_FEEDBACK_DIR=/data/archer_feedback
```

The SFT cell weights `feedback_correction` 6x. The DPO cell appends those
preference pairs after the synthetic cap. If there are no feedback files, both
loaders print "none found" and training continues with the corpus; that is not
an error, it just means no human corrections have been added yet.

## 8. Durable execution: your laptop can turn off

Training runs on the GPU server. Your laptop is only a terminal: once the
command is detached from the SSH session, closing the lid, losing Wi-Fi or
shutting the laptop down does not touch the process. The two things that can
still stop it are the server rebooting and the process crashing, and both are
recoverable: the trainer writes a stage marker after every finished stage and a
checkpoint every `--save-minutes`, so restarting the same command resumes from
the last finished unit of work instead of from zero.

`scripts/train_archer_14b.sh` runs the whole sequence in order, writes one log
per stage, and exits non-zero on the first failure. Running it twice is safe:
finished stages are skipped, an interrupted stage resumes. It also refuses to
start a second copy against the same work folder, because two drivers writing
the same checkpoints corrupt them.

### 8.1 tmux (simplest)

```bash
ssh user@server
cd /opt/archer
tmux new -s archer

export CREE_WORK=/data/cree_llm
export HF_HOME=/data/hf
export CREE_PROFILE=big
bash scripts/train_archer_14b.sh

# detach without stopping anything: Ctrl-b, then d
```

Close the laptop. Come back and watch it:

```bash
ssh user@server
tmux ls
tmux attach -t archer                  # live view
tail -f /data/cree_llm/logs/cpt.log    # or just the stage log
```

### 8.2 nohup (no tmux installed)

```bash
cd /opt/archer
nohup bash scripts/train_archer_14b.sh \
  > /data/cree_llm/logs/driver.log 2>&1 &
echo $! > /data/cree_llm/logs/driver.pid
disown
```

`disown` removes the job from the shell's table, so closing the terminal does
not send it SIGHUP. Check it and stop it by PID, never by process name:

```bash
kill -0 "$(cat /data/cree_llm/logs/driver.pid)" && echo running
tail -f /data/cree_llm/logs/driver.log
kill "$(cat /data/cree_llm/logs/driver.pid)"    # deliberate stop only
```

### 8.3 systemd (survives a reboot, restarts after a crash)

This is the version to use if the server may reboot while you are away. Create
`/etc/systemd/system/archer-train.service`:

```ini
[Unit]
Description=Archer 14B Cree training
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/archer
Environment=CREE_WORK=/data/cree_llm
Environment=HF_HOME=/data/hf
Environment=CREE_PROFILE=big
Environment=ARCHER_FEEDBACK_DIR=/data/archer_feedback
ExecStart=/bin/bash /opt/archer/scripts/train_archer_14b.sh
Restart=on-failure
RestartSec=120
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USER`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now archer-train
systemctl status archer-train
journalctl -u archer-train -f
```

`Restart=on-failure` restarts the driver after a crash, and the driver resumes
the interrupted stage from its checkpoint. A finished stage is never repeated.

### 8.4 What a reboot or crash costs

Nothing already finished. Check where the run stands:

```bash
cat /data/cree_llm/run_state.json
ls -1 /data/cree_llm/checkpoints/*
tail -n 40 /data/cree_llm/logs/*.log
```

Then start it the same way again, or `sudo systemctl start archer-train`. Do not
delete the work folder and do not use `--force` to "clean up": a forced stage
throws away the checkpoint it would otherwise resume from.

### 8.5 Do not run the training on the laptop

There is no GPU in the laptop, and a session that dies with the lid would stop
the run. The laptop's only jobs are copying files and reading logs.

## 9. Run the training sequence

Set the environment once, then start the driver as described in section 8:

```bash
cd /opt/archer
. .venv/bin/activate
export CREE_WORK=/data/cree_llm
export HF_HOME=/data/hf
export ARCHER_FEEDBACK_DIR=/data/archer_feedback
bash scripts/train_archer_14b.sh
```

The stages are `corpus`, `clean`, `tokenizer`, `cpt`, `sft`, `dpo`, `eval`,
`report`, `export`. The driver runs them in order, and a stage marker is written
only after that stage succeeds, so a crash can be resumed without recomputing
earlier work. The per-stage commands in 9.1 to 9.7 are exactly what the driver
calls; use them when you want to run or rerun one stage by hand.

### 9.1 Clean and audit contamination

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to clean --log-file "$CREE_WORK/clean.log"
```

This writes the held-out split and the 12-gram contamination audit. It raises if
the test split overlaps the training split. Do not disable that audit.

It also writes `$CREE_WORK/processed/duplicate_audit.json`, which is the
duplicate-word check. It records the raw and kept token and unique-word counts,
duplicate document/sentence/pair counts, the spelling-variant groups (forms that
differ only by case, apostrophe or macron and collapse to one word), and the
words removed by the language filter. The stage fails if any exact duplicate row
survives, so "no duplicate rows" is a check, not a promise. Inspect it before
the GPU is busy:

```bash
python - <<'PY'
import json, os
root = os.environ.get("CREE_WORK", "/data/cree_llm")
audit = json.load(open(f"{root}/processed/duplicate_audit.json", encoding="utf-8"))
print(json.dumps({k: audit[k] for k in ("tokens", "unique_words", "documents", "sentences", "pairs", "checks")}, indent=2))
print("spelling-variant groups:", audit["spelling_variants"]["groups"])
PY
```

### 9.2 Train the tokenizer

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to tokenizer --log-file "$CREE_WORK/tokenizer.log"
```

The tokenizer is trained on the training side only. Check the reported
fertility; a tokenizer that spends too many tokens on Cree morphology makes
every later stage slower and weaker.

### 9.3 Continued pretraining

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to cpt --log-file "$CREE_WORK/cpt.log"
```

This is the language-adaptation pass. The 14B profile uses 2048 context, LoRA
rank 64, and three passes. On one 80 GB card this is the longest stage. Save
checkpoints every 30 minutes:

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to cpt --save-minutes 30 --log-file "$CREE_WORK/cpt.log"
```

If the run is interrupted, rerun the same command. The trainer resumes from the
last checkpoint. Do not use `--force cpt` unless you intend to discard that
progress.

### 9.4 Instruction tuning

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to sft --log-file "$CREE_WORK/sft.log"
```

This is where identity, purpose, language identification, the English-default
answer policy, translation, and song-note behaviour are taught together.

### 9.5 Preference tuning

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to dpo --log-file "$CREE_WORK/dpo.log"
```

The preference set contains feedback pairs and echo negatives: the model is
shown that repeating the user's message is worse than answering it. If the
model starts repeating one symbol, the preference step has collapsed. Keep the
DPO learning rate at the configured `1e-6`; do not "fix" repetition by raising
it.

### 9.6 Held-out evaluation and report

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to eval --log-file "$CREE_WORK/eval.log"

python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to report --log-file "$CREE_WORK/report.log"
```

`eval` writes `out/creebench_report.json`, including chrF, exact match,
language accuracy, and the echo rate. The echo rate must be zero or the run
fails its gate. `report` writes the report card and the site bundle.

### 9.7 Export the adapter and GGUF

```bash
python train_cree_llm.py --profile big --work "$CREE_WORK" \
  --to export --log-file "$CREE_WORK/export.log"
```

The export derives the tag from the actual base model: `archer:14b` for the 14B
profiles, not `archer:7b`. It writes the merged model and the quantised GGUF
under `$CREE_WORK/out/export/`. Check:

```bash
find "$CREE_WORK/out/export" -maxdepth 2 -type f -printf '%p %s\n' | sort
```

If the merge runs out of RAM, rerun with `--merge-device cpu` and make sure the
machine has enough swap or system memory. Do not export a 7B file under a 14B
name.

## 10. Install the model on the machine that runs Archer

Copy the GGUF and its `Modelfile` to the Ollama host:

```bash
scp /data/cree_llm/out/export/archer-14b.Q4_K_M.gguf user@app-server:/opt/archer/models/
scp /data/cree_llm/out/export/Modelfile user@app-server:/opt/archer/models/
```

On the app server:

```bash
cd /opt/archer/models
ollama create archer:14b -f Modelfile
ollama list
```

Test it directly before changing the app:

```bash
ollama run archer:14b "Who are you?"
ollama run archer:14b "tânisi"
```

The second answer should be English by default. Then test an explicit request:

```bash
ollama run archer:14b "Answer in Cree: hello, how are you?"
```

Set the app's model only after the 14B GGUF exists:

```yaml
chat:
  model: archer:14b
```

Restart the app and check its health endpoint.

## 11. Generate the 1,000-line test set

Generate it after the clean stage, from the held-out split, on the server:

```bash
cd /opt/archer
python tools/make_model_benchmark.py \
  --work "$CREE_WORK" \
  --out data/eval/model_benchmark_1000.jsonl \
  --require-held-out
```

It writes exactly 1,000 rows:

| Category | Rows | Metric |
|---|---:|---|
| translate English to Cree | 200 | chrF++ |
| translate Cree to English | 200 | chrF++ |
| identify Cree | 150 | exact |
| identify English | 150 | exact |
| identify mixed | 100 | exact |
| explain a dictionary word | 150 | contains |
| identity and purpose | 25 | contains |
| greeting echo | 25 | not the prompt repeated |

`--require-held-out` fails if `parallel_test.jsonl` is missing. That is the
correct behaviour: the benchmark must not be built from the training rows.

The repository checkout already contains `data/eval/model_benchmark_1000.jsonl`
as a dry-run file. It was generated from the repository corpus before a clean
split existed, so its rows say `"held_out": false`. Regenerate it on the server
with the command above before you quote its numbers.

## 12. Score the installed model

Start Ollama and run the benchmark:

```bash
ollama serve
```

In another shell:

```bash
cd /opt/archer
python tools/eval_model_benchmark.py \
  --model archer:14b \
  --rows data/eval/model_benchmark_1000.jsonl \
  --out data/eval/model_benchmark_report.json
```

For a quick smoke test before the full 1,000:

```bash
python tools/eval_model_benchmark.py --model archer:14b --limit 40
```

The evaluator writes every prompt, expected answer, actual reply, score, and
error. It writes a category table and `data/eval/model_benchmark.png`.

Do not compare this number to a leaderboard score. It is this model, this test
set, this prompt format. A low category is a real signal to inspect the rows,
not something to round up.

Optional installed-model audit:

```bash
python tools/audit_chat_model.py
```

## 13. Train and measure the learned language identifier

The model has learned language behaviour in its weights. The app also has a
separate learned character n-gram identifier for fast routing. Train and measure
it on the same corpus:

```bash
python tools/train_language_id.py
python tools/make_language_tests.py
python tools/eval_language_id.py
```

`train_language_id.py` writes `chat/language_id_model.json` and
`data/processed/language_id_metrics.json`. It also compares itself with the old
diacritic rule and exits non-zero if the learned model is not better. Do not
ship a replacement that loses to the rule it replaces.

`eval_language_id.py` scores language, task, and answer-language decisions. Add
`--model archer:14b` to score the installed model's decisions through Ollama.

## 14. Publish the real numbers to the frontend

The publisher reads run files and copies them into frontend assets. It does not
invent a number when a file is missing:

```bash
cd /opt/archer
python tools/publish_metrics.py --work "$CREE_WORK"
```

It reads:

- `$CREE_WORK/out/creebench_report.json`
- `$CREE_WORK/out/web/cree-llm.json`
- `data/processed/language_id_metrics.json`
- `data/eval/model_benchmark_report.json`
- optional `data/eval/audit_report.json`

It writes:

```text
frontend/assets/evidence/cree-llm.json
frontend/assets/evidence/creebench.png
frontend/assets/evidence/sft_mix.png
frontend/assets/evidence/tokenizer_fertility.png
frontend/assets/evidence/model_benchmark.png
```

The JSON contains `training`, `language_id`, `benchmark`, `audit`, `figures`,
and a `sources` map. If a section is absent, the file did not exist. If the
command exits 1, it found neither training nor benchmark metrics; run the
missing stage rather than editing the JSON by hand.

A frontend page can load it with:

```javascript
fetch("/static/assets/evidence/cree-llm.json")
  .then((response) => response.json())
  .then((data) => {
    console.log(data.training?.tuned);
    console.log(data.benchmark?.categories);
  });
```

Use the values from `data.benchmark.categories` rather than retyping numbers.
The images are beside the JSON and can be used directly as chart images.

## 15. What to do when a stage fails

### Corpus size warning

This is the expected result on the current repository. The message reports the
estimated tokens and the 20M target. Add more Cree or aligned data in
`CREE_UPLOAD_DIR`, rerun the corpus stage, and check the ledger. Training still
continues and uses every verified token, but the resulting model must not be
reported as a high-end model until the corpus is actually large enough. Do not
lower `min_corpus_tokens`; the warning is the honest record.

### A source is down

Run `python tools/check_cree_sources.py --json`. Fix the endpoint or wait and
rerun the corpus stage. Do not enable missing sources for a final run.

### CUDA out of memory

- On one 24 GB card, use `--profile big24`, context 1024, LoRA rank 32.
- On 80 GB, use `--profile big`; do not raise batch size beyond what the
  preset uses.
- For export, use `--merge-device cpu`.
- Confirm no other process is using the GPU with `nvidia-smi`.

### bitsandbytes or CUDA kernel error

The torch, CUDA, and bitsandbytes versions do not match. Reinstall the torch
wheel for the driver, then reinstall bitsandbytes. `train_cree_llm.py --check`
must show a CUDA device and no CPU-only warning before the run.

### Training stops after a disconnect

Rerun the same driver command from section 8. The checkpoint and stage marker
let it resume at the stage that was interrupted. Under systemd, `Restart=on-failure`
starts it again without you. Check `$CREE_WORK/run_state.json`,
`$CREE_WORK/checkpoints/` and the stage log first. Do not delete the work folder.

### DPO produces repetition

That is a collapsed policy, not a translation quality issue. Keep the DPO
learning rate small, inspect the preference pairs, and rerun from the SFT
checkpoint with `--force dpo` only after fixing the data. The echo metric in
`eval` is the gate that catches this.

### The model still answers Cree messages in Cree

Check three things, in order:

1. The SFT log reports the language identification and answer-language example
   counts. They should not be zero.
2. The installed tag is the model you just trained, not the old `archer:7b`.
3. `eval` and `model_benchmark` show the language-policy category. If the model
   itself is wrong, add feedback corrections and retrain SFT and DPO; do not
   patch a word list into the prompt.

### Export fails

Install `cmake` and `build-essential`, free disk space, and use
`--merge-device cpu`. Check that the base weights are fully downloaded. Do not
copy a partial GGUF into the app.

## 16. Final acceptance checklist

Run these on the server, in order:

```bash
python tools/verify_training_pipeline.py
python train_cree_llm.py --check
python tools/check_cree_sources.py
python tools/build_cree_corpus.py --min-tokens 20000000 || true
cat "$CREE_WORK/raw/corpus_size.json"
cat "$CREE_WORK/processed/duplicate_audit.json"
python tools/make_model_benchmark.py --work "$CREE_WORK" --require-held-out
python tools/eval_model_benchmark.py --model archer:14b
python tools/publish_metrics.py --work "$CREE_WORK"
```

Accept the run only when:

- every expected corpus source contributed
- the corpus-size report exists and its warning was reviewed; if it is below
  20M, the result is not presented as high-end
- `verify_training_pipeline.py` passes
- SFT reports non-zero language identification and answer-language examples
- the contamination audit passes
- `duplicate_audit.json` reports zero duplicate documents, sentences and pairs,
  and its spelling-variant groups were reviewed
- the echo rate is zero
- the 1,000-row benchmark has no `"held_out": false` rows
- the report and benchmark JSON exist under the paths above
- `ollama run archer:14b` answers identity, Cree-default, explicit-Cree, and
  translation prompts correctly
- the frontend bundle was generated from those files, not typed by hand

If any item fails, the honest result is "not ready", not a lower gate.

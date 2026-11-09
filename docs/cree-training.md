# Training the Archer model on a GPU server

This is the whole procedure, from a bare server to a trained model the app can
use. It assumes you have SSH access and a CUDA GPU. Nothing here needs Colab.

The trainer is one standalone file. It downloads its own corpus, builds its own
tokenizer, trains, benchmarks, and exports. The server does not need a copy of
this repository.

---

## 1. What the server needs

| Item | Minimum | Comfortable |
|---|---|---|
| GPU | 16 GB (T4 profile) | A100 40 GB or 80 GB |
| System RAM | 32 GB | 64 GB or more |
| Disk | 80 GB free for the 7B | 150 GB for the 14B, 300 GB for the 32B |
| Python | 3.10 | 3.12 |
| Network | Hugging Face, GitHub, archive.org | the same, with an `HF_TOKEN` for rate limits |
| OS | Linux | Linux |

Two decisions are worth knowing before you start.

The base weights are cached separately from the checkpoints. If the work folder
is a network volume, put the weights on local disk with `--hf-home` so 15 GB of
model files are not read over NFS on every restart. The checkpoints stay on the
big volume.

The export merges the adapter into a full fp16 copy before quantising. That needs
roughly 4.4 GB of RAM per billion parameters: about 31 GB for the 7B, 62 GB for
the 14B, 141 GB for the 32B. The script checks this and stops with a clear
message instead of filling the machine.

---

## 2. The files to copy

These instructions assume the server does not have the repository: the files are
copied into `/data/cree`, and every command below runs from that folder. If the
repository is already checked out on the server, work from its root instead, use
`training/requirements-cree.txt`, and point `--work` at a scratch folder such as
`$HOME/cree-work`. The trainer does not care where it is run from.

Only these two are required:

```bash
scp train_cree_llm.py training/requirements-cree.txt you@server:/data/cree/
```

Optional, only if you have Jupyter on the server and prefer the notebook. The
Python file is the supported path:

```bash
scp training/notebooks/Cree_LLM_Training.ipynb you@server:/data/cree/
```

This runbook is the third file if you want it on the server:

```bash
scp docs/cree-training.md you@server:/data/cree/
```

To make one archive of everything:

```bash
tar -czf cree-server-bundle.tar.gz train_cree_llm.py training/requirements-cree.txt docs/cree-training.md
```

`train_cree_llm.py` is generated from `training/cree_llm/cells/` in the
repository and is always in step with the Colab notebook. On the server it is
the only thing you run.

---

## 3. One-time environment setup

```bash
cd /data/cree
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `python3 -m venv` reports a missing module on Debian or Ubuntu, install it
first with `apt install python3-venv` and run the block again.

Install PyTorch first, using the build that matches the driver. For CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The second line must print `True`. If it prints `False`, the CPU wheel is the
one that got installed and nothing below will use the card.

If the server has a different CUDA, pick the matching index from
`https://pytorch.org/get-started/locally/`. Then everything else:

```bash
pip install -r requirements-cree.txt
```

`bitsandbytes` needs CUDA. On a driver older than CUDA 11.8 it will not import,
and the 7B will not fit in 4-bit.

For faster and more reliable downloads, set a Hugging Face token in the shell
you run training from. Do not put it in a file that gets committed:

```bash
export HF_TOKEN=hf_...
```

---

## 4. Preflight

```bash
python train_cree_llm.py --check --work /data/cree/work
```

This prints the Python version, torch and CUDA, every GPU with its memory, every
required package with its version, and the free space on the work and cache
volumes. If something is missing it prints the exact `pip install` line and
exits non-zero. Nothing downloads until this passes.

It also tells you which profile to use:

| GPU memory | Profile |
|---|---|
| 16 GB | `--profile t4` (7B) |
| 24 GB | `--profile t4` (7B, fastest) or `--profile big24` (14B, slower but stronger) |
| 40 GB | `--profile a100` |
| 80 GB | `--profile big` for the 14B, or `--profile a100` for the 7B |
| 80 GB and a second machine for the merge | `--profile huge` for the 32B |

`--profile auto` picks `a100` on 38 GB or more and `t4` below that. Being
explicit is better on a server you are paying for. `big24` is the 14B in 4-bit
at a 1024-token context: one 24 GB card fits the training, and the merge runs on
CPU, so that machine needs about 64 GB of RAM for the export stage.

### Which card it uses

The trainer reads `nvidia-smi` before it imports torch and pins itself to the
card with the most free memory. On a shared box that is what stops the
five-gigabyte allocation failure inside `caching_allocator_warmup` when device 0
belongs to somebody else's job. The choice is printed as
`gpu: using device 1 ...`, and `--gpu N` overrides it. `CUDA_VISIBLE_DEVICES`
already set in the shell is respected, not overridden.

The training itself is single-GPU: a second card does not make the model wider
or the run faster. What it does give you is a free card to train on while the
other is busy, and the option of a stronger base: `--profile big24` trains the
14B on one 24 GB card instead of the 7B. That is the biggest quality step
available on two 4090s, and it is slower and needs about 64 GB of RAM for the
merge. Run the 7B first, read its report, then decide whether to spend the time
on the 14B.

---

## 5. Start the run

Use tmux, or nohup if tmux is not installed. Either way the run keeps going
after you close the SSH session.

With tmux:

```bash
tmux new -s cree
cd /data/cree && source .venv/bin/activate
python train_cree_llm.py --profile a100 \
  --work /data/cree/work \
  --hf-home /data/cree/hf \
  2>&1 | tee /data/cree/console.log
```

Detach with `Ctrl+b` then `d`. Return with `tmux attach -t cree`.

With nohup:

```bash
cd /data/cree && source .venv/bin/activate
nohup python train_cree_llm.py --profile a100 \
  --work /data/cree/work \
  --hf-home /data/cree/hf \
  > /data/cree/console.log 2>&1 &
echo $! > /data/cree/train.pid
```

The script writes its own timestamped log under `<work>/logs/` as well, so the
console file is a convenience, not the only record.

### Watching it

```bash
tail -f /data/cree/work/logs/$(ls -t /data/cree/work/logs | head -1)
```

and, to confirm it is still alive:

```bash
ps -p $(cat /data/cree/train.pid) -o pid,etime,pcpu,pmem,cmd
```

A stage that is downloading or packing data prints nothing for minutes at a
time. That is expected. `nvidia-smi` showing 0% during the corpus stage is also
expected: that stage is network and CPU work.

---

## 6. What it does, in order

| Stage | Work | Rough time on an A100 |
|---|---|---|
| `corpus` | Downloads the corpus sources, verifies every file, writes the licence ledger | 20 to 60 min, network bound |
| `clean` | Language filter, dedupe, document-disjoint split, leakage pass | 5 to 15 min |
| `tokenizer` | Trains SentencePiece on Cree, grafts it onto the base vocabulary | 5 to 10 min |
| `cpt` | Continued pretraining, packed Cree with 10% English replay | 1.5 to 2.5 h |
| `sft` | Instruction tuning, completion-only loss | 30 to 60 min |
| `dpo` | Preference tuning on faithfulness pairs | 20 to 40 min |
| `eval` | CreeBench against the base model and the old stack, then the language-model checks | 20 to 40 min |
| `report` | Report card, model card, figures | under a minute |
| `export` | Merge, GGUF, quantise, Modelfile, zip | 20 to 60 min |

The whole run on a 7B A100 is about 3 to 5 hours, plus whatever the corpus
download costs on your network. A T4 is roughly double. The 14B is 6 to 10 hours
on an 80 GB card.

Watch for these lines to know it is progressing:

```
[..] corpus: sources recorded ... ledger written
[..] sentences kept 66071  rejected {...}
[..] tokenizer: ...
[..] CPT: 65470 Cree sentences -> ...
[..] SFT: kept ... examples ...
[..] CreeBench: 203 held-out pairs, both directions, base and tuned
[..] the bar: the best score measured before this model on each job ...
[..] LLM checks: 6 probe pairs ...
[..] pipeline finished in ...
```

---

## 7. If it stops

Do not delete the work folder. Run the same command again.

```bash
python train_cree_llm.py --profile a100 --work /data/cree/work --hf-home /data/cree/hf
```

Every finished stage is recorded in `<work>/run_state.json` against a
fingerprint of its inputs and is skipped. An interrupted training stage continues
from its newest intact `checkpoint-N` with the optimizer and scheduler state, so
the learning rate does not jump back to its peak. A stage's root adapter is
written only after training returns, so an interrupted stage never looks
finished.

Checkpoints are written every ten minutes by default. On a slow card, shorten it:

```bash
python train_cree_llm.py --profile t4 --save-minutes 5 --work /data/cree/work
```

To stop it cleanly, `kill $(cat /data/cree/train.pid)` or `Ctrl+C` in tmux. You
lose at most the last checkpoint interval.

To redo one stage:

```bash
python train_cree_llm.py --profile a100 --work /data/cree/work --force eval
```

`--force cpt` also reruns SFT, DPO and the evaluation, because those were trained
on the CPT adapter being replaced. The previous folders are renamed, never
deleted, so a force is reversible.

To stop after a stage, for example to inspect the benchmark before spending time
on the GGUF:

```bash
python train_cree_llm.py --profile a100 --work /data/cree/work --to eval
```

---

## 7b. Pulling new code without losing the run

A `git pull` cannot remove a finished stage. Everything the run produces lives in
the folder you pass as `--work`, which is outside the repository, and the folder
the script uses when `--work` is omitted (`./cree_llm_work`) is gitignored. Code
changes arrive in the checkout; checkpoints, `run_state.json` and `out/` do not
move.

Confirm it in two commands, before and after the pull:

```bash
cat /data/cree/work/run_state.json | head -20
git pull --ff-only && git log --oneline -1
```

The pull also does not change which stages rerun. A stage is skipped only when
the fingerprint of its inputs matches the one in `run_state.json`, and those
inputs are data files, the base model, the context length, the LoRA rank and the
schedule. Editing Python does not touch them, so a bug fix is free: pull, rerun,
and the finished stages walk past.

Two things do cause a rerun, and both are visible in the first ten lines of the
log, in the stage table:

- **A changed input.** A regenerated corpus, a different base model, a different
  context length or LoRA rank invalidates the stages built on it. This is
  deliberate: a score computed from other data is not a score for this model.
- **`--force STAGE`.** The stage and everything downstream of it rerun.

When a stage reruns, its previous folder is renamed to `<stage>.stale-<time>`,
never deleted, so a mistake is reversible.

The one way to appear to lose progress is to omit `--work`. The default is
`cree_llm_work` relative to the current directory, so a run started from a
different folder looks like a fresh run. Always pass the absolute path:

```bash
python train_cree_llm.py --profile t4 --work "$HOME/cree-work" --hf-home "$HOME/cree-hf"
```

If the stage table says `cpt done`, `sft done`, `dpo done` and the run still
starts training, that is the fingerprint doing its job, and the log line names
which stage and why.

---

## 8. Where everything lands

```
/data/cree/work/
  raw/                      downloaded documents, archives, ledger.json
  processed/                cleaned sentences, splits, tokenizer, lexicon
  checkpoints/cpt/          adapter plus checkpoint-N folders
  checkpoints/sft/
  checkpoints/dpo/
  out/creebench_report.json scores, the old-setup comparison, every example
  out/llm_tests.json        stability, abstention, retention, fertility, calibration, cost
  out/report_card.md        the readable report
  out/plots/                charts from the run
  out/web/cree-llm.json     compact bundle for the project site
  out/web/corpus.json       source list and totals from the licence ledger
  out/web/llm-*.svg         the site figures, in the site's palette
  out/export/               merged model, f16 GGUF, the quantised GGUF, Modelfile, manifest, zip
  run_state.json            the stage ledger
  logs/                     one log per run
```

The export file names follow the model actually trained, not a default. A 7B run
writes `archer-7b.Q4_K_M.gguf`; a 14B run writes
`archer-14b.Q4_K_M.gguf`.

Quantising to Q4_K_M needs `llama-quantize`, which is a compiled binary. The
export installs `cmake` into the virtualenv and builds it once. On a host with no
compiler at all it does not fail: the converter writes `archer-7b.Q8_0.gguf`
instead, about 8.2 GB rather than 4.7 GB and slightly higher quality. The
Modelfile, the zip and `manifest.json` all name the file that was actually
written, so `ollama create` is unchanged either way.

### Read the results

```bash
less /data/cree/work/out/report_card.md
python -m json.tool /data/cree/work/out/creebench_report.json | less
python -m json.tool /data/cree/work/out/web/cree-llm.json | less
```

The report card carries the data table, the licence of every source, the
translation scores, the tokenizer audit and the language-model checks.

The section to read first is **Against the setup this replaces**. It lists, per
job, this model's score, the best score any part of the old three-part setup got
on the same held-out data, and whether the single model met or beat that bar.
Pass `--external-model <ollama-tag>` to put the old Cree model into the same
table; without it the bar is the base instruction model and the base model
reading the same retrieved notes.

### Send the site bundle back

This is what the Archer Bot page will be built from:

```bash
scp -r you@server:/data/cree/work/out/web ./cree-web-results
scp you@server:/data/cree/work/out/creebench_report.json ./creebench_report.json
scp you@server:/data/cree/work/out/report_card.md ./report_card.md
```

---

## 9. Install the model on the machine that runs the app

Copy the GGUF and the Modelfile out of the export folder:

```bash
scp you@server:/data/cree/work/out/export/archer-14b.Q4_K_M.gguf .
scp you@server:/data/cree/work/out/export/Modelfile .

mkdir -p assets/archer_llm
cp archer-14b.Q4_K_M.gguf assets/archer_llm/
cp Modelfile assets/archer_llm/
cd assets/archer_llm
ollama create archer:14b -f Modelfile
```

This is the only chat model the app installs. It answers conversation, song
notes, translation and script conversion, so there is no second tag to create.

Confirm it is installed and answering:

```bash
ollama list | grep '^archer'
curl -s http://127.0.0.1:11434/api/chat \
  -d '{"model":"archer:14b","messages":[{"role":"user","content":"tânisi"}],"stream":false}' \
  | python -m json.tool
```

Point the app at it in `config/config.yaml`:

```yaml
chat:
  backend: ollama
  ollama_url: http://127.0.0.1:11434
  model: archer:14b
  temperature: 0.3
  max_tokens: 500
```

Restart the app, then check the chat endpoint:

```bash
curl -s http://127.0.0.1:8000/api/chat/health | python -m json.tool
```

`model` should name the tag. If the tag is missing, the chat returns a 503 with
the `ollama create` line; it does not fall back to another model.

The same model can be used without Ollama through llama.cpp by setting
`chat.backend: llama_cpp` and `chat.gguf_path` to the same GGUF file.

---

## 10. Command reference

| Flag | What it does |
|---|---|
| `--work DIR` | Corpora, checkpoints, results. Default `./cree_llm_work` |
| `--profile NAME` | `auto`, `smoke`, `t4`, `a100`, `big`, `huge` |
| `--gpu N` | Train on card N. Without it, the run takes the card with the most free memory |
| `--base-model MODEL` | Any Hugging Face causal LM; overrides the profile |
| `--seq-len N` | Context length. Forced down to 1024 on cards under 24 GB |
| `--lora-r N` | Adapter rank. Default 64 on A100, 32 on T4 |
| `--cpt-epochs N` | Passes over the Cree text |
| `--sft-epochs N` | Passes over the instruction set |
| `--dpo-epochs N` | Preference passes. `0` disables DPO for this run |
| `--eval-n N` | Held-out pairs scored by CreeBench |
| `--save-minutes N` | Checkpoint interval. Default 10 |
| `--hf-home DIR` | Where the base weights are cached. Default `<work>/hf` |
| `--uploads DIR` | Folder of `.txt` and `.md` Cree text to add to the corpus |
| `--external-model TAG` | Ollama tag to score on the same prompts, for comparison |
| `--ollama-url URL` | Where that Ollama runs. Default `http://127.0.0.1:11434` |
| `--merge-device auto/cpu/cuda` | Where the export merges the adapter |
| `--to STAGE` | Stop after `corpus`, `clean`, `tokenizer`, `cpt`, `sft`, `dpo`, `eval`, `report` or `export` |
| `--force STAGE[,STAGE]` | Rerun stages even though they are marked finished |
| `--log-file PATH` | Where the run log goes |
| `--no-log` | Do not write a log file |
| `--check` | Verify the environment and exit |

Every run writes its own log to `<work>/logs/train-YYYYmmdd-HHMMSS.log`.

---

## 11. Troubleshooting

| Message | What it means | What to do |
|---|---|---|
| `CUDA out of memory` at load | Another job is holding the card, or the profile is bigger than it | Read the `gpu:` line it prints, then `--gpu N` for a free card, or `--profile t4` with `--seq-len 1024` |
| `CUDA out of memory` mid-training | The profile is bigger than the card | `--profile t4`, or add `--seq-len 1024` |
| `torch.cuda.is_available()` is `False` | A CPU wheel got installed | Reinstall torch from the CUDA index that matches the driver |
| `SafetensorError: header too large` | A model shard was written incompletely | Delete that one file under `<hf-home>/hub` and rerun. The loader validates every shard and names the bad file |
| Corpus stage silent for 10 minutes | Downloading or streaming a dataset | Check `nvidia-smi` is idle and the log is moving. Retry if the log has not changed for 15 minutes |
| `HTTP 429` from Hugging Face | Anonymous rate limit | Set `HF_TOKEN` and rerun |
| `bitsandbytes` import error | Driver older than CUDA 11.8, or no CUDA | Update the driver, or run a smaller profile on a newer machine |
| `Merging ... needs about 31 GB of CPU RAM` | The merge does not fit | Run `--to eval` now and export later on a bigger machine, or pass `--merge-device cuda` with an 80 GB card |
| `cmake is not installed` | llama.cpp cannot be built for the GGUF | `apt install cmake`, or skip export with `--to eval` |
| `No space left on device` | Checkpoints plus weights filled the disk | `du -sh` the work folder, free space, or rerun with `--hf-home` on another volume |
| A stage is skipped that you wanted rerun | Its marker matches its inputs | `--force STAGE` |
| The run says `interrupted, last checkpoint` but does not continue | The newest checkpoint was torn by the stop | It is skipped automatically in favour of the previous one. If none are usable, `--force STAGE` starts that stage again |

The GPU does not need to be idle between stages. The script frees the model
before the merge and empties the CUDA cache, and the merge chooses CPU or GPU
from the memory actually available.

---

## 12. Checklist

- [ ] `train_cree_llm.py` and `requirements-cree.txt` on the server
- [ ] venv created, CUDA torch installed, requirements installed
- [ ] `--check` passes and names the right GPU
- [ ] free space is at least 80 GB for the 7B, 150 GB for the 14B
- [ ] `HF_TOKEN` exported if downloads are rate limited
- [ ] run started under tmux or nohup, log path noted
- [ ] `out/report_card.md` read when the run finishes
- [ ] `out/web/` copied back for the site
- [ ] GGUF and Modelfile copied to the app machine, `ollama create` done
- [ ] `chat.model` set in `config.yaml`, `/api/chat/health` reports the tag and `/api/chat` answers

# Cree LLM notebook sources

The notebook to open in Colab is **[`Cree_LLM_Training.ipynb`](../../Cree_LLM_Training.ipynb)**
in the repository root. It is generated from the files in `cells/`. Edit the
cell sources, never the notebook, then rebuild:

```bash
python training/cree_llm/build_notebook.py
python training/cree_llm/build_notebook.py --check   # CI: fail if out of date
```

The builder compiles every code cell and rejects an empty or malformed
notebook, so a syntax error is caught before it reaches Colab.

## Running the cells locally

`run_cells.py` executes the same code the notebook contains, in one namespace,
with Colab magics skipped. It is the smoke harness:

```bash
python -m venv --system-site-packages training/.venv
training/.venv/bin/pip install -r training/requirements-cree.txt

CREE_SMOKE=1 CREE_WORK=/tmp/cree_smoke \
  training/.venv/bin/python training/cree_llm/run_cells.py
```

`CREE_SMOKE=1` selects the 0.5B model, a handful of steps and a tiny corpus. It
proves every stage runs; its scores mean nothing.

Use `--stop N` to stop after the Nth code cell (0-based, in filename order).
The indices are the order of `cells/*.py`, not the notebook's full cell count.

## Resuming after a disconnect

Colab ends a session without asking, so the notebook is written to be interrupted.
Two things make that work:

* `run_state.json` in the work directory records every finished stage against a
  fingerprint of its inputs. A stage whose marker matches is skipped, and its
  outputs are read back from disk. Change the corpus, the base model, the context
  length or the LoRA rank and the affected stages lose their markers and run
  again on their own.
* Every training stage writes `checkpoint-N` folders beside its adapter, and
  `run_training` resumes from the newest one that has a trainer state and an
  intact safetensors header. A checkpoint torn by a disconnect is skipped in
  favour of the previous one.

| Stage | Marker needs | Resume point |
|---|---|---|
| `corpus` | `raw/docs.jsonl`, `ledger.json` | not applicable, rebuilt if missing |
| `clean` | `processed/stats.json` and the split files | not applicable |
| `tokenizer` | `processed/tokenizer/`, `tokenizer_audit.json` | not applicable |
| `cpt` | `checkpoints/cpt/adapter_config.json` | `checkpoints/cpt/checkpoint-N` |
| `sft` | `checkpoints/sft/adapter_config.json` | `checkpoints/sft/checkpoint-N` |
| `dpo` | `checkpoints/dpo/adapter_config.json` | `checkpoints/dpo/checkpoint-N` |
| `eval` | `out/creebench_report.json`, keyed to the adapter | not applicable |
| `llmtests` | `out/llm_tests.json`, keyed to the adapter | not applicable |

`CREE_FORCE=eval` (or any comma-separated list of stage names) re-runs those
stages regardless of their markers. `CREE_SAVE_MINUTES=5` changes the checkpoint
interval, which defaults to roughly ten minutes of work per save.

A stage root adapter is written only after its training returns, so an
interrupted stage never looks finished. The next cell in the chain loads the
newest *finished* adapter, which means running only the SFT cell in a fresh
kernel continues from CPT instead of silently training the bare base model.

## On a GPU server

The notebook is generated for Colab. The same cells also build a single Python
file that runs the identical pipeline on a normal machine with no notebook, no
Drive and no session timer:

```bash
python training/cree_llm/build_trainer.py
scp train_cree_llm.py gpu-box:/data/

python -m pip install -r training/requirements-cree.txt
python train_cree_llm.py --check
python train_cree_llm.py --profile a100 --work /data/cree_llm
```

It takes `--work`, `--profile`, `--base-model`, `--seq-len`, `--lora-r`, the
three epoch counts, `--eval-n`, `--hf-home`, `--to STAGE` and `--force STAGE`,
and it writes a timestamped log under `<work>/logs/` as well as the terminal.
Resuming works exactly as it does in the notebook: re-run the same command.

`train_cree_llm.py` is generated, so edit `cells/` and rebuild. `--check` fails
if the file on disk no longer matches its sources.

Every run also writes the evaluation bundle for the project site to
`<work>/out/web/`: `cree-llm.json`, `corpus.json` and the SVG charts. The files
are generated from the run's own report and licence ledger, not retyped.

## Files

| Path | Purpose |
|---|---|
| `cells/` | the readable sources, one file per notebook cell |
| `build_notebook.py` | assembles and validates the `.ipynb` |
| `build_trainer.py` | assembles and validates the standalone `.py` |
| `run_cells.py` | executes the cell sources outside Colab |
| `../../Cree_LLM_Training.ipynb` | the generated notebook, this is the file you run in Colab |
| `../../train_cree_llm.py` | the generated trainer, the file you run on a server |

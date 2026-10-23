# GPU RUNBOOK — MYSONG LARGE MODEL

This is the end-to-end recipe for training MySong on the 4,571-score
deduplicated AutoHarmonizer/Wikifonia corpus on a GPU server, then exporting a checkpoint
that the CPU package can load for inference.  It is deliberately separate from
the SingSong runbook: this model is small, trains in hours rather than weeks,
and does not need the MUSDB/Demucs stack.

## 0. What you need

- Ubuntu 22.04/24.04, an NVIDIA GPU with 8 GB VRAM or more (24 GB is plenty),
  16 GB RAM and 20 GB free disk.
- CUDA 12.1 or 12.4 driver.  A CPU-only PyTorch wheel will train on CPU but
  will not use the GPU.
- The repository checked out on the server.
- `ffmpeg`, `fluidsynth`, and a General MIDI SoundFont for listening tests.

## 1. SSH in and install

```bash
ssh you@your-server
sudo apt-get update
sudo apt-get install -y git ffmpeg fluidsynth \
    python3-venv python3-pip build-essential libsndfile1 unzip p7zip-full

git clone <your-repo-url> archer
cd archer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Match the wheel to the driver.  This is the CUDA 12.1 build.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Confirm the GPU is visible before downloading anything:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

Do not continue if `torch.cuda.is_available()` is `False`.

## 2. Build the corpora

The base corpus is OpenEWLD.  The auxiliary corpus is the 5,204-score
AutoHarmonizer archive, which is distributed as a split ZIP.  The builder
removes title duplicates and melody-signature duplicates against OpenEWLD;
the latter is what prevents the test split from leaking.

```bash
mkdir -p data/mysong/model data/autoharmonizer
python -m mysong.cli corpus

curl -L --fail -o /tmp/dataset.z01 \
  https://raw.githubusercontent.com/sander-wood/autoharmonizer/main/dataset/dataset.z01
curl -L --fail -o /tmp/dataset.zip \
  https://raw.githubusercontent.com/sander-wood/autoharmonizer/main/dataset/dataset.zip
7z x -y -odata/autoharmonizer /tmp/dataset.z01

python tools/build_autoharmonizer_corpus.py \
  --root data/autoharmonizer/dataset \
  --out data/mysong/corpus_extra.jsonl \
  --workers 16
```

Expected: **4,571 usable scores** after removing 626 duplicate scores and a
handful of unparseable files.  The builder checks both normalized titles and
a transposition-invariant melody signature against the whole OpenEWLD corpus;
a title-only filter left 40 of the 44 test tunes in the training archive.

## 3. Train the tagger on the GPU

The streaming trainer keeps only per-beat arrays, not thousands of `Song`
objects, so it fits in a few GB of RAM.  `--device cuda` moves the GRU and each
batch to the GPU; the checkpoint is still written as CPU tensors.

```bash
python tools/train_mysong_tagger_large.py \
  --base data/mysong/corpus.jsonl \
  --extra data/mysong/corpus_extra.jsonl \
  --out-model data/mysong/model/mysong_large.npz \
  --device cuda \
  --epochs 60 \
  --hidden 256
```

Run it in `tmux` so an SSH disconnect does not kill the job:

```bash
tmux new -s mysong
# paste the training command
# Ctrl-b, then d to detach; `tmux attach -t mysong` to return
```

The trainer prints one line per epoch:

```
tagger epoch  1/60 loss 2.20 val root 0.50 state 0.47
```

`val root` is measured on a held-out slice of the training songs, not on
OpenEWLD dev or test.  Expect the first epoch in a few minutes on a GPU; the
whole 60-epoch run is hours, not weeks.  If you need a quick sanity run first:

```bash
python tools/train_mysong_tagger_large.py --epochs 2 --hidden 128 \
  --out-model /tmp/mysong_smoke.npz
```

The output directory must contain both:

```
data/mysong/model/mysong_large.npz
data/mysong/model/mysong_large_tagger.npz
```

`MySongModel.load("data/mysong/model/mysong_large.npz")` looks for the tagger
next to the structural model.  The `.pt` copy is optional.

## 4. Evaluate on the clean test split

```bash
python -m mysong.cli evaluate \
  --model data/mysong/model/mysong_large.npz \
  --split test
```

The command prints root/core/full/key/change F1 against the 44 OpenEWLD test
songs.  Do not tune anything on this split.  Tune on `--split dev`; the test
number is the one you report.

To use the large model as the package default:

```bash
cp data/mysong/model/mysong.npz data/mysong/model/mysong_small_backup.npz
cp data/mysong/model/mysong_large.npz data/mysong/model/mysong.npz
cp data/mysong/model/mysong_large_tagger.npz data/mysong/model/mysong_tagger.npz
```

Keep the backup; the small model is the fallback on machines without PyTorch.

## 5. Install the good-sounding renderer

The package renders through FluidSynth when both the binary and a General MIDI
SoundFont are present.  On Ubuntu the FluidR3 bank is packaged separately:

```bash
sudo apt-get install -y fluid-soundfont-gm
fluidsynth --version
ls -lh /usr/share/sounds/sf2/FluidR3_GM.sf2
```

Then render with the sampled instruments:

```bash
python -m mysong.cli render take.wav --out renders/take --style jazz \
  --renderer soundfont
```

`--renderer auto` is the default: it uses the SoundFont when available and
falls back to the numpy synth otherwise.  `--renderer synth` is useful for
deterministic tests and for machines without FluidSynth.

## 6. What “done” means

- [ ] `data/mysong/corpus_extra.jsonl` has about 4,571 scores and zero melody-signature overlap with the OpenEWLD test split.
- [ ] `data/mysong/model/mysong_large.npz` and `mysong_large_tagger.npz` exist.
- [ ] `python -m mysong.cli evaluate --model ... --split test` runs and the
      number is recorded in `docs/mysong.md`.
- [ ] `--renderer soundfont` produces finite, non-silent audio.
- [ ] A held-out take has been listened to, not just scored.
- [ ] The model card records the exact corpus counts, licenses and split.
- [ ] The OpenEWLD test split was never used for gradients, early stopping, or
      hyperparameter selection.

## 7. Honest limits

This is a stronger symbolic harmonizer, not a claim of universal superiority.
The auxiliary archive is Wikifonia-derived and its files do not carry a
blanket open license for the underlying songs; check before publishing weights.
The model still assumes one global key, symbolic supervision, and Western
popular/jazz harmony.  A listening test remains the final arbiter.

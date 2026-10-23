# TRAIN SINGSONG ON A GPU SERVER — STEP BY STEP

This is the complete runbook. It assumes a fresh Ubuntu GPU server with SSH
access. Replace the `<...>` values with yours.

The four artifacts you are producing:

```
checkpoints/singsong/soundstream.pt
checkpoints/singsong/semantic_kmeans.joblib
checkpoints/singsong/stage12.pt
checkpoints/singsong/fine.pt
```

## DATASET: WHERE GOOGLE'S DATA IS

Google's paper says:

> "11 million audio-only sources resulting in 46k hours of music."

It is an **unnamed private licensed corpus**. Google never released it and never
gave a download link. There is no public URL for the exact training data.

MUSDB18 is **not** the training set. Google used MUSDB18 only for evaluation:
150 mixes, about 10 hours, studio stems. It is the public benchmark you can
actually download.

- MUSDB18-HQ download: https://zenodo.org/records/3338373
- Dataset page: https://sigsep.github.io/datasets/musdb.html
- License: research/educational only; commercial use needs permission.

So: exact Google data = impossible publicly. Real project-owned model on
MUSDB18-HQ = possible with the steps below.

---

## 1. CONNECT AND CHECK THE GPU

```bash
ssh <user>@<server-ip>
nvidia-smi
```

Do not continue unless `nvidia-smi` lists a GPU.

Start tmux so SSH disconnects do not kill training:

```bash
tmux new -s singsong
```

Detach: `Ctrl-b` then `d`. Reattach: `tmux attach -t singsong`.

## 2. INSTALL SYSTEM PACKAGES

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg unzip wget python3-venv python3-pip
```

## 3. CLONE AND INSTALL PYTHON

```bash
git clone <YOUR-REPO-URL> Archer-Robot
cd Archer-Robot

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch wheel matching the server CUDA version. Pick the command
from https://pytorch.org/get-started/locally/. Example for CUDA 12.4:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-singsong.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
PY
```

`cuda:` must print `True`.

## 4. DOWNLOAD MUSDB18-HQ

```bash
mkdir -p data/singsong/corpus
wget -c "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1" \
  -O data/singsong/musdb18hq.zip

unzip -q data/singsong/musdb18hq.zip -d data/singsong/corpus
```

Check the layout:

```bash
find data/singsong/corpus -maxdepth 3 -name vocals.wav | head
```

You need paths like:

```
data/singsong/corpus/train/<track>/vocals.wav
```

If the files are under `corpus/musdb18hq/train/`, fix it:

```bash
mv data/singsong/corpus/musdb18hq/* data/singsong/corpus/
```

Each track directory must contain:

```
vocals.wav
drums.wav
bass.wav
other.wav
mixture.wav
```

## 4B. DOWNLOAD THE CC BY CREE CORPUS (CREE ADAPTATION)

This is separate from MUSDB. It downloads real Cree singing from uploads whose
licence is explicitly **Creative Commons Attribution (reuse allowed)**: 27
recordings, **2.196 hours**, about 1.4 GB. Attribution and source URLs are kept
in `data/cree_songs/metadata.jsonl`.

```bash
python tools/fetch_cree_songs.py --download
python tools/fetch_cree_songs.py --verify
```

Expected lines in this checkout today:

```
mixed Cree songs: 27 tracks, 2.196 h
paired Cree stems: 12 tracks, 0.526 h
```

After the full GPU separation in [GPU_RUNBOOK.md](GPU_RUNBOOK.md), the second
line becomes `27 tracks, 2.196 h`.

If the paired line says `0 tracks`, build a few Demucs pairs first:

```bash
python tools/fetch_cree_songs.py \
  --separate-ids RkRTiAiKNq0,ui1eVd_O3AI,QL5dpYIlUQc,2Xjn4jijq3A
```

Demucs is source separation, not a studio multitrack. The `0.526 h` pair set is
for a pipeline smoke test and a first adaptation run, not for claiming
production-quality stems. On the development CPU it ran at roughly 1.5x faster
than real time; use a GPU for the full corpus.

Check the paired set:

```bash
python tools/check_cree_singsong_dataset.py \
  --root data/cree_singsong --min-hours 0.2
```

**For the Cree model, use [GPU_RUNBOOK.md](GPU_RUNBOOK.md).** Do not train a
codec or transformer from scratch on 0.526 h. The Cree workflow separates all
2.196 h, reuses the base MUSDB codec, refreshes the semantic codebook on Cree
singing, and fine-tunes stage 1+2 and stage 3 from the base checkpoints.

`config/cree.yaml` points at `data/cree_singsong/train_audio` (the singer-disjoint
training view), not the full `audio/` directory. It writes artifacts to
`checkpoints/cree/`. Do not mix the CC BY corpus with the uncleared Archer MP3s.

## 5. DOWNLOAD THE SEMANTIC FEATURIZER

This is Meta's official w2v-BERT 2.0. It is the paper's feature extractor, not a
SingSong checkpoint. It does not generate music by itself.

```bash
mkdir -p data/singsong/semantic
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/w2v-bert-2.0",
    local_dir="data/singsong/semantic/w2v-bert-2.0",
)
PY
```

No SingSong weights are downloaded. The four artifacts are trained below.

## 6. CONFIGURE GRADIENT CHECKPOINTING

Edit:

```bash
nano config/config.yaml
```

Find `singsong:` and set the model block to:

```yaml
singsong:
  model:
    stage12:
      preset: base
      gradient_checkpointing: true
    fine:
      preset: base
      gradient_checkpointing: true
```

Keep `preset: base`. Do not use `xl` on one GPU.

## 7. BUILD INSTRUMENTAL AUDIO FOR THE CODEC

```bash
python -m sing_song.cli prepare-codec
```

This writes one 16 kHz mono instrumental WAV per track:

```
data/singsong/codec_audio/<track>.wav
```

## 8. TRAIN THE CODEC

Smoke test:

```bash
python -m sing_song.cli train-codec \
  --audio data/singsong/codec_audio \
  --out checkpoints/singsong/soundstream.pt \
  --steps 2000 \
  --batch-size 16 \
  --segment-seconds 1.0 \
  --device cuda \
  --workers 8
```

Full run:

```bash
python -m sing_song.cli train-codec \
  --audio data/singsong/codec_audio \
  --out checkpoints/singsong/soundstream.pt \
  --steps 200000 \
  --batch-size 16 \
  --segment-seconds 1.0 \
  --device cuda \
  --workers 8
```

Output: `checkpoints/singsong/soundstream.pt`.

## 9. FIT THE SEMANTIC K-MEANS

```bash
python -m sing_song.cli train-kmeans \
  --max-frames 2000000 \
  --device cuda
```

Output: `checkpoints/singsong/semantic_kmeans.joblib`.

## 10. TOKENIZE THE CORPUS

```bash
python -m sing_song.cli prepare
```

Outputs:

```
data/singsong/tokens/index.jsonl
data/singsong/tokens/manifest.json
data/singsong/tokens/*.npz
```

Check the manifest:

```bash
head -40 data/singsong/tokens/manifest.json
```

If every clip was dropped, your stems or levels are wrong.

## 11. TRAIN STAGE 1+2

Single GPU:

```bash
python -m sing_song.cli train-stage12 \
  --steps 200000 \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Eight GPUs:

```bash
torchrun --nproc_per_node=8 -m sing_song.cli train-stage12 \
  --steps 200000 \
  --batch-size 64 \
  --grad-accum 1 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Output:

```
checkpoints/singsong/stage12.pt
checkpoints/singsong/stage12_train.jsonl
```

Resume:

```bash
python -m sing_song.cli train-stage12 \
  --steps 200000 \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --resume checkpoints/singsong/stage12.pt
```

For a first run, replace `--steps 200000` with `--steps 20000` **and use a
separate `--checkpoint-dir checkpoints/singsong_smoke`**. The LR schedule is
built for the `--steps` value you pass; resuming a 20k smoke checkpoint with
`--steps 200000` resets the schedule, so do not mix the two unless that is what
you want.

## 12. TRAIN STAGE 3 (FINE)

```bash
python -m sing_song.cli train-fine \
  --steps 200000 \
  --batch-size 1 \
  --grad-accum 512 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Output:

```
checkpoints/singsong/fine.pt
checkpoints/singsong/fine_train.jsonl
```

Resume:

```bash
python -m sing_song.cli train-fine \
  --steps 200000 \
  --batch-size 1 \
  --grad-accum 512 \
  --device cuda \
  --dtype bf16 \
  --resume checkpoints/singsong/fine.pt
```

For a first run, use `--steps 20000`.

## 13. VERIFY AND RENDER

```bash
python -m sing_song.cli status
```

It must show `"ready": true` and no missing artifacts.

Render:

```bash
python -m sing_song.cli render \
  --input path/to/heldout_vocal.wav \
  --out renders/first-take
```

Outputs:

```
renders/first-take/mix.wav
renders/first-take/instrumental.wav
```

## 14. POST-TRAINING VERIFICATION

Run the unit tests:

```bash
python -m pytest -p no:hydra_pytest \
  tests/test_sing_song_*.py \
  tests/test_fetch_cree_songs.py \
  tests/test_cree_singsong_dataset.py
```

Check the Cree corpus and the paired smoke set:

```bash
python tools/fetch_cree_songs.py --verify
python tools/check_cree_singsong_dataset.py \
  --root data/cree_singsong --min-hours 0.2
```

Run the artifact verifier. It checks every checkpoint, loads the codec,
semantic encoder, k-means, stage 1+2, and stage 3, and reports the exact failure
if one is mismatched:

```bash
python tools/check_singsong_training.py --config config/config.yaml
```

Smoke-render 10 seconds of a held-out vocal:

```bash
python tools/check_singsong_training.py \
  --config config/config.yaml \
  --render path/to/heldout_vocal.wav \
  --seconds 10 \
  --out renders/verify.wav
```

Both commands must exit 0. The smoke render must not be silent and must not
contain NaN or infinity.

## 15. WATCH PROGRESS

```bash
nvidia-smi
tail -f checkpoints/singsong/stage12_train.jsonl
tail -f checkpoints/singsong/fine_train.jsonl
```

## 16. TIME EXPECTATIONS

- Codec 200k steps: several hours to about a day on an A100.
- Stage 1+2 200k steps at effective batch 512: weeks on one GPU.
- Stage 3 200k steps: similar or slower.
- A first 20k-step run is for listening, not Google-scale quality.

## 17. COMMON FIXES

**CUDA out of memory in stage 1+2:** lower `--batch-size` to 4 and raise
`--grad-accum` to 128. Keep effective batch 512.

**CUDA out of memory in stage 3:** batch 1 is the minimum. Use a larger GPU or
keep gradient checkpointing on.

**Wrong PyTorch/CUDA:** reinstall from the PyTorch selector; plain PyPI may give
a CPU-only wheel.

**No tracks found:** the corpus layout is wrong. Check
`find data/singsong/corpus -maxdepth 3 -name vocals.wav`.

**Tests fail with a Hydra error:** the old Hydra pytest plugin is incompatible
with Python 3.12. Run tests with:

```bash
python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py
```

## 18. WHAT IS AND IS NOT EXACT

Implemented exactly at the algorithm level: 25 Hz semantic tokens, 4x50 Hz
coarse, 8x50 Hz fine, 5122/8194 vocabularies, T5 stage 1+2, decoder-only stage
3, constrained sampling, 10 s clips, 50% overlap, σ=0.01 vocal noise, −25 dB /
+5 dB filtering, 200k steps at effective batch 512.

Not reproducible outside Google: the 46k-hour private corpus and the exact 3B
TPU checkpoint. A MUSDB18-HQ model is a real project-owned model, not Google's.

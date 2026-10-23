# Training SingSong from scratch

This is the runbook for producing the four project-owned artifacts that the
SingSong panel in the Live Demo page needs:

```
checkpoints/singsong/soundstream.pt
checkpoints/singsong/semantic_kmeans.joblib
checkpoints/singsong/stage12.pt
checkpoints/singsong/fine.pt
```

No SingSong checkpoint is downloaded anywhere in this process. The codec,
k-means codebook, stage 1+2 transformer, and stage 3 transformer are all trained
by the commands below.

## 0. Read this before renting a GPU

The paper's §4.1 says: **“11 million audio-only sources resulting in 46k hours
of music.”** It describes an unnamed, large, diverse music corpus, resampled to
44.1 kHz and mixed to mono, then cut into 10 s clips and separated with MDXNet.
The paper does **not** name the corpus, its provider, or its license, and neither
the paper nor the online supplement provides a dataset download. There is no
official Google SingSong dataset URL to give you; it was not released.

**MUSDB18 is not the training set.** Google used MUSDB18 only for evaluation and
hyperparameter comparisons (its “dev” set): 150 mixes, about 10 hours, with
studio-isolated stems. Training on MUSDB18-HQ produces a real, project-owned
benchmark model, not the 46k-hour model. Any site claiming to host “the SingSong
training dataset” is not the paper's corpus.

The paper trained on TPUs for 200k steps at effective batch 512. Reproducing
that exact checkpoint requires a comparably large licensed corpus and a
GPU/TPU cluster.

The commands and tensor layouts are the paper's. The data scale is whatever you
own or license.

The minimum useful training machine is one 24 GB CUDA GPU. An 8x A100/H100 node
is the realistic reproduction machine. CPU training is only for the unit tests
and tiny smoke runs.

Disk:

- MUSDB18-HQ archive: **22.7 GB**
- MUSDB instrumentals at 16 kHz mono PCM16: about **1.2 GB**
- Token cache for MUSDB: well under **1 GB**
- Token cache for a 46k-hour corpus: roughly **200 GB** before filesystem
  overhead

## 1. Install on the GPU machine

```bash
git clone <your-repo> Archer-Robot
cd Archer-Robot

python -m venv .venv
source .venv/bin/activate

# Install the CUDA wheel that matches the machine, from
# https://pytorch.org/get-started/locally/
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-singsong.txt
sudo apt-get install -y ffmpeg
```

Verify the GPU before doing anything expensive:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0)
PY
```

`cuda` must be `True`. If it is `False`, fix the wheel before continuing; the
training loops will silently fall back to CPU and take months.

## 2. Get a corpus

### Option A: MUSDB18-HQ (public benchmark, non-commercial research)

Download the official archive from Zenodo:

```bash
mkdir -p data/singsong/corpus
wget -c "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1" \
  -O data/singsong/musdb18hq.zip

unzip -q data/singsong/musdb18hq.zip -d data/singsong/corpus
# Verify the layout. It must be corpus/train/<track>/vocals.wav, not
# corpus/musdb18hq/train/...
find data/singsong/corpus -maxdepth 2 -type d | head
```

Required layout per track:

```
data/singsong/corpus/train/<track>/vocals.wav
data/singsong/corpus/train/<track>/drums.wav
data/singsong/corpus/train/<track>/bass.wav
data/singsong/corpus/train/<track>/other.wav
data/singsong/corpus/train/<track>/mixture.wav
```

MUSDB18-HQ is licensed for educational/research use only. Do not train a
commercial model on it without permission from the rights holders.

### Option B: your own licensed recordings

Use either layout:

```
corpus/<track>/vocals.wav
corpus/<track>/instrumental.wav
```

or, if you only have mixtures:

```
corpus/<track>/mixture.wav
corpus/<track>/vocals.wav       # optional; can be produced by Demucs
```

Mixture-only corpora need `prepare --separate`, which runs Demucs two-stem
separation. That is slower and introduces separation artifacts; studio stems
are always better.

### Option C: CC BY Cree corpus (2.196 hours, mixed songs)

Download the 27 recordings whose uploaders explicitly published them under
Creative Commons Attribution (reuse allowed):

```bash
python tools/fetch_cree_songs.py --download
python tools/fetch_cree_songs.py --verify
```

This writes `data/cree_songs/` and keeps the title, performer, source URL and
licence in `metadata.jsonl`. It is **real Cree singing**, but it is mixed, not
studio stems. Build a small paired smoke set first:

```bash
python tools/fetch_cree_songs.py \
  --separate-ids RkRTiAiKNq0,ui1eVd_O3AI,QL5dpYIlUQc,2Xjn4jijq3A
python tools/check_cree_singsong_dataset.py \
  --root data/cree_singsong --min-hours 0.2
```

The current checkout has **2.196 h** mixed Cree singing and **0.526 h** Demucs
pairs. For the full 27-track separation and the correct Cree fine-tune order,
use **[GPU_RUNBOOK.md](GPU_RUNBOOK.md)**. Do not train a codec or transformer
from scratch on the smoke set.

`config/cree.yaml` points at the singer-disjoint `data/cree_singsong/train_audio`
view and writes artifacts to `checkpoints/cree/`. It reuses the base MUSDB codec
and fine-tunes the base stage 1+2 and stage 3 checkpoints.

## 3. Put the semantic featurizer on disk

The paper conditions on **w2v-BERT** features. Those are a pretrained
self-supervised encoder, not the SingSong generative model. This repository does
not bundle or download one at inference time, so put an official checkpoint in
the configured directory:

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

This is Meta's official w2v-BERT 2.0 release. It is the featurizer the paper
uses; it is not a SingSong checkpoint and it generates no accompaniment by
itself. If you have trained your own w2v-BERT/HuBERT encoder, point
`sing_song.semantic.model_path` at that directory instead.

The adapter detects the checkpoint type automatically:

- `wav2vec2-bert`: feeds Kaldi mel `input_features` at 16 kHz;
- HuBERT / wav2vec 2.0: feeds raw `input_values`;
- audio is resampled to 16 kHz, never to the 50 Hz feature frame rate;
- 50 Hz encoder frames are average-pooled to the paper's 25 Hz.

## 4. Check the configuration

Open `config/config.yaml`, section `singsong:`. The defaults already point at:

```yaml
singsong:
  paths:
    codec: "checkpoints/singsong/soundstream.pt"
    stage12: "checkpoints/singsong/stage12.pt"
    fine: "checkpoints/singsong/fine.pt"
    semantic_kmeans: "checkpoints/singsong/semantic_kmeans.joblib"
  semantic:
    model_path: "data/singsong/semantic/w2v-bert-2.0"
    sample_rate: 16000
    native_hz: 50
    output_hz: 25
  data:
    corpus: "data/singsong/corpus"
    token_dir: "data/singsong/tokens"
```

For a 24 GB GPU, set both transformer blocks to use gradient checkpointing
(keep the existing `model:` nesting):

```yaml
  model:
    stage12:
      preset: base
      gradient_checkpointing: true
    fine:
      preset: base
      gradient_checkpointing: true
```

Leave `preset: base` unless you have a cluster. `preset: xl` is the 3B
direction and does not fit on one GPU.

Check readiness at any time:

```bash
python -m sing_song.cli status
```

It prints every missing artifact and exits non-zero if one is absent. That is
expected before training.

## 5. Build an instrumental-only folder for the codec

The codec should learn the instrumental distribution, not vocal stems. This
writes one 16 kHz mono instrumental WAV per track:

```bash
python -m sing_song.cli prepare-codec
```

For MUSDB this sums `drums + bass + other` exactly. No separation residual is
introduced. The default output is `data/singsong/codec_audio/`; pass `--out` to
put it elsewhere.

## 6. Train the SoundStream codec

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

What this trains:

- convolutional encoder/decoder at 16 kHz, 50 Hz frames;
- 12 residual vector quantizers, 1024 entries each;
- multi-scale mel reconstruction;
- RVQ commitment/codebook loss;
- multi-scale STFT hinge GAN and feature matching.

The trainer reads only a random one-second crop from each WAV per step; it does
not decode the whole track. Checkpoints are written every 2,000 steps and at
the end.

A first MUSDB smoke run can use `--steps 2000` to verify the pipeline before
committing to the long run. The final artifact must be the long run; a 2k-step
codec will sound rough.

**Expected time:** roughly 0.1–0.4 s/step on an A100 for batch 16, so plan on
several hours to a day for 200k steps. If throughput is far below that, check
that `--workers` is not competing with the GPU and that the audio is on a local
NVMe, not a network mount.

## 7. Fit the semantic k-means

```bash
python -m sing_song.cli train-kmeans \
  --max-frames 2000000 \
  --device cuda
```

This extracts intermediate w2v-BERT features from the training split only
(`data/singsong/corpus/train` for MUSDB), takes a bounded random sample, and
fits 1024 centres. The held-out `test/` split is never read by this step. It
writes:

```
checkpoints/singsong/semantic_kmeans.joblib
```

For a very large corpus, raise `--max-frames`; for MUSDB, 2M frames is already
more than the corpus contains and is a safe cap. FAISS is used when installed;
scikit-learn MiniBatchKMeans is the fallback.

This step can run before or after the codec. It only needs the semantic encoder
and the corpus audio.

## 8. Tokenize the corpus

```bash
python -m sing_song.cli prepare
```

For mixture-only corpora:

```bash
python -m sing_song.cli prepare --separate --limit 20
```

This applies the paper's filters and writes the cache:

- non-overlapping 10 s clips;
- drop instrumental peak RMS below **−25 dBFS**;
- drop clips where the vocal is **5 dB louder** than the instrumental;
- add white noise at **σ = 0.01** to the vocal only;
- semantic: `(250,)` int16 at 25 Hz;
- coarse: `(4, 500)` int16 at 50 Hz;
- fine: `(8, 500)` int16 at 50 Hz;
- `index.jsonl` and `manifest.json` under `data/singsong/tokens/`.

Check the manifest. If every clip was dropped, the corpus is too quiet, the
stems are mislabeled, or the vocal/instrumental balance is wrong. Do not start
the transformers until the cache contains usable clips.

## 9. Train stage 1+2

Single 24–80 GB GPU, effective batch 512:

```bash
python -m sing_song.cli train-stage12 \
  --steps 200000 \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

On an 8-GPU node:

```bash
torchrun --nproc_per_node=8 -m sing_song.cli train-stage12 \
  --steps 200000 \
  --batch-size 64 \
  --grad-accum 1 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Outputs:

```
checkpoints/singsong/stage12.pt
checkpoints/singsong/stage12_train.jsonl
```

Resume after a restart:

```bash
python -m sing_song.cli train-stage12 \
  --steps 200000 --batch-size 8 --grad-accum 64 \
  --resume checkpoints/singsong/stage12.pt
```

Monitor:

```bash
tail -f checkpoints/singsong/stage12_train.jsonl
```

The logged `nll` is validation cross-entropy, not a perceptual score. Do not
call a checkpoint good because NLL fell; render audio and listen.

**Scale warning:** 200k optimizer steps at effective batch 512 is the paper's
run. On one GPU that is weeks, not hours. For a first MUSDB model, use
`--steps 20000` with a separate `--checkpoint-dir checkpoints/singsong_smoke`
and accept that it is a demonstration, not a reproduction. The LR schedule is
built from the `--steps` value, so resuming a 20k checkpoint with
`--steps 200000` changes the schedule; use a fresh directory for the full run
unless that reset is intentional.

## 10. Train stage 3 (fine)

```bash
python -m sing_song.cli train-fine \
  --steps 200000 \
  --batch-size 1 \
  --grad-accum 512 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

The fine sequence is 6,000 tokens per sample (2,000 coarse prefix + 4,000 fine
targets) and the vocabulary is 8,194. Its activations are much larger than
stage 1+2. Batch 1 with accumulation is the safe default; only raise it if you
have measured VRAM headroom.

Outputs:

```
checkpoints/singsong/fine.pt
checkpoints/singsong/fine_train.jsonl
```

Resume with `--resume checkpoints/singsong/fine.pt`.

## 11. Verify all four artifacts

```bash
python -m sing_song.cli status
```

It should now print `"ready": true` and an empty `missing` list. If not, it
names the exact path that is absent or mismatched.

Render a held-out vocal:

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

Then run the post-training checks:

```bash
python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py
python tools/check_singsong_training.py --config config/config.yaml
```

The verifier loads every artifact, checks the codec round trip, loads the
semantic encoder and both transformers, and exits non-zero on the first
mismatch. For a smoke render:

```bash
python tools/check_singsong_training.py \
  --config config/config.yaml \
  --render heldout_vocal.wav \
  --seconds 10 \
  --out renders/verify.wav
```

Then open the SingSong controls inside the Live Demo page:

```
http://localhost:8000/live-demo#singsong
```

The status card should switch from “checkpoints missing” to ready, and the
upload form should return a WAV instead of 503.

## 12. Sanity gates before calling it usable

- The render is finite, non-silent, and does not collapse to one pitch.
- The instrumental follows the vocal's tempo and key at least locally.
- The mix does not clip.
- `python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py` still passes.
- A held-out vocal that was never in training produces a plausible result.
- Compare against the input with the vocal muted; if the “instrumental” is just
  a blurry copy of the vocal, the model learned the separation artifact. The
  σ=0.01 vocal noise and semantic-only conditioning are the paper's defenses;
  do not remove them.

## 13. Common failures

**`cuda` is False.** Wrong PyTorch wheel. Reinstall from the PyTorch selector,
not plain PyPI.

**CUDA out of memory in stage 1+2.** Lower `--batch-size`, keep effective batch
at 512 by raising `--grad-accum`, and set `gradient_checkpointing: true`.

**CUDA out of memory in stage 3.** Batch 1 is already the minimum. Use a larger
GPU, gradient checkpointing, or a shorter clip length (but then rebuild the
token cache and keep the layouts consistent).

**`no (vocal, instrumental) pairs found`.** The corpus layout is wrong. The
discovery code expects `corpus/train/<track>/vocals.wav` or
`corpus/<track>/vocals.wav`; check `find data/singsong/corpus -maxdepth 3`.

**`token index not found`.** Run `prepare` before `train-stage12`/`train-fine`.

**`semantic encoder not found`.** The w2v-BERT directory is missing or
`semantic.model_path` points at the wrong path.

**`k-means is not loaded`.** Run `train-kmeans`; `prepare` needs the codebook.

**Hydra pytest error on Python 3.12.** The repository's old Hydra plugin is
incompatible with Python 3.12. Use `-p no:hydra_pytest` for the test suite; it
is unrelated to SingSong.

## 14. What “exact” means here

Exact and implemented:

- 25 Hz semantic / 4x50 Hz coarse / 8x50 Hz fine streams;
- 5122 and 8194 token vocabularies with per-position masks;
- T5-style stage 1+2 and decoder-only stage 3;
- 10 s clips, 50% overlap, constrained sampling;
- noisy-vocal S-SA conditioning at σ=0.01;
- −25 dB / +5 dB clip filtering;
- 200k steps at effective batch 512 as the published recipe.

Not reproducible outside Google:

- the 46,000-hour private corpus;
- the exact 3B TPU checkpoint;
- Google's internal FAD/listening evaluation.

A model trained with this runbook belongs to this project. It is not a
downloaded SingSong checkpoint, and it should not be described as one.

# GPU RUNBOOK — SINGSONG BASE + CREE FINE-TUNE

This is the full path from a bare GPU server to four trained checkpoints, a
separated Cree corpus, and a held-out render. It is deliberately explicit: do
the steps in order, do not skip the checks, and do not start stage 1+2 until the
codec and semantic codebook exist.

**Current measured data in this checkout**

| Corpus | Amount | What it is |
|---|---:|---|
| MUSDB18-HQ | ~10 h | Studio vocal/instrumental stems, research licence |
| CC BY Cree singing | **2.196 h** | 27 real Cree singing recordings, mixed |
| Cree paired stems | **0.526 h** (12 tracks) | Demucs pseudo-stems built so far |
| Target after full GPU separation | **2.196 h** (27 tracks) | Every Cree track gets a paired stem |

The 12 paired tracks are a smoke set. The GPU job below separates the other 15,
including the 45-minute Little Island Cree round-dance recording. The full
2.196 h is paired, but training uses only the 22-track train split, **1.928 h**;
the 3.7-minute Big River Cree dev track and four test recordings stay held out.
The 45-minute Little Island Cree set is in train, where it belongs.

**Two different models, one pipeline**

1. **Base model** — trained on MUSDB18-HQ. This gives you a working codec,
   semantic codebook, stage 1+2 model and stage 3 model.
2. **Cree model** — starts from the base checkpoints, refreshes the semantic
   codebook on Cree singing, and fine-tunes stage 1+2 and stage 3 on the Cree
   pairs. It is not trained from scratch on 2.196 h; that would be worse than
   fine-tuning.

Do not mix the uncleared Archer MP3s into either dataset.

---

## 0. Machine requirements

Use a Linux box with:

- Ubuntu 22.04 or 24.04, root or passwordless sudo.
- NVIDIA GPU with **at least 24 GB VRAM**. A100 40 GB is the comfortable choice.
- **100 GB free disk**. MUSDB18-HQ zip alone is 22.66 GB; unpacked corpus,
  Cree audio, tokens and four checkpoints need room.
- CUDA 12.1 or 12.4 driver. The exact wheel must match the driver.
- 16+ CPU cores and 32 GB RAM are useful for tokenization and Demucs.

Check:

```bash
nvidia-smi
nproc
free -h
df -h
```

Do not continue if `nvidia-smi` fails or free disk is below 80 GB.

Start a persistent shell:

```bash
tmux new -s singsong
```

Detach with `Ctrl-b` then `d`. Reattach with:

```bash
tmux attach -t singsong
```

---

## 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg unzip wget curl tmux \
  python3-venv python3-pip build-essential libsndfile1
```

`libsndfile1` is required by `soundfile`; `ffmpeg` is required by `yt-dlp` and
Demucs.

## 2. Clone and install Python

```bash
git clone <YOUR-REPO-URL> Archer-Robot
cd Archer-Robot

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the CUDA PyTorch wheel. The example is CUDA 12.4; use the command from
https://pytorch.org/get-started/locally/ if your driver differs:

```bash
python -m pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-singsong.txt
```

`requirements-singsong.txt` now includes `demucs` and `yt-dlp`.

Enable gradient checkpointing before training on 24–40 GB cards. In
`config/config.yaml`, under `singsong.model`, set both entries to `true`:

```yaml
stage12:
  preset: base
  gradient_checkpointing: true
fine:
  preset: base
  gradient_checkpointing: true
```

`config/cree.yaml` already has both enabled. Keep `preset: base`; do not use
`xl` on one GPU.

Verify everything:

```bash
python - <<'PY'
import torch, torchaudio, demucs, soundfile, sklearn
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("demucs", demucs.__version__)
print("soundfile", soundfile.__version__)
PY

ffmpeg -version | head -1
python -m demucs --help >/dev/null && echo "demucs ok"
python -m yt_dlp --version
```

`cuda` must be `True`. If it is `False`, stop and fix the wheel; a CPU wheel will
make the run take weeks.

Run the unit tests once before downloading anything:

```bash
python -m pytest -q -p no:hydra_pytest \
  tests/test_sing_song_*.py \
  tests/test_fetch_cree_songs.py \
  tests/test_cree_singsong_dataset.py
```

Expected: **49 passed**. The `-p no:hydra_pytest` is required on Python 3.12;
the old Hydra pytest plugin fails during collection otherwise.

## 3. Create directories and check space

```bash
mkdir -p \
  data/singsong/corpus data/singsong/codec_audio \
  data/cree_songs data/cree_singsong \
  checkpoints/singsong checkpoints/cree \
  logs renders

df -h .
du -sh data checkpoints 2>/dev/null || true
```

The Cree audio directories are git-ignored. Keep them out of git.

## 4. Download MUSDB18-HQ

MUSDB18-HQ is the public benchmark with studio stems. It is **not** Google's
private 46,000-hour corpus, and it is research/educational use only.

```bash
wget -c \
  "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1" \
  -O data/singsong/musdb18hq.zip

unzip -q data/singsong/musdb18hq.zip -d data/singsong/corpus

find data/singsong/corpus -maxdepth 3 -name vocals.wav | head
```

You want paths like:

```text
data/singsong/corpus/train/<track>/vocals.wav
data/singsong/corpus/train/<track>/drums.wav
data/singsong/corpus/train/<track>/bass.wav
data/singsong/corpus/train/<track>/other.wav
data/singsong/corpus/train/<track>/mixture.wav
```

If the unzip produces `corpus/musdb18hq/train/`, move it up:

```bash
mv data/singsong/corpus/musdb18hq/* data/singsong/corpus/
```

Check the count:

```bash
find data/singsong/corpus/train -maxdepth 1 -mindepth 1 -type d | wc -l
```

If `vocals.wav` is missing, the scrape or unzip is wrong; do not continue.

## 5. Download the semantic featurizer

This is Meta's official w2v-BERT 2.0. It is a feature extractor, not a SingSong
checkpoint.

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

Check:

```bash
find data/singsong/semantic/w2v-bert-2.0 -maxdepth 2 -type f | head
```

The path must match `config/config.yaml` and `config/cree.yaml`:
`data/singsong/semantic/w2v-bert-2.0`.

---

## 6. Download the CC BY Cree corpus

This is the real Cree singing download. It uses uploads whose licence is
**Creative Commons Attribution (reuse allowed)**. It is not the commercial
Canyon Records/Smithsonian/SICC catalogue.

```bash
python tools/fetch_cree_songs.py --download
python tools/fetch_cree_songs.py --verify
```

Expected:

```text
mixed Cree songs: 27 tracks, 2.196 h
paired Cree stems: 12 tracks, 0.526 h
```

The 12 stems are only the smoke set. The next step builds the other 15.

If YouTube returns 429, retry later. `yt-dlp` is using the Android client
because the normal web client is rate-limited. Keep the attribution file:
`data/cree_songs/metadata.jsonl`.

## 7. Separate all remaining Cree tracks on the GPU

This is the step that fixes the “only 12 tracks have stems” limitation. It will
process the missing 15 tracks, including the 45-minute Little Island Cree set.

```bash
python tools/fetch_cree_songs.py \
  --separate-all \
  --device cuda \
  --segment 7 \
  --shifts 2
```

What the flags mean:

- `--separate-all`: process every track that does not already have a pair.
- `--device cuda`: use the GPU. On CPU, 2.196 h takes hours; GPU is the point.
- `--segment 7`: lower memory than the model default. If CUDA OOMs, use
  `--segment 5` or `--segment 3`.
- `--shifts 2`: higher-quality separation, roughly twice as slow. Use
  `--shifts 1` for a quick smoke run.

If the 45-minute track is too large for the GPU, split it into shorter source
files first or run it alone with a smaller segment:

```bash
python tools/fetch_cree_songs.py \
  --separate-ids 03xrspXK9pY \
  --device cuda --segment 3 --shifts 1
```

Check the result:

```bash
python tools/fetch_cree_songs.py --verify
python tools/check_cree_singsong_dataset.py \
  --root data/cree_singsong --min-hours 2
```

Target after a successful full separation:

```text
mixed Cree songs: 27 tracks, 2.196 h
paired Cree stems: 27 tracks, 2.196 h
```

The validator must say `OK`. If any track has no `vocals.wav` or
`instrumental.wav`, fix it before training; do not lower the minimum hours to
make the check pass.

The script creates singer-disjoint split views:

```text
data/cree_singsong/train_audio/
data/cree_singsong/dev_audio/
data/cree_singsong/test_audio/
```

`config/cree.yaml` points at `train_audio`, not the whole `audio/` directory.
That is deliberate: it keeps the test singer out of training. Do not change it
back to `audio/` just to make a command run.

---

## 8. Build codec audio

The codec learns instrumentals. Do MUSDB first:

```bash
python -m sing_song.cli prepare-codec \
  --out data/singsong/codec_audio
```

Then the Cree training split:

```bash
python -m sing_song.cli --config config/cree.yaml prepare-codec \
  --out data/cree_singsong/codec_audio
```

Check both:

```bash
ls data/singsong/codec_audio | wc -l
ls data/cree_singsong/codec_audio | wc -l
```

The Cree count should equal the number of train tracks (22 if all 27 separated;
12 in the current smoke set).

---

# PART A — BASE MODEL (MUSDB18-HQ)

## 9. Train the codec

Smoke first, then the full run. Do not jump straight to 200k.

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
BASE_STEPS=200000

python -m sing_song.cli train-codec \
  --audio data/singsong/codec_audio \
  --out checkpoints/singsong/soundstream.pt \
  --steps $BASE_STEPS \
  --batch-size 16 \
  --segment-seconds 1.0 \
  --device cuda \
  --workers 8
```

If CUDA OOMs, use `--batch-size 8` and `--segment-seconds 0.5`.

Output: `checkpoints/singsong/soundstream.pt`.

## 10. Fit the base semantic codebook

```bash
python -m sing_song.cli train-kmeans \
  --max-frames 2000000 \
  --device cuda
```

Output: `checkpoints/singsong/semantic_kmeans.joblib`.

## 11. Tokenize MUSDB

```bash
python -m sing_song.cli prepare
```

Outputs:

```text
data/singsong/tokens/index.jsonl
data/singsong/tokens/manifest.json
data/singsong/tokens/*.npz
```

Inspect the manifest:

```bash
head -40 data/singsong/tokens/manifest.json
```

If every clip was dropped, the stems or levels are wrong; do not train on an
empty manifest.

## 12. Train stage 1+2

Smoke run first:

```bash
SMOKE_STEPS=20000

python -m sing_song.cli train-stage12 \
  --steps $SMOKE_STEPS \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --workers 8 \
  --checkpoint-dir checkpoints/singsong_smoke
```

Full run:

```bash
python -m sing_song.cli train-stage12 \
  --steps $BASE_STEPS \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Output:

```text
checkpoints/singsong/stage12.pt
checkpoints/singsong/stage12_train.jsonl
```

## 13. Train stage 3

```bash
python -m sing_song.cli train-fine \
  --steps $BASE_STEPS \
  --batch-size 1 \
  --grad-accum 512 \
  --device cuda \
  --dtype bf16 \
  --workers 8
```

Output:

```text
checkpoints/singsong/fine.pt
checkpoints/singsong/fine_train.jsonl
```

Keep the base checkpoints. The Cree fine-tune starts from them.

---

# PART B — CREE FINE-TUNE

## 14. Reuse the base codec

The Cree run uses the base codec (the codec is a general audio tokenizer; the
Cree corpus is too small to train it from scratch). Copy it into the Cree
checkpoint directory:

```bash
mkdir -p checkpoints/cree
cp checkpoints/singsong/soundstream.pt \
   checkpoints/cree/soundstream.pt
```

Do **not** copy the MUSDB semantic k-means; refit that on Cree singing.

## 15. Fit the Cree semantic codebook

```bash
python -m sing_song.cli --config config/cree.yaml train-kmeans \
  --max-frames 2000000 \
  --device cuda
```

Output: `checkpoints/cree/semantic_kmeans.joblib`.

## 16. Tokenize the Cree training split

```bash
python -m sing_song.cli --config config/cree.yaml prepare
```

Outputs:

```text
data/cree_tokens/index.jsonl
data/cree_tokens/manifest.json
data/cree_tokens/*.npz
```

`config/cree.yaml` points at `data/cree_singsong/train_audio`, so dev/test
singers are not tokenized into the training set. After full separation the
train view is 22 tracks, **1.928 h**; dev is 1 track, test is 4 tracks.

## 17. Fine-tune stage 1+2 from the base checkpoint

The resume step matters. Start from the base checkpoint and train **10k more
steps**, not 200k from step zero. The shell variables do not survive a new tmux
session; set `BASE_STEPS` again if you reconnect.

```bash
BASE_STEPS=${BASE_STEPS:-200000}
FINETUNE_STEPS=$((BASE_STEPS + 10000))

python -m sing_song.cli --config config/cree.yaml train-stage12 \
  --steps $FINETUNE_STEPS \
  --batch-size 8 \
  --grad-accum 64 \
  --device cuda \
  --dtype bf16 \
  --workers 8 \
  --resume checkpoints/singsong/stage12.pt \
  --checkpoint-dir checkpoints/cree
```

What happens: the checkpoint loader restores the base model, optimizer and
scheduler; `FINETUNE_STEPS` extends the cosine schedule so there are 10k steps
left. If you pass `--steps 10000` instead, the loop exits immediately because
the base checkpoint is already at 200k.

If you used a 20k smoke base instead of 200k, set `BASE_STEPS=20000` before this
command; `FINETUNE_STEPS` becomes 30k. Keep the pair consistent.

Output: `checkpoints/cree/stage12.pt`.

## 18. Fine-tune stage 3 from the base checkpoint

```bash
python -m sing_song.cli --config config/cree.yaml train-fine \
  --steps $FINETUNE_STEPS \
  --batch-size 1 \
  --grad-accum 512 \
  --device cuda \
  --dtype bf16 \
  --workers 8 \
  --resume checkpoints/singsong/fine.pt \
  --checkpoint-dir checkpoints/cree
```

Output: `checkpoints/cree/fine.pt`.

## 19. Confirm all four artifacts exist

```bash
ls -lh \
  checkpoints/cree/soundstream.pt \
  checkpoints/cree/semantic_kmeans.joblib \
  checkpoints/cree/stage12.pt \
  checkpoints/cree/fine.pt

python -m sing_song.cli --config config/cree.yaml status
```

`status` must show `"ready": true` and an empty `missing` list.

---

## 20. Run the artifact verifier

This loads every model and checks the codec round trip:

```bash
python tools/check_singsong_training.py \
  --config config/cree.yaml
```

Exit code **0** means all four artifacts load. Exit code 1 before training is
expected; after training it is a real failure. Do not ignore it.

## 21. Render a held-out take

Pick a test-split vocal, not a training singer:

```bash
HELDOUT=$(find -L data/cree_singsong/test_audio -name vocals.wav | head -1)
echo "$HELDOUT"

python -m sing_song.cli --config config/cree.yaml render \
  --input "$HELDOUT" \
  --out renders/cree-first-take
```

If the test view is empty, the full separation did not finish; go back to step 7.

The artifact verifier can render too:

```bash
python tools/check_singsong_training.py \
  --config config/cree.yaml \
  --render "$HELDOUT" \
  --seconds 10 \
  --out renders/cree-verify.wav
```

Check the output is finite, non-silent, and the same approximate length as the
input:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf

for path in sorted(Path("renders").glob("cree*/*.wav")) + list(Path("renders").glob("cree-verify.wav")):
    audio, sr = sf.read(str(path), always_2d=True)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"{path}: {audio.shape[0]/sr:.2f}s sr={sr} peak={peak:.4f} rms={rms:.5f}")
    assert np.isfinite(audio).all(), f"non-finite samples in {path}"
    assert peak > 0.001, f"{path} is effectively silent"
PY
```

## 22. Listening gate — do this before calling the model good

Use headphones and check, in this order:

1. **Vocal bleed** — mute the input vocal and listen to the generated
   instrumental. If you can hear the Cree singer in it, the model learned
   separation artifacts. Stop and fix the stem quality.
2. **Timing** — the instrumental must follow the vocal; check the first 10 s and
   the last 10 s, not only the middle.
3. **Keys and tuning** — a round-dance drum should not drift in pitch or tempo.
4. **Style** — Northern Cree, Little Island Cree and Falynn Baptiste should not
   collapse into the same generic pad.
5. **Held-out singer** — use a test singer, never a training singer.

A low loss is not a listening test. Do not ship a checkpoint based only on
`nll`.

---

## 23. Resume, monitor and stop safely

Monitor in another tmux pane:

```bash
watch -n 5 nvidia-smi
tail -f checkpoints/cree/stage12_train.jsonl
tail -f checkpoints/cree/fine_train.jsonl
```

Read the saved step first (command at the end of this section), then set
`--steps` to **saved step + 10000**. The `220000` below is only an example for a
200k base that was interrupted at 210k.

Resume stage 1+2:

```bash
python -m sing_song.cli --config config/cree.yaml train-stage12 \
  --steps 220000 \
  --batch-size 8 --grad-accum 64 \
  --device cuda --dtype bf16 --workers 8 \
  --resume checkpoints/cree/stage12.pt \
  --checkpoint-dir checkpoints/cree
```

Resume stage 3:

```bash
python -m sing_song.cli --config config/cree.yaml train-fine \
  --steps 220000 \
  --batch-size 1 --grad-accum 512 \
  --device cuda --dtype bf16 --workers 8 \
  --resume checkpoints/cree/fine.pt \
  --checkpoint-dir checkpoints/cree
```

The `--steps` value must be **greater than the step saved in the checkpoint**,
otherwise the loop has nothing to do. Check the saved step with:

```bash
python - <<'PY'
import torch
for path in ("checkpoints/cree/stage12.pt", "checkpoints/cree/fine.pt"):
    try:
        print(path, torch.load(path, map_location="cpu", weights_only=False)["step"])
    except FileNotFoundError:
        print(path, "missing")
PY
```

## 24. Troubleshooting

**CUDA out of memory in Demucs**

```bash
python tools/fetch_cree_songs.py \
  --separate-ids 03xrspXK9pY \
  --device cuda --segment 3 --shifts 1
```

Then separate the rest with `--segment 5`.

**YouTube 429 / bot check**

Retry later; `--download` skips files that already exist. Do not commit cookies.
If it persists, download the CC BY tracks manually and place the WAVs at
`data/cree_songs/audio/<youtube-id>.wav`, then run `--verify`.

**`prepare` finds no tracks**

Check that `data/cree_singsong/train_audio` contains symlinks:

```bash
find data/cree_singsong/train_audio -maxdepth 1 -mindepth 1 -type l | wc -l
```

If it is empty, run `python tools/fetch_cree_songs.py --verify`; if it is still
empty, no track is marked `train` in `data/cree_singsong/metadata.jsonl`.

**Every clip is dropped**

Inspect `data/cree_tokens/manifest.json`. The usual causes are a silent
instrumental or vocals that are louder than the instrumental threshold. Fix the
stems; do not lower the filter thresholds just to get a non-empty manifest.

**Stage 1+2 resumes but trains zero steps**

`--steps` is not larger than the checkpoint step. Read step 23.

**`status` says semantic k-means missing**

`config/cree.yaml` has two k-means paths. Confirm:
`semantic.kmeans_path` and `paths.semantic_kmeans` both point at
`checkpoints/cree/semantic_kmeans.joblib`.

**`pytest` fails during collection with Hydra**

Use `-p no:hydra_pytest`; this is expected on Python 3.12.

**`cuda: False`**

You installed a CPU wheel. Reinstall PyTorch with the CUDA index URL that matches
the driver.

---

## 25. Serve the Cree model from the app

`--config config/cree.yaml` only affects the CLI. The HTTP app calls
`load_sing_song_config()`, which always reads `config/config.yaml`, so
`/singsong/status` and `/api/singsong/render` keep serving the MUSDB base model
until you edit that file. Change the four `singsong.paths` entries:

```yaml
  paths:
    codec: "checkpoints/cree/soundstream.pt"
    stage12: "checkpoints/cree/stage12.pt"
    fine: "checkpoints/cree/fine.pt"
    semantic_kmeans: "checkpoints/cree/semantic_kmeans.joblib"
```

Leave `singsong.semantic.model_path` at `data/singsong/semantic/w2v-bert-2.0`;
the featurizer is shared and is not fine-tuned. Also set
`singsong.model.stage12.gradient_checkpointing: true` and
`singsong.model.fine.gradient_checkpointing: true` in `config/config.yaml` if
you have not already (step 2) — it costs a little speed and saves VRAM at
inference too.

Restart the server; the engine caches its config on first use, so a running
process will not pick up the edit:

```bash
python server.py          # or: PORT=8080 python server.py
```

Verify through the same interface a browser uses:

```bash
curl -s localhost:8000/singsong/status | python -m json.tool | head -40
```

`ready` must be `true` and `error` absent. Then render an upload (`HELDOUT` is
re-derived here because a new tmux session does not keep step 21's variable):

```bash
HELDOUT=$(find -L data/cree_singsong/test_audio -name vocals.wav | head -1)

curl -s -X POST localhost:8000/api/singsong/render \
  -F "file=@$HELDOUT" \
  -F "return_mix=false" \
  -o renders/cree-http.wav

python - <<'PY'
import numpy as np, soundfile as sf
audio, sr = sf.read("renders/cree-http.wav", always_2d=True)
print(audio.shape, sr, float(np.max(np.abs(audio))))
assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.001
PY
```

A 503 with a `missing` list means a path in `config/config.yaml` is still
wrong; fix the path rather than copying files around to silence it.

## 26. Write the model card

`docs/cree-model-card.md` is the template; fill in every `TODO` before you call
the run finished. It must carry:

- the four CC BY source URLs and the per-track splits, taken from
  `data/cree_songs/metadata.jsonl`;
- the exact command lines used, so the run is reproducible;
- the Demucs version and the fact that the stems are **pseudo-stems**, not
  studio multitracks;
- the held-out test singers and the fact that they were never trained on;
- final `nll`, step counts and checkpoint SHA-256 values;
- the listening verdict from step 22, with the reviewer's role (Cree speaker or
  not) and the checkpoint ID.

Record hashes so a later checkpoint cannot be confused with this one:

```bash
sha256sum checkpoints/cree/*.pt checkpoints/cree/*.joblib \
  > checkpoints/cree/SHA256SUMS.txt
cat checkpoints/cree/SHA256SUMS.txt
```

## 27. What “done” means

Do not call the model finished until all of these are true:

- [ ] `data/cree_singsong` has 27 paired tracks, 2.196 h, validator `OK`.
- [ ] `config/cree.yaml` points at `train_audio`, not the full `audio/`.
- [ ] Base codec, k-means, stage 1+2 and stage 3 exist.
- [ ] Base codec copied to `checkpoints/cree/soundstream.pt`; Cree k-means,
      stage 1+2 and stage 3 exist.
- [ ] `python -m sing_song.cli --config config/cree.yaml status` says ready.
- [ ] `tools/check_singsong_training.py --config config/cree.yaml` exits 0.
- [ ] A held-out test-singer render is finite, non-silent, and has no audible
      vocal bleed.
- [ ] At least one Cree-speaking listener has heard the output and the result is
      recorded with the checkpoint ID.
- [ ] `config/config.yaml` points at `checkpoints/cree/*` and
      `/singsong/status` returns `"ready": true` in a restarted server.
- [ ] `sha256sum` of the four artifacts is recorded in
      `checkpoints/cree/SHA256SUMS.txt`.
- [ ] The model card names the CC BY sources, the Demucs limitation, and the
      exact test split, with no `TODO` left in
      `docs/cree-model-card.md`.

The current checkout already satisfies the download, 12-track pair, test and
metadata checks. The GPU job in step 7 is what takes it from 12 paired tracks to
the full 2.196 h paired corpus, of which the 1.928 h train split feeds the Cree
fine-tune; the base model in Part A is what makes Part B meaningful.

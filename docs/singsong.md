# SingSong in this repository

This is a paper-faithful implementation of **SingSong: Generating musical
accompaniments from singing** (Donahue, Caillon, Roberts et al., Google Research
/ IRCAM, 2023). It is a separate system from the RVC voice models used by the
live demo: those *re-sing* an existing performance, while SingSong *composes an
instrumental* that follows the performance.

The code lives in [`sing_song/`](../sing_song). The controls are part of the
Live Demo page (`/live-demo#singsong`), not a separate route, and the API is
documented at the end of this file.

---

## 1. What Google actually did

The paper models a distribution `P(instrumental | vocal)` over waveforms, but
never models waveforms directly. It models discrete codes:

| Stage | Representation | Rate | Vocabulary |
|---|---|---|---|
| semantic | w2v-BERT intermediate features, k-means quantized | 25 Hz | 1024 |
| coarse | SoundStream RVQ books 0–3, flattened frame-major | 200 codes/s (4 × 50) | 1024 each |
| fine | SoundStream RVQ books 4–11, flattened frame-major | 400 codes/s (8 × 50) | 1024 each |

The codec is 16 kHz mono SoundStream with 12 residual vector quantizers and a
50 Hz frame rate. Coarse codes alone can be decoded, but fine codes restore the
top end.

The factorization is:

```
P(Sem(y), Coarse(y) | Feats(x)) =
    P(Fine(y) | Coarse(y))
  · P(Coarse(y) | Sem(y), Feats(x))
  · P(Sem(y) | Feats(x))
```

**Stage 1+2** is one T5 encoder-decoder. It reads `Feats(x)` (250 vocal
semantic tokens for 10 s) and emits `[Sem(y); Coarse(y)]` (250 + 2000 = 2250
tokens). It is trained with teacher forcing and cross-entropy on every target
code.

**Stage 3** is a decoder-only transformer. It reads the 2000 coarse tokens as a
prefix and emits 4000 fine tokens (8 streams × 500 frames).

Both stages use:

- fixed sinusoidal positional encodings (not T5's relative bias);
- pre-RMSNorm, gated GEGLU feed-forwards, tied output embeddings;
- dropout 0.1, fixed sinusoidal positions, and vocabulary subsets that are
  disjoint per quantizer. At sampling time, every invalid subset is masked to
  `-inf`.
- temperature 0.85. No top-k in the paper; `top_k` is available here but
  defaults to 0.

The published training recipe is 200k steps at batch 512 on 10 s clips. The
base model is a `t5.1.1.base` shape (250M parameters in the paper's 32k-vocab
accounting); the XL model scales to 3B. The paper used TPUs.

### The two featurization choices that matter

Training data is made by source separation, which leaves instrumental artifacts
in the "vocal". A model can learn to reconstruct the instrumental from those
artifacts and then fail on real isolated vocals. The paper's fix, and this
implementation's default, is **Noisy / S-SA**:

1. add white noise at **σ = 0.01** to the vocal before feature extraction, at
   both training and inference time;
2. use **only semantic codes** of the vocal as conditioning (never vocal coarse
   codes).

The paper reports this improves isolated-vocal FAD by 55% relative to a naive
AudioLM adaptation. Training clips are also filtered:

- drop when the instrumental's peak RMS is below **−25 dBFS**;
- drop when the vocal's peak RMS is **5 dB louder** than the instrumental's.

### Data

Google trained on **11 million tracks / 46,000 hours** of a private corpus,
separated with MDXNet. That corpus is not public, so nobody outside Google can
reproduce that exact model or that exact data. This project trains on a corpus
you supply, with **MUSDB18-HQ** (150 tracks, ~10 hours, public) as the reference
benchmark. MUSDB18-HQ ships studio stems, so its instrumental is exactly
`drums + bass + other`; no separation error is introduced for that corpus.

---

## 2. What this repository implements

Everything in the diagram above, from scratch, in PyTorch:

| File | Responsibility |
|---|---|
| `sing_song/tokens.py` | exact 5122 / 8194 vocabularies, offsets, interleaving, per-position masks |
| `sing_song/codec.py` | SoundStream encoder/decoder + 12-book residual VQ |
| `sing_song/codec_training.py` | multi-scale STFT discriminator, hinge GAN, feature matching, mel loss, codec training |
| `sing_song/semantic.py` | w2v-BERT/HuBERT adapter, k-means tokenizer, feature extraction |
| `sing_song/semantic_training.py` | semantic k-means fitting on a corpus |
| `sing_song/model.py` | T5-style stage 1+2 model and stage 3 decoder, KV cache |
| `sing_song/sampling.py` | constrained temperature sampling, prompting |
| `sing_song/data.py` | stem discovery, Demucs separation, filtering, noising, token cache |
| `sing_song/training.py` | stage 1+2 and fine training, DDP, checkpoints, validation NLL |
| `sing_song/inference.py` | 10 s / 50%-overlap generation, mixing, WAV rendering |
| `sing_song/cli.py` | `status`, `prepare-codec`, `prepare`, `train-codec`, `train-kmeans`, `train-stage12`, `train-fine`, `render` |
| `frontend/index.html` SingSong panel | status-driven controls inside the Live Demo page |
| `server.py` `/singsong/status`, `/api/singsong/render` | integration with the app; `/singsong` redirects to the Live Demo panel |

### Deliberate differences from the paper's exact stack

These are stated plainly rather than hidden:

1. **No private corpus and no 3B TPU run.** The code supports the base and XL
   shapes, but training at the published scale requires a GPU cluster and a
   licensed corpus. This machine (16-thread Ryzen, no CUDA, 27 GB RAM) cannot
   train it, and the repository does not pretend otherwise.
2. **The semantic encoder is an adapter, not a reimplementation of w2v-BERT.**
   The paper uses a w2v-BERT checkpoint pre-trained on its corpus. A
   reimplementation of w2v-BERT would not be checkpoint-compatible. The adapter
   detects the checkpoint type: w2v-BERT 2.0 gets its own Kaldi mel frontend
   (`input_features`), while HuBERT/wav2vec 2.0 get raw `input_values`. Audio is
   resampled to the encoder's 16 kHz sample rate, never to the 50 Hz feature
   frame rate; the 50 Hz features are averaged in pairs to the paper's 25 Hz.
   Point `sing_song.semantic.model_path` at a locally trained w2v-BERT/HuBERT
   directory. The adapter takes an intermediate hidden layer and averages
   50 Hz frames down to the paper's 25 Hz. If you have a true 25 Hz encoder,
   set `native_hz: 25` and no pooling happens.
3. **The codec is trained here, not downloaded.** `sing_song/codec.py` is the
   full SoundStream architecture and `codec_training.py` is its objective.
   A 46k-hour codec needs a GPU; the code is the same recipe.
4. **The long-form scheduler uses 50% overlap with a complementary crossfade.**
   The paper describes prompting the decoder with the previous 5 s of generated
   instrumental. `generate_stage12(..., prefix_tokens=...)` and
   `generate_fine(..., prefix_tokens=...)` implement a same-layout prompt for a
   checkpoint trained with that continuation objective. The default scheduler
   generates each 10 s window independently and crossfades, because a 5 s token
   prefix is out of distribution for a model trained on 10 s block layouts.
   This is the one place where the paper's prose is underspecified; the code
   states the choice instead of silently inventing a prompt.

Nothing is downloaded at inference time, and no SingSong checkpoint is
committed or fetched. The four artifacts must be produced by this repository
(or by an equivalent training run you own).

---

## 3. Hardware and licensing reality

**Hardware for the published run:** many GPUs for weeks. A useful personal
run is much smaller: the base shape at 10 s clips fits on one 24 GB GPU with
mixed precision and gradient accumulation, but it will not match the paper's
FAD without the paper's data scale.

**Disk:** MUSDB18-HQ is 22.7 GB compressed; the 16 kHz token cache is much
smaller than the audio but still tens of GB at full scale. This machine has
66 GB free, which is enough for a MUSDB experiment but not a 46k-hour corpus.

**Licensing:** MUSDB18-HQ is distributed for non-commercial research
(CC BY-NC-SA 4.0 for the MedleyDB subset; check the Zenodo record). A model
trained on it inherits restrictions on the data use; it is not a commercial
training corpus. Use your own licensed recordings for anything else. The code
is original to this repository; the paper and the open implementations below
are cited, not vendored.

---

## 4. Setup on a training machine

```bash
python -m pip install -r requirements-singsong.txt
```

You need:

- Python 3.10+
- PyTorch with CUDA for training (CPU works for the unit tests and tiny
  inference, not for the real model)
- `ffmpeg` on PATH for Demucs / MP3
- a locally available semantic encoder directory (w2v-BERT or HuBERT)

Put the corpus under `data/singsong/corpus/`. Two layouts are supported:

```
data/singsong/corpus/train/<track>/vocals.wav
data/singsong/corpus/train/<track>/drums.wav
data/singsong/corpus/train/<track>/bass.wav
data/singsong/corpus/train/<track>/other.wav        # MUSDB18-HQ
```

or

```
data/singsong/corpus/<track>/vocals.wav
data/singsong/corpus/<track>/instrumental.wav
```

A folder of mixtures is also accepted, but only with `prepare --separate`
(Demucs two-stem); otherwise the pipeline refuses rather than training on the
mixture as if it were an instrumental.

Check readiness at any time:

```bash
python -m sing_song.cli status
```

It prints the exact missing paths and exits non-zero if anything is missing.

---

## 5. Training

For the complete GPU runbook — corpus download, featurizer setup, all five
commands, resume behavior, time/VRAM expectations, and troubleshooting — see
**[singsong-training.md](singsong-training.md)**. The summary below is the
short version.

### 5.1 SoundStream codec

```bash
python -m sing_song.cli train-codec \
  --audio data/singsong/codec_audio \
  --out checkpoints/singsong/soundstream.pt \
  --steps 200000 --batch-size 16 --segment-seconds 1.0
```

The objective is multi-scale mel reconstruction + RVQ commitment/codebook loss
+ multi-scale STFT hinge GAN + feature matching. The checkpoint includes the
codec config, so inference reconstructs the same architecture.

### 5.2 Semantic k-means

```bash
python -m sing_song.cli train-kmeans --max-frames 2000000
```

This extracts intermediate w2v-BERT/HuBERT features from the corpus, samples
frames, and fits 1024 centres (FAISS when available, scikit-learn otherwise).
The result is `checkpoints/singsong/semantic_kmeans.joblib`.

If you are training the semantic encoder itself, do that first and point
`singsong.semantic.model_path` at its output directory. The paper pre-trained
w2v-BERT on the same corpus; that pre-training is a separate, much larger run
and is not reimplemented here.

### 5.3 Token cache

```bash
python -m sing_song.cli prepare
```

This applies the paper's 10 s / non-overlap / −25 dB / +5 dB filters, adds
σ=0.01 noise to the vocal, and writes `semantic` (250,), `coarse` (4, 500) and
`fine` (8, 500) arrays per clip plus `index.jsonl`.

For mixture-only corpora:

```bash
python -m sing_song.cli prepare --separate --limit 20
```

### 5.4 Stage 1+2

Single GPU:

```bash
python -m sing_song.cli train-stage12 \
  --steps 200000 --batch-size 8 --grad-accum 64
```

A cluster (effective batch 512):

```bash
torchrun --nproc_per_node=8 -m sing_song.cli train-stage12 \
  --steps 200000 --batch-size 64 --grad-accum 1
```

Checkpoints go to `checkpoints/singsong/stage12.pt`; the log is
`stage12_train.jsonl`. Resume with `--resume checkpoints/singsong/stage12.pt`.

### 5.5 Stage 3 (fine)

```bash
python -m sing_song.cli train-fine \
  --steps 200000 --batch-size 1 --grad-accum 512
```

The fine stage has a 6000-token sequence (2000 coarse + 4000 fine) and an 8194
vocabulary, so its activations are much larger per sample than stage 1+2. Batch
1 with accumulation is the safe default; raise it only with enough GPU memory.

### 5.6 What "done" looks like

The paper's quantitative metric is Fréchet Audio Distance on mixes against
MUSDB18-test, reported separately for isolated (`FAD_i`) and source-separated
(`FAD_s`) vocal inputs. The training loop logs validation NLL, which the paper
notes correlates poorly with perception; do not treat a falling NLL as proof
of musical quality. The listening test is the real criterion.

A reasonable engineering gate before calling a checkpoint usable:

- every voice/instrumental pair in a held-out set generates 10 s without NaNs;
- the output is not silent and does not collapse to one pitch;
- the generated instrumental follows the vocal's tempo and key at least
  locally;
- the mix is not clipping;
- `python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py` still passes.

---

## 6. Inference

```bash
python -m sing_song.cli render --input vocal.wav --out renders/my-take
```

Outputs `renders/my-take/mix.wav` and `instrumental.wav` at the input's sample
rate. The original vocal is mixed at full quality; it is never codec-round-
tripped. Long inputs are capped by `singsong.inference.max_seconds` (120 s by
default).

The controls are at `http://localhost:8000/live-demo#singsong`; the API is:

```
GET  /singsong/status
POST /api/singsong/render     multipart file=<audio>, return_mix=true|false, band=<style>
```

The old `/singsong` URL redirects to that anchor.

`/singsong/status` never loads weights and always reports the missing artifact
list.  It also reports `band_available: true` and `engine: "built-in"`, because
this route never has to refuse: with no weights installed, the panel renders
with the built-in symbolic band engine (the chord model and bands behind
MySong, default `round_dance`) instead of returning 503.  `X-SingSong-Engine` on
the response says which engine ran: `neural` for the trained cascade, `built-in`
otherwise.  The four artifacts still switch the panel to the trained cascade as
soon as they exist, and the panel says so in its status card instead of reading
as broken.

Both engines return the band at the app's one reference level plus the
`X-SingSong-Band-Gain-Db` placement for the take, and both build the mix with
`mysong.accompaniment.mix_with_vocal`, so the voice is over the band in the
panel and in the saved WAV.  One upload renders both stems in a single pass
(`_singsong_render_cache`), so the page's second request for the band it plays
is a cache hit rather than a second model pass.  The same path is used by the
trained cascade: its decoder output is the band, and it is levelled against the
take before anything is returned.

---

## 7. Exact tensor contracts

For a 10 s clip:

```
Stage 1+2 input           [B, 250]                 semantic vocal codes
Stage 1+2 target          [B, 2250]                [250 semantic | 2000 coarse]
Stage 1+2 vocab           5122                     1024 sem + 4*1024 coarse + 2
Stage 3 coarse prefix     [B, 2000]                4 coarse books, interleaved
Stage 3 fine target       [B, 4000]                8 fine books, interleaved
Stage 3 vocab             8194                     8*1024 fine + 2
Codec output              16 kHz mono, 50 frames/s
```

At every target position the sampler masks all but that position's own 1024-token
block. `sing_song/tokens.py` is the single source of truth for these offsets;
`tests/test_sing_song_tokens.py` holds it to the paper's numbers.

---

## 8. Tests

```bash
python -m pytest -p no:hydra_pytest tests/test_sing_song_*.py -v
```

The tests need no checkpoints, no GPU, and no audio corpus. They cover:

- exact sequence lengths and vocabulary offsets;
- target build/split round trips and rejection of invalid tokens;
- stage 1+2 and stage 3 forward shapes and gradients;
- **KV-cache logits equal teacher-forced logits** (1.5e-7 max difference);
- constrained sampling never emits an out-of-range token;
- codec encode/decode/reconstruct shapes and gradients;
- corpus discovery, −25 dB / +5 dB filtering, non-overlap, noise sigma;
- token-cache writing and dataset loading;
- window weighting and the end-to-end cascade with tiny models.

The `-p no:hydra_pytest` flag is required in this environment because the
installed Hydra pytest plugin is incompatible with Python 3.12; it is unrelated
to SingSong.

---

## 9. References

- Donahue, Caillon, Roberts, Manilow, Esling, Agostinelli, Verzetti, Simon,
  Pietquin, Zeghidour, Engel. *SingSong: Generating musical accompaniments from
  singing.* arXiv:2301.12662, 2023.
  <https://arxiv.org/abs/2301.12662>
- Online supplement: <https://storage.googleapis.com/sing-song/index.html>
- Borsos et al. *AudioLM: a language modeling approach to audio generation.*
  arXiv:2209.03143, 2022.
- Zeghidour et al. *SoundStream: an end-to-end neural audio codec.* 2021.
- Chung et al. *w2v-BERT.* 2021.
- Rafii et al. *MUSDB18.* 2017.
- Open reference implementations consulted for the open-component
  substitutions, not vendored:
  - <https://github.com/zhvng/open-musiclm> (MIT)
  - <https://github.com/jihoojung0106/open-singsong> (MIT)

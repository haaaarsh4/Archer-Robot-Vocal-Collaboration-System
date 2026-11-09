# Cree SingSong model card

Status: **template**. Fill every `TODO` from the GPU run in
[singsong-training.md](singsong-training.md) before publishing. Anything still marked
`TODO` means the run is not finished.

## 1. What the model does

Given a monophonic Cree singing recording, the model generates an instrumental
accompaniment that follows the vocal. It does **not** re-sing the input, translate
it, or generate new Cree lyrics. Architecture and training recipe follow
Donahue et al., *SingSong: Generating musical accompaniments from singing*
(2023): a SoundStream codec, a w2v-BERT + k-means semantic tokenizer, a T5-style
stage 1+2 model and a decoder-only stage 3 model.

## 2. Training and evaluation data

All Cree audio is downloaded from YouTube uploads whose licence is
**Creative Commons Attribution (reuse allowed)**. No all-rights-reserved Canyon
Records, Smithsonian Folkways or SICC material is included. Per-track attribution
(source URL, uploader, title, licence) lives in `data/cree_songs/metadata.jsonl`
and is reproduced in section 7.

| Split | Recordings | Duration | Use |
|---|---:|---:|---|
| train | 22 | 1.928 h | codec corpus, semantic k-means, stage 1+2, stage 3 |
| dev | 1 | 0.061 h | validation / early stopping |
| test | 4 | 0.207 h | held-out render and listening gate |
| **total** | **27** | **2.196 h** | |

Held-out recordings (never trained on):

| Split | Recording | Performer | Length |
|---|---|---|---:|
| dev | `yOuSag1GfZE` | Big River Cree | 3.66 min |
| test | `liCZRqx202M` | Falynn Baptiste | 4.39 min |
| test | `l4Td_Ur0rUE` | Falynn Baptiste | 2.95 min |
| test | `r0tFyijTkLk` | Falynn Baptiste | 2.93 min |
| test | `zSGqAfCiJHI` | Jaguar Bird / Cree vocal | 2.18 min |

Splits are **singer-disjoint**: no performer appears in more than one split, so a
test score is not a memorised training singer.

**Stems are pseudo-stems.** Vocal/instrumental pairs come from Demucs 4.x
(`htdemucs`, two-stem mode) run on mixed masters, not from studio multitracks.
Bleed and smearing are therefore part of the training target, and the model can
learn to reproduce separation artifacts. This is the single largest quality
limitation of the Cree model.

The base model additionally uses **MUSDB18-HQ** (~10 h, 150 tracks) for the
codec, semantic codebook and both transformers. MUSDB18-HQ is distributed for
research and educational use; check its terms before any commercial deployment.

## 3. Training recipe

| Stage | Initialisation | Steps | Batch × accum | Hardware | Command |
|---|---|---:|---|---|---|
| codec (base) | random | TODO | TODO | TODO | `train-codec` (see runbook §9) |
| semantic k-means (base) | random | 2,000,000 frames | n/a | TODO | `train-kmeans` (§10) |
| stage 1+2 (base) | random | TODO | 8 × 64 | TODO | `train-stage12` (§12) |
| stage 3 (base) | random | TODO | 1 × 512 | TODO | `train-fine` (§13) |
| semantic k-means (Cree) | refit on Cree train split | 2,000,000 frames | n/a | TODO | `train-kmeans --config config/cree.yaml` (§15) |
| stage 1+2 (Cree) | base checkpoint | +10,000 | 8 × 64 | TODO | `train-stage12 --resume …` (§17) |
| stage 3 (Cree) | base checkpoint | +10,000 | 1 × 512 | TODO | `train-fine --resume …` (§18) |

The Cree codec is the base codec copied unchanged: 2.196 h is far too little to
train a codec, and re-training it would make the stage models incompatible with
the base checkpoints.

Final losses: train `TODO`, validation `TODO` (stage 1+2), `TODO` / `TODO`
(stage 3). Optimiser step at which training stopped: `TODO`.

## 4. Evaluation

**Listening gate** (runbook §22) on the four test recordings, headphones:

| Check | Verdict | Notes |
|---|---|---|
| No audible vocal bleed in the instrumental | TODO | |
| Timing follows the vocal (first 10 s and last 10 s) | TODO | |
| Round-dance tempo / pitch does not drift | TODO | |
| Styles do not collapse into one generic pad | TODO | |
| Held-out singer only | TODO | |

Reviewer: `TODO` (Cree speaker: yes/no). Checkpoint ID: `TODO`.

Loss is not a quality metric here; do not report `nll` alone.

## 5. Artifacts

| File | SHA-256 |
|---|---|
| `checkpoints/cree/soundstream.pt` | TODO |
| `checkpoints/cree/semantic_kmeans.joblib` | TODO |
| `checkpoints/cree/stage12.pt` | TODO |
| `checkpoints/cree/fine.pt` | TODO |

Produce the table with:

```bash
sha256sum checkpoints/cree/*.pt checkpoints/cree/*.joblib \
  > checkpoints/cree/SHA256SUMS.txt
```

## 6. Limitations and responsible use

- **Small corpus.** 1.928 h of training audio across 22 recordings and roughly a
  dozen performers. Expect over-fitting to those singers' drum patterns and
  keys; a new style or a solo hand-drum singer will sound generic.
- **Pseudo-stems.** Vocal bleed and smeared transients are in the targets.
- **Language coverage.** Plains Cree / nêhiyawêwin round-dance and Christmas
  songs dominate. Swampy Cree, Woods Cree, Michif and syllabics-first repertoire
  are not represented.
- **Consent is not implied by a licence.** CC BY permits training and
  adaptation, not disregard for community protocol. Ceremonial or restricted
  songs must be excluded even when the upload is CC BY, and a recording must be
  removed from future runs if a singer or Nation asks.
- **Attribution is mandatory.** Any published demo or generated audio must credit
  the performers and uploaders listed below.
- **Not a substitute for musicians.** This is an accompaniment sketch tool. It
  should not be presented as Cree music authored by the model.

## 7. Attribution

Every training recording, with its source URL and licence, is listed in
`data/cree_songs/metadata.jsonl`; that file is the authoritative list and must be
shipped next to the model. Performer totals:

| Performer / group | Recordings | Minutes |
|---|---:|---:|
| Little Island Cree | 1 | 45.12 |
| Northern Cree | 8 | 25.83 |
| Falynn Baptiste | 3 | 10.27 |
| Confederation Park School | 1 | 8.60 |
| Cree Confederation | 3 | 6.76 |
| Powwow Times singers | 2 | 6.01 |
| Hall of Hidden Wisdom | 1 | 4.21 |
| Young Spirit | 1 | 3.94 |
| rabbit86musician | 1 | 3.93 |
| Big River Cree | 1 | 3.66 |
| Mountain Cree | 1 | 3.45 |
| Bad Lands Cree | 1 | 3.39 |
| Iron Boy Style Singers | 1 | 2.88 |
| Jaguar Bird / Cree vocal | 1 | 2.18 |
| Brian Tomaszewski | 1 | 1.55 |

Licence for all of the above: Creative Commons Attribution (CC BY), as selected
by the uploader on YouTube. See `data/cree_songs/metadata.jsonl` for the exact
`license_url` of each recording.

# Cree-focused SingSong: data inventory and plan

## 0. Decisions recorded

- **Dialect:** Plains Cree / nêhiyawêwin.
- **Targets:** both a Cree-style instrumental model (SingSong) and a Cree
  singing-voice model (RVC/TTS). These are separate models with separate data
  and evaluation.
- **Existing Archer recordings:** no stems and no confirmed training
  permission. Do not assume the website copies may be used for training.

As of 19 October 2026 there is a real CC BY Cree singing corpus in
`data/cree_songs/`: **27 recordings, 2.196 hours**, downloaded by
`tools/fetch_cree_songs.py`. Twelve tracks have Demucs pseudo-stems
(**0.526 hours**) in `data/cree_singsong/`. That is enough to refit the semantic
codebook and smoke-test a stage 1+2 / stage 3 fine-tune on the reused base
codec; it is not enough to train a codec, and it is not the 5-hour
multi-singer target.

The existing Archer recordings are still a different matter: they have no
confirmed training permission and must not be treated as Cree training data
just because they are used on the website.

## 1. What is actually in this checkout

Measured from the audio files in the repository, excluding duplicate frontend
copies and virtual environments:

| Material | Duration | What it is | Usable for SingSong training? |
|---|---:|---|---|
| VocalSet11 | **9.29 h** (3,614 files, 11 singers) | Generic Western vocal exercises: scales, arpeggios, long tones, excerpts | No. No instrumentals, and not Cree. Useful for voice/timbre work, not accompaniment. |
| Archer / Indigenous song takes | **44.9 min** (10 tracks) | Four long Archer Pechawis works (40.7 min) plus six short takes by Indigenous artists | Not as-is. These are mixed masters, not vocal/instrumental stems, and the tracks have no per-track language/consent metadata. |
| Verified spoken Cree | **1.5 min** (9 files) | Cree Literacy Organization recordings | No. Speech, not singing. Useful for ASR/pronunciation evaluation. |
| **CC BY Cree singing corpus** | **2.196 h** (27 recordings, 15 performers/groups) | Northern Cree, Little Island Cree, Cree Confederation, Mountain Cree, Young Spirit, Falynn Baptiste and others | Yes for semantic/k-means fitting and a stage 1+2 / stage 3 fine-tune; too small to train a codec. Mixed songs; 12 tracks have Demucs pseudo-stems. |
| **Demucs paired Cree stems** | **0.526 h** (12 tracks) | Vocal/no-vocal pairs generated from the CC BY mixes | Smoke-test only. Separation artifacts, not studio stems. |
| Everything else | ~4 min | DSP demos, neural voice samples, generated auditions | No. Not training data. |
| **Total singing audio in checkout** | **12.24 h** | 10.04 h pre-existing + 2.196 h CC BY Cree corpus | Cree singing: **2.196 h**; paired Cree stems: **0.526 h**. |

**Bottom line:** **12.24 hours of singing audio total**, including **2.196
hours of real Cree singing** under CC BY. **0.526 hours** of that is currently
paired with Demucs pseudo-stems. The old 1.5-minute figure was only the local
spoken-Cree check-in; it was never the size of the Cree music world.

The four long Archer works are not usable training data: they are mixed songs,
they would need separation, and the repository does not contain written
permission to train on them merely because they are used on the website. The CC
BY corpus above is the usable material.

### The 1.5 minutes was a local-corpus measurement, not the Cree music world

The 1.5 minutes was only the **verified spoken Cree** that was checked into this
repo at the time. It is useless for training a singing model and was never
evidence that Cree singing is rare. There is a large commercial and community
Cree music scene, and there are CC BY recordings usable for training.

### Open CC BY Cree singing corpus (now downloaded)

`tools/fetch_cree_songs.py` downloads 27 recordings whose uploaders selected
YouTube's **Creative Commons Attribution (reuse allowed)** licence. The script
records attribution, source URL, licence, singer/group, style and split in
`data/cree_songs/metadata.jsonl`. It does not download the all-rights-reserved
Canyon Records, Smithsonian Folkways, or SICC albums.

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

Download and verify:

```bash
python tools/fetch_cree_songs.py --download
python tools/fetch_cree_songs.py --verify
```

Build Demucs pseudo-stems for selected tracks, or separate the whole corpus on
a GPU:

```bash
# small smoke set
python tools/fetch_cree_songs.py \
  --separate-ids RkRTiAiKNq0,2Xjn4jijq3A,QL5dpYIlUQc,ui1eVd_O3AI

# full corpus (27 tracks, 2.196 h); use --shifts 2 for final quality
python tools/fetch_cree_songs.py \
  --separate-all --device cuda --segment 7 --shifts 2
```

**Attribution is part of the licence.** Keep the source URL and uploader with
the audio, and credit them in any published model card or demo. CC BY permits
adaptation and training; it does not erase the need to respect community
protocols for ceremonial or restricted songs. If a singer or Nation asks for a
recording to be removed, remove it from future runs.

Other Cree singing examples (all rights reserved unless stated):

- Young Spirit, *Mewasinsational – Cree Round Dance Songs* —
  https://canyonrecords.bandcamp.com/album/mewasinsational-cree-round-dance-songs
  (includes “Cree Prayer Song”; stream or paid download, no stems)
- Northern Cree, *Ewipihcihk – Cree Round Dance Songs* —
  https://canyonrecords.bandcamp.com/album/ewipihcihk-cree-round-dance-songs
- Northern Cree, *It's A Cree Thing – Cree Round Dance Songs* —
  https://canyonrecords.bandcamp.com/album/its-a-cree-thing-cree-round-dance-songs
- Northern Cree, *When It's Cold – Cree Round Dance Songs* —
  https://canyonrecords.bandcamp.com/album/when-its-cold-cree-round-dance-songs
- Northern Cree Handgame Singers, *Onion Lake: Plains Cree Handgame Songs* —
  SICC store listing: https://store.sicc.sk.ca/products/onion-lake-plains-cree-handgame-songs-cree
- William Burn Stick, “Plains Cree Prisoner's Song” —
  https://www.youtube.com/watch?v=SAWj65gltfY (Smithsonian Folkways recording)
- Northern Cree, Tiny Desk (Home) Concert —
  https://www.youtube.com/watch?v=maDO3z41wXA

These are for listening and rights conversations. Buying an album or streaming a
track does **not** grant permission to extract stems, train on it, or
redistribute a model. Some round-dance, handgame, prayer, and ceremonial songs
carry restrictions beyond copyright; ask the artists and the relevant community
before using them.

## 2. What "Cree-focused" means technically

SingSong generates an **instrumental** conditioned on a vocal. It does not sing
lyrics. A Cree-focused SingSong therefore needs:

```
Cree vocal stem  +  matching instrumental stem  +  metadata
```

The language matters because it changes phrasing, vowel colour, vocables,
melisma, rhythm and repertoire. The model still learns instrumental style from
the paired instrumental target.

If the goal is a **Cree singing voice**, that is the RVC/TTS side of this
project, not SingSong. Train that separately, with separate consent and
pronunciation evaluation.

## 3. Minimum viable Cree dataset

```
data/cree_singsong/
  audio/
    <song_id>/
      vocals.wav            # isolated, 44.1 kHz or 48 kHz, stereo/mono
      instrumental.wav      # exact same length, same start, no vocal bleed
  metadata.jsonl
```

One metadata row per song:

```json
{
  "song_id": "cree_0001",
  "singer_id": "singer_003",
  "community": "community name, with permission",
  "dialect": "Plains Cree (nêhiyawêwin)",
  "style": "round dance | powwow | folk | other",
  "language": "cree",
  "bpm": 112.0,
  "key": "F#m",
  "duration_s": 214.5,
  "consent_id": "signed-form-id",
  "license": "training-only | research | commercial",
  "split": "train | dev | test"
}
```

Validate the dataset before training:

```bash
python tools/check_cree_singsong_dataset.py \
  --root data/cree_singsong --min-hours 5
```

It checks missing stems, mismatched lengths/sample rates, duplicate song IDs,
missing consent/licence fields, and singer leakage across splits. It exits
non-zero on failure.

Rules:

- **Singer-disjoint splits.** No singer appears in both train and test. No song
  appears in two splits. The fetch tool creates `train_audio/`, `dev_audio/` and
  `test_audio/` symlink views; `config/cree.yaml` points at `train_audio/`, not
  the full `audio/` directory, so test singers cannot leak into training. After
  full separation the split is 22 train tracks (**1.928 h**), 1 dev track and 4
  test tracks.
- **Stems, not mixes.** A full mix cannot supervise the instrumental target.
  Separation is a fallback, not a substitute for multitracks.
- **Metadata is not optional.** Without singer/community/license IDs you cannot
  audit leakage, pay people correctly, or honour a withdrawal request.
- **Private by default.** Do not commit Cree audio to git. Keep it in an
  access-controlled data directory. The manifest can live in git without audio.

### How much data?

These are engineering targets, not guarantees:

- **5–10 h** of clean, paired, diverse Cree singing: enough for a first
  style-adaptation fine-tune and a meaningful listening test.
- **20–50 h** across multiple singers and styles: a robust Cree-focused model.
- **Hundreds of hours**: only with a large licensed/commissioned archive.

Five hours from one singer will overfit that singer. Five hours across ten
singers is much more useful.

## 4. Recording protocol for new Plains Cree material

SingSong needs the vocal and instrumental to be **sample-aligned**. Record them
as multitracks, not as a stereo mix:

- Record to a click track at a declared BPM; log the key and tuning reference.
- Export every stem at the **same sample rate, same start sample, same length**.
- Keep the lead vocal dry: no reverb, delay, compression, or autotune on the
  exported vocal stem. Effects can be printed on a separate reference mix.
- Keep the instrumental as a true instrumental: no guide vocal, no vocal bleed
  from headphones.
- Leave 0.5–1 s of silence at the start of every file so alignment is auditable.
- Deliver 44.1 kHz or 48 kHz WAV; the pipeline resamples to 16 kHz mono.
- Log: singer ID, community, dialect, style, BPM, key, song title, language,
  consent ID, license, and whether the song is restricted/ceremonial.
- For the voice model, also capture isolated sustained vowels, spoken
  pronunciation of the lyrics, and several takes at different dynamics.

Do not record or use ceremonial, sacred, or restricted songs unless the proper
authority has given explicit permission and the community has approved the use.

## 5. Consent, sovereignty, and rights

This is not optional paperwork. It determines whether the dataset can exist.

- Follow **OCAP** (First Nations principles of Ownership, Control, Access and
  Possession) and the relevant Nation's own research protocol.
- Get **written, specific consent** for: recording, model training, evaluation,
  redistribution, and commercial use (separately). Website playback permission
  is not training permission.
- Pay singers, knowledge keepers, translators, and reviewers.
- Give the community a **review and withdrawal path**. If consent is withdrawn,
  remove the item from future training runs and document how checkpoints will be
  handled.
- Do not train on **ceremonial, sacred, or restricted songs** without explicit
  protocol and permission from the proper authority. Some songs belong to a
  drum, family, or ceremony, not to a public dataset.
- Do not scrape arbitrary YouTube, Spotify, TikTok, or commercial CDs. A
  stream or purchase is not a training license. The exception is an upload
  whose licence explicitly says CC BY (reuse allowed); the acquisition script
  only accepts that case and keeps attribution.
- Store consent forms and audio separately from the public repository.

Public archives worth talking to (none of these is an open training license):

- Saskatchewan Indigenous Cultural Centre audio collection:
  https://www.sicc.sk.ca/audio-collection
- Cree Literacy Network songs/resources: https://creeliteracy.org/
- Smithsonian Folkways, *Indian Music of the Canadian Plains*:
  https://folkways.si.edu/indian-music-of-the-canadian-plains/american-indian/album/smithsonian
- Library and Archives Canada, Cree recordings:
  https://recherche-research.bac-lac.gc.ca/eng/public/list/46438
- Canadian Museum of History, Cree songs recorded by Christian Leden:
  https://www.historymuseum.ca/collections/archive/3351891

The honest conclusion after the public-source search: there is **no openly
licensed Plains Cree corpus with studio-isolated vocal/instrumental pairs**.
The 2.196-hour CC BY corpus is real, but it is mixed; its 0.526 hours of Demucs
stems are approximations. A production model still needs commissioned or
licensed multitracks, or a community partnership. The CC BY corpus is enough to
refit the semantic codebook, reuse the MUSDB codec and smoke-test the paired
pipeline now, instead of waiting for the full 5-hour target.

## 6. Two datasets, not one

| Model | Data | Minimum first milestone |
|---|---|---|
| SingSong (instrumental) | paired Cree vocal + instrumental stems | 5 h across 5+ singers and 3+ styles |
| Cree singing voice (RVC/TTS) | isolated Cree singing takes, plus spoken pronunciation | 30–60 min of a consenting target voice for a first voice; more speakers for a general voice |

The same recording session can produce both, but the exports and consent scopes
are different. A voice model needs the singer's permission for voice cloning
specifically; a SingSong pair does not automatically grant that.

## 7. Training plan

### Stage A — base model

Train the general model on MUSDB18-HQ as already documented. This gives the
codec, semantic codebook, stage 1+2 and fine model a working starting point.

### Stage B — Cree adaptation

1. **Codec**: reuse the base codec unchanged. 2.196 h is too little to train a
   codec, and re-training it would make the Cree stage checkpoints incompatible
   with the base ones. Copy `checkpoints/singsong/soundstream.pt` to
   `checkpoints/cree/soundstream.pt` (runbook step 14) and revisit codec
   adaptation only once there are tens of hours of Cree stems.
2. **Semantic tokenizer**: re-fit the 1024-centre k-means on Cree vocal features,
   and optionally continue-train the w2v-BERT/HuBERT encoder on consenting Cree
   speech and singing. Keep a copy of the general codebook for A/B tests.
3. **Stage 1+2**: fine-tune from the general checkpoint on the Cree train split
   only, +10k steps with the cosine schedule extended (runbook step 17). The
   trainer takes one corpus per run, so the "70% Cree / 30% general" mixed
   batches below are **not implemented**; they would need a weighted dataset
   reader first. Cree-only fine-tuning is what the current runbook does, and it
   is why keeping the base checkpoints matters.
4. **Stage 3**: fine-tune on Cree instrumental codes. The fine stage controls
   timbre and top-end detail, so it matters for drum texture and vocal-adjacent
   artifacts.
5. **Style conditioning (optional)**: add a learned style embedding for
   round dance / powwow / folk / other if the dataset has enough examples per
   style. Without style labels, the model will average them.

### Stage C — evaluation

- Held-out **singers**, not just held-out songs.
- Cree-speaking listeners compare outputs; preference and appropriateness, not
  only FAD.
- Rhythm/key alignment against the held-out instrumental.
- Check for vocal bleed: mute the input vocal and listen to the generated
  instrumental. If the vocal is audible, the model learned separation artifacts.
- Report results separately for each style and dialect. A single average hides
  the failure modes that matter.
- Record all of the above in [the model card](cree-model-card.md); a `TODO` left
  in it means the evaluation is not finished.

## 8. Code changes needed

The current pipeline assumes a flat MUSDB-style corpus. A Cree dataset needs:

- a metadata-aware dataset reader (`metadata.jsonl`, not directory-name
  inference) — **not implemented**; the reader walks
  `data/cree_singsong/train_audio` and ignores the metadata beyond the
  singer-disjoint views;
- singer-disjoint split logic — **done** in `tools/fetch_cree_songs.py`, which
  writes `metadata.jsonl` and the `train_audio` / `dev_audio` / `test_audio`
  views;
- a Cree training profile — **done**: `config/cree.yaml` (not a section inside
  `config/config.yaml`);
- optional style conditioning — not implemented;
- a private-data path excluded from git — **done**: `data/cree_songs/`,
  `data/cree_singsong/` and `checkpoints/` are git-ignored;
- a weighted mixed Cree/general corpus loader — not implemented, so the
  fine-tune is Cree-only;
- an evaluation manifest that records consent/licence IDs — the per-track
  `license`, `license_url` and `consent_id` fields in `metadata.jsonl` serve this
  role.

The metadata and split checks already exist in
`tools/check_cree_singsong_dataset.py`; the CC BY corpus is a smoke-test set,
not a reason to skip the remaining schema work.

## 9. First practical milestone

1. Reuse the MUSDB codec unchanged, re-fit the semantic k-means on the 2.196 h
   CC BY corpus, and smoke-test the paired path on the 0.526 h of Demucs stems
   before running the full separation and fine-tune in
   [GPU_RUNBOOK.md](../GPU_RUNBOOK.md).
2. Separate more of the CC BY mixes (GPU makes this practical), or replace the
   approximations with commissioned multitracks.
3. Collect **5 hours** of paired stems across **at least 5 singers** and **at
   least 3 styles**, with singer-disjoint splits.
4. Run the base MUSDB model, then the Cree fine-tune, and evaluate with Cree
   listeners.
5. Only scale the dataset after the first listening test shows the style
   adaptation is real.

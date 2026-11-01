# MySong: automatic accompaniment for a vocal melody

This is the `mysong/` package: a from-scratch, downloadable-data system that
takes a melody (symbolic notes or a sung take), chooses a chord sequence, and
renders an arrangement with piano, bass, drums or pad, a mix and a MIDI file.
It is a strict superset of Simon, Morris and Basu's CHI 2008 MySong, with the
parts of that design that no longer hold up replaced and measured.

The package is deliberately separate from the live robot pipeline. It has no
side effects at import, runs on CPU, and its model is a few hundred kilobytes
plus an optional 2 MB neural tagger.

---

## 1. What the 2008 paper actually did

Source: [MySong: Automatic Accompaniment Generation for Vocal Melodies
(PDF)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/mysongchi2008.pdf)

The paper is worth reading closely because most summaries miss the details
that make the system work.

### Data and vocabulary

* **298 lead sheets**, about half from wikifonia.org and half from a private
  collection; pop, rock, R&B, jazz and country.
* Every chord is reduced to one of **five triads**: major, minor, diminished,
  augmented and suspended (sus2, sus4, or no third). That gives
  **12 roots x 5 qualities = 60 chord states**, plus virtual *start* and *end*
  states, so the transition matrix is **62 x 62**.
* Songs are transposed to C. The model is therefore key-independent; key is
  recovered at inference.

### The two learned tables

1. **Chord transition matrix.** Count every adjacent chord pair in every song,
   including start/end, and normalise each row. This is a first-order chord
   bigram.
2. **Melody observation matrix.** For each of the 60 chords, sum the *duration*
   of every melody pitch class that sounds while that chord is active, then
   normalise the 60 x 12 table by row. A row is `P(melody note | chord)`. A
   tiny imaginary duration is added for every note over every chord (a
   conjugate prior) so no entry is zero and the distribution is smoothed.

At inference a measure's melody is reduced to a 12-bin vector `x` (total time
on each pitch class), and the chord log-likelihood is the dot product
`x . log P(note | chord)`. That is a bag-of-pitch-classes model: it ignores
note order, octave, and which note is on the beat.

### Major/minor clustering

The corpus has key labels but not mode labels. The paper initialises each song
as major or minor with four heuristics (I/vii ratio, IV/ii ratio, V->I and
III->vii transitions), builds separate transition matrices, relabels every
song by which matrix gives the whole chord sequence a higher probability, and
repeats until stable. Only the transition matrices are split by mode; the
melody matrix is shared because pilot testing found the two modes similar.

### Decoding

* One HMM node per **measure**; 60 states per node.
* Viterbi finds the most likely chord sequence.
* The **"jazz factor"** multiplies the observation log-likelihood. High values
  follow the melody literally and accept stranger progressions; low values
  favour conventional progressions.
* The **"happy factor"** linearly blends the major and minor transition
  matrices before decoding: `(1-h) * minor + h * major`.
* **Key** is found by running the whole procedure twelve times, once per
  transposition, and keeping the key whose chord sequence has the highest
  probability. The paper reports under 50 ms for this on a typical PC.

### Audio front end

Autocorrelation pitch tracking at **100 samples/second**, deliberately *without*
the dynamic-programming cleanup a normal tracker uses. The authors argue that
keeping the noisy, continuous pitch trace is more robust for untrained singers
than quantising it into notes. The recording's tuning offset is estimated by
minimising the mean squared error between each frequency and the nearest
equal-tempered note, then the whole trace is shifted and folded to pitch
classes; octave is discarded. Per measure, sample counts become the 12-bin
vector.

### Accompaniment and evaluation

The 2008 system renders a **fixed, pre-defined piano pattern** and points at
Band-in-a-Box for better audio. Its evaluation is subjective: 26 sung clips,
30 musicians, mean scores **6.3 (MySong), 6.6 (manual), 2.9 (Band-in-a-Box)**.
A second study with 13 non-musicians found 73% of MySong accompaniments were
judged acceptable versus 23% for a chord-audition tool. There is no standard
held-out chord-label accuracy in the paper, and no public implementation of the
original HMM was released.

### What later work changed

| Work | Change | Why it matters here |
|---|---|---|
| Lim et al., *Chord Generation from Symbolic Melody Using BLSTM Networks* (2017); [reproduction](https://github.com/sander-wood/chord_generation) | Bar-level BiLSTM over melody, 24 major/minor triads | Learns context instead of a bag of pitch classes |
| [AutoHarmonizer](https://github.com/sander-wood/autoharmonizer) (Wu et al., 2021/22) | 1,462 chord labels, 16th-note frames, BiLSTM encoder-decoder, density control, 5,204 Wikifonia lead sheets | Flexible harmonic rhythm and a much larger vocabulary |
| Yeh et al., *Automatic Melody Harmonization with Triad Chords* (2020), [paper](https://arxiv.org/html/2001.02360v3) | Reproducible HMM baseline: 49 labels, half-bar PCPs, diagonal Gaussians, bigram smoothing beta=0.08, Viterbi; MTHarmonizer adds a chord-function head | Confirms a well-built HMM is still competitive; exposes how much the observation model matters |
| Sun et al., *Orderless NADE for Melody Harmonization* (2020), [paper](https://arxiv.org/html/2010.13468v2) | Masked-chord BiLSTM, 96 chords, blocked Gibbs sampling, class balancing, 18,005 Hooktheory examples | Progression-level context and rare-chord handling |
| MelodyT5 (2024), [paper](https://arxiv.org/html/2407.02277v1); SongDriver (2022), [paper](https://arxiv.org/html/2209.06054v2) | Transformer / CRF hybrids | Long context and real-time use, at a much larger data and compute cost |

Two useful negative results from that literature: there is **no standard
melody-to-chord accuracy benchmark on OpenEWLD**, and a single reference chord
sequence is not the only valid harmonisation. This package therefore reports
several metrics and several baselines instead of one leaderboard number.

---

## 2. What this implementation does differently

### 2.1 Flexible harmonic rhythm instead of one chord per measure

The decoder is a **semi-Markov HMM**. A chord may last 1, 2, 3, 4, 6, 8, 12 or
16 quarter notes; the duration prior is learned **per meter** (2/4, 3/4, 4/4,
6/8), and a separate prior models where in the bar changes happen. A truncated
final chord is allowed, so a 67-beat take does not need to be a multiple of
four. The paper's fixed grid is still available as
`Harmonizer.harmonize_fixed_bars`, which is the evaluation baseline.

### 2.2 A discriminative chord classifier instead of a duration table

`mysong/chord_model.py` fits a multinomial logistic regression on 52 features
of a candidate span:

* metric-weighted chroma, downbeat-only chroma, onset chroma and tail chroma
  (12 each, L1-normalised),
* log duration, bar phase (sin/cos), and mode.

Training rows are the annotated chord spans **plus every beat-aligned 1, 2 and
4-beat window** whose chord covers at least 60% of the window. That matters:
the decoder scores windows at inference that never appear as annotated spans.
The classifier is exported to numpy, so inference has no scikit-learn
dependency.

The classifier's output is a probability over the 60 core states. Its learned
class prior is partly divided out and the row renormalised, so rare chords do
not get free probability mass; without the renormalisation the decoder pays
*less* for adding segments and key selection collapses to minor keys.

### 2.3 A learned 24-way key classifier

Krumhansl-Schmuckler correlation over the melody gets 55% exact key accuracy on
held-out OpenEWLD: it confuses a key with its relative major/minor or dominant.
A linear classifier over the twelve pitch classes, trained on the corpus's key
labels and transposition-augmented so all 24 keys are represented, reaches
**93.2% exact key accuracy on the test split**. It is exported as a 24 x 12
numpy matrix. The decoder still considers several candidate keys and adds the
sequence likelihood, but the classifier's log-probability is weighted strongly
enough that it normally decides.

### 2.4 A beat-level BiGRU tagger (optional but recommended)

The span classifier cannot see the phrase around it. `mysong/neural.py` adds a
two-layer bidirectional GRU (128 units per direction) over per-beat features:
chroma, downbeat chroma, onset chroma, bar phase, mode and a voiced flag. It
emits 60 chord log-probabilities per beat, trained with label smoothing and
early stopping on a validation split carved out of the training songs.

The tagger is used as the decoder's observation model when its weights are
present. It is the largest single quality jump in the package. If PyTorch or
the weights are missing, the numpy classifier is used instead and everything
still works.

### 2.5 Exact inference and posterior alternatives

* Viterbi over the semi-Markov model is exact (no beam, no sampling).
* Forward-backward gives a posterior over chords for every decoded span. The
  UI can show real alternatives and confidences instead of pretending the MAP
  path is certain. Because the tagger's per-beat errors are correlated, the
  span posterior is tempered per beat before display; without that, every long
  chord reads as 100%. On the test split the mean chosen confidence is 0.62
  against 0.53 core accuracy. It is a ranking signal, not a calibrated error
  rate.
* Chord extensions (6, 7, 9, sus, ...) are chosen from learned
  `P(extension | core triad, mode)` tables, weighted by whether the melody
  actually contains the added pitch classes. The decoder always reasons over
  the five core triads, so the state space stays small.

### 2.6 Sampled instruments, not a toy synth

The first renderer was additive numpy: clean, dependency-free, and plainly an
instrument-shaped oscillator rather than a piano.  That is no longer the
default.  `render_arrangement(..., renderer="auto")` now renders each stem
through **FluidSynth** and a General MIDI SoundFont when both are installed,
then applies its own balance, stereo width, convolution room, bus compression
and limiter.  The sampled piano, acoustic bass, pad and drum kit are a large
listening improvement over the additive fallback, and the MIDI file remains
the interchange path for replacing them with a better sample library.

The fallback is still important: `renderer="synth"` is deterministic and needs
only numpy.  Tests use it, and machines without FluidSynth still work.  The
sampled path is selected only when it can actually run; it never silently
produces silence.

One real bug surfaced while wiring this up: MIDI track-name meta events were
written as `FF 03` followed directly by the name, without the required
variable-length length byte.  Every parser then read the first character as a
huge length and the track was misread.  FluidSynth rendered 64 samples of
silence from a file that looked plausible in a hex dump.  The encoder now
writes `FF 03 <length> <name>`, and a regression test parses the generated
track names with music21.

### 2.7 An actual arrangement, not a block-chord piano

`mysong/accompaniment.py` adds:

* a voice-leading search over inversions,
* style-specific piano, bass, pad and drum patterns (pop, ballad, folk, jazz,
  waltz, ambient, rock),
* humanised timing and velocity, swing, count-in,
* sampled General MIDI piano, bass, pad and drums through FluidSynth when
  available, with the additive synth as a dependency-free fallback,
* convolution reverb, vocal ducking, bus compression and a soft limiter,
* a Standard MIDI File for replacing the bank with a better instrument.

---

## 3. Training

```bash
python -m mysong.cli corpus       # parse OpenEWLD into data/mysong/corpus.jsonl
python -m mysong.cli train        # fit tables, chord classifier, key model, tagger
```

The base corpus is [OpenEWLD](https://github.com/00sapo/OpenEWLD), the
public-domain subset of EWLD. The checked-in cache has **502 songs / 52,417
melody notes / 19,569 chord spans**. Splits are by song, 80/10/10: **409
train, 49 dev, 44 test**.

The training code is deterministic given `--seed`. The tagger uses a 10%
validation split of the training songs for early stopping; dev and test are
never used for gradients or stopping.

### 3.1 The large corpus

Five hundred lead sheets is enough to prove a decoder, not enough to make a
strong model.  `tools/build_autoharmonizer_corpus.py` builds an auxiliary
corpus from the 5,204-score AutoHarmonizer/Wikifonia archive.  It removes both
normalized-title duplicates and, more importantly, any melody whose
**transposition-invariant note signature** matches an OpenEWLD score.  Title
matching alone missed 40 of the 44 test songs; the signature check removes
them, and the resulting **4,571 songs** have zero signature overlap with
OpenEWLD train, dev or test.

`tools/train_mysong_tagger_large.py` streams that file one score at a time and
keeps only the per-beat feature arrays.  This matters: a naive load of 5,000
`Song` objects plus the training graph exceeded 10 GB on a 27 GB laptop, while
the streaming path stays around 1.5 GB.  One malformed 21,759-beat score is
skipped for the same reason; one score out of thousands is not a distribution
change.

The structural tables (transitions, durations, extensions, bass) still come
from OpenEWLD in the streaming run.  The large tagger is the observation model;
this isolates the effect of adding data and keeps the evaluation honest.

```bash
python tools/build_autoharmonizer_corpus.py --workers 8
python tools/train_mysong_tagger_large.py \
  --epochs 60 --hidden 256 \
  --out-model data/mysong/model/mysong_large.npz
```

On a GPU server use `--device cuda` and [GPU_MYSONG.md](../GPU_MYSONG.md).  The
checkpoint is always written as CPU tensors, so inference does not need CUDA.
The supported GPU path for the shipped weights — 60 epochs, two seeds for the
ensemble, checkpoints every 10 epochs, and the tagger metadata rewritten on the
structural model — is the Colab notebook
[`MySong_Colab_Training.ipynb`](../MySong_Colab_Training.ipynb).

Model artefacts:

| File | Contents | Size |
|---|---|---|
| `data/mysong/model/mysong.npz` | transitions, start/end, duration/phase priors, extension/bass tables, chord classifier, key classifier, key profiles, metadata — including the attached tagger's architecture and run report | ~75 KiB |
| `data/mysong/model/mysong_tagger.npz` | 256-unit BiGRU weights + architecture metadata, plain numpy (a `.pt` copy is written too) | ~7.1 MiB |
| `data/mysong/model/mysong_tagger2.npz` | the second 256-unit tagger (same data, different seed); `MySongModel.load` loads it as a 2-model log-probability ensemble | ~7.1 MiB |
| `data/mysong/model/mysong_large.npz` | the same structural model, paired with the large-corpus tagger | ~75 KiB |
| `data/mysong/model/mysong_large_tagger.npz` | the ensemble's first 256-unit BiGRU, trained on 4,571 deduplicated extra scores | ~7.1 MiB |
| `data/mysong/model/mysong_large_tagger2.npz` | the ensemble's second tagger | ~7.1 MiB |
| `data/mysong/model/mysong_tagger_archive_small.npz` | 128-unit BiGRU kept for the ablation row | ~1.9 MB |

`python -m mysong.cli status` reports which parts are present.  The shipped
weights come from the Colab notebook [`MySong_Colab_Training.ipynb`](../MySong_Colab_Training.ipynb);
`reports/mysong_colab/model_card.md` records the run, the corpus counts, the
assembly hashes and the ablation behind the table below.

---

## 4. Evaluation

The harness is `mysong/evaluate.py`. Accuracy is frame-weighted at quarter-beat
resolution against the lead sheet, with three levels: **root**, **core** (root +
one of the five MySong triads) and **full** (exact extension). **Key** is exact
key and mode; **changes** is F1 of chord-change positions within half a beat.
The baselines are the paper's fixed one-chord-per-bar model, a key-aware
template matcher, and the paper's generative observation with everything else
unchanged.

### Test split (44 held-out songs, 2026-10-27)

| System | root | core | full | key | changes |
|---|---:|---:|---:|---:|---:|
| **256-unit two-tagger ensemble, auto key (shipped)** | **0.604** | **0.576** | **0.312** | **0.977** | **0.591** |
| 256-unit two-tagger ensemble, oracle key | 0.606 | 0.577 | 0.312 | 1.000 | 0.592 |
| single 256-unit tagger, auto key (ensemble ablation) | 0.592 | 0.560 | 0.315 | 0.955 | 0.600 |
| 192-unit tagger, auto key (2026-10-20 run) | 0.592 | 0.560 | 0.322 | 0.977 | 0.595 |
| 128-unit tagger, auto key (ablation) | 0.544 | 0.522 | 0.299 | 0.932 | 0.552 |
| fixed bar, large tagger | 0.584 | 0.559 | - | 1.000 | - |
| template matching | 0.406 | 0.353 | - | 1.000 | - |
| paper's generative observation | 0.249 | 0.206 | 0.146 | 1.000 | 0.600 |

The generative baseline is the paper's `P(note | chord)` dot product with the
same transitions, durations and decoder. The discriminative/tagger observation
roughly doubles its root accuracy.

The shipped tagger is trained on the OpenEWLD training split (409 songs) plus
4,571 auxiliary scores. The test split is never used for gradients or early
stopping, and no test melody appears in the auxiliary corpus. The table above
is therefore the honest held-out number for the shipped model. The previous
128-unit tagger, trained on the same training split without the auxiliary
corpus, scored 0.544 root / 0.932 exact key; the larger network and the extra
in-domain examples bought about five points of root accuracy.

The second tagger is a different seed over the same data, so it costs one extra
training run and no extra data; averaging the two log-probability fields buys
0.012 root and 0.022 exact key over the single tagger, which is what the table
above ships.

One intermediate result is worth recording because it was wrong: deduplicating
the auxiliary archive by title alone gave a **0.73 root** test score. Forty of
the 44 test melodies were still in the training archive under a different
title. Matching a transposition-invariant melody signature removed them, and
the leaked score is not reported as a result. The clean number was 0.592 for a
single tagger and is 0.604 for the shipped ensemble.

### What the metrics do and do not mean

OpenEWLD is a jazz/pop lead-sheet corpus. Its chord symbols are dense (often
two or three per bar), include extensions and inversions, and its key labels
occasionally disagree with the sounding tonic (a song notated in B minor can
sit in D major). Exact root accuracy against one reference is therefore a
lower bound on musical quality, not a verdict. The paper's own evaluation was
a listening study; no objective benchmark existed for this task on this
corpus.

---

## 5. Using it

### The bands

The bands for this project's own music come first, in the code, in `python -m
mysong.cli render` and in the app; the Western presets are still there under
"other bands".

| Band | What it plays |
|---|---|
| `hand_drum` (default) | The heartbeat a hand drum keeps under a song, a low drone per chord, and two off-beat strums. The middle of the bar is left to the voice. |
| `round_dance` | Steady hand-drum pulse with rattles on the eighths and a boom-chuck guitar. |
| `grass_dance` | The big drum on every beat (the honour beat), the dancers' bells on the eighths, a pickup into the next bar, and one low root per beat on the bass. |
| `jingle_dress` | The side-step: a hand drum on the pulse with the cone jingles ringing on the beat and the step after it, over a soft boom-chuck guitar. |
| `metis_fiddle` | A reel line on the fiddle (GM 41) over a two-step guitar and an upright two-feel bass. No drum kit. |
| `cedar_flute` | Long flute tones (GM 73) with a warm pad and two soft drum strokes a bar. |
| `pop`, `ballad`, `folk`, `jazz`, `waltz`, `ambient`, `rock` | The other bands. `jazz` swings, `ambient` has no drums. |

A band is a `ArrangementStyle`; the "piano" slot is whichever chordal or
melodic instrument the band points at it through `programmes`, which is why the
same slot carries the fiddle in one band and the flute in another. Percussion
comes from the drum channel of the SoundFont, so the drum-led bands use toms and
shakers rather than a rock kit, and they never add a cymbal.

These bands are played, not programmed: `humanize` (0.62-0.82 for the six
indigenous bands, against 0.5 for the other presets) jitters every onset by a
few milliseconds and every velocity by a few units, the hand drum's accents are
softer than a solo drum's because they sit under a voice, and `level_db` moves a
band from the common level -- `-3` for the dense hand-drum trio, `+2.5` for the
drum-and-bass grass dance -- so a busier band does not mask the words a sparse
one leaves alone. That is the difference between a band that was arranged for
the take and a grid with instruments on it.

### Levels: the voice is on top of the band

`mix_with_vocal` levels both parts instead of trusting the recording. The voice
is normalised to 0.18 RMS (up to +24 dB of boost, which covers a quiet phone
recording) and the band is then placed 10 dB under it (`BAND_OFFSET_DB`), moved
by the band's own `level_db` (-3 for the dense hand-drum trio, +2.5 for the
drum-and-bass grass dance), and ducked a further 6 dB while the voice sounds.

**The panel does not play that mix.**  It plays two sources -- your untouched
take and the generated band -- so the band file has to be levelled too, and this
was the bug that made the band louder than the singing: the player used to get
the renderer's raw, full-scale band against the raw (often 20 dB quieter)
recording. The band is now written at one reference RMS (`BAND_STEM_RMS`, the
same idea as the DSP vocable's fixed level) and the server measures the take and
returns `X-MySong-Band-Gain-Db`, the amount to pull the band down by for *that*
take. The page applies it with the ROBOT VOLUME slider, so the two sources at
100% are the mix, and a quiet phone recording gets the same balance as a loud
studio take.

Measured on the project's own take, with the panel at its defaults (ROBOT
VOLUME 100%, TRACK VOLUME 78%), the voice sits **9.4-13.1 dB above the band** in
the 300 Hz-3.4 kHz band across the six indigenous bands -- 9.4 for the dense
hand-drum trio, 12.8 for the round dance, 13.1 for the jingle dress -- and the
same take recorded 22 dB quieter lands at exactly the same 9.4 dB. The sum peaks
between 0.61 and 0.80, so nothing clips.

Those are also the defaults of the Sing with Archer panel, which is deliberate:
the same two labels, the same two numbers, the same meaning. The sliders are
playback trims; the same positions are sent to the render as `band_gain_db` and
`vocal_gain_db` so the WAV that gets saved is the balance that is playing. The
band's own trim is part of the arrangement, so it is applied to the mix and to
the player's placement, never to the slider.

### Symbolic melody

```python
from mysong import Harmonizer, MySongModel, Style
from mysong.corpus import NoteEvent
from mysong.melody import Melody

notes = [NoteEvent(0, 2, 60), NoteEvent(2, 2, 64), NoteEvent(4, 4, 67)]
melody = Melody.from_notes(notes, beats_per_bar=4)
model = MySongModel.load()
harmony = Harmonizer(model).harmonize(melody, style=Style())
print(harmony.summary())          # "C: C | F | G"
print(harmony.to_dict())          # segments, roman numerals, confidence
```

### A sung take

```bash
python -m mysong.cli analyze take.wav --bpm 96 --key C
python -m mysong.cli render take.wav --out renders/take --style jazz
```

`render` writes `mix.wav`, `accompaniment.wav`, `vocal.wav`, `song.mid` and
`chords.json`. In Python, `mysong.pipeline.create_song` / `analyze_song` are
the same entry points.

### Web app

The Live Demo page (`/live-demo#mysong`) has the same pipeline behind a small
panel: drop a take, choose the band preset, choose auto-detect or a fixed key,
switch between vocal + band and band only, then play or download the WAV. It
calls `POST /api/mysong/render`; `/api/mysong/status` reports whether the
model is present.

### Style controls

`Style` exposes the paper's controls and a few more:

| Field | Meaning |
|---|---|
| `observation_weight` | paper's jazz factor (numpy classifier path) |
| `tagger_weight` | scale on the neural tagger's log-probabilities |
| `transition_weight` | trust in the chord bigram |
| `density`, `duration_weight` | harmonic rhythm / duration prior temperature |
| `change_bonus` | extra log-odds for changing chord; raise for busier harmony |
| `color` | 0 = plain triads, 1 = let extensions follow the corpus |
| `happy` | 0 = minor transition matrix, 1 = major, in between = blend |
| `key`, `key_candidates`, `key_weight` | force or search key |
| `min_duration`, `max_duration`, `durations` | allowed chord lengths |

---

## 6. Limitations and honest gaps

* **One global key.** The decoder assumes a single key per performance, as the
  paper does. Local modulation is not tracked; a song that changes key will
  lose accuracy after the change. The key classifier is trained on whole-song
  chroma for the same reason.
* **Symbolic supervision.** Training uses lead sheets, not audio. The audio
  path is a separate front end (pYIN + tuning + beat grid); its transcription
  errors are not represented in training.
* **Corpus bias.** OpenEWLD and the AutoHarmonizer/Wikifonia archive are
  Western popular, jazz and standards material.  The model is not a model of
  Cree, folk, or non-Western harmony unless it is retrained on that material.
* **Auxiliary-corpus licensing.** The 4,571 extra scores are Wikifonia-derived.
  The archive's code license does not automatically clear the underlying songs;
  check before publishing derived weights.
* **Reference ambiguity.** The metrics compare against one annotator's chord
  sequence. A different but valid harmonisation can score poorly.
* **Fallback instruments.** The default renderer is now a sampled General MIDI
  bank through FluidSynth.  The additive synth remains the fallback on machines
  without FluidSynth, and it is plainly a fallback, not a production piano.
* **No listening benchmark.** The accuracy numbers are against one reference
  chord sequence.  They are not a substitute for a listening panel; the MIDI
  and WAV outputs exist so that judgement can be made on real material.
* **Tagger cost.** The BiGRU adds a PyTorch dependency and roughly 50-100 ms
  per song on CPU. The numpy classifier fallback is faster but weaker.
* **No lyrics or phonemes.** Chords are chosen from pitch/rhythm only; the
  words being sung do not influence harmony.

---

## 7. Reproducing the numbers

```bash
python -m mysong.cli corpus        # if data/mysong/corpus.jsonl is absent
python -m mysong.cli train         # structural model + small tagger

# large corpus (download/build it first; see GPU_MYSONG.md section 2)
python tools/build_autoharmonizer_corpus.py --workers 8
python tools/train_mysong_tagger_large.py \
  --epochs 60 --hidden 256 \
  --out-model data/mysong/model/mysong_large.npz

python -m mysong.cli evaluate --model data/mysong/model/mysong_large.npz --split dev
python -m mysong.cli evaluate --model data/mysong/model/mysong_large.npz --split test
python -m pytest -q tests/test_mysong.py
```

The small tagger takes a few minutes on CPU.  The large-corpus run took about
50 minutes on this 16-core CPU and about 20 minutes on a Colab T4 for the
shipped 60-epoch, two-seed recipe; `--device cuda`,
[GPU_MYSONG.md](../GPU_MYSONG.md) and the Colab notebook are the server paths.
`--no-tagger` trains only the numpy models and is enough for the CLI fallback.

### References

* Simon, Morris, Basu. *MySong: Automatic Accompaniment Generation for Vocal
  Melodies.* CHI 2008. [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/mysongchi2008.pdf)
* Lim, Lee, Lee. *Chord Generation from Symbolic Melody Using BLSTM Networks.*
  2017. [Reproduction](https://github.com/sander-wood/chord_generation)
* Wu, Chen, Yang. *AutoHarmonizer: Automatic Melody Harmonization with
  Transformer and Density Control.* 2021/22. [Paper](https://arxiv.org/html/2112.11122v3), [code](https://github.com/sander-wood/autoharmonizer)
* Yeh et al. *Automatic Melody Harmonization with Triad Chords: A Comparative
  Study.* 2020. [Paper](https://arxiv.org/html/2001.02360v3)
* Sun et al. *Orderless NADE for Melody Harmonization.* 2020. [Paper](https://arxiv.org/html/2010.13468v2)
* OpenEWLD. [Repository](https://github.com/00sapo/OpenEWLD)
* EWLD. [Zenodo record](https://zenodo.org/records/1476555)

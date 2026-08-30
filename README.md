# Archer-Robot Vocal Collaboration System

A real time system where a robot listens to a singer and harmonizes back live, using synthesized and neurally rendered vocables that echo the phonetic character of Cree singing without speaking the language. Built around a live pitch and rhythm tracking pipeline, a harmony decision engine, a two stage vocal synthesizer (DSP plus optional neural voice conversion), and a companion research web app for Cree language tools, sentiment analysis, and transcription.

---

## Collaborators

This project is a collaboration between:

- **Sara**
- **Archer Pechawis**, lead artist. [Archer's website, add link]
- **Orus Mateo Castaño Suárez**. [orusmateo.com](https://orusmateo.com)
- **Harsh Upadhyay**

Part of the **Abundant Intelligences** research program. [Abundant Intelligences website, add T'Karonto Pod link]

Read more in our article: [add link once published]

---

## What this actually does

There are two ways to run this project, and they share almost all of the same underlying code.

1. **Live performance mode.** A microphone feeds a real time pipeline: pitch detection, then rhythm and phrase tracking, then a harmony engine that decides what the robot should sing (unison, third, fifth, octave, drone, call and response, and so on), then a vocable synthesizer that renders the actual sound, optionally re-skinned through a trained neural singing voice.
2. **Web app mode.** `server.py` runs a FastAPI server and browser frontend with everything above, plus a set of standalone research tools built on the same audio stack: Cree word and morphology lookup, sentiment analysis, offline speech to text transcription, live streaming captions, and a notes-grounded chat assistant.

### The core pipeline (live performance)

```
mic -> AudioCapture -> Preprocessor -> PitchDetector (yin/rmvpe)
                                     -> RhythmAnalyzer (tempo, phrase state)
                                     -> CreeTokenizer (phoneme profile)
                                             |
                                     HarmonyEngine.decide()
                                             |
                                 VocableSynthesizer.synthesize()
                                             |
                        [optional] Neural Timbre Converter (RVC sidecar)
                                             |
                                     TimingSync -> speaker
```

- **Pitch detection.** `yin` is the default: fast and cheap to run on CPU. `rmvpe`, a real neural pitch model, is also available and pre-warmed at server startup for higher accuracy.
- **Harmony engine.** Decides the musical response (interval, mode, texture, number of voices) based on what the singer is doing, the tempo, and where phrases begin and end. It also includes a configurable sound cue that can trigger a silence response.
- **Vocable synthesizer.** The DSP layer (sinusoidal, wavetable, or formant synthesis) always works with zero extra setup. Its output can then be routed through a neurally trained voice.
- **Neural voice conversion (RVC).** This runs as its own separate process on purpose. `rvc_server.py` is a local sidecar (`http://127.0.0.1:8801` by default) so a heavy PyTorch and RVC stack never blocks the real time audio loop. `server.py` calls it over HTTP per note when `synthesis.neural.enabled` is `true`, and falls back to plain DSP audio automatically if the sidecar is down or a voice fails to load. Check `/neural/status` on the main app and `/health` on the sidecar itself.
  - One honest limitation, straight from the code: the neural stage currently re-skins the whole mixed DSP ensemble through one trained voice, not each choir layer separately. A true multi-timbre neural choir would need the synthesizer to expose per-voice audio stems before mixing. That is a real gap in the current version, not a bug.

### The web app's other tools

| Feature | Endpoint(s) | Backend |
|---|---|---|
| Cree word analysis (morphology, lemma, part of speech, spelling suggestions) | `/api/cree/analyze`, `/api/cree/health` | ALTLab's open source Plains Cree finite state analyzer |
| Cree to English translation | `/api/translate`, `/api/translate/health` | Custom trained transformer (`translation/translate_transformer.py`) |
| Sentiment analysis | `/api/sentiment`, `/api/sentiment/health` | RoBERTa as the primary model, with a VADER fallback. The response always says which one actually ran |
| Offline transcription of an uploaded track | `/api/transcribe`, `/api/transcribe/annotated` | faster-whisper, in two model sizes, with instrument and percussion span detection |
| Live streaming captions from the mic | `ws://.../ws/live-transcribe` | Vosk, fully local, no network calls at request time |
| Whole track neural voice rendering | `/api/neural/render-track`, `/api/neural/convert-track` | RVC sidecar |
| Pitch analysis of an uploaded track | `/api/pitch/analyze-track` | yin/rmvpe |
| Notes grounded chat assistant | see the "Chat" tab in the frontend | Only answers from notes it has explicitly been given |

**On the live transcription specifically:** it actually runs two ASR models at once, for a good reason. Vosk streams continuously and shows a rough caption almost immediately, in faded italic text. Whisper is much more accurate but only produces a result once it detects a pause in the audio, which barely happens during singing. So Vosk keeps something on screen the whole time, and whenever Whisper catches up, its cleaner result quietly replaces Vosk's rough guess. Neither model transcribes Cree. Both are English only, and that does not change with any config setting.

---

## Project structure

```
archer_robot/
├── server.py                    # main entry point: web API, serves the frontend, drives the live pipeline
├── requirements.txt
├── config/
│   ├── config.yaml               # all tunable settings: audio, pitch, harmony, synthesis, neural voice, etc.
│   └── config_loader.py
├── core/
│   ├── audio_capture.py
│   ├── preprocessor.py
│   └── pipeline.py
├── analysis/
│   ├── pitch_detector.py
│   ├── rhythm_analyzer.py
│   └── phonetic_analysis.py       # CreeTokenizer
├── synthesis/
│   ├── harmony_engine.py
│   ├── vocable_synthesizer.py
│   └── samples/                   # recorded/rendered vocable source takes
├── translation/
│   └── translate_transformer.py   # Cree to English model
├── output/
│   └── timing_sync.py
├── frontend/
│   └── index.html                 # single file frontend, served at "/" and mounted under /static
├── neural_env/                    # separate environment for the RVC sidecar
│   ├── rvc_server.py              # FastAPI sidecar, voice conversion only
│   └── rvc_pipeline.py            # RVCOfflineVoice / RMVPEPitchExtractor
├── data/models/                    # downloaded model checkpoints go here (mostly gitignored)
├── assets/                         # RVC checkpoints (hubert, rmvpe, trained voices) go here
├── tests/
│   └── test_pipeline.py
├── download_whisper_model.py       # downloads the faster-whisper models
├── download_vosk_model.py          # downloads the Vosk streaming ASR model
├── download_sentiment_model.py     # downloads the RoBERTa sentiment model
└── download_rvc_core_models.py     # downloads the universal RVC assets (hubert_base.pt, rmvpe.pt)
```

A note on `main.py`: it has been removed, and that is correct, not a bug. Its old job, listing audio devices and running the mic only pipeline, is now handled inside `server.py` itself, through `GET /devices` and `POST /pipeline/start` / `/pipeline/stop`. If you see old instructions anywhere that say `python main.py`, they are out of date.

---

## Setup

**You will need:** Python 3.11 or newer, and `ffmpeg` on your system path (used to decode uploaded audio tracks).

```bash
git clone https://github.com/haaaarsh4/Archer-Robot-Vocal-Collaboration-System.git
cd Archer-Robot-Vocal-Collaboration-System
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

On Raspberry Pi or Linux, install the system audio dependencies first:

```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

The base `requirements.txt` leaves out a few heavier, optional packages on purpose, so a minimal install stays light. Depending on which features you actually want, also run:

```bash
pip install transformers faiss-cpu vaderSentiment faster-whisper vosk pyaudio
```

- `transformers` and `faiss-cpu`: needed for the sentiment model and for RVC's retrieval index
- `vaderSentiment`: the sentiment fallback, works with zero downloads
- `faster-whisper`: offline track transcription
- `vosk`: live streaming captions
- `pyaudio`: local microphone capture, only needed for live performance mode, not for a headless or cloud deployment of the web app

**If you also want the neural singing voice**, it runs in a separate Python environment (`neural_env/`), because it needs its own PyTorch and RVC specific packages, kept isolated from the main app so a heavy import can never block real time audio:

```bash
cd neural_env
python -m venv venv
source venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" python-multipart pyyaml numpy scipy soundfile librosa faiss-cpu
```

---

## Pre-work: models you need before everything runs end to end

Almost every feature here fails safely if its model is missing. You will see a clear `[WARN]` or `[ERROR]` in the startup log, and the matching `/health` endpoint will report something like `"model_not_loaded"`, but the rest of the app keeps running with just that one feature turned off. Nothing breaks silently. Check the startup log to see what actually loaded.

### 1. Public models, one command each

```bash
python download_whisper_model.py        # faster-whisper small.en (about 150MB) and large-v3-turbo (about 1.6GB)
python download_vosk_model.py            # Vosk en-us streaming model (about 1.8GB)
python download_sentiment_model.py       # cardiffnlp RoBERTa sentiment model (about 500MB)
```

Each script is safe to re-run and only needs internet the first time. After that, every model loads from local files with no further network calls at runtime.

### 2. RVC core assets, only needed for the neural singing voice

```bash
python download_rvc_core_models.py       # hubert_base.pt (about 190MB) and rmvpe.pt (about 181MB)
```

These are the two universal building blocks that any RVC voice conversion needs, published by the RVC project itself. They are not specific to any one trained voice.

### 3. The Cree morphological analyzer, public but versioned, grab it by hand

The Cree word validation feature uses ALTLab's open source Plains Cree finite state analyzer. Download the compiled file from the releases page and place it where the code expects it:

1. Go to `github.com/UAlbertaALTLab/plains-cree-fsts`, then Releases
2. Download `crk-descriptive-analyzer.hfstol`
3. Place it at `data/models/crk-descriptive-analyzer.hfstol`
4. Run `pip install hfst`

### 4. Your trained neural singing voices, bring your own

`config.yaml` points at two RVC voice models:

```yaml
model_paths:
  - "assets/female2/female2.pth"
  - "assets/synthesis_models/mi-test.pth"
index_paths:
  - "assets/female2/female2_added_IVF1701_Flat_nprobe_1_female2_v2.index"
  - "assets/synthesis_models/mi-test_added_IVF1003_Flat_nprobe_1_mi-test_v2.index"
```

Both of these are voices trained with the [RVC WebUI project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). Training your own voice this way needs a set of clean vocal recordings of the person or voice you are training, plus a GPU (or patience, on CPU) to run the RVC training pipeline itself. That training process is outside the scope of this repo. Once you have a trained voice, each one needs two files to be usable here:

- a `.pth` weights file (the trained voice itself)
- a matching `.index` file (RVC's retrieval index, which improves timbre accuracy; it helps but is not strictly required)

Place them at the paths shown above, or update `config.yaml` to point wherever you keep them. Then start `rvc_server.py` and check `GET /health` on the sidecar. It reports exactly which voices loaded and, for any that failed, why.

**A note before naming or sharing any trained voice publicly:** a voice model trained on someone's singing is a recording of their voice in a very literal sense. Before a `.pth` file like this goes into a public repo or a model card, it is worth being explicit about whose voice it is and what permission exists to use it, the same way you would credit and get consent for a recording. That applies here regardless of what the file happens to be named.

### 5. The Cree to English translation model, project specific

`translation/translate_transformer.py` loads a custom trained model from:

```
data/models/transformer_mt.pt
data/models/spm.model
data/models/spm_config.json
```

This model was trained specifically for this project, not adapted from a public checkpoint, so there is no generic download script for it. Whoever trained it needs to provide these three files directly.

### 6. Everything else downloads itself automatically

- **PANNs** (instrument and percussion detection, used in `/api/transcribe/annotated`) downloads and caches its own weights the first time it runs. It needs internet once, and there is no manual step.
- **The RMVPE pitch model** for the live pipeline's `pitch.rmvpe` setting is the same `rmvpe.pt` file from step 2.

---

## Running it

**Web app** (recommended, this covers everything: live pipeline controls, Cree tools, sentiment, transcription, chat):

```bash
python server.py
# or: PORT=8080 python server.py
```

Then open `http://localhost:8000` (or whichever port you set).

**Neural voice sidecar** (only needed if you want the neural singing voice, and it should be started before `server.py`, or you can set `synthesis.neural.enabled: false` in `config.yaml` to skip it entirely):

```bash
cd neural_env
python rvc_server.py
# or: RVC_SIDECAR_PORT=8801 ARCHER_ROBOT_CONFIG=/path/to/config.yaml python rvc_server.py
```

Check `http://127.0.0.1:8801/health` to confirm your voices loaded, then `http://localhost:8000/neural/status` to confirm the main app can reach the sidecar.

**Finding your microphone** for live performance mode: open the web app and check `GET /devices`, or call `POST /pipeline/start` with the `input_device` index you want. You can also set a default in `config/config.yaml` under `audio.input_device`.

---

## Configuration

All settings live in `config/config.yaml`. A few of the most important ones:

| Setting | What it does |
|---|---|
| `pitch.method` | `yin` (fast, runs fine on CPU) or `rmvpe` (neural, more accurate, needs `rmvpe.pt`) |
| `harmony.default_mode` | The starting harmony mode: unison, third, fifth, octave, drone, and so on |
| `synthesis.engine` | `sinusoidal`, `wavetable`, or `neural_wavetable` (pre-rendered neural takes) |
| `synthesis.neural.enabled` | Turns on live neural voice re-skinning through the RVC sidecar |
| `synthesis.neural.sidecar_url` | Where `rvc_server.py` can be reached |
| `timing.response_delay_ms` | How much lag the robot's response has, in milliseconds |
| `cree_tokenizer.enabled` | Whether a Cree phoneme profile shapes the robot's timbre |

---

## Running tests

```bash
pytest tests/test_pipeline.py -v
```

Tests run without a microphone, using synthetic audio, so you do not need any hardware to check that the pipeline logic itself is working.

---

## A note on hardware

If you are running this on a laptop with an integrated GPU (for example, an AMD Phoenix or Radeon iGPU), keep `device: cpu` for both `pitch.rmvpe` and `synthesis.neural` in `config.yaml`. These chips generally are not officially supported by ROCm, and the usual workaround to force GPU use is a known hard hang on at least one commonly used chip. CPU performance is kept reasonable through int8 HuBERT quantization and explicit thread count tuning in `rvc_server.py`. That is a deliberate design choice, not a fallback someone forgot to fix.

---

## Current status and known limitations

This project is under active development, and it is worth being upfront about what is and is not finished yet, the same way the rest of this README tries to be:

- **Cree speech recognition does not exist yet.** Both ASR engines (Vosk and faster-whisper) are English only. Cree language support in this project currently lives in the text based tools: morphological analysis and translation.
- **Translation quality has not been formally measured.** The Cree to English model is trained and running, but there is no BLEU score, no held out test evaluation, and no human review process in place yet. Treat its output as a research prototype, not a verified translation.
- **The neural voice choir is one voice at a time.** As noted above, live neural re-skinning currently applies to the whole mixed ensemble rather than each harmony voice individually.
- **Dependency versions are not pinned yet.** `requirements.txt` uses lower bounds rather than exact versions in most places, so a fresh install months from now may pull in different versions than what this was built and tested against. Pinning exact versions, or adding a lockfile, would make this more reproducible.
- **Trained voice provenance should be documented before wider release.** As mentioned in the setup section above, any `.pth` voice file that goes into this repo or its model hub should be paired with a plain statement of whose voice it is and what permission exists to use it.

None of this blocks running the project locally for development or performance. It is here so that anyone picking this repo up later, including future collaborators, knows exactly where the edges are.
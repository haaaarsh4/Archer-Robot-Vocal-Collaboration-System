# Archer Robot Vocal Collaboration System

A real-time system in which a robot listens to a singer and harmonizes back with synthesized and neurally rendered vocables. The same repository also runs a research web application for Cree language tools, sentiment analysis, transcription, and a notes-grounded assistant.

The project is part of the **Abundant Intelligences** research program.

## Collaborators

- **Sara Diamond** ([LinkedIn](https://www.linkedin.com/in/sadiamond))
- **Archer Pechawis** ([website](http://apxo.net/))
- **Orus Mateo Castaño Suárez** ([website](https://orusmateo.com/))
- **Harsh Upadhyay** ([website](http://www.harshupadhyayy.com/))
- **Joshua Pothen** ([website](https://www.joshuapothen.com/))

## What it does

There are two ways to run the project. They share the audio, synthesis, and language code.

1. **Live performance.** A microphone feeds a real-time pipeline: pitch detection, rhythm and phrase tracking, a harmony engine, and a vocable synthesizer. The synthesized part can be re-skinned through a trained neural singing voice.
2. **Web application.** `server.py` runs a FastAPI server and a browser frontend with the live pipeline plus Cree word analysis, Cree and English translation, sentiment analysis, offline transcription, live captions, and a notes-grounded chat assistant.

### Live pipeline

```
mic -> AudioCapture -> Preprocessor -> PitchDetector (yin or rmvpe)
                                     -> RhythmAnalyzer
                                     -> CreeTokenizer
                                             |
                                     HarmonyEngine.decide()
                                             |
                                 VocableSynthesizer.synthesize()
                                             |
                        [optional] Neural Timbre Converter (RVC sidecar)
                                             |
                                     TimingSync -> speaker
```

The DSP synthesizer works without any downloaded model. `yin` pitch tracking runs on CPU. `rmvpe` and the neural voice converter require the model files listed below.

### Web application

| Feature | Endpoint | Backend |
|---|---|---|
| Cree word analysis | `/api/cree/analyze`, `/api/cree/health` | ALTLab Plains Cree finite-state analyzer |
| Cree to English translation | `/api/translate`, `/api/translate/health` | Project transformer in `translation/` |
| Sentiment analysis | `/api/sentiment`, `/api/sentiment/health` | RoBERTa, VADER fallback |
| Track transcription | `/api/transcribe`, `/api/transcribe/annotated` | faster-whisper |
| Live captions | `ws://.../ws/live-transcribe` | Vosk |
| Neural voice rendering | `/api/neural/render-track`, `/api/neural/convert-track` | RVC sidecar |
| Pitch analysis | `/api/pitch/analyze-track` | yin or rmvpe |
| Notes assistant | `/api/chat` | Local Archer model through Ollama or llama.cpp |

The speech recognizers are English-only. Cree speech recognition is not implemented.

## Quick start

Requirements: Python 3.11 or newer, and `ffmpeg` on the system path.

```bash
git clone https://github.com/haaaarsh4/Archer-Robot-Vocal-Collaboration-System.git
cd Archer-Robot-Vocal-Collaboration-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://localhost:8000`. On Linux or Raspberry Pi, install PortAudio first:

```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg
```

Features are optional. Install what you need:

```bash
pip install transformers faiss-cpu vaderSentiment faster-whisper vosk pyaudio
```

Download the public model files once:

```bash
python scripts/downloads/download_whisper_model.py
python scripts/downloads/download_vosk_model.py
python scripts/downloads/download_sentiment_model.py
python scripts/downloads/download_rvc_core_models.py
python scripts/downloads/download_cree_model.py
```

The Cree morphological analyzer is a separate release from ALTLab. Download `crk-descriptive-analyzer.hfstol` from the `UAlbertaALTLab/plains-cree-fsts` releases page, place it at `data/models/crk-descriptive-analyzer.hfstol`, and run `pip install hfst`.

The Cree and English translation model uses project files in `data/models/transformer_mt.pt`, `data/models/spm.model`, and `data/models/spm_config.json`. These are not public downloads.

The neural singing voice runs in a separate environment:

```bash
cd neural_env
python -m venv .venv
source .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" python-multipart pyyaml numpy scipy soundfile librosa faiss-cpu
python rvc_server.py
```

The sidecar listens on `127.0.0.1:8801` by default. Set `synthesis.neural.enabled: false` in `config/config.yaml` to run without it.

## The Archer model

The chat assistant runs one local model trained for this project. The same weights handle conversation about Archer's work, answers from the song notes, English and Cree translation, SRO and syllabics conversion, and word explanations.

The supported training path is [`train_cree_llm.py`](train_cree_llm.py), generated from [`training/cree_llm/cells/`](training/cree_llm/cells/). The Colab notebook is at [`training/notebooks/Cree_LLM_Training.ipynb`](training/notebooks/Cree_LLM_Training.ipynb).

- Model and data policy: [`docs/cree-llm.md`](docs/cree-llm.md)
- GPU server runbook: [`docs/cree-training.md`](docs/cree-training.md)
- Cree SingSong model card: [`docs/cree-model-card.md`](docs/cree-model-card.md)

The model is a research preview. Its held-out Cree translation scores are low, and its output should be reviewed by a fluent speaker before publication. It is not a substitute for an Elder or a fluent speaker.

## Training

| System | Guide |
|---|---|
| Archer Cree assistant | [`docs/cree-llm.md`](docs/cree-llm.md), [`docs/cree-training.md`](docs/cree-training.md) |
| SingSong accompaniment | [`docs/singsong-training.md`](docs/singsong-training.md) |
| Cree SingSong data | [`docs/cree-singsong.md`](docs/cree-singsong.md) |
| MySong accompaniment | [`docs/mysong-training.md`](docs/mysong-training.md), [`docs/mysong.md`](docs/mysong.md) |

The SingSong and MySong systems are separate from the live RVC voice path. SingSong composes an instrumental that follows a vocal. MySong chooses chords and renders a symbolic arrangement. RVC re-sings an existing performance.

## Project layout

```
analysis/       pitch, rhythm, phonetics and corpus tools
assets/         site art, neural voice models, local LLM files
chat/           assistant API, retrieval and local LLM client
config/         configuration and loader
core/           audio capture, preprocessing, pipeline and timing
data/           corpora and local model cache
docs/           model, data and training documentation
frontend/       browser application and static assets
mysong/         symbolic accompaniment system
neural_env/     isolated RVC sidecar
reports/        evaluation evidence and model reports
scripts/        setup, download and training helpers
sing_song/      SingSong implementation
song_notes/     notes used for grounded answers
synthesis/      harmony engine and vocable synthesis
tests/          pytest suite
tools/          evaluation and build utilities
training/       Cree LLM cells and training notebooks
translation/    Cree and English translation model
server.py       FastAPI application
train_cree_llm.py
```

## Tests

```bash
pytest tests/test_pipeline.py -v
```

The core tests use synthetic audio and do not need a microphone. Individual feature tests are under `tests/`.

## Data and credits

- Archer Pechawis's song notes and project text are used with permission and are project-owned.
- The Cree text corpus draws on EdTeKLA's Indigenous Languages Corpora, the Plains Cree Figurative Language corpus, Cree Wikipedia, FineWeb-2, FinePDFs, FineTranslations, GlotCC, GlotStoryBook, Tatoeba, the Plains Cree phrase recordings, the project's aligned pairs, Watkins 1865, and public-domain books from the Internet Archive. Every source is recorded with its licence, dialect, and script in the training ledger. The full list is in [`docs/cree-llm.md`](docs/cree-llm.md).
- Cree singing recordings used for SingSong and voice work come from YouTube uploads marked Creative Commons Attribution. Attribution and licence metadata are stored in `data/cree_songs/metadata.jsonl`. Consent is separate from a licence; do not publish a voice model without the singer's permission.
- MUSDB18-HQ is used under its research licence. VocalSet is described in [`docs/cree-singsong.md`](docs/cree-singsong.md). These datasets are not redistributed in this repository.
- RVC, HuBERT, RMVPE, Whisper, Vosk, RoBERTa, and the ALTLab analyzer remain under their own licences.

## Known limitations

- Speech recognition is English-only. Cree support covers text analysis and translation.
- Cree translation quality is measured but low. Treat it as a draft.
- The neural voice stage re-skins the mixed DSP ensemble, not each harmony layer separately.
- Dependency versions are lower bounds, not a lockfile.
- Trained voice files need documented provenance and permission before public release.

## Licence

The repository code and the third-party data and models have separate terms. Check the source and model documentation before redistributing weights, audio, or derived datasets.

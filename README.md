# Archer-Robot Vocal Collaboration System

A real-time system where a robot listens to Archer sing and harmonizes
back using vocables that mirror the feeling and
phonetic character of Cree singing without speaking the language.

---

## Project structure

```
archer_robot/
├── main.py                        
├── requirements.txt
├── config/
│   ├── config.yaml                
│   └── config_loader.py
├── core/
│   ├── audio_capture.py           
│   ├── preprocessor.py            
│   └── pipeline.py                
├── analysis/
│   ├── pitch_detector.py          
│   ├── rhythm_analyzer.py         
│   └── cree_tokenizer.py          
├── synthesis/
│   ├── harmony_engine.py          
│   ├── vocable_synthesizer.py     
│   └── samples/                   
│       ├── aah.wav
│       ├── ooo.wav
│       ├── mmm.wav
│       └── hey.wav
├── output/
│   └── timing_sync.py             
└── tests/
    └── test_pipeline.py
```

---

## Setup

**Requirements: Python 3.11+**

```bash
cd Archer-Robot-Vocal-Collaboration-System
pip install -r requirements.txt
```

On Raspberry Pi, install system audio dependencies first:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

---

## Find your USB mic

```bash
python main.py --list-devices
```

Find your USB mic in the list. Note its index number.
Set it in `config/config.yaml` under `audio.input_device`.

---

## Run

```bash
python main.py
```

With options:
```bash
python main.py --interval fifth     # change harmony interval
python main.py --debug              # verbose logging
python main.py --config my.yaml     # custom config file
```

Press **Ctrl+C** to stop.

---

## Configuration

All settings are in `config/config.yaml`. Key ones:

| Setting | What it does |
|---|---|
| `audio.input_device` | USB mic device index (null = default) |
| `pitch.engine` | `crepe` (better) or `pyin` (faster) |
| `harmony.default_interval` | `third`, `fifth`, `octave`, or `unison` |
| `synthesis.engine` | `sinusoidal`, `wavetable`, or `ddsp` |
| `timing.response_delay_ms` | Robot response lag in ms (default 70) |
| `cree_tokenizer.enabled` | `false` until model is ready |

---

## Wavetable mode (recommended for performance)

Record Archer (or any voice) singing sustained vowel sounds at a
comfortable pitch into four WAV files:

- `synthesis/samples/aah.wav`
- `synthesis/samples/ooo.wav`
- `synthesis/samples/mmm.wav`
- `synthesis/samples/hey.wav`

Then set `synthesis.engine: wavetable` in config.
The system will pitch-shift these samples to any harmony note in real time.

---

## Enabling the Cree tokenizer

When the Cree phoneme model is ready:

1. Set `cree_tokenizer.enabled: true` in config
2. Set `cree_tokenizer.model_path` to the model file path
3. Adjust `cree_tokenizer.phoneme_influence` (0.0–1.0) to control
   how strongly the phoneme profile shapes the robot's timbre

---

## Run tests

```bash
pytest tests/test_pipeline.py -v
```

Tests run without a microphone using synthetic audio.

---

## Hardware tested at York

- USB mic
- Raspberry Pi 5 or small laptop
- USB audio interface — Focusrite Scarlett Solo
- Powered monitor speaker — Yamaha HS5 or similar

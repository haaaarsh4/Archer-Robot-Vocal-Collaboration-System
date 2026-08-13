"""
chat/svs_pipeline.py

"Turn my song idea into Archer singing it" has two different inputs, and
they need two different pipelines. Don't skip this distinction — it's the
difference between something you can ship this week and a second ML
project.

Path A — hum/sing a scratch vocal (MVP, ships now)
---------------------------------------------------
    mic recording of you humming/singing the idea
        --> RVC (existing NeuralTimbreConverter / rvc_server.py sidecar)
        --> audio in Archer's timbre, same melody/rhythm/words you gave it

RVC is *voice conversion*: it re-skins the timbre of an audio performance
that already exists. It cannot invent a melody from text alone. This path
works today because your repo already has a trained RVC voice model and a
working sidecar (neural_env/rvc_server.py) — this class just calls the
same `NeuralTimbreConverter.convert_blocking()` your offline track
rendering already uses, on a user-supplied scratch vocal instead of a
DSP-synthesized one.

Path B — lyrics + melody, no scratch vocal (bigger lift, scaffolded here)
---------------------------------------------------------------------------
    lyrics + a melody (as a MIDI file, or a simple note-list you define)
        --> DiffSinger (singing voice synthesis: generates a sung
            scratch vocal directly from lyrics+notes, in a generic voice)
        --> RVC (same as Path A)
        --> audio in Archer's timbre, singing your written words to your
            written melody, with no human scratch performance needed

DiffSinger is a full second local ML pipeline: its own repo, its own
environment, its own pretrained acoustic + vocoder checkpoints, and
(for anything beyond its default voice/language) its own fine-tuning
data and training run. That's out of scope to fully stand up here, but
the orchestration below is real and works once you've set DiffSinger up
locally, following https://github.com/openvpi/DiffSinger — install it as
its own local sidecar service (same private-localhost pattern as the RVC
sidecar) and point svs.diffsinger_url at it. Until that URL is configured
and reachable, calls to `lyrics_and_melody_to_archer()` fail loudly rather
than silently degrading, since there's no DSP fallback for "sing this
melody from scratch" the way there is for single-note synthesis.

Both paths are 100% local — this file only ever calls localhost sidecars.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import requests
import soundfile as sf
from loguru import logger


@dataclass
class SVSConfig:
    diffsinger_url: str = "http://127.0.0.1:8802"
    diffsinger_timeout_s: float = 600.0
    default_voice_index: int = 0


class ScratchToSungVoice:
    def __init__(self, neural_timbre, config: SVSConfig | None = None):
        """
        neural_timbre: an already-constructed synthesis.neural_timbre.NeuralTimbreConverter
                        (the same instance server.py already builds — reused here rather
                        than opening a second connection to the RVC sidecar).
        """
        self.neural_timbre = neural_timbre
        self.cfg = config or SVSConfig()

    # ---------- Path A: hum/sing -> Archer's voice ----------
    def hum_to_archer(self, scratch_audio: np.ndarray, sample_rate: int,
                        voice_index: int | None = None) -> np.ndarray:
        if not self.neural_timbre or not self.neural_timbre.enabled:
            raise RuntimeError(
                "Neural timbre stage isn't ready — check synthesis.neural.enabled in "
                "config.yaml and that the RVC sidecar (neural_env/rvc_server.py) is running."
            )
        voice_index = self.cfg.default_voice_index if voice_index is None else voice_index
        result = self.neural_timbre.convert_blocking(scratch_audio, sample_rate, voice_index=voice_index)
        if result is None:
            raise RuntimeError("RVC conversion failed or timed out — see server logs.")
        return result

    # ---------- Path B: lyrics + melody -> Archer's voice ----------
    def lyrics_and_melody_to_archer(self, lyrics: str, notes: list[dict],
                                      voice_index: int | None = None) -> np.ndarray:
        """
        notes: a simple melody spec the caller builds (e.g. from a MIDI file
               parsed with `pretty_midi`, or typed directly), one dict per
               syllable/note:
                   {"pitch_midi": 60, "duration_s": 0.4, "lyric": "sun"}
               This shape maps directly onto DiffSinger's per-note input —
               translate from an uploaded .mid with pretty_midi.PrettyMIDI()
               plus your own lyric-to-note alignment before calling this.
        """
        scratch_audio, sr = self._call_diffsinger(lyrics, notes)
        return self.hum_to_archer(scratch_audio, sr, voice_index=voice_index)

    def _call_diffsinger(self, lyrics: str, notes: list[dict]) -> tuple[np.ndarray, int]:
        try:
            resp = requests.post(
                f"{self.cfg.diffsinger_url}/synthesize",
                json={"lyrics": lyrics, "notes": notes},
                timeout=self.cfg.diffsinger_timeout_s,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"DiffSinger sidecar not reachable at {self.cfg.diffsinger_url}: {e}\n"
                "This path needs DiffSinger set up as its own local service first — see the "
                "module docstring in chat/svs_pipeline.py for the setup pointer. Path A "
                "(hum your idea, /api/chat/sing) works right now without this."
            ) from e

        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

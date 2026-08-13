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
        self.neural_timbre = neural_timbre
        self.cfg = config or SVSConfig()

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

    def lyrics_and_melody_to_archer(self, lyrics: str, notes: list[dict],
                                      voice_index: int | None = None) -> np.ndarray:
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

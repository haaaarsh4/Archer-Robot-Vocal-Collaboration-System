from __future__ import annotations

import io
import subprocess
from pathlib import Path

from loguru import logger


class LocalTTS:
    def __init__(self, voice_path: str | None = None):
        self.voice_path = voice_path
        self._available = voice_path is not None and Path(voice_path).exists()
        if voice_path and not self._available:
            logger.warning(f"TTS voice not found at '{voice_path}' — spoken replies will be "
                           "unavailable until a valid Piper .onnx voice path is configured.")

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str) -> bytes:
        if not self._available:
            raise RuntimeError("Local TTS voice not configured — set tts.voice_path in config.yaml")

        proc = subprocess.run(
            ["piper", "--model", self.voice_path, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        pcm = proc.stdout  # raw 16-bit PCM, 22050 Hz mono
        return _pcm16_to_wav(pcm, sample_rate=22050)


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

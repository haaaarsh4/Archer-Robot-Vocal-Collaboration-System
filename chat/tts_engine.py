"""
chat/tts_engine.py

Local text-to-speech for the chatbot's "spoken audio" reply mode.
Uses Piper (https://github.com/rhasspy/piper) — small, fast neural TTS
that runs comfortably on CPU with no internet after the one-time voice
download.

Setup:
    pip install piper-tts
    # download a voice, e.g.:
    python -m piper.download_voices en_US-lessac-medium
This puts a .onnx + .onnx.json pair in Piper's voices dir; point
tts.voice_path at the .onnx file in config.yaml.

This is deliberately a separate, swappable stage from RVC. Piper gives you
a *neutral* spoken voice reading the chatbot's text reply (e.g. "This song
was written about..."). RVC is for the *singing* path — converting a sung
scratch vocal into Archer's singing timbre. Don't conflate the two: running
spoken chat replies through the RVC singing model would sound wrong (RVC
voice models here are trained on singing, not conversational speech).
"""

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
        """Returns 22.05kHz mono WAV bytes."""
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

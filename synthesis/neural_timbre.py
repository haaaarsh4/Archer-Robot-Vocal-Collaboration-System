import io
import threading
import time

import numpy as np
import requests
import soundfile as sf
from loguru import logger

from config.config_loader import get_config


class NeuralTimbreConverter:
    def __init__(self):
        cfg = get_config()
        ncfg = cfg.get("synthesis", {}).get("neural", {})

        self.enabled = bool(ncfg.get("enabled", False))
        self.sidecar_url = ncfg.get("sidecar_url", "http://127.0.0.1:8801").rstrip("/")
        self.timeout_s = float(ncfg.get("sidecar_timeout_s", 5.0))
        self.max_latency_warn_s = float(ncfg.get("max_latency_warn_s", 0.5))
        self.num_voices_configured = len(ncfg.get("model_paths", []))

        self._reachable = False
        self._call_count = 0
        # Two separate locks, not one shared _inflight_lock: convert() (the
        # live mic pipeline's real-time, one-note-at-a-time path) and
        # convert_blocking() (the web upload panel's Live/Full-render
        # chunk/track path) are genuinely different callers with different
        # needs -- one drops instantly if busy, the other waits patiently.
        # Sharing a single lock meant a multi-second track/chunk conversion
        # holding the lock would make every real-time note conversion
        # during that window fail its non-blocking acquire and silently
        # fall back to DSP audio, even though the actual bottleneck (the
        # sidecar's own _inference_lock, one model, one hardware) was
        # already the correct place for that serialization to happen.
        # Splitting these means each caller only ever contends with
        # *itself*, not with the other feature.
        self._live_lock = threading.Lock()
        self._track_lock = threading.Lock()

        if self.enabled:
            self._check_sidecar()
        else:
            logger.info("NeuralTimbreConverter disabled (synthesis.neural.enabled=false) "
                        "— pipeline runs on pure DSP output, unchanged from before.")

    def _check_sidecar(self):
        try:
            resp = requests.get(f"{self.sidecar_url}/health", timeout=self.timeout_s)
            resp.raise_for_status()
            status = resp.json()
            loaded = status.get("voices_loaded", 0)
            configured = status.get("voices_configured", 0)
            if loaded == 0:
                logger.warning(
                    f"Neural sidecar at {self.sidecar_url} is reachable but has 0/{configured} "
                    "voices loaded — check its terminal output. Falling back to DSP output for now."
                )
                self.enabled = False
                return
            self._reachable = True
            logger.info(f"Neural sidecar reachable at {self.sidecar_url}: "
                       f"{loaded}/{configured} voice(s) loaded and ready.")
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Neural sidecar not reachable at {self.sidecar_url}: {e}\n"
                "Is it running? See neural_env/README.md. Falling back to pure DSP output "
                "until it's up."
            )
            self.enabled = False

    def convert(self, audio: np.ndarray, sample_rate: int, target_hz: float,
                voice_index: int = 0, pad_seconds: float | None = None) -> np.ndarray:
        if not self.enabled or not self._reachable:
            return audio

        if not self._live_lock.acquire(blocking=False):
            return audio

        try:
            start = time.perf_counter()
            result = self._call_sidecar(audio, sample_rate, voice_index, pad_seconds)
            elapsed = time.perf_counter() - start
            self._call_count += 1
            if elapsed > self.max_latency_warn_s:
                logger.debug(f"Neural conversion took {elapsed*1000:.0f}ms "
                            f"(threshold {self.max_latency_warn_s*1000:.0f}ms) — "
                            "CPU inference is slower than the note rate; this is expected "
                            "without a GPU, notes that arrive mid-conversion fall back to DSP.")
            return result
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 503:
                logger.debug("Neural sidecar busy with another conversion — using DSP audio.")
            else:
                logger.error(f"Neural sidecar returned an error: {e} — using DSP audio for this note.")
            return audio
        except requests.exceptions.RequestException as e:
            logger.error(f"Neural sidecar call failed: {e} — using DSP audio for this note.")
            return audio
        except Exception as e:
            logger.error(f"Neural timbre conversion failed unexpectedly: {e} — "
                        "using DSP audio for this note.")
            return audio
        finally:
            self._live_lock.release()

    def _call_sidecar(self, audio: np.ndarray, sample_rate: int, voice_index: int,
                       pad_seconds: float | None = None) -> np.ndarray:
        buf = _encode_wav(audio, sample_rate)
        data = {"voice_index": voice_index, "sample_rate": sample_rate}
        if pad_seconds is not None:
            data["pad_seconds"] = pad_seconds

        resp = requests.post(
            f"{self.sidecar_url}/convert",
            files={"file": ("scratch.wav", buf, "audio/wav")},
            data=data,
            timeout=self.timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"sidecar returned {resp.status_code}: {resp.text}")
        return _decode_wav_response(resp.content, sample_rate)

    def convert_blocking(self, audio: np.ndarray, sample_rate: int, voice_index: int = 0,
                          timeout_s: float | None = None,
                          pad_seconds: float | None = None) -> np.ndarray | None:
        if not self.enabled or not self._reachable:
            return None

        timeout_s = float(timeout_s if timeout_s is not None else max(self.timeout_s, 900.0))
        acquired = self._track_lock.acquire(timeout=timeout_s)
        if not acquired:
            logger.error(f"Neural offline render: timed out after {timeout_s:.0f}s waiting for "
                         "another track/chunk conversion already in progress.")
            return None
        try:
            buf = _encode_wav(audio, sample_rate)
            data = {"voice_index": voice_index, "sample_rate": sample_rate, "wait": "true"}
            if pad_seconds is not None:
                data["pad_seconds"] = pad_seconds
            start = time.perf_counter()
            resp = requests.post(
                f"{self.sidecar_url}/convert",
                files={"file": ("track.wav", buf, "audio/wav")},
                data=data,
                timeout=timeout_s,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"sidecar returned {resp.status_code}: {resp.text}")
            round_trip_ms = (time.perf_counter() - start) * 1000
            inference_ms = resp.headers.get("X-Inference-Ms")
            if inference_ms is not None:
                # The gap between round-trip and reported inference time is
                # queueing (waiting on the sidecar's own lock) + network +
                # WAV encode/decode -- worth knowing apart from actual
                # model compute time when something feels slow.
                logger.debug(f"Neural chunk convert: {round_trip_ms:.0f}ms round-trip, "
                            f"{float(inference_ms):.0f}ms actual inference.")
            return _decode_wav_response(resp.content, sample_rate)
        except requests.exceptions.RequestException as e:
            logger.error(f"Neural offline render: sidecar call failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Neural offline render: unexpected failure: {e}")
            return None
        finally:
            self._track_lock.release()


def _encode_wav(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf


def _decode_wav_response(content: bytes, sample_rate: int) -> np.ndarray:
    out_buf = io.BytesIO(content)
    converted, out_sr = sf.read(out_buf, dtype="float32", always_2d=False)
    if converted.ndim > 1:
        converted = converted.mean(axis=1)
    if out_sr != sample_rate:
        import librosa
        converted = librosa.resample(converted, orig_sr=out_sr, target_sr=sample_rate)
    return converted

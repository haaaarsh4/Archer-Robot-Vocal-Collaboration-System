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

        # How many voices in a choir/duet get individually re-skinned
        # through the sidecar in a live, real-time call. None means no
        # cap (convert every voice) -- fine for an offline whole-track
        # render, not fine for anything with a latency budget, since the
        # sidecar processes one conversion at a time (see convert_stems).
        raw_cap = ncfg.get("max_ensemble_voices", 2)
        self.max_ensemble_voices = int(raw_cap) if raw_cap is not None else None

        # When true, and more than one voice is listed in model_paths,
        # harmony stems (everything except the lead) round-robin across
        # every loaded voice instead of all singing through the same one
        # -- this is what actually makes a neural choir sound like more
        # than one trained voice repeated at different pitches.
        self.distribute_voices_across_choir = bool(ncfg.get("distribute_voices_across_choir", True))

        self._reachable = False
        self._call_count = 0
        self._live_lock = threading.Lock()
        self._track_lock = threading.Lock()

        if self.enabled:
            self._check_sidecar()
        else:
            logger.info("NeuralTimbreConverter disabled — pipeline runs on pure DSP output.")

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
                    "voices loaded — falling back to DSP output."
                )
                self.enabled = False
                return
            self._reachable = True
            logger.info(f"Neural sidecar reachable at {self.sidecar_url}: "
                       f"{loaded}/{configured} voice(s) loaded and ready.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Neural sidecar not reachable at {self.sidecar_url}: {e}")
            self.enabled = False

    def convert(self, audio: np.ndarray, sample_rate: int, target_hz: float,
                voice_index: int = 0, pad_seconds: float | None = None,
                transpose_semitones: float | None = None) -> np.ndarray:
        if not self.enabled or not self._reachable:
            return audio

        if not self._live_lock.acquire(blocking=False):
            return audio

        try:
            start = time.perf_counter()
            result = self._call_sidecar(audio, sample_rate, voice_index, pad_seconds, transpose_semitones)
            elapsed = time.perf_counter() - start
            self._call_count += 1
            if elapsed > self.max_latency_warn_s:
                logger.debug(f"Neural conversion took {elapsed*1000:.0f}ms")
            return result
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 503:
                logger.debug("Neural sidecar busy — using DSP audio.")
            else:
                logger.error(f"Neural sidecar returned an error: {e}")
            return audio
        except requests.exceptions.RequestException as e:
            logger.error(f"Neural sidecar call failed: {e}")
            return audio
        except Exception as e:
            logger.error(f"Neural timbre conversion failed unexpectedly: {e}")
            return audio
        finally:
            self._live_lock.release()

    def _call_sidecar(self, audio: np.ndarray, sample_rate: int, voice_index: int,
                       pad_seconds: float | None = None,
                       transpose_semitones: float | None = None) -> np.ndarray:
        buf = _encode_wav(audio, sample_rate)
        data = {"voice_index": voice_index, "sample_rate": sample_rate}
        if pad_seconds is not None:
            data["pad_seconds"] = pad_seconds
        if transpose_semitones is not None:
            data["transpose_semitones"] = transpose_semitones

        resp = requests.post(
            f"{self.sidecar_url}/convert",
            files={"file": ("scratch.wav", buf, "audio/wav")},
            data=data,
            timeout=self.timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"sidecar returned {resp.status_code}: {resp.text}")
        return _decode_wav_response(resp.content, sample_rate)

    def convert_stems(self, stems: list[dict], sample_rate: int, voice_index: int = 0,
                       transpose_semitones: float | None = None,
                       max_voices: int | None = "__default__",
                       blocking: bool = False, timeout_s: float | None = None) -> list[dict]:
        """
        Runs each choir/duet voice stem through the sidecar individually,
        instead of mixing the whole ensemble down first and running one
        conversion pass over the mix. RVC's pitch stage (RMVPE) estimates
        a single f0 curve per call -- handing it several simultaneous,
        deliberately-detuned pitches at once doesn't produce a richer
        result, it produces a mistracked pitch curve and smeared timbre,
        because the model was never built to separate overlapping voices.
        Converting one already-detuned voice at a time means RVC always
        sees exactly the one real pitch that voice was synthesized at.

        Returns a new list of stems, same shape as the input, with
        "audio" replaced wherever a conversion was attempted and
        succeeded. Any stem beyond max_voices, or where conversion fails
        or the sidecar isn't reachable, keeps its original DSP audio --
        so a choir always degrades to "fewer voices are neurally-voiced,
        the rest are DSP" rather than to silence or an error. The input
        list itself is never mutated.

        max_voices caps how many stems are actually sent to the sidecar,
        picking the ones closest to the lead (cents nearest zero) first.
        Leave it unset to use synthesis.neural.max_ensemble_voices from
        config.yaml (meant for anything with a real-time budget, since
        the sidecar only ever processes one conversion at a time -- see
        _inference_lock in rvc_server.py). Pass max_voices=None explicitly
        to convert every stem regardless of that config value, which is
        the right call for an offline whole-track render where time isn't
        the constraint.

        The lead stem (cents == 0, always stems[0] out of
        VocableSynthesizer._render_ensemble_stems) always converts through
        the requested voice_index, since that's the voice the person
        actually chose for this performance. Harmony stems round-robin
        across every voice in synthesis.neural.model_paths when
        distribute_voices_across_choir is on and more than one is
        configured (see __init__) -- that's what gives a choir more than
        one distinct trained timbre instead of one voice cloned at
        several pitches.
        """
        if not self.enabled or not self._reachable or not stems:
            return stems

        if max_voices == "__default__":
            max_voices = self.max_ensemble_voices

        ordered = sorted(range(len(stems)), key=lambda i: abs(stems[i].get("cents", 0.0)))
        if max_voices is not None:
            # max(0, ...), not max(1, ...) on purpose -- 0 is a legitimate,
            # explicit choice (see neural_voice_count in server.py): "use
            # the Neural engine, but this ensemble stays entirely DSP."
            # That's different from disabling the neural stage outright,
            # since the lead voice or other stems in a different call can
            # still ask for conversion; it just means this call converts
            # nothing and every stem keeps its original DSP audio.
            ordered = ordered[:max(0, int(max_voices))]

        out = [dict(s) for s in stems]  # shallow copy -- caller's original stems are left untouched
        harmony_rank = 0

        for i in ordered:
            stem = out[i]
            audio = np.asarray(stem["audio"], dtype=np.float32)
            n_samples_in = len(audio)
            is_lead = abs(stem.get("cents", 0.0)) < 1e-6

            if is_lead or not self.distribute_voices_across_choir or self.num_voices_configured <= 1:
                this_voice_index = voice_index
            else:
                this_voice_index = harmony_rank % self.num_voices_configured
                harmony_rank += 1

            if blocking:
                converted = self.convert_blocking(
                    audio, sample_rate, voice_index=this_voice_index,
                    timeout_s=timeout_s, transpose_semitones=transpose_semitones,
                )
            else:
                result = self.convert(
                    audio, sample_rate, stem.get("f0_hz", 0.0), voice_index=this_voice_index,
                    transpose_semitones=transpose_semitones,
                )
                # convert() degrades silently to returning the input audio
                # unchanged on any failure/busy/disabled state (see its
                # own docstring) -- treat that the same as "didn't convert"
                # here rather than mixing the same audio in twice.
                converted = None if result is audio else result

            if converted is None:
                continue

            out[i] = {**stem, "audio": _fit_length(np.asarray(converted, dtype=np.float32), n_samples_in)}

        return out

    def convert_blocking(self, audio: np.ndarray, sample_rate: int, voice_index: int = 0,
                          timeout_s: float | None = None,
                          pad_seconds: float | None = None,
                          transpose_semitones: float | None = None) -> np.ndarray | None:
        if not self.enabled or not self._reachable:
            return None

        timeout_s = float(timeout_s if timeout_s is not None else max(self.timeout_s, 900.0))
        acquired = self._track_lock.acquire(timeout=timeout_s)
        if not acquired:
            logger.error(f"Neural offline render: timed out after {timeout_s:.0f}s")
            return None
        try:
            buf = _encode_wav(audio, sample_rate)
            data = {"voice_index": voice_index, "sample_rate": sample_rate, "wait": "true"}
            if pad_seconds is not None:
                data["pad_seconds"] = pad_seconds
            if transpose_semitones is not None:
                data["transpose_semitones"] = transpose_semitones
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


def _fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Pads or trims audio to exactly n_samples. The sidecar resamples
    internally (project rate -> 16kHz -> back), so a converted stem can
    come back a handful of samples off from what went in -- this keeps
    it aligned with the other stems before VocableSynthesizer.combine_stems()
    mixes them back together.
    """
    if len(audio) == n_samples:
        return audio
    if len(audio) > n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio)))


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
        converted = librosa.resample(converted, orig_sr=out_sr, target_sr=sample_rate, res_type="soxr_vhq")
    return converted

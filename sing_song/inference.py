"""End-to-end SingSong inference.

The system is a three-stage cascade:

    vocal waveform
      -> semantic tokens (25 Hz)
      -> stage 1+2: instrumental semantic + coarse tokens
      -> stage 3: fine tokens conditioned on coarse
      -> SoundStream decode
      -> instrumental, mixed with the untouched input vocal

Long audio is generated with the paper's 50%-overlap sliding window: each 10 s
window is conditioned on that window of vocal, decoded independently, and the
overlap is equal-power crossfaded into the output.  The model API also accepts a
same-layout decoder prompt (``prompt_tokens``) for checkpoints trained with the
paper's continuation objective; the default scheduler does not invent a prompt
that would be out of distribution.
"""
from __future__ import annotations

import io
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .codec import SoundStream, load_codec
from .config import SingSongConfig, load_sing_song_config
from .model import FineModel, Stage12Model, build_fine, build_stage12
from .sampling import GenerationConfig, generate_fine, generate_stage12
from .semantic import SemanticEncoder, load_semantic
from .tokens import FineLayout, Stage12Layout


class SingSongNotReady(RuntimeError):
    pass


class SingSongEngine:
    def __init__(self, cfg: SingSongConfig | None = None):
        self.cfg = cfg or load_sing_song_config()
        self.device = self._resolve_device(self.cfg.inference.device)
        self.dtype = self._resolve_dtype(self.cfg.inference.dtype, self.device)
        self.codec: SoundStream | None = None
        self.semantic: SemanticEncoder | None = None
        self.stage12: Stage12Model | None = None
        self.fine: FineModel | None = None
        self.loaded = False
        self.load_error: str | None = None
        self.load_seconds: float | None = None

    # Loading and status
    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value in (None, "", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    @staticmethod
    def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
        if value in ("float16", "fp16", "half") and device.type == "cuda":
            return torch.float16
        if value in ("bfloat16", "bf16") and device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    def status(self) -> dict:
        status = self.cfg.status()
        status.update({
            "loaded": self.loaded,
            "load_error": self.load_error,
            "load_seconds": self.load_seconds,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        })
        return status

    def load(self) -> "SingSongEngine":
        if self.loaded:
            return self
        missing = self.cfg.missing_artifacts()
        if missing:
            self.load_error = "missing artifacts: " + "; ".join(missing)
            raise SingSongNotReady(self.load_error)

        start = time.perf_counter()
        try:
            self.codec = load_codec(str(self.cfg.codec_path), map_location="cpu")
            self.codec.to(self.device, dtype=self.dtype).eval()

            self.semantic = load_semantic(
                self.cfg.semantic, kmeans_path=self.cfg.semantic_kmeans_path
            )
            if self.semantic.device != self.device:
                self.semantic.device = self.device
                if self.semantic.model is not None:
                    self.semantic.model.to(self.device)

            self.stage12 = self._load_transformer(
                build_stage12(self.cfg.stage12), self.cfg.stage12_path
            )
            self.fine = self._load_transformer(
                build_fine(self.cfg.fine), self.cfg.fine_path
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through /singsong/status
            self.load_error = str(exc)
            raise
        self.load_seconds = round(time.perf_counter() - start, 2)
        self.loaded = True
        self.load_error = None
        return self

    def _load_transformer(self, model, path):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        if state and all(str(k).startswith("module.") for k in state):
            state = {str(k)[len("module."):]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {path} does not match the configured architecture: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        return model.to(self.device, dtype=self.dtype).eval()

    # Generation
    @torch.no_grad()
    def generate(
        self,
        vocal: np.ndarray,
        sample_rate: int,
        *,
        prompt_tokens: torch.Tensor | None = None,
    ) -> dict:
        """Generate an instrumental for a mono vocal waveform.

        Returns ``instrumental`` and ``mix`` at ``sample_rate`` plus metadata.
        The input vocal is never codec-round-tripped; the paper 
        mixes the original waveform with the decoded instrumental.
        """
        if not self.loaded:
            self.load()
        assert self.codec is not None and self.semantic is not None
        assert self.stage12 is not None and self.fine is not None

        vocal = np.asarray(vocal, dtype=np.float32).reshape(-1)
        if vocal.size == 0:
            raise ValueError("empty vocal")
        max_samples = int(self.cfg.inference.max_seconds * sample_rate)
        if vocal.size > max_samples:
            vocal = vocal[:max_samples]

        window_s = float(self.cfg.inference.window_seconds)
        hop_s = max(0.1, window_s - float(self.cfg.inference.overlap_seconds))
        window_n = int(round(window_s * sample_rate))
        hop_n = int(round(hop_s * sample_rate))
        sr16 = self.cfg.codec.sample_rate
        window_16 = int(round(window_s * sr16))

        layout = Stage12Layout.for_seconds(window_s)
        fine_layout = FineLayout.for_seconds(window_s)
        if self.codec.cfg.frame_rate != 50:
            raise RuntimeError(
                f"codec frame rate {self.codec.cfg.frame_rate} Hz is not 50 Hz; "
                "the SingSong token layout requires 50 Hz acoustic frames"
            )
        if layout.semantic_frames != int(round(window_s * 25)):
            raise RuntimeError("semantic layout does not match the 25 Hz paper rate")

        starts = list(range(0, max(1, vocal.size - window_n + 1), hop_n))
        if not starts or starts[-1] + window_n < vocal.size:
            starts.append(max(0, vocal.size - window_n))

        instrumental = np.zeros(vocal.size + window_n, dtype=np.float32)
        weight = np.zeros_like(instrumental)
        rng = np.random.default_rng(self.cfg.inference.seed)
        windows: list[dict] = []

        for wi, start in enumerate(starts):
            chunk = vocal[start:start + window_n]
            if chunk.size < window_n:
                chunk = np.pad(chunk, (0, window_n - chunk.size))
            if self.cfg.inference.add_noise and self.cfg.data.noise_sigma > 0:
                chunk = chunk + rng.normal(
                    0.0, self.cfg.data.noise_sigma, size=chunk.shape
                ).astype(np.float32)

            vocal_sr = torch.from_numpy(
                _resample_np(chunk, sample_rate, sr16)
            ).to(self.device).unsqueeze(0)
            sem = self.semantic.encode(vocal_sr, sample_rate=sr16)
            sem = _fit_frames(sem, layout.semantic_frames)

            gen_cfg = GenerationConfig(
                temperature=self.cfg.inference.temperature,
                top_k=self.cfg.inference.top_k,
                seed=int(self.cfg.inference.seed) + wi,
            )
            stage12_tokens = generate_stage12(
                self.stage12, sem, layout, config=gen_cfg,
                prefix_tokens=prompt_tokens, device=self.device,
            )
            _, coarse = layout.split_target(stage12_tokens[0].tolist())
            coarse_t = torch.tensor(coarse, dtype=torch.long, device=self.device).unsqueeze(0)
            coarse_prefix = fine_layout.build_coarse_prefix(coarse)
            coarse_prefix_t = torch.tensor(
                coarse_prefix, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            fine_tokens = generate_fine(
                self.fine, coarse_prefix_t, fine_layout, config=gen_cfg, device=self.device
            )
            fine = fine_layout.split_fine(fine_tokens[0].tolist())
            fine_t = torch.tensor(fine, dtype=torch.long, device=self.device).unsqueeze(0)

            decoded = self.codec.reconstruct_from_codes(
                coarse_t, fine_t, length=window_16
            )[0, 0].float().cpu().numpy()
            if sample_rate != sr16:
                decoded = _resample_np(decoded, sr16, sample_rate)
            decoded = _fit_array(decoded, window_n)

            # Equal-power overlap-add.  Every window has the same weight shape,
            # so the sum of weights is 1 in the interior and the fade-in/out at
            # the two ends is a raised cosine.
            w = _window_weights(window_n, hop_n, first=wi == 0, last=wi == len(starts) - 1)
            instrumental[start:start + window_n] += decoded * w
            weight[start:start + window_n] += w
            windows.append({
                "start_s": round(start / sample_rate, 3),
                "semantic_tokens": int(layout.semantic_frames),
                "coarse_tokens": int(layout.coarse_length),
                "fine_tokens": int(fine_layout.fine_length),
            })

        nz = weight > 1e-6
        instrumental[nz] /= weight[nz]
        instrumental = instrumental[:vocal.size]

        if self.cfg.inference.instrumental_gain_db:
            instrumental *= 10 ** (self.cfg.inference.instrumental_gain_db / 20.0)
        if self.cfg.inference.vocal_gain_db:
            vocal = vocal * 10 ** (self.cfg.inference.vocal_gain_db / 20.0)
        mix = vocal + instrumental if self.cfg.inference.mix_vocal else instrumental.copy()
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.99:
            gain = 0.99 / peak
            mix *= gain
            instrumental *= gain

        return {
            "instrumental": instrumental.astype(np.float32),
            "mix": mix.astype(np.float32),
            "sample_rate": sample_rate,
            "windows": windows,
            "duration_s": round(vocal.size / sample_rate, 3),
            "device": str(self.device),
        }

    def render_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        return_mix: bool = True,
    ) -> dict:
        import librosa

        audio, sr = librosa.load(str(input_path), sr=None, mono=True)
        result = self.generate(audio, sr)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        instrumental_path = output_dir / "instrumental.wav"
        mix_path = output_dir / "mix.wav"
        sf.write(instrumental_path, result["instrumental"], sr, subtype="PCM_16")
        sf.write(mix_path, result["mix"], sr, subtype="PCM_16")
        result["instrumental_path"] = str(instrumental_path)
        result["mix_path"] = str(mix_path)
        result["return_mix"] = bool(return_mix)
        return result

    def render_bytes(
        self, input_bytes: bytes, *, return_mix: bool = True
    ) -> tuple[bytes, dict]:
        import librosa

        audio, sr = librosa.load(io.BytesIO(input_bytes), sr=None, mono=True)
        result = self.generate(audio, sr)
        buf = io.BytesIO()
        sf.write(buf, result["mix"] if return_mix else result["instrumental"], sr, format="WAV")
        buf.seek(0)
        return buf.read(), result


def _resample_np(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=target, res_type="soxr_hq").astype(np.float32)


def _fit_array(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size == length:
        return audio.astype(np.float32, copy=False)
    if audio.size > length:
        return audio[:length].astype(np.float32, copy=False)
    return np.pad(audio, (0, length - audio.size)).astype(np.float32)


def _fit_frames(tokens: torch.Tensor, frames: int) -> torch.Tensor:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.shape[1] == frames:
        return tokens
    if tokens.shape[1] > frames:
        return tokens[:, :frames]
    pad = torch.zeros(
        (tokens.shape[0], frames - tokens.shape[1]),
        dtype=tokens.dtype, device=tokens.device,
    )
    return torch.cat((tokens, pad), dim=1)


def _window_weights(n: int, hop: int, *, first: bool, last: bool) -> np.ndarray:
    """Raised-cosine window that sums to one across a 50% overlap."""
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, max(2, min(hop, n // 2)), endpoint=False)
    # Linear, complementary fades sum to exactly one across the overlap.  An
    # equal-power (sin/cos) pair would sum in energy but dip in amplitude when
    # the two windows are correlated, which is audible on sustained notes.
    fade_in = ramp.astype(np.float32)
    fade_out = (1.0 - ramp).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    m = len(fade_in)
    w[:m] = fade_in
    w[-m:] = fade_out
    if first:
        w[:m] = 1.0
    if last:
        w[-m:] = 1.0
    return w


def engine_from_config(cfg: SingSongConfig | None = None) -> SingSongEngine:
    return SingSongEngine(cfg)

from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from infer.hubert import load_hubert_model, extract_hubert_features
from infer.rmvpe import RMVPE
from infer.module.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

try:
    import faiss
except ImportError:
    faiss = None  

_SYNTH_CLASSES = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
}


def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


_HP_B, _HP_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def _change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    import librosa
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1_t = torch.from_numpy(rms1)
    rms1_t = F.interpolate(rms1_t.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2_t = torch.from_numpy(rms2)
    rms2_t = F.interpolate(rms2_t.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2_t = torch.max(rms2_t, torch.zeros_like(rms2_t) + 1e-6)
    scaled = data2 * (
        torch.pow(rms1_t, torch.tensor(1 - rate)) * torch.pow(rms2_t, torch.tensor(rate - 1))
    ).numpy()
    return scaled.astype(np.float32)


class RMVPEPitchExtractor:
    def __init__(self, model_path: str, device: str = "cpu", is_half: bool = False):
        self._rmvpe = RMVPE(model_path, is_half=is_half, device=device)
        self._lock = threading.Lock()

    def infer(self, audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        with self._lock:
            return self._rmvpe.infer_from_audio(audio_16k, thred=thred)


class RVCOfflineVoice:
    _hubert_cache: dict = {}
    _hubert_lock = threading.Lock()

    def __init__(self, model_path: str, index_path: str | None = None, device: str = "cpu",
                 is_half: bool = False):
        self.device = torch.device(device)
        self.model_path = model_path
        self.index_path = index_path
        self.is_half = bool(is_half) and self.device.type == "cuda"
        self.quantized = self.device.type == "cpu"

        cpt = torch.load(model_path, map_location="cpu")
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = int(cpt.get("f0", 1))
        self.version = cpt.get("version", "v1")

        try:
            synth_cls = _SYNTH_CLASSES[(self.version, self.if_f0)]
        except KeyError:
            raise ValueError(
                f"Unsupported checkpoint combination: version={self.version!r}, f0={self.if_f0!r}"
            )

        self.net_g = synth_cls(*cpt["config"], is_half=self.is_half)
        del self.net_g.enc_q

        load_result = self.net_g.load_state_dict(cpt["weight"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"[WARN] Voice checkpoint load mismatch for {model_path}: "
                f"missing_keys={load_result.missing_keys}, "
                f"unexpected_keys={load_result.unexpected_keys}"
            )

        self.net_g = (self.net_g.half() if self.is_half else self.net_g.float()).eval().to(self.device)
        self.net_g.remove_weight_norm()
        self.tgt_sr = cpt["config"][-1]

        self.index = None
        self.big_npy = None
        if index_path and faiss is not None and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = max(self.index.nprobe, 8)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        key = f"{self.device}:{'half' if self.is_half else 'float'}"
        with RVCOfflineVoice._hubert_lock:
            if key not in RVCOfflineVoice._hubert_cache:
                hubert = load_hubert_model(self.device, is_half=self.is_half)
                if self.device.type == "cpu":
                    hubert = torch.quantization.quantize_dynamic(
                        hubert, {torch.nn.Linear}, dtype=torch.qint8
                    )
                RVCOfflineVoice._hubert_cache[key] = hubert
            self.hubert = RVCOfflineVoice._hubert_cache[key]

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        rmvpe: RMVPEPitchExtractor,
        f0_up_key: float = 0,
        index_rate: float = 0.66,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        pad_seconds: float = 1.0,
        filter_radius: int = 3,
    ) -> tuple[np.ndarray, int]:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        audio_16k = _resample_np(audio, sample_rate, 16000)
        audio_16k = signal.filtfilt(_HP_B, _HP_A, audio_16k).astype(np.float32)
        audio_16k_for_rms = audio_16k

        pad_16k = int(round(max(pad_seconds, 0.0) * 16000))
        if pad_16k > 0:
            audio_16k = np.pad(audio_16k, (pad_16k, pad_16k), mode="reflect")

        audio_16k_t = torch.from_numpy(audio_16k).to(self.device)
        audio_16k_t = audio_16k_t.half() if self.is_half else audio_16k_t.float()

        feats_in = audio_16k_t.view(1, -1)
        padding_mask = torch.zeros(feats_in.shape, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            feats = extract_hubert_features(self.hubert, feats_in, self.version, padding_mask=padding_mask)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            feats0 = feats.clone()

        if self.index is not None and index_rate > 0:
            npy = feats[0].float().cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / np.maximum(score, 1e-9))
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                retrieved = torch.from_numpy(npy.astype("float32")).unsqueeze(0).to(self.device)
                retrieved = retrieved.half() if self.is_half else retrieved.float()
                feats = retrieved * index_rate + (1 - index_rate) * feats

        p_len = audio_16k_t.shape[0] // 160
        pitch = pitchf = None
        if self.if_f0:
            f0 = rmvpe.infer(audio_16k, thred=0.03)
            f0 = f0 * pow(2.0, f0_up_key / 12.0)
            if len(f0) < p_len:
                f0 = np.pad(f0, (0, p_len - len(f0)))
            f0 = f0[:p_len]
            uv = f0 == 0
            if np.any(~uv) and np.any(uv):
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            if filter_radius and filter_radius >= 3:
                r = filter_radius if filter_radius % 2 == 1 else filter_radius + 1
                f0 = signal.medfilt(f0, kernel_size=r)

            f0_min, f0_max = 50.0, 1100.0
            f0_mel_min = 1127 * np.log(1 + f0_min / 700)
            f0_mel_max = 1127 * np.log(1 + f0_max / 700)
            f0_mel = 1127 * np.log(1 + f0 / 700)
            voiced = f0_mel > 0
            f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
            f0_mel = np.clip(f0_mel, 1, 255)
            f0_coarse = np.rint(f0_mel).astype(np.int64)

            pitch = torch.from_numpy(f0_coarse).unsqueeze(0).to(self.device)
            pitchf = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).to(self.device)
            pitchf = pitchf.half() if self.is_half else pitchf.float()

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = min(feats.shape[1], p_len)
        feats = feats[:, :p_len, :]
        if self.if_f0:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

        if self.if_f0 and protect < 0.5:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            feats0 = feats0[:, :p_len, :]
            voiced_mask = pitchf.clone()
            voiced_mask[pitchf > 0] = 1
            voiced_mask[pitchf < 1] = protect
            feats = feats * voiced_mask.unsqueeze(-1) + feats0 * (1 - voiced_mask.unsqueeze(-1))

        phone_lengths = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)

        with torch.no_grad():
            if self.if_f0:
                out = self.net_g.infer(feats, phone_lengths, pitch, pitchf, sid)[0]
            else:
                out = self.net_g.infer(feats, phone_lengths, sid)[0]

        out = out.squeeze().float().cpu().numpy()

        if pad_16k > 0:
            pad_tgt = int(round(pad_seconds * self.tgt_sr))
            if pad_tgt > 0 and len(out) > 2 * pad_tgt:
                out = out[pad_tgt:-pad_tgt]

        if rms_mix_rate != 1.0:
            out = _change_rms(audio_16k_for_rms, 16000, out, self.tgt_sr, rms_mix_rate)

        return out, self.tgt_sr

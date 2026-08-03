"""
neural_env/rvc_pipeline.py

Offline (whole-clip) RVC voice conversion, built directly from the official
RVC-Project inference code -- infer/hubert.py, infer/rmvpe.py, and
infer/module/{models,modules,attentions,commons,transforms}.py, the exact
files this project's own infer/ folder ships -- instead of going through the
third-party `rvc-python` PyPI package the sidecar used before.

WHY THIS REPLACED rvc-python
-----------------------------
rvc-python is a community reimplementation of RVC-Project's inference code,
not the original. Loading the same trained .pth checkpoint through DIFFERENT
inference code does not reliably give the same result: HuBERT layer
selection, the exact f0-to-pitch-bin conversion, feature-index retrieval
blending, and the generator's exact forward pass all have to match what the
checkpoint was actually trained against, bit for bit, or the vocoder gets
fed inputs shaped/scaled in ways it never saw in training -- which is exactly
the "sounds like garbled noise, no intelligible words" failure mode, versus
the official WebUI's Model Inference tab (pm/rmvpe/fcpe + index-rate +
protect, same pipeline this file ports) producing clean, intelligible
output from the identical checkpoint.

This module is a direct port of that official pipeline's logic, using the
project's own files verbatim (see neural_env/infer/), wired up for OFFLINE
whole-track use instead of the realtime skip_head/return_length streaming
API in infer/rtrvc.py (that class exists for live, low-latency mic input;
an offline render has no such constraint and doesn't need the realtime
class's cache_pitch bookkeeping at all).
"""
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
    faiss = None  # index-based retrieval simply won't be available; index_rate is ignored


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


# Official pipeline.py's very first step on every input: a 5th-order
# Butterworth highpass at 48Hz, computed once at import time (bh/ah in
# their code) and reused for every call. It assumes its input is already
# 16kHz (that's what fs=16000 means to scipy.signal.butter -- Wn=48 is
# then interpreted as 48Hz against that sample rate). Our convert() below
# applies it AFTER resampling to 16k for that reason, not before, since
# our caller can hand us audio at any sample_rate.
_HP_B, _HP_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def _change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    """
    Ported verbatim from official pipeline.py's change_rms(). This is the
    WebUI's "Adjust the volume envelope scaling" slider (rms_mix_rate).

    data1/sr1: the pre-conversion input audio (what the DSP synth made).
    data2/sr2: the post-conversion generator output.
    rate: 1.0 keeps data2's own loudness envelope entirely; 0.0 fully
    replaces it with data1's envelope. The WebUI's own default is 0.25 --
    mostly following the generator's natural dynamics, with a bit of the
    original's shape mixed in to mask low-level synthesis noise.
    """
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
    """
    Thin, shared wrapper around infer/rmvpe.py's RMVPE so every loaded voice
    doesn't load its own separate copy of the (~180MB) checkpoint. One
    instance is created at sidecar startup and passed into every
    RVCOfflineVoice.convert() call.
    """

    def __init__(self, model_path: str, device: str = "cpu", is_half: bool = False):
        self._rmvpe = RMVPE(model_path, is_half=is_half, device=device)
        self._lock = threading.Lock()  # RMVPE isn't documented as call-thread-safe; be conservative

    def infer(self, audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        with self._lock:
            return self._rmvpe.infer_from_audio(audio_16k, thred=thred)


class RVCOfflineVoice:
    """
    One loaded voice model (.pth [+ optional .index]), ready to convert
    whole audio clips offline in a single pass. This mirrors
    get_synthesizer()/RVC.__init__() from infer/rtrvc.py, minus the
    realtime-only state (cache_pitch/cache_pitchf, block-frame accounting)
    that exists purely to support incremental, low-latency streaming.
    """

    _hubert_cache: dict = {}  # device-str -> loaded HuBERT model, shared across all voices
    _hubert_lock = threading.Lock()

    def __init__(self, model_path: str, index_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.index_path = index_path

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

        self.net_g = synth_cls(*cpt["config"], is_half=False)
        del self.net_g.enc_q  # only needed for training; matches rtrvc.get_synthesizer()

        # NOTE: strict=False means any checkpoint key that doesn't exactly
        # match a layer name in this model class is silently skipped --
        # that layer then keeps its random initial weights instead of the
        # trained ones. Loading still "succeeds" and shapes still line up
        # everywhere downstream, but a partially-random generator produces
        # exactly the symptom reported against this voice: confident,
        # correctly-pitched audio with no real phonetic content. The load
        # result was previously discarded; it's now captured and any
        # mismatch is logged so this can actually be ruled in or out
        # instead of assumed fine.
        load_result = self.net_g.load_state_dict(cpt["weight"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"[WARN] Voice checkpoint load mismatch for {model_path}: "
                f"missing_keys={load_result.missing_keys}, "
                f"unexpected_keys={load_result.unexpected_keys}"
            )

        self.net_g = self.net_g.float().eval().to(self.device)
        self.net_g.remove_weight_norm()
        self.tgt_sr = cpt["config"][-1]

        self.index = None
        self.big_npy = None
        if index_path and faiss is not None and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        key = str(self.device)
        with RVCOfflineVoice._hubert_lock:
            if key not in RVCOfflineVoice._hubert_cache:
                RVCOfflineVoice._hubert_cache[key] = load_hubert_model(self.device, is_half=False)
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
    ) -> tuple[np.ndarray, int]:
        """
        audio: mono float32 at `sample_rate`, the WHOLE clip -- there is no
        multi-chunk splitting here on purpose (official pipeline.py's
        opt_ts sliding-window split exists mainly to bound VRAM/compute on
        very long files; for the short single-note and short-track clips
        this sidecar actually receives, official pipeline.py effectively
        runs them through in one `vc()` call too). That's what makes this
        the offline path: infer/rtrvc.py's skip_head/return_length
        machinery is for a realtime caller re-running this generator on
        overlapping small blocks without an audible seam; an offline
        render has no such requirement.

        pad_seconds: mirrors official pipeline.py's x_pad/t_pad/t_pad_tgt.
        `pad_seconds` of reflect-padded context is added to BOTH ends
        before HuBERT/RMVPE/the generator ever see the audio, then the
        matching amount is trimmed back off the generator's OUTPUT
        afterward. Without this, the very first and last stretch of every
        clip runs with zero left/right context on one side -- exactly
        where you'd expect a chewed-sounding onset/offset -- and since
        this sidecar mostly renders short single notes, that edge is a
        much bigger fraction of the total clip here than it is for a
        30-second WebUI file. Set to 0 to disable.

        rms_mix_rate: mirrors the WebUI's "Adjust the volume envelope
        scaling" slider; see _change_rms() above. Set to 1.0 to disable
        (keep the generator's own loudness untouched).

        Returns (converted_audio, tgt_sr) -- tgt_sr is the voice model's own
        native output sample rate (e.g. 40000/48000), NOT necessarily
        `sample_rate`; resample the result yourself if you need it to match.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        audio_16k = _resample_np(audio, sample_rate, 16000)

        # Official pipeline.py's highpass, applied here (post-resample)
        # since _HP_B/_HP_A were designed assuming a 16kHz signal.
        audio_16k = signal.filtfilt(_HP_B, _HP_A, audio_16k).astype(np.float32)
        # Keep an unpadded copy for the RMS-envelope step at the end --
        # official change_rms() compares against the original (highpass-
        # filtered, but NOT context-padded) input.
        audio_16k_for_rms = audio_16k

        pad_16k = int(round(max(pad_seconds, 0.0) * 16000))
        if pad_16k > 0:
            audio_16k = np.pad(audio_16k, (pad_16k, pad_16k), mode="reflect")

        audio_16k_t = torch.from_numpy(audio_16k).float().to(self.device)

        # ---- HuBERT content features (50Hz) ----
        feats_in = audio_16k_t.view(1, -1)
        padding_mask = torch.zeros(feats_in.shape, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            feats = extract_hubert_features(self.hubert, feats_in, self.version, padding_mask=padding_mask)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            feats0 = feats.clone()  # cloned AFTER padding so it stays frame-aligned with
            # feats through interpolation -- cloning before the pad left feats0 one frame
            # short at 50Hz (two frames short at 100Hz), which is what caused the
            # "size of tensor a (N) must match size of tensor b (N+1)" crash in the
            # protect blend below.

        # ---- index-based feature retrieval (accent-strength slider in the
        # official UI), exactly as in infer/rtrvc.py's RVC.infer() ----
        if self.index is not None and index_rate > 0:
            npy = feats[0].cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / np.maximum(score, 1e-9))
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                retrieved = torch.from_numpy(npy.astype("float32")).unsqueeze(0).to(self.device)
                feats = retrieved * index_rate + (1 - index_rate) * feats

        # ---- pitch (100Hz via RMVPE) ----
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

        # ---- upsample HuBERT features 50Hz -> 100Hz to align with pitch,
        # then truncate everything to a common frame count ----
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = min(feats.shape[1], p_len)
        feats = feats[:, :p_len, :]
        if self.if_f0:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

        # ---- "protect": blend the least-voiced-sounding frames back toward
        # the un-retrieved HuBERT features, so unvoiced consonants/breath
        # don't get pulled all the way toward the target voice's retrieved
        # timbre and turn into the tearing/artifact sound the slider's
        # tooltip describes. Skipped entirely at protect=0.5 ("disabled"). ----
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

        # Trim the context padding back off, now in tgt_sr-space (mirrors
        # official pipeline.py's `[t_pad_tgt : -t_pad_tgt]` slice).
        if pad_16k > 0:
            pad_tgt = int(round(pad_seconds * self.tgt_sr))
            if pad_tgt > 0 and len(out) > 2 * pad_tgt:
                out = out[pad_tgt:-pad_tgt]

        # Volume envelope mix -- official pipeline.py runs this as the
        # very last step before final int16-range normalization.
        if rms_mix_rate != 1.0:
            out = _change_rms(audio_16k_for_rms, 16000, out, self.tgt_sr, rms_mix_rate)

        return out, self.tgt_sr
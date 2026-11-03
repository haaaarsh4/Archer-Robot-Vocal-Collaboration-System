"""The two transformers SingSong trains, written from the paper's description.

Stage 1+2 is a T5-style encoder-decoder.  The encoder reads the vocal's
semantic codes; the decoder emits ``[instrumental semantic; coarse]``.  Stage 3
is a decoder-only transformer that reads the coarse codes as a prefix and emits
the eight fine streams.  Both use fixed sinusoidal positional encodings (the
paper replaces T5's relative bias with these), pre-RMSNorm, gated GEGLU
feed-forwards, and a tied output embedding.

The implementation exposes an incremental decoding path with a
per-layer KV cache.  Generating 2250 + 4000 tokens one at a time is the normal
inference mode; recomputing the whole prefix for every token would make a
10-second clip cost minutes even on a GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FineConfig, Stage12Config
from .tokens import (
    CODEBOOK_SIZE,
    FINE_VOCAB_SIZE,
    NUM_COARSE_QUANTIZERS,
    NUM_FINE_QUANTIZERS,
    STAGE12_VOCAB_SIZE,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos positions, as in the original Transformer and in SingSong."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        if dim % 2:
            raise ValueError("d_model must be even for sinusoidal positions")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        length = x.shape[1]
        if offset + length > self.max_len:
            raise ValueError(
                f"position {offset + length} exceeds the configured maximum {self.max_len}"
            )
        return x + self.pe[:, offset:offset + length].to(dtype=x.dtype, device=x.device)


def _causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    """True where a query may attend.  Handles a cached prefix correctly."""
    q_pos = torch.arange(key_len - query_len, key_len, device=device)
    k_pos = torch.arange(key_len, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"d_model {dim} is not divisible by {heads} heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out = nn.Linear(dim, dim, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        memory: torch.Tensor | None = None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        causal: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        b, t, _ = x.shape
        if memory is not None and past is not None:
            q = self._split(self.qkv(x)[..., : self.dim])
            k, v = past
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q, k, v = self._split(q), self._split(k), self._split(v)
            if memory is not None:
                mkv = self.qkv(memory)
                _, k, v = mkv.chunk(3, dim=-1)
                k, v = self._split(k), self._split(v)
            elif past is not None:
                k = torch.cat((past[0], k), dim=2)
                v = torch.cat((past[1], v), dim=2)

        if memory is not None:
            mask = None
        elif causal:
            mask = _causal_mask(t, k.shape[2], x.device)
        else:
            mask = None

        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(b, t, self.dim)
        out = self.out(attended)
        cache = (k, v) if use_cache else None
        return out, cache


class GEGLU(nn.Module):
    """T5's gated feed-forward: gelu(W0 x) * W1 x, then W2."""

    def __init__(self, dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w0 = nn.Linear(dim, d_ff, bias=False)
        self.w1 = nn.Linear(dim, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w0(x)) * self.w1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = Attention(dim, heads, attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = GEGLU(dim, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x))[0])
        x = x + self.ffn(self.norm2(x))
        return x


class CausalLayer(nn.Module):
    """Decoder-only block used by the fine stage (no cross-attention)."""

    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = Attention(dim, heads, attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = GEGLU(dim, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        causal: bool = True,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        s_out, cache = self.self_attn(
            self.norm1(x), past=past, causal=causal, use_cache=use_cache
        )
        x = x + self.dropout(s_out)
        x = x + self.ffn(self.norm2(x))
        return x, cache


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, d_ff: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = Attention(dim, heads, attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = Attention(dim, heads, attn_dropout)
        self.norm3 = RMSNorm(dim)
        self.ffn = GEGLU(dim, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        *,
        causal: bool = True,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        self_past = None if past is None else (past[0], past[1])
        cross_past = None if past is None else (past[2], past[3])

        s_out, s_cache = self.self_attn(
            self.norm1(x), past=self_past, causal=causal, use_cache=use_cache
        )
        x = x + self.dropout(s_out)
        c_out, c_cache = self.cross_attn(
            self.norm2(x), memory=memory, past=cross_past, use_cache=use_cache
        )
        x = x + self.dropout(c_out)
        x = x + self.ffn(self.norm3(x))

        cache = None
        if use_cache:
            cache = (s_cache[0], s_cache[1], c_cache[0], c_cache[1])  # type: ignore[index]
        return x, cache


class Stage12Model(nn.Module):
    """T5-style encoder-decoder: vocal semantics -> instrumental semantic + coarse."""

    def __init__(self, cfg: Stage12Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(STAGE12_VOCAB_SIZE, cfg.d_model)
        self.position = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(cfg.d_model, cfg.heads, cfg.d_ff, cfg.dropout, cfg.attention_dropout)
            for _ in range(cfg.encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(cfg.d_model, cfg.heads, cfg.d_ff, cfg.dropout, cfg.attention_dropout)
            for _ in range(cfg.decoder_layers)
        )
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.decoder_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, STAGE12_VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def vocab_size(self) -> int:
        return STAGE12_VOCAB_SIZE

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.position(self.token_embedding(input_ids))
        if self.cfg.gradient_checkpointing and self.training:
            for layer in self.encoder_layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.encoder_layers:
                x = layer(x)
        return self.encoder_norm(x)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits for every target position.

        ``target_ids`` are the *targets* (no BOS).  The decoder is shifted right
        internally, so a target of length T is predicted from [BOS, y_0..y_{T-2}].
        """
        if target_ids.ndim != 2:
            raise ValueError("target_ids must be [batch, time]")
        memory = self.encode(input_ids)
        bos = torch.full(
            (target_ids.shape[0], 1), STAGE12_VOCAB_SIZE - 2,
            dtype=target_ids.dtype, device=target_ids.device,
        )
        dec_in = torch.cat((bos, target_ids[:, :-1]), dim=1)
        x = self.position(self.token_embedding(dec_in))
        if self.cfg.gradient_checkpointing and self.training:
            for layer in self.decoder_layers:
                x, _ = torch.utils.checkpoint.checkpoint(
                    lambda h, layer=layer: layer(
                        h, memory, None, causal=True, use_cache=False
                    ),
                    x, use_reentrant=False,
                )
        else:
            for layer in self.decoder_layers:
                x, _ = layer(x, memory, causal=True, use_cache=False)
        return self.lm_head(self.decoder_norm(x))

    @torch.no_grad()
    def decode_step(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        cache: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """One incremental decoding step.  ``token`` is [batch] or [batch, 1]."""
        if token.ndim == 1:
            token = token.unsqueeze(1)
        offset = 0 if cache is None else int(cache[0][0].shape[2])
        x = self.position(self.token_embedding(token), offset=offset)
        new_cache: list = []
        if cache is None:
            cache = [None] * len(self.decoder_layers)
        for layer, past in zip(self.decoder_layers, cache):
            x, layer_cache = layer(
                x, memory, past, causal=True, use_cache=True
            )
            new_cache.append(layer_cache)
        logits = self.lm_head(self.decoder_norm(x))[:, -1, :]
        return logits, new_cache


class FineModel(nn.Module):
    """Decoder-only model: coarse codes (prefix) -> eight fine streams."""

    def __init__(self, cfg: FineConfig):
        super().__init__()
        self.cfg = cfg
        self.coarse_embedding = nn.Embedding(
            NUM_COARSE_QUANTIZERS * CODEBOOK_SIZE, cfg.d_model
        )
        self.fine_embedding = nn.Embedding(FINE_VOCAB_SIZE, cfg.d_model)
        self.position = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.layers = nn.ModuleList(
            CausalLayer(
                cfg.d_model, cfg.heads, cfg.d_ff, cfg.dropout, cfg.attention_dropout
            )
            for _ in range(cfg.layers)
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, FINE_VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.fine_embedding.weight
        self.apply(Stage12Model._init_weights)

    @property
    def fine_vocab_size(self) -> int:
        return FINE_VOCAB_SIZE

    def forward(self, coarse_ids: torch.Tensor, fine_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced fine logits for every requested fine position.

        ``coarse_ids`` is the flattened 4-stream prefix, already in the coarse
        vocabulary (0..4095).  ``fine_ids`` are the shifted-right targets, i.e.
        [BOS, y_0..y_{T-1}], and the returned logits align with them one-to-one.
        """
        if coarse_ids.ndim != 2 or fine_ids.ndim != 2:
            raise ValueError("coarse_ids and fine_ids must both be [batch, time]")
        prefix = self.position(self.coarse_embedding(coarse_ids), offset=0)
        suffix = self.position(
            self.fine_embedding(fine_ids), offset=coarse_ids.shape[1]
        )
        x = torch.cat((prefix, suffix), dim=1)
        if self.cfg.gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    lambda h, layer=layer: layer(h, causal=True, use_cache=False)[0],
                    x, use_reentrant=False,
                )
        else:
            for layer in self.layers:
                x, _ = layer(x, causal=True, use_cache=False)
        x = self.norm(x)[:, coarse_ids.shape[1]:, :]
        return self.lm_head(x)

    @torch.no_grad()
    def prefill_prefix(self, coarse_ids: torch.Tensor) -> list:
        """Run the coarse prefix once and return per-layer self-attention caches."""
        x = self.position(self.coarse_embedding(coarse_ids), offset=0)
        caches: list = []
        for layer in self.layers:
            x, cache = layer(x, causal=True, use_cache=True)
            caches.append(cache)
        return caches

    @torch.no_grad()
    def decode_step(
        self,
        token: torch.Tensor,
        cache: list,
        offset: int,
    ) -> tuple[torch.Tensor, list]:
        if token.ndim == 1:
            token = token.unsqueeze(1)
        x = self.position(self.fine_embedding(token), offset=offset)
        new_cache: list = []
        for layer, past in zip(self.layers, cache):
            x, layer_cache = layer(x, past, causal=True, use_cache=True)
            new_cache.append(layer_cache)
        return self.lm_head(self.norm(x))[:, -1, :], new_cache


def build_stage12(cfg: Stage12Config) -> Stage12Model:
    return Stage12Model(cfg)


def build_fine(cfg: FineConfig) -> FineModel:
    return FineModel(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> dict[str, int | str]:
    cfg = model.cfg
    return {
        "class": type(model).__name__,
        "parameters": count_parameters(model),
        "parameters_m": round(count_parameters(model) / 1e6, 1),
        "d_model": cfg.d_model,
        "layers": getattr(cfg, "layers", getattr(cfg, "encoder_layers", None)),
        "heads": cfg.heads,
        "d_ff": cfg.d_ff,
    }

"""Constrained sampling for the two SingSong stages.

The model has one output vocabulary per stage, but not every token is legal at
every position.  A semantic position may only emit 0..1023, coarse position
(q, t) may only emit the 1024 ids belonging to quantizer q, and a fine position
may only emit its own quantizer's block.  The paper enforces this by truncating
invalid subsets from the distribution; this module does the same thing with a
mask before temperature/top-k sampling.

Both generators accept a prompt.  That is how the paper's long-form mode works:
a 10 s vocal window is conditioned on, while the decoder is primed with the
preceding 5 s of already-generated instrumental, so successive windows keep the
same key, tempo and instrumentation instead of restarting.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import FineModel, Stage12Model
from .tokens import FineLayout, Stage12Layout


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.85
    top_k: int = 0
    seed: int | None = None


def _constrain(logits: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Set everything outside ``[lo, hi)`` to -inf, without copying the model."""
    if lo <= 0 and hi >= logits.shape[-1]:
        return logits
    out = logits.clone()
    if lo > 0:
        out[..., :lo] = float("-inf")
    if hi < out.shape[-1]:
        out[..., hi:] = float("-inf")
    return out


def _pick(logits: torch.Tensor, cfg: GenerationConfig, generator: torch.Generator | None) -> torch.Tensor:
    if cfg.temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / float(cfg.temperature)
    if cfg.top_k and cfg.top_k > 0:
        k = min(int(cfg.top_k), logits.shape[-1])
        values, _ = torch.topk(logits, k, dim=-1)
        cutoff = values[..., -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


@torch.no_grad()
def generate_stage12(
    model: Stage12Model,
    semantic_ids: torch.Tensor,
    layout: Stage12Layout,
    *,
    config: GenerationConfig | None = None,
    prefix_tokens: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate ``layout.target_length`` semantic+coarse tokens.

    ``prefix_tokens`` is a [batch, P] target-token prompt in the same packed
    layout.  When supplied, the returned sequence is P tokens of prompt followed
    by ``target_length - P`` generated tokens, exactly the paper's 5 s
    continuation setup.
    """
    cfg = config or GenerationConfig()
    device = device or next(model.parameters()).device
    semantic_ids = semantic_ids.to(device)
    model.eval()

    batch = semantic_ids.shape[0]
    target_len = layout.target_length
    if prefix_tokens is None:
        prefix = torch.empty((batch, 0), dtype=torch.long, device=device)
    else:
        prefix = prefix_tokens.to(device)
        if prefix.shape[0] != batch:
            raise ValueError("prefix batch does not match semantic batch")
        if prefix.shape[1] >= target_len:
            raise ValueError("prefix is already as long as the target sequence")

    prefix_len = prefix.shape[1]
    new_tokens = target_len - prefix_len
    memory = model.encode(semantic_ids)
    generator = _make_generator(device, cfg.seed)

    # Decoder input is [BOS, y_0, ..., y_{P-1}]; the logits after BOS predict y_0,
    # and the logits after y_{P-1} predict the first token we actually sample.
    current = torch.full((batch,), layout.bos_id, dtype=torch.long, device=device)
    cache = None
    logits, cache = model.decode_step(current, memory, cache)
    for pos in range(prefix_len):
        current = prefix[:, pos]
        logits, cache = model.decode_step(current, memory, cache)

    generated: list[torch.Tensor] = []
    for pos in range(prefix_len, target_len):
        lo, hi = layout.position_range(pos)
        step_logits = _constrain(logits, lo, hi)
        token = _pick(step_logits, cfg, generator)
        generated.append(token)
        if pos + 1 < target_len:
            logits, cache = model.decode_step(token, memory, cache)

    if generated:
        new = torch.stack(generated, dim=1)
        return torch.cat((prefix, new), dim=1)
    return prefix


@torch.no_grad()
def generate_fine(
    model: FineModel,
    coarse_ids: torch.Tensor,
    layout: FineLayout,
    *,
    config: GenerationConfig | None = None,
    prefix_tokens: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate the ``layout.fine_length`` fine tokens given a coarse prefix."""
    cfg = config or GenerationConfig()
    device = device or next(model.parameters()).device
    coarse_ids = coarse_ids.to(device)
    model.eval()

    batch = coarse_ids.shape[0]
    total = layout.fine_length
    if prefix_tokens is None:
        prefix = torch.empty((batch, 0), dtype=torch.long, device=device)
    else:
        prefix = prefix_tokens.to(device)
        if prefix.shape[0] != batch:
            raise ValueError("fine prefix batch does not match coarse batch")
        if prefix.shape[1] >= total:
            raise ValueError("fine prefix is already as long as the target sequence")

    prefix_len = prefix.shape[1]
    cache = model.prefill_prefix(coarse_ids)
    generator = _make_generator(device, cfg.seed)

    # Position P is the first fine target position; BOS occupies it in the
    # decoder input, and its logits predict fine token 0.
    offset = coarse_ids.shape[1]
    current = torch.full((batch,), layout.bos_id, dtype=torch.long, device=device)
    logits, cache = model.decode_step(current, cache, offset)
    for pos in range(prefix_len):
        current = prefix[:, pos]
        offset += 1
        logits, cache = model.decode_step(current, cache, offset)

    generated: list[torch.Tensor] = []
    for pos in range(prefix_len, total):
        lo, hi = layout.fine_position_range(pos)
        step_logits = _constrain(logits, lo, hi)
        token = _pick(step_logits, cfg, generator)
        generated.append(token)
        if pos + 1 < total:
            offset += 1
            logits, cache = model.decode_step(token, cache, offset)

    if generated:
        new = torch.stack(generated, dim=1)
        return torch.cat((prefix, new), dim=1)
    return prefix


def greedy_stage12(
    model: Stage12Model,
    semantic_ids: torch.Tensor,
    layout: Stage12Layout,
    **kwargs,
) -> torch.Tensor:
    return generate_stage12(
        model, semantic_ids, layout, config=GenerationConfig(temperature=0.0), **kwargs
    )


def greedy_fine(
    model: FineModel,
    coarse_ids: torch.Tensor,
    layout: FineLayout,
    **kwargs,
) -> torch.Tensor:
    return generate_fine(
        model, coarse_ids, layout, config=GenerationConfig(temperature=0.0), **kwargs
    )

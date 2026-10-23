"""The transformers and sampler, held to their shapes and constraints.

The cache equivalence test is the important one: generation is incremental, so
if the cached path drifts from the teacher-forced path the model still trains
and still produces audio, but the audio is sampled from a different model than
the one that was trained.  The tolerance is tight because both paths are the
same arithmetic in a different order.
"""
from __future__ import annotations

import torch

from sing_song.config import FineConfig, Stage12Config
from sing_song.model import build_fine, build_stage12
from sing_song.sampling import GenerationConfig, generate_fine, generate_stage12
from sing_song.tokens import FineLayout, Stage12Layout


def _stage12() -> tuple:
    cfg = Stage12Config(
        d_model=32, encoder_layers=1, decoder_layers=1, heads=4, d_ff=64, dropout=0.0
    )
    return build_stage12(cfg).eval(), Stage12Layout(semantic_frames=4, coarse_frames=3)


def _fine() -> tuple:
    cfg = FineConfig(d_model=32, layers=1, heads=4, d_ff=64, dropout=0.0)
    return build_fine(cfg).eval(), FineLayout(coarse_frames=3, fine_frames=3)


def test_stage12_forward_shape_and_finite_loss():
    model, layout = _stage12()
    semantic = torch.randint(0, 1024, (2, layout.semantic_frames))
    semantic_ids = torch.randint(0, 1024, (2, layout.semantic_frames)).tolist()
    coarse = [
        torch.randint(0, 1024, (2, layout.coarse_frames)).tolist()
        for _ in range(layout.num_coarse)
    ]
    # transpose the per-batch construction into [batch][quantizer][frame]
    targets = []
    for b in range(2):
        targets.append(layout.build_target(
            semantic_ids[b],
            [[coarse[q][b][t] for t in range(layout.coarse_frames)] for q in range(layout.num_coarse)],
        ))
    target = torch.tensor(targets)
    logits = model(semantic, target)
    assert logits.shape == (2, layout.target_length, 5122)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_stage12_cached_logits_match_the_teacher_forced_path():
    model, layout = _stage12()
    semantic = torch.randint(0, 1024, (2, layout.semantic_frames))
    target = torch.randint(0, 1024, (2, layout.target_length))
    # rebuild valid targets
    valid = []
    for b in range(2):
        sem = torch.randint(0, 1024, (layout.semantic_frames,)).tolist()
        coarse = [torch.randint(0, 1024, (layout.coarse_frames,)).tolist()
                  for _ in range(layout.num_coarse)]
        valid.append(layout.build_target(sem, coarse))
    target = torch.tensor(valid)

    with torch.no_grad():
        full = model(semantic, target)
        memory = model.encode(semantic)
        current = torch.full((2,), layout.bos_id, dtype=torch.long)
        logits, cache = model.decode_step(current, memory, None)
        incremental = []
        for pos in range(layout.target_length):
            incremental.append(logits)
            if pos + 1 < layout.target_length:
                logits, cache = model.decode_step(target[:, pos], memory, cache)
        incremental = torch.stack(incremental, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5, rtol=1e-4)


def test_generated_tokens_respect_every_position_range():
    model, layout = _stage12()
    semantic = torch.randint(0, 1024, (2, layout.semantic_frames))
    out = generate_stage12(
        model, semantic, layout, config=GenerationConfig(temperature=0.0)
    )
    assert out.shape == (2, layout.target_length)
    for pos in range(layout.target_length):
        lo, hi = layout.position_range(pos)
        assert ((out[:, pos] >= lo) & (out[:, pos] < hi)).all(), pos


def test_stage12_prefix_is_preserved_and_extends_to_the_same_length():
    model, layout = _stage12()
    semantic = torch.randint(0, 1024, (2, layout.semantic_frames))
    prefix = torch.randint(0, 1024, (2, 2))
    # make the prefix legal
    for pos in range(2):
        lo, hi = layout.position_range(pos)
        prefix[:, pos] = lo + (prefix[:, pos] % (hi - lo))
    out = generate_stage12(
        model, semantic, layout, config=GenerationConfig(temperature=0.0),
        prefix_tokens=prefix,
    )
    assert out.shape == (2, layout.target_length)
    assert torch.equal(out[:, :2], prefix)


def test_fine_forward_shape_and_cache_equivalence():
    model, layout = _fine()
    coarse = torch.randint(0, 4096, (2, layout.coarse_prefix_length))
    fine = torch.randint(0, 8192, (2, layout.fine_length))
    # valid fine targets
    fine = torch.stack([
        torch.tensor(layout.build_fine([
            torch.randint(0, 1024, (layout.fine_frames,)).tolist()
            for _ in range(layout.num_fine)
        ])) for _ in range(2)
    ])
    bos = torch.full((2, 1), layout.bos_id, dtype=torch.long)
    with torch.no_grad():
        full = model(coarse, torch.cat((bos, fine[:, :-1]), dim=1))
        cache = model.prefill_prefix(coarse)
        offset = coarse.shape[1]
        logits, cache = model.decode_step(bos.squeeze(1), cache, offset)
        incremental = []
        for pos in range(layout.fine_length):
            incremental.append(logits)
            if pos + 1 < layout.fine_length:
                offset += 1
                logits, cache = model.decode_step(fine[:, pos], cache, offset)
        incremental = torch.stack(incremental, dim=1)
    assert full.shape == (2, layout.fine_length, 8194)
    assert torch.allclose(full, incremental, atol=1e-5, rtol=1e-4)


def test_fine_generation_stays_inside_each_quantizer_block():
    model, layout = _fine()
    coarse = torch.randint(0, 4096, (2, layout.coarse_prefix_length))
    out = generate_fine(
        model, coarse, layout, config=GenerationConfig(temperature=0.0)
    )
    assert out.shape == (2, layout.fine_length)
    for pos in range(layout.fine_length):
        lo, hi = layout.fine_position_range(pos)
        assert ((out[:, pos] >= lo) & (out[:, pos] < hi)).all(), pos

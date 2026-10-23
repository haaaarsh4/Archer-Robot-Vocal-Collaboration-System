"""Training loops for the two SingSong transformer stages.

The paper trains stage 1+2 and stage 3 independently: stage 1+2 on
``P(Sem(y), Coarse(y) | Sem(noisy x))`` and stage 3 on
``P(Fine(y) | Coarse(y))``.  This module mirrors that split.  Both loops use
teacher forcing, cross-entropy on every target code, gradient accumulation, a
warmup + cosine schedule, mixed precision on CUDA, and resumable checkpoints.

The published run is 200k steps at an effective batch of 512 on 10 s clips.  The
defaults below express that recipe with a smaller per-step batch and a larger
accumulation, so the same command works on one GPU; a cluster can raise
``--batch-size`` and set ``--grad-accum`` to 1.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .config import FineConfig, Stage12Config
from .data import TokenDataset
from .model import FineModel, Stage12Model, build_fine, build_stage12
from .tokens import FineLayout, Stage12Layout


@dataclass
class TrainConfig:
    steps: int = 200_000
    batch_size: int = 8
    grad_accum: int = 64
    lr: float = 1e-4
    warmup_steps: int = 4_000
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    log_every: int = 50
    save_every: int = 1_000
    val_every: int = 2_000
    val_batches: int = 20
    valid_frac: float = 0.05
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    checkpoint_dir: str = "checkpoints/singsong"
    resume_from: str | None = None
    compile: bool = False
    distributed: bool = True

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TrainConfig":
        raw = dict(raw or {})
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _dist_info() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world, local
    return 0, 1, 0


def _setup_distributed(cfg: TrainConfig) -> tuple[int, int, int]:
    rank, world, local = _dist_info()
    if world > 1 and cfg.distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local)
    return rank, world, local


def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device not in (None, "", "auto"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    if cfg.dtype in ("float16", "fp16"):
        return torch.float16
    if cfg.dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    if cfg.dtype == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _split_dataset(dataset: TokenDataset, frac: float, seed: int) -> tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(n * frac)))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def _stage12_collate(layout: Stage12Layout):
    def collate(batch: list[dict]) -> dict:
        semantic = torch.stack([item["semantic"] for item in batch])
        coarse = torch.stack([item["coarse"] for item in batch])  # [B, 4, T]
        offsets = torch.arange(layout.num_coarse).view(1, -1, 1) * layout.codebook_size
        coarse_ids = coarse + offsets
        target = torch.cat(
            (semantic, coarse_ids.transpose(1, 2).reshape(semantic.shape[0], -1)),
            dim=1,
        )
        return {"semantic": semantic, "coarse": coarse, "target": target}
    return collate


def _fine_collate(layout: FineLayout):
    def collate(batch: list[dict]) -> dict:
        coarse = torch.stack([item["coarse"] for item in batch])  # [B, 4, T]
        fine = torch.stack([item["fine"] for item in batch])      # [B, 8, T]
        coarse_offsets = torch.arange(layout.num_coarse).view(1, -1, 1) * layout.codebook_size
        fine_offsets = torch.arange(layout.num_fine).view(1, -1, 1) * layout.codebook_size
        coarse_prefix = (coarse + coarse_offsets).transpose(1, 2).reshape(coarse.shape[0], -1)
        fine_target = (fine + fine_offsets).transpose(1, 2).reshape(fine.shape[0], -1)
        return {"coarse": coarse, "fine": fine, "coarse_prefix": coarse_prefix, "target": fine_target}
    return collate


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    cfg,
    train_cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "model_config": asdict(cfg),
        "train_config": asdict(train_cfg),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if all(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))


def _nll_stage12(model: Stage12Model, loader, device, dtype, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            semantic = batch["semantic"].to(device)
            target = batch["target"].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(semantic, target)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total += float(loss)
            count += 1
    model.train()
    return total / max(1, count)


def _nll_fine(model: FineModel, loader, device, dtype, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            coarse = batch["coarse_prefix"].to(device)
            target = batch["target"].to(device)
            bos = torch.full((target.shape[0], 1), model.fine_vocab_size - 2, dtype=torch.long, device=device)
            dec_in = torch.cat((bos, target[:, :-1]), dim=1)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(coarse, dec_in)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total += float(loss)
            count += 1
    model.train()
    return total / max(1, count)


def train_stage12(
    model_cfg: Stage12Config,
    data_cfg,
    train_cfg: TrainConfig,
    *,
    layout: Stage12Layout | None = None,
) -> dict:
    layout = layout or Stage12Layout.for_seconds(data_cfg.clip_seconds)
    rank, world, local = _setup_distributed(train_cfg)
    device = _resolve_device(train_cfg)
    dtype = _resolve_dtype(train_cfg, device)
    torch.manual_seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)

    dataset = TokenDataset(data_cfg.token_dir, expected_frames=layout.semantic_frames)
    train_set, val_set = _split_dataset(dataset, train_cfg.valid_frac, train_cfg.seed)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=train_cfg.num_workers, drop_last=True,
        collate_fn=_stage12_collate(layout), pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=max(1, train_cfg.num_workers // 2),
        collate_fn=_stage12_collate(layout),
    )

    model = build_stage12(model_cfg).to(device)
    if train_cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world > 1 and train_cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None
        )
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, train_cfg.warmup_steps, train_cfg.steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    start_step = 0
    if train_cfg.resume_from:
        start_step = _load_checkpoint(train_cfg.resume_from, raw_model, optimizer, scheduler)

    out_dir = Path(train_cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stage12_train.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    t0 = time.time()
    last: dict = {"step": step, "loss": None, "nll": None}

    while step < train_cfg.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            semantic = batch["semantic"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(semantic, target)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
                loss = loss / train_cfg.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % train_cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last = {"step": step, "loss": float(loss.detach()) * train_cfg.grad_accum}
                if rank == 0 and step % train_cfg.log_every == 0:
                    nll = _nll_stage12(raw_model, val_loader, device, dtype, train_cfg.val_batches)
                    last["nll"] = nll
                    record = {
                        **last,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    print(f"[stage12] step {step} loss={last['loss']:.3f} nll={nll:.3f} lr={record['lr']:.2e}")
                if rank == 0 and step % train_cfg.save_every == 0:
                    _save_checkpoint(out_dir / "stage12.pt", raw_model, optimizer, scheduler, step, model_cfg, train_cfg)
                    print(f"[stage12] checkpoint at {step}")
                if step >= train_cfg.steps:
                    break
    if rank == 0:
        _save_checkpoint(out_dir / "stage12.pt", raw_model, optimizer, scheduler, step, model_cfg, train_cfg)
    if world > 1 and train_cfg.distributed:
        dist.barrier()
        dist.destroy_process_group()
    return last


def train_fine(
    model_cfg: FineConfig,
    data_cfg,
    train_cfg: TrainConfig,
    *,
    layout: FineLayout | None = None,
) -> dict:
    layout = layout or FineLayout.for_seconds(data_cfg.clip_seconds)
    rank, world, local = _setup_distributed(train_cfg)
    device = _resolve_device(train_cfg)
    dtype = _resolve_dtype(train_cfg, device)
    torch.manual_seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)

    expected_semantic = int(round(data_cfg.clip_seconds * 25))
    dataset = TokenDataset(data_cfg.token_dir, expected_frames=expected_semantic)
    train_set, val_set = _split_dataset(dataset, train_cfg.valid_frac, train_cfg.seed)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        train_set, batch_size=train_cfg.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=train_cfg.num_workers, drop_last=True,
        collate_fn=_fine_collate(layout), pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=max(1, train_cfg.num_workers // 2),
        collate_fn=_fine_collate(layout),
    )

    model = build_fine(model_cfg).to(device)
    if train_cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world > 1 and train_cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None
        )
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, train_cfg.warmup_steps, train_cfg.steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))
    start_step = 0
    if train_cfg.resume_from:
        start_step = _load_checkpoint(train_cfg.resume_from, raw_model, optimizer, scheduler)

    out_dir = Path(train_cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fine_train.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    t0 = time.time()
    last: dict = {"step": step, "loss": None, "nll": None}

    while step < train_cfg.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            coarse = batch["coarse_prefix"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            bos = torch.full(
                (target.shape[0], 1), raw_model.fine_vocab_size - 2,
                dtype=torch.long, device=device,
            )
            dec_in = torch.cat((bos, target[:, :-1]), dim=1)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(coarse, dec_in)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
                loss = loss / train_cfg.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % train_cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last = {"step": step, "loss": float(loss.detach()) * train_cfg.grad_accum}
                if rank == 0 and step % train_cfg.log_every == 0:
                    nll = _nll_fine(raw_model, val_loader, device, dtype, train_cfg.val_batches)
                    last["nll"] = nll
                    record = {
                        **last,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    print(f"[fine] step {step} loss={last['loss']:.3f} nll={nll:.3f} lr={record['lr']:.2e}")
                if rank == 0 and step % train_cfg.save_every == 0:
                    _save_checkpoint(out_dir / "fine.pt", raw_model, optimizer, scheduler, step, model_cfg, train_cfg)
                    print(f"[fine] checkpoint at {step}")
                if step >= train_cfg.steps:
                    break
    if rank == 0:
        _save_checkpoint(out_dir / "fine.pt", raw_model, optimizer, scheduler, step, model_cfg, train_cfg)
    if world > 1 and train_cfg.distributed:
        dist.barrier()
        dist.destroy_process_group()
    return last

import math
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, random_split

from translation.dataset import PAD_ID, TranslationDataset, collate_batch
from translation.model import TransformerMT

CORPUS_PATH = Path("data/processed/parallel_cr_en.jsonl")
SPM_MODEL_PATH = "data/models/spm.model"
CHECKPOINT_PATH = Path("data/models/transformer_mt.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs: int = 60, batch_size: int = 32, lr: float = 3e-4,
          val_split: float = 0.1, patience: int = 8, weight_decay: float = 0.01):
    if not Path(SPM_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"{SPM_MODEL_PATH} not found. Run 'python -m translation.tokenizer' first."
        )

    dataset = TranslationDataset(CORPUS_PATH, SPM_MODEL_PATH)
    logger.info(f"Loaded {len(dataset)} training pairs, vocab size {dataset.sp.get_piece_size()}")
    if len(dataset) < 50:
        raise ValueError(
            f"Only {len(dataset)} usable pairs after filtering - too little data to train "
            f"a transformer meaningfully. Expand the corpus first."
        )

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train/val split: {train_size}/{val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = TransformerMT(
        vocab_size=dataset.sp.get_piece_size(), d_model=192, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=384, dropout=0.3,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": dataset.sp.get_piece_size(),
                    "model_config": {
                        "d_model": 192, "nhead": 4,
                        "num_encoder_layers": 2, "num_decoder_layers": 2,
                        "dim_feedforward": 384, "dropout": 0.3,
                    },
                },
                CHECKPOINT_PATH,
            )
            logger.info(f"New best val_loss={val_loss:.4f}, checkpoint saved to {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"No improvement for {patience} epochs, stopping early")
                break


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH,
                         help="Path to the jsonl training corpus, e.g. "
                              "parallel_cr_en_combined.jsonl to include Watkins data.")
    args = parser.parse_args()
    CORPUS_PATH = args.corpus
    train()

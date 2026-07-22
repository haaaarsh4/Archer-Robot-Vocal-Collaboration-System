import json
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class TranslationDataset(Dataset):

    def __init__(self, jsonl_path: Path, spm_model_path: str, max_len: int = 128):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.max_len = max_len
        self.pairs: List[Tuple[List[int], List[int]]] = []

        self.morfessor_model = None
        config_path = Path(spm_model_path.rsplit(".", 1)[0] + "_config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("use_morfessor"):
                from translation.cree_morfessor import get_model
                self.morfessor_model = get_model()

        dropped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                src = self._encode_cree(row["cree"])
                tgt = self._encode(row["english"])
                if 2 < len(src) <= max_len and 2 < len(tgt) <= max_len:
                    self.pairs.append((src, tgt))
                else:
                    dropped += 1

        if dropped:
            from loguru import logger
            logger.info(f"Dropped {dropped} pairs that were empty or exceeded max_len={max_len}")

    def _encode_cree(self, text: str) -> List[int]:
        if self.morfessor_model is not None:
            from translation.cree_morfessor import segment_cree_text
            text = segment_cree_text(text, self.morfessor_model)
        return self._encode(text)

    def _encode(self, text: str) -> List[int]:
        ids = self.sp.encode(text.strip(), out_type=int)
        return [BOS_ID] + ids + [EOS_ID]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        src, tgt = self.pairs[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_batch(batch):
    """Pads a batch of (src, tgt) tensors to the longest sequence in the batch."""
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)

    src_padded = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, :len(s)] = s
        tgt_padded[i, :len(t)] = t
    return src_padded, tgt_padded

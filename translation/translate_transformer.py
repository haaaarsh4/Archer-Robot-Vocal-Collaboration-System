import json
from pathlib import Path

import sentencepiece as spm
import torch
from loguru import logger

from translation.utils import tokenize
from translation.dataset import BOS_ID, EOS_ID, PAD_ID
from translation.model import TransformerMT

SPM_MODEL_PATH = "data/models/spm.model"
CHECKPOINT_PATH = "data/models/transformer_mt.pt"
PHRASE_TABLE_PATH = "data/models/phrase_table.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralTranslator:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH,
                 spm_model_path: str = SPM_MODEL_PATH,
                 phrase_table_path: str = PHRASE_TABLE_PATH):
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)

        self.morfessor_model = None
        config_path = Path(spm_model_path.rsplit(".", 1)[0] + "_config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("use_morfessor"):
                from translation.cree_morfessor import get_model
                self.morfessor_model = get_model()
                logger.info("Morfessor pre-segmentation enabled for Cree input (matches training config)")

        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model_config = ckpt.get("model_config", {})
        self.model = TransformerMT(vocab_size=ckpt["vocab_size"], **model_config).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.phrase_table = {}
        pt_path = Path(phrase_table_path)
        if pt_path.exists():
            with open(pt_path, "r", encoding="utf-8") as f:
                self.phrase_table = json.load(f)
            logger.info(f"Loaded phrase table ({len(self.phrase_table)} entries)")

    def translate(self, cree_text: str, beam_size: int = 5, max_len: int = 100, min_len: int = 2) -> str:
        normalized = " ".join(tokenize(cree_text))
        if normalized in self.phrase_table:
            return self.phrase_table[normalized]
        return self._beam_search(cree_text, beam_size, max_len, min_len)

    @torch.no_grad()
    def _beam_search(self, text: str, beam_size: int, max_len: int, min_len: int = 2) -> str:
        if self.morfessor_model is not None:
            from translation.cree_morfessor import segment_cree_text
            text = segment_cree_text(text, self.morfessor_model)
        src_ids = [BOS_ID] + self.sp.encode(text.strip(), out_type=int) + [EOS_ID]
        src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
        memory, src_pad_mask = self.model.encode(src)

        beams = [([BOS_ID], 0.0)]
        for step in range(max_len):
            all_finished = all(seq[-1] == EOS_ID for seq, _ in beams)
            if all_finished:
                break

            candidates = []
            for seq, score in beams:
                if seq[-1] == EOS_ID:
                    candidates.append((seq, score))
                    continue
                tgt = torch.tensor([seq], dtype=torch.long, device=DEVICE)
                logits = self.model.decode_step(tgt, memory, src_pad_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                if step < min_len:
                    log_probs[EOS_ID] = -1e9
                topk_log_probs, topk_ids = log_probs.topk(beam_size)
                for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    candidates.append((seq + [idx], score + lp))

            # length-normalized score so beam search doesn't just prefer
            # short outputs
            candidates.sort(key=lambda c: c[1] / len(c[0]), reverse=True)
            beams = candidates[:beam_size]

        finished = [b for b in beams if b[0][-1] == EOS_ID]
        pool = finished if finished else beams
        best_seq = max(pool, key=lambda c: c[1] / len(c[0]))[0]
        piece_ids = [i for i in best_seq if i not in (BOS_ID, EOS_ID, PAD_ID)]
        translation = self.sp.decode(piece_ids)

        if not translation.strip():
            fallback_seq = max(
                beams,
                key=lambda c: len([i for i in c[0] if i not in (BOS_ID, EOS_ID, PAD_ID)]),
            )[0]
            piece_ids = [i for i in fallback_seq if i not in (BOS_ID, EOS_ID, PAD_ID)]
            translation = self.sp.decode(piece_ids)

        return translation


if __name__ == "__main__":
    translator = NeuralTranslator()
    test_lines = [
        "nêhiyawêwin katawasisin",
        "kihci kiskinwahamâtowikamik",
        "kwâyas kanawêyimisok",
    ]
    for line in test_lines:
        print(f"{line!r} -> {translator.translate(line)!r}")

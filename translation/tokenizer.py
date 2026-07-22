import json
import re
from pathlib import Path

import sentencepiece as spm
from loguru import logger

CORPUS_PATH = Path("data/processed/parallel_cr_en.jsonl")
SPM_INPUT_PATH = Path("data/processed/spm_train_input.txt")
SPM_MODEL_PREFIX = "data/models/spm"
VOCAB_SIZE = 4000  

def write_spm_training_text(corpus_path: Path, out_path: Path, use_morfessor: bool = False) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    morfessor_model = None
    if use_morfessor:
        from translation.cree_morfessor import get_model, segment_cree_text
        morfessor_model = get_model()
        logger.info("Pre-segmenting Cree side with Morfessor before SentencePiece training")

    n = 0
    with open(corpus_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            cree_text = row["cree"].strip()
            if use_morfessor:
                from translation.cree_morfessor import segment_cree_text
                cree_text = segment_cree_text(cree_text, morfessor_model)
            fout.write(cree_text + "\n")
            fout.write(row["english"].strip() + "\n")
            n += 1
    logger.info(f"Wrote {n * 2} raw lines for SentencePiece training from {n} pairs")
    return n


def train_tokenizer(vocab_size: int = VOCAB_SIZE, use_morfessor: bool = False):
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"{CORPUS_PATH} not found. Run "
            f"'python -m translation.build_translation_corpus' first."
        )

    n_pairs = write_spm_training_text(CORPUS_PATH, SPM_INPUT_PATH, use_morfessor=use_morfessor)
    if n_pairs < 200:
        logger.warning(
            f"Only {n_pairs} sentence pairs found. A transformer trained on "
            f"this little data will likely underperform the phrase-table "
            f"lookup for anything outside the training set - consider "
            f"pulling in more of the PlainsCree subfolders (Bible, "
            f"Dictionary, ChildrenBooks, etc.) before training."
        )

    Path(SPM_MODEL_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    _train_with_fallback(vocab_size)

    config_path = Path(SPM_MODEL_PREFIX + "_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"use_morfessor": use_morfessor}, f)
    logger.info(f"Saved tokenizer config -> {config_path} (use_morfessor={use_morfessor})")


def _train_with_fallback(vocab_size: int, attempts_left: int = 5):
    try:
        spm.SentencePieceTrainer.train(
            input=str(SPM_INPUT_PATH),
            model_prefix=SPM_MODEL_PREFIX,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        logger.info(f"Trained SentencePiece model (vocab_size={vocab_size}) -> {SPM_MODEL_PREFIX}.model")
    except RuntimeError as e:
        match = re.search(r"<=\s*(\d+)", str(e))
        if match and attempts_left > 0:
            suggested = int(match.group(1))
            logger.warning(
                f"vocab_size={vocab_size} too large for this corpus, "
                f"retrying with SentencePiece's suggested max: {suggested}"
            )
            _train_with_fallback(suggested, attempts_left - 1)
        else:
            raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH,
                         help="Path to the jsonl corpus to train the tokenizer on. "
                              "Point this at parallel_cr_en_combined.jsonl to include "
                              "Watkins dictionary vocabulary.")
    parser.add_argument("--use-morfessor", action="store_true",
                         help="Pre-segment Cree into morphemes with Morfessor before "
                              "SentencePiece training. Requires cree_morfessor.py's model "
                              "already trained (python -m translation.cree_morfessor first).")
    args = parser.parse_args()
    CORPUS_PATH = args.corpus
    train_tokenizer(use_morfessor=args.use_morfessor)

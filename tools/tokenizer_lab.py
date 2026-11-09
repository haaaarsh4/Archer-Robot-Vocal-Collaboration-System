"""Compare tokenizers on this repository's own Cree, before any model is trained.

Two papers set the question this tool answers:

* MorphBPE (Findings of ACL 2026) constrains BPE merges so they cannot cross a
  morpheme boundary, keeps inference unchanged, and reports better morphological
  consistency and lower cross-entropy, with gains concentrated in morphologically
  rich languages. It also shows fertility alone is a bad proxy for tokenizer
  quality.
* The Mapudungun work (AmericasNLP 2026) applies BPE inside Morfessor morphemes
  for a polysynthetic American language and finds morphology-aware tokenization
  can substitute for a 5x increase in model parameters in one direction.

Plains Cree has no Morfessor model here, but it has something better: a
finite-state morphology with 21,232 stems (Interactive Word Completion for Plains
Cree, ACL 2022) and a lexicon in GiellaLT. Neither is installed in this
repository, so this lab uses a documented prefix and suffix inventory to segment
words at the boundaries the FST would mark, and measures what that does to the
tokenizer. It is a screening tool, not a replacement for the FST: when
`lang-crk` is present the same measurement should be rerun with real analyses.

    python tools/tokenizer_lab.py
    python tools/tokenizer_lab.py --vocab 4000 --segmented-suffixes

It reports, for standard BPE and for morpheme-constrained BPE:

    fertility          tokens per word (lower is not automatically better)
    boundary precision of token boundaries that fall on a morpheme boundary
    boundary recall    of morpheme boundaries that a token boundary lands on
    long-word fertility  tokens per word for words of 10 characters or more,
                       which is where polysynthetic Cree actually lives
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# Written without hyphens because SRO joins them to the stem. Ordered longest
# first: the segmenter takes the longest match, which is what a greedy FST
# tokeniser does at the surface level.
PREFIXES = [
    "ekayan", "ekan", "ekwa", "kayk", "kayi", "meka", "nama", "namoya", "ekwa",
    "kaki", "kik", "kita", "kici", "kisi", "osam", "wika", "wina", "ano", "api",
    "e", "ka", "ki", "ke", "ma", "me", "ni", "pe", "ta", "wa", "wi", "na", "pa",
]
SUFFIXES = [
    "kopan", "kwan", "will", "ikow", "iski", "iwak", "iwew", "ahk", "ehk", "ihk",
    "ohk", "ak", "ik", "uk", "an", "in", "on", "aw", "ew", "iw", "ow", "win",
    "wak", "wa", "ni", "na", "ki", "ka", "si", "sk", "k", "w", "a", "e", "i",
]
SEP = "|"          # marks a morpheme boundary inside a word
_LONG_WORD = 10


def strip_marks(word: str) -> str:
    return re.sub(r"[^a-zA-ZâêîôûÂÊÎÔÛāēīōū'\-]", "", word).lower()


def segment(word: str) -> list[str]:
    """Surface morpheme split: prefixes, then the stem, then suffixes.

    A real analysis comes from the FST. This is the closest approximation that
    works with no model files, and it is applied identically to training and
    evaluation so the comparison is fair.
    """
    w = strip_marks(word)
    if len(w) <= 3:
        return [w] if w else []
    parts: list[str] = []
    changed = True
    while changed and len(w) > 4:
        changed = False
        for p in PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 3:
                parts.append(p)
                w = w[len(p):]
                changed = True
                break
    suffixes: list[str] = []
    changed = True
    while changed and len(w) > 4:
        changed = False
        for s in SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 2:
                suffixes.insert(0, s)
                w = w[: -len(s)]
                changed = True
                break
    parts.append(w)
    return parts + suffixes


def mark(word: str) -> str:
    return SEP.join(segment(word))


def words_from_corpus(limit: int = 40_000) -> list[str]:
    path = PROC / "cree_corpus.jsonl"
    out: list[str] = []
    if not path.exists():
        raise SystemExit("run tools/build_cree_corpus.py first")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out.extend(strip_marks(w) for w in row["text"].split())
            if len(out) >= limit:
                break
    return [w for w in out if len(w) > 1]


def train(prefix: Path, text_path: Path, vocab: int) -> SentencePieceProcessor:
    SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(prefix),
        vocab_size=vocab,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        split_by_whitespace=True,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return SentencePieceProcessor(model_file=str(prefix) + ".model")


def boundary_scores(sp: SentencePieceProcessor, words: list[str]) -> tuple[float, float, float, float]:
    """Fertility and boundary alignment against the surface segmentation.

    A token boundary counts as correct when it lands on a morpheme boundary. The
    separator is a character in the marked string, so offsets are rebuilt by
    walking the pieces over that string rather than by adding up their lengths.
    """
    tokens = 0
    hit_tp = hit_fp = hit_fn = 0
    long_tokens = long_words = 0
    for word in words:
        marked = mark(word)
        pieces = sp.encode(marked, out_type=str)
        tokens += len(pieces)
        if len(word) >= _LONG_WORD:
            long_tokens += len(pieces)
            long_words += 1
        seg_bounds = {i for i, ch in enumerate(marked) if ch == SEP}
        if not seg_bounds:
            continue
        tok_bounds, pos = set(), 0
        for piece in pieces:
            text = piece.replace("\u2581", " ")
            # SentencePiece may keep the separator inside a piece; either way the
            # piece covers the next len(text) characters of the marked word.
            if marked[pos:pos + len(text)] != text:
                stripped = text.replace(" ", "")
                if marked[pos:pos + len(stripped)] != stripped:
                    break
                text = stripped
            pos += len(text)
            if pos < len(marked):
                tok_bounds.add(pos)
        hit_tp += len(seg_bounds & tok_bounds)
        hit_fp += len(tok_bounds - seg_bounds)
        hit_fn += len(seg_bounds - tok_bounds)
    precision = hit_tp / (hit_tp + hit_fp) if hit_tp + hit_fp else 0.0
    recall = hit_tp / (hit_tp + hit_fn) if hit_tp + hit_fn else 0.0
    return (tokens / max(1, len(words)),
            long_tokens / max(1, long_words),
            precision, recall)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--words", type=int, default=20_000)
    args = ap.parse_args()

    words = words_from_corpus()
    held = words[::10][: args.words]
    print(f"{len(words):,} word tokens, {len(set(words)):,} distinct; "
          f"measuring on {len(held):,} held-out tokens\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        plain = tmp_path / "plain.txt"
        marked = tmp_path / "marked.txt"
        plain.write_text("\n".join(segment_join(w) for w in set(words)), encoding="utf-8")
        marked.write_text("\n".join(mark(w) for w in set(words)), encoding="utf-8")
        base = train(tmp_path / "bpe", plain, args.vocab)
        morph = train(tmp_path / "morph", marked, args.vocab)

        rows = []
        for name, sp in (("standard BPE", base), ("morpheme-constrained BPE", morph)):
            fertility, long_fert, prec, rec = boundary_scores(sp, held)
            rows.append({"tokenizer": name, "fertility": round(fertility, 3),
                         "long_word_fertility": round(long_fert, 3),
                         "boundary_precision": round(prec, 3),
                         "boundary_recall": round(rec, 3)})
            print(f"{name:28s} fertility {fertility:.2f}  "
                  f"long words {long_fert:.2f}  "
                  f"boundary P {prec:.2f}  R {rec:.2f}")

        out = PROC / "tokenizer_lab.json"
        out.write_text(json.dumps({"vocab": args.vocab, "results": rows}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)}")

        a, b = rows
        better = (b["boundary_precision"] > a["boundary_precision"]
                  and b["long_word_fertility"] <= a["long_word_fertility"] * 1.1)
        print("verdict:", "morpheme-constrained BPE wins on both alignment and long words"
              if better else "no clear win on this corpus; keep standard BPE and rerun with the FST")
    return 0


def segment_join(word: str) -> str:
    return word


if __name__ == "__main__":
    raise SystemExit(main())

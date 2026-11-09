"""Build one Cree corpus out of every source in the repo, and say how big it is.

The training pipeline can only be as fluent as the text it is given. This walks
every Cree source on disk, normalises the text, drops duplicates and junk,
writes ``data/processed/cree_corpus.jsonl`` and prints the count. It is the
number to look at before blaming the model for bad Cree.

    python tools/build_cree_corpus.py
    python tools/build_cree_corpus.py --min-tokens 20000000   # gate a run

What the repository actually contains, checked on every run:

    data/processed/parallel_cr_en_combined.jsonl   aligned pairs, mixed quality
    data/processed/parallel_cr_en.jsonl            aligned pairs
    data/processed/watkins_dictionary_pairs.jsonl  1865 dictionary pairs
    data/processed/watkins_entries.jsonl           1865 entries, many forms each
    data/PlainsCree/Dictionary/dictionary_cr.txt   word list
    data/Parse (Watkins 1865)/raw_ocr_text.txt     1865 OCR page text
    data/PlainsCree/Twitter/Twitter_cr.txt         a handful of lines

Two quality problems are repaired or reported rather than hidden:

* Some rows in the pair files have their columns swapped (``cree: "thus"``,
  ``english: "kwema"``). If the Cree side looks English and the English side
  looks Cree, the row is swapped back.
* Dictionary rows carry sense numbers (``niyânan 2``, ``okay 6``). Those are not
  sentences and are dropped from the corpus, and counted separately so the
  report says how much of the "parallel data" is really a dictionary.

Token count is estimated at 3.2 characters per token, which is about what the
project tokenizer produces on Cree. It is an estimate and is labelled as one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "cree_corpus.jsonl"
PAIRS_OUT = PROC / "parallel_clean.jsonl"

_CREE_LETTERS = set("âêîôûāēīōūȃȇȋȏȗᐁᐃᐅᐊᐯᑌᑭᑐᑕᒣᓀᓯᓴᔦᔨᔪᔭᕁᖬ")
_CREE_WORDS = re.compile(r"\b(?:ê|kâ|kî|mâ|nî|pê|tâ|wâ|ô|â)[a-z]{2,}\b")
_WORD = re.compile(r"[A-Za-z\u00c0-\u024f\u1400-\u167f'’-]+")
_DIGITS = re.compile(r"\d")
_URL = re.compile(r"https?://|www\.", re.I)
_ENGLISH_HINT = re.compile(
    r"\b(?:the|and|of|to|in|is|are|was|were|with|that|this|these|those|his|her|"
    r"they|you|not|for|from|by|at|on|as|or|but|if|a|an|he|she|it|there|what|how)\b",
    re.I,
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()


def looks_cree(text: str, single_word_ok: bool = False) -> bool:
    """Cree-looking text. A dictionary form is one word, a sentence is not.

    The 1865 forms carry no diacritics ("wapinio", "nukutao"), so requiring a
    macron would throw away most of the historical dictionary. They are accepted
    as words when the source is a word list, and rejected as sentences.
    """
    if _DIGITS.search(text) or _URL.search(text):
        return False
    words = _WORD.findall(text)
    if single_word_ok and len(words) == 1:
        return not looks_english(text)
    if len(words) < 2:
        return False
    if any(ch in _CREE_LETTERS for ch in text):
        return True
    return bool(_CREE_WORDS.search(text.lower()))


def looks_english(text: str) -> bool:
    return bool(_ENGLISH_HINT.search(text)) and not any(ch in _CREE_LETTERS for ch in text)


def key_of(text: str) -> str:
    return re.sub(r"[^a-z\u00c0-\u024f\u1400-\u167f ]+", "", text.lower()).strip()


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _core(text: str) -> str:
    """The text without parenthetical qualifiers, which are often English.

    ``(from lying) wuniskow`` is a Cree form with an English note in front of
    it. Judging the row on the note alone would call it English and swap a row
    that was already the right way round.
    """
    return normalise(re.sub(r"\([^)]*\)", " ", text))


def fixed_pair(row: dict) -> tuple[str, str] | None:
    """The (cree, english) of a pair row, with swapped columns put back."""
    cree = (row.get("cree") or "").strip()
    english = (row.get("english") or "").strip()
    if not cree or not english:
        return None
    # A swapped dictionary row holds a multi-word English gloss in ``cree`` and
    # one Cree form in ``english``. The test is on the gloss, not on the whole
    # cell, so a Cree form carrying an English parenthetical is left alone.
    cree_core, english_core = _core(cree), _core(english)
    if (len(cree_core.split()) >= 2 and looks_english(cree_core)
            and len(english_core.split()) == 1):
        cree, english = english, cree
        cree_core = _core(cree)
    if looks_english(cree_core):
        return None
    return normalise(cree), normalise(english)


def work_dir() -> Path | None:
    """Where a training run puts what it downloaded.

    The sources named in the project's own table (FinePDFs, FineWeb-2, the
    Internet Archive books, EdTeKLA, GlotCC, Tatoeba, GiellaLT) are fetched by
    ``training/cree_llm/cells/03_corpus.py`` at run time, into ``ROOT/raw``. ROOT
    is ``/content/cree_llm`` on Colab, the Drive folder on a long run, or
    ``$CREE_WORK`` / ``cree_llm_work`` locally. They are not committed to the
    repository, which is why the checkout on its own holds only the small
    extracts in ``data/processed``.
    """
    candidates = [
        Path(os.environ["CREE_WORK"]).expanduser() if os.environ.get("CREE_WORK") else None,
        ROOT / "cree_llm_work",
        Path("/content/cree_llm"),
        Path("/content/drive/MyDrive/archer_cree_llm"),
    ]
    for path in candidates:
        if path and (path / "raw").is_dir():
            return path
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="exit 1 if the corpus estimates fewer tokens than this")
    ap.add_argument("--min-sentences", type=int, default=0)
    args = ap.parse_args()

    seen: set[str] = set()
    kept: list[dict] = []
    per_source: dict[str, int] = {}

    dropped = 0
    swapped = 0

    def add(text: str, source: str, single_word_ok: bool = False) -> None:
        nonlocal dropped
        text = normalise(text)
        if not (3 <= len(text) <= 400) or not looks_cree(text, single_word_ok):
            dropped += 1
            return
        k = key_of(text)
        if not k or k in seen:
            dropped += 1
            return
        seen.add(k)
        kept.append({"text": text, "source": source})

    # --- aligned pairs, with the columns checked -----------------------------
    clean_pairs: list[dict] = []
    pair_rows = dict_digits = 0
    for name in ("parallel_cr_en_combined.jsonl", "parallel_cr_en.jsonl"):
        path = PROC / name
        if not path.exists():
            print(f"  missing, skipped: {path.relative_to(ROOT)}")
            continue
        before = len(kept)
        for row in jsonl_rows(path):
            pair_rows += 1
            pair = fixed_pair(row)
            if pair is None:
                dropped += 1
                continue
            cree, english = pair
            if _DIGITS.search(cree) or _DIGITS.search(english):
                dict_digits += 1
                continue
            if looks_english(cree) and looks_cree(english, single_word_ok=True):
                swapped += 1
            if len(cree.split()) < 2:
                dropped += 1
                continue
            clean_pairs.append({"cree": cree, "english": english,
                                "source": row.get("source", name)})
            add(cree, name)
        per_source[name] = len(kept) - before
        print(f"  {name:58s} {per_source[name]:>8,} kept")

    # --- 1865 dictionary entries, every form --------------------------------
    path = PROC / "watkins_entries.jsonl"
    if path.exists():
        before = len(kept)
        for row in jsonl_rows(path):
            for form in row.get("cree_forms") or []:
                if isinstance(form, str):
                    add(form, path.name, single_word_ok=True)
        per_source[path.name] = len(kept) - before
        print(f"  {path.name:58s} {per_source[path.name]:>8,} kept")

    for name, key in (("watkins_dictionary_pairs.jsonl", "cree"),
                      ("data/PlainsCree/Dictionary/dictionary_cr.txt", None),
                      ("data/Parse (Watkins 1865)/raw_ocr_text.txt", None),
                      ("data/PlainsCree/Twitter/Twitter_cr.txt", None)):
        path = ROOT / name if "/" in name else PROC / name
        if not path.exists():
            print(f"  missing, skipped: {path.relative_to(ROOT)}")
            continue
        before = len(kept)
        word_list = "dictionary" in path.name.lower() or "watkins" in path.name.lower()
        if path.suffix == ".jsonl":
            for row in jsonl_rows(path):
                add((row.get(key) or ""), path.name, single_word_ok=word_list)
        else:
            for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
                add(line, path.name, single_word_ok=word_list)
        per_source[path.name] = len(kept) - before
        print(f"  {path.relative_to(ROOT)!s:58s} {per_source[path.name]:>8,} kept")

    # --- the downloaded corpus, if a training run has already fetched it -----
    work = work_dir()
    if work is None:
        print("\n  downloaded corpus: not on this machine")
        print("    FinePDFs / FineWeb-2 / Internet Archive / EdTeKLA / GlotCC / Tatoeba / GiellaLT")
        print("    are fetched by training/cree_llm/cells/03_corpus.py into ROOT/raw at run time.")
        print("    Point CREE_WORK at that folder (or sync it here) and this count includes them.")
    else:
        for name, key, single in (("raw/docs.jsonl", "text", False),
                                  ("raw/pairs.jsonl", "cree", False)):
            path = work / name
            if not path.exists():
                print(f"  work dir {work} has no {name}")
                continue
            before = len(kept)
            for row in jsonl_rows(path):
                add((row.get(key) or ""), path.stem, single_word_ok=single)
            per_source[name] = len(kept) - before
            print(f"  {work.name + '/' + name:58s} {per_source[name]:>8,} kept")

    chars = sum(len(r["text"]) for r in kept)
    words = sum(len(r["text"].split()) for r in kept)
    tokens = int(chars / 3.2)
    PROC.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with PAIRS_OUT.open("w", encoding="utf-8") as fh:
        for row in clean_pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {OUT.relative_to(ROOT)} and {PAIRS_OUT.relative_to(ROOT)}")
    print(f"  sentences {len(kept):,}")
    print(f"  words     {words:,}")
    print(f"  chars     {chars:,}")
    print(f"  tokens    ~{tokens:,} (estimate at 3.2 chars/token)")
    print(f"  dropped   {dropped:,} (duplicates, English, OCR noise, too short or long)")
    print(f"  pair rows read {pair_rows:,}: {dict_digits:,} were dictionary entries with "
          f"sense numbers, {len(clean_pairs):,} clean pairs, {swapped:,} swapped back")

    if tokens < 2_000_000:
        verdict = "thin: enough to teach spelling and common sentences, not enough for fluency"
    elif tokens < 10_000_000:
        verdict = "workable: covers frequent grammar, will still fail on uncommon wording"
    elif tokens < 50_000_000:
        verdict = "solid: the range where a small model starts writing its own sentences"
    else:
        verdict = "large for this project: quality, not volume, is the limit now"
    print(f"  verdict   {verdict}")

    if args.min_tokens and tokens < args.min_tokens:
        print(f"\nFAILED: {tokens:,} estimated tokens is under the required {args.min_tokens:,}")
        return 1
    if args.min_sentences and len(kept) < args.min_sentences:
        print(f"\nFAILED: {len(kept):,} sentences is under the required {args.min_sentences:,}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

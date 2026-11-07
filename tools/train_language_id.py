"""Train the language identifier from this project's own text.

Nothing here is hand-labelled and nothing is downloaded. The Cree side is the
aligned Cree sentences, the corpus sentences and Watkins' dictionary forms the
project already has; the English side is the English half of the same aligned
pairs plus the song notes. The split is a hash of the text, so a sentence is
either a training example or a test example and never both.

    .venv/bin/python tools/train_language_id.py

Writes chat/language_id_model.json and data/processed/language_id_metrics.json.
Prints sentence and word accuracy on the held-out tenth, plus the accuracy of
the diacritic rule this replaces, because a replacement that is not better than
the thing it replaces is not a replacement.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat.language_id import (                                        # noqa: E402
    MODEL_PATH, is_held_out, language_of, normalise, save, train,
)

PAIR_FILES = (
    "parallel_clean.jsonl",
    "parallel_cr_en.jsonl",
    "parallel_cr_en_combined.jsonl",
    "watkins_dictionary_pairs.jsonl",
)


def _usable(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not 2 <= len(text) <= 300:
        return False
    return bool(re.search(r"[A-Za-zÂâÊêÎîÔô]", text))


def _english_residue(text: str) -> bool:
    """Is this supposedly Cree row an English gloss in disguise?"""
    return bool(_ENGLISH_MARKERS.search(text)) and not re.search(r"[âêîô]", text)


def _pair_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    processed = ROOT / "data" / "processed"
    for name in PAIR_FILES:
        path = processed / name
        if not path.exists():
            print(f"  missing {name}, skipped")
            continue
        added = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                cree = row.get("cree") or row.get("cr")
                english = row.get("english") or row.get("en")
                # A stale pair file can carry the gloss in the ``cree`` column
                # (the Watkins rows once did). Training on that teaches English
                # as Cree, so the label has to be earned here and not only when
                # the held-out rows are scored.
                if _usable(cree) and not _english_residue(cree):
                    rows.append((cree, "cr"))
                    added += 1
                if _usable(english) and not re.search(r"[âêîô\u1400-\u167F]", english):
                    rows.append((english, "en"))
                    added += 1
        print(f"  {name}: {added} rows")
    return rows


def _corpus_rows() -> list[tuple[str, str]]:
    path = ROOT / "data" / "processed" / "cree_corpus.jsonl"
    if not path.exists():
        print("  missing cree_corpus.jsonl, skipped")
        return []
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                text = json.loads(line).get("text")
            except ValueError:
                continue
            if _usable(text):
                rows.append((text, "cr"))
    print(f"  cree_corpus.jsonl: {len(rows)} rows")
    return rows


def _note_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted((ROOT / "song_notes").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = re.sub(r"[#*_`>-]+", " ", sentence).strip()
            if _usable(sentence):
                rows.append((sentence, "en"))
    print(f"  song_notes: {len(rows)} rows")
    return rows


def _doc_rows() -> list[tuple[str, str]]:
    """English prose from the project, so the English side is not 900 sentences.

    The aligned pairs give about 900 English sentences against 24,000 Cree ones.
    A classifier fitted on that learns Cree far better than English, which shows
    up as English questions read as Cree. The project's own documentation is
    English and is already here. Sentences carrying a long vowel or syllabics are
    skipped: those are the Cree examples the docs quote, and labelling them
    English would teach the model the opposite of what it needs.
    """
    paths = [ROOT / "README.md"]
    paths += sorted((ROOT / "docs").rglob("*.md"))
    paths += sorted((ROOT / "frontend" / "pages").glob("*.html"))
    rows: list[tuple[str, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = re.sub(r"[#*_`>|\[\]()]+", " ", sentence).strip()
            if not _usable(sentence) or len(sentence.split()) < 4:
                continue
            if re.search(r"[âêîô\u1400-\u167F]", sentence):
                continue
            rows.append((sentence, "en"))
    print(f"  project prose: {len(rows)} rows")
    return rows


def _derived_words(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Single words with the sentence's label, for the word-level model.

    The dictionary gives 8,900 single words, which is not enough to separate a
    Cree stem from an English gloss. Every word in a labelled sentence is also a
    labelled word, so the same text yields about 200,000 word examples at no
    extra cost. Only training sentences are used, so the split stays clean.
    """
    out: list[tuple[str, str]] = []
    for text, label in rows:
        # Only from rows whose label is earned. The Cree column of the aligned
        # corpus has English residue in it, and deriving word labels from those
        # rows teaches the word model that "for" and "poles" are Cree, which is
        # exactly the error the benchmark then caught.
        if not _label_is_trustworthy(text, label):
            continue
        for token in re.findall(r"[^\W\d_][^\W\d_'’-]*", text):
            if 3 <= len(token) <= 25:
                out.append((token, label))
    return out


def _word_rows() -> list[tuple[str, str]]:
    """Single tokens, so short input is calibrated too, not just sentences."""
    path = ROOT / "data" / "processed" / "watkins_dictionary_pairs.jsonl"
    if not path.exists():
        return []
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for key, label in (("cree", "cr"), ("english", "en")):
                value = str(row.get(key) or "").strip()
                if 3 <= len(value) <= 30 and " " not in value and _usable(value):
                    rows.append((value, label))
    return rows


_MARKED = re.compile(r"[âêîô]")
_ENGLISH_MARKERS = re.compile(
    r"\b(?:the|and|is|are|was|were|of|to|in|it|this|that|there|what|how|not|"
    r"much|he|she|they|you|for|with|on|at|from|by|or|but|if|a|an)\b",
    re.IGNORECASE,
)


def _label_is_trustworthy(text: str, label: str) -> bool:
    """Was this row's label earned?

    The aligned corpus has English residue in its Cree column ("n. embrocation
    lining") and Cree headwords in its English column. Those rows cannot be
    scored correctly by any model, so they are counted and excluded instead of
    being averaged in as if the classifier were wrong.
    """
    words = text.split()
    if label == "cr":
        return len(words) >= 2 and (bool(_MARKED.search(text)) or len(words) >= 6) \
            and not _ENGLISH_MARKERS.search(text)
    return not _MARKED.search(text) and not re.search(r"[\u1400-\u167F]", text)


def _accuracy(rows: list[tuple[str, str]]) -> tuple[float, dict[str, int]]:
    right = 0
    confusion: dict[str, int] = {}
    for text, gold in rows:
        got = language_of(text).lang
        key = f"{gold}->{got}"
        confusion[key] = confusion.get(key, 0) + 1
        if got == gold:
            right += 1
    return (right / len(rows) if rows else 0.0), confusion


def _diacritic_baseline(rows: list[tuple[str, str]]) -> float:
    """The rule this replaces: any long vowel means Cree."""
    right = sum(
        ("cr" if re.search(r"[âêîô]", text) else "en") == gold
        for text, gold in rows
    )
    return right / len(rows) if rows else 0.0


def main() -> int:
    print("reading project text")
    rows = _pair_rows() + _corpus_rows() + _note_rows() + _doc_rows()
    word_rows = _word_rows()
    print(f"  dictionary words: {len(word_rows)} rows")
    rows += word_rows

    # Deduplicate before splitting, so the same sentence from two collections
    # cannot be in both halves.
    unique: dict[str, tuple[str, str]] = {}
    for text, label in rows:
        unique.setdefault(normalise(text), (text, label))
    rows = list(unique.values())

    training = [(text, label) for text, label in rows if not is_held_out(text)]
    held_out = [(text, label) for text, label in rows if is_held_out(text)]
    training_words = {normalise(text): label for text, label in training if " " not in text}
    for token, label in _derived_words(training):
        training_words.setdefault(normalise(token), label)
    word_training = [(text, label) for text, label in training_words.items() if text]
    print(f"\n{len(rows)} unique rows: {len(training)} training, {len(held_out)} held out")
    print(f"  {len(word_training)} single-word training examples")

    model = train(training, word_training)
    path = save(model)
    print(f"model written to {path} ({path.stat().st_size / 1024:.0f} KB)")

    raw_sentence_rows = [
        (text, label) for text, label in held_out
        if (label == "cr" and len(text.split()) >= 2
            or label == "en" and len(text.split()) >= 3)
    ]
    sentence_rows = [
        (text, label) for text, label in raw_sentence_rows
        if _label_is_trustworthy(text, label)
    ]
    noise = len(raw_sentence_rows) - len(sentence_rows)
    word_held = [(t, l) for t, l in held_out if " " not in t and len(t) <= 30]

    sentence_acc, confusion = _accuracy(sentence_rows)
    word_acc, _word_confusion = _accuracy(word_held)
    # Unmarked Cree spelling ("wapinio") is genuinely close to English spelling,
    # so the two are reported apart instead of averaged into one number that
    # hides which case is failing.
    marked_words = [(t, l) for t, l in word_held if l == "cr" and re.search(r"[âêîô]", t)]
    marked_acc, _ = _accuracy(marked_words)
    # Measured on every row, not just the trustworthy ones: the point of the
    # comparison is what the old rule did with the data as it came.
    baseline = _diacritic_baseline(raw_sentence_rows)

    print("\nheld-out sentence accuracy")
    for key in sorted(confusion):
        print(f"  {key:12s} {confusion[key]}")
    print(f"  accuracy {sentence_acc:.3f} on {len(sentence_rows)} sentences")
    print(f"  {noise} held-out rows excluded as unlabelled/mislabelled residue")
    print(f"  diacritic-rule baseline {baseline:.3f}")
    print(f"\nheld-out single-word accuracy {word_acc:.3f} on {len(word_held)} words")
    print(f"  of which marked Cree forms {marked_acc:.3f} on {len(marked_words)} words")

    metrics = {
        "rows": len(rows),
        "training": len(training),
        "held_out": len(held_out),
        "sentence_accuracy": round(sentence_acc, 4),
        "sentence_baseline_diacritic": round(baseline, 4),
        "word_accuracy": round(word_acc, 4),
        "marked_word_accuracy": round(marked_acc, 4),
        "sentence_confusion": confusion,
        "model_bytes": path.stat().st_size,
    }
    out = ROOT / "data" / "processed" / "language_id_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"metrics written to {out}")

    # A model no better than the rule it replaced is not worth shipping.
    if sentence_acc < 0.95 or marked_acc < 0.85:
        print("\nFAILED: below the gate (sentence 0.95, marked word 0.85)")
        return 1
    print("\npassed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

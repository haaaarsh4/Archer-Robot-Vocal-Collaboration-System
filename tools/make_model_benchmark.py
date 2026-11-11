#!/usr/bin/env python3
"""Build the 1,000-line model benchmark.

The rows come from the cleaned corpus, not from a hand-written list of chatbot
answers. Translation prompts use the held-out split when it exists. Language
identification uses the labelled columns of the same corpus. Identity and echo
probes are a small fixed probe set because those are behaviours, not corpus
sentences, and they still carry an explicit expected result.

Every row says how it must be scored:

    chrf       translation, scored against the reference with chrF++
    language   the reply must name the language the row is labelled with
    exact      an answer that must match exactly
    contains   identity or explanation, must contain the expected fact
    not_echo   the reply must not be the prompt repeated back

Run it after the clean stage on the server:

    python tools/make_model_benchmark.py --work /data/cree_llm \
        --out data/eval/model_benchmark_1000.jsonl --require-held-out

The fallback to the repository corpus is for a dry run before training. It marks
`held_out` false, so nobody can mistake it for a clean benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text.strip(" .") if text else ""


def _sentence(text: str) -> bool:
    return 2 <= len(_clean(text).split()) <= 40


def _usable(text: str) -> bool:
    """A sentence, not a concordance entry or a verse number.

    The Cree Bible sources were split on verse markers, so "boanerges 1 caesar 3"
    and "herod 11 heroda 2" survived cleaning as if they were sentences. Each is
    a name plus a sense number. A language row built from one asks the model to
    call Hebrew names Plains Cree, and a translation row built from one asks it
    to translate a verse number, so both measure nothing. A digit is the tell,
    and no real sentence here starts with one.
    """
    clean = _clean(text)
    if not clean or not clean[0].isalpha():
        return False
    if re.search(r"\d", clean):
        return False
    return _sentence(clean)


# English function words are a closed class: Cree does not use "the", "is" or
# "why", and no Plains Cree sentence reaches a third of its words from this list.
# A row whose Cree column holds English is not a Cree sample, and asking a model
# to identify one taught nothing: the test was wrong, not the answer.
_ENGLISH_FUNCTION = frozenset(
    "a an and are as at be been being but by can could did do does for from had "
    "has have he her here him his how i if in into is it its me more most my no "
    "not of on or our out she should so some than that the their them then there "
    "these they this those to too up us was we were what when where which who "
    "whom why will with would you your".split()
)


def _english_share(text: str) -> float:
    words = re.findall(r"[^\W\d_]+", text.lower())
    if not words:
        return 0.0
    return sum(word in _ENGLISH_FUNCTION for word in words) / len(words)


def _looks_cree(text: str) -> bool:
    """Is this the Cree side of a pair, or English sitting in that column?

    A Cree diacritic or a syllabics character settles it positively. Otherwise
    the closed class of English function words decides, which keeps unmarked Cree
    like "maskisin" and drops "why are you going home ?".
    """
    clean = _clean(text)
    if not clean:
        return False
    if re.search(r"[\u00e2\u00ea\u00ee\u00f4\u0101\u0113\u012b\u014d]", clean, re.I):
        return True
    if re.search(r"[\u1400-\u167f]", clean):
        return True
    return _english_share(clean) < 0.34


def _usable_pair(row: dict) -> bool:
    """Both sides are sentences, the Cree side is Cree, and it is not Watkins."""
    return (
        _usable(row.get("cree"))
        and _usable(row.get("english"))
        and _looks_cree(row.get("cree"))
        and row.get("source") != "watkins1865"
    )


def _choose_processed(work: Path | None) -> tuple[Path, bool]:
    if work is not None:
        processed = work / "processed"
        if (processed / "parallel_test.jsonl").exists() or (processed / "parallel_cr_en.jsonl").exists():
            return processed, True
    repo = ROOT / "data" / "processed"
    return repo, False


def _pairs(processed: Path, require_held_out: bool) -> tuple[list[dict], bool]:
    test = processed / "parallel_test.jsonl"
    if test.exists():
        rows = [r for r in _load(test) if _usable_pair(r)]
        if rows:
            return rows, True
    if require_held_out:
        raise SystemExit(f"{test} is missing or empty; run the clean stage first")
    fallback = processed / "parallel_cr_en.jsonl"
    rows = [r for r in _load(fallback) if _usable_pair(r)]
    return rows, False


def _sense_free(text: str) -> str:
    """Drop the sense number a dictionary entry carries: "hard 1" is "hard"."""
    return re.sub(r"\s+", " ", re.sub(r"\b\d+\b", " ", _clean(text))).strip(" .,")


def _gloss_pair(row: dict) -> dict | None:
    """One dictionary word and its English gloss, or None.

    A dictionary entry is a word, not a sentence, so the two-word rule that keeps
    concordance fragments out of the sentence pools cannot apply here: it would
    throw away every entry like "maskisin" -> "shoe", which is the whole pool.
    What marks a fragment is the sense number, and that is stripped rather than
    used to reject the row.
    """
    cree, english = _clean(row.get("cree")), _clean(row.get("english"))
    if not cree or not english or not cree[0].isalpha() or not english[0].isalpha():
        return None
    if len(cree.split()) > 3 or len(english.split()) > 4:
        return None
    cree, english = _sense_free(cree), _sense_free(english)
    if not cree or not english:
        return None
    # "jspimik. (more) awusim4" is a fragment of a dictionary entry, not an
    # entry: a digit stuck to a letter survives the sense strip, and a bracketed
    # editorial note is the dictionary talking about the word, not the word.
    if re.search(r"[\d()\[\]]", cree + english):
        return None
    return {"cree": cree, "english": english}


def _words(processed: Path, pairs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for row in _load(processed / "watkins_dictionary_pairs.jsonl"):
        pair = _gloss_pair(row)
        if pair and pair["cree"] not in seen:
            seen.add(pair["cree"])
            rows.append(pair)
    if rows:
        return rows
    for row in pairs:
        pair = _gloss_pair(row)
        if pair and pair["cree"] not in seen:
            seen.add(pair["cree"])
            rows.append(pair)
    return rows


IDENTITY = [
    ("Who are you?", "Archer's assistant"),
    ("What is your purpose?", "answer from Archer's song notes"),
    ("Are you Archer?", "No"),
    ("What model are you based on?", "local"),
    ("Do you speak for Cree people?", "No"),
    ("What can you do?", "translate"),
    ("What language do you work in?", "Plains Cree"),
    ("Where do your song answers come from?", "notes"),
    ("Are you a person?", "No"),
    ("What is this project?", "vocal collaboration"),
    ("Who is this system for?", "Archer"),
    ("How accurate is your Cree?", "fluent"),
    ("What should you do when the notes do not say?", "say the notes do not say"),
    ("Do you invent facts?", "No"),
    ("What is your name?", "Archer's assistant"),
    ("Can you translate English into Cree?", "Yes"),
    ("Can you explain a Cree word?", "Yes"),
    ("Do you answer Cree in Cree by default?", "No"),
    ("When do you answer in Cree?", "when explicitly asked"),
    ("What are your limits?", "not a person"),
    ("Do you have a body?", "No"),
    ("Are you the robot?", "No"),
    ("What is Archer?", "project"),
    ("What do you do with corrections?", "learn"),
    ("How should you answer a greeting?", "briefly"),
]

ECHO = [
    "hi", "hey", "hello", "hi there", "howdy", "good morning",
    "yo", "hey there", "morning", "hello there", "how's it going?", "what's up?",
    "good afternoon", "good evening", "greetings", "hiya", "hey hey",
    "hello friend", "morning friend", "hey you", "hi again", "hello again",
    "good day", "how are you?", "what's new?",
]


def _id(category: str, index: int) -> str:
    return f"{category}-{index:04d}"


def _translation_row(category: str, index: int, row: dict, held_out: bool) -> dict:
    if category == "translate_en_cr":
        prompt = ("<|translate|><|english|><|sro|>\n"
                  f"Translate into Plains Cree (SRO):\n{row['english']}")
        expected, language = row["cree"], "cr"
    else:
        prompt = ("<|translate|><|cree|><|english|>\n"
                  f"Translate into English:\n{row['cree']}")
        expected, language = row["english"], "en"
    return {
        "id": _id(category, index), "category": category, "prompt": prompt,
        "expected": expected, "metric": "chrf", "answer_language": language,
        "held_out": held_out, "source": "parallel_test" if held_out else "repository",
    }


def _language_row(category: str, index: int, row: dict, held_out: bool) -> dict:
    if category == "language_id_cree":
        text, expected = row["cree"], "Plains Cree."
    elif category == "language_id_english":
        text, expected = row["english"], "English."
    else:
        text, expected = f"{row['cree']} {row['english']}", "Mixed."
    return {
        "id": _id(category, index),
        "category": category,
        "prompt": f"<|song|>\nWhich language is this?\n{text}",
        "expected": expected,
        "metric": "language",
        "answer_language": "en",
        "held_out": held_out,
        "source": "parallel_test" if held_out else "repository",
    }


def _word_row(index: int, row: dict, held_out: bool) -> dict:
    cree, english = _clean(row["cree"]), _clean(row["english"])
    return {
        "id": _id("word_explain", index),
        "category": "word_explain",
        "prompt": f"<|song|>\nWhat does {cree} mean in English?",
        "expected": f"{cree} means {english}.",
        "metric": "contains",
        "must_contain": english,
        "answer_language": "en",
        "held_out": held_out,
        "source": "dictionary",
    }


def _identity_row(index: int, question: str, fact: str) -> dict:
    return {
        "id": _id("identity", index),
        "category": "identity",
        "prompt": f"<|song|>\n{question}",
        "expected": fact,
        "metric": "contains",
        "must_contain": fact,
        "answer_language": "en",
        "held_out": True,
        "source": "probe",
    }


def _echo_row(index: int, prompt: str) -> dict:
    return {
        "id": _id("echo", index),
        "category": "echo",
        "prompt": f"<|song|>\n{prompt}",
        "expected": prompt,
        "metric": "not_echo",
        "answer_language": "en",
        "held_out": True,
        "source": "probe",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, help="training work folder, for example /data/cree_llm")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "model_benchmark_1000.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--require-held-out", action="store_true")
    args = ap.parse_args()

    processed, from_work = _choose_processed(args.work)
    pairs, held_out = _pairs(processed, args.require_held_out)
    words = _words(processed, pairs)
    if not pairs:
        raise SystemExit(f"no usable sentence pairs in {processed}")
    if not words:
        raise SystemExit(f"no usable dictionary rows in {processed}")
    if not held_out:
        print("warning: no held-out split found; using the repository corpus and marking rows held_out=false")
    print(f"pairs {len(pairs)}, dictionary rows {len(words)}, held_out={held_out}, processed={processed}")

    # Fixed target counts, so the file is exactly 1,000 rows and every capability
    # has enough examples to produce a rate rather than a single anecdote.
    targets = {
        "translate_en_cr": 200,
        "translate_cr_en": 200,
        "language_id_cree": 150,
        "language_id_english": 150,
        "language_id_mixed": 100,
        "word_explain": 150,
        "identity": 25,
        "echo": 25,
    }
    scale = max(args.n / 1000, 1e-9)
    counts = {k: int(round(v * scale)) for k, v in targets.items()}
    while sum(counts.values()) < args.n:
        counts["translate_en_cr"] += 1
    while sum(counts.values()) > args.n:
        counts["identity"] = max(0, counts["identity"] - 1)

    rows: list[dict] = []
    seen_prompts: set[str] = set()

    def _add(row: dict) -> bool:
        if row["prompt"] in seen_prompts:
            return False
        seen_prompts.add(row["prompt"])
        rows.append(row)
        return True

    for category, count in list(counts.items()):
        pool_size = {
            "word_explain": len(words),
            "identity": len(IDENTITY),
            "echo": len(ECHO),
        }.get(category, len(pairs))
        if pool_size < count:
            # The benchmark is bounded by the corpus that exists. Stopping here
            # turned a small held-out split into no benchmark at all, which is
            # how a 1,000-row target ended up with nothing to score.
            print(f"note: {category} pool has {pool_size} rows; building {pool_size}, not {count}")
            counts[category] = count = pool_size
        added = 0
        candidate = 0
        while added < count:
            if category in {"translate_en_cr", "translate_cr_en"}:
                row = _translation_row(category, candidate, pairs[candidate % len(pairs)], held_out)
            elif category in {"language_id_cree", "language_id_english", "language_id_mixed"}:
                row = _language_row(category, candidate, pairs[candidate % len(pairs)], held_out)
            elif category == "word_explain":
                row = _word_row(candidate, words[candidate % len(words)], held_out)
            elif category == "identity":
                question, fact = IDENTITY[candidate % len(IDENTITY)]
                row = _identity_row(candidate, question, fact)
            else:
                row = _echo_row(candidate, ECHO[candidate % len(ECHO)])
            if _add(row):
                added += 1
            candidate += 1
            if candidate > count * 50:
                raise SystemExit(f"could not build {count} unique {category} rows from {pool_size} candidates")

    # Deterministic order, stable across runs and machines.
    rows.sort(key=lambda row: hashlib.sha1(row["id"].encode()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    for category, count in counts.items():
        print(f"  {category:20s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

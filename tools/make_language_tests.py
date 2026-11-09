"""Generate thousands of labelled language samples to test the chat with.

Every row is built from this project's own text and carries what the app should
conclude about it: which language the message is in, which prompt shape it
should take, and which language the answer should come back in. That makes it a
test of the routing and of the model's reply language, not a list of questions
somebody thought of.

The pools are filtered, because a benchmark is only worth its labels. The
aligned corpus contains rows that are half English and dictionary rows whose
English side is a Cree headword, so:

* a Cree sentence must carry at least one long vowel (marked orthography), have
  three words or more, and not contain the English article "the"; the mixed rows
  and abbreviation residue are dropped;
* an English sentence comes from the project's own English prose, or from the
  English side of a pair with no long vowels and none of the words that turn a
  sentence into a language request;
* an English word must occur in that prose, so "jsewao" cannot be labelled
  English just because it sat in the English column of a dictionary row.

The split is the held-out tenth of chat/language_id.py, so the identifier was
not trained on these sentences.

    .venv/bin/python tools/make_language_tests.py               # 4000 rows
    .venv/bin/python tools/make_language_tests.py --n 8000

Writes data/eval/language_tests.jsonl and prints the family counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat.language_id import is_held_out, normalise                  # noqa: E402

OUT = ROOT / "data" / "eval" / "language_tests.jsonl"

# A sentence that is itself a language request is not a chat sentence, and
# labelling it "chat" would test the wrong thing.
_INTENT = re.compile(
    r"\bcree\b|\btranslate\b|\bsyllabics?\b|\bsro\b|\bword for\b|\bhow do you say\b",
    re.IGNORECASE,
)
_MARKED = re.compile(r"[âêîô]")
_SYLLABICS = re.compile(r"[\u1400-\u167F]")


def _clean(text: str) -> str:
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# English function words. Used only to throw contaminated rows out of the
# benchmark: a row in the Cree column of the aligned data that contains "the"
# or "is" is not a Cree sentence, and labelling it Cree would make the test
# measure nothing. This is data hygiene for the benchmark, never routing.
_ENGLISH_MARKERS = re.compile(
    r"\b(?:the|and|is|are|was|were|of|to|in|it|this|that|there|what|how|not|"
    r"much|he|she|they|you|for|with|on|at|from|by|or|but|if|a|an)\b",
    re.IGNORECASE,
)


def _trustworthy_cree(text: str) -> bool:
    """A Cree sentence worth putting in the benchmark.

    Either it is written in marked orthography (a long vowel, which no English
    word has) or it is long enough that a stray English gloss could not be the
    whole row. Short unmarked rows in the Cree column are often English
    residue and are dropped, because a benchmark with wrong labels measures
    nothing.
    """
    words = text.split()
    return (
        len(words) >= 3
        and (bool(_MARKED.search(text)) or len(words) >= 6)
        and not re.search(r"\d", text)
        and "inv." not in text.lower()
        and not _ENGLISH_MARKERS.search(text)
    )


def _trustworthy_english(text: str) -> bool:
    return (
        len(text.split()) >= 4
        and not _MARKED.search(text)
        and not _SYLLABICS.search(text)
        and not re.search(r"\d", text)
        and not _INTENT.search(text)
    )


def _prose_sentences() -> list[str]:
    paths = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
    sentences: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"[#*_`>|\[\]()]+", " ", text)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = _clean(sentence)
            if _trustworthy_english(sentence):
                sentences.append(sentence)
    return sentences


def _read_pools() -> tuple[list[str], list[str], list[str], list[str]]:
    cree_sentences: list[str] = []
    english_sentences: list[str] = list(_prose_sentences())
    cree_words: list[str] = []
    english_words: list[str] = []
    processed = ROOT / "data" / "processed"

    for name in ("parallel_cr_en.jsonl", "parallel_clean.jsonl", "cree_corpus.jsonl"):
        path = processed / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                cree = _clean(str(row.get("cree") or row.get("text") or ""))
                english = _clean(str(row.get("english") or ""))
                if cree and _trustworthy_cree(cree):
                    cree_sentences.append(cree)
                if english and _trustworthy_english(english):
                    english_sentences.append(english)

    prose_vocabulary = {
        token.lower()
        for sentence in english_sentences
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
    }
    # Cree headwords: the dictionary is written in unmarked orthography, so a
    # marked word has to come from the marked sentences. Every token of a
    # trustworthy Cree sentence that carries a long vowel is a Cree word, which
    # gives thousands without guessing at any label.
    for sentence in cree_sentences:
        for token in re.findall(r"[^\W\d_][^\W\d_'’-]*", sentence):
            if 3 <= len(token) <= 20 and _MARKED.search(token):
                cree_words.append(token)

    words = processed / "watkins_dictionary_pairs.jsonl"
    if words.exists():
        with words.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                english = str(row.get("english") or "").strip()
                if (
                    3 <= len(english) <= 20 and " " not in english
                    and not _MARKED.search(english)
                    and english.lower() in prose_vocabulary
                ):
                    english_words.append(english)
    return cree_sentences, english_sentences, cree_words, english_words


def _held_out(pool: list[str]) -> list[str]:
    """Deterministic order: sort by hash, keep the reserved tenth."""
    unique = {normalise(text): _clean(text) for text in pool if normalise(text)}
    rows = [(hashlib.sha1(key.encode()).hexdigest(), text)
            for key, text in unique.items() if is_held_out(text)]
    rows.sort()
    return [text for _digest, text in rows]


# Each template carries its own expectation. "translate this" asks for a
# translation, "what does this mean" asks for an explanation, and a message
# written in Cree is answered in English, not in Cree.
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "note_question": [
        ("what is {title} about?", "song_notes", "en"),
        ("tell me about {title}", "song_notes", "en"),
        ("explain {title}", "song_notes", "en"),
        ("what does {title} mean?", "song_notes", "en"),
    ],
    "translate_request": [
        ("how do you say {english} in Cree?", "translate", "cr"),
        ("translate this into Cree: {english}", "translate", "cr"),
        ("say {english} in Cree", "translate", "cr"),
        ("what is the Cree word for {english}?", "translate", "cr"),
    ],
    "word_question": [
        ("what does {cree} mean?", "chat", "en"),
        ("explain the Cree word {cree}", "chat", "en"),
        ("what does {cree} mean in English?", "chat", "en"),
    ],
    "code_switch": [
        ('translate this: "{cree}"', "translate", "en"),
        ('what does "{cree}" mean?', "chat", "en"),
        ('how would you say "{cree}" in English?', "translate", "en"),
    ],
}

FAMILIES = (
    "cree_sentence",
    "english_sentence",
    "cree_word",
    "english_word",
    "note_question",
    "translate_request",
    "word_question",
    "code_switch",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    cree_sentences, english_sentences, cree_words, english_words = _read_pools()
    pools = {
        "cree_sentence": _held_out(cree_sentences),
        "english_sentence": _held_out(english_sentences),
        "cree_word": _held_out(cree_words),
        "english_word": _held_out(english_words),
    }
    for name, pool in pools.items():
        print(f"  pool {name}: {len(pool)} held-out examples")
        if not pool:
            print(f"FAILED: no held-out examples for {name}")
            return 1

    per_family = max(args.n // len(FAMILIES), 1)
    out: list[dict] = []

    def take(pool: list[str], index: int) -> str:
        return pool[index % len(pool)]

    for index in range(per_family):
        cree_sentence = take(pools["cree_sentence"], index)
        english_sentence = take(pools["english_sentence"], index)
        cree_word = take(pools["cree_word"], index)
        english_word = take(pools["english_word"], index)

        rows = [
            ("cree_sentence", cree_sentence, "cr", "translate", "en", False),
            ("english_sentence", english_sentence, "en", "chat", "en", False),
            ("cree_word", cree_word, "cr", "translate", "en", False),
            ("english_word", english_word, "en", "chat", "en", False),
        ]
        for family in ("note_question", "translate_request", "word_question", "code_switch"):
            template, task, answer = TEMPLATES[family][index % len(TEMPLATES[family])]
            text = template.format(
                title="Horse",
                english=english_sentence,
                cree=cree_word if family == "word_question" else cree_sentence,
            )
            rows.append((family, text, "mixed" if family == "code_switch" else "en",
                         task, answer, family == "note_question"))

        for family, text, lang, task, answer, notes in rows:
            out.append({
                "id": f"{family}-{index:05d}",
                "family": family,
                "text": text,
                "expect": {
                    "lang": lang,
                    "task": task,
                    "answer_language": answer,
                    "notes": notes,
                },
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in out:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    print(f"\n{len(out)} samples written to {OUT}")
    for family in FAMILIES:
        print(f"  {family:20s} {counts.get(family, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

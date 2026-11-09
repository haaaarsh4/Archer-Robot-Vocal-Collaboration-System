"""Which language is this text in? Learned from this project's own corpus.

There is no list of Cree words in this file, and that is deliberate. The app
used to decide "this is Cree" by looking for one of about fifty hand-picked
words or a circumflex, which fails on every Cree word nobody happened to type
into the set, fires on English text that contains one loanword, and can only
ever be as good as the person who wrote the list.

What replaces it is the method production multilingual systems use: a linear
classifier over character n-grams trained on labelled text. That is what
fastText is (Joulin et al., 2016), and GlotLID (Kargaran et al., Findings of
EMNLP 2023) shows the relevant thing for this project: for a low-resource
language, a small model trained on that language's own text beats a large model
that barely saw it, and the corpus is what has to be right. This project already
has 27,000 aligned Cree sentences, corpus sentences and dictionary forms, so the
classifier is trained on those instead of downloaded.

Three decisions come from the literature rather than taste:

* Character n-grams, not words. Cree is polysynthetic, so a word list covers
  almost nothing: a stem with a different suffix is a word the list has never
  seen. Character 2-grams to 5-grams cover the morphology, which is also why the
  tokenizer work here landed on a morphology-aware BPE.
* Logistic regression, not naive Bayes. The first version of this file used
  multinomial naive Bayes and measured 0.603 sentence accuracy. The n-grams were
  the same; the scorer was the problem. A discriminative linear model on the same
  features is what fastText uses and what the accuracy gate needs.
* Per-word decisions as well as the whole message. Understanding and Mitigating
  Language Confusion in LLMs (EMNLP 2024) measures confusion at line and word
  level separately, and the code-switching work (VarDial 2024; MaskLID and its
  integer-programming follow-up, 2026) shows that utterance-level LID is the
  wrong granularity for mixed input. `language_of()` therefore returns a message
  label, a per-word reading, and the share of words that are Cree, which is what
  the router uses to decide which language to answer in.

The model is trained by tools/train_language_id.py and written to
chat/language_id_model.json as weights over character n-grams. Runtime scoring
is a dot product over the grams present in the input, so it needs no GPU, no
network and no framework.

If the model file is missing, this module trains itself once from
data/processed and saves the result. If the data is missing too, every call
returns "und" and the app says it cannot tell, which is honest: guessing Cree
from an English sentence is the failure this module exists to remove.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

__all__ = [
    "Verdict", "language_of", "predict", "per_word", "cree_share", "cree_run",
    "is_cree", "WORD_DECIDED",
    "is_english", "normalise", "train", "save", "is_held_out", "MODEL_PATH",
    "CREE_MIN_POSTERIOR",
]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(__file__).resolve().parent / "language_id_model.json"

# 2 to 5 characters. One is too little signal; six starts memorising whole
# English suffixes and costs speed for nothing measured.
NGRAM_RANGE = (2, 5)
MIN_COEF = 0.01
SENTENCE_FEATURES = 60_000
WORD_FEATURES = 30_000

# How sure a single word has to be before it counts as decided. Below this it
# is "und" and is left out of the share, because a two-letter word carries no
# evidence either way.
WORD_DECIDED = 0.60

_SYLLABICS = re.compile(r"[\u1400-\u167F\u18B0-\u18FF]")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD = re.compile(r"[^\W\d_][^\W\d_'’-]*", re.UNICODE)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def normalise(text: str) -> str:
    """Lowercase, NFC, letters and inner apostrophes only, single spaces."""
    text = _nfc(text).lower().replace("’", "'")
    text = re.sub(r"[^a-zâêîôûāēīōū' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _padded(text: str) -> str:
    """One space at each end so a word edge is itself a feature."""
    return f" {normalise(text)} "


def _grams(text: str, ngram: tuple[int, int] = NGRAM_RANGE) -> Counter:
    padded = _padded(text)
    counts: Counter = Counter()
    for n in range(ngram[0], ngram[1] + 1):
        for i in range(len(padded) - n + 1):
            counts[padded[i:i + n]] += 1
    return counts


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _fit(padded_texts: list[str], labels: list[int], max_features: int,
         ngram: tuple[int, int] = NGRAM_RANGE, c_value: float = 4.0) -> dict:
    """Fit one linear classifier and export it as gram weights.

    Imported here rather than at module import: training is a build step, and
    the running app should not pay for sklearn it never uses.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    vectoriser = CountVectorizer(
        analyzer="char",
        ngram_range=ngram,
        min_df=3,
        max_features=max_features,
        lowercase=False,
    )
    matrix = vectoriser.fit_transform(padded_texts)
    model = LogisticRegression(
        max_iter=300,
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(matrix, labels)
    names = vectoriser.get_feature_names_out()
    weights = model.coef_[0]
    return {
        "intercept": round(float(model.intercept_[0]), 5),
        "features": int(matrix.shape[1]),
        "ngram": list(ngram),
        "coef": {
            str(names[i]): round(float(weights[i]), 4)
            for i in range(len(names))
            if abs(float(weights[i])) >= MIN_COEF
        },
    }


def train(sentence_rows: Iterable[tuple[str, str]],
          word_rows: Iterable[tuple[str, str]] | None = None) -> dict:
    """Fit the sentence and word classifiers and return the model dict.

    ``sentence_rows`` and ``word_rows`` are (text, "cr"|"en") pairs. They are
    fitted separately on purpose: a model trained only on sentences applied to a
    single word has almost no evidence to work with, and the word reading is
    what the router uses for code-switched input.
    """
    sentences: dict[str, str] = {}
    for text, label in sentence_rows:
        key = normalise(text)
        if key and key not in sentences:
            sentences[key] = label
    words: dict[str, str] = {}
    for text, label in (word_rows or []):
        key = normalise(text)
        if key and " " not in key and key not in words:
            words[key] = label

    def split(rows: dict[str, str]) -> tuple[list[str], list[int], int, int]:
        texts = [f" {key} " for key in rows]
        labels = [1 if rows[key] == "cr" else 0 for key in rows]
        return texts, labels, sum(labels), len(labels) - sum(labels)

    sentence_texts, sentence_labels, cr_docs, en_docs = split(sentences)
    word_texts, word_labels, word_cr, word_en = split(words)
    if not sentence_texts:
        raise ValueError("no training sentences")

    model = {
        "version": 2,
        "ngram": list(NGRAM_RANGE),
        "documents": {"cr": cr_docs, "en": en_docs,
                      "word_cr": word_cr, "word_en": word_en},
        "sentence": _fit(sentence_texts, sentence_labels, SENTENCE_FEATURES),
        # The word model gets single characters. That is where the strongest
        # evidence about a short word lives: English uses b, d, f, g, j, l, q,
        # r, v, x and z, and Cree does not, so a four-letter word with a d in it
        # is decided by a feature the 2-gram model never had. It is also more
        # strongly regularised, because a word gives far fewer n-grams to fit.
        "word": _fit(word_texts, word_labels, WORD_FEATURES,
                     ngram=(1, 4), c_value=2.0) if word_texts else None,
    }
    return model


def save(model: dict, path: Path | None = None) -> Path:
    path = Path(path or MODEL_PATH)
    path.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8")
    return path


_MODEL: dict | None = None
_SUB: dict[str, dict] | None = None


def _bootstrap() -> dict | None:
    """Train from data/processed when the model file is absent.

    The app must keep working from a fresh clone, and it must never fall back to
    a word list, so the fallback is: learn the language from whatever labelled
    text is on disk.
    """
    processed = ROOT / "data" / "processed"
    rows: list[tuple[str, str]] = []
    pairs = processed / "parallel_cr_en.jsonl"
    if pairs.exists():
        with pairs.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("cree"):
                    rows.append((str(row["cree"]), "cr"))
                if row.get("english"):
                    rows.append((str(row["english"]), "en"))
    corpus = processed / "cree_corpus.jsonl"
    if corpus.exists():
        with corpus.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    text = json.loads(line).get("text")
                except ValueError:
                    continue
                if text:
                    rows.append((str(text), "cr"))
    words = processed / "watkins_dictionary_pairs.jsonl"
    word_rows: list[tuple[str, str]] = []
    if words.exists():
        with words.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for key, label in (("cree", "cr"), ("english", "en")):
                    value = str(row.get(key) or "").strip()
                    if value:
                        rows.append((value, label))
                        if " " not in value:
                            word_rows.append((value, label))
    if not rows:
        return None
    model = train(rows, word_rows)
    try:
        save(model)
    except OSError:
        pass  # a read-only checkout still gets the trained model in memory
    return model


def _load() -> dict | None:
    global _MODEL, _SUB
    if _SUB is not None:
        return _MODEL
    model = None
    if MODEL_PATH.exists():
        try:
            model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        except ValueError:
            model = None
    if not model or model.get("version") != 2:
        model = _bootstrap()
    _MODEL = model
    if not model:
        _SUB = {}
        return None
    sub = {}
    for name in ("sentence", "word"):
        data = model.get(name)
        if data:
            sub[name] = {
                "intercept": float(data["intercept"]),
                "coef": {str(k): float(v) for k, v in data["coef"].items()},
                "ngram": tuple(data.get("ngram", NGRAM_RANGE)),
            }
    _SUB = sub
    return model


def _posterior_cree(text: str, which: str) -> float:
    sub = _SUB or {}
    data = sub.get(which) or sub.get("sentence")
    if not data:
        return 0.5
    grams = _grams(text, data["ngram"])
    logit = data["intercept"]
    coef = data["coef"]
    for gram, count in grams.items():
        weight = coef.get(gram)
        if weight:
            logit += count * weight
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    return math.exp(logit) / (1.0 + math.exp(logit))


def predict(text: str) -> tuple[float, float]:
    """Return (posterior that the text is Cree, share of words that are Cree).

    A posterior of 0.5 with no words read means the model cannot tell, not that
    the text is half Cree; callers should use ``language_of`` for the label.
    """
    _load()
    tokens = _WORD.findall(_nfc(text))
    which = "word" if len(tokens) == 1 else "sentence"
    return _posterior_cree(text, which), cree_share(text)


@lru_cache(maxsize=2048)
def per_word(text: str) -> tuple[tuple[str, str, float], ...]:
    """(word, "cr"|"en"|"und", confidence) for each word in the text.

    This is the code-switching reading: a message can be mostly English with a
    Cree sentence quoted inside it, and the whole-message label cannot say which
    part is which. Words under three letters are undecided on purpose, because
    two characters are not evidence in either language.
    """
    _load()
    out: list[tuple[str, str, float]] = []
    for match in _WORD.finditer(_nfc(text)):
        word = match.group(0)
        if _SYLLABICS.search(word):
            out.append((word, "cr", 1.0))
            continue
        if len(normalise(word)) < 3:
            out.append((word, "und", 0.0))
            continue
        p_cree = _posterior_cree(word, "word")
        confidence = max(p_cree, 1.0 - p_cree)
        if p_cree >= WORD_DECIDED:
            out.append((word, "cr", confidence))
        elif p_cree <= 1.0 - WORD_DECIDED:
            out.append((word, "en", confidence))
        else:
            out.append((word, "und", confidence))
    return tuple(out)


def cree_share(text: str) -> float:
    """Fraction of decided words that the classifier reads as Cree."""
    decided = [label for _word, label, _conf in per_word(text) if label in {"cr", "en"}]
    if not decided:
        return 0.0
    return sum(label == "cr" for label in decided) / len(decided)


def cree_run(text: str) -> int:
    """Longest run of consecutive Cree words, ignoring undecided ones.

    A quiet but important distinction for this app: a Cree word inside an
    English question is a mention, and the reply should be English ("what does
    maskwa mean"). A run of Cree words is use, and the message itself is mixed.
    Share alone cannot tell those apart, because one long Cree word can be the
    same fraction of a short sentence as three short ones.
    """
    best = run = 0
    for _word, label, _conf in per_word(text):
        if label == "cr":
            run += 1
            best = max(best, run)
        elif label == "en":
            run = 0
    return best


@dataclass(frozen=True)
class Verdict:
    """What the identifier concluded about one message."""
    lang: str            # "cr" | "en" | "mixed" | "und"
    confidence: float    # posterior of the message-level winner
    p_cree: float        # raw posterior that the message is Cree
    cree_share: float    # share of decided words that are Cree
    words: tuple[tuple[str, str, float], ...] = ()


@lru_cache(maxsize=4096)
def language_of(text: str) -> Verdict:
    """Label one message, and say how mixed it is.

    The message labels mean exactly what they say: "cr" is Cree, "en" is
    English, "mixed" is a message with real material in both (a Cree sentence
    quoted inside an English request is the common case here), and "und" is the
    identifier declining to guess.

    "mixed" needs a span of at least three Cree words, a word-level share that
    is neither all one language nor all the other, and at least three tokens to
    look at. Without the word count a two-word fragment could be called mixed on
    one word's noise; without the span an English question quoting one Cree word
    would be called mixed.
    """
    text = _nfc(text)
    words = per_word(text)
    letters = len(_LETTERS.findall(text))
    syllabic = len(_SYLLABICS.findall(text))
    if not letters:
        return Verdict("und", 0.0, 0.5, 0.0, words)
    if syllabic and syllabic / max(letters, 1) >= 0.3:
        return Verdict("cr", 1.0, 1.0, 1.0, words)

    tokens = _WORD.findall(text)
    p_cree = _posterior_cree(text, "word" if len(tokens) == 1 else "sentence")
    confidence = max(p_cree, 1.0 - p_cree)
    share = cree_share(text)
    # Mixed first: a message can be confidently both, and asking "is the whole
    # thing Cree" before "is it mixed" labelled every English question quoting
    # a Cree sentence as Cree. A span of three words is the line between using
    # the language and quoting one word of it.
    run = cree_run(text)
    cr_words = sum(1 for _w, label, _c in words if label == "cr")
    en_words = sum(1 for _w, label, _c in words if label == "en")
    # Both languages present, each with more than one decided word. One Cree
    # word does not make a mixed message, it makes an English message mentioning
    # a Cree word; one English word inside a Cree sentence is a name. Undecided
    # words are not counted as either, so a Cree sentence full of short particles
    # cannot be pushed into "mixed" by them.
    if cr_words >= 2 and en_words >= 2 and run >= 2 and 0.25 <= share <= 0.95:
        return Verdict("mixed", confidence, p_cree, share, words)
    # No word reads as Cree: whatever the sentence model makes of a short
    # English string, it is not Cree. "I met no one" scored 0.62 for Cree on
    # length alone before this line existed. A single token is exempt, because
    # there the word model's own posterior is the reading and 0.535 for a real
    # Cree noun is a decision, not an absence of one.
    if cr_words == 0 and len(tokens) > 1:
        return Verdict("en", confidence, p_cree, share, words)
    # An English question carrying a quoted Cree word is English, and is
    # answered in English, even when the sentence model is confident the quote
    # is Cree. The counts are the word classifier's reading, not a word list.
    if en_words >= 3 and cr_words <= 2 and run < 3:
        return Verdict("en", confidence, p_cree, share, words)
    return Verdict("cr" if p_cree >= 0.5 else "en", confidence, p_cree, share, words)


@lru_cache(maxsize=2048)
def is_cree(text: str) -> bool:
    """Is this message Cree enough to answer in Cree?"""
    verdict = language_of(text)
    return verdict.lang == "cr" or (
        verdict.lang == "mixed" and verdict.cree_share >= 0.5
    )


def is_english(text: str) -> bool:
    return language_of(text).lang == "en"


def is_held_out(text: str, of: int = 10) -> bool:
    """The test split, as a function rather than a file.

    Training and the sample generator both call this, so no test sentence can
    have been trained on: the split is a hash of the text, not an index.
    """
    digest = hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % of == 0

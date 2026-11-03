"""The nêhiyawêwin text side of the transcriber: lexicon, character LM, phonotactics.

An acoustic model alone cannot transcribe a language it has few hours of. Meta's
MMS crk adapter is trained on read Plains Cree and, on a sung take with drums
under it, its frame posteriors are noisy enough that a greedy decode walks off
into whatever character sequence is locally cheapest -- which, for this project,
was English words. Three things fix that, and all three are *language* knowledge
rather than more audio:

  * a **lexicon**, every word of Plains Cree this project can find, weighted by
    how often it actually occurs;
  * a **character n-gram model** over the same text, which knows what a Cree word
    looks like long before it knows which word it is -- the language's
    orthography is regular enough that a 6-gram over characters rules out almost
    every non-word;
  * **phonotactics**, the syllable inventory (onset, nucleus, coda) read out of
    the lexicon, which accepts a word the lexicon has never seen as long as it is
    built from syllables the language actually uses.

Those three are fused into the CTC beam search in analysis/cree_asr.py. This
module is only the text side, and it is buildable offline: the
corpus is this project's own `data/PlainsCree/`, plus whatever optional lexicon
files have been dropped into `data/models/cree/lm/`.

Two things worth knowing about the corpus before reading the numbers it reports:

  * The `*_cr.txt` files are Cree in Standard Roman Orthography, and the
    `*_syllabics.txt` files are the same books in syllabics; the latter are
    converted through analysis/cree_orthography so both contribute the same
    words. `*_en.txt` files are English and are not read at all.
  * Some files in this corpus write long vowels doubled (`aa`, `ee`, `ii`, `oo`)
    instead of with a diacritic. Those are folded before counting, because a
    lexicon that holds `nêhiyawêwin` and `nehiyawewin` as two different words is
    a worse lexicon than one that holds them as one. Folding `ee` is the one
    ambiguous case (see `fold_doubled_vowels`) and is reported, not hidden.

Output of the build is one JSON file, `data/models/cree/lm/cre_text_model.json`,
so the transcriber starts in milliseconds instead of re-reading the corpus.
"""

from __future__ import annotations

import glob
import json
import math
import os
import pathlib
import re
import unicodedata
from collections import Counter, defaultdict

from analysis.cree_orthography import (
    fold_doubled_vowels,
    is_syllabics,
    looks_like_syllabics,
    normalize_sro,
    syllabify,
    to_sro,
)

__all__ = [
    "CreTextModel", "WORD_BOUNDARY", "load_text_model", "build_text_model",
    "CREE_LETTERS", "cree_letters_only", "text_model_path",
]

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_LM_DIR = _ROOT / "data" / "models" / "cree" / "lm"
_MODEL_PATH = _LM_DIR / "cre_text_model.json"

# The letters nêhiyawêwin is written with in Standard Roman Orthography: the
# twelve consonants of the language (p t k c m n s h w y, plus the l and r of
# the western dialects and the th of Woods Cree), the four short vowels, the four
# long ones, and the `hk` sequence, which is a single coda. b d f g j q u v x z
# are NOT here on purpose: they only ever appear in this corpus inside loanwords
# and proper names (biscuit, Alphaeus), and an English word that survives the
# filter below is exactly the failure this model exists to prevent.
CREE_LETTERS = set("ptkcsmnhwy" "lr" "aeio" "âêîô")

# Where the corpus lives, in the order its files are read. `*_cr.txt` are the
# canonical SRO texts; `*_syllabics.txt` are converted, so between them every
# Cree sentence in the corpus is counted exactly once even where a book ships in
# both scripts.
_CORPUS_GLOBS = (
    "data/PlainsCree/**/*_cr.txt",
    "data/PlainsCree/**/*_syllabics.txt",
)

# Optional extra lexicons, dropped in by tools/fetch_cree_lexicon.py. Each is
# read as newline- or whitespace-separated Cree words/tokens with no counts, and
# contributes presence (a small count) rather than frequency.
_LEXICON_GLOBS = (
    "data/models/cree/lm/lexicon*.txt",
)

WORD_BOUNDARY = " "


def text_model_path() -> pathlib.Path:
    return _MODEL_PATH


def cree_letters_only(word: str) -> bool:
    """Is every letter of this token a letter Cree is written with?"""
    if not word:
        return False
    return all(ch in CREE_LETTERS for ch in word)


# corpus read

def _tokens_from_line(line: str, fold_doubled: bool) -> list[str]:
    """Normalize one line of corpus text and return its Cree tokens.

    Syllabics is converted to SRO first, then normalized, then the doubled-vowel
    convention is folded. A token keeps only its letters: punctuation, digits and
    the stray footnote marks these texts carry are not part of a word.
    """
    line = str(line).strip()
    if not line:
        return []
    if looks_like_syllabics(line):
        line = to_sro(line)
    line = normalize_sro(line, fold_doubled=fold_doubled)
    out: list[str] = []
    for token in line.split(" "):
        core = "".join(ch for ch in token if ch in CREE_LETTERS or ch.isalpha() or ch in "âêîô")
        if not core:
            continue
        if not cree_letters_only(core):
            # A token with a non-Cree letter in it is a loanword or a proper
            # name. Dropping it here is what keeps `biscuit` out of the lexicon.
            continue
        if len(core) < 2:
            continue
        out.append(core)
    return out


def read_corpus(root: pathlib.Path | None = None,
                extra_globs: tuple[str, ...] = _LEXICON_GLOBS) -> tuple[Counter, dict]:
    """Read every Cree token this project has, with counts, and report provenance.

    Returns `(word_counts, report)`. The report is not decoration: the number of
    words each source contributed is the only way to tell whether the corpus
    changed under the model, and the transcriber's accuracy depends on it.
    """
    root = root or _ROOT
    counts: Counter = Counter()
    doubled_words: set[str] = set()
    sources: dict[str, dict] = {}

    files: list[pathlib.Path] = []
    for pattern in _CORPUS_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    for pattern in extra_globs:
        files.extend(sorted(root.glob(pattern)))

    # A file whose path contains `_en` is English; skip it even if a glob caught
    # it, so adding a glob can never quietly put English into the lexicon.
    for path in files:
        name = path.name
        if "_en" in name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        is_lexicon = "lexicon" in name
        tokens: list[str] = []
        for line in text.splitlines():
            if is_lexicon:
                # a bare word list: one entry per line, no folding (these are
                # already canonical SRO with diacritics)
                line = normalize_sro(line)
                core = "".join(ch for ch in line if ch in CREE_LETTERS)
                if core and cree_letters_only(core) and len(core) > 1:
                    tokens.append(core)
                    if re.search(r"aa|ee|ii|oo|uu", core):
                        doubled_words.add(core)
                continue
            fold = bool(re.search(r"aa|ee|ii|oo|uu", line))
            toks = _tokens_from_line(line, fold_doubled=fold)
            if fold:
                for t in toks:
                    if re.search(r"aa|ee|ii|oo|uu", t):
                        doubled_words.add(t)
            tokens.extend(toks)
        for t in tokens:
            counts[t] += 1
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        key = os.path.dirname(rel) or rel
        entry = sources.setdefault(key, {"files": 0, "tokens": 0})
        entry["files"] += 1
        entry["tokens"] += len(tokens)

    report = {
        "files": len(files),
        "tokens": sum(counts.values()),
        "types": len(counts),
        "doubled_vowel_types": len(doubled_words),
        "sources": dict(sorted(sources.items())),
    }
    return counts, report


# the model

def _syllable_parts(syllable: str) -> tuple[str, str, str] | None:
    """Split one syllable into (onset, nucleus, coda), or None if unparseable.

    Nucleus is a run of vowels (Cree writes long vowels as a single marked
    letter, but a sequence like `iya` really is two syllables and `syllabify`
    has already separated those, so what is left inside a syllable is one vowel,
    or a vowel plus the `w`/`y` glide that rode in with the onset).
    """
    m = re.match(r"^([^aeiouâêîô]*)([aeiouâêîô]+)([^aeiouâêîô]*)$", syllable)
    if not m:
        return None
    onset, nucleus, coda = m.group(1), m.group(2), m.group(3)
    if not nucleus:
        return None
    return onset, nucleus, coda


class CreTextModel:
    """Lexicon + character n-gram LM + phonotactics for nêhiyawêwin.

    Scoring is log-probability throughout, and every method is safe to call on
    arbitrary input: unseen words, empty strings and stray symbols all score
    finite values, because the beam search calls these millions of times and a
    single exception would take the decoder down mid-utterance.
    """

    def __init__(self, words: dict[str, int], char_ctx: dict[str, dict[str, int]],
                 report: dict | None = None, order: int = 6, discount: float = 0.75):
        self.words = words
        self.order = order
        self.discount = discount
        self.report = report or {}
        self._char_ctx = char_ctx
        # totals and uniqueness per context, precomputed: the beam search asks
        # for these on every extension.
        self._ctx_total: dict[str, int] = {}
        self._ctx_types: dict[str, int] = {}
        for ctx, row in self._char_ctx.items():
            self._ctx_total[ctx] = sum(row.values())
            self._ctx_types[ctx] = len(row)
        self._accepted_vocab = set(CREE_LETTERS) | {"▲"}
        self._uni_total = 0
        self._uni_counts: dict[str, int] = {}
        for row in self._char_ctx.get("", {}).items():
            self._uni_counts[row[0]] = row[1]
            self._uni_total += row[1]

        self.total_words = sum(words.values())
        self._word_alpha = 0.35
        self._word_denom = self.total_words + self._word_alpha * max(1, len(words))

        # phonotactics, read out of the lexicon itself
        onsets: Counter = Counter()
        codas: Counter = Counter()
        nuclei: Counter = Counter()
        for word, count in words.items():
            for syl in syllabify(word):
                parts = _syllable_parts(syl)
                if parts is None:
                    continue
                o, n, c = parts
                onsets[o] += count
                nuclei[n] += count
                codas[c] += count
        self.onsets = dict(onsets)
        self.nuclei = dict(nuclei)
        self.codas = dict(codas)
        self._max_onset = max((len(o) for o in onsets), default=3)
        self._max_coda = max((len(c) for c in codas), default=3)
        self._prob_cache: dict[str, dict[str, float]] = {}
        self._word_logprob_cache: dict[str, float] = {}

        # baselines the decoder measures *against*, not against zero
        #
        # An absolute log-probability is the wrong thing to add to an acoustic
        # score. Both get more negative as a transcription gets longer, so a term
        # of that shape rewards saying nothing: a decoder given one will happily
        # answer a whole sentence with a single syllable, and it did, until these
        # two numbers were subtracted off. What the decoder needs from the text
        # model is not "how probable is this text" -- the acoustic score already
        # settles the question of how much was said -- but "how much *better than
        # this language's own average* is it": the average log-probability of a
        # letter, and of a syllable, so that ordinary nêhiyawêwin scores near zero
        # and a deviation from it scores its size.
        self.char_baseline = self._entropy(self._char_ctx.get("", {}))
        self.syllable_baseline = (self._entropy(onsets) + self._entropy(nuclei)
                                  + self._entropy(codas))

    @staticmethod
    def _entropy(counts: dict[str, int]) -> float:
        """The mean `log p` of a distribution's own symbols: its negative entropy.

        This is the expected log-probability of a symbol drawn from the
        distribution itself, which is exactly the number a term has to be
        measured against for a well-formed input to score near zero.
        """
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        out = 0.0
        for count in counts.values():
            if count <= 0:
                continue
            p = count / total
            out += p * math.log(p)
        return out

    # lexicon

    def is_known(self, word: str) -> bool:
        return word in self.words

    def word_count(self, word: str) -> int:
        return self.words.get(word, 0)

    def word_logprob(self, word: str) -> float:
        """Unigram probability of a word, with an explicit penalty for the unknown.

        The unknown penalty is what turns "the decoder produced an English word"
        from a thing that happens into a thing that costs: `alpha` is small
        enough against the corpus total that an unseen word is heavily
        dispreferred next to any real Cree word with a count, while still
        affordable if the acoustics are emphatic enough (a name, a vocable).
        """
        if word in self._word_logprob_cache:
            return self._word_logprob_cache[word]
        c = self.words.get(word, 0)
        if c:
            val = math.log((c + self._word_alpha) / self._word_denom)
        else:
            # ZERO for the unknown, not a constant: see `unknown_word_logprob`.
            val = math.log(self._word_alpha / self._word_denom)
        self._word_logprob_cache[word] = val
        return val

    @property
    def unknown_word_logprob(self) -> float:
        return math.log(self._word_alpha / self._word_denom)

    # phonotactics

    def is_legal_word(self, word: str) -> bool:
        """Could this be a nêhiyawêwin word the lexicon has never seen?

        Every syllable has to be one the language builds: an onset the lexicon
        uses, a nucleus it uses, a coda it uses. Not a lookup -- `mîkisowin` is
        legal whether or not it is in the corpus, because all three of its
        syllables are ordinary Cree.
        """
        if not word or not cree_letters_only(word):
            return False
        syllables = syllabify(word)
        if not syllables:
            return False
        for syl in syllables:
            parts = _syllable_parts(syl)
            if parts is None:
                return False
            onset, nucleus, coda = parts
            if onset not in self.onsets and onset:
                return False
            if nucleus not in self.nuclei:
                return False
            if coda not in self.codas and coda:
                return False
        return True

    def syllable_baseline_of(self, word: str) -> float:
        """What :meth:`phonotactic_score` would return for an ordinary word of this length.

        The baseline a word's syllable score has to be compared against. A
        five-syllable word is not worse than a two-syllable one; only a word with
        *unusual* syllables in it is worse, and this is the number that makes that
        distinction instead of measuring length.
        """
        return self.syllable_baseline * len(syllabify(word))

    def phonotactic_relative(self, word: str) -> float:
        """How much *better or worse* than ordinary this word's syllables are.

        Zero for a word built out of the syllables the language uses normally,
        negative for one built out of rare or impossible ones, and length-neutral
        either way -- see the baselines in `__init__` for why that matters.
        """
        return self.phonotactic_score(word) - self.syllable_baseline_of(word)

    def phonotactic_score(self, word: str) -> float:
        """Syllable phonotactics of a word, **summed** over its syllables.

        Each syllable's onset, nucleus and coda is scored by how often the lexicon
        uses it, and the scores are added up. The sum, not the mean, is what the
        decoder charges: a score that is averaged over syllables is a *constant*
        per word, and a constant charged once per word is a penalty on having
        words at all. A decoder given that penalty does the only thing it can and
        welds every sentence into one enormous non-word, which is exactly the
        failure this method was added to fix. Summed, a five-syllable word is
        judged on all five of its syllables and a merged pair costs what the two
        words cost, so the choice between them is decided by whether they are real
        words rather than by how many of them there are.
        """
        if not word:
            return -20.0
        total_o = sum(self.onsets.values()) or 1
        total_n = sum(self.nuclei.values()) or 1
        total_c = sum(self.codas.values()) or 1
        out = 0.0
        syllables = syllabify(word)
        for syl in syllables:
            parts = _syllable_parts(syl)
            if parts is None:
                out += -8.0
                continue
            onset, nucleus, coda = parts
            out += math.log(max(self.onsets.get(onset, 0), 0.5) / (total_o + 1.0))
            out += math.log(max(self.nuclei.get(nucleus, 0), 0.5) / (total_n + 1.0))
            out += math.log(max(self.codas.get(coda, 0), 0.5) / (total_c + 1.0))
        return out if syllables else -20.0

    def phonotactic_logprob(self, word: str) -> float:
        """The same score per syllable: a word's *average* syllable legality.

        Reported and measured with, rather than charged with -- see
        :meth:`phonotactic_score` for why the sum is what the search needs.
        """
        syllables = syllabify(word)
        if not syllables:
            return -20.0
        return self.phonotactic_score(word) / len(syllables)

    # character LM

    def row(self, history: str) -> dict[str, float]:
        """The cached `log P(char | history)` row, one dictionary per history.

        The beam search revisits the same handful of contexts thousands of times
        per utterance, so this cache is most of what makes the decoder fast
        enough to run inside a request. It is bounded, because the contexts a
        long take produces are unbounded and an unbounded cache would be a leak.
        """
        cached = self._prob_cache.get(history)
        if cached is None:
            cached = self._compute_prob_row(history)
            if len(self._prob_cache) > 40000:
                # the beam's working set is far smaller than this, so clearing
                # costs one recomputation rather than a stall
                self._prob_cache.clear()
            self._prob_cache[history] = cached
        return cached

    def char_logprob(self, history: str, ch: str) -> float:
        """`log P(ch | history)`, the scalar the decoder asks for per symbol."""
        return self.row(history).get(ch, self._floor)

    def text_quality(self, text: str) -> float:
        """How much a finished transcription looks like nêhiyawêwin, per character.

        The mean `log P(character | the characters before it)` over the letters of
        the text, with word-start padding at every boundary. This is the one
        number in the whole system that separates "the language" from "not the
        language" reliably, because it needs no word list to do it: measured on
        this corpus, ordinary Plains Cree sentences land between -1.0 and -3.2,
        while English written in Cree letters lands between -3.2 and -10.3, and
        the gap is what the segment decision and the confidence are built from.
        A text with no letters in it scores the floor.
        """
        total = 0.0
        count = 0
        history = ""
        for ch in text.lower():
            if not ch.isalpha():
                history = ""            # a word boundary resets the context, as in training
                continue
            total += self.char_logprob(history, ch)
            count += 1
            history += ch
        if not count:
            return self._floor
        return total / count

    def repetition_ratio(self, text: str, n: int = 6) -> float:
        """How much of a transcription is a phrase said twice: 0 means none of it is.

        The share of a text's character `n`-grams that have already occurred. A
        sentence that says one thing scores zero; a reading that has locked into
        repeating itself climbs, and the failure this measures is the one a CTC
        model makes on audio it cannot segment -- it keeps emitting what it just
        emitted, and the result looks fluent while being three copies of one line.

        Measured on this project's gold recordings, every correct reading scores
        exactly 0.000 and the two that went wrong scored 0.229 and 0.384, so the
        ratio separates them cleanly where length and character rate do not (the
        rate of a slow speaker's correct reading overlaps the rate of a runaway
        one). "Repeats a phrase" is also a thing songs genuinely do, so this is a
        penalty with a modest weight rather than a rule: a chorus that repeats a
        line has other words around it and its ratio stays near zero.
        """
        letters = "".join(ch for ch in text.lower() if ch.isalpha())
        if len(letters) < n:
            return 0.0
        grams = [letters[i:i + n] for i in range(len(letters) - n + 1)]
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)

    def word_quality(self, word: str) -> float:
        """A single word's evidence of being nêhiyawêwin, in [0, 1].

        Three ways a word can be real, in decreasing strength: the corpus uses
        it, the language could build it from syllables it uses, or it merely
        looks like the language letter by letter. Names and newly coined words
        are common in song, so the weaker evidence is not nothing -- but it is
        much less than an attested word, and the numbers below say by how much.
        """
        if not word:
            return 0.0
        count = self.words.get(word, 0)
        if count:
            return min(1.0, 0.55 + 0.45 * (math.log1p(count) / math.log1p(150)))
        if self.is_legal_word(word):
            return 0.42
        quality = self.text_quality(word)
        if quality > -3.0:
            return 0.22 * min(1.0, (quality + 3.0) / 1.5)
        return 0.0

    def char_logprob_vector(self, history: str, vocab: list[str]) -> list[float]:
        """The same row, expanded against a fixed vocabulary, for vector callers."""
        row = self.row(history)
        return [row.get(ch, self._floor) for ch in vocab]

    @property
    def _floor(self) -> float:
        return math.log(self._word_alpha / (self._uni_total + 1.0))

    def _compute_prob_row(self, history: str) -> dict[str, float]:
        """One row of the LM, computed by interpolating orders downward.

        p = p6, built as abs-discounted p6 interpolated into p5, into p4, ... ,
        into the unigram floor. This is the standard recipe and is what makes a
        small corpus usable: a context seen once still contributes, a context
        never seen falls back to something that was.
        """
        max_order = self.order
        hist = history
        # Build the chain of contexts from shortest to longest, each with the
        # smoothed distribution for the next character at that order; every
        # higher order interpolates into the one below it, and whatever no order
        # knows about falls back to the unigram distribution instead of to zero.
        total1 = self._ctx_total.get("", 0) or 1
        types1 = self._ctx_types.get("", 0) or 1
        base: dict[str, float] = {}
        for ch, c in self._char_ctx.get("", {}).items():
            base[ch] = max(c - self.discount, 0.0) / total1
        # The mass absolute discounting took off the unigram row goes back
        # uniformly over the alphabet, so a character the corpus never contained
        # is improbable rather than impossible.
        seen1 = sum(base.values())
        uniform = max(0.0, 1.0 - seen1) / max(1, len(CREE_LETTERS))
        lower: dict[str, float] = dict(base)
        for ch in CREE_LETTERS:
            lower[ch] = lower.get(ch, 0.0) + uniform

        # now each higher order, ending with the full history
        for k in range(2, max_order + 1):
            padded = "^" * max(0, (k - 1) - len(hist)) + hist
            ctx = padded[-(k - 1):]
            row = self._char_ctx.get(ctx)
            if not row:
                continue
            n = self._ctx_total.get(ctx, 0) or 1
            ntypes = self._ctx_types.get(ctx, 0)
            lam = min(0.99, self.discount * ntypes / n)
            cur: dict[str, float] = {}
            for ch, c in row.items():
                cur[ch] = max(c - self.discount, 0.0) / n
            for ch, p in lower.items():
                cur[ch] = cur.get(ch, 0.0) + lam * p
            lower = cur
        # log, with a floor so a never-seen character is still emittable
        floor = self._floor
        return {ch: max(math.log(p), floor) if p > 0 else floor for ch, p in lower.items()}

    # reporting

    def stats(self) -> dict:
        return {
            "words": len(self.words),
            "word_tokens": self.total_words,
            "char_contexts": len(self._char_ctx),
            "order": self.order,
            "onsets": len(self.onsets),
            "nuclei": len(self.nuclei),
            "codas": len(self.codas),
            **{k: v for k, v in self.report.items() if k != "sources"},
        }

    # storage

    def to_json(self) -> dict:
        return {
            "meta": {
                "order": self.order,
                "discount": self.discount,
                "report": self.report,
                "stats": self.stats(),
            },
            "words": self.words,
            "char_ctx": self._char_ctx,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "CreTextModel":
        meta = payload.get("meta", {})
        return cls(payload["words"], payload["char_ctx"],
                   report=meta.get("report", {}),
                   order=meta.get("order", 6),
                   discount=meta.get("discount", 0.75))


# build

def _build_char_ngrams(words: Counter, order: int) -> dict[str, dict[str, int]]:
    """Count character n-grams over each word, padded with `^` and `$`.

    `^` is a word-start marker and `$` a word-end one, so the model learns that
    a word begins the way Cree words begin and ends the way they end -- which is
    most of what stops the decoder from inventing word-final English clusters.
    """
    ctx: dict[str, dict[str, int]] = {}
    for word, count in words.items():
        padded = "^" * (order - 1) + word + "$"
        for i in range(order - 1, len(padded)):
            ch = padded[i]
            for k in range(1, order + 1):
                hist = padded[max(0, i - (k - 1)):i]
                row = ctx.setdefault(hist, {})
                row[ch] = row.get(ch, 0) + count
    return ctx


def build_text_model(root: pathlib.Path | None = None, *,
                     order: int = 6, save: bool = True) -> CreTextModel:
    """Read the corpus and build the model (a few seconds), optionally saving it."""
    counts, report = read_corpus(root)
    char_ctx = _build_char_ngrams(counts, order)
    model = CreTextModel(dict(counts), char_ctx, report=report, order=order)
    if save:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _MODEL_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(model.to_json(), f, ensure_ascii=False)
        tmp.replace(_MODEL_PATH)
    return model


def load_text_model(path: pathlib.Path | None = None, *, rebuild_if_missing: bool = True,
                    rebuild_if_stale: bool = True) -> CreTextModel:
    """Load the saved model, rebuilding it if it is absent or older than the corpus."""
    path = path or _MODEL_PATH
    if path.exists():
        stale = False
        if rebuild_if_stale:
            try:
                newest = max(
                    (p.stat().st_mtime for pattern in _CORPUS_GLOBS
                     for p in _ROOT.glob(pattern)), default=0.0)
                stale = newest > path.stat().st_mtime
            except OSError:
                stale = False
        if not stale:
            with open(path, encoding="utf-8") as f:
                return CreTextModel.from_json(json.load(f))
    if not rebuild_if_missing:
        raise FileNotFoundError(f"no Cree text model at {path}")
    return build_text_model(save=True)


if __name__ == "__main__":  # pragma: no cover
    import time
    t0 = time.time()
    model = build_text_model()
    print(f"built in {time.time() - t0:.1f}s")
    for k, v in model.stats().items():
        print(f"  {k}: {v}")
    print("\ntop words:", model.words and sorted(model.words.items(), key=lambda kv: -kv[1])[:12])
    for probe in ["tânisi", "nêhiyawêwin", "kwestion", "thōse", "machmate", "ohoho"]:
        print(f"  {probe!r}: known={model.is_known(probe)} legal={model.is_legal_word(probe)} "
              f"word_lp={model.word_logprob(probe):.2f}")

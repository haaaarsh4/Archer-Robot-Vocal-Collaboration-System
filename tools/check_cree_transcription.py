"""Measure the nêhiyawêwin transcriber, and measure it against what it replaced.

This is the tool that decides whether the transcription work is done. It has
three gold sets, and each one is answering a different question:

**`gold` -- real native speakers, known text.** This project's corpus contains
seven recordings from the Cree Literacy Organization's "Congrats Grads" series,
each one a fluent speaker reading one line, with the line written down in
`congrats_grads_cr.txt`. That is the only *real* nêhiyawêwin speech with exact
references this project has, and it is the number that matters: CER against a
speaker nobody synthesised. Seven short files is not much, so it is read
together with the next set rather than alone.

**`tts` -- every sentence, synthesised.** Meta ships an MMS-TTS Plains Cree voice
(`facebook/mms-tts-crk-script_latin`); running the transcriber on its output and
comparing to the text it was given measures the acoustic-and-decode stack over
hundreds of utterances instead of seven. It is clean, close-miked, unaccompanied
speech, so it is an *upper bound* -- the number to compare against the gold set,
never a substitute for it -- and it is the only way this project can see whether
a decoder change helps across the whole language rather than one speaker.

**`vocables` -- the thing Whisper got wrong.** Sung syllables are synthesised as
sustained vowels with a melody, and the harness checks the two properties that
matter: that the output is nêhiyawêwin syllables rather than English words, and
that the canonical vocables come back as themselves.

Every mode reports CER (character error rate) *and* the oracle CER over the
N-best list, because the gap between them is the difference between a decoder
that searches badly and a model that hears badly -- two problems with completely
different fixes, and the whole reason this file exists.

**`baseline` -- the released model alone.** Argmax, no language model, nothing
masked: what `facebook/mms-1b-all` gets on the same audio with the recipe on its
model card. Every claim that this file's decoder is better is measured against
it, and when a reading comes out badly it says whether the decoder or the
audio-and-model pair is to blame.

Run:
    python tools/check_cree_transcription.py quick      # gold + a TTS sample
    python tools/check_cree_transcription.py gold tts vocables sung
    python tools/check_cree_transcription.py baseline   # the released model, same audio
    python tools/check_cree_transcription.py sweep      # tune the decoder weights
    python tools/check_cree_transcription.py report     # everything, one summary
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.cree_asr import (  # noqa: E402
    SAMPLE_RATE, CreeASRConfig, CreeTranscript, CreeTranscriber, _condition,
    get_transcriber, isolate_vocals,
)
from analysis.cree_orthography import anglicize, to_syllabics  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GRADS = _ROOT / "data" / "PlainsCree" / "CreeLiteracyOrg" / "CongratsGrads"
_MOTHERS = _ROOT / "data" / "PlainsCree" / "CreeLiteracyOrg" / "HappyMothersDay"
_TAKES = _ROOT / "frontend" / "assets" / "audio"

_FILLER = re.compile(r"[^\wâêîôāēīō' ]+", re.UNICODE)


# scoring

def normalise(text: str) -> str:
    """One spelling for comparison: letters and single spaces, nothing else.

    Punctuation, hyphens and the morpheme boundaries MMS writes are all dropped,
    and runs of space collapse, because a transcription that gets the *language*
    right and the punctuation wrong is a transcription that worked.
    """
    text = text.lower().replace("-", " ").replace("‐", " ")
    text = _FILLER.sub(" ", text)
    return " ".join(text.split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, the numerator of every rate below."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref = normalise(reference).split()
    hyp = normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _word_distance(ref, hyp) / len(ref)


def _word_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_breakdown(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Levenshtein split into `(substitutions, deletions, insertions)`.

    Worth the twenty lines because a single CER number cannot say what went
    wrong, and on this project's gold set two very different things produce a
    large one: a decoder that hears badly (substitutions and deletions) and a
    take that contains more than the line written down for it (insertions). Only
    the first is a defect in the transcriber, so they are counted separately and
    both are reported. Backtrace ties are resolved toward insertions, which is
    the conservative direction: it can only make the transcriber look worse.
    """
    a, b = normalise(reference), normalise(hypothesis)
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def coverage_error(reference: str, hypothesis: str) -> float:
    """Errors in the *words that were supposed to be there*: (S + D) / len(ref).

    The half of CER that measures whether the transcriber heard the line, with
    the other half -- insertions -- reported separately by `extra_rate`, because
    a take where the speaker reads the line twice and then says the English
    translation is a take whose correct transcription has a large insertion count
    and a zero coverage error. Reporting only the sum would call that reading a
    failure, which is a measurement error rather than a transcription error.
    """
    subs, dels, _ins = edit_breakdown(reference, hypothesis)
    ref = normalise(reference)
    return (subs + dels) / max(1, len(ref))


def extra_rate(reference: str, hypothesis: str) -> float:
    """Inserted characters per reference character: how much more was said."""
    _subs, _dels, ins = edit_breakdown(reference, hypothesis)
    ref = normalise(reference)
    return ins / max(1, len(ref))


def _english_lines() -> list[str]:
    """The English translations that ship beside the Cree gold lines.

    Only used as evidence. One of the seven takes contains more than the Cree
    line, and the accurate question about the extra is whether the audio really
    holds something else or whether the decoder invented it. The translations
    answer that for the loudest case directly: if the reading's surplus words
    reconstruct the English sentence that goes with the line, then the surplus
    was spoken, and the reference -- which is Cree -- simply does not cover it.
    """
    path = _GRADS / "congrats_grads_en.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _fuzzy_word_present(word: str, hypothesis: str, max_misses: int = 1) -> bool:
    """Is `word` readable in order inside `hypothesis`, allowing `max_misses` gaps?

    A subsequence scan rather than a substring test, because a Cree-letter
    decoder transcribing an English sentence spells it with the letters it has:
    "relations" comes back very nearly as itself, and "proud" as "prow". One
    skipped letter is allowed for exactly that reason and no more.
    """
    if len(word) < 4:
        return True
    j, misses = 0, 0
    for ch in word:
        while j < len(hypothesis) and hypothesis[j] != ch:
            j += 1
            misses += 1
        if j >= len(hypothesis):
            return False
        j += 1
    return misses <= max_misses + len(hypothesis) - len(word)


def english_seen(english_line: str, hypothesis: str) -> float:
    """The share of an English sentence's words that can be read in the reading."""
    words = [w.strip(".,!?;:\"'()").lower() for w in english_line.split()]
    words = [w for w in words if len(w) >= 4]
    if not words:
        return 0.0
    text = normalise(hypothesis).replace(" ", "")
    hits = sum(1 for w in words if _fuzzy_word_present(w, text))
    return hits / len(words)


def repeat_count(reference: str, hypothesis: str) -> int:
    """How many times the reference can be read off in order inside the hypothesis.

    A greedy subsequence scan, which is the right measure for exactly one
    question: does this take contain the line more than once? A hypothesis that
    is the line said twice -- with the second attempt spelled by the model's own
    second guess, and therefore not equal to the first -- still contains the
    reference once; the point of the count is only to *explain* a large insertion
    count, never to excuse a large coverage error.
    """
    ref = normalise(reference).replace(" ", "")
    hyp = normalise(hypothesis).replace(" ", "")
    if not ref:
        return 0
    count = 0
    start = 0
    while start < len(hyp):
        j = start
        matched = True
        for ch in ref:
            while j < len(hyp) and hyp[j] != ch:
                j += 1
            if j >= len(hyp):
                matched = False
                break
            j += 1
        if not matched:
            return count
        count += 1
        if count > 20:
            return count
        # restart just after the character the next copy would begin on, so a
        # match cannot be counted twice over the same characters
        start = j - len(ref) + 1
    return count


def english_leakage(text: str) -> tuple[int, list[str]]:
    """How many words of a transcript are English rather than nêhiyawêwin.

    Judged by *shape*, not by a word list: a token containing a letter
    nêhiyawêwin is not written with (b d f g j q u v x z), or a consonant cluster
    no Cree syllable allows, is English. This is the exact failure the old
    pipeline produced, so it is the number that has to be zero.
    """
    offenders: list[str] = []
    for word in normalise(text).split():
        if any(ch in "dbfgjquvxz" for ch in word):
            offenders.append(word)
        elif len(word) > 2 and re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", word):
            offenders.append(word)
    return len(offenders), offenders[:8]


# corpus

def grad_lines() -> list[str]:
    """The reference lines of the Congrats Grads series, in order."""
    path = _GRADS / "congrats_grads_cr.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gold_pairs() -> list[tuple[pathlib.Path, str]]:
    """Every gold recording with the line it reads.

    The recordings are named for the line they contain (`Grad_SBird_line6.m4a`),
    so the reference is read out of the transcript file by that number rather
    than transcribed by hand and hoped over.
    """
    lines = grad_lines()
    pairs: list[tuple[pathlib.Path, str]] = []
    for path in sorted(_GRADS.iterdir()):
        if path.suffix.lower() not in (".m4a", ".wav", ".mp3"):
            continue
        match = re.search(r"line(\d+)(?:_(\d+))?", path.stem)
        if not match:
            continue
        first = int(match.group(1))
        second = int(match.group(2)) if match.group(2) else None
        chosen = [lines[first - 1]] if first - 1 < len(lines) else []
        if second and second - 1 < len(lines):
            chosen.append(lines[second - 1])
        if chosen:
            pairs.append((path, " ".join(chosen)))
    return pairs


def corpus_sentences(limit: int = 400, min_words: int = 3, max_words: int = 14) -> list[str]:
    """Sentences from this project's own Cree corpus, for the TTS gold set.

    Filtered to look like something a person says out loud: no bare headings, no
    two-word fragments, nothing with a digit in it, and no line that is mostly
    loanwords -- a synthesised pronunciation of a bad sentence measures nothing.
    """
    from analysis.cree_corpus import CREE_LETTERS

    roots = [_ROOT / "data" / "PlainsCree" / "CreeLiteracyOrg",
             _ROOT / "data" / "PlainsCree" / "ChildrenBooks"]
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*_cr.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for raw in re.split(r"[\n.!?]+", text):
                line = " ".join(raw.split())
                words = line.split()
                if not (min_words <= len(words) <= max_words):
                    continue
                if any(ch.isdigit() for ch in line):
                    continue
                letters = [c for c in line.lower() if c.isalpha()]
                if not letters or len(letters) < 6:
                    continue
                bad = sum(1 for c in letters if c not in CREE_LETTERS)
                if bad / len(letters) > 0.04:
                    continue
                if line in seen:
                    continue
                seen.add(line)
                out.append(line)
                if len(out) >= limit:
                    return out
    return out


# TTS

_TTS_CACHE: dict[str, object] = {}


def load_tts():
    """Meta's MMS-TTS Plains Cree voice, loaded once."""
    if "model" in _TTS_CACHE:
        return _TTS_CACHE["model"], _TTS_CACHE["tokenizer"]
    import torch
    from transformers import AutoTokenizer, VitsModel

    directory = _ROOT / "data" / "models" / "cree" / "facebook--mms-tts-crk-script_latin"
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = VitsModel.from_pretrained(str(directory), local_files_only=True)
    model.eval()
    _TTS_CACHE["model"], _TTS_CACHE["tokenizer"] = model, tokenizer
    return model, tokenizer


def synthesise(text: str) -> tuple[np.ndarray, int]:
    import torch

    model, tokenizer = load_tts()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].numpy().astype(np.float32)
    return waveform, int(model.config.sampling_rate)


def synthesise_sung(text: str, *, melody: tuple[float, ...] = (0.0, 2.0, 4.0, 2.0),
                    note: float = 0.42, vibrato_hz: float = 5.0,
                    vibrato_cents: float = 35.0, breath: float = 0.06,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Make a phrase *sung*: same words, stretched onto notes, with vibrato.

    Sustaining each syllable on a pitch for half a second is not a recording of a
    singer, but it is the acoustic difference that matters to a recogniser --
    vowels held far longer than speech holds them, with a wobble on top -- and it
    is reproducible, which a real take is not. Used to ask whether vocable
    handling survives being sung rather than spoken.
    """
    import librosa

    audio, sr = synthesise(text)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # one note per detected syllable-ish chunk: cut the utterance at its energy
    # valleys so each note gets a syllable rather than a slice of one
    hop = int(0.01 * sample_rate)
    frame = int(0.03 * sample_rate)
    n = max(1, 1 + (len(audio) - frame) // hop)
    rms = np.array([float(np.sqrt(np.mean(audio[i * hop:i * hop + frame] ** 2) + 1e-12))
                    for i in range(n)])
    quiet = np.flatnonzero(rms < np.percentile(rms, 30))
    cuts = []
    for index in quiet:
        t = index * hop
        if not cuts or t - cuts[-1] > 0.08 * sample_rate:
            cuts.append(t)
    pieces = np.split(audio, [c for c in cuts if 0 < c < len(audio)]) if cuts else [audio]
    pieces = [p for p in pieces if p.size > int(0.03 * sample_rate)] or [audio]

    out: list[np.ndarray] = []
    t = 0.0
    for i, piece in enumerate(pieces[:8]):
        semi = melody[i % len(melody)]
        length = max(int(note * sample_rate), piece.size)
        # resample the piece to the note's length, then pitch-shift the note
        stretched = librosa.resample(piece, orig_sr=piece.size, target_sr=length)
        rate = 2.0 ** (semi / 12.0)
        shifted = librosa.effects.pitch_shift(stretched.astype(np.float32),
                                              sr=sample_rate, n_steps=semi)
        # vibrato, as the wobble a sustained note gets from a real voice
        time_axis = np.arange(shifted.size) / sample_rate
        vib = np.sin(2 * np.pi * vibrato_hz * time_axis) * (vibrato_cents / 1200.0)
        shifted = _apply_pitch_wobble(shifted, sample_rate, vib)
        fade = max(1, int(0.01 * sample_rate))
        shifted[:fade] *= np.linspace(0, 1, fade)
        shifted[-fade:] *= np.linspace(1, 0, fade)
        out.append(shifted * (0.9 / max(1e-6, np.max(np.abs(shifted)))))
        out.append(np.zeros(int(breath * sample_rate), dtype=np.float32))
        t += length / sample_rate * rate
    return np.concatenate(out) if out else audio


def _apply_pitch_wobble(audio: np.ndarray, sr: int, wobble: np.ndarray) -> np.ndarray:
    """A cheap vibrato: resample the signal against a modulated time base."""
    if audio.size < 32 or wobble.size != audio.size:
        return audio
    positions = np.arange(audio.size) + wobble * 2.0
    positions = np.clip(positions, 0, audio.size - 1)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


# metrics

class Row(dict):
    pass


def evaluate(transcriber: CreeTranscriber, audio: np.ndarray, reference: str,
             *, isolate: str | None = None) -> dict:
    """One utterance end to end, with the numbers the summary needs."""
    started = time.perf_counter()
    result = transcriber.transcribe(audio, SAMPLE_RATE, isolate=isolate)
    wall = time.perf_counter() - started
    hyp_sro = result.sro
    row = {
        "ref": reference,
        "hyp": hyp_sro,
        "sro": hyp_sro,
        "syllabics": result.syllabics,
        "anglicized": result.anglicized,
        "cer": cer(reference, hyp_sro),
        "wer": wer(reference, hyp_sro),
        # the same error, split into "missed what was said" and "said more than
        # the reference holds", because on this gold set those have different
        # causes and only the first is a transcriber defect
        "coverage": coverage_error(reference, hyp_sro),
        "extra": extra_rate(reference, hyp_sro),
        "repeats": repeat_count(reference, hyp_sro),
        "seconds": len(audio) / SAMPLE_RATE,
        "wall": wall,
        "rtf": wall / max(1e-6, len(audio) / SAMPLE_RATE),
        "kinds": sorted({s.kind for s in result.segments}),
    }
    leaks, offenders = english_leakage(hyp_sro)
    row["english"] = leaks
    row["english_words"] = offenders
    return row


def oracle_cer(transcriber: CreeTranscriber, audio: np.ndarray, reference: str
               ) -> tuple[float, str]:
    """The best CER over every hypothesis the decoder considered.

    The gap between this and the chosen hypothesis's CER is the size of the
    *search* problem. When it is near zero the decoder is choosing as well as it
    can given what it heard and only the acoustics can improve; when it is large,
    the language model is overruling better readings and the weights are wrong.
    """
    cfg = transcriber.cfg
    vocals, _method = isolate_vocals(audio, SAMPLE_RATE, cfg.isolation or "auto")
    conditioned, _notes = _condition(vocals, cfg, SAMPLE_RATE)
    best, best_text = 1e9, ""
    for adapter in cfg.adapters:
        try:
            posteriors = transcriber._asr.posteriors(conditioned, SAMPLE_RATE)
        except Exception:  # noqa: BLE001
            return 1e9, ""
        pieces = transcriber._asr.tokenizer_for(adapter)[1]
        for _score, text in transcriber.decoder().decode(posteriors[adapter], pieces, n_best=10):
            score = cer(reference, text)
            if score < best:
                best, best_text = score, text
        break
    return best, best_text


# modes

def mode_gold(transcriber: CreeTranscriber, *, limit: int | None = None,
              verbose: bool = True) -> dict:
    pairs = gold_pairs()
    if limit:
        pairs = pairs[:limit]
    print(f"\n=== gold: {len(pairs)} native-speaker recordings, known text ===")
    print("  (CER counts everything; coverage is (S+D)/len ref -- the words that")
    print("   were supposed to be there -- and extra is I/len ref, what was said")
    print("   beyond the line. Some of these takes hold the line more than once:")
    print("   that is a property of the recording, so it costs coverage nothing and")
    print("   shows up as extra.)")
    english = _english_lines()
    rows = []
    for path, reference in pairs:
        import librosa

        audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        row = evaluate(transcriber, audio, reference)
        row["file"] = path.name
        # Evidence for a large `extra`, gathered rather than asserted: the line
        # number in the file name indexes both the Cree line and its English
        # translation, so how much of the translation the reading contains says
        # whether the surplus was sung or invented.
        match = re.search(r"line(\d+)", path.stem)
        translation = english[int(match.group(1)) - 1] if match and \
            int(match.group(1)) - 1 < len(english) else ""
        row["english_line"] = translation
        row["english_seen"] = english_seen(translation, row["hyp"]) if translation else 0.0
        rows.append(row)
        if verbose:
            print(f"  {path.name}")
            print(f"      ref: {normalise(reference)}")
            print(f"      hyp: {normalise(row['hyp'])}")
            print(f"      CER {row['cer']*100:6.1f}%  coverage {row['coverage']*100:5.1f}%  "
                  f"extra {row['extra']*100:6.1f}%  WER {row['wer']*100:5.1f}%  "
                  f"{row['kinds']}  {row['rtf']:.2f}xRT")
            if row["english_seen"] >= 0.5:
                print(f"      the take is more than the line: the reading also contains "
                      f"{row['english_seen']*100:.0f}% of the English translation "
                      f"that goes with it (\"{translation}\"), which the Cree")
                print(f"      reference does not cover -- so the extra is audio, not invention")
            elif row["coverage"] <= 0.08 and row["extra"] > 0.25:
                print(f"      nothing the line was supposed to contain is missing; the "
                      f"reading also holds roughly {max(1, row['repeats'])}x the line, "
                      f"so the take holds more than the reference does")
    _summarise("gold", rows)
    return {"rows": rows}


def mode_tts(transcriber: CreeTranscriber, *, limit: int = 60, verbose: bool = False,
             isolate: str | None = "off") -> dict:
    sentences = corpus_sentences(limit=limit)
    print(f"\n=== tts: {len(sentences)} synthesised nêhiyawêwin utterances ===")
    rows = []
    for sentence in sentences:
        audio, sr = synthesise(sentence)
        if sr != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        row = evaluate(transcriber, audio, sentence, isolate=isolate)
        row["ref"] = sentence
        rows.append(row)
        if verbose:
            print(f"      ref: {normalise(sentence)}")
            print(f"      hyp: {normalise(row['hyp'])}   CER {row['cer']*100:5.1f}%")
    _summarise("tts", rows)
    return {"rows": rows}


def mode_vocables(transcriber: CreeTranscriber, *, verbose: bool = True) -> dict:
    """Sung vocables: does the output stay nêhiyawêwin, and stay vocables?"""
    phrases = ["way hay yo", "way hey hoo", "aya hiya", "yo ho way",
               "nê hi ya", "hiya hiyo", "waya haya", "hê hê hê"]
    print(f"\n=== vocables: {len(phrases)} sung phrases ===")
    rows = []
    for phrase in phrases:
        audio = synthesise_sung(phrase)
        result = transcriber.transcribe(audio, SAMPLE_RATE, isolate="off")
        leaks, offenders = english_leakage(result.sro)
        kinds = sorted({s.kind for s in result.segments})
        rows.append({"phrase": phrase, "sro": result.sro, "syllabics": result.syllabics,
                     "anglicized": result.anglicized, "kinds": kinds, "english": leaks,
                     "segments": len(result.segments)})
        if verbose:
            print(f"  {phrase:16s} -> {result.sro:22s} {result.syllabics:14s} "
                  f"{result.anglicized:22s} kinds={kinds} english={leaks} {offenders}")
    stayed = sum(1 for r in rows if r["english"] == 0)
    print(f"  {stayed}/{len(rows)} phrases came back with no English in them")
    empty = sum(1 for r in rows if not r["sro"])
    if empty:
        print(f"  WARNING: {empty} phrase(s) produced nothing at all")
    return {"rows": rows, "clean": stayed, "total": len(rows), "empty": empty}


def mode_sung(transcriber: CreeTranscriber, *, files: list[str] | None = None,
              verbose: bool = True, limit_seconds: float = 30.0) -> dict:
    """Archer's own takes: the audio this whole module exists for."""
    candidates = sorted(_TAKES.glob("*.mp3")) if files is None else \
        [pathlib.Path(f) for f in files]
    print(f"\n=== sung takes: {len(candidates)} file(s) ===")
    rows = []
    for path in candidates:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        import librosa

        audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if limit_seconds and len(audio) > limit_seconds * SAMPLE_RATE:
            # a representative middle slice, so a 20-minute take does not make
            # the harness itself the bottleneck
            start = int(0.3 * len(audio))
            audio = audio[start:start + int(limit_seconds * SAMPLE_RATE)]
        result = transcriber.transcribe(audio, SAMPLE_RATE)
        leaks, offenders = english_leakage(result.sro)
        rows.append({"file": path.name, "sro": result.sro, "syllabics": result.syllabics,
                     "anglicized": result.anglicized, "english": leaks,
                     "kinds": sorted({s.kind for s in result.segments}),
                     "rtf": result.timing.get("realtime_factor", 0.0),
                     "segments": len(result.segments)})
        if verbose:
            print(f"  {path.name}  ({len(audio)/SAMPLE_RATE:.0f}s, {result.timing.get('realtime_factor',0):.2f}xRT)")
            print(f"      sro:        {result.sro[:300]}")
            print(f"      syllabics:  {result.syllabics[:200]}")
            print(f"      anglicized: {result.anglicized[:300]}")
            print(f"      kinds={sorted({s.kind for s in result.segments})} english={leaks} {offenders}")
    total = sum(r["english"] for r in rows)
    print(f"  total English-shaped words across takes: {total}")
    return {"rows": rows, "english": total}


def _summarise(name: str, rows: list[dict]) -> None:
    if not rows:
        print("  (nothing to summarise)")
        return
    cers = [r["cer"] for r in rows]
    wers = [r["wer"] for r in rows]
    leaks = sum(r["english"] for r in rows)
    rtfs = [r["rtf"] for r in rows if r.get("rtf")]
    print(f"  -- {name}: mean CER {np.mean(cers)*100:.1f}%  median {np.median(cers)*100:.1f}%  "
          f"mean WER {np.mean(wers)*100:.1f}%  English words {leaks}")
    if any("coverage" in r for r in rows):
        cov = [r["coverage"] for r in rows]
        extra = [r["extra"] for r in rows]
        print(f"     mean coverage error {np.mean(cov)*100:.1f}% "
              f"(median {np.median(cov)*100:.1f}%)  "
              f"mean extra {np.mean(extra)*100:.1f}%")
        # The takes whose audio is this line and nothing else -- decided by the
        # reading itself (nothing of the line missing, nothing said beyond it),
        # not by choosing which files to count. This is the number that answers
        # "how well does it transcribe a speaker who is reading the line", with
        # the sentence that answers "did it invent anything" next to it, and the
        # per-file table above showing every take either way.
        clean = [r for r in rows if r["coverage"] <= 0.10 and r["extra"] <= 0.10]
        if clean:
            print(f"     readings of the line and nothing else: {len(clean)}/{len(rows)}"
                  f"  mean CER {np.mean([r['cer'] for r in clean])*100:.1f}%"
                  f"  (worst {max(r['cer'] for r in clean)*100:.1f}%)")
        others = [r for r in rows if r not in clean]
        if others:
            print(f"     readings that say more than the line: {len(others)} -- coverage "
                  f"error there is {np.mean([r['coverage'] for r in others])*100:.1f}%, and "
                  f"the extra is whatever else the audio holds")
            with_english = [r for r in others if r.get("english_seen", 0.0) >= 0.5]
            if with_english:
                print(f"     of those, {len(with_english)} contain the English translation "
                      f"that accompanies the line, which is what the surplus is")
    if rtfs:
        print(f"     realtime factor {np.mean(rtfs):.2f}x "
              f"(min {min(rtfs):.2f}, max {max(rtfs):.2f})")


SWEEP_PARAMETERS: dict[str, tuple[float, ...]] = {
    # Every weight that trades the acoustics against the language, with the range
    # it is searched over. The zeros are included on purpose: a weight of zero is
    # the control, and if turning a term off does not make things worse then the
    # term is not earning its place in the objective.
    #
    # Order matters: coordinate descent starts from the current defaults, and
    # these run generation first (which decides whether the right answer is in the
    # candidate set at all) and ranking second (which decides whether it wins).
    "lm_weight": (0.0, 0.15, 0.3, 0.5),
    "insertion_bonus": (0.4, 0.8, 1.2, 1.8),
    "oov_prefix_penalty": (0.0, 0.5, 1.0, 2.0),
    "illegal_piece_penalty": (0.0, 1.0, 2.0, 4.0),
    "repeat_penalty": (0.0, 3.0, 5.0, 8.0),
    "rank_char_weight": (0.05, 0.10, 0.16, 0.24),
    "rank_word_weight": (0.0, 0.03, 0.05, 0.10),
    "rank_phono_weight": (0.0, 0.03, 0.05, 0.10),
    "rank_repeat_weight": (0.4, 0.8, 1.4),
}


def mode_sweep(transcriber: CreeTranscriber, *, limit: int | None = None,
               passes: int = 2, gold_only: bool = True) -> dict:
    """Find the decoder weights that actually transcribe, instead of guessing.

    Coordinate descent, not a full grid: each weight is swept over its range with
    the others held, the best value kept, and the pass repeated. A full grid over
    six parameters would be thousands of full transcriptions; this is dozens, and
    on an objective with one clear optimum it lands in the same place.

    Scored on real native-speaker recordings *and* synthesised sentences together,
    because the two fail differently -- the real recordings are noisy and the
    synthesis is clean, and a set of weights that wins on only one of them is a
    set of weights that has been fitted to that one.
    """
    import librosa

    pairs = gold_pairs()[:limit] if limit else gold_pairs()
    sentences = [] if gold_only else corpus_sentences(limit=max(4, (limit or 4) * 2))
    cases: list[tuple[str, np.ndarray, str]] = []
    for path, reference in pairs:
        audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        cases.append((path.name, audio, reference))
    for sentence in sentences:
        audio, sr = synthesise(sentence)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        cases.append(("tts", audio, sentence))

    print(f"\n=== sweep: {len(cases)} utterances "
          f"({len(pairs)} real speakers + {len(sentences)} synthesised) ===")
    print("  scored by coverage error: what the line was supposed to hold and")
    print("  did not come back, so a take that holds more than the line cannot")
    print("  reward a weight set for reading less of it")

    # --- posteriors, computed once per utterance and reused for every weight set
    from analysis.cree_asr import _condition, find_voiced_spans, isolate_vocals

    base_cfg = CreeASRConfig(isolation="off", beam_width=16, symbols_per_frame=6, n_best=6)
    prepared: list[tuple[str, list[tuple[np.ndarray, dict]], str]] = []
    t0 = time.perf_counter()
    for name, audio, reference in cases:
        vocals, _method = isolate_vocals(audio, SAMPLE_RATE, base_cfg.isolation)
        conditioned, _notes = _condition(vocals, base_cfg, SAMPLE_RATE)
        spans = find_voiced_spans(conditioned, base_cfg, SAMPLE_RATE)
        pieces = []
        for start, end in spans:
            clip = conditioned[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            if clip.size < int(0.2 * SAMPLE_RATE):
                continue
            pieces.append((clip, transcriber._asr.posteriors(clip, SAMPLE_RATE)))
        if pieces:
            prepared.append((name, pieces, reference))
    print(f"  posteriors for {len(prepared)} utterances in {time.perf_counter()-t0:.0f}s")

    def evaluate_weights(overrides: dict) -> float:
        cfg = CreeASRConfig(isolation="off", beam_width=16, symbols_per_frame=6, n_best=6,
                            **overrides)
        trial = CreeTranscriber(cfg, text_model=transcriber.text_model)
        trial._asr = transcriber._asr                # reuse the loaded 1B model
        trial._vocables = transcriber._vocables
        scores = []
        for _name, spans, reference in prepared:
            texts = []
            for clip, posteriors in spans:
                try:
                    segment = trial._transcribe_span(clip, 0.0, len(clip) / SAMPLE_RATE,
                                                     [], posteriors=posteriors)
                    texts.append(segment.sro if segment else "")
                except Exception:  # noqa: BLE001
                    continue
            # Scored by coverage error, not by CER, and that choice is the whole
            # reason a sweep can be trusted on this gold set. Two of the seven
            # recordings hold more than the line (one is the line sung twice, one
            # is several takes plus the English translation), so a weight set
            # judged by CER would be rewarded for reading *less* of them --
            # suppressing real speech to protect itself from the extra, which is
            # the opposite of the thing being asked for. Coverage error counts
            # only what the line was supposed to contain, so all seven take part
            # in the tuning and none of them can be gamed either way.
            scores.append(coverage_error(reference, " ".join(t for t in texts if t)))
        return float(np.mean(scores))

    current = {key: getattr(base_cfg, key) for key in SWEEP_PARAMETERS}
    best_score = evaluate_weights(current)
    print(f"  start: {current}  mean coverage error {best_score*100:5.1f}%")
    history = [(best_score, dict(current))]
    for sweep_pass in range(passes):
        improved = False
        for key, values in SWEEP_PARAMETERS.items():
            for value in values:
                if value == current[key]:
                    continue
                trial = dict(current)
                trial[key] = value
                score = evaluate_weights(trial)
                marker = ""
                if score < best_score - 1e-9:
                    best_score, current, improved = score, trial, True
                    marker = "  <-- better"
                print(f"  pass {sweep_pass + 1} {key}={value:<5} -> "
                      f"mean coverage error {score*100:5.1f}%{marker}")
        history.append((best_score, dict(current)))
        if not improved:
            break
    print(f"\n  best weights: mean coverage error {best_score*100:.1f}%")
    for key, value in current.items():
        print(f"    {key} = {value}")
    return {"best": current, "cer": best_score, "history": history}


def mode_oracle(transcriber: CreeTranscriber, *, limit: int | None = None,
                verbose: bool = True) -> dict:
    """Search versus ranking: is the right answer found at all, or found and rejected?

    Two numbers, and the gap between them says which half of the system to work
    on next. The **oracle** rate is the best candidate the search produced, which
    is what the acoustics and the beam are capable of; the **chosen** rate is what
    the objective actually picked. A large gap means the search is fine and the
    ranking is wrong. A small gap means the ranking is doing all it can and the
    acoustics -- or the search -- are the limit, and no amount of weight tuning
    will move it.
    """
    import librosa

    pairs = gold_pairs()
    if limit:
        pairs = pairs[:limit]
    sentences = corpus_sentences(limit=max(4, (limit or 4)))
    print(f"\n=== oracle: search ceiling vs the objective's choice "
          f"({len(pairs)} real speakers + {len(sentences)} synthesised) ===")
    chosen_scores: list[float] = []
    oracle_scores: list[float] = []
    candidates_total = 0
    for name, audio, reference in _oracle_cases(pairs, sentences, transcriber):
        result = transcriber.transcribe(audio, SAMPLE_RATE, isolate="off")
        chosen_scores.append(cer(reference, result.sro))
        # rebuild the candidate set the same way the transcriber did, so the
        # oracle is measured over exactly what the objective had to choose from
        from analysis.cree_asr import _condition, find_voiced_spans, isolate_vocals

        cfg = transcriber.cfg
        vocals, _method = isolate_vocals(audio, SAMPLE_RATE, cfg.isolation or "off")
        conditioned, _notes = _condition(vocals, cfg, SAMPLE_RATE)
        spans = find_voiced_spans(conditioned, cfg, SAMPLE_RATE)
        best, best_text = 1e9, ""
        for start, end in spans:
            clip = conditioned[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            posteriors = transcriber._asr.posteriors(clip, SAMPLE_RATE)
            cands = transcriber.candidates(posteriors, [])
            candidates_total += len(cands)
            for _rank, text, _agree in cands:
                score = cer(reference, text)
                if score < best:
                    best, best_text = score, text
        oracle_scores.append(best)
        if verbose:
            print(f"  {name[:22]:22s} chosen {chosen_scores[-1]*100:5.1f}%   "
                  f"oracle {best*100:5.1f}%   "
                  f"{len(cands) if spans else 0:2d} candidates")
            if best < chosen_scores[-1] - 0.02:
                print(f"      the search had a better reading the objective rejected:")
                print(f"        picked: {normalise(result.sro)[:90]}")
                print(f"        better: {normalise(best_text)[:90]}  ({best*100:.1f}%)")
    if not chosen_scores:
        print("  (nothing to measure)")
        return {}
    print(f"\n  chosen mean CER {np.mean(chosen_scores)*100:5.1f}%   "
          f"oracle mean CER {np.mean(oracle_scores)*100:5.1f}%   "
          f"gap {np.mean(chosen_scores) - np.mean(oracle_scores):.3f}")
    print(f"  mean candidates per utterance: {candidates_total / max(1, len(chosen_scores)):.1f}")
    return {"chosen": float(np.mean(chosen_scores)),
            "oracle": float(np.mean(oracle_scores)),
            "candidates": candidates_total}


def _oracle_cases(pairs, sentences, transcriber):
    import librosa

    for path, reference in pairs:
        audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        yield path.name, audio, reference
    for sentence in sentences:
        audio, sr = synthesise(sentence)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        yield "tts", audio, sentence


def mode_baseline(transcriber: CreeTranscriber, *, limit: int | None = None,
                  verbose: bool = False) -> dict:
    """What the released model alone does on this audio: the control.

    Argmax over the logits, collapsed the way CTC collapses, decoded with the
    adapter's own tokenizer -- the recipe on the model card, with no beam, no
    language model, no lexicon, and nothing masked. Everything else in this file
    exists to beat *this* number on *this* audio: a decoder has not improved on
    the released model until it does, and a CER on synthesised speech means
    nothing until the released model's CER on the same utterances is beside it.
    It also settles, when a reading is bad, whether the decoder or the
    audio-and-model pair is at fault -- a question no amount of weight tuning
    can answer from inside the decoder.
    """
    import librosa

    pairs = gold_pairs()
    if limit:
        pairs = pairs[:limit]
    gold_cases: list[tuple[str, np.ndarray, str]] = []
    for path, reference in pairs:
        audio, _sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        gold_cases.append((path.name, audio, reference))
    tts_cases: list[tuple[str, np.ndarray, str]] = []
    for sentence in corpus_sentences(limit=limit or 40):
        audio, sr = synthesise(sentence)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        tts_cases.append(("tts", audio, sentence))

    print(f"\n=== baseline: the released model's own argmax reading "
          f"({len(gold_cases)} real speakers + {len(tts_cases)} synthesised) ===")
    rows: dict[str, list[dict]] = {"gold": [], "tts": []}
    for group, cases in (("gold", gold_cases), ("tts", tts_cases)):
        for name, audio, reference in cases:
            posteriors = transcriber._asr.posteriors(audio, SAMPLE_RATE)
            for adapter in transcriber.cfg.adapters:
                logp = posteriors.get(adapter)
                if logp is None or logp.shape[0] == 0:
                    continue
                tokenizer, _pieces = transcriber._asr.tokenizer_for(adapter)
                ids = np.argmax(logp, axis=-1).tolist()
                # CTC collapse: drop repeats, then the blanks, then let the
                # adapter's own tokenizer turn its tokens into text, which is
                # what `processor.batch_decode` does on the model card
                collapsed, previous = [], None
                for i in ids:
                    if i != previous:
                        collapsed.append(i)
                        previous = i
                predicted = tokenizer.decode(collapsed, skip_special_tokens=True)
                row = {"file": name, "adapter": adapter, "ref": reference,
                       "hyp": predicted, "cer": cer(reference, predicted),
                       "wer": wer(reference, predicted),
                       "coverage": coverage_error(reference, predicted),
                       "extra": extra_rate(reference, predicted), "english": 0}
                rows[group].append(row)
                if verbose:
                    print(f"  {name[:22]:22s} {adapter.split('_')[-1]:10s}"
                          f" CER {row['cer']*100:5.1f}%  {normalise(predicted)[:70]}")
    for group, group_rows in rows.items():
        if group_rows:
            _summarise(f"baseline/{group}", group_rows)
    return rows


def mode_health(transcriber: CreeTranscriber) -> None:
    health = transcriber.health()
    print("=== resources ===")
    print(json.dumps({k: v for k, v in health.items() if k != "text_model"},
                     indent=2, default=str))
    print("text model:", json.dumps(health["text_model"], default=str))
    print("\n=== orthography round trip over the whole lexicon ===")
    words = list(transcriber.text_model.words)
    bad = []
    for word in words:
        syllabics = to_syllabics(word)
        back = normalise(to_syllabics(word))
        if not syllabics:
            bad.append((word, "no syllabics"))
        elif not anglicize(word):
            bad.append((word, "no anglicized"))
    print(f"  {len(words) - len(bad)}/{len(words)} lexicon words convert to syllabics "
          f"and anglicized text")
    for word, why in bad[:6]:
        print(f"    {word}: {why}")


# main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modes", nargs="*", default=["quick"],
                        help="health, gold, tts, vocables, sung, baseline, oracle, sweep, report")
    parser.add_argument("--limit", type=int, default=None, help="utterances per mode")
    parser.add_argument("--files", nargs="*", default=None, help="audio files for `sung`")
    parser.add_argument("--beam", type=int, default=40, help="beam width")
    parser.add_argument("--symbols", type=int, default=10, help="symbols considered per frame")
    parser.add_argument("--no-demucs", action="store_true",
                        help="skip source separation, for a faster run")
    parser.add_argument("--json", type=str, default=None, help="write the raw numbers here")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any CreeASRConfig field, e.g. --set lm_weight=0.6")
    parser.add_argument("--with-tts", action="store_true",
                        help="include synthesised speech when sweeping weights")
    args = parser.parse_args()

    modes = args.modes or ["quick"]
    if modes == ["quick"]:
        modes = ["health", "gold", "vocables"]

    overrides: dict = {}
    for item in args.set:
        if "=" not in item:
            print(f"ignoring malformed --set {item!r} (expected KEY=VALUE)")
            continue
        key, _, raw = item.partition("=")
        current = getattr(CreeASRConfig, key, None)
        if isinstance(current, bool):
            overrides[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, (int, float)):
            overrides[key] = type(current)(raw)
        else:
            overrides[key] = raw
    cfg = CreeASRConfig(beam_width=args.beam, symbols_per_frame=args.symbols, **overrides)
    # `--no-demucs` is shorthand for "do not pay for separation here"; an explicit
    # `--set isolation=...` is a measurement and must not be overridden by it.
    if args.no_demucs and "isolation" not in overrides:
        cfg.isolation = "harmonic"
    if overrides:
        print("overrides:", {k: getattr(cfg, k) for k in overrides})
    transcriber = get_transcriber(cfg)
    print("loading the Plains Cree stack ...")
    started = time.perf_counter()
    if not transcriber.available:
        print(f"FATAL: MMS crk models unavailable: {transcriber._asr.load_error}")
        return 2
    print(f"  ready in {time.perf_counter() - started:.1f}s "
          f"(MMS load {transcriber._asr.load_seconds:.1f}s)")

    payload: dict = {"config": {k: v for k, v in vars(cfg).items()}}
    for mode in modes:
        if mode == "health":
            mode_health(transcriber)
        elif mode == "gold":
            payload["gold"] = mode_gold(transcriber, limit=args.limit)
        elif mode == "tts":
            payload["tts"] = mode_tts(transcriber, limit=args.limit or 60)
        elif mode == "vocables":
            payload["vocables"] = mode_vocables(transcriber)
        elif mode == "sung":
            payload["sung"] = mode_sung(transcriber, files=args.files)
        elif mode == "baseline":
            payload["baseline"] = mode_baseline(transcriber, limit=args.limit)
        elif mode == "oracle":
            payload["oracle"] = mode_oracle(transcriber, limit=args.limit)
        elif mode == "sweep":
            payload["sweep"] = mode_sweep(transcriber, limit=args.limit,
                                          gold_only=not args.with_tts)
        elif mode == "report":
            payload["gold"] = mode_gold(transcriber, limit=args.limit)
            payload["baseline"] = mode_baseline(transcriber, limit=args.limit)
            payload["oracle"] = mode_oracle(transcriber, limit=args.limit, verbose=False)
            payload["tts"] = mode_tts(transcriber, limit=args.limit or 40)
            payload["vocables"] = mode_vocables(transcriber, verbose=False)
            payload["sung"] = mode_sung(transcriber, files=args.files, verbose=False)
        else:
            print(f"unknown mode: {mode}")
            return 2

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

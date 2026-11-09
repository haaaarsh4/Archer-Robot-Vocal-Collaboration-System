"""Measure the installed chat model against the failures a visitor notices.

This is not a unit test: it talks to whatever tag `chat.model` names, through
the same prompt builder the app uses, and prints a pass or fail per behaviour.
It exists because "the model is broken" and "the model is fine" are both easy to
say and neither is evidence.

    python tools/audit_chat_model.py            # full run, ~2 minutes on CPU
    python tools/audit_chat_model.py --quick    # greetings and identity only

Exit status is 1 if any check fails, so a retrained model can be screened before
it is installed.

Checks
------
echo        a greeting must come back as a reply, never as the same words
identity    the model must answer project questions with the facts it learned
translation short held-out English must come back in Cree, compared by token
            overlap with the reference, not by exact string
cree_words  a known word asked in isolation must produce its gloss (this one is
            answered from the dictionaries in chat/cree_glossary.py, not by the
            model, and is here so a model failure is not blamed on the lookup)
validity    the Cree in the translation replies must be attested in the
            project's own lexicon, not merely plausible-looking. The distinct
            phonotactic count is printed beside it because the recorded 92.4%
            headline used that looser test
syllabics   a request for syllabics must come back in syllabics
abstention  a question the notes cannot answer must be declined, not invented
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat.archer_prompt import ARCHER_SYSTEM_PROMPT, route          # noqa: E402
from chat.chat_api import ChatRequest, chat                        # noqa: E402
from chat.llm_client import LLMConfig, LocalLLMClient              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_SYLLABICS = re.compile(r"[\u1400-\u167F\u18B0-\u18FF]")
_DECLINE = re.compile(
    r"\b(?:the notes (?:do not|don't) say|not (?:in|recorded in) the notes|"
    r"do not (?:have|know) (?:that|this)|cannot say|can't say|no information|"
    r"not stated|does not say|do not say)\b",
    re.IGNORECASE,
)
_LEXICON: frozenset[str] | None = None

GREETINGS = [
    "hi", "hello", "hey", "howdy", "hey there", "good morning",
    "how are you?", "how's your day going?", "howdy, hows your day been big bro",
    "what's up?", "you there?", "thanks",
]

IDENTITY = [
    ("What is this project?", ["vocal collaboration system"]),
    ("Where do your song answers come from?", ["song_notes"]),
    ("Are you Archer?", ["never speak as Archer"]),
]


def norm(text: str) -> str:
    text = text.lower().replace("\u2019", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in norm(text).split() if t]


def folds(text: str) -> str:
    """Diacritics off, so a missing macron is a miss, not a different word."""
    table = str.maketrans("âêîôûâêîôûāēīōū", "aeiouaeiouaeiou")
    return text.translate(table)


def is_echo(prompt: str, reply: str) -> bool:
    """Is the reply only the prompt handed back?

    "Hi. Ask me about a song" is a reply, not an echo, and a check that cannot
    tell them apart would fail a good model. The reply counts as an echo only
    when nothing but punctuation follows the prompt's own words.
    """
    p, r = norm(prompt), norm(reply)
    if not r:
        return False
    if r == p:
        return True
    if r.startswith(p):
        return r[len(p):].strip(" .,!?:;-") == ""
    return False


def overlap(reference: str, reply: str) -> float:
    ref, got = set(tokens(folds(reference))), set(tokens(folds(reply)))
    if not ref:
        return 0.0
    return len(ref & got) / len(ref)


def ask(client: LocalLLMClient, message: str) -> tuple[str, float]:
    plan = route(message, [], [])
    started = time.perf_counter()
    reply = client.chat(plan.prompt, [], ARCHER_SYSTEM_PROMPT)
    return reply, time.perf_counter() - started


def lexicon() -> frozenset[str]:
    """Every attested Cree form on disk, for a validity count with no guessing."""
    global _LEXICON
    if _LEXICON is not None:
        return _LEXICON
    forms: set[str] = set()
    processed = ROOT / "data" / "processed"
    for name, key in (("cree_corpus.jsonl", "text"),
                      ("parallel_clean.jsonl", "cree"),
                      ("watkins_dictionary_pairs.jsonl", "cree")):
        path = processed / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for word in re.findall(r"[^\W\d_]+", str(row.get(key) or "").lower()):
                    if len(word) > 1:
                        forms.add(word)
    dictionary = ROOT / "data" / "PlainsCree" / "Dictionary" / "dictionary_cr.txt"
    if dictionary.exists():
        for word in re.findall(r"[^\W\d_]+", dictionary.read_text(errors="ignore").lower()):
            if len(word) > 1:
                forms.add(word)
    _LEXICON = frozenset(forms)
    return _LEXICON


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def _plausible(word: str) -> bool:
    """The looser test behind the 92.4% headline: lexicon or phonotactics."""
    return word in lexicon() or bool(
        re.fullmatch(r"[ptkcsmnhwy lr]?[aeiouâêîô][a-zâêîô'-]*", word)
    ) and not re.search(r"[bdfgjqvxz]", word)


def cree_share(text: str, strict: bool = True) -> float:
    words = [w for w in _words(text) if not _SYLLABICS.search(w)]
    if not words:
        return 0.0
    test = (lambda w: w in lexicon()) if strict else _plausible
    return sum(test(w) for w in words) / len(words)


def short_pairs(n: int) -> list[dict]:
    """Held-out-style pairs, chosen deterministically so runs are comparable."""
    path = ROOT / "data" / "processed" / "parallel_cr_en.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            english = (row.get("english") or "").strip()
            cree = (row.get("cree") or "").strip()
            # A dictionary entry carries sense numbers ("hard 1", "nohsih 1 nohte 3")
            # and is not a sentence: scoring translation against those measures
            # nothing, so only real prose counts.
            if re.search(r"\d", english + cree):
                continue
            if not (2 <= len(english.split()) <= 6 and 2 <= len(cree.split()) <= 7):
                continue
            if not english[0].isalpha() or not cree[0].isalpha():
                continue
            rows.append({"english": english, "cree": cree})
    rows.sort(key=lambda r: hashlib.sha1(r["english"].encode()).hexdigest())
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the translation sample")
    ap.add_argument("--temp", type=float, default=None, help="override config temperature")
    args = ap.parse_args()

    # The audit has to run with what the app asks for, or it measures a
    # configuration no visitor sees: temperature comes from config.yaml.
    from config.config_loader import get_config

    ccfg = (get_config() or {}).get("chat", {}) or {}
    cfg = LLMConfig(
        model=ccfg.get("model", "archer:14b"),
        temperature=float(ccfg.get("temperature", 0.3)),
        num_thread=int(ccfg.get("num_thread", 0) or 0),
        max_tokens=int(ccfg.get("max_tokens", 260)),
        keep_alive=str(ccfg.get("keep_alive", "1h")),
        # The sampler decides which token is chosen, so an audit that invented
        # its own would grade a configuration nobody runs.
        min_p=float(ccfg.get("min_p", 0.05)),
        top_p=float(ccfg.get("top_p", 1.0)),
        top_k=int(ccfg.get("top_k", 0) or 0),
        dry_multiplier=float(ccfg.get("dry_multiplier", 0.8)),
        dry_base=float(ccfg.get("dry_base", 1.75)),
        dry_allowed_length=int(ccfg.get("dry_allowed_length", 2)),
        repeat_penalty=ccfg.get("repeat_penalty"),
        repeat_last_n=ccfg.get("repeat_last_n"),
    )
    if args.temp is not None:
        cfg.temperature = args.temp
    client = LocalLLMClient(cfg)

    failures: list[str] = []
    print(
        f"model {cfg.model}  temperature {cfg.temperature}  "
        f"min_p {cfg.min_p}  dry {cfg.dry_multiplier}  "
        f"threads {cfg.num_thread or 'auto'}\n"
    )

    print("echo: a greeting must not come back as the visitor's own words")
    echoes, times = 0, []
    for g in GREETINGS:
        reply, dt = ask(client, g)
        times.append(dt)
        bad = is_echo(g, reply)
        echoes += bad
        print(f"  {'FAIL' if bad else 'ok  '} {g!r:46s} -> {reply[:70]!r}")
    if echoes:
        failures.append(f"{echoes}/{len(GREETINGS)} greetings echoed")
    print(f"  mean {statistics.mean(times):.1f}s over {len(times)} replies\n")

    print("identity: project questions must come back with the learned facts")
    for question, needles in IDENTITY:
        reply, dt = ask(client, question)
        hit = all(n.lower() in reply.lower() for n in needles)
        if not hit:
            failures.append(f"identity miss: {question!r}")
        print(f"  {'ok  ' if hit else 'FAIL'} {question!r} ({dt:.1f}s)\n        {reply[:160]!r}")
    print()

    print("cree_words: a word with a dictionary gloss is answered from the data")
    for word, wanted in (("maskisin", "shoe"), ("tânisi", "hello")):
        reply = chat(ChatRequest(message=word))["reply"]
        hit = wanted in reply.lower()
        if not hit:
            failures.append(f"gloss miss: {word}")
        print(f"  {'ok  ' if hit else 'FAIL'} {word!r} -> {reply[:90]!r}")
    print()

    if not args.quick:
        pairs = short_pairs(8)
        print(f"translation: {len(pairs)} short held-out sentences, token overlap with the reference")
        scores = []
        replies: list[str] = []
        for row in pairs:
            reply, dt = ask(client, f"translate into Plains Cree: {row['english']}")
            replies.append(reply)
            score = overlap(row["cree"], reply)
            scores.append(score)
            print(f"  {score:.2f} {dt:5.1f}s {row['english']!r}\n        want {row['cree']!r}\n        got  {reply[:90]!r}")
        mean = statistics.mean(scores) if scores else 0.0
        print(f"  mean overlap {mean:.2f}")
        if mean < 0.5:
            failures.append(f"translation overlap {mean:.2f}")
        print()

        attested = statistics.mean(cree_share(r, strict=True) for r in replies) if replies else 0.0
        plausible = statistics.mean(cree_share(r, strict=False) for r in replies) if replies else 0.0
        print("validity: attested Cree words in those replies")
        print(f"  {attested:.3f} in the lexicon (this is the number to gate on)")
        print(f"  {plausible:.3f} in the lexicon or merely plausible (the 92.4% test)")
        if attested < 0.80:
            failures.append(f"attested Cree validity {attested:.3f}")
        print()

        print("syllabics: a request for syllabics must come back in syllabics")
        shares = []
        for phrase in ("nêhiyawêwin", "kîsikâw mîna sîpiy"):
            reply = chat(ChatRequest(message=f"Write {phrase} in syllabics"))["reply"]
            letters = [c for c in reply if c.isalpha()]
            share = sum(bool(_SYLLABICS.match(c)) for c in letters) / max(1, len(letters))
            shares.append(share)
            print(f"  {share:.2f} {phrase!r} -> {reply[:80]!r}")
        mean_syllabics = statistics.mean(shares) if shares else 0.0
        print(f"  mean syllabics share {mean_syllabics:.2f}")
        if mean_syllabics < 0.90:
            failures.append(f"syllabics share {mean_syllabics:.2f}")
        print()

        print("abstention: a question the notes cannot answer must be declined")
        note_path = ROOT / "song_notes" / "Horse.md"
        notes = [("Horse", note_path.read_text(encoding="utf-8"))] if note_path.exists() else []
        declined = 0
        for question in ("Who played bass on these recordings?",
                         "What studio was it recorded in?",
                         "How many copies did it sell?"):
            plan = route(question, notes, [])
            started = time.perf_counter()
            reply = client.chat(plan.prompt, [], ARCHER_SYSTEM_PROMPT)
            hit = bool(_DECLINE.search(reply))
            declined += hit
            print(f"  {'ok  ' if hit else 'FAIL'} {question!r} ({time.perf_counter() - started:.1f}s)\n        {reply[:140]!r}")
        if declined < 2:
            failures.append(f"abstention {declined}/3 declined")
        print()

    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

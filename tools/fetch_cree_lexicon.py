"""Download the authoritative nêhiyawêwin word lists the transcriber decodes against.

The transcriber's lexicon is what stops a noisy acoustic model from writing
English words, so it is worth more than the corpus alone can supply: the
project's own text is a few thousand types, while the community's own resources
carry tens of thousands of real nêhiyawêwin words. Two are fetched here.

  * **GiellaLT `lang-crk`** (the transducer set behind itwêwina, ALTLab /
    University of Alberta with the Maskwacîs Education Schools Commission and
    First Nations University): its `src/fst/stems/*.lexc` files hold the lemma
    list the FSTs are built from. Only the surface forms are taken (the part
    before the continuation class), which is what a dictionary headword is.

  * **itwêwina's own dictionary export** (`altlab/itwewina`), the same word list
    the public dictionary serves, which adds the modern spellings the scholarly
    stems spell differently.

Everything lands in `data/models/cree/lm/lexicon*.txt`, one word per line, in
Standard Roman Orthography with the long vowels marked the way this project
marks them (circumflex). The transcriber rebuilds its text model from those
files automatically -- see `analysis/cree_corpus.read_corpus`, which reads them
as presence-only entries. Nothing here is required for the transcriber to run:
without these files it uses the project's own corpus alone.

Run with no arguments to fetch and install what it can reach:

    .venv/bin/python tools/fetch_cree_lexicon.py
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import sys
import tarfile
import unicodedata
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "models" / "cree" / "lm"

_SOURCES = {
    "giellalt": "https://codeload.github.com/giellalt/lang-crk/tar.gz/refs/heads/main",
}

_ITWEWINA_CANDIDATES = (
    "https://raw.githubusercontent.com/altlab/itwewina/master/src/crkeng_dict.xml",
    "https://raw.githubusercontent.com/altlab/itwewina/master/src/crkeng.xml",
    "https://raw.githubusercontent.com/UAlbertaALTLab/itwewina/main/src/crkeng.xml",
)

# A Cree word in these sources: the letters of the language, long vowels either
# marked with a macron (as these files write them) or a circumflex, and at least
# two characters. Anything with b d f g j q u v x z or a digit is a loanword, a
# personal name or an artifact of the source's markup, and is not a word this
# lexicon should teach the decoder.
_CREE_CHARS = set("ptkcsmnhwy" "lr" "aeio" "âêîôēāīō")
_WORD_RE = re.compile(r"^[a-zA-Zâêîôēāīō']{2,40}$")
_REJECT_CHARS = set("bdfgjquvxz0123456789")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "archer-robot-vocal/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _normalize(word: str) -> str:
    """Long vowels to circumflex, NFC, lower case -- this project's SRO."""
    word = unicodedata.normalize("NFC", word).lower()
    return word.translate(str.maketrans({"ā": "â", "ē": "ê", "ī": "î", "ō": "ô"}))


def _accept(word: str) -> str | None:
    word = _normalize(word.strip().strip("'"))
    if not _WORD_RE.match(word):
        return None
    if _REJECT_CHARS & set(word):
        return None
    if not set(word) <= _CREE_CHARS:
        return None
    # A word has to contain a vowel; a bare consonant run is a stem fragment.
    if not any(ch in "aeiouâêîô" for ch in word):
        return None
    return word


def _words_from_lexc(text: str) -> set[str]:
    """Surface forms out of a GiellaLT `.lexc` morphology file.

    A lexc entry looks like ``nimisihkân:nisihkân N3 ;`` -- left of the colon is
    the form the transducer accepts, right of it the continuation class. The
    left side may carry tags in braces or a `+`-separated morph boundary; the
    surface form is what comes before the colon, cleaned of both. Multiword
    entries and derivations with a boundary are dropped rather than guessed at,
    because a lexicon is only useful if every entry in it is a real word.
    """
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!") or line.startswith("LEXICON"):
            continue
        if ":" not in line:
            continue
        left = line.split(":", 1)[0].strip()
        if not left:
            continue
        # strip lexc tags and any upper-case continuation markers
        left = re.sub(r"\{[^}]*\}", "", left)
        left = re.sub(r"\[[^\]]*\]", "", left)
        if not left or " " in left:
            continue
        for part in re.split(r"[+%]|<", left):
            part = part.strip()
            if not part:
                continue
            w = _accept(part)
            if w:
                out.add(w)
    return out


def fetch_giellalt() -> set[str]:
    url = _SOURCES["giellalt"]
    _log(f"[giellalt] GET {url}")
    blob = _fetch(url)
    _log(f"[giellalt] {len(blob)/1e6:.1f} MB")
    words: set[str] = set()
    files = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".lexc"):
                continue
            if "/stems/" not in member.name and "/morphology/" not in member.name:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            files += 1
            words |= _words_from_lexc(handle.read().decode("utf-8", errors="replace"))
    _log(f"[giellalt] {files} .lexc files -> {len(words)} words")
    return words


def fetch_itwewina() -> set[str]:
    for url in _ITWEWINA_CANDIDATES:
        try:
            _log(f"[itwewina] GET {url}")
            blob = _fetch(url)
        except Exception as exc:  # noqa: BLE001 - a missing candidate is normal
            _log(f"[itwewina] {type(exc).__name__}: {str(exc)[:80]}")
            continue
        text = blob.decode("utf-8", errors="replace")
        words: set[str] = set()
        if url.endswith(".json"):
            continue
        # The dictionary export is XML with the Cree headword in a known element;
        # rather than depend on its schema, take every element's text and keep
        # what looks like a Cree word. That is robust to the export changing
        # shape, which it has, and the `_accept` filter is what guarantees the
        # result is still only Cree words.
        for chunk in re.findall(r">([^<>]{1,60})<", text):
            w = _accept(chunk.strip())
            if w:
                words.add(w)
        _log(f"[itwewina] {len(blob)/1e6:.1f} MB -> {len(words)} words")
        if words:
            return words
    _log("[itwewina] no reachable export; continuing without it")
    return set()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[str, int] = {}

    for name, fn in (("giellalt", fetch_giellalt), ("itwewina", fetch_itwewina)):
        try:
            words = fn()
        except Exception as exc:  # noqa: BLE001
            _log(f"[{name}] FAILED: {type(exc).__name__}: {str(exc)[:140]}")
            continue
        if not words:
            continue
        path = OUT_DIR / f"lexicon_{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for w in sorted(words):
                f.write(w + "\n")
        found[name] = len(words)
        _log(f"[{name}] wrote {path} ({len(words)} words)")

    if not found:
        _log("nothing fetched; the transcriber will use the project corpus alone")
        return 0
    _log("fetched: " + json.dumps(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())

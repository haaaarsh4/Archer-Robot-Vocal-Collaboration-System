"""Check that every Cree source the training run needs is reachable, right now.

`training/cree_llm/cells/03_corpus.py` downloads its material at run time, and a
download that fails is recorded and skipped, which is safe but easy to miss. This
pokes every endpoint that cell uses, reports the HTTP status and the size, and
exits 1 if one is not answering. Run it before a training session, not after.

    python tools/check_cree_sources.py
    python tools/check_cree_sources.py --json    # machine-readable

The checks are the same URLs the cell uses:

    EdTeKLA corpus          codeload.github.com/EdTeKLA/...        tarball
    GiellaLT lang-crk       codeload.github.com/giellalt/...       tarball
    Tatoeba Cree            downloads.tatoeba.org/.../crk_...      bz2 file
    FineWeb-2 / FinePDFs    huggingface.co/api/datasets/...        dataset API
    FineTranslations        huggingface.co/api/datasets/...
    GlotCC-V1               huggingface.co/api/datasets/...
    GlotStoryBook           huggingface.co/api/datasets/...
    plains-cree-figurative  huggingface.co/api/datasets/...
    Cree Wikipedia          huggingface.co/api/datasets/...
    Internet Archive        archive.org/advancedsearch.php?...     search API
    project data            raw.githubusercontent.com/...           pairs file

A dataset API answering proves the repository exists and is public; it does not
prove the per-config slice has rows, which is what `load_*` checks at run time.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 30
SOURCES = [
    ("EdTeKLA Cree Corpus", "tarball",
     "https://codeload.github.com/EdTeKLA/IndigenousLanguages_Corpora/tar.gz/refs/heads/master"),
    ("GiellaLT lang-crk", "tarball",
     "https://codeload.github.com/giellalt/lang-crk/tar.gz/refs/heads/main"),
    ("Tatoeba Cree", "file",
     "https://downloads.tatoeba.org/exports/per_language/crk/crk_sentences.tsv.bz2"),
    ("FineWeb-2", "dataset", "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2"),
    ("FinePDFs", "dataset", "https://huggingface.co/api/datasets/HuggingFaceFW/finepdfs"),
    ("FineTranslations", "dataset", "https://huggingface.co/api/datasets/HuggingFaceFW/finetranslations"),
    ("GlotCC-V1", "dataset", "https://huggingface.co/api/datasets/cis-lmu/GlotCC-V1"),
    ("GlotStoryBook", "dataset", "https://huggingface.co/api/datasets/cis-lmu/GlotStoryBook"),
    ("Little Cree Books", "tarball",
     "https://codeload.github.com/global-asp/lcb-source/tar.gz/refs/heads/master"),
    ("Plains Cree Figurative", "dataset", "https://huggingface.co/api/datasets/KonradBRG/plains-cree-figurative"),
    ("Cree Wikipedia", "dataset", "https://huggingface.co/api/datasets/wikimedia/wikipedia"),
    ("Internet Archive", "search",
     "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
         {"q": 'language:(cree OR crk)', "fl[]": ["identifier"], "rows": 1, "output": "json"},
         doseq=True)),
    # The repository name is the one in 03_corpus.py's PROJECT_REPO. It has four
    # a's; a checker that guesses it reports a 404 and blames the source.
    ("Project pairs", "file",
     "https://raw.githubusercontent.com/haaaarsh4/Archer-Robot-Vocal-Collaboration-System/main/data/processed/parallel_cr_en.jsonl"),
]


def probe(url: str) -> tuple[bool, int, str]:
    """(ok, bytes reported, note). A HEAD is enough for the big tarballs."""
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "archer-corpus-check"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            size = resp.headers.get("Content-Length") or ""
            if url.endswith((".jsonl", ".json")) or "/api/datasets/" in url or "advancedsearch" in url:
                body = resp.read(2_000_000)
                return True, len(body), f"HTTP {resp.status}, {len(body):,} bytes read"
            resp.read(1024)
            if not size:
                return True, 0, f"HTTP {resp.status}, no length header"
            return True, int(size), f"HTTP {resp.status}, size {int(size):,} bytes"
    except urllib.error.HTTPError as e:
        # A 206 on a range request is a success; anything else is a real failure.
        if e.code in (200, 206):
            return True, 0, f"HTTP {e.code}"
        return False, 0, f"HTTP {e.code} {e.reason}"
    except Exception as e:  # noqa: BLE001 - the point is to report, not to raise
        return False, 0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = []
    failed = 0
    for name, kind, url in SOURCES:
        ok, size, note = probe(url)
        failed += not ok
        results.append({"source": name, "kind": kind, "url": url, "ok": ok, "note": note})
        if not args.json:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:24s} {kind:8s} {note}")

    if args.json:
        print(json.dumps(results, indent=2))

    if failed:
        print(f"\n{failed} of {len(SOURCES)} sources are not answering. "
              "The corpus cell would skip them and record the partial corpus, "
              "and 03_corpus.py now stops the run instead.")
        return 1
    print(f"\nall {len(SOURCES)} sources answered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

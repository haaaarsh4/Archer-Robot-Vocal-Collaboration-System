"""Build a large auxiliary MySong corpus from the AutoHarmonizer archive.

The model shipped with the package is trained on OpenEWLD (502 public-domain
lead sheets).  The AutoHarmonizer repository distributes 5,204 Wikifonia
lead sheets in MusicXML (`dataset.z01` + `dataset.zip`).  Wikifonia is the same
family of source MySong used, and the archive is ten times larger, which is the
single biggest quality lever available for a symbolic harmonizer.

This script does not silently mix the two corpora.  It:

* parses every MXL in parallel;
* removes scores whose normalized title **or transposition-invariant melody
  signature** matches an OpenEWLD score, so the held-out test split cannot leak
  through a duplicate arrangement (title matching alone missed 40 of 44 test
  tunes);
* deduplicates the auxiliary archive itself;
* writes a JSONL cache in the same format as `data/mysong/corpus.jsonl`.

The archive's own license is not a blanket license for the underlying songs;
check `data/autoharmonizer/README` and the source repository before publishing
derived weights.  The parser is the same one used for OpenEWLD.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mysong.corpus import load_corpus, melody_signature, parse_musicxml


def normalize_title(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^]]*\]", " ", value)
    value = re.sub(r"\b(transcribed|transcription|arranged|arrangement|cover)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def parse_one(path_text: str) -> tuple[object | None, str | None]:
    path = Path(path_text)
    try:
        song = parse_musicxml(path)
    except Exception as exc:  # noqa: BLE001 - a corpus build reports failures
        return None, f"{path.name}: {type(exc).__name__}: {exc}"
    # The archive is flat, so `path.parent.name` is just "dataset".  The file
    # stem carries the composer and title.
    song.title = path.stem
    song.composer = "AutoHarmonizer/Wikifonia"
    song.source = str(path)
    song.song_id = f"ah-{path.stem}"
    return song, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/autoharmonizer/dataset")
    parser.add_argument("--out", default="data/mysong/corpus_extra.jsonl")
    parser.add_argument("--openewld", default="data/mysong/corpus.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.root)
    paths = sorted(root.glob("*.mxl")) + sorted(root.glob("*.musicxml"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no scores under {root}", file=sys.stderr)
        return 1

    openewld_titles: set[str] = set()
    openewld_sigs: set[tuple] = set()
    if Path(args.openewld).exists():
        for song in load_corpus(args.openewld):
            openewld_titles.add(normalize_title(song.title))
            signature = melody_signature(song)
            if signature is not None:
                openewld_sigs.add(signature)
    print(
        f"OpenEWLD titles: {len(openewld_titles)}, melody signatures: "
        f"{len(openewld_sigs)}; archive files: {len(paths)}",
        flush=True,
    )

    kept: list[dict] = []
    seen: set[str] = set(openewld_titles)
    seen_sigs: set[tuple] = set(openewld_sigs)
    failures: list[str] = []
    duplicates = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(parse_one, str(path)) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            song, error = future.result()
            if error:
                failures.append(error)
            elif song is not None:
                key = normalize_title(song.title)
                signature = melody_signature(song)
                if key in seen or (signature is not None and signature in seen_sigs):
                    duplicates += 1
                else:
                    seen.add(key)
                    if signature is not None:
                        seen_sigs.add(signature)
                    payload = song.to_dict()
                    payload["dataset"] = "autoharmonizer"
                    kept.append(payload)
            if index % 500 == 0:
                print(f"  parsed {index}/{len(paths)} kept {len(kept)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for payload in kept:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        f"wrote {out}: {len(kept)} songs, {duplicates} duplicates, "
        f"{len(failures)} failures",
        flush=True,
    )
    for failure in failures[:10]:
        print(f"  failed: {failure}", file=sys.stderr)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Train a MySong tagger on OpenEWLD + the 4,571-score auxiliary corpus.

The important difference from ``python -m mysong.train_large`` is memory: this
tool streams the auxiliary JSONL one score at a time and keeps only the compact
per-beat feature/label arrays.  That fits in a few hundred MB even with 5,000
scores, so it runs on a laptop as well as on a GPU server.

The OpenEWLD dev/test split is created exactly as in the package and is never
used for gradients or early stopping.  The auxiliary corpus is train-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mysong.corpus import Song, load_corpus, split_corpus
from mysong.evaluate import chord_accuracy
from mysong.harmonize import Harmonizer, Style
from mysong.melody import Melody
from mysong.model import MySongModel
from mysong.neural import song_sequence, train_tagger_sequences


def _evaluate(model: MySongModel, songs, *, style: Style | None = None) -> dict:
    harmonizer = Harmonizer(model)
    style = style or Style()
    root = core = full = 0.0
    for song in songs:
        melody = Melody.from_notes(song.notes, beats_per_bar=song.beats_per_bar)
        harmony = harmonizer.harmonize(
            melody,
            key=song.key,
            style=style,
            beats_per_bar=song.beats_per_bar,
        )
        scores = chord_accuracy(song, harmony)
        root += scores["root"]
        core += scores["core"]
        full += scores["full"]
    n = max(1, len(songs))
    return {"n": len(songs), "root": round(root / n, 4),
            "core": round(core / n, 4), "full": round(full / n, 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/mysong/corpus.jsonl")
    parser.add_argument("--extra", default="data/mysong/corpus_extra.jsonl")
    parser.add_argument("--base-model", default="data/mysong/model/mysong.npz")
    parser.add_argument("--out-model", default="data/mysong/model/mysong_large.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--limit-extra", type=int, default=0)
    parser.add_argument(
        "--base-repeat", type=int, default=1,
        help="repeat OpenEWLD training sequences this many times to keep the "
             "small, in-domain corpus from being drowned by the auxiliary one",
    )
    parser.add_argument("--init-tagger", default=None,
                        help="fine-tune from an existing .pt tagger checkpoint")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    base = load_corpus(args.base)
    split = split_corpus(base, seed=args.seed)
    base_sequences = []
    for song in split["train"]:
        item = song_sequence(song)
        if item is not None:
            base_sequences.append(item)
    sequences = base_sequences * max(1, int(args.base_repeat))

    streamed = 0
    if args.extra and Path(args.extra).exists():
        with Path(args.extra).open(encoding="utf-8") as handle:
            for line in handle:
                if args.limit_extra and streamed >= args.limit_extra:
                    break
                line = line.strip()
                if not line:
                    continue
                song = Song.from_dict(json.loads(line))
                item = song_sequence(song)
                if item is not None:
                    sequences.append(item)
                    streamed += 1
                if streamed and streamed % 500 == 0:
                    print(f"  streamed {streamed} extra scores", flush=True)
    print(
        f"base train {len(split['train'])} x{max(1, args.base_repeat)}, "
        f"extra {streamed}, sequences {len(sequences)}",
        flush=True,
    )

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    tagger_pt = out_model.with_name(out_model.stem + "_tagger.pt")
    report = train_tagger_sequences(
        sequences,
        tagger_pt,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
        validation_fraction=0.08,
        device=args.device,
        init_path=args.init_tagger,
        log=print,
    )

    # Reuse the trained structural model (transitions, durations, key model)
    # and replace only its observation network.  This keeps the run bounded and
    # makes the comparison to the shipped model a pure tagger comparison.
    if out_model.resolve() != Path(args.base_model).resolve():
        shutil.copy2(args.base_model, out_model)
    model = MySongModel.load(out_model)
    result = {
        "model": str(out_model),
        "tagger": report.to_dict(),
        "extra_scores": streamed,
        "test": _evaluate(model, split["test"]),
        "dev": _evaluate(model, split["dev"]),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"model: {out_model}")
        print(f"test: {result['test']}")
        print(f"dev:  {result['dev']}")
        print(f"elapsed: {result['elapsed_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

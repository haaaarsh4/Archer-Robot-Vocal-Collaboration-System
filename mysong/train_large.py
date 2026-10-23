"""Train MySong on OpenEWLD plus a large auxiliary lead-sheet corpus.

The package's default model is trained on the 409 OpenEWLD training scores.
That is enough to prove the decoder works, not enough to make it strong.  This
entry point adds the 4,571 deduplicated AutoHarmonizer/Wikifonia lead sheets,
keeps the OpenEWLD dev/test split untouched, and reports the number on that
held-out split.  The auxiliary corpus is built by
``tools/build_autoharmonizer_corpus.py``.

Usage on CPU::

    python -m mysong.train_large --epochs 10 --hidden 128

On a GPU box::

    python -m mysong.train_large --device cuda --epochs 60 --hidden 256 \
        --out data/mysong/model/mysong_large.npz

The test split is never used for gradients or early stopping.  The auxiliary
corpus is train-only; duplicate titles and transposition-invariant melody
signatures were removed against the whole OpenEWLD corpus when the cache was
built.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mysong.corpus import load_corpus, split_corpus
from mysong.evaluate import chord_accuracy, chord_change_f1
from mysong.harmonize import Harmonizer, Style
from mysong.melody import Melody
from mysong.model import DEFAULT_MODEL, MySongModel
from mysong.train import TrainConfig, fit_model


def _evaluate(model: MySongModel, songs, *, style: Style | None = None) -> dict:
    harmonizer = Harmonizer(model)
    style = style or Style()
    rows = []
    for song in songs:
        melody = Melody.from_notes(song.notes, beats_per_bar=song.beats_per_bar)
        harmony = harmonizer.harmonize(
            melody,
            key=song.key,
            style=style,
            beats_per_bar=song.beats_per_bar,
        )
        scores = chord_accuracy(song, harmony)
        changes = chord_change_f1(song, harmony)
        rows.append({
            "root": scores["root"],
            "core": scores["core"],
            "full": scores["full"],
            "key": float(harmony.key == song.key),
            "change_f1": changes["f1"],
            "segments": len(harmony.segments),
        })
    return {
        "n": len(rows),
        **{
            name: round(float(np.mean([row[name] for row in rows])), 4)
            for name in ("root", "core", "full", "key", "change_f1", "segments")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/mysong/corpus.jsonl")
    parser.add_argument("--extra", default="data/mysong/corpus_extra.jsonl")
    parser.add_argument("--out", default="data/mysong/model/mysong_large.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--limit-extra", type=int, default=0)
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    base = load_corpus(args.base)
    split = split_corpus(base, seed=args.seed)
    extra: list = []
    if Path(args.extra).exists():
        extra = load_corpus(args.extra)
        if args.limit_extra:
            extra = extra[: args.limit_extra]
    train_songs = list(split["train"]) + list(extra)
    print(
        f"base {len(base)} (train {len(split['train'])}, dev {len(split['dev'])}, "
        f"test {len(split['test'])}) + extra {len(extra)} => train {len(train_songs)}",
        flush=True,
    )

    config = TrainConfig(
        corpus=Path(args.base),
        out=Path(args.out),
        seed=args.seed,
        tagger_epochs=args.epochs,
        tagger_hidden=args.hidden,
        train_classifier=not args.skip_classifier,
    )
    # `fit_model` trains the tagger with early stopping on a slice of the
    # *training* songs, so dev/test stay clean.  The base classifier is reused
    # when the large corpus is too big to fit a second dense logistic model in
    # memory; the neural tagger is the primary observation model anyway.
    model, report = fit_model(train_songs, config, split={
        "train": len(train_songs), "dev": len(split["dev"]), "test": len(split["test"]),
        "base_train": len(split["train"]), "extra": len(extra),
    }, verbose=True)

    if model.classifier is None and Path(DEFAULT_MODEL).exists():
        try:
            model.classifier = MySongModel.load(DEFAULT_MODEL).classifier
            report["classifier"] = {"reused": str(DEFAULT_MODEL)}
        except Exception as exc:  # pragma: no cover - optional fallback
            report["classifier"] = {"error": str(exc)}

    path = model.save(Path(args.out))
    report["model_path"] = str(path)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    model = MySongModel.load(path)
    report["test"] = _evaluate(model, split["test"])
    report["dev"] = _evaluate(model, split["dev"])
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"model: {path}")
        print(f"test: {report['test']}")
        print(f"dev:  {report['dev']}")
        print(f"elapsed: {report['elapsed_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry points.

    python -m mysong.cli corpus          # parse OpenEWLD into a cache
    python -m mysong.cli train           # estimate the model (CPU, ~1 minute)
    python -m mysong.cli status          # what is installed / trained
    python -m mysong.cli analyze take.wav
    python -m mysong.cli render take.wav --out renders/take --style jazz

The CLI deliberately uses the same functions as the HTTP app, so a result
reproduced on the command line is the result the app returns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mysong.accompaniment import ArrangementStyle, DEFAULT_BAND
from mysong.corpus import DEFAULT_CACHE, DEFAULT_CORPUS, build_corpus, corpus_stats, load_corpus
from mysong.harmonize import Style
from mysong.model import DEFAULT_MODEL, MySongModel
from mysong.train import TrainConfig, train_from_cache


def _cmd_corpus(args: argparse.Namespace) -> int:
    songs, failures = build_corpus(
        args.root, args.cache, limit=args.limit, verbose=not args.quiet
    )
    stats = corpus_stats(songs)
    print(json.dumps({
        "songs": stats.get("songs", 0),
        "melody_notes": stats.get("melody_notes", 0),
        "chords": stats.get("chords", 0),
        "modes": stats.get("modes", {}),
        "unique_chords": stats.get("unique_chords", 0),
        "failures": len(failures),
        "cache": str(args.cache),
    }, indent=2))
    for path, error in failures[:5]:
        print(f"  failed: {Path(path).name}: {error}", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = TrainConfig(
        corpus=Path(args.cache),
        out=Path(args.out),
        seed=args.seed,
        train_classifier=not args.no_classifier,
        train_tagger=not args.no_tagger,
        tagger_epochs=args.tagger_epochs,
    )
    model, report, splits = train_from_cache(config, verbose=not args.quiet)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"model: {args.out}")
        print(f"split: {report['split']}")
        if report.get("classifier"):
            classifier = report["classifier"]
            print(
                f"classifier: {classifier['samples']} examples, "
                f"{classifier['features']} features, "
                f"train top-1 {classifier['train_top1']:.3f} / top-3 {classifier['train_top3']:.3f}"
            )
        tagger = report.get("tagger") or {}
        if tagger and "error" not in tagger:
            print(
                f"tagger: {tagger.get('train_songs', 0)} train / "
                f"{tagger.get('val_songs', 0)} val songs, "
                f"val root {tagger.get('best_val_root', 0):.3f}, "
                f"{tagger.get('parameters', 0):,} parameters"
            )
        elif tagger.get("error"):
            print(f"tagger: skipped ({tagger['error']})")
        key_report = report.get("key_classifier") or {}
        if key_report:
            print(
                f"key classifier: {key_report.get('samples', 0)} songs, "
                f"train accuracy {key_report.get('train_accuracy', 0):.3f}"
            )
        print("done")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    payload: dict = {"model": str(args.model), "corpus": str(args.cache)}
    try:
        model = MySongModel.load(args.model)
        payload["model_info"] = {
            "songs": model.meta.songs,
            "transitions": model.meta.transitions,
            "trained_at": model.meta.trained_at,
            "split": model.meta.split,
            "classifier": model.classifier is not None,
            "key_classifier": model.key_classifier is not None,
            "tagger": model.tagger is not None,
            "meta": model.meta.config.get("classifier", {}),
            "key_meta": model.meta.config.get("key_classifier", {}),
            "tagger_meta": model.meta.config.get("tagger", {}),
        }
    except FileNotFoundError as exc:
        payload["model_error"] = str(exc)
    try:
        songs = load_corpus(args.cache)
        payload["corpus_info"] = corpus_stats(songs)
    except FileNotFoundError as exc:
        payload["corpus_error"] = str(exc)
    print(json.dumps(payload, indent=2))
    return 0 if "model_error" not in payload else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    from mysong.pipeline import analyze_song

    style = Style(
        observation_weight=args.obs_weight,
        density=args.density,
        color=args.color,
        key_weight=args.key_weight,
        key_candidates=args.key_candidates,
    )
    analysis, harmony = analyze_song(
        args.audio,
        key=args.key,
        style=style,
        bpm=args.bpm,
        beat_offset=args.offset,
        beats_per_bar=args.beats_per_bar,
        backend=args.backend,
    )
    if args.json:
        payload = {
            "analysis": {
                "tempo_bpm": analysis.tempo.bpm,
                "beat_offset": analysis.tempo.beat_offset,
                "tuning_cents": analysis.tuning_cents,
                "voiced_ratio": analysis.voiced_ratio,
                "key_guess": analysis.key_guess.name(),
                "warnings": analysis.warnings,
            },
            "harmony": harmony.to_dict(),
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(f"tempo: {analysis.tempo.bpm:.1f} BPM (offset {analysis.tempo.beat_offset:.2f}s)")
    print(f"tuning: {analysis.tuning_cents:+.1f} cents")
    print(f"key: {harmony.key.name()} (confidence {harmony.key_confidence:.0%})")
    print(f"chords: {harmony.summary()}")
    for segment in harmony.segments:
        print(
            f"  {segment.start_beat:6.1f}-{segment.end_beat:<6.1f} "
            f"{segment.chord.figure(harmony.key):8} "
            f"{harmony.key.roman(segment.chord):8} "
            f"conf {segment.confidence:.0%}"
        )
    for warning in analysis.warnings + harmony.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from mysong.pipeline import create_song

    style = Style(
        observation_weight=args.obs_weight,
        density=args.density,
        color=args.color,
        key_weight=args.key_weight,
        key_candidates=args.key_candidates,
    )
    project = create_song(
        args.audio,
        key=args.key,
        style=style,
        arrangement_style=args.style,
        bpm=args.bpm,
        beat_offset=args.offset,
        beats_per_bar=args.beats_per_bar,
        sample_rate=args.sample_rate,
        mix_vocal=not args.no_vocal,
        vocal_gain_db=args.vocal_gain_db,
        backend=args.backend,
        renderer=args.renderer,
    )
    paths = project.save(args.out)
    if args.json:
        print(json.dumps({"paths": paths, "result": project.to_dict()}, indent=2))
    else:
        print(f"key: {project.harmony.key.name()} (confidence {project.harmony.key_confidence:.0%})")
        print(f"tempo: {project.analysis.tempo.bpm:.1f} BPM, style {args.style}")
        print(f"chords: {project.harmony.summary()}")
        for name, path in paths.items():
            print(f"  {name:14} {path}")
        for warning in project.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from mysong.evaluate import evaluate_corpus, format_report

    report = evaluate_corpus(
        MySongModel.load(args.model),
        cache=args.cache,
        split=args.split,
        limit=args.limit,
        include_generative=not args.fast,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mysong", description="MySong-style accompaniment generation")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="parse the OpenEWLD lead-sheet corpus")
    corpus.add_argument("--root", default=str(DEFAULT_CORPUS))
    corpus.add_argument("--cache", default=str(DEFAULT_CACHE))
    corpus.add_argument("--limit", type=int, default=0)
    corpus.add_argument("--quiet", action="store_true")
    corpus.set_defaults(func=_cmd_corpus)

    train = sub.add_parser("train", help="train the model from the cached corpus")
    train.add_argument("--cache", default=str(DEFAULT_CACHE))
    train.add_argument("--out", default=str(DEFAULT_MODEL))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--no-classifier", action="store_true",
                       help="skip the discriminative observation model (generative fallback)")
    train.add_argument("--no-tagger", action="store_true",
                       help="skip the neural chord tagger (numpy classifier only)")
    train.add_argument("--tagger-epochs", type=int, default=25)
    train.add_argument("--quiet", action="store_true")
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=_cmd_train)

    status = sub.add_parser("status", help="show model and corpus status")
    status.add_argument("--model", default=str(DEFAULT_MODEL))
    status.add_argument("--cache", default=str(DEFAULT_CACHE))
    status.set_defaults(func=_cmd_status)

    analyze = sub.add_parser("analyze", help="print the key and chords for a take")
    analyze.add_argument("audio")
    analyze.add_argument("--key", default=None)
    analyze.add_argument("--bpm", type=float, default=None)
    analyze.add_argument("--offset", type=float, default=None, help="seconds of the first beat")
    analyze.add_argument("--beats-per-bar", type=float, default=4.0)
    analyze.add_argument("--obs-weight", type=float, default=2.5)
    analyze.add_argument("--density", type=float, default=1.0)
    analyze.add_argument("--color", type=float, default=0.75)
    analyze.add_argument("--key-weight", type=float, default=8.0)
    analyze.add_argument("--key-candidates", type=int, default=12)
    analyze.add_argument("--backend", default="pyin", choices=["pyin", "rmvpe"])
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(func=_cmd_analyze)

    render = sub.add_parser("render", help="render an accompaniment and mix")
    render.add_argument("audio")
    render.add_argument("--out", required=True)
    render.add_argument("--style", default=DEFAULT_BAND,
                        choices=list(ArrangementStyle.presets()))
    render.add_argument("--key", default=None)
    render.add_argument("--bpm", type=float, default=None)
    render.add_argument("--offset", type=float, default=None)
    render.add_argument("--beats-per-bar", type=float, default=4.0)
    render.add_argument("--obs-weight", type=float, default=2.5)
    render.add_argument("--density", type=float, default=1.0)
    render.add_argument("--color", type=float, default=0.75)
    render.add_argument("--key-weight", type=float, default=8.0)
    render.add_argument("--key-candidates", type=int, default=12)
    render.add_argument("--sample-rate", type=int, default=44100)
    render.add_argument("--vocal-gain-db", type=float, default=0.0)
    render.add_argument("--no-vocal", action="store_true")
    render.add_argument("--renderer", default="auto", choices=["auto", "soundfont", "synth"],
                        help="auto uses a General MIDI SoundFont when FluidSynth is installed")
    render.add_argument("--backend", default="pyin", choices=["pyin", "rmvpe"])
    render.add_argument("--json", action="store_true")
    render.set_defaults(func=_cmd_render)

    evaluate = sub.add_parser("evaluate", help="objective evaluation on held-out songs")
    evaluate.add_argument("--split", default="test", choices=["train", "dev", "test"])
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--cache", default=str(DEFAULT_CACHE))
    evaluate.add_argument("--model", default=str(DEFAULT_MODEL))
    evaluate.add_argument("--fast", action="store_true", help="skip the generative ablation")
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

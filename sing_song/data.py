"""Build the (vocal, instrumental) token corpus SingSong trains on.

The paper's recipe, in order:

1. take a music corpus and separate every mix into vocals and instrumental;
2. keep non-overlapping 10 s clips;
3. drop clips whose instrumental peak RMS is below -25 dBFS, or whose vocal
   peak RMS is 5 dB louder than the instrumental;
4. add white noise at sigma=0.01 to the *vocal* (the "Noisy" recipe that stops
   the model learning source-separation artifacts);
5. tokenize the vocal with the semantic encoder and the instrumental with the
   SoundStream codec.

MUSDB18-HQ already ships studio stems, so for that corpus step 1 is free: the
instrumental is drums + bass + other.  For an arbitrary folder of mixes, the
pipeline can call Demucs (installed in this project) or any configured
separation command, then use the resulting ``vocals``/``no_vocals`` pair.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class TrackPair:
    name: str
    vocals: Path
    instrumental: Path


@dataclass(frozen=True)
class Clip:
    index: int
    vocal: np.ndarray
    instrumental: np.ndarray
    sample_rate: int
    start_sample: int


@dataclass(frozen=True)
class ClipDecision:
    keep: bool
    reason: str
    vocal_peak_db: float
    instrumental_peak_db: float


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    import librosa

    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def peak_rms_db(audio: np.ndarray, sample_rate: int, frame_ms: float = 50.0) -> float:
    """Peak short-time RMS in dBFS, the paper's loudness measure."""
    if audio.size == 0:
        return float("-inf")
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, frame // 2)
    if audio.size < frame:
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        return 20.0 * np.log10(max(rms, 1e-12))
    # strided frames, no padding
    n = 1 + (audio.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n, frame), strides=(audio.strides[0] * hop, audio.strides[0])
    )
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return float(20.0 * np.log10(max(float(rms.max()), 1e-12)))


def decide_clip(
    vocal: np.ndarray,
    instrumental: np.ndarray,
    sample_rate: int,
    *,
    instrumental_floor_db: float = -25.0,
    vocal_over_instrumental_db: float = 5.0,
) -> ClipDecision:
    v_db = peak_rms_db(vocal, sample_rate)
    i_db = peak_rms_db(instrumental, sample_rate)
    if i_db < instrumental_floor_db:
        return ClipDecision(False, "instrumental_below_floor", v_db, i_db)
    if v_db - i_db >= vocal_over_instrumental_db:
        return ClipDecision(False, "vocal_too_loud", v_db, i_db)
    return ClipDecision(True, "ok", v_db, i_db)


def add_noise(audio: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return audio.astype(np.float32, copy=False)
    noise = rng.normal(0.0, sigma, size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def iter_clips(
    vocal: np.ndarray,
    instrumental: np.ndarray,
    sample_rate: int,
    *,
    clip_seconds: float = 10.0,
) -> Iterator[Clip]:
    if vocal.shape != instrumental.shape:
        raise ValueError(
            f"vocal and instrumental lengths differ: {vocal.shape} vs {instrumental.shape}"
        )
    length = int(round(clip_seconds * sample_rate))
    if length <= 0:
        raise ValueError("clip_seconds must be positive")
    for index, start in enumerate(range(0, len(vocal) - length + 1, length)):
        yield Clip(
            index=index,
            vocal=vocal[start:start + length],
            instrumental=instrumental[start:start + length],
            sample_rate=sample_rate,
            start_sample=start,
        )


def discover_tracks(corpus: str | Path, split: str = "train") -> list[TrackPair]:
    """Find (vocals, instrumental) pairs in MUSDB18-HQ or a generic folder.

    Recognised layouts::

        corpus/<split>/<track>/{vocals.wav, drums.wav, bass.wav, other.wav}
        corpus/<track>/{vocals.wav, instrumental.wav|accompaniment.wav}
        corpus/<track>/{vocals.wav, mixture.wav}          # instruments = mix - vocal
    """
    root = Path(corpus)
    if not root.exists():
        raise FileNotFoundError(f"corpus not found: {root}")
    base = root / split if (root / split).is_dir() else root
    tracks: list[TrackPair] = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        vocals = _first_existing(directory, ("vocals.wav", "vocal.wav", "vocals.flac"))
        if vocals is None:
            continue
        accompaniment = _first_existing(
            directory, ("instrumental.wav", "accompaniment.wav", "no_vocals.wav", "instrumental.flac")
        )
        if accompaniment is not None:
            tracks.append(TrackPair(directory.name, vocals, accompaniment))
            continue
        # MUSDB18-HQ: sum the non-vocal stems.  This is exact, whereas
        # mixture - vocals would bake the separation residual into the target.
        stems = [directory / name for name in ("drums.wav", "bass.wav", "other.wav")]
        if all(p.exists() for p in stems):
            tracks.append(TrackPair(directory.name, vocals, directory / "__musdb_instrumental__.wav"))
            continue
        mixture = _first_existing(directory, ("mixture.wav", "mix.wav"))
        if mixture is not None:
            tracks.append(TrackPair(directory.name, vocals, mixture))
    return tracks


def _first_existing(directory: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def materialize_instrumental(pair: TrackPair, sample_rate: int) -> np.ndarray:
    """Load the instrumental, summing MUSDB stems when necessary."""
    if pair.instrumental.name == "__musdb_instrumental__.wav":
        directory = pair.instrumental.parent
        total = None
        for stem in ("drums.wav", "bass.wav", "other.wav"):
            audio = load_audio(directory / stem, sample_rate)
            total = audio if total is None else total + audio
        if total is None:
            raise FileNotFoundError(f"no instrumental stems in {directory}")
        return total.astype(np.float32)
    if pair.instrumental.name in ("mixture.wav", "mix.wav"):
        # A mix with no separate instrumental: fall back to the mixture.  The
        # caller should separate it; this is only a last resort for evaluation.
        return load_audio(pair.instrumental, sample_rate)
    return load_audio(pair.instrumental, sample_rate)


def write_codec_corpus(
    cfg,
    out_dir: str | Path = "data/singsong/codec_audio",
    *,
    log=print,
) -> dict:
    """Materialise one instrumental WAV per track for codec training.

    The codec should learn the distribution it will later encode: the
    instrumental, not the vocal stem.  For MUSDB this sums ``drums + bass +
    other``; for a generic corpus it copies ``instrumental.wav``.  A
    mixture-only track is written as-is with a warning because there is no
    separated target to learn from.
    """
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = discover_tracks(cfg.corpus, cfg.musdb_split)
    if not pairs:
        raise RuntimeError(f"no (vocal, instrumental) pairs found under {cfg.corpus}")
    result = {"out": str(out), "tracks": 0, "seconds": 0.0, "warnings": []}
    for i, pair in enumerate(pairs, 1):
        if pair.instrumental.name in ("mixture.wav", "mix.wav"):
            result["warnings"].append(
                f"{pair.name}: no instrumental stem; writing the mixture"
            )
        audio = materialize_instrumental(pair, cfg.sample_rate)
        sf.write(
            out / f"{pair.name}.wav", audio, cfg.sample_rate, subtype="PCM_16"
        )
        result["tracks"] += 1
        result["seconds"] += len(audio) / cfg.sample_rate
        log(
            f"[singsong] codec audio {i}/{len(pairs)} {pair.name}: "
            f"{len(audio) / cfg.sample_rate:.1f}s"
        )
    result["seconds"] = round(result["seconds"], 2)
    return result


def separate_with_demucs(
    mixture: str | Path,
    output_dir: str | Path,
    *,
    model: str = "htdemucs",
    python: str | None = None,
) -> tuple[Path, Path]:
    """Run Demucs' two-stem mode and return (vocals, no_vocals)."""
    mixture = Path(mixture)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python or sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals", "-n", model, "-o", str(output_dir), str(mixture),
    ]
    subprocess.run(cmd, check=True)
    stem_dir = output_dir / model / mixture.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError(
            f"Demucs finished but did not produce the expected stems under {stem_dir}"
        )
    return vocals, instrumental


def separate_with_command(
    mixture: str | Path,
    output_dir: str | Path,
    command: str,
) -> tuple[Path, Path]:
    """Run an MDXNet/other separator via a shell command template.

    ``{input}``, ``{output}`` and ``{stem}`` are substituted.  The command must
    create ``{output}/{stem}/vocals.wav`` and ``{output}/{stem}/no_vocals.wav``.
    """
    mixture = Path(mixture)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = command.format(
        input=str(mixture), output=str(output_dir), stem=mixture.stem
    )
    subprocess.run(rendered, shell=True, check=True)
    stem_dir = output_dir / mixture.stem
    return stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"


def tokenize_clip(
    clip: Clip,
    *,
    codec,
    semantic,
    target_sample_rate: int,
    semantic_rate: int,
    noise_sigma: float,
    rng: np.random.Generator,
    device=None,
) -> dict[str, np.ndarray]:
    """Add noise, run both tokenizers, and return integer arrays.

    The vocal gets the noise; the instrumental does not.  This is the paper's
    Noisy/S-SA recipe: the noise sits above source-separation artifacts and
    below the vocal, so the semantic encoder cannot hear the leftover
    instrumental.
    """
    import torch
    import torchaudio.functional as AF

    if clip.sample_rate != target_sample_rate:
        vocal = AF.resample(
            torch.from_numpy(clip.vocal), clip.sample_rate, target_sample_rate
        ).numpy()
        instrumental = AF.resample(
            torch.from_numpy(clip.instrumental), clip.sample_rate, target_sample_rate
        ).numpy()
    else:
        vocal = clip.vocal
        instrumental = clip.instrumental

    vocal = add_noise(vocal, noise_sigma, rng)
    # Keep the target at the same length as the vocal even after resampling.
    n = min(len(vocal), len(instrumental))
    vocal, instrumental = vocal[:n], instrumental[:n]

    with torch.no_grad():
        vocal_t = torch.from_numpy(vocal).unsqueeze(0)
        inst_t = torch.from_numpy(instrumental).unsqueeze(0)
        sem = semantic.encode(vocal_t, sample_rate=target_sample_rate)
        if isinstance(sem, tuple):
            sem = sem[0]
        codes = codec.encode(inst_t)
    sem = sem[0].detach().cpu().numpy().astype(np.int16)
    codes = codes[0].detach().cpu().numpy().astype(np.int16)
    coarse = codes[:4]
    fine = codes[4:12]
    return {"semantic": sem, "coarse": coarse, "fine": fine}


def prepare_dataset(
    cfg,
    *,
    codec,
    semantic,
    separate: bool = False,
    limit: int | None = None,
    log=print,
) -> dict:
    """Convert a corpus into cached token shards and an index.jsonl.

    Returns a small manifest with counts and dropped-clip reasons.  The function
    is intentionally resumable: an existing ``<track>__<clip>.npz`` is skipped
    unless ``cfg.overwrite`` is true.
    """
    token_dir = Path(cfg.token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_tracks(cfg.corpus, cfg.musdb_split)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"no (vocal, instrumental) pairs found under {cfg.corpus}")

    manifest: dict = {
        "corpus": str(cfg.corpus),
        "clip_seconds": cfg.clip_seconds,
        "sample_rate": cfg.sample_rate,
        "noise_sigma": cfg.noise_sigma,
        "clips": 0,
        "dropped": {},
        "tracks": [],
    }
    rng = np.random.default_rng(0)
    index_path = token_dir / "index.jsonl"
    existing_index: dict[str, dict] = {}
    if index_path.exists() and not cfg.overwrite:
        for entry in read_index(token_dir):
            existing_index[entry["path"]] = entry
    index_fh = open(index_path, "w", encoding="utf-8")
    for entry in existing_index.values():
        index_fh.write(json.dumps(entry) + "\n")
    index_fh.flush()

    try:
        for ti, pair in enumerate(pairs):
            try:
                vocal = load_audio(pair.vocals, cfg.sample_rate)
                instrumental = materialize_instrumental(pair, cfg.sample_rate)
                if pair.instrumental.name == "__musdb_instrumental__.wav":
                    # For MUSDB the summed stems are already the target.
                    pass
                elif pair.instrumental.name in ("mixture.wav", "mix.wav"):
                    if not separate:
                        raise ValueError(
                            "a mixture-only track needs separation; run with separate=True "
                            "or provide an instrumental.wav next to vocals.wav"
                        )
                    sep_dir = token_dir / "_separated" / pair.name
                    v_path, i_path = separate_with_demucs(pair.instrumental, sep_dir)
                    vocal = load_audio(v_path, cfg.sample_rate)
                    instrumental = load_audio(i_path, cfg.sample_rate)
            except Exception as exc:  # noqa: BLE001 - one bad track must not stop a corpus
                log(f"[singsong] skip track {pair.name}: {exc}")
                manifest["tracks"].append({"name": pair.name, "error": str(exc)})
                continue

            kept = 0
            for clip in iter_clips(vocal, instrumental, cfg.sample_rate, clip_seconds=cfg.clip_seconds):
                decision = decide_clip(
                    clip.vocal, clip.instrumental, cfg.sample_rate,
                    instrumental_floor_db=cfg.instrumental_peak_rms_db,
                    vocal_over_instrumental_db=cfg.vocal_over_instrumental_db,
                )
                if not decision.keep:
                    manifest["dropped"][decision.reason] = manifest["dropped"].get(decision.reason, 0) + 1
                    continue
                name = f"{pair.name}__{clip.index:05d}"
                out = token_dir / f"{name}.npz"
                if out.exists() and not cfg.overwrite:
                    kept += 1
                    manifest["clips"] += 1
                    continue
                try:
                    tokens = tokenize_clip(
                        clip, codec=codec, semantic=semantic,
                        target_sample_rate=cfg.sample_rate,
                        semantic_rate=cfg.semantic_rate,
                        noise_sigma=cfg.noise_sigma, rng=rng,
                    )
                except Exception as exc:  # noqa: BLE001
                    log(f"[singsong] tokenize failed {name}: {exc}")
                    manifest["dropped"]["tokenize_error"] = manifest["dropped"].get("tokenize_error", 0) + 1
                    continue
                np.savez_compressed(out, **tokens)
                entry = {
                    "path": out.name,
                    "track": pair.name,
                    "clip": clip.index,
                    "start_sample": clip.start_sample,
                    "vocal_peak_db": decision.vocal_peak_db,
                    "instrumental_peak_db": decision.instrumental_peak_db,
                }
                existing_index[out.name] = entry
                # The file is rewritten once at the end, so an interrupted run
                # never leaves duplicate lines for the same clip.
                kept += 1
                manifest["clips"] += 1
            manifest["tracks"].append({"name": pair.name, "clips": kept})
            log(f"[singsong] {ti + 1}/{len(pairs)} {pair.name}: {kept} clips")
            index_fh.flush()
    finally:
        index_fh.close()

    # Rewrite the index deterministically, one line per clip, in track order.
    index_fh.close()
    with open(index_path, "w", encoding="utf-8") as fh:
        for entry in sorted(existing_index.values(), key=lambda e: (e.get("track", ""), e.get("clip", 0))):
            fh.write(json.dumps(entry) + "\n")
    (token_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_index(token_dir: str | Path) -> list[dict]:
    path = Path(token_dir) / "index.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"token index not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TokenDataset:
    """Map-style token dataset for the training loops.

    Each item is the fixed 10 s token set: semantic [250], coarse [4, 500],
    fine [8, 500].  The training script uses a DistributedSampler when run
    under torchrun, so this class stays small.
    """

    def __init__(self, token_dir: str | Path, *, expected_frames: int | None = None):
        self.token_dir = Path(token_dir)
        self.entries = read_index(self.token_dir)
        if not self.entries:
            raise RuntimeError(f"no token clips in {self.token_dir}")
        self.expected_frames = expected_frames

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        import torch

        entry = self.entries[index]
        data = np.load(self.token_dir / entry["path"])
        semantic = torch.from_numpy(data["semantic"].astype(np.int64))
        coarse = torch.from_numpy(data["coarse"].astype(np.int64))
        fine = torch.from_numpy(data["fine"].astype(np.int64))
        if self.expected_frames is not None:
            if semantic.numel() != self.expected_frames:
                raise ValueError(
                    f"{entry['path']}: expected {self.expected_frames} semantic frames, "
                    f"got {semantic.numel()}; rebuild with the same clip_seconds"
                )
        return {
            "semantic": semantic,
            "coarse": coarse,
            "fine": fine,
            "path": entry["path"],
        }

    def stats(self) -> dict:
        return {
            "clips": len(self.entries),
            "tracks": len({entry["track"] for entry in self.entries}),
            "token_dir": str(self.token_dir),
        }

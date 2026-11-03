#!/usr/bin/env python3
"""Download CC BY Cree singing recordings and optionally build paired stems.

The 27 recordings below were selected because the uploader published them under
YouTube's "Creative Commons Attribution license (reuse allowed)" setting. That
is a real reuse licence for the recording, unlike the all-rights-reserved
commercial Cree albums. The script records the source URL, uploader, title and
licence next to the audio so attribution survives.

The mixed songs are useful for codec/semantic adaptation and for listening.
SingSong stage 1+2/3 also needs a vocal/instrumental pair. ``--separate`` uses
Demucs to make one; that is source separation, not a studio stem, so treat the
result as weakly paired training data and keep the original mix.

Usage::

    python tools/fetch_cree_songs.py --download
    python tools/fetch_cree_songs.py --download --separate-ids RkRTiAiKNq0,2Xjn4jijq3A

Outputs::

    data/cree_songs/audio/<youtube-id>.wav
    data/cree_songs/metadata.jsonl
    data/cree_songs/manifest.json
    data/cree_songs/README.md

    # only with --separate:
    data/cree_singsong/audio/<youtube-id>/{vocals,instrumental}.wav
    data/cree_singsong/metadata.jsonl

The data directories are git-ignored. Run ``--verify`` to print the measured
hours without touching the network.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "cree_songs"
DEFAULT_PAIRED = PROJECT_ROOT / "data" / "cree_singsong"
LICENCE = "CC BY"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
CONSENT_ID = "youtube-cc-by-upload"


@dataclass(frozen=True)
class Track:
    youtube_id: str
    title: str
    singer: str
    style: str
    dialect: str = "Plains Cree"
    split: str = "train"
    notes: str = ""

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"


# A conservative list: every entry is either an explicitly Cree
# group/performer, a Cree-language song, or a Cree round-dance recording. The
# long Little Island Cree video is a full round-dance set, not one song.
TRACKS: tuple[Track, ...] = (
    Track(
        "03xrspXK9pY",
        "Little Island Cree Classic Round Dance Songs",
        "Little Island Cree",
        "round dance",
        notes="45-minute round-dance set; one continuous recording.",
    ),
    Track("Sj2DqRb-CBY", "Northern Cree Intertribal Song", "Northern Cree", "intertribal"),
    Track(
        "SD7KpH8B-6g",
        "Northern Cree Powerful Traditional Song",
        "Northern Cree",
        "traditional",
    ),
    Track("v5YAjS76R6I", "Northern Cree Sings a Jammer", "Northern Cree", "jammer"),
    Track(
        "goDU25CnLYU",
        "Northern Cree Chicken Dance Trick Song",
        "Northern Cree",
        "chicken dance",
    ),
    Track(
        "nQXDV2A61jI",
        "Northern Cree Tricky Fancy Action",
        "Northern Cree",
        "fancy dance",
    ),
    Track(
        "_LyUCJbBwuQ",
        "Northern Cree Smooth Grass Dance Song",
        "Northern Cree",
        "grass dance",
    ),
    Track("2Xjn4jijq3A", "Northern Cree Fancy Jammer", "Northern Cree", "fancy dance"),
    Track(
        "RU7RQ4kO_f4",
        "Northern Cree Ladies Group 2 Song 1",
        "Northern Cree",
        "ladies dance",
    ),
    Track(
        "c_SMinISiac",
        "Cree Confederation Jamming Song",
        "Cree Confederation",
        "jammer",
    ),
    Track(
        "QL5dpYIlUQc",
        "Cree Confederation Fancy Dance Song",
        "Cree Confederation",
        "fancy dance",
    ),
    Track(
        "ui1eVd_O3AI",
        "Cree Confederation Fancy Dance Song 2",
        "Cree Confederation",
        "fancy dance",
    ),
    Track("W-FGFS-Ei-s", "Mountain Cree Singers", "Mountain Cree", "intertribal"),
    Track(
        "yOuSag1GfZE",
        "Big River Cree Singers",
        "Big River Cree",
        "intertribal",
        split="dev",
    ),
    Track("X7Y3odW000w", "Bad Lands Cree Singers", "Bad Lands Cree", "intertribal"),
    Track("EWqvc7BrCaE", "Young Spirit Mind Blowing Song", "Young Spirit", "round dance"),
    Track(
        "f9N1PFigIC4",
        "Sorrel Rider Round Dance Jam",
        "Powwow Times singers",
        "round dance",
    ),
    Track(
        "_BGn-E3uss0",
        "Iron Boy Style Singers Intertribal Song",
        "Iron Boy Style Singers",
        "intertribal",
    ),
    Track(
        "dr-KqSimIHc",
        "Jr Mens Northern Fancy Jam",
        "Powwow Times singers",
        "fancy dance",
    ),
    Track(
        "liCZRqx202M",
        "O Holy Night (Plains Cree)",
        "Falynn Baptiste",
        "Christmas hymn",
        split="test",
    ),
    Track(
        "l4Td_Ur0rUE",
        "Silent Night (Plains Cree)",
        "Falynn Baptiste",
        "Christmas hymn",
        split="test",
    ),
    Track(
        "r0tFyijTkLk",
        "Angels We Have Heard On High (Plains Cree)",
        "Falynn Baptiste",
        "Christmas hymn",
        split="test",
    ),
    Track(
        "zSGqAfCiJHI",
        "O Canada (Cree)",
        "Jaguar Bird / Cree vocal",
        "anthem",
        split="test",
    ),
    Track(
        "RkRTiAiKNq0",
        "Cree Language Song",
        "Brian Tomaszewski",
        "children's song",
    ),
    Track(
        "lUyAZP82V3Y",
        "Cree Songs by Confederation Park School Students",
        "Confederation Park School",
        "children's song",
    ),
    Track(
        "bA2_x9upCiw",
        "The Edge of the Journey (Cree)",
        "Hall of Hidden Wisdom",
        "contemporary",
    ),
    Track(
        "U3ksUpjXtnc",
        "Mamii Jii Makw Jesus",
        "rabbit86musician",
        "hymn",
    ),
)


def _run(command: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.frames <= 0:
        raise RuntimeError(f"{path} is empty")
    return float(info.frames) / float(info.samplerate)


def _download(track: Track, out: Path, *, dry_run: bool = False) -> Path:
    destination = out / "audio" / f"{track.youtube_id}.wav"
    if destination.exists():
        try:
            _duration(destination)
            print(f"have {destination.name} ({_duration(destination) / 60:.1f} min)")
            return destination
        except Exception as exc:  # noqa: BLE001 - replace a corrupt download
            print(f"replacing {destination.name}: {exc}")
            destination.unlink(missing_ok=True)

    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--extractor-args",
        "youtube:player_client=android",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--no-progress",
        "--sleep-requests",
        "1",
        "-o",
        str(out / "audio" / "%(id)s.%(ext)s"),
        track.url,
    ]
    try:
        _run(command, dry_run=dry_run)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"download failed for {track.youtube_id} ({track.url}). "
            "YouTube may be rate-limiting this IP; retry later or pass a "
            "cookies file with yt-dlp."
        ) from exc

    if dry_run:
        return destination
    if not destination.exists():
        raise RuntimeError(f"yt-dlp reported success but {destination} is missing")
    return destination


def _write_manifest(tracks: list[Track], out: Path) -> dict:
    rows = []
    for track in tracks:
        path = out / "audio" / f"{track.youtube_id}.wav"
        if not path.exists():
            continue
        rows.append(
            {
                "song_id": track.youtube_id,
                "title": track.title,
                "singer_id": track.singer,
                "dialect": track.dialect,
                "style": track.style,
                "language": "Cree",
                "split": track.split,
                "duration_seconds": round(_duration(path), 3),
                "license": LICENCE,
                "license_url": LICENCE_URL,
                "consent_id": CONSENT_ID,
                "source_url": track.url,
                "source": "YouTube",
                "notes": track.notes,
            }
        )

    metadata = out / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    seconds = sum(row["duration_seconds"] for row in rows)
    manifest = {
        "license": LICENCE,
        "license_url": LICENCE_URL,
        "consent_id": CONSENT_ID,
        "tracks": len(rows),
        "hours": round(seconds / 3600, 3),
        "metadata": str(metadata),
        "rows": rows,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_readme(rows, out)
    return manifest


def _write_readme(rows: list[dict], out: Path) -> None:
    lines = [
        "# Cree singing corpus",
        "",
        "These are **real Cree singing recordings** published by their uploaders",
        "under YouTube's Creative Commons Attribution (reuse allowed) setting.",
        "They are not the all-rights-reserved commercial albums. Attribution and",
        "the source URL are kept in `metadata.jsonl`.",
        "",
        "| Track | Performer / group | Style | Minutes | Licence |",
        "|---|---|---|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: item["title"].lower()):
        title = str(row["title"]).replace("|", "\\|")
        singer = str(row["singer_id"]).replace("|", "\\|")
        style = str(row["style"]).replace("|", "\\|")
        minutes = row["duration_seconds"] / 60
        lines.append(
            f"| {title} | {singer} | {style} | {minutes:.2f} | CC BY |"
        )
    lines += [
        "",
        f"Total: **{len(rows)} recordings, "
        f"{sum(row['duration_seconds'] for row in rows) / 3600:.3f} hours**.",
        "",
        "The paired directory, when built with `--separate`, contains Demucs",
        "separations. They are useful training pairs but are not studio stems.",
        "Do not present a separation artifact as an original master.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _separate(
    track: Track,
    out: Path,
    paired: Path,
    *,
    device: str,
    segment: int = 0,
    shifts: int = 1,
    dry_run: bool = False,
) -> None:
    source = out / "audio" / f"{track.youtube_id}.wav"
    vocal = paired / "audio" / track.youtube_id / "vocals.wav"
    instrumental = paired / "audio" / track.youtube_id / "instrumental.wav"
    if vocal.exists() and instrumental.exists():
        print(f"have paired {track.youtube_id}")
        return
    if not source.exists():
        raise FileNotFoundError(f"download {track.youtube_id} before separating it")

    paired.mkdir(parents=True, exist_ok=True)
    work = out / "separated"
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "-d",
        device,
        "-o",
        str(work),
    ]
    if segment > 0:
        command += ["--segment", str(segment)]
    if shifts != 1:
        command += ["--shifts", str(shifts)]
    command.append(str(source))
    _run(command, dry_run=dry_run)
    if dry_run:
        return

    produced = work / "htdemucs" / track.youtube_id
    vocal_source = produced / "vocals.wav"
    instrumental_source = produced / "no_vocals.wav"
    if not vocal_source.exists() or not instrumental_source.exists():
        raise RuntimeError(f"Demucs did not produce both stems under {produced}")

    target = paired / "audio" / track.youtube_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocal_source, vocal)
    shutil.copy2(instrumental_source, instrumental)
    print(f"paired {track.youtube_id}: {_duration(vocal) / 60:.2f} min")


def _write_split_views(paired: Path) -> None:
    """Expose singer-disjoint split views without copying audio.

    ``prepare_dataset`` walks a plain directory tree; it does not read the
    metadata split. Giving it ``train_audio`` prevents the dev/test singers
    from leaking into training when the whole corpus has been separated.
    """
    metadata = paired / "metadata.jsonl"
    if not metadata.exists():
        return
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    for split in ("train", "dev", "test"):
        view = paired / f"{split}_audio"
        view.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row.get("split") != split:
                continue
            source = paired / "audio" / str(row["song_id"])
            target = view / str(row["song_id"])
            if source.exists() and not target.exists():
                target.symlink_to(source.resolve(), target_is_directory=True)


def _write_paired_metadata(tracks: list[Track], paired: Path) -> None:
    rows = []
    for track in tracks:
        if not (paired / "audio" / track.youtube_id / "vocals.wav").exists():
            continue
        rows.append(
            {
                "song_id": track.youtube_id,
                "singer_id": track.singer,
                "dialect": track.dialect,
                "style": track.style,
                "language": "Cree",
                "consent_id": CONSENT_ID,
                "license": LICENCE,
                "split": track.split,
                "source_url": track.url,
                "title": track.title,
                "separation": "demucs-htdemucs-two-stems",
            }
        )
    if rows:
        paired.mkdir(parents=True, exist_ok=True)
        (paired / "metadata.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_split_views(paired)


def _verify(out: Path, paired: Path) -> int:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}; run --download first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"mixed Cree songs: {manifest['tracks']} tracks, {manifest['hours']:.3f} h")
    paired_hours = 0.0
    paired_tracks = 0
    if (paired / "audio").exists():
        for directory in sorted((paired / "audio").iterdir()):
            vocal = directory / "vocals.wav"
            if vocal.exists():
                paired_hours += _duration(vocal) / 3600
                paired_tracks += 1
    print(f"paired Cree stems: {paired_tracks} tracks, {paired_hours:.3f} h")
    if paired_tracks and (paired / "metadata.jsonl").exists():
        print(f"paired metadata: {paired / 'metadata.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paired-root", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--download", action="store_true", help="download missing WAVs")
    parser.add_argument(
        "--separate-ids",
        default="",
        help="comma-separated YouTube IDs to Demucs-separate after download",
    )
    parser.add_argument(
        "--separate-all",
        action="store_true",
        help="separate every downloaded track (slow on CPU)",
    )
    parser.add_argument("--device", default="cpu", help="Demucs device, e.g. cpu or cuda")
    parser.add_argument(
        "--segment",
        type=int,
        default=0,
        help="Demucs segment seconds; lower it if the GPU runs out of memory",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Demucs random shifts; 1 is fastest, 2+ can improve separation",
    )
    parser.add_argument("--limit", type=int, default=0, help="download at most N tracks")
    parser.add_argument("--verify", action="store_true", help="print measured hours")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selected = list(TRACKS)
    if args.limit:
        selected = selected[: args.limit]

    if args.download:
        for track in selected:
            _download(track, args.out, dry_run=args.dry_run)

    if not args.dry_run:
        manifest = _write_manifest(selected, args.out)
        print(
            f"wrote {args.out / 'metadata.jsonl'}: "
            f"{manifest['tracks']} tracks, {manifest['hours']:.3f} h"
        )
        # Keep the paired manifest in sync with any stems already on disk,
        # even when this invocation only runs --verify.
        _write_paired_metadata(selected, args.paired_root)

    separate_ids = {
        item.strip() for item in args.separate_ids.split(",") if item.strip()
    }
    if args.separate_all:
        separate_ids.update(track.youtube_id for track in selected)
    if separate_ids:
        for track in selected:
            if track.youtube_id in separate_ids:
                _separate(
                    track,
                    args.out,
                    args.paired_root,
                    device=args.device,
                    segment=args.segment,
                    shifts=args.shifts,
                    dry_run=args.dry_run,
                )
        if not args.dry_run:
            _write_paired_metadata(selected, args.paired_root)

    if args.verify or not args.download:
        return _verify(args.out, args.paired_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

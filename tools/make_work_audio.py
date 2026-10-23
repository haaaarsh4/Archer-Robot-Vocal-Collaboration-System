#!/usr/bin/env python3
"""Put the songs and the performances where the site can serve them.

The About page's list is a player: a row's play button plays a real take from
this project, in the artist's own voice, and a row that has only footage opens
the work's video on the artist's own page instead. This script is the first
half of that, it copies everything Archer has delivered in
`music samples/shortlorezarchersongs/` into the frontend's own asset folder,
under the slug its row is keyed by:

  * frontend/assets/audio/ahasiw.mp3           Ahasiw
  * frontend/assets/audio/aim.mp3              AIM
  * frontend/assets/audio/alanise-obomsawin.mp3
  * frontend/assets/audio/halfbred.mp3         Halfbred, performance
  * frontend/assets/audio/horse.mp3            Horse, Winnipeg Art Gallery
  * frontend/assets/audio/hummingbird.mp3
  * frontend/assets/audio/memory-v2.mp3        Memory_V2, performance
  * frontend/assets/audio/muwekma-ohlone.mp3
  * frontend/assets/audio/sara-rocque.mp3
  * frontend/assets/audio/suzanne-kite.mp3

Copies rather than re-encodes, at the bitrate they were delivered in: Horse is
twelve megabytes and Halfbred thirty, full performances, and a re-encode of an
mp3 only ever loses. Sources are read only, and nothing is deleted from
`music samples/`.

Usage:  .venv/bin/python tools/make_work_audio.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "music samples" / "shortlorezarchersongs"
OUT = ROOT / "frontend" / "assets" / "audio"

# row slug -> the file in `music samples/` it is played from
RECORDINGS = {
    "ahasiw": "Ahasiw F 120.mp3",
    "aim": "AIM - Db 125.mp3",
    "alanise-obomsawin": "Alanise Obomsawin. Cm 144.mp3",
    "halfbred": "Halfbred performance by Archer Pechawis - Archer Pechawis (192k).mp3",
    "horse": "Horse - performance by Archer Pechawis Winnipeg Art Gallery - Archer Pechawis (192k).mp3",
    "hummingbird": "Hummingbird A (F#m)  160.mp3",
    "memory-v2": "Memory_V2 performance by Archer Pechawis - Archer Pechawis (192k).mp3",
    "muwekma-ohlone": "Muwekma Ohlone. Fm 130 bpm.mp3",
    "sara-rocque": "Sara Rocque Cm 100.mp3",
    "suzanne-kite": "Suzanne Kite Fm 120.mp3",
}


def duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    out = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)]
    )
    return float(out.decode().strip())


def main() -> int:
    missing = [n for n in RECORDINGS.values() if not (SRC / n).is_file()]
    if missing:
        print("not found under %s:" % SRC.relative_to(ROOT))
        for name in missing:
            print("  ", name)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    print("%-20s %-9s %9s %7s" % ("row", "recording", "length", "size"))
    total = 0
    for slug, name in RECORDINGS.items():
        src = SRC / name
        dst = OUT / (slug + ".mp3")
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        total += dst.stat().st_size
        print("%-20s %-9s %8.1fs %6dM" % (slug, "copied", duration(dst), dst.stat().st_size // (1024 * 1024)))

    print("\n%d recordings, %.1f MB -> %s" % (len(RECORDINGS), total / 1024 / 1024, OUT.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

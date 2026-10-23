"""Run the real /api/transcribe/annotated pipeline on a file and print the rows
it produces, so the labels can be read exactly as the panel would show them.

    .venv/bin/python scratch/diag_annotated_rows.py <path> [seconds-to-trim]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os

sys.path.insert(0, ".")
sys.path.insert(0, "tools")


def main() -> int:
    path = sys.argv[1]
    trim = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    work = path
    tmp = None
    if trim:
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-t", str(trim),
                        "-c", "copy", tmp], check=True)
        work = tmp
    with open(work, "rb") as fh:
        raw = fh.read()

    import time
    import server
    t0 = time.perf_counter()
    result = server._transcribe_track_annotated(raw, "en")
    print(f"\n=== {path} ===\n{len(result['segments'])} row(s) in "
          f"{time.perf_counter() - t0:.1f}s, has_lexical_speech={result['has_lexical_speech']}",
          flush=True)
    for s in result["segments"]:
        print(f"  {s['start']:6.2f}-{s['end']:6.2f}  type={s.get('type'):13s} "
              f"confident={s.get('confident')!s:5s}  {s['label']!r}", flush=True)
    if tmp:
        os.unlink(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

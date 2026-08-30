"""
Downloads the two universal RVC assets that rvc_server.py / rvc_pipeline.py
need to run at all, regardless of which trained voice you use:

  1. hubert_base.pt   -- the feature extractor every RVC voice conversion
                          runs audio through before it ever touches a
                          specific trained voice model.
  2. rmvpe.pt          -- the pitch-extraction model config.yaml points at
                          via pitch.rmvpe.model_path.

Both are published by the original RVC author (lj1995) on HuggingFace and
are the same files every RVC-based project uses -- they are NOT specific
to this project's trained voices (female2, mi-test), which have to be
supplied separately (see the README's "Neural singing voices" section).

Usage:
    python download_rvc_core_models.py

Safe to re-run: existing files are left alone.
"""

import os
import urllib.request

FILES = [
    (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "assets/hubert/hubert_base.pt",
        "~190MB",
    ),
    (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "assets/rmvpe/rmvpe.pt",
        "~181MB",
    ),
]


def _download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        print(f"\r  {pct:3d}%  ({done // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()


if __name__ == "__main__":
    for url, dest, size_hint in FILES:
        if os.path.exists(dest):
            print(f"Already present: {dest} -- skipping.")
            continue
        print(f"Downloading {os.path.basename(dest)} ({size_hint}) from {url}")
        try:
            _download(url, dest)
            print(f"  Saved to {dest}")
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  You can also download it by hand from {url} and place it at {dest}")

    print(
        "\nDone. rmvpe.pt's location (assets/rmvpe/rmvpe.pt) matches "
        "pitch.rmvpe.model_path in config.yaml exactly -- no further setup needed for that one.\n"
        "hubert_base.pt is saved to assets/hubert/hubert_base.pt, the conventional RVC "
        "location. rvc_pipeline.py loads it via infer.hubert.load_hubert_model(), which "
        "does not take an explicit path -- if that module has a different hardcoded path "
        "baked in, move (or symlink) the file there instead."
    )

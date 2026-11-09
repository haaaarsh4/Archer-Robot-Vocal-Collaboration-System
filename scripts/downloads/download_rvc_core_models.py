import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        ROOT / "assets" / "hubert" / "hubert_base.pt",
        "~190MB",
    ),
    (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        ROOT / "assets" / "rmvpe" / "rmvpe.pt",
        "~181MB",
    ),
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        print(f"\r  {pct:3d}%  ({done // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()


if __name__ == "__main__":
    for url, dest, size_hint in FILES:
        if dest.exists():
            print(f"Already present: {dest}, skipping.")
            continue
        print(f"Downloading {dest.name} ({size_hint}) from {url}")
        try:
            _download(url, dest)
            print(f"  Saved to {dest}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            print(f"  Download it by hand from {url} and place it at {dest}")

    print(
        "\nDone. assets/rmvpe/rmvpe.pt matches pitch.rmvpe.model_path in "
        "config.yaml. assets/hubert/hubert_base.pt is the conventional RVC "
        "location used by rvc_pipeline.py."
    )

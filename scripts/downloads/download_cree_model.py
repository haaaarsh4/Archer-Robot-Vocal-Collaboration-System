"""Download the Plains Cree speech models, the ones nêhiyawêwin transcription uses.

Whisper is English only -- `language="cr"` is the English decoder with a language
tag on it -- so the language Archer sings in is read by Meta's MMS stack instead:
one 1B acoustic model with a Plains Cree adapter per script, and the MMS-TTS crk
voice that `tools/check_cree_transcription.py` measures the transcriber against.

None of this is optional for the `tts` and `baseline` modes of that check, and
the acoustic model and its `crk` adapters are what `/api/cree/transcribe` needs.
Everything lands under `data/models/cree/`, in the layout
`analysis/cree_asr.py` looks in, so nothing has to be passed on the command line.

Run it once:

    python scripts/downloads/download_cree_model.py
"""

from __future__ import annotations

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "data" / "models" / "cree"

# (hub id, directory name under data/models/cree/, what it is for, rough size)
MODELS = [
    ("facebook/mms-1b-all", "facebook--mms-1b-all",
     "the acoustic model (both crk adapters are separate files inside it)", "4.0GB"),
    ("facebook/mms-tts-crk-script_latin", "facebook--mms-tts-crk-script_latin",
     "the Plains Cree voice the transcriber is measured against", "145MB"),
    ("facebook/mms-tts-crk-script_syllabics", "facebook--mms-tts-crk-script_syllabics",
     "the same voice written in syllabics, for the syllabics check", "145MB"),
    ("facebook/wav2vec2-lv-60-espeak-cv-ft", "facebook--wav2vec2-lv-60-espeak-cv-ft",
     "the phone recogniser behind the vocable reading", "1.2GB"),
]

# The per-language dictionaries MMS stores as `token count` text, which the
# tokenizer builder in analysis/cree_asr.py turns into the JSON transformers
# wants. They live in the repository, not in the main snapshot, so each one is
# fetched by name.
VOCABS = ["crk-script_latin.txt", "crk-script_syllabics.txt"]

# The 1B repository holds an adapter for each of its 1100+ languages, so the
# snapshot is asked for exactly the files this project reads: the base weights,
# the five small config files, and the two crk adapters. A bare `*.safetensors`
# here would pull every other language's adapter as well, which is more download
# than the base model itself and none of it used.
BASES: dict[str, list[str]] = {
    "facebook--mms-1b-all": [
        "config.json", "preprocessor_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "model.safetensors", "model.safetensors.index.json",
        "adapter.crk-script_latin.safetensors",
        "adapter.crk-script_syllabics.safetensors",
    ],
}


def _download(repo_id: str, name: str, what: str, size: str) -> None:
    from huggingface_hub import snapshot_download

    out = TARGET / name
    if (out / "config.json").exists() and any(out.glob("*.safetensors")):  # already there
        print(f"  {name}: already present, skipping")
        return
    print(f"Downloading {repo_id} ({size}) to {out} ...")
    print(f"  {what}")
    # allow_patterns rather than the whole repository: the 1B snapshot also
    # ships 1000+ other languages' adapters and their dictionaries, which are
    # 99 per cent of the download and none of what this project uses.
    snapshot_download(repo_id=repo_id, local_dir=str(out),
                      allow_patterns=BASES.get(name))
    print(f"  done -> {out}")


def _download_vocabs() -> None:
    from huggingface_hub import hf_hub_download

    dest = TARGET / "facebook--mms-1b-all" / "vocabs"
    dest.mkdir(parents=True, exist_ok=True)
    for name in VOCABS:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  vocabs/{name}: already present, skipping")
            continue
        print(f"Downloading vocabs/{name} ...")
        path = hf_hub_download("facebook/mms-1b-all", f"vocabs/{name}")
        shutil.copy(path, target)
        print(f"  done -> {target}")


if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    for repo_id, name, what, size in MODELS:
        _download(repo_id, name, what, size)
    _download_vocabs()
    print("\nAll Plains Cree speech models are in place. Check them with:")
    print("    python tools/check_cree_transcription.py health gold")
    print("The transcriber loads them lazily, so server.py starts either way -- "
          "/api/cree/transcribe/health says what is missing if anything is.")

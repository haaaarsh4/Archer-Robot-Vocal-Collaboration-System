#!/usr/bin/env python3
"""
Offline vocable bank builder.

Run this once, or whenever you want to refresh the vocable set. Not while
the robot is performing. It takes short raw takes of someone actually
saying or singing "oh", "ahh", "hey", "yeah" (or whatever set you pick) and
pushes each one through the neural voice pipeline that's already in the
project (the same one behind /api/chat/sing), exactly one time each, and
saves the converted result into synthesis/samples/neural/.

From that point on, the live synthesizer never touches the neural model at
all for these sounds. It only pitch-shifts the pre-rendered files (see
neural_vocable_bank.py), which is fast enough on CPU to keep up with a
live singer.

Setup:
    1. Make sure server.py is running with synthesis.neural.enabled: true
       in config.yaml, and the RVC sidecar (neural_env/rvc_server.py) is up
       and reachable. Check GET /neural/status if you're not sure.
    2. Record short, 1 to 3 second, raw takes and drop them in
       synthesis/source_takes/, named like:

           oh_low.wav    oh_mid.wav    oh_high.wav
           ahh_low.wav   ahh_mid.wav   ahh_high.wav
           hey_low.wav   hey_mid.wav   hey_high.wav
           yeah_low.wav  yeah_mid.wav  yeah_high.wav

       Say or sing them the way you actually want the robot to sound,
       real inflection, a real onset, not a flat monotone hum. This is
       the one place in the whole pipeline where a human performance
       actually matters, since everything downstream is just pitch
       shifting this original character around.

       "low/mid/high" don't need to be exact. They just tell the loader
       roughly where in the singing range each take sits, so it can pick
       the closest starting point later and shift less.

    3. Run:
           python synthesis/build_vocable_bank.py

    4. In config.yaml, set:
           synthesis:
             engine: neural_wavetable

You can rerun this any time you add or re-record a take. It only converts
files that don't already have a matching output, unless --force is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from loguru import logger

SOURCE_DIR = Path(__file__).parent / "source_takes"
OUTPUT_DIR = Path(__file__).parent / "samples" / "neural"


def build(server_url: str, voice_index: int, force: bool) -> None:
    if not SOURCE_DIR.exists():
        logger.error(
            f"No source takes found at {SOURCE_DIR}. Record a few short "
            "WAV takes there first, see the module docstring above for "
            "naming and what to actually say."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    takes = sorted(SOURCE_DIR.glob("*.wav"))
    if not takes:
        logger.error(f"{SOURCE_DIR} exists but has no .wav files in it yet.")
        return

    converted, skipped, failed = 0, 0, 0

    for take_path in takes:
        out_path = OUTPUT_DIR / take_path.name
        if out_path.exists() and not force:
            logger.info(f"Skipping {take_path.name} (already converted, pass --force to redo it)")
            skipped += 1
            continue

        logger.info(f"Converting {take_path.name} through the neural voice, this may take a bit...")
        try:
            with open(take_path, "rb") as f:
                resp = requests.post(
                    f"{server_url}/api/neural/convert-track",
                    files={"file": (take_path.name, f, "audio/wav")},
                    data={"voice_index": str(voice_index)},
                    timeout=300,
                )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.info(f"Saved {out_path}")
            converted += 1
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Couldn't convert {take_path.name}: {e}\n"
                "Check that server.py is running with the neural stage "
                "enabled and that neural_env/rvc_server.py is up. This "
                "script needs the neural pipeline running once, offline. "
                "The live singer path never calls it again after this."
            )
            failed += 1

    logger.info(f"Done. Converted {converted}, skipped {skipped}, failed {failed}.")
    if converted:
        logger.info(
            "Set synthesis.engine to 'neural_wavetable' in config.yaml to "
            "start using these instead of the plain formant synthesis."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8000",
                         help="Main app server URL (check the port your server.py actually runs on)")
    parser.add_argument("--voice-index", type=int, default=0, help="Which trained voice to use")
    parser.add_argument("--force", action="store_true", help="Reconvert files that already have output")
    args = parser.parse_args()
    build(args.server, args.voice_index, args.force)

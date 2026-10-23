"""Check the nêhiyawêwin transcription endpoint through the real app.

`tools/check_cree_transcription.py` measures the transcriber by calling it
directly, which is the right instrument for accuracy and the wrong one for the
question this file answers: does the thing the browser actually talks to work?
It mounts `server.app` with FastAPI's test client -- the same routes, the same
multipart upload, the same JSON encoding -- and posts real recordings to
`/api/cree/transcribe` and checks the answer against the line they read, so a
green result here means the endpoint is wired, the audio decodes off a *file on
disk* rather than a numpy array, and the JSON that comes back is the reading.

Run:

    python tools/check_cree_endpoint.py                 # the gold recordings
    python tools/check_cree_endpoint.py --file take.mp3 # any audio file
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from check_cree_transcription import cer, coverage_error, normalise  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GRADS = _ROOT / "data" / "PlainsCree" / "CreeLiteracyOrg" / "CongratsGrads"


def _gold_cases(limit: int | None) -> list[tuple[pathlib.Path, str]]:
    lines = [ln.strip() for ln in
             (_GRADS / "congrats_grads_cr.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    cases: list[tuple[pathlib.Path, str]] = []
    for path in sorted(_GRADS.iterdir()):
        if path.suffix.lower() not in (".m4a", ".wav", ".mp3"):
            continue
        # A file can hold more than one line -- `Grad_SRatt_line7_8.m4a` is lines
        # 7 and 8 read together -- so both numbers in the name are part of the
        # reference, exactly as `gold_pairs` reads them in the main harness.
        match = re.search(r"line(\d+)(?:_(\d+))?", path.stem)
        if not match:
            continue
        chosen = []
        for group in match.groups():
            if group and int(group) - 1 < len(lines):
                chosen.append(lines[int(group) - 1])
        if chosen:
            cases.append((path, " ".join(chosen)))
    return cases[:limit] if limit else cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action="append", default=None,
                        help="audio file to post instead of the gold recordings")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("importing server.py (this loads the app's models) ...")
    from fastapi.testclient import TestClient

    import server

    client = TestClient(server.app)

    health = client.get("/api/cree/transcribe/health")
    print(f"  GET /api/cree/transcribe/health -> {health.status_code}")
    if health.status_code != 200:
        print(f"  {health.text[:300]}")
        return 2
    body = health.json()
    print(f"  engine: {body.get('engine')}")
    print(f"  text model: {body.get('text_model')}")
    if body.get("status") != "ok":
        print("  the Cree transcriber is not available; nothing to post to")
        return 2

    if args.file:
        cases = [(pathlib.Path(f), "") for f in args.file]
    else:
        cases = _gold_cases(args.limit)

    print(f"\nposting {len(cases)} file(s) to /api/cree/transcribe")
    failures = 0
    for path, reference in cases:
        with open(path, "rb") as handle:
            response = client.post(
                "/api/cree/transcribe",
                files={"file": (path.name, handle, "application/octet-stream")})
        if response.status_code != 200:
            print(f"  {path.name}: HTTP {response.status_code} {response.text[:200]}")
            failures += 1
            continue
        data = response.json()
        sro = data.get("sro", "")
        keys = sorted(data.keys())
        print(f"  {path.name}")
        print(f"      sro:        {sro[:110]}")
        print(f"      syllabics:  {data.get('syllabics', '')[:70]}")
        print(f"      anglicized: {data.get('anglicized', '')[:90]}")
        print(f"      keys: {keys}")
        print(f"      duration {data.get('duration')}s  segments "
              f"{len(data.get('segments', []))}  engine {data.get('engine')}  "
              f"rtf {round((data.get('timing') or {}).get('realtime_factor', 0), 2)}x")
        # The three scripts have to be three scripts: an endpoint that answered
        # the same string three times would pass a naive check and be useless.
        if not sro or not data.get("syllabics") or not data.get("anglicized"):
            print("      FAIL: a script came back empty")
            failures += 1
        elif data["syllabics"] == sro:
            print("      FAIL: syllabics came back identical to SRO")
            failures += 1
        if reference:
            print(f"      ref: {normalise(reference)}")
            print(f"      CER {cer(reference, sro)*100:5.1f}%  "
                  f"coverage {coverage_error(reference, sro)*100:5.1f}%")
    # The page asks for this and shows it, so it has to be answerable.
    missing = client.post("/api/cree/transcribe", files={})
    print(f"\n  a request with no file -> {missing.status_code} (422 is correct)")
    print(f"\n{len(cases) - failures}/{len(cases)} file(s) came back with a reading")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

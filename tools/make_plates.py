#!/usr/bin/env python
"""Cut Archer's photographs into the home page's plate assets.

The home strip is the reference site's, value for value: a plate is 12% of the
window height wide and 449.2307692308% of that tall, and it is a *window* onto
a piece of art twice as wide (24% of the window height), so

    the resting plate shows   the middle 50% of the art,  ratio 1 : 4.4923
    the focused plate shows   all of the art,             ratio 1 : 2.2458

Both cuts have to read well, so each source photo is cropped to the art ratio
(2.2458) with the subject kept centred, centred precisely, because the middle
half of that crop is what a resting plate shows and 11% of the frame either way
is all the room there is.

Everything is driven by PLATES, one line per card, in strip order. `cx` and
`cy` are where the crop's centre sits in the *source* (0-1, so 0.5 is dead
centre) and `zoom` tightens the cut (1.0 keeps the tallest possible crop).

    ./venv/bin/python tools/make_plates.py            # write the plates
    ./venv/bin/python tools/make_plates.py --preview  # + a strip contact sheet

Sources live in frontend/images/ and are never modified; the derived art goes
to frontend/assets/art/plate-01..07.jpg, which is what the home page loads.
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(ROOT, "frontend", "images")
OUT_DIR = os.path.join(ROOT, "frontend", "assets", "art")

ART_RATIO = 24.0 / (12.0 * 4.492307692308)   # the media box: 1 : 2.2458
ART_W = 644                                   # the reference serves 644 x 1448
ART_H = 1448

# strip order, section -> photograph. `zoom` > 1 cuts in tighter, `cx`/`cy` are
# where the crop's centre sits in the source. Every one of these is picked so
# the *middle half* of the cut, all a resting plate ever shows, still holds
# the subject's face or hands, and so the whole cut reads when it opens.
PLATES = [
    # 01 the practice: the seated portrait, the most recognisable frame of him
    dict(out="plate-01.jpg", src="archer1.jpeg", cx=0.50, cy=0.50, zoom=1.06),
    # 02 the system, nine stages: the solo juggler, timing and balance made visible
    dict(out="plate-02.jpg", src="archer2.jpeg", cx=0.50, cy=0.50, zoom=1.15),
    # 03 three Cree voicings under one voice: the troupe, many voices at once
    dict(out="plate-03.jpg", src="archer6.jpeg", cx=0.50, cy=0.50, zoom=1.25),
    # 04 syllables held and shaped: the open mouth, caught mid-note
    dict(out="plate-04.jpg", src="archer8.jpeg", cx=0.52, cy=0.50, zoom=1.15),
    # 05 meaning under the words: the torch bearer, theatrical, sepia
    dict(out="plate-05.jpg", src="archer9.jpeg", cx=0.50, cy=0.50, zoom=1.10),
    # 06 the bot: the clown juggling fire in the doorway, a figure mid-answer
    dict(out="plate-06.jpg", src="archer4.jpeg", cx=0.50, cy=0.50, zoom=1.15),
    # 07 the demo, the voice as fire: the fire eater, torches at the mouth
    dict(out="plate-07.jpg", src="archer3.jpeg", cx=0.37, cy=0.40, zoom=1.15),
]


def cut(im: Image.Image, cx: float, cy: float, zoom: float) -> Image.Image:
    """Crop *im* to the art ratio, centred on (cx, cy) of the source."""
    w, h = im.size
    # the tallest crop that fits, then tightened by the zoom
    cw = min(float(w), h * ART_RATIO) / max(zoom, 0.01)
    ch = cw / ART_RATIO
    if cw > w or ch > h:                      # keep it inside the source
        k = min(w / cw, h / ch)
        cw, ch = cw * k, ch * k
    x = min(max(cx * w - cw / 2.0, 0.0), w - cw)
    y = min(max(cy * h - ch / 2.0, 0.0), h - ch)
    box = (int(round(x)), int(round(y)), int(round(x + cw)), int(round(y + ch)))
    return im.crop(box).resize((ART_W, ART_H), Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="also write /tmp/plate_preview.jpg: the plates cut "
                         "down to what a resting window shows, side by side")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    resting = []
    for n, p in enumerate(PLATES, start=1):
        src = os.path.join(SRC_DIR, p["src"])
        if not os.path.exists(src):
            print("missing source: %s" % src, file=sys.stderr)
            return 1
        im = Image.open(src).convert("RGB")
        art = cut(im, p["cx"], p["cy"], p["zoom"])
        # a touch of contrast: the strip is washed out at rest and this is what
        # comes back up at full strength when a plate opens
        art = ImageEnhance.Contrast(art).enhance(1.04)
        dest = os.path.join(OUT_DIR, p["out"])
        art.save(dest, "JPEG", quality=80, optimize=True, progressive=True)
        # what a resting plate shows: the middle half of the art
        mid = art.crop((ART_W // 4, 0, ART_W * 3 // 4, ART_H))
        resting.append(mid)
        print("%-13s <- %-13s %s  %d KB" % (
            p["out"], p["src"], "%dx%d" % art.size, os.path.getsize(dest) // 1024))

    if args.preview:
        # row 1: what a resting plate shows (the middle half of the art)
        # row 2: what a focused plate shows (all of it)
        rest_cell = (96, 431)
        open_cell = (192, 431)
        w = 8 + sum(rest_cell[0] + 8 for _ in resting) + 16 + sum(open_cell[0] + 8 for _ in resting)
        sheet = Image.new("RGB", (w, rest_cell[1] + 24), "#e7e5df")
        x = 8
        for mid in resting:
            sheet.paste(mid.resize(rest_cell, Image.LANCZOS), (x, 8))
            x += rest_cell[0] + 8
        x += 16
        for i, p in enumerate(PLATES):
            art = Image.open(os.path.join(OUT_DIR, p["out"])).convert("RGB")
            sheet.paste(art.resize(open_cell, Image.LANCZOS), (x, 8))
            x += open_cell[0] + 8
        out = "/tmp/plate_preview.jpg"
        sheet.save(out, quality=90)
        print("preview (resting strip | focused plates) -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

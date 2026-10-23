#!/usr/bin/env python3
"""Generate the site's artwork locally (no external images, no licences).

Everything the redesigned site shows is drawn here, from scratch, in one
consistent screen-print art direction:

  * frontend/assets/art/poster-NN.jpg    tall filmstrip cards for the home page
                                         (512 x 1080, the focused card's shape)
  * frontend/assets/art/banner-*.jpg     wide plates for the inner page headers
  * frontend/assets/art/thumb-*.jpg      small square tiles for the work list

The motifs are Plains Cree visual language, four-direction stars, beadwork
chevrons, hand-drum rings, thunderbird zigzags, star-blanket diamond fields,
Cree syllabics, combined with signal motifs from this project (waveforms,
pipeline nodes). Text is set in Inter Tight (system) with syllabics in
Noto Sans Canadian Aboriginal.

Usage:  .venv/bin/python tools/make_art.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "assets" / "art"

DISPLAY_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
DISPLAY_MED = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
DISPLAY_REG = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SYLLABICS = "/usr/share/fonts/truetype/noto/NotoSansCanadianAboriginal-Bold.ttf"

# ================================================================ palettes
PALETTES = {
    "ochre": {
        "deep": (168, 62, 34),
        "mid": (206, 122, 60),
        "light": (233, 205, 158),
        "paper": (242, 232, 213),
        "ink": (26, 22, 18),
        "pop": (47, 107, 94),
    },
    "indigo": {
        "deep": (36, 46, 88),
        "mid": (78, 104, 156),
        "light": (186, 202, 224),
        "paper": (231, 236, 240),
        "ink": (18, 20, 30),
        "pop": (206, 156, 66),
    },
    "sage": {
        "deep": (66, 92, 72),
        "mid": (140, 160, 128),
        "light": (208, 216, 190),
        "paper": (238, 239, 228),
        "ink": (20, 24, 18),
        "pop": (176, 74, 48),
    },
    "rust": {
        "deep": (124, 48, 30),
        "mid": (196, 108, 68),
        "light": (226, 194, 168),
        "paper": (244, 234, 222),
        "ink": (24, 18, 15),
        "pop": (58, 90, 122),
    },
}


# ================================================================ helpers
def fractal(size: tuple[int, int], octaves: int = 6, seed: int = 1, gain: float = 0.52) -> np.ndarray:
    """Multi-octave value noise in 0..1, the paper/stone/cloud base."""
    rng = np.random.default_rng(seed)
    w, h = size
    out = np.zeros((h, w), dtype=np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(2 ** (o + 1)))
        gh = max(2, int(gw * h / w))
        g = rng.random((gh, gw)).astype(np.float32)
        layer = np.asarray(
            Image.fromarray((g * 255).astype(np.uint8), "L")
            .resize((w, h), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(1.0 + o * 0.7)),
            dtype=np.float32,
        ) / 255.0
        out += layer * amp
        tot += amp
        amp *= gain
    out /= tot
    lo, hi = float(out.min()), float(out.max())
    return (out - lo) / max(1e-6, hi - lo)


def duotone(noise: np.ndarray, dark: tuple, light: tuple, curve: float = 1.0) -> Image.Image:
    n = np.clip(noise, 0.0, 1.0) ** curve
    a = np.array(dark, dtype=np.float32)
    b = np.array(light, dtype=np.float32)
    arr = a[None, None, :] * (1 - n[..., None]) + b[None, None, :] * n[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def lightshaft(img: Image.Image, angle_deg: float, width: float, color, strength: float, center: float = 0.4) -> None:
    """A soft diagonal band of light, the thing that makes flat art look lit."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = (xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a)
    m = np.exp(-((u - (center - 0.5)) ** 2) / (2 * (width / 2) ** 2)) * strength
    arr = np.asarray(img, dtype=np.float32)
    tint = np.array(color, dtype=np.float32)
    arr = arr * (1 - m[..., None]) + tint[None, None, :] * m[..., None]
    img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), (0, 0))


def scanlines(img: Image.Image, step: int = 3, alpha: float = 0.05) -> None:
    arr = np.asarray(img, dtype=np.float32)
    mask = (np.arange(img.height) % step == 0).astype(np.float32)[:, None, None]
    arr *= (1.0 - mask * alpha)
    img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), (0, 0))


# Moire-free washed base: dark value structure + one saturated pop band.
def poster_base(size: tuple[int, int], pal: dict, seed: int, warmth: float = 0.5) -> Image.Image:
    w, h = size
    deep = np.array(pal["deep"], dtype=np.float32)
    ink = np.array(pal["ink"], dtype=np.float32)
    paper = np.array(pal["paper"], dtype=np.float32)
    mid = np.array(pal["mid"], dtype=np.float32)

    n1 = fractal(size, 6, seed)                      # broad tonal structure
    n2 = fractal(size, 7, seed + 101, gain=0.62)     # detail
    struct = np.clip(n1 * 0.68 + n2 * 0.32, 0, 1)

    # value curve: keep a dark third, a mid third and a lit third
    v = np.clip((struct - 0.30) / 0.55, 0, 1) ** 0.9
    col = (
        ink[None, None, :] * (1 - v[..., None]) ** 2
        + deep[None, None, :] * (2 * v * (1 - v))[..., None] * (0.6 + warmth)
        + paper[None, None, :] * (v[..., None] ** 2) * 0.85
    )
    # a saturated band bleeding through the middle
    band = np.exp(-((np.linspace(0, 1, h)[:, None] - 0.52) ** 2) / (2 * 0.10 ** 2)).astype(np.float32)
    col = col * (1 - band[..., None] * 0.30) + mid[None, None, :] * (band[..., None] * 0.30)
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
    lightshaft(img, 62 + 20 * (seed % 3), 0.42, tuple(paper.astype(int)), 0.16, 0.30 + 0.2 * ((seed % 5) / 5))
    return img


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def text_w(draw: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont, tracking: float = 0.0) -> float:
    if not s:
        return 0.0
    return draw.textlength(s, font=f) + tracking * (len(s) - 1)


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    s: str,
    f: ImageFont.FreeTypeFont,
    fill,
    tracking: float = 0.0,
    anchor_y: str = "la",
) -> float:
    """Draw text with manual letter-spacing. Returns the x advance."""
    x, y = xy
    for ch in s:
        draw.text((x, y), ch, font=f, fill=fill, anchor="l" + anchor_y[1])
        x += draw.textlength(ch, font=f) + tracking
    return x


def center_tracked(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    s: str,
    f: ImageFont.FreeTypeFont,
    fill,
    tracking: float = 0.0,
) -> None:
    x = cx - text_w(draw, s, f, tracking) / 2
    draw_tracked(draw, (x, y), s, f, fill, tracking)


def wrap(draw: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=f) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def vertical_gradient(size: tuple[int, int], top, bottom) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_a = np.array(top, dtype=np.float32)
    bot_a = np.array(bottom, dtype=np.float32)
    col = top_a[None, :] * (1 - t) + bot_a[None, :] * t
    arr = np.repeat(col[:, None, :], w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def glow(img: Image.Image, cx: float, cy: float, rx: float, ry: float, color, strength: float) -> None:
    """Soft radial bloom added in place."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    m = np.clip(1.0 - d, 0.0, 1.0) ** 2 * strength
    arr = np.asarray(img, dtype=np.float32)
    tint = np.array(color, dtype=np.float32)
    arr = arr * (1 - m[..., None]) + tint[None, None, :] * m[..., None]
    img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), (0, 0))


def halftone(img: Image.Image, step: int = 8, alpha: float = 0.055, color=(0, 0, 0)) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for y in range(0, img.height, step):
        for x in range(0, img.width, step):
            r = 1.1 if ((x // step + y // step) % 2 == 0) else 0.7
            d.ellipse([x, y, x + 2 * r, y + 2 * r], fill=color + (int(255 * alpha),))
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def grain(img: Image.Image, amount: float = 7.0, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, amount, arr.shape[:2])[..., None]
    img.paste(Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB"), (0, 0))


def vignette(img: Image.Image, strength: float = 0.30) -> None:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    r = np.sqrt(nx**2 * 0.7 + ny**2)
    m = np.clip((r - 0.55) / 0.75, 0, 1) ** 1.6 * strength
    arr = np.asarray(img, dtype=np.float32)
    arr *= (1.0 - m[..., None])
    img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), (0, 0))


# ================================================================ motifs
def motif_star(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float, col, col2, lw: int, pts: int = 8) -> None:
    """Four-direction star (medicine wheel) built from two rotated squares."""
    for phase, c in ((0, col), (math.pi / pts, col2)):
        poly = []
        for i in range(pts * 2):
            rr = r if i % 2 == 0 else r * 0.42
            a = phase + i * math.pi / pts
            poly.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(poly, fill=c)
    d.ellipse([cx - r * 0.30, cy - r * 0.30, cx + r * 0.30, cy + r * 0.30], outline=col2, width=lw)
    d.ellipse([cx - r * 1.32, cy - r * 1.32, cx + r * 1.32, cy + r * 1.32], outline=col, width=max(2, lw // 2))


def motif_chevrons(d: ImageDraw.ImageDraw, w: int, y0: float, y1: float, col, col2, lw: int, bands: int = 9) -> None:
    """Beadwork chevron bands stacked down the card."""
    span = (y1 - y0) / bands
    for i in range(bands):
        y = y0 + i * span
        c = col if i % 2 == 0 else col2
        step = w / 7.0
        x = -step
        pts = []
        up = True
        while x <= w + step:
            pts.append((x, y + (0 if up else span * 0.42)))
            x += step
            up = not up
        d.line(pts, fill=c, width=lw, joint="curve")


def motif_wave(d: ImageDraw.ImageDraw, w: int, h: int, y: float, amp: float, col, lw: int, cycles: float = 1.6, phase: float = 0.0) -> None:
    pts = []
    for i in range(0, w + 2, 2):
        t = i / w
        pts.append((i, y + math.sin(t * cycles * 2 * math.pi + phase) * amp * (0.35 + 0.65 * math.sin(t * math.pi))))
    d.line(pts, fill=col, width=lw, joint="curve")


def motif_rings(d: ImageDraw.ImageDraw, cx: float, cy: float, r0: float, r1: float, col, lw: int, n: int = 6) -> None:
    for i in range(n):
        r = r0 + (r1 - r0) * i / max(1, n - 1)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=lw)


def motif_diamonds(d: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, col, col2, cell: float, lw: int) -> None:
    """Star-blanket diamond field."""
    i = 0
    y = y0
    while y < y1:
        x = x0
        j = 0
        while x < x1:
            r = cell * (0.5 if (i + j) % 2 else 0.34)
            c = col if (i + j) % 2 else col2
            d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], outline=c, width=lw)
            x += cell
            j += 1
        y += cell
        i += 1


def motif_teepees(d: ImageDraw.ImageDraw, w: int, y0: float, y1: float, col, col2, lw: int, n: int = 4) -> None:
    step = w / n
    for i in range(n):
        cx = step * (i + 0.5)
        h = (y1 - y0) * (0.62 if i % 2 else 0.92)
        base = y1
        d.line([(cx - h * 0.42, base), (cx, base - h), (cx + h * 0.42, base)], fill=col if i % 2 else col2, width=lw)
        d.line([(cx, base - h), (cx, base)], fill=col2, width=max(1, lw // 2))


def motif_syllabics(img: Image.Image, text: str, size: int, col, x: float, y: float, tracking: float = 0.0) -> None:
    d = ImageDraw.Draw(img)
    f = font(SYLLABICS, size)
    draw_tracked(d, (x, y), text, f, col, tracking)


def motif_pipeline(d: ImageDraw.ImageDraw, w: int, h: int, col, col2, lw: int) -> None:
    """Little node/stage graph, the system's own shape."""
    nodes = [(0.16, 0.30), (0.5, 0.22), (0.84, 0.34), (0.30, 0.52), (0.68, 0.60), (0.5, 0.80)]
    f = font(MONO, 15)
    for i, (nx, ny) in enumerate(nodes):
        x, y = nx * w, ny * h
        r = 15
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        d.text((x, y), f"{i+1:02d}", font=f, fill=col2, anchor="mm")
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (3, 5), (5, 4)]:
        ax, ay = nodes[a][0] * w, nodes[a][1] * h
        bx, by = nodes[b][0] * w, nodes[b][1] * h
        d.line([(ax, ay), (bx, by)], fill=col2, width=max(1, lw // 3))


# ================================================================ still
# The home filmstrip card. Sized for the shape it is actually shown at: at
# rest the card is half as wide as when it takes focus, so the artwork is
# composed on the focused shape and the resting card shows its middle half.
# Nothing important may sit in the outer quarters, and nothing is cropped at
# the smallest size the card is ever drawn at.
STILL = (512, 1080)


def still_field(size: tuple[int, int], pal: dict, seed: int) -> Image.Image:
    """A high key plate: pale paper, a soft horizon, one quiet glow."""
    w, h = size
    paper = np.array(pal["paper"], dtype=np.float32)
    light = np.array(pal["light"], dtype=np.float32)
    mid = np.array(pal["mid"], dtype=np.float32)
    deep = np.array(pal["deep"], dtype=np.float32)

    n = fractal(size, 6, seed, gain=0.56)
    n = np.clip(0.5 + (n - 0.5) * 0.8, 0, 1)
    col = (
        paper[None, None, :] * (1 - n[..., None] * 0.52)
        + light[None, None, :] * (n[..., None] * 0.52)
    )
    # a soft horizon two thirds down, and a slightly deeper floor under it
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    hz = np.clip((t - 0.50) / 0.30, 0, 1) ** 1.5
    col = col * (1 - hz * 0.38) + mid[None, None, :] * (hz * 0.38)
    sky = np.clip((0.42 - t) / 0.36, 0, 1) ** 1.3
    col = col * (1 - sky * 0.20) + deep[None, None, :] * (sky * 0.20)
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
    lightshaft(img, 74.0, 0.30, pal["paper"], 0.20, 0.40 + 0.14 * ((seed % 5) / 5.0))
    glow(img, w * 0.5, h * 0.36, w * 0.72, h * 0.26, pal["paper"], 0.52)
    glow(img, w * 0.5, h * 0.76, w * 0.60, h * 0.18, pal["light"], 0.24)
    vignette(img, 0.30)
    return img


def still(art: str, pal_name: str, size: tuple[int, int] = STILL, seed: int = 3) -> Image.Image:
    """One filmstrip card, drawn on its own, no type, no scrim."""
    p = PALETTES[pal_name]
    w, h = size
    img = still_field(size, p, seed)
    d = ImageDraw.Draw(img)
    lw = max(2, int(w * 0.0075))
    cy = h * 0.45
    ink = p["deep"]

    if art == "star":
        motif_rings(d, w / 2, cy, w * 0.40, w * 1.55, ink, max(1, lw // 2), 5)
        motif_star(d, w / 2, cy, w * 0.36, p["light"], ink, lw)
        motif_star(d, w / 2, cy, w * 0.16, p["pop"], ink, lw)
    elif art == "chevrons":
        motif_chevrons(d, w, h * 0.20, h * 0.52, ink, p["pop"], lw, 7)
        motif_chevrons(d, w, h * 0.60, h * 0.88, ink, p["pop"], lw, 6)
    elif art == "wave":
        for k, (amp, ph, col) in enumerate([(0.030, 0.0, ink), (0.062, 1.1, p["pop"]), (0.100, 2.2, ink)]):
            motif_wave(d, w, h, cy + h * (0.085 * k - 0.085), h * amp, col, lw + 2, 1.05, ph)
        m = w * 0.10
        d.ellipse([w * 0.25 - m, cy + h * 0.24 - m, w * 0.25 + m, cy + h * 0.24 + m], fill=p["paper"])
        d.ellipse([w * 0.25 - m * 0.5, cy + h * 0.24 - m * 0.5, w * 0.25 + m * 0.5, cy + h * 0.24 + m * 0.5], fill=p["pop"])
        d.line([(w * 0.33, cy + h * 0.24), (w * 0.78, cy + h * 0.24)], fill=ink, width=max(1, lw // 2))
    elif art == "drums":
        motif_rings(d, w / 2, cy, w * 0.24, w * 0.50, ink, lw)
        motif_star(d, w / 2, cy, w * 0.16, ink, p["pop"], max(2, lw // 2), 4)
        motif_chevrons(d, w, h * 0.66, h * 0.88, ink, p["pop"], lw, 6)
    elif art == "diamonds":
        motif_diamonds(d, -w * 0.08, h * 0.16, w * 1.08, h * 0.70, ink, p["pop"], w / 2.6,
                       max(1, lw // 2))
        motif_star(d, w / 2, h * 0.43, w * 0.20, p["pop"], ink, max(2, lw // 2), 4)
    elif art == "syllabics":
        motif_syllabics(img, "\u14c0\u1426\u1421", int(w * 0.52), ink, w * 0.06, h * 0.24)
        motif_syllabics(img, "\u1526\u142f", int(w * 0.38), p["pop"], w * 0.14, h * 0.40)
        d = ImageDraw.Draw(img)
        motif_wave(d, w, h, h * 0.62, h * 0.010, ink, max(2, lw // 2), 2.2)
    elif art == "pipeline":
        motif_pipeline(d, w, int(h * 0.56), ink, p["pop"], max(2, lw + 1))
    elif art == "teepees":
        motif_teepees(d, w, h * 0.20, h * 0.60, ink, p["pop"], lw, 4)
        motif_wave(d, w, h, h * 0.74, h * 0.020, ink, max(2, lw // 2), 1.4)
    elif art == "rings":
        motif_rings(d, w / 2, cy, w * 0.16, w * 0.62, ink, max(1, lw // 2), 7)
        motif_star(d, w / 2, cy, w * 0.13, p["pop"], ink, max(2, lw // 2), 6)
        motif_wave(d, w, h, h * 0.70, h * 0.016, ink, max(2, lw // 2), 1.2)
    else:
        motif_wave(d, w, h, cy, h * 0.05, ink, lw)

    # The card is cropped to its middle when it sits in the filmstrip (the
    # focused card shows the whole plate), so every important element is
    # centred horizontally and nothing is left only in the outer quarters.

    # a foot: one hairline across the plate with a single mark on it, so the
    # lower third is not dead space when the card is the focused one
    if art not in ("chevrons", "drums"):
        fy = h * 0.80
        d.line([(w * 0.16, fy), (w * 0.84, fy)], fill=ink, width=max(1, lw // 2))
        r = max(3, int(w * 0.017))
        mx = w * (0.28 + 0.14 * ((seed % 4) / 3.0))
        d.ellipse([mx - r, fy - r, mx + r, fy + r], fill=p["pop"])

    vignette(img, 0.20)
    grain(img, 5.0, seed=abs(hash(art)) % 9999)
    return img


# ================================================================ banners
def banner(art: str, pal_name: str, word: str, size=(1680, 900), seed: int = 9) -> Image.Image:
    p = PALETTES[pal_name]
    w, h = size
    img = poster_base(size, p, seed, warmth=0.45)
    glow(img, w * 0.20, h * 0.40, w * 0.5, h * 0.8, p["mid"], 0.42)
    glow(img, w * 0.88, h * 0.16, w * 0.34, h * 0.5, p["light"], 0.34)
    scrim = Image.new("RGBA", size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    for i in range(120):
        xx = w - 1 - i
        a = int(170 * (i / 119.0) ** 1.4)
        sd.line([(xx, 0), (xx, h)], fill=p["ink"] + (a,))
    img = Image.alpha_composite(img.convert("RGBA"), scrim).convert("RGB")
    d = ImageDraw.Draw(img)
    lw = 10

    if art == "star":
        motif_star(d, w * 0.30, h * 0.48, w * 0.13, p["light"], p["pop"], lw)
    elif art == "wave":
        for k, (amp, ph, col) in enumerate([(0.10, 0.0, p["light"]), (0.17, 1.1, p["pop"])]):
            motif_wave(d, w, h, h * (0.62 + 0.08 * k), h * amp, col, lw + 3, 1.1, ph)
    elif art == "chevrons":
        motif_chevrons(d, int(w * 0.52), h * 0.18, h * 0.72, p["light"], p["pop"], lw)
    elif art == "rings":
        motif_rings(d, w * 0.28, h * 0.5, w * 0.04, w * 0.22, p["light"], lw, 7)
    elif art == "diamonds":
        motif_diamonds(d, w * 0.55, h * 0.12, w * 1.02, h * 0.9, p["light"], p["pop"], w / 11.0, 3)
    elif art == "syllabics":
        motif_syllabics(img, "\u14c0\u1426\u1421\u140a", int(w * 0.13), p["light"], w * 0.05, h * 0.14)
        d = ImageDraw.Draw(img)
    elif art == "plates":
        rng = np.random.default_rng(5)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        for i in range(9):
            bw = int(w * rng.uniform(0.18, 0.42))
            bh = int(h * rng.uniform(0.25, 0.5))
            x = int(w * 0.45 + rng.uniform(-0.1, 0.5) * w)
            y = int(h * rng.uniform(0.05, 0.6))
            col = [p["light"], p["pop"], p["paper"], p["mid"]][i % 4]
            ImageDraw.Draw(layer).rectangle([x, y, x + bw, y + bh], fill=col + ((140 if i % 2 else 80),))
        img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))
    else:
        motif_wave(d, w, h, h * 0.5, h * 0.12, p["light"], lw)

    f = font(DISPLAY_BOLD, int(h * 0.24))
    lines = wrap(d, word, f, w * 0.62)
    lh = int(h * 0.24 * 1.02)
    y = h * 0.86 - lh * (len(lines) - 1)
    for ln in lines:
        draw_tracked(d, (w * 0.06, y), ln, f, p["paper"], -3.0)
        y += lh
    draw_tracked(d, (w * 0.06, h * 0.06), "ARCHER PECHAWIS \u00b7 VOCAL COLLABORATION SYSTEM",
                 font(DISPLAY_MED, int(h * 0.030)), p["paper"], 5.0)
    halftone(img, 9, 0.055, (0, 0, 0))
    scanlines(img, 3, 0.04)
    vignette(img, 0.34)
    grain(img, 6.5, seed=len(word))
    return img


# ================================================================ thumbs
def thumb(art: str, pal_name: str, size=(160, 160), label: str = "", seed: int = 2) -> Image.Image:
    p = PALETTES[pal_name]
    img = poster_base(size, p, seed, warmth=0.4)
    glow(img, size[0] * 0.5, size[1] * 0.35, size[0] * 0.9, size[1] * 0.8, p["mid"], 0.42)
    d = ImageDraw.Draw(img)
    w, h = size
    if art == "star":
        motif_star(d, w / 2, h / 2, w * 0.26, p["light"], p["pop"], 3)
    elif art == "wave":
        motif_wave(d, w, h, h * 0.55, h * 0.16, p["light"], 4, 1.2)
    elif art == "chevrons":
        motif_chevrons(d, w, h * 0.3, h * 0.72, p["light"], p["pop"], 4, 4)
    elif art == "diamonds":
        motif_diamonds(d, -4, h * 0.16, w + 4, h * 0.86, p["light"], p["pop"], w / 3.4, 2)
    elif art == "rings":
        motif_rings(d, w / 2, h / 2, w * 0.1, w * 0.44, p["light"], 3, 4)
    else:
        motif_wave(d, w, h, h * 0.5, h * 0.2, p["light"], 4)
    if label:
        d.text((w / 2, h - 12), label, font=font(MONO, 15), fill=p["paper"], anchor="mm")
    grain(img, 5.0, seed=len(art) + 3)
    return img


# ================================================================ content
POSTERS = [
    # motif, palette, section
    ("wave", "ochre", "demo"),
    ("chevrons", "sage", "harmony"),
    ("syllabics", "indigo", "synthesis"),
    ("star", "rust", "translation"),
    ("diamonds", "ochre", "chat"),
    ("pipeline", "indigo", "architecture"),
    ("drums", "sage", "about"),
    # the works that sit at the end of the home strip
    ("teepees", "rust", "horse"),
    ("rings", "indigo", "indian-act"),
    ("drums", "indigo", "memory-v2"),
]

BANNERS = [
    ("wave", "ochre", "Sing with Archer", "sing-with-archer"),
    ("star", "rust", "The Artist", "artist"),
    ("bands", "sage", "The Works", "works"),
    ("syllabics", "indigo", "Transitional Cree Culture", "culture"),
    ("rings", "indigo", "Contact", "contact"),
    ("plates", "rust", "The Archive", "archive"),
]

THUMBS = [
    ("star", "rust", "HORSE"),
    ("wave", "ochre", "MEMORY"),
    ("chevrons", "sage", "ELEGY"),
    ("diamonds", "ochre", "BINARY"),
    ("rings", "indigo", "CHOICE"),
    ("wave", "indigo", "BIG RED DICE"),
    ("chevrons", "rust", "INDIAN ACT"),
    ("star", "sage", "TRADITIONAL"),
    ("diamonds", "indigo", "HALFBRED"),
    ("rings", "rust", "VOICES"),
    ("wave", "sage", "SHOOT THE INDIAN"),
    ("chevrons", "ochre", "FOR MIKE"),
    # the songs: one tile each, so no two rows of the list wear the same one
    ("rings", "sage", "AHASIW"),
    ("star", "indigo", "AIM"),
    ("wave", "rust", "ALANISE OBOMSAWIN"),
    ("diamonds", "sage", "HUMMINGBIRD"),
    ("chevrons", "indigo", "MUWEKMA OHLONE"),
    ("rings", "ochre", "SARA ROCQUE"),
    ("star", "ochre", "SUZANNE KITE"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (art, pal, _section) in enumerate(POSTERS, 1):
        img = still(art, pal, STILL, seed=i * 37 + 5)
        path = OUT / f"poster-{i:02d}.jpg"
        img.save(path, "JPEG", quality=86, optimize=True, progressive=True)
        print(f"{path.name} {path.stat().st_size / 1024:.0f} KB")
    for art, pal, word, name in BANNERS:
        img = banner(art, pal, word)
        path = OUT / f"banner-{name}.jpg"
        img.save(path, "JPEG", quality=84, optimize=True, progressive=True)
        print(f"{path.name}  {path.stat().st_size / 1024:.0f} KB")
    for i, (art, pal, label) in enumerate(THUMBS, 1):
        img = thumb(art, pal, (160, 160), "", seed=i * 13 + 2)
        path = OUT / f"thumb-{i:02d}.jpg"
        img.save(path, "JPEG", quality=86, optimize=True)
    print(f"{len(THUMBS)} thumbnails written")
    total = sum(f.stat().st_size for f in OUT.glob("*.jpg")) / 1024 / 1024
    print(f"total art weight: {total:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

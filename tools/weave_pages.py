#!/usr/bin/env python3
"""Weave the reading-page fragments into frontend/index.html.

The site is one static HTML file with every page inside it, and the page
sources live in frontend/pages/ so they can be edited as whole documents
instead of as 2,000-line slices of a 14,000-line file.  This script replaces
the five rebuilt page blocks, inserts the three new ones, and keeps the menu
and the client-side routes in step.

    python tools/weave_pages.py            # write
    python tools/weave_pages.py --check    # report drift, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
PAGES = FRONTEND / "pages"
INDEX = FRONTEND / "index.html"

MARKERS = {
    "architecture": "<!-- ===================== ARCHITECTURE PAGE ===================== -->",
    "harmony": "<!-- ===================== HARMONY PAGE ===================== -->",
    "synthesis": "<!-- ===================== SYNTHESIS / NEURAL VOICE PAGE ===================== -->",
    "translation": "<!-- ===================== TRANSLATION PAGE ===================== -->",
    "chat": "<!-- ===================== CHAT / LLM PAGE ===================== -->",
    "demo": "<!-- ===================== DEMO PAGE ===================== -->",
    "about": "<!-- ===================== ABOUT ===================== -->",
}
NEW_PAGES = [
    ("singsong", "frontend/pages/singsong.html", "<!-- ===================== SINGSONG PAGE ===================== -->"),
    ("mysong", "frontend/pages/mysong.html", "<!-- ===================== MYSONG PAGE ===================== -->"),
    ("measurements", "frontend/pages/measurements.html", "<!-- ===================== MEASUREMENTS PAGE ===================== -->"),
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def demo(name: str) -> str:
    return read(PAGES / "_demos" / f"{name}.html")


def render(name: str) -> str:
    text = read(PAGES / f"{name}.html")
    for key in ("arch", "harmony", "cree0", "cree1"):
        text = text.replace(f"<!--DEMO:{key}-->", demo(key))
        text = text.replace('<div id="arch-demo-mount"></div>', demo("arch"))
    return text


def replace_block(html: str, start_marker: str, end_marker: str, body: str) -> str:
    start = html.index(start_marker)
    end = html.index(end_marker, start)
    return html[:start] + start_marker + "\n" + body + "\n\n" + html[end:]


def build(html: str) -> str:
    for name in ("architecture", "harmony", "synthesis", "translation", "chat"):
        this = MARKERS[name]
        following = {
            "architecture": MARKERS["harmony"],
            "harmony": MARKERS["synthesis"],
            "synthesis": MARKERS["translation"],
            "translation": MARKERS["chat"],
            "chat": MARKERS["demo"],
        }[name]
        html = replace_block(html, this, following, render(name))

    # The three new pages form one block immediately before the demo page.
    # Rebuild that block wholesale, so the build is idempotent and order is
    # fixed even when a previous run left only part of it behind.
    new_block = "\n\n".join(marker + "\n" + render(name)
                           for name, _path, marker in NEW_PAGES)
    existing = [html.index(marker) for _n, _p, marker in NEW_PAGES if marker in html]
    if existing:
        start = min(existing)
        end = html.index(MARKERS["demo"], start)
        html = html[:start] + new_block + "\n\n" + html[end:]
    else:
        html = html.replace(MARKERS["demo"], new_block + "\n\n" + MARKERS["demo"], 1)

    # The menu: everything except the first and last entries is a reading page.
    menu = (
        '    <div class="menu-label" style="--mi:3">Instruments</div>\n'
        '    <ul class="menu-list">\n'
        '      <li style="--mi:4"><button data-page="architecture" onclick="showPage(\'architecture\')"><span class="mi-num">02</span><span class="mi-name">Architecture</span><span class="mi-note">The system</span></button></li>\n'
        '      <li style="--mi:5"><button data-page="harmony" onclick="showPage(\'harmony\')"><span class="mi-num">03</span><span class="mi-name">Harmony</span><span class="mi-note">Ten functions</span></button></li>\n'
        '      <li style="--mi:6"><button data-page="synthesis" onclick="showPage(\'synthesis\')"><span class="mi-num">04</span><span class="mi-name">Vocable Synthesis</span><span class="mi-note">The voice</span></button></li>\n'
        '      <li style="--mi:7"><button data-page="translation" onclick="showPage(\'translation\')"><span class="mi-num">05</span><span class="mi-name">Cree Sentiment</span><span class="mi-note">The language</span></button></li>\n'
        '      <li style="--mi:8"><button data-page="chat" onclick="showPage(\'chat\')"><span class="mi-num">06</span><span class="mi-name">Archer Bot</span><span class="mi-note">The notes and the language</span></button></li>\n'
        '      <li style="--mi:9"><button data-page="singsong" onclick="showPage(\'singsong\')"><span class="mi-num">07</span><span class="mi-name">SingSong</span><span class="mi-note">Generated band</span></button></li>\n'
        '      <li style="--mi:10"><button data-page="mysong" onclick="showPage(\'mysong\')"><span class="mi-num">08</span><span class="mi-name">MySong</span><span class="mi-note">Chord band</span></button></li>\n'
        '      <li style="--mi:11"><button data-page="measurements" onclick="showPage(\'measurements\')"><span class="mi-num">09</span><span class="mi-name">Measurements</span><span class="mi-note">The evidence</span></button></li>\n'
        '      <li style="--mi:12"><button data-page="demo" onclick="showPage(\'demo\')"><span class="mi-num">10</span><span class="mi-name">Sing With Archer</span><span class="mi-note">Live demo</span></button></li>\n'
        '    </ul>'
    )
    html = re.sub(
        r'    <div class="menu-label" style="--mi:3">Instruments</div>\n    <ul class="menu-list">.*?</ul>',
        menu, html, count=1, flags=re.S,
    )
    html = html.replace('<div class="menu-foot" style="--mi:12">',
                        '<div class="menu-foot" style="--mi:13">', 1)

    # Client routes.
    html = html.replace(
        "const PAGE_TO_ROUTE = { home: '/', about: '/about', architecture: '/architecture', harmony: '/harmony', synthesis: '/synthesis', translation: '/sentiment', chat: '/chat', demo: '/live-demo' };",
        "const PAGE_TO_ROUTE = { home: '/', about: '/about', architecture: '/architecture', harmony: '/harmony', synthesis: '/synthesis', translation: '/sentiment', chat: '/chat', singsong: '/singsong-guide', mysong: '/mysong-guide', measurements: '/measurements', demo: '/live-demo' };",
    )
    html = html.replace(
        "const ROUTE_TO_PAGE = { '': 'home', 'about': 'about', 'architecture': 'architecture', 'harmony': 'harmony', 'synthesis': 'synthesis', 'sentiment': 'translation', 'chat': 'chat', 'live-demo': 'demo' };",
        "const ROUTE_TO_PAGE = { '': 'home', 'about': 'about', 'architecture': 'architecture', 'harmony': 'harmony', 'synthesis': 'synthesis', 'sentiment': 'translation', 'chat': 'chat', 'singsong-guide': 'singsong', 'mysong-guide': 'mysong', 'measurements': 'measurements', 'live-demo': 'demo' };",
    )

    # The reading layer's stylesheet and its small instruments.
    old_css = '<link rel="stylesheet" href="/static/fonts/fonts.css?v=11">    <link rel="stylesheet" href="/static/assets/site.css?v=91">'
    # The build must be idempotent; drop any copies an earlier run left and
    # add exactly one, directly after the site sheet.
    html = re.sub(r'\n<link rel="stylesheet" href="/static/assets/paper\.css\?v=\d+">', '', html)
    if old_css in html:
        html = html.replace(old_css, old_css + '\n<link rel="stylesheet" href="/static/assets/paper.css?v=12">', 1)
    else:
        raise SystemExit("could not find the stylesheet line to add paper.css after")

    js_tag = '<script src="/static/assets/site.js?v=60"></script>'
    if js_tag in html and 'paper.js' not in html:
        html = html.replace(js_tag, '<script src="/static/assets/paper.js?v=1"></script>\n' + js_tag, 1)
    elif 'paper.js' not in html:
        raise SystemExit("could not find the site.js tag to add paper.js before")

    return html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    before = read(INDEX)
    after = build(before)
    if args.check:
        if before != after:
            print("frontend/index.html is out of date; run python tools/weave_pages.py")
            return 1
        print("frontend/index.html matches the page sources")
        return 0
    INDEX.write_text(after, encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)} ({len(after.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

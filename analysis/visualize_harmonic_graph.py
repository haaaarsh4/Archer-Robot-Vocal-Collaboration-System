from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

BG_COLOR     = "#07070d"
TEXT_COLOR   = "#f2f0ff"
MUTED_COLOR  = "#8a86a8"
RING_COLOR   = "#1c1c2e"
ACCENT_PURPLE = "#7c6af7"
ACCENT_TEAL   = "#34d399"
ACCENT_RED    = "#f87171"

RELATION_COLORS = {
    "transitions_to": ACCENT_PURPLE,
    "phrase_final_candidate": ACCENT_TEAL,
}
RELATION_LABELS = {
    "transitions_to": "moves to",
    "phrase_final_candidate": "ends a phrase on",
}


def load_graph(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def register_color(hz: float, min_hz: float, max_hz: float) -> tuple:
    span = math.log2(max_hz) - math.log2(min_hz) if max_hz > min_hz else 1.0
    t = (math.log2(hz) - math.log2(min_hz)) / span
    t = min(max(t, 0.0), 1.0)
    stops = [(0.0, (217, 119, 6)), (0.5, (124, 106, 247)), (1.0, (52, 211, 153))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple((c0[i] + f * (c1[i] - c0[i])) / 255 for i in range(3))
    return tuple(v / 255 for v in stops[-1][1])


def build_positions(tones: list, drone_id: str | None) -> dict:
    others = sorted((t for t in tones if t["id"] != drone_id), key=lambda t: t["reference_hz"])
    n = len(others)
    positions = {}
    if drone_id:
        positions[drone_id] = (0.0, 0.0)
    for i, t in enumerate(others):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        positions[t["id"]] = (math.cos(angle), math.sin(angle))
    return positions


def bezier_arc(p0: tuple, p1: tuple, curvature: float = 0.25) -> MplPath:
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    cx, cy = mx * (1 - curvature), my * (1 - curvature)
    return MplPath([p0, (cx, cy), p1], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def draw_background_rings(ax):
    for r in (0.35, 0.65, 1.0):
        ax.add_patch(mpatches.Circle((0, 0), r, fill=False, edgecolor=RING_COLOR, linewidth=1.0, zorder=1))


def hz_to_note_name(hz: float) -> str:
    return librosa.hz_to_note(hz)


def main():
    parser = argparse.ArgumentParser(description="Render a CreeHarmonicGraph YAML file as a circular diagram.")
    parser.add_argument("graph_path")
    parser.add_argument("--output", default=None, help="Output image path (default: <graph_name>_tree.png)")
    parser.add_argument("--min-edge-weight", type=float, default=0.12,
                         help="Hide relations weaker than this (0 to 1) to reduce clutter")
    parser.add_argument("--max-edges", type=int, default=140,
                         help="Draw at most this many of the strongest relations, keeps dense graphs legible")
    parser.add_argument("--title", default="Archer's Harmonic Graph")
    args = parser.parse_args()

    data = load_graph(args.graph_path)
    tones = data.get("tones", [])
    relations = data.get("relations", [])
    if not tones:
        print("No tones found in this graph file, nothing to draw.")
        return

    drone = next((t for t in tones if t.get("role") == "drone_root"), None)
    drone_id = drone["id"] if drone else None
    hz_values = [t["reference_hz"] for t in tones]
    min_hz, max_hz = min(hz_values), max(hz_values)
    positions = build_positions(tones, drone_id)

    degree: dict[str, float] = {t["id"]: 0.0 for t in tones}
    for r in relations:
        w = r.get("weight", 1.0)
        degree[r["from_tone"]] = degree.get(r["from_tone"], 0.0) + w
        degree[r["to_tone"]] = degree.get(r["to_tone"], 0.0) + w
    max_degree = max(degree.values()) if degree else 1.0

    fig, ax = plt.subplots(figsize=(18, 18), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_background_rings(ax)

    drawable = [r for r in relations
                if r.get("weight", 1.0) >= args.min_edge_weight
                and r["from_tone"] in positions and r["to_tone"] in positions
                and r["from_tone"] != r["to_tone"]]
    drawable.sort(key=lambda r: -r.get("weight", 1.0))
    drawable = drawable[:args.max_edges]

    for r in drawable:
        w = r.get("weight", 1.0)
        a, b = r["from_tone"], r["to_tone"]
        color = RELATION_COLORS.get(r.get("relation_type"), MUTED_COLOR)
        path = bezier_arc(positions[a], positions[b])
        alpha = 0.08 + 0.35 * w
        lw_base = 0.5 + 1.6 * w
        for glow_lw, glow_alpha in ((lw_base * 2.6, alpha * 0.12), (lw_base * 1.5, alpha * 0.25), (lw_base, alpha)):
            ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=color,
                                             lw=glow_lw, alpha=glow_alpha, capstyle="round", zorder=2))

    for t in tones:
        tid = t["id"]
        if tid not in positions:
            continue
        x, y = positions[tid]
        hz = t["reference_hz"]
        is_drone = tid == drone_id
        is_sensitive = bool(t.get("protocol_sensitive", False))
        color = register_color(hz, min_hz, max_hz)
        size = (0.045 + 0.06 * (degree.get(tid, 0.0) / max_degree)) if max_degree else 0.05
        if is_drone:
            size = max(size, 0.12)

        for glow_r, glow_alpha in ((size * 2.3, 0.10), (size * 1.5, 0.18)):
            ax.add_patch(mpatches.Circle((x, y), glow_r, color=color, alpha=glow_alpha, zorder=3, linewidth=0))

        edge_color = ACCENT_RED if is_sensitive else "#ffffff"
        ax.add_patch(mpatches.Circle((x, y), size, facecolor=color, edgecolor=edge_color,
                                      linewidth=2.4 if is_drone else 1.4, zorder=4))

        if x == 0 and y == 0:
            lx, ly, ha = 0.0, -size - 0.10, "center"
        else:
            norm = math.hypot(x, y)
            ux, uy = x / norm, y / norm
            lx, ly = x + ux * (size + 0.055), y + uy * (size + 0.055)
            ha = "left" if ux >= -0.05 else "right"

        label = f"{hz:.0f} Hz"
        try:
            note_name = hz_to_note_name(hz)
        except Exception:
            note_name = ""
        fontsize = 15 if is_drone else 10
        fontweight = "bold" if is_drone else "normal"
        ax.text(lx, ly, label, color=TEXT_COLOR, fontsize=fontsize, fontweight=fontweight,
                ha=ha, va="center", zorder=5, family="serif")
        sub = f"{note_name}  \u00b7  drone" if is_drone else note_name
        if sub:
            ax.text(lx, ly + 0.036, sub, color=MUTED_COLOR, fontsize=9.5 if is_drone else 8,
                    ha=ha, va="center", zorder=5, family="serif", fontstyle="italic")

    legend_handles = [
        Line2D([0], [0], color=ACCENT_PURPLE, lw=2.5, label="moves toward"),
        Line2D([0], [0], color=ACCENT_TEAL, lw=2.5, label="ends a phrase on"),
        Line2D([0], [0], marker="o", color=BG_COLOR, markerfacecolor=ACCENT_PURPLE,
               markeredgecolor="#ffffff", markersize=14, label="drone", lw=0),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
                        ncol=3, frameon=False, fontsize=13, labelcolor=MUTED_COLOR,
                        handletextpad=0.6, columnspacing=1.8, prop={"family": "serif", "size": 13})

    fig.text(0.5, 0.02, "built from Archer's own singing, not from music theory",
              color=RING_COLOR, fontsize=11, ha="center", family="serif", fontstyle="italic")

    out_path = args.output or (Path(args.graph_path).stem + "_tree.png")
    fig.savefig(out_path, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.4)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

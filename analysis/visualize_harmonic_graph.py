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
import matplotlib.patheffects as path_effects
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


def hz_to_note_name(hz: float) -> str:
    try:
        return librosa.hz_to_note(hz)
    except Exception:
        return ""


def bezier_arc(p0: tuple, p1: tuple, curvature: float = 0.25, pivot: tuple = (0.0, 0.0)) -> MplPath:
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    cx = pivot[0] + (mx - pivot[0]) * (1 - curvature)
    cy = pivot[1] + (my - pivot[1]) * (1 - curvature)
    return MplPath([p0, (cx, cy), p1], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def draw_glow_edge(ax, p0, p1, color, weight, pivot=(0.0, 0.0), curvature=0.25):
    path = bezier_arc(p0, p1, curvature=curvature, pivot=pivot)
    alpha = 0.08 + 0.35 * weight
    lw_base = 0.5 + 1.6 * weight
    for glow_lw, glow_alpha in ((lw_base * 2.6, alpha * 0.12), (lw_base * 1.5, alpha * 0.25), (lw_base, alpha)):
        ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=color,
                                         lw=glow_lw, alpha=glow_alpha, capstyle="round", zorder=2))


def draw_node(ax, pos, color, size, is_drone, is_sensitive, zorder=4):
    x, y = pos
    for glow_r, glow_alpha in ((size * 2.3, 0.10), (size * 1.5, 0.18)):
        ax.add_patch(mpatches.Circle((x, y), glow_r, color=color, alpha=glow_alpha, zorder=zorder - 1, linewidth=0))
    edge_color = ACCENT_RED if is_sensitive else "#ffffff"
    ax.add_patch(mpatches.Circle((x, y), size, facecolor=color, edgecolor=edge_color,
                                  linewidth=2.4 if is_drone else 1.4, zorder=zorder))


def draw_node_label(ax, pos, pivot, size, hz, is_drone, extra_sub=None):
    x, y = pos
    if abs(x - pivot[0]) < 1e-6 and abs(y - pivot[1]) < 1e-6:
        lx, ly, ha = x, y - size - 0.09, "center"
    else:
        dx, dy = x - pivot[0], y - pivot[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        lx, ly = x + ux * (size + 0.05), y + uy * (size + 0.05)
        ha = "left" if ux >= -0.05 else "right"

    label = f"{hz:.0f} Hz"
    note_name = hz_to_note_name(hz)
    fontsize = 16 if is_drone else 11.5
    fontweight = "bold" if is_drone else "normal"
    halo = [path_effects.withStroke(linewidth=3.2, foreground=BG_COLOR)]
    ax.text(lx, ly, label, color=TEXT_COLOR, fontsize=fontsize, fontweight=fontweight,
            ha=ha, va="center", zorder=7, family="serif", path_effects=halo)
    sub_parts = [p for p in (note_name, "drone" if is_drone else None, extra_sub) if p]
    if sub_parts:
        ax.text(lx, ly + 0.038, "  \u00b7  ".join(sub_parts), color="#c9c6e0",
                fontsize=10 if is_drone else 9, ha=ha, va="center", zorder=7,
                family="serif", fontstyle="italic", path_effects=halo)


def draw_legend_and_footer(fig, ax, extra_handles=None, footer=None):
    legend_handles = [
        Line2D([0], [0], color=ACCENT_PURPLE, lw=2.5, label=RELATION_LABELS["transitions_to"]),
        Line2D([0], [0], color=ACCENT_TEAL, lw=2.5, label=RELATION_LABELS["phrase_final_candidate"]),
        Line2D([0], [0], marker="o", color=BG_COLOR, markerfacecolor=ACCENT_PURPLE,
               markeredgecolor="#ffffff", markersize=14, label="drone", lw=0),
    ]
    if extra_handles:
        legend_handles += extra_handles
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(legend_handles), frameon=False, fontsize=12, labelcolor=MUTED_COLOR,
              handletextpad=0.6, columnspacing=1.6, prop={"family": "serif", "size": 12})
    if footer:
        fig.text(0.5, 0.02, footer, color=RING_COLOR, fontsize=11, ha="center",
                  family="serif", fontstyle="italic")


def draw_title(fig, title, subtitle):
    fig.text(0.5, 0.965, title, color=TEXT_COLOR, fontsize=30, fontweight="bold",
              ha="center", family="serif")
    if subtitle:
        fig.text(0.5, 0.935, subtitle, color=MUTED_COLOR, fontsize=14, ha="center", family="serif")

def draw_background_rings(ax):
    for r in (0.35, 0.65, 1.0):
        ax.add_patch(mpatches.Circle((0, 0), r, fill=False, edgecolor=RING_COLOR, linewidth=1.0, zorder=1))


def build_ring_positions(tones: list, drone_id: str | None) -> dict:
    others = sorted((t for t in tones if t["id"] != drone_id), key=lambda t: t["reference_hz"])
    n = len(others)
    positions = {}
    if drone_id:
        positions[drone_id] = (0.0, 0.0)
    for i, t in enumerate(others):
        angle = 2 * math.pi * i / max(n, 1) - math.pi / 2
        positions[t["id"]] = (math.cos(angle), math.sin(angle))
    return positions


def render_ring(data: dict, args) -> None:
    tones = data.get("tones", [])
    relations = data.get("relations", [])
    if not tones:
        print("No tones found in this graph file, nothing to draw.")
        return

    drone = next((t for t in tones if t.get("role") == "drone_root"), None)
    drone_id = drone["id"] if drone else None
    hz_values = [t["reference_hz"] for t in tones]
    min_hz, max_hz = min(hz_values), max(hz_values)
    positions = build_ring_positions(tones, drone_id)

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

    draw_title(fig, args.title, args.subtitle)
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
        draw_glow_edge(ax, positions[a], positions[b], color, w)

    for t in tones:
        tid = t["id"]
        if tid not in positions:
            continue
        pos = positions[tid]
        hz = t["reference_hz"]
        is_drone = tid == drone_id
        is_sensitive = bool(t.get("protocol_sensitive", False))
        color = register_color(hz, min_hz, max_hz)
        size = (0.045 + 0.06 * (degree.get(tid, 0.0) / max_degree)) if max_degree else 0.05
        if is_drone:
            size = max(size, 0.12)
        draw_node(ax, pos, color, size, is_drone, is_sensitive)
        draw_node_label(ax, pos, (0.0, 0.0), size, hz, is_drone)

    draw_legend_and_footer(fig, ax)
    out_path = args.output or (Path(args.graph_path).stem + "_ring.png")
    fig.savefig(out_path, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.4)
    print(f"Wrote {out_path}")

def build_relation_index(relations: list) -> dict:
    adj: dict[str, dict[str, dict]] = {}
    for r in relations:
        a, b = r["from_tone"], r["to_tone"]
        w = r.get("weight", 1.0)
        rtype = r.get("relation_type")
        for src, dst in ((a, b), (b, a)):
            bucket = adj.setdefault(src, {})
            if dst not in bucket or w > bucket[dst]["weight"]:
                bucket[dst] = {"other": dst, "weight": w, "type": rtype}
    return adj


def tone_strength(tone_id: str, adj: dict) -> float:
    return sum(e["weight"] for e in adj.get(tone_id, {}).values())


def neighbors_of(tone_id: str, adj: dict, min_weight: float, exclude: set) -> list:
    edges = [e for other, e in adj.get(tone_id, {}).items()
             if other not in exclude and e["weight"] >= min_weight]
    return sorted(edges, key=lambda e: -e["weight"])


def pick_center(tones: list, adj: dict) -> str:
    drone = next((t for t in tones if t.get("role") == "drone_root"), None)
    if drone:
        return drone["id"]
    return max(tones, key=lambda t: tone_strength(t["id"], adj))["id"]


def tone_metric(tone: dict, adj: dict) -> float:
    if tone.get("total_duration_s"):
        return float(tone["total_duration_s"])
    return tone_strength(tone["id"], adj) or 1.0


def build_flower_layout(tones_by_id: dict, adj: dict, min_edge_weight: float,
                         max_petals: int | None, max_satellites: int | None) -> dict:
    center_id = pick_center(list(tones_by_id.values()), adj)

    petal_edges = neighbors_of(center_id, adj, min_edge_weight, exclude={center_id})
    if max_petals is not None:
        petal_edges = petal_edges[:max_petals]
    n_petals = len(petal_edges)

    # More petals -> each one has to shrink to stay legible. sqrt rather
    # than linear so this degrades gracefully instead of collapsing fast.
    petal_scale = 1.0 / math.sqrt(max(n_petals, 1))
    petal_hub_radius = 1.35

    positions = {center_id: (0.0, 0.0)}
    petals = []

    for i, edge in enumerate(petal_edges):
        pid = edge["other"]
        angle = (2 * math.pi * i / n_petals - math.pi / 2) if n_petals else 0.0
        hub_pos = (petal_hub_radius * math.cos(angle), petal_hub_radius * math.sin(angle))
        positions[pid] = hub_pos

        sat_edges = neighbors_of(pid, adj, min_edge_weight, exclude={center_id, pid})
        if max_satellites is not None:
            sat_edges = sat_edges[:max_satellites]
        n_sat = len(sat_edges)

        local_r = 0.15 + 0.55 * petal_scale
        sat_positions = {}
        for j, sedge in enumerate(sat_edges):
            sid = sedge["other"]
            sangle = (2 * math.pi * j / n_sat - math.pi / 2) if n_sat else 0.0
            spos = (hub_pos[0] + local_r * math.cos(sangle), hub_pos[1] + local_r * math.sin(sangle))
            sat_positions[sid] = spos
            positions[sid] = spos

        petals.append({
            "hub_id": pid,
            "hub_pos": hub_pos,
            "hub_edge_from_center": edge,
            "satellite_edges": sat_edges,
            "satellite_positions": sat_positions,
            "local_radius": local_r,
        })

    return {"center_id": center_id, "positions": positions, "petals": petals, "n_petals": n_petals}


def petal_boundary_points(hub_pos: tuple, satellite_positions: dict, local_radius: float) -> list | None:
    n = len(satellite_positions)
    if n < 3:
        return None  # not enough points for a meaningful polygon
    hx, hy = hub_pos
    boundary_r = local_radius * 1.6
    pts = []
    for spos in satellite_positions.values():
        ang = math.atan2(spos[1] - hy, spos[0] - hx)
        pts.append((hx + boundary_r * math.cos(ang), hy + boundary_r * math.sin(ang)))
    # order points by angle so the polygon doesn't self-intersect
    pts.sort(key=lambda p: math.atan2(p[1] - hy, p[0] - hx))
    return pts


def render_tree(data: dict, args) -> None:
    tones = data.get("tones", [])
    relations = data.get("relations", [])
    if not tones:
        print("No tones found in this graph file, nothing to draw.")
        return
    if len(tones) < 2:
        print("Only one tone in this graph — nothing to relate it to yet, "
              "so there's no tree to draw. Feed in more recordings first.")
        return

    tones_by_id = {t["id"]: t for t in tones}
    hz_values = [t["reference_hz"] for t in tones]
    min_hz, max_hz = min(hz_values), max(hz_values)
    adj = build_relation_index(relations)

    layout = build_flower_layout(tones_by_id, adj, args.min_edge_weight,
                                  args.max_petals, args.max_satellites_per_petal)
    center_id = layout["center_id"]
    positions = layout["positions"]
    petals = layout["petals"]

    if not petals:
        print(f"'{center_id}' has no relations at or above --min-edge-weight "
              f"{args.min_edge_weight}, so there are no petals to draw. Try "
              f"lowering --min-edge-weight, or check that the graph actually "
              f"has relations in it.")
        return

    metrics = {t["id"]: tone_metric(t, adj) for t in tones}
    max_metric = max(metrics.values()) if metrics else 1.0

    # Figure scales with how many petals there are, same "size decided by
    # the data" principle as the petals themselves.
    side = max(16, min(30, 14 + 1.6 * math.sqrt(len(petals))))
    fig, ax = plt.subplots(figsize=(side, side), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    extent = 1.35 + max(p["local_radius"] for p in petals) * 1.8
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_title(fig, args.title, args.subtitle)
    draw_background_rings(ax)

    # --- petal boundaries (drawn first, underneath everything else) ---
    for petal in petals:
        pts = petal_boundary_points(petal["hub_pos"], petal["satellite_positions"], petal["local_radius"])
        if pts:
            poly = mpatches.Polygon(pts, closed=True, fill=False,
                                     edgecolor=RING_COLOR, linewidth=1.3, zorder=1)
            ax.add_patch(poly)
        elif petal["satellite_positions"]:
            ax.add_patch(mpatches.Circle(petal["hub_pos"], petal["local_radius"] * 1.6,
                                          fill=False, edgecolor=RING_COLOR, linewidth=1.0, zorder=1))

    # --- edges: center <-> each petal hub ---
    for petal in petals:
        e = petal["hub_edge_from_center"]
        color = RELATION_COLORS.get(e["type"], MUTED_COLOR)
        draw_glow_edge(ax, positions[center_id], petal["hub_pos"], color, e["weight"], pivot=(0.0, 0.0))

    # --- edges: each petal hub <-> its own satellites ---
    for petal in petals:
        for e in petal["satellite_edges"]:
            color = RELATION_COLORS.get(e["type"], MUTED_COLOR)
            draw_glow_edge(ax, petal["hub_pos"], positions[e["other"]], color, e["weight"], pivot=petal["hub_pos"])

    # --- nodes: satellites first, then petal hubs, then center on top ---
    def draw_tone_node(tid, pos, pivot, is_hub):
        tone = tones_by_id[tid]
        hz = tone["reference_hz"]
        is_drone = tid == center_id and tone.get("role") == "drone_root"
        is_sensitive = bool(tone.get("protocol_sensitive", False))
        color = register_color(hz, min_hz, max_hz)
        m = metrics.get(tid, 0.0)
        base = 0.06 if tid == center_id else (0.045 if is_hub else 0.032)
        span = 0.09 if tid == center_id else (0.055 if is_hub else 0.03)
        size = base + span * (m / max_metric if max_metric else 0.0)
        occ = tone.get("occurrence_count")
        extra = f"seen {occ}x" if occ else None
        draw_node(ax, pos, color, size, is_drone, is_sensitive, zorder=5 if is_hub else 4)
        draw_node_label(ax, pos, pivot, size, hz, is_drone, extra_sub=extra)

    for petal in petals:
        for sid, spos in petal["satellite_positions"].items():
            draw_tone_node(sid, spos, petal["hub_pos"], is_hub=False)
    for petal in petals:
        draw_tone_node(petal["hub_id"], petal["hub_pos"], (0.0, 0.0), is_hub=True)
    draw_tone_node(center_id, positions[center_id], (0.0, 0.0), is_hub=True)

    footer = f"{len(tones)} tones  \u00b7  {len(petals)} directly related to the center"
    draw_legend_and_footer(fig, ax, footer=footer)

    out_path = args.output or (Path(args.graph_path).stem + "_tree.png")
    fig.savefig(out_path, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.4)
    print(f"Wrote {out_path}")
    print(f"  center: {center_id} ({tones_by_id[center_id]['reference_hz']:.0f} Hz)  "
          f"\u00b7  {len(petals)} petals  \u00b7  "
          f"{sum(len(p['satellite_positions']) for p in petals)} satellite tones total")


def main():
    parser = argparse.ArgumentParser(description="Render a CreeHarmonicGraph YAML file as a diagram.")
    parser.add_argument("graph_path")
    parser.add_argument("--style", choices=["ring", "tree"], default="ring",
                         help="'ring': every tone on one circle (original behaviour). "
                              "'tree': nested flower layout, one center + petals of related "
                              "tones, each with its own sub-petal of relations.")
    parser.add_argument("--output", default=None, help="Output image path (default: <graph_name>_<style>.png)")
    parser.add_argument("--min-edge-weight", type=float, default=0.12,
                         help="Hide relations weaker than this (0 to 1) to reduce clutter")
    parser.add_argument("--max-edges", type=int, default=140,
                         help="[ring style] Draw at most this many of the strongest relations")
    parser.add_argument("--max-petals", type=int, default=None,
                         help="[tree style] Cap how many tones connect directly to the center. "
                              "Default: no cap — draw as many as the data supports.")
    parser.add_argument("--max-satellites-per-petal", type=int, default=None,
                         help="[tree style] Cap how many relatives each petal shows. "
                              "Default: no cap — draw as many as the data supports.")
    parser.add_argument("--title", default=None, help="Default: 'Archer's Harmonic Tree' or 'Archer's Harmonic Graph' depending on --style")
    parser.add_argument("--subtitle", default=None,
                         help="Text under the title. Default: none.")
    args = parser.parse_args()

    if args.title is None:
        args.title = "Archer's Harmonic Tree" if args.style == "tree" else "Archer's Harmonic Graph"

    data = load_graph(args.graph_path)

    if args.style == "tree":
        render_tree(data, args)
    else:
        render_ring(data, args)


if __name__ == "__main__":
    main()
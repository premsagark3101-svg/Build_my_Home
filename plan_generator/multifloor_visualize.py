"""
multifloor_visualize.py
=======================
Rich visualization for multi-floor building layouts.

Outputs:
  1. Per-floor 2D plan (colour-coded rooms)
  2. Vertical section view (shows structural alignment)
  3. Structural core overlay (staircase, elevator, plumbing stacks)
  4. Training curves comparison (L1 vs L2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np

GRID = 20
CELL = 28   # pixels per cell in floor plan

ROOM_COLORS = {
    "master_bedroom": "#4169E1",
    "bedroom":        "#6495ED",
    "bathroom":       "#20B2AA",
    "kitchen":        "#FF8C00",
    "living_room":    "#2E8B57",
    "dining_room":    "#6B8E23",
    "study":          "#9932CC",
    "parking":        "#708090",
    "staircase":      "#DC143C",
    "elevator":       "#B8860B",
    "gym":            "#CD5C5C",
    "home_office":    "#DEB887",
    "terrace":        "#BDB76B",
    "storage":        "#A0522D",
    "laundry":        "#5F9EA0",
}

STRUCTURAL_HATCH = {
    "staircase": "///",
    "elevator":  "xxx",
}

FLOOR_BG = ["#0d1b2a", "#1b2838", "#162032", "#0f1c2e"]


def _color(room: str) -> str:
    return ROOM_COLORS.get(room, "#888888")


# ─────────────────────────────────────────────────────────────────────────────
# Floor plan panel
# ─────────────────────────────────────────────────────────────────────────────

def _draw_floor(
    ax:         plt.Axes,
    layout:     list[dict],
    floor_num:  int,
    core_dict:  Optional[dict] = None,
    show_grid:  bool = True,
):
    bg = FLOOR_BG[min(floor_num - 1, len(FLOOR_BG) - 1)]
    ax.set_facecolor(bg)

    if show_grid:
        for i in range(GRID + 1):
            ax.axhline(i, color="#ffffff10", linewidth=0.4)
            ax.axvline(i, color="#ffffff10", linewidth=0.4)

    # Plot boundary
    ax.add_patch(patches.Rectangle(
        (0, 0), GRID, GRID,
        linewidth=2.5, edgecolor="#FF6B6B", facecolor="none", zorder=15
    ))

    # Draw rooms
    for r in layout:
        name = r["room"]
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        color = _color(name)
        is_structural = r.get("structural", False)
        hatch = STRUCTURAL_HATCH.get(name, None) if is_structural else None

        rect = patches.FancyBboxPatch(
            (x + 0.07, y + 0.07), w - 0.14, h - 0.14,
            boxstyle="round,pad=0.05",
            facecolor=to_rgba(color, 0.80 if not is_structural else 0.65),
            edgecolor="white" if not is_structural else "#FFD700",
            linewidth=2.0 if is_structural else 1.2,
            zorder=5,
            hatch=hatch,
        )
        ax.add_patch(rect)

        # Label
        label = name.replace("_", "\n")
        fs = max(5.5, min(8.5, (w * h) ** 0.4 * 3))
        ax.text(
            x + w/2, y + h/2, label,
            ha="center", va="center",
            fontsize=fs, fontweight="bold",
            color="white", zorder=6,
        )

        # Size tag
        if w * h >= 6:
            ax.text(
                x + w - 0.15, y + 0.2,
                f"{w}×{h}",
                ha="right", va="bottom",
                fontsize=5, color="white", alpha=0.6, zorder=6,
            )

    # Overlay structural core annotations
    if core_dict:
        _draw_core_overlay(ax, core_dict)

    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_aspect("equal")
    ax.set_title(
        f"Floor {floor_num}",
        color="white", fontsize=11, fontweight="bold", pad=6
    )
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#FF6B6B")
        sp.set_linewidth(1.5)


def _draw_core_overlay(ax: plt.Axes, core: dict):
    """Draw structural core markers on a floor plan."""
    # Plumbing stack markers
    for (sx, sy) in core.get("plumbing_stacks", []):
        ax.plot(sx + 1, sy + 1, "o",
                color="#00FFFF", markersize=6, zorder=20, alpha=0.8)
        ax.annotate("P", (sx + 1, sy + 1), color="#00FFFF",
                    fontsize=5, ha="center", va="center", zorder=21)

    # Kitchen stack
    ks = core.get("kitchen_stack")
    if ks:
        ax.plot(ks[0] + 1.5, ks[1] + 1.5, "D",
                color="#FFA500", markersize=5, zorder=20, alpha=0.8)

    # Column grid (tiny dots)
    for (cx, cy) in core.get("columns", []):
        ax.plot(cx, cy, ".",
                color="#ffffff40", markersize=3, zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# Vertical section view
# ─────────────────────────────────────────────────────────────────────────────

def _draw_section(
    ax:       plt.Axes,
    building: dict,
    n_floors: int,
    core:     Optional[dict] = None,
):
    """
    Schematic vertical cross-section showing room stacking.
    X-axis = grid X position, Y-axis = floor number.
    """
    ax.set_facecolor("#0d1b2a")
    floor_h = 1.0   # height of each floor band

    for floor_num in range(1, n_floors + 1):
        key    = f"floor_{floor_num}"
        layout = building.get(key, [])
        y_base = (floor_num - 1) * floor_h

        # Floor band background
        ax.add_patch(patches.Rectangle(
            (0, y_base), GRID, floor_h,
            facecolor="#ffffff08", edgecolor="#ffffff20",
            linewidth=0.5, zorder=1,
        ))
        ax.text(-0.5, y_base + floor_h/2, f"F{floor_num}",
                color="white", fontsize=8, va="center", ha="right")

        # Draw room bars
        for r in layout:
            name = r["room"]
            x, w = r["x"], r["w"]
            color = _color(name)
            ax.add_patch(patches.Rectangle(
                (x + 0.05, y_base + 0.05), w - 0.1, floor_h - 0.1,
                facecolor=to_rgba(color, 0.7),
                edgecolor="white", linewidth=0.8, zorder=5,
            ))

    # Plumbing stack lines
    if core:
        for (sx, _) in core.get("plumbing_stacks", []):
            ax.axvline(sx + 1, color="#00FFFF", linewidth=1.5,
                       linestyle="--", alpha=0.6, zorder=10)
        ks = core.get("kitchen_stack")
        if ks:
            ax.axvline(ks[0] + 1.5, color="#FFA500", linewidth=1.5,
                       linestyle=":", alpha=0.6, zorder=10)

        # Staircase column
        sc = core.get("staircase")
        if sc:
            ax.add_patch(patches.Rectangle(
                (sc[0], 0), 3, n_floors * floor_h,
                facecolor=to_rgba("#DC143C", 0.15),
                edgecolor="#DC143C", linewidth=1.5,
                linestyle="--", zorder=8,
            ))

        # Elevator column
        el = core.get("elevator")
        if el:
            ax.add_patch(patches.Rectangle(
                (el[0], 0), 2, n_floors * floor_h,
                facecolor=to_rgba("#B8860B", 0.15),
                edgecolor="#B8860B", linewidth=1.5,
                linestyle="--", zorder=8,
            ))

    ax.set_xlim(-1, GRID)
    ax.set_ylim(0, n_floors * floor_h)
    ax.set_xlabel("X position (m)", color="white", fontsize=9)
    ax.set_title("Vertical Section View", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")


# ─────────────────────────────────────────────────────────────────────────────
# Main visualize_building
# ─────────────────────────────────────────────────────────────────────────────

def visualize_building(
    building:  dict,
    title:     str  = "Multi-Floor Building — PPO Hierarchical Planner",
    save_path: Optional[str] = None,
    show:      bool = False,
) -> plt.Figure:

    floor_keys = sorted([k for k in building if k.startswith("floor_")])
    n_floors   = len(floor_keys)
    core       = building.get("structural_core")

    # Layout: floor plans in a row, section view + legend below
    fig = plt.figure(figsize=(5 * n_floors + 2, 16))
    fig.patch.set_facecolor("#0a0f1e")

    gs = gridspec.GridSpec(
        3, max(n_floors, 1),
        figure   = fig,
        height_ratios = [8, 4, 1.2],
        hspace   = 0.45,
        wspace   = 0.18,
    )

    # ── Row 1: floor plans ────────────────────────────────────────────
    for i, key in enumerate(floor_keys):
        ax = fig.add_subplot(gs[0, i])
        _draw_floor(
            ax        = ax,
            layout    = building[key],
            floor_num = i + 1,
            core_dict = core,
        )

    # ── Row 2: vertical section ───────────────────────────────────────
    ax_sec = fig.add_subplot(gs[1, :])
    _draw_section(ax_sec, building, n_floors, core)

    # ── Row 3: legend ─────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_facecolor("#0a0f1e")
    ax_leg.axis("off")

    rooms_shown = set()
    for key in floor_keys:
        for r in building[key]:
            rooms_shown.add(r["room"])

    legend_handles = [
        patches.Patch(color=_color(r), label=r.replace("_"," ").title())
        for r in sorted(rooms_shown)
    ]
    # Structural legend
    legend_handles += [
        Line2D([0],[0], color="#00FFFF", linewidth=2, linestyle="--",
               label="Plumbing Stack"),
        Line2D([0],[0], color="#FFA500", linewidth=2, linestyle=":",
               label="Kitchen Stack"),
        patches.Patch(facecolor=to_rgba("#DC143C",0.3), edgecolor="#DC143C",
                      linestyle="--", label="Staircase Zone"),
        patches.Patch(facecolor=to_rgba("#B8860B",0.3), edgecolor="#B8860B",
                      linestyle="--", label="Elevator Zone"),
    ]
    ax_leg.legend(
        handles    = legend_handles,
        loc        = "center",
        ncol       = min(len(legend_handles), 7),
        framealpha = 0.2,
        facecolor  = "#1a1a2e",
        edgecolor  = "#FF6B6B",
        labelcolor = "white",
        fontsize   = 8,
    )

    fig.suptitle(title, color="white", fontsize=14,
                 fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Building visualization saved → {save_path}")

    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curves (L1 vs L2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:   list[dict],
    save_path: Optional[str] = None,
    show:      bool = False,
) -> plt.Figure:

    l1 = [h for h in history if h.get("phase") == "L1"]
    l2 = [h for h in history if h.get("phase") == "L2"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0a0f1e")

    def styled(ax, title):
        ax.set_facecolor("#0d1b2a")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.grid(alpha=0.15, color="#444")

    # L1 rewards
    ax = axes[0]
    styled(ax, "L1 Agent — Ground Floor Rewards")
    if l1:
        eps = [h["episode"]    for h in l1]
        rws = [h["avg_reward"] for h in l1]
        ax.plot(eps, rws, color="#FF6B6B", linewidth=2, label="Avg Reward")
        ax.fill_between(eps, rws, alpha=0.15, color="#FF6B6B")
        if len(rws) > 3:
            sm = np.convolve(rws, np.ones(3)/3, mode="valid")
            ax.plot(eps[2:], sm, color="#FFD700", linewidth=1.5,
                    linestyle="--", label="Smoothed")
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg Reward")
    ax.legend(labelcolor="white", facecolor="#1a1a2e", edgecolor="#555")

    # L2 rewards
    ax = axes[1]
    styled(ax, "L2 Agent — Upper Floor Rewards")
    if l2:
        eps = [h["episode"]    for h in l2]
        rws = [h["avg_reward"] for h in l2]
        ax.plot(eps, rws, color="#4FC3F7", linewidth=2, label="Avg Reward")
        ax.fill_between(eps, rws, alpha=0.15, color="#4FC3F7")
        if len(rws) > 3:
            sm = np.convolve(rws, np.ones(3)/3, mode="valid")
            ax.plot(eps[2:], sm, color="#81C784", linewidth=1.5,
                    linestyle="--", label="Smoothed")
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg Reward")
    ax.legend(labelcolor="white", facecolor="#1a1a2e", edgecolor="#555")

    fig.suptitle("Hierarchical PPO Training Curves",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Training curves saved → {save_path}")
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with a hand-crafted building
    demo_building = {
        "floor_1": [
            {"room": "living_room", "x": 0,  "y": 0,  "w": 6, "h": 5, "floor": 1},
            {"room": "kitchen",     "x": 6,  "y": 0,  "w": 4, "h": 4, "floor": 1},
            {"room": "dining_room", "x": 6,  "y": 4,  "w": 4, "h": 4, "floor": 1},
            {"room": "bathroom",    "x": 10, "y": 0,  "w": 2, "h": 3, "floor": 1},
            {"room": "parking",     "x": 12, "y": 0,  "w": 3, "h": 5, "floor": 1},
            {"room": "staircase",   "x": 0,  "y": 5,  "w": 3, "h": 3, "floor": 1, "structural": True},
            {"room": "elevator",    "x": 3,  "y": 5,  "w": 2, "h": 2, "floor": 1, "structural": True},
        ],
        "floor_2": [
            {"room": "master_bedroom","x": 0,  "y": 0, "w": 5, "h": 4, "floor": 2},
            {"room": "bedroom",       "x": 5,  "y": 0, "w": 4, "h": 4, "floor": 2},
            {"room": "bathroom",      "x": 9,  "y": 0, "w": 2, "h": 2, "floor": 2},
            {"room": "study",         "x": 11, "y": 0, "w": 3, "h": 3, "floor": 2},
            {"room": "bedroom",       "x": 0,  "y": 4, "w": 4, "h": 4, "floor": 2},
            {"room": "staircase",     "x": 0,  "y": 8, "w": 3, "h": 3, "floor": 2, "structural": True},
            {"room": "elevator",      "x": 3,  "y": 8, "w": 2, "h": 2, "floor": 2, "structural": True},
        ],
        "floor_3": [
            {"room": "bedroom",       "x": 0,  "y": 0, "w": 4, "h": 4, "floor": 3},
            {"room": "study",         "x": 4,  "y": 0, "w": 3, "h": 3, "floor": 3},
            {"room": "bathroom",      "x": 7,  "y": 0, "w": 2, "h": 2, "floor": 3},
            {"room": "master_bedroom","x": 9,  "y": 0, "w": 5, "h": 4, "floor": 3},
            {"room": "staircase",     "x": 0,  "y": 4, "w": 3, "h": 3, "floor": 3, "structural": True},
            {"room": "elevator",      "x": 3,  "y": 4, "w": 2, "h": 2, "floor": 3, "structural": True},
        ],
        "structural_core": {
            "staircase":       [0, 8],
            "staircase_size":  [3, 3],
            "elevator":        [3, 8],
            "elevator_size":   [2, 2],
            "plumbing_stacks": [[10, 0]],
            "wet_zones":       [[10, 0]],
            "kitchen_stack":   [6, 0],
            "columns":         [[0,0],[4,0],[8,0],[12,0],[0,4],[4,4],[8,4],[12,4]],
        }
    }

    out = Path("outputs"); out.mkdir(exist_ok=True)
    visualize_building(
        demo_building,
        title     = "3-Floor Building — Hierarchical PPO (Demo)",
        save_path = str(out / "building_demo.png"),
        show      = False,
    )
    print("Demo visualization saved to outputs/building_demo.png")

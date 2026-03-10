"""
visualize.py
============
Visualization utilities for floor plan layouts and PPO training metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

ROOM_COLORS = {
    "master_bedroom": "#4682B4",
    "bedroom":        "#6495ED",
    "bathroom":       "#20B2AA",
    "kitchen":        "#FFA500",
    "living_room":    "#3CB371",
    "dining_room":    "#9ACD32",
    "study":          "#BA55D3",
    "parking":        "#A9A9A9",
    "gym":            "#CD5C5C",
    "home_office":    "#DEB887",
    "terrace":        "#F0E68C",
    "storage":        "#D2B48C",
    "laundry":        "#B0C4DE",
    "utility_room":   "#C0C0C0",
}

GRID = 20


def _room_color(room_name: str) -> str:
    return ROOM_COLORS.get(room_name, "#DDDDDD")


# ─────────────────────────────────────────────────────────────────────────────
# Floor plan visualizer
# ─────────────────────────────────────────────────────────────────────────────

def visualize_layout(
    layout:    list[dict],
    title:     str  = "Generated Floor Plan",
    grid_size: int  = GRID,
    save_path: Optional[str] = None,
    show:      bool = True,
) -> plt.Figure:
    """
    Render a floor plan layout as a colour-coded grid.

    Parameters
    ----------
    layout    : list of room dicts  {"room","x","y","w","h"}
    title     : figure title
    grid_size : grid dimensions
    save_path : if set, save PNG here
    show      : if True, call plt.show()
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Grid lines
    for i in range(grid_size + 1):
        ax.axhline(i, color="#0f3460", linewidth=0.5, alpha=0.6)
        ax.axvline(i, color="#0f3460", linewidth=0.5, alpha=0.6)

    # Plot boundary
    boundary = patches.Rectangle(
        (0, 0), grid_size, grid_size,
        linewidth=3, edgecolor="#e94560", facecolor="none", zorder=10
    )
    ax.add_patch(boundary)

    # Draw rooms
    for room_dict in layout:
        name = room_dict["room"]
        x, y = room_dict["x"], room_dict["y"]
        w, h = room_dict["w"], room_dict["h"]
        color = _room_color(name)

        rect = patches.FancyBboxPatch(
            (x + 0.05, y + 0.05), w - 0.1, h - 0.1,
            boxstyle="round,pad=0.05",
            facecolor=to_rgba(color, 0.85),
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        ax.add_patch(rect)

        # Room label
        label = name.replace("_", "\n")
        fontsize = max(6, min(9, int(w * h * 0.5)))
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white", zorder=6,
            wrap=True,
        )

        # Dimensions label
        ax.text(
            x + w - 0.15, y + 0.25,
            f"{w}×{h}",
            ha="right", va="bottom",
            fontsize=6, color="white", alpha=0.7, zorder=6,
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xlabel("X (metres)", color="white", fontsize=11)
    ax.set_ylabel("Y (metres)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e94560")
        spine.set_linewidth(2)

    # Legend
    legend_patches = [
        patches.Patch(color=_room_color(r["room"]), label=r["room"].replace("_", " ").title())
        for r in layout
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        framealpha=0.3,
        facecolor="#1a1a2e",
        edgecolor="#e94560",
        labelcolor="white",
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Layout saved → {save_path}")

    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history:   list[dict],
    save_path: Optional[str] = None,
    show:      bool = True,
) -> plt.Figure:
    """Plot PPO training metrics over episodes."""
    if not history:
        print("No training history to plot.")
        return

    episodes   = [h["episode"]    for h in history]
    avg_rewards= [h["avg_reward"] for h in history]
    pg_losses  = [h.get("pg_loss",  0) for h in history]
    vf_losses  = [h.get("vf_loss",  0) for h in history]
    entropies  = [h.get("entropy",  0) for h in history]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")
        return ax

    # 1. Average reward
    ax1 = styled_ax(gs[0, 0])
    ax1.plot(episodes, avg_rewards, color="#e94560", linewidth=2, label="Avg Reward")
    ax1.fill_between(episodes, avg_rewards, alpha=0.2, color="#e94560")
    # Smoothed
    if len(avg_rewards) > 5:
        smooth = np.convolve(avg_rewards, np.ones(5)/5, mode="valid")
        ax1.plot(episodes[4:], smooth, color="#FFA500", linewidth=1.5,
                 linestyle="--", label="Smoothed")
    ax1.set_title("Average Reward per Update")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(labelcolor="white", facecolor="#1a1a2e", edgecolor="#0f3460")
    ax1.grid(alpha=0.2, color="#0f3460")

    # 2. Policy gradient loss
    ax2 = styled_ax(gs[0, 1])
    ax2.plot(episodes, pg_losses, color="#4682B4", linewidth=2)
    ax2.fill_between(episodes, pg_losses, alpha=0.2, color="#4682B4")
    ax2.set_title("Policy (PG) Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.2, color="#0f3460")

    # 3. Value function loss
    ax3 = styled_ax(gs[1, 0])
    ax3.plot(episodes, vf_losses, color="#3CB371", linewidth=2)
    ax3.fill_between(episodes, vf_losses, alpha=0.2, color="#3CB371")
    ax3.set_title("Value Function Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.grid(alpha=0.2, color="#0f3460")

    # 4. Entropy
    ax4 = styled_ax(gs[1, 1])
    ax4.plot(episodes, entropies, color="#BA55D3", linewidth=2)
    ax4.fill_between(episodes, entropies, alpha=0.2, color="#BA55D3")
    ax4.set_title("Policy Entropy (Exploration)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Entropy")
    ax4.grid(alpha=0.2, color="#0f3460")

    fig.suptitle("PPO Training Curves — Floor Plan RL",
                 color="white", fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Curves saved → {save_path}")

    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layout comparison grid
# ─────────────────────────────────────────────────────────────────────────────

def visualize_multiple_layouts(
    layouts:   list[list[dict]],
    titles:    Optional[list[str]] = None,
    save_path: Optional[str] = None,
    show:      bool = True,
) -> plt.Figure:
    """Show up to 6 layouts in a comparison grid."""
    n   = min(len(layouts), 6)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    fig.patch.set_facecolor("#1a1a2e")

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (ax, layout) in enumerate(zip(axes_flat, layouts[:n])):
        ax.set_facecolor("#16213e")
        for j in range(GRID + 1):
            ax.axhline(j, color="#0f3460", linewidth=0.4, alpha=0.5)
            ax.axvline(j, color="#0f3460", linewidth=0.4, alpha=0.5)

        for room_dict in layout:
            name  = room_dict["room"]
            x, y  = room_dict["x"], room_dict["y"]
            w, h  = room_dict["w"], room_dict["h"]
            color = _room_color(name)
            rect  = patches.Rectangle(
                (x, y), w, h,
                facecolor=to_rgba(color, 0.8),
                edgecolor="white", linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, name.replace("_", "\n"),
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")

        boundary = patches.Rectangle((0, 0), GRID, GRID,
                                      linewidth=2, edgecolor="#e94560", facecolor="none")
        ax.add_patch(boundary)
        ax.set_xlim(0, GRID)
        ax.set_ylim(0, GRID)
        ax.set_aspect("equal")
        title = titles[i] if titles and i < len(titles) else f"Layout {i+1}"
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Floor Plan Comparison", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Comparison saved → {save_path}")

    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: visualize a hand-crafted layout (no training required)
    demo_layout = [
        {"room": "living_room",    "x": 0,  "y": 0,  "w": 6, "h": 5},
        {"room": "kitchen",        "x": 6,  "y": 0,  "w": 4, "h": 4},
        {"room": "dining_room",    "x": 6,  "y": 4,  "w": 4, "h": 4},
        {"room": "master_bedroom", "x": 0,  "y": 5,  "w": 5, "h": 4},
        {"room": "bedroom",        "x": 5,  "y": 8,  "w": 4, "h": 4},
        {"room": "bathroom",       "x": 5,  "y": 5,  "w": 2, "h": 3},
        {"room": "study",          "x": 10, "y": 0,  "w": 3, "h": 3},
        {"room": "parking",        "x": 10, "y": 3,  "w": 3, "h": 5},
    ]
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    visualize_layout(demo_layout, title="Demo Floor Plan (No Training)",
                     save_path=str(out_dir / "demo_layout.png"), show=False)
    print("Demo layout saved to outputs/demo_layout.png")

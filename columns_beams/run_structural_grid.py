"""
run_structural_grid.py
======================
End-to-end runner: room layout → structural grid → visualizations.

Usage:
    python run_structural_grid.py                 # demo layout
    python run_structural_grid.py --json layout.json
    python run_structural_grid.py --floors 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from structural_grid import StructuralGridGenerator, MultiFloorStructuralGrid
from structural_visualize import visualize_structural_grid, visualize_multifloor_section

# ─────────────────────────────────────────────────────────────────────────────
# Demo layouts
# ─────────────────────────────────────────────────────────────────────────────

DEMO_BUILDING = {
    "floor_1": [
        {"room":"living_room",  "x":0,  "y":0,  "w":6, "h":5},
        {"room":"kitchen",      "x":6,  "y":0,  "w":4, "h":4},
        {"room":"dining_room",  "x":6,  "y":4,  "w":4, "h":4},
        {"room":"bathroom",     "x":10, "y":0,  "w":2, "h":3},
        {"room":"parking",      "x":12, "y":0,  "w":4, "h":5},
        {"room":"staircase",    "x":0,  "y":8,  "w":3, "h":3, "structural":True},
        {"room":"elevator",     "x":3,  "y":8,  "w":2, "h":2, "structural":True},
    ],
    "floor_2": [
        {"room":"master_bedroom","x":0,  "y":0,  "w":5, "h":4},
        {"room":"bedroom",       "x":5,  "y":0,  "w":4, "h":4},
        {"room":"bathroom",      "x":9,  "y":0,  "w":2, "h":2},
        {"room":"study",         "x":11, "y":0,  "w":3, "h":3},
        {"room":"bedroom",       "x":0,  "y":4,  "w":4, "h":4},
        {"room":"staircase",     "x":0,  "y":8,  "w":3, "h":3, "structural":True},
        {"room":"elevator",      "x":3,  "y":8,  "w":2, "h":2, "structural":True},
    ],
    "floor_3": [
        {"room":"bedroom",       "x":0,  "y":0,  "w":4, "h":4},
        {"room":"study",         "x":4,  "y":0,  "w":3, "h":3},
        {"room":"bathroom",      "x":10, "y":0,  "w":2, "h":2},
        {"room":"master_bedroom","x":9,  "y":0,  "w":5, "h":4},
        {"room":"staircase",     "x":0,  "y":8,  "w":3, "h":3, "structural":True},
        {"room":"elevator",      "x":3,  "y":8,  "w":2, "h":2, "structural":True},
    ],
}


def run(building: dict, out_dir: Path, verbose: bool = True):
    out_dir.mkdir(exist_ok=True)

    mf = MultiFloorStructuralGrid(grid_size=20)

    # Train ML model
    print("\n" + "═"*60)
    print("  STRUCTURAL GRID GENERATOR")
    print("  Scikit-learn GradientBoosting Column Spacing Predictor")
    print("═"*60)
    metrics = mf.train(verbose=verbose)
    print(f"  Model CV RMSE: {metrics['cv_rmse_m']:.4f} m")

    # Generate all floors
    grids = mf.generate_all(building, verbose=verbose)

    # Print summaries
    for fn, grid in sorted(grids.items()):
        print(f"\n  Floor {fn}:")
        print(grid.summary())

    # Full JSON output
    result = mf.to_dict()

    # Save JSON
    with open(out_dir / "structural_grid.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Full JSON → {out_dir}/structural_grid.json")

    # Print spec-format output for floor 1
    print("\n── Spec-format output (floor_1) ──")
    f1 = result.get("floor_1", {})
    spec_out = {
        "columns": f1.get("columns", [])[:10],
        "beams":   [{"start":b["start"],"end":b["end"]} for b in f1.get("beams",[])[:8]],
        "slabs":   [{"corners":s["corners"],"area_m2":s["area_m2"]} for s in f1.get("slabs",[])[:4]],
    }
    print(json.dumps(spec_out, indent=2))

    # Visualize each floor
    for fn, grid in sorted(grids.items()):
        layout     = building.get(f"floor_{fn}", [])
        load_arr   = grid.load_map.load_array if grid.load_map is not None else None
        grid_dict  = grid.to_dict()
        visualize_structural_grid(
            grid_data  = grid_dict,
            layout     = layout,
            load_array = load_arr,
            floor_num  = fn,
            title      = f"Floor {fn} — Structural Grid Analysis",
            save_path  = str(out_dir / f"structural_floor_{fn}.png"),
        )

    # Multi-floor section
    all_grids_dict = {f"floor_{fn}": g.to_dict() for fn, g in grids.items()}
    visualize_multifloor_section(
        all_grids  = all_grids_dict,
        building   = building,
        save_path  = str(out_dir / "structural_section.png"),
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",   type=str, default=None,
                        help="Path to building layout JSON")
    parser.add_argument("--floors", type=int, default=3)
    parser.add_argument("--out",    type=str, default="outputs")
    args = parser.parse_args()

    if args.json:
        with open(args.json) as f:
            building = json.load(f)
    else:
        # Trim demo to requested floors
        building = {
            k: v for k, v in DEMO_BUILDING.items()
            if k.startswith("floor_") and int(k.split("_")[1]) <= args.floors
        }

    run(building, Path(args.out))

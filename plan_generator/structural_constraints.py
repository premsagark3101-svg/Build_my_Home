"""
structural_constraints.py
=========================
Defines and enforces structural constraints that must hold
across ALL floors of a multi-floor building:

  1. Structural columns / load-bearing walls must align vertically
  2. Plumbing stacks — bathrooms must share X-column across floors
  3. Staircases — same (x,y) footprint on every floor
  4. Elevators  — same (x,y) footprint on every floor
  5. Wet walls  — kitchen drain stacks should align

These constraints are computed from the GROUND FLOOR layout and
then passed as hard constraints to all upper-floor agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

GRID = 20

# Size constants (grid cells)
STAIRCASE_W, STAIRCASE_H = 3, 3
ELEVATOR_W,  ELEVATOR_H  = 2, 2
COLUMN_GRID  = 4          # structural columns every N cells


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructuralCore:
    """
    The 'core' of the building — elements that must appear at the
    same (x, y) on every floor.
    """
    staircase:      Optional[tuple[int,int]] = None   # (x, y) top-left
    elevator:       Optional[tuple[int,int]] = None   # (x, y) top-left
    plumbing_stacks: list[tuple[int,int]]    = field(default_factory=list)
    # Bathroom zones: list of (x,y) top-lefts for 2×2 wet-zone columns
    wet_zones:       list[tuple[int,int]]    = field(default_factory=list)
    # Structural columns: grid of (x,y) anchor points
    columns:         list[tuple[int,int]]    = field(default_factory=list)
    # Kitchen drain stack
    kitchen_stack:   Optional[tuple[int,int]]= None

    def staircase_cells(self) -> list[tuple[int,int]]:
        if not self.staircase:
            return []
        x, y = self.staircase
        return [(x+dx, y+dy) for dx in range(STAIRCASE_W) for dy in range(STAIRCASE_H)]

    def elevator_cells(self) -> list[tuple[int,int]]:
        if not self.elevator:
            return []
        x, y = self.elevator
        return [(x+dx, y+dy) for dx in range(ELEVATOR_W) for dy in range(ELEVATOR_H)]

    def reserved_cells(self) -> set[tuple[int,int]]:
        """All cells reserved by the structural core on any floor."""
        cells = set(self.staircase_cells() + self.elevator_cells())
        return cells

    def to_dict(self) -> dict:
        return {
            "staircase":       list(self.staircase) if self.staircase else None,
            "staircase_size":  [STAIRCASE_W, STAIRCASE_H],
            "elevator":        list(self.elevator) if self.elevator else None,
            "elevator_size":   [ELEVATOR_W, ELEVATOR_H],
            "plumbing_stacks": [list(p) for p in self.plumbing_stacks],
            "wet_zones":       [list(z) for z in self.wet_zones],
            "kitchen_stack":   list(self.kitchen_stack) if self.kitchen_stack else None,
            "columns":         [list(c) for c in self.columns],
        }


@dataclass
class FloorConstraints:
    """
    Per-floor constraints derived from the StructuralCore.
    Passed to each upper-floor agent to guide placement.
    """
    floor_num:         int
    core:              StructuralCore
    # Cells that MUST be occupied by structural elements (pre-placed)
    pre_placed:        list[dict]   = field(default_factory=list)
    # Cells that rooms MUST NOT overlap
    reserved_cells:    set[tuple[int,int]] = field(default_factory=set)
    # Preferred zones for bathrooms (near plumbing stacks)
    bathroom_anchors:  list[tuple[int,int]] = field(default_factory=list)
    # Preferred zones for wet rooms (kitchen)
    kitchen_anchors:   list[tuple[int,int]] = field(default_factory=list)

    def occupied_map(self) -> np.ndarray:
        """Binary grid marking all reserved/pre-placed cells."""
        grid = np.zeros((GRID, GRID), dtype=np.float32)
        for (x, y) in self.reserved_cells:
            if 0 <= x < GRID and 0 <= y < GRID:
                grid[x, y] = 1.0
        return grid


# ─────────────────────────────────────────────────────────────────────────────
# Core Extractor  — reads ground floor layout → produces StructuralCore
# ─────────────────────────────────────────────────────────────────────────────

class StructuralCoreExtractor:
    """
    Analyses a ground floor layout to extract all vertical structural
    elements that must align across upper floors.
    """

    def extract(
        self,
        ground_layout:  list[dict],
        n_floors:       int,
        place_elevator: bool = True,
    ) -> StructuralCore:
        core = StructuralCore()

        # ── 1. Plumbing stacks — from bathrooms ────────────────────────
        bathrooms = [r for r in ground_layout if "bathroom" in r["room"]]
        for bath in bathrooms:
            # Stack anchored at top-left of bathroom
            core.plumbing_stacks.append((bath["x"], bath["y"]))
            core.wet_zones.append((bath["x"], bath["y"]))

        # ── 2. Kitchen drain stack ─────────────────────────────────────
        kitchens = [r for r in ground_layout if r["room"] == "kitchen"]
        if kitchens:
            k = kitchens[0]
            core.kitchen_stack = (k["x"], k["y"])

        # ── 3. Staircase placement ─────────────────────────────────────
        # Place staircase near the centre of the building,
        # away from large living spaces, respecting existing rooms.
        occupied = self._occupied_set(ground_layout)
        core.staircase = self._find_staircase_pos(ground_layout, occupied)

        # ── 4. Elevator (for 3+ floors) ────────────────────────────────
        if place_elevator and n_floors >= 3:
            stair_cells = set(core.staircase_cells()) if core.staircase else set()
            core.elevator = self._find_elevator_pos(occupied | stair_cells)

        # ── 5. Structural columns ──────────────────────────────────────
        core.columns = self._grid_columns()

        return core

    # ------------------------------------------------------------------

    def _occupied_set(self, layout: list[dict]) -> set[tuple[int,int]]:
        cells = set()
        for r in layout:
            for dx in range(r["w"]):
                for dy in range(r["h"]):
                    cells.add((r["x"]+dx, r["y"]+dy))
        return cells

    def _find_staircase_pos(
        self, layout: list[dict], occupied: set[tuple[int,int]]
    ) -> tuple[int,int]:
        """
        Find best staircase position: near centre, not overlapping rooms,
        preferably touching a corridor area.
        """
        cx, cy = GRID // 2, GRID // 2
        best_pos  = (GRID - STAIRCASE_W - 1, 0)
        best_score = -999.0

        for x in range(1, GRID - STAIRCASE_W):
            for y in range(1, GRID - STAIRCASE_H):
                cells = set((x+dx, y+dy)
                            for dx in range(STAIRCASE_W)
                            for dy in range(STAIRCASE_H))
                if cells & occupied:
                    continue
                # Score: prefer centre proximity + edge proximity
                dist_centre = abs(x + STAIRCASE_W/2 - cx) + abs(y + STAIRCASE_H/2 - cy)
                # Bonus for being adjacent to a room (corridor junction)
                adjacent_rooms = sum(
                    1 for (ax, ay) in cells
                    for (nx, ny) in [(ax-1,ay),(ax+1,ay),(ax,ay-1),(ax,ay+1)]
                    if (nx, ny) in occupied
                )
                score = adjacent_rooms * 2.0 - dist_centre * 0.3
                if score > best_score:
                    best_score = score
                    best_pos   = (x, y)

        return best_pos

    def _find_elevator_pos(self, occupied: set[tuple[int,int]]) -> tuple[int,int]:
        """Place elevator adjacent to staircase if possible."""
        # Try positions near staircase
        for x in range(1, GRID - ELEVATOR_W):
            for y in range(1, GRID - ELEVATOR_H):
                cells = set((x+dx, y+dy)
                            for dx in range(ELEVATOR_W)
                            for dy in range(ELEVATOR_H))
                if not (cells & occupied):
                    return (x, y)
        return (GRID - ELEVATOR_W - 1, GRID - ELEVATOR_H - 1)

    def _grid_columns(self) -> list[tuple[int,int]]:
        """Structural column grid — every COLUMN_GRID cells."""
        cols = []
        for x in range(0, GRID, COLUMN_GRID):
            for y in range(0, GRID, COLUMN_GRID):
                cols.append((x, y))
        return cols


# ─────────────────────────────────────────────────────────────────────────────
# Floor Constraint Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_floor_constraints(
    core:      StructuralCore,
    floor_num: int,
    is_ground: bool = False,
) -> FloorConstraints:
    """
    Build the FloorConstraints object for a given floor number,
    using the extracted StructuralCore.
    """
    reserved = core.reserved_cells()

    # Pre-placed structural rooms (staircase + elevator on every floor)
    pre_placed = []
    if core.staircase:
        pre_placed.append({
            "room": "staircase",
            "x": core.staircase[0],
            "y": core.staircase[1],
            "w": STAIRCASE_W,
            "h": STAIRCASE_H,
            "structural": True,
        })
    if core.elevator:
        pre_placed.append({
            "room": "elevator",
            "x": core.elevator[0],
            "y": core.elevator[1],
            "w": ELEVATOR_W,
            "h": ELEVATOR_H,
            "structural": True,
        })

    # Bathroom anchors — prefer placement near existing stack positions
    bath_anchors = []
    for (sx, sy) in core.plumbing_stacks:
        # Offer a 3×3 neighbourhood around the stack
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < GRID-2 and 0 <= ny < GRID-2 and (nx,ny) not in reserved:
                    bath_anchors.append((nx, ny))

    # Kitchen anchors — near the kitchen drain stack
    kitchen_anchors = []
    if core.kitchen_stack:
        kx, ky = core.kitchen_stack
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = kx + dx, ky + dy
                if 0 <= nx < GRID-3 and 0 <= ny < GRID-4 and (nx,ny) not in reserved:
                    kitchen_anchors.append((nx, ny))

    return FloorConstraints(
        floor_num       = floor_num,
        core            = core,
        pre_placed      = pre_placed,
        reserved_cells  = reserved,
        bathroom_anchors= bath_anchors,
        kitchen_anchors = kitchen_anchors,
    )

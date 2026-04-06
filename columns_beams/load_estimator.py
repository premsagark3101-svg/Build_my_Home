"""
load_estimator.py
=================
Estimates structural load distribution from room layout coordinates.

Engineering basis:
  - Dead load  : self-weight of slabs, walls, finishes
  - Live load  : occupancy loads per room type (IS 875 / ASCE 7)
  - Total load  = Dead + Live  (kN/m²)

The load map drives column placement — high-load zones need
closer column spacing; light zones can use longer spans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Occupancy loads  (kN/m²)  — IS 875 Part 2 / ASCE 7-22
# ─────────────────────────────────────────────────────────────────────────────

LIVE_LOADS: dict[str, float] = {
    "living_room":    2.0,   # residential
    "dining_room":    2.0,
    "bedroom":        1.5,
    "master_bedroom": 1.5,
    "bathroom":       2.0,
    "kitchen":        3.0,   # heavier equipment
    "study":          2.5,
    "home_office":    2.5,
    "parking":        5.0,   # vehicle loads
    "staircase":      3.0,
    "elevator":       5.0,   # machinery
    "gym":            5.0,
    "storage":        5.0,
    "laundry":        3.0,
    "terrace":        4.0,
    "balcony":        3.0,
    "utility_room":   3.0,
    "garage":         5.0,
    "guest_room":     1.5,
}

DEAD_LOAD_SLAB    = 3.75   # kN/m²  (150mm RCC slab)
DEAD_LOAD_FINISH  = 1.00   # kN/m²  (screed + tiles)
DEAD_LOAD_WALLS   = 2.50   # kN/m²  (partition walls — averaged over slab)
TOTAL_DEAD        = DEAD_LOAD_SLAB + DEAD_LOAD_FINISH + DEAD_LOAD_WALLS

LOAD_FACTOR_DL    = 1.5    # IS 456 partial safety factor
LOAD_FACTOR_LL    = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellLoad:
    x:          int
    y:          int
    room_type:  str
    dead_kn_m2: float
    live_kn_m2: float
    total_kn_m2:float
    factored:   float

    @property
    def is_heavy(self) -> bool:
        return self.live_kn_m2 >= 4.0

    @property
    def is_wet(self) -> bool:
        return self.room_type in ("bathroom", "kitchen", "laundry", "utility_room")


@dataclass
class LoadMap:
    grid_size:   int
    cells:       list[list[Optional[CellLoad]]]     # [x][y]
    load_array:  np.ndarray                          # (grid, grid) factored kN/m²
    floor_num:   int = 1
    n_floors:    int = 1

    def max_load(self)  -> float: return float(self.load_array.max())
    def mean_load(self) -> float: return float(self.load_array[self.load_array > 0].mean()) if (self.load_array > 0).any() else 0.0

    def load_at(self, x: int, y: int) -> float:
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return float(self.load_array[x, y])
        return 0.0

    def get_feature_vector(self, x: int, y: int, radius: int = 2) -> np.ndarray:
        """
        Extract a feature vector around cell (x,y) for ML prediction.
        Features: local load stats + position encoding.
        """
        x1 = max(0, x - radius); x2 = min(self.grid_size, x + radius + 1)
        y1 = max(0, y - radius); y2 = min(self.grid_size, y + radius + 1)
        patch = self.load_array[x1:x2, y1:y2]

        load_mean  = float(patch.mean())
        load_max   = float(patch.max())
        load_std   = float(patch.std())
        load_sum   = float(patch.sum())
        nonzero    = float((patch > 0).sum()) / max(patch.size, 1)
        heavy_frac = float((patch >= 6.0).sum()) / max(patch.size, 1)

        # Position encoding (normalised)
        pos_x = x / self.grid_size
        pos_y = y / self.grid_size

        # Distance from boundary
        dist_edge = min(x, y, self.grid_size - x, self.grid_size - y) / self.grid_size

        # Floor factor
        floor_factor = (self.n_floors - self.floor_num + 1) / self.n_floors

        return np.array([
            load_mean, load_max, load_std, load_sum, nonzero,
            heavy_frac, pos_x, pos_y, dist_edge, floor_factor,
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Load Estimator
# ─────────────────────────────────────────────────────────────────────────────

class LoadEstimator:
    """
    Converts a room layout into a cell-by-cell factored load map.

    Parameters
    ----------
    grid_size : grid dimension in metres (1 cell = 1 m)
    floor_num : which floor (affects cumulative load)
    n_floors  : total floors (ground floor carries more load)
    """

    def __init__(self, grid_size: int = 20, floor_num: int = 1, n_floors: int = 1):
        self.grid_size = grid_size
        self.floor_num = floor_num
        self.n_floors  = n_floors

    def estimate(self, layout: list[dict]) -> LoadMap:
        """
        Parameters
        ----------
        layout : list of room dicts {"room", "x", "y", "w", "h"}

        Returns
        -------
        LoadMap  with per-cell factored loads
        """
        G = self.grid_size
        cells: list[list[Optional[CellLoad]]] = [[None]*G for _ in range(G)]
        load_array = np.zeros((G, G), dtype=np.float32)

        # Cumulative load factor for lower floors (carry upper floors)
        cumulative_factor = (self.n_floors - self.floor_num + 1)

        for room in layout:
            rtype = room.get("room", "bedroom")
            x0, y0 = room["x"], room["y"]
            w,  h  = room["w"], room["h"]
            live   = LIVE_LOADS.get(rtype, 2.0)
            dead   = TOTAL_DEAD

            for dx in range(w):
                for dy in range(h):
                    rx, ry = x0 + dx, y0 + dy
                    if not (0 <= rx < G and 0 <= ry < G):
                        continue

                    # Edge cells have higher wall load
                    is_edge = (dx == 0 or dx == w-1 or dy == 0 or dy == h-1)
                    wall_extra = 1.5 if is_edge else 0.0

                    cell_dead = dead + wall_extra
                    factored  = (
                        LOAD_FACTOR_DL * cell_dead * cumulative_factor +
                        LOAD_FACTOR_LL * live
                    )

                    cells[rx][ry] = CellLoad(
                        x=rx, y=ry, room_type=rtype,
                        dead_kn_m2=cell_dead,
                        live_kn_m2=live,
                        total_kn_m2=cell_dead + live,
                        factored=factored,
                    )
                    load_array[rx, ry] = factored

        return LoadMap(
            grid_size  = G,
            cells      = cells,
            load_array = load_array,
            floor_num  = self.floor_num,
            n_floors   = self.n_floors,
        )

    def combine_floors(self, floor_maps: list[LoadMap]) -> LoadMap:
        """
        Sum loads from all floors to get ground-floor cumulative load.
        Used for foundation design.
        """
        if not floor_maps:
            raise ValueError("No floor maps provided.")
        G      = floor_maps[0].grid_size
        combined = np.zeros((G, G), dtype=np.float32)
        for fm in floor_maps:
            combined += fm.load_array
        return LoadMap(
            grid_size  = G,
            cells      = floor_maps[0].cells,
            load_array = combined,
            floor_num  = 0,
            n_floors   = len(floor_maps),
        )

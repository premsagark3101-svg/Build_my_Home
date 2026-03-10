"""
floor_plan_env.py
=================
Gymnasium-compatible Reinforcement Learning environment for
sequential floor plan generation using PPO.

State space:
  - placed_rooms     : binary occupancy grid  [GRID x GRID]
  - empty_space_map  : float distance-to-wall  [GRID x GRID]
  - remaining_rooms  : one-hot room type vector [NUM_ROOM_TYPES]
  - room_progress    : scalar (fraction placed)

Action space (Discrete):
  place_room(x, y)  → GRID*GRID positions
  Each action = grid cell to place top-left corner of current room.
  Width/height come from the room spec; rotation toggles aspect ratio.

Reward shaping:
  +2.0   valid placement (no overlap, in boundary)
  +0.5   each satisfied adjacency constraint
  +0.3   compact layout bonus (room touching existing room)
  +1.0   all rooms placed successfully
  -1.0   placement causes overlap
  -1.5   placement goes outside boundary
  -0.2   wasted action (already tried position)
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRID = 20          # grid cells (each cell = 1 metre)
NUM_ROOM_TYPES = 8

ROOM_TYPES = [
    "master_bedroom", "bedroom", "bathroom", "kitchen",
    "living_room", "dining_room", "study", "parking"
]

# Default room sizes [w, h] in grid cells — (rotatable)
DEFAULT_SIZES: dict[str, list[int]] = {
    "master_bedroom": [4, 5],
    "bedroom":        [3, 4],
    "bathroom":       [2, 2],
    "kitchen":        [3, 4],
    "living_room":    [5, 6],
    "dining_room":    [4, 4],
    "study":          [3, 3],
    "parking":        [3, 5],
}

# Adjacency preferences: room_a should be near room_b
ADJACENCY_PREFS: list[tuple[str, str]] = [
    ("master_bedroom", "bathroom"),
    ("bedroom",        "bathroom"),
    ("kitchen",        "living_room"),
    ("kitchen",        "dining_room"),
    ("living_room",    "dining_room"),
    ("parking",        "living_room"),  # near entrance
]

# Separation preferences: room_a should NOT touch room_b
SEPARATION_PREFS: list[tuple[str, str]] = [
    ("bedroom",        "kitchen"),
    ("master_bedroom", "kitchen"),
    ("parking",        "bedroom"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlacedRoom:
    room:    str
    x:       int
    y:       int
    w:       int
    h:       int
    rotated: bool = False

    def to_dict(self) -> dict:
        return {"room": self.room, "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def cells(self) -> list[tuple[int, int]]:
        return [(self.x + dx, self.y + dy)
                for dx in range(self.w) for dy in range(self.h)]

    def bbox(self) -> tuple[int, int, int, int]:
        """x1, y1, x2, y2 (inclusive)"""
        return self.x, self.y, self.x + self.w - 1, self.y + self.h - 1

    def is_adjacent_to(self, other: "PlacedRoom") -> bool:
        """True if rooms share an edge (1-cell gap or touching)."""
        x1, y1, x2, y2 = self.bbox()
        ox1, oy1, ox2, oy2 = other.bbox()
        # Expand self by 1 and check overlap
        return not (x2 + 1 < ox1 or ox2 + 1 < x1 or y2 + 1 < oy1 or oy2 + 1 < y1)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class FloorPlanEnv:
    """
    Gymnasium-compatible environment for sequential room placement.

    Observation shape : (GRID, GRID, 3) + (NUM_ROOM_TYPES + 1,)
      Channel 0 : occupancy map   (0=empty, 1=placed)
      Channel 1 : room-id map     (0=empty, room_type_index+1)
      Channel 2 : distance map    (normalised distance to nearest wall/edge)
    Vector     : remaining_rooms one-hot + progress scalar

    Action : int in [0, GRID*GRID)  → (x, y) = (action % GRID, action // GRID)
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, rooms_to_place: Optional[list[str]] = None, seed: int = 42):
        self.grid_size  = GRID
        self.n_actions  = GRID * GRID

        # Observation spaces (described for PPO policy)
        self.obs_grid_shape = (3, GRID, GRID)       # CHW for CNN
        self.obs_vec_size   = NUM_ROOM_TYPES + 1    # remaining + progress

        if rooms_to_place is None:
            rooms_to_place = [
                "living_room", "kitchen", "dining_room",
                "master_bedroom", "bedroom", "bathroom", "study"
            ]
        self.rooms_to_place = rooms_to_place
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.reset()

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed: int = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self.occupancy      = np.zeros((GRID, GRID), dtype=np.float32)
        self.room_id_map    = np.zeros((GRID, GRID), dtype=np.float32)
        self.placed_rooms:  list[PlacedRoom] = []
        self.remaining:     list[str] = list(self.rooms_to_place)
        self._tried_actions: set[int] = set()
        self.step_count     = 0
        self.total_reward   = 0.0
        self.done           = False

        # Shuffle remaining to add variety
        self._rng.shuffle(self.remaining)

        obs  = self._get_obs()
        info = {"remaining": list(self.remaining), "placed": []}
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        action : int  [0, GRID*GRID)
        Returns (obs, reward, terminated, truncated, info)
        """
        assert not self.done, "Call reset() before stepping."
        self.step_count += 1

        x = int(action % GRID)
        y = int(action // GRID)
        room_name = self.remaining[0]
        w, h = DEFAULT_SIZES.get(room_name, [3, 3])

        # Try rotation with 30% probability during training
        rotated = False
        if self._rng.random() < 0.3 and w != h:
            w, h = h, w
            rotated = True

        reward, placement_ok = self._compute_reward(room_name, x, y, w, h, action)

        if placement_ok:
            room = PlacedRoom(room_name, x, y, w, h, rotated)
            self._place(room)
            self.remaining.pop(0)
            self._tried_actions.clear()
        else:
            self._tried_actions.add(action)

        terminated = len(self.remaining) == 0
        truncated  = self.step_count >= GRID * GRID * 2   # safety limit

        if terminated:
            reward += self._final_layout_bonus()
            self.done = True
        elif truncated:
            reward -= 0.5
            self.done = True

        self.total_reward += reward
        obs  = self._get_obs()
        info = {
            "placed":    [r.to_dict() for r in self.placed_rooms],
            "remaining": list(self.remaining),
            "step":      self.step_count,
            "ok":        placement_ok,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Return RGB grid image."""
        COLORS = {
            "master_bedroom": (70,  130, 180),
            "bedroom":        (100, 149, 237),
            "bathroom":       (32,  178, 170),
            "kitchen":        (255, 165,  0),
            "living_room":    (60,  179, 113),
            "dining_room":    (154, 205,  50),
            "study":          (186,  85, 211),
            "parking":        (169, 169, 169),
        }
        EMPTY  = (240, 240, 230)
        BORDER = (50, 50, 50)
        CELL   = 30   # pixels per cell

        img = np.full((GRID * CELL, GRID * CELL, 3), EMPTY, dtype=np.uint8)

        for room in self.placed_rooms:
            color = COLORS.get(room.room, (200, 200, 200))
            x1 = room.x * CELL
            y1 = room.y * CELL
            x2 = (room.x + room.w) * CELL
            y2 = (room.y + room.h) * CELL
            img[y1:y2, x1:x2] = color
            # Inner border
            img[y1:y1+2, x1:x2] = BORDER
            img[y2-2:y2, x1:x2] = BORDER
            img[y1:y2, x1:x1+2] = BORDER
            img[y1:y2, x2-2:x2] = BORDER

        # Grid lines
        for i in range(0, GRID * CELL, CELL):
            img[i, :] = (200, 200, 200)
            img[:, i] = (200, 200, 200)

        return img

    def get_layout_json(self) -> list[dict]:
        return [r.to_dict() for r in self.placed_rooms]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        dist_map = self._distance_map()
        grid_obs = np.stack([
            self.occupancy,
            self.room_id_map / (NUM_ROOM_TYPES + 1),
            dist_map,
        ], axis=0).astype(np.float32)  # (3, GRID, GRID)

        remaining_vec = np.zeros(NUM_ROOM_TYPES, dtype=np.float32)
        for r in self.remaining:
            idx = ROOM_TYPES.index(r) if r in ROOM_TYPES else 0
            remaining_vec[idx] = min(remaining_vec[idx] + 1, 1.0)

        progress = 1.0 - len(self.remaining) / max(len(self.rooms_to_place), 1)
        vec_obs  = np.append(remaining_vec, progress).astype(np.float32)

        return {"grid": grid_obs, "vec": vec_obs}

    def _distance_map(self) -> np.ndarray:
        """Normalised distance of each cell from nearest occupied cell / wall."""
        if not self.placed_rooms:
            # Full empty — distance from centre
            cx, cy = GRID // 2, GRID // 2
            xs = np.arange(GRID)
            ys = np.arange(GRID)
            xg, yg = np.meshgrid(xs, ys, indexing='ij')
            d = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
            return (1.0 - d / d.max()).astype(np.float32)
        # Distance transform approximation
        occ = self.occupancy.copy()
        d   = np.ones((GRID, GRID), dtype=np.float32) * GRID
        for r in self.placed_rooms:
            x1, y1, x2, y2 = r.bbox()
            for x in range(GRID):
                for y in range(GRID):
                    dist = min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2))
                    d[x, y] = min(d[x, y], dist)
        return (1.0 - np.clip(d / GRID, 0, 1)).astype(np.float32)

    def _compute_reward(
        self, room: str, x: int, y: int, w: int, h: int, action: int
    ) -> tuple[float, bool]:
        reward = 0.0

        # ── Boundary check ────────────────────────────────────────────
        if x < 0 or y < 0 or x + w > GRID or y + h > GRID:
            return -1.5, False

        # ── Overlap check ─────────────────────────────────────────────
        test_cells = set((x + dx, y + dy) for dx in range(w) for dy in range(h))
        for pr in self.placed_rooms:
            existing = set(pr.cells())
            if test_cells & existing:
                return -1.0, False

        # ── Repeated action penalty ────────────────────────────────────
        if action in self._tried_actions:
            return -0.2, False

        # ── Valid placement ────────────────────────────────────────────
        reward += 2.0

        # ── Adjacency reward ──────────────────────────────────────────
        test_room = PlacedRoom(room, x, y, w, h)
        for pr in self.placed_rooms:
            if test_room.is_adjacent_to(pr):
                pair = (room, pr.room)
                rpair = (pr.room, room)
                if pair in ADJACENCY_PREFS or rpair in ADJACENCY_PREFS:
                    reward += 0.5
                if pair in SEPARATION_PREFS or rpair in SEPARATION_PREFS:
                    reward -= 0.4

        # ── Compactness reward ────────────────────────────────────────
        if self.placed_rooms and test_room.is_adjacent_to(self.placed_rooms[-1]):
            reward += 0.3

        # ── Centre bias (prefer not placing far from centre) ──────────
        cx = abs(x + w / 2 - GRID / 2) / GRID
        cy = abs(y + h / 2 - GRID / 2) / GRID
        reward += 0.2 * (1.0 - (cx + cy) / 2)

        return reward, True

    def _final_layout_bonus(self) -> float:
        """Bonus computed once all rooms are placed."""
        bonus = 1.0   # completion bonus

        # Check all adjacency preferences satisfied
        satisfied = 0
        for a, b in ADJACENCY_PREFS:
            rooms_a = [r for r in self.placed_rooms if r.room == a]
            rooms_b = [r for r in self.placed_rooms if r.room == b]
            for ra in rooms_a:
                for rb in rooms_b:
                    if ra.is_adjacent_to(rb):
                        satisfied += 1
                        break

        bonus += satisfied * 0.3

        # Compactness: bounding box efficiency
        if self.placed_rooms:
            xs = [r.x for r in self.placed_rooms]
            ys = [r.y for r in self.placed_rooms]
            xe = [r.x + r.w for r in self.placed_rooms]
            ye = [r.y + r.h for r in self.placed_rooms]
            bbox_area   = (max(xe) - min(xs)) * (max(ye) - min(ys))
            total_room  = sum(r.w * r.h for r in self.placed_rooms)
            if bbox_area > 0:
                efficiency = total_room / bbox_area
                bonus += efficiency * 0.5

        return bonus

    def _place(self, room: PlacedRoom):
        for dx in range(room.w):
            for dy in range(room.h):
                self.occupancy[room.x + dx, room.y + dy]   = 1.0
                idx = ROOM_TYPES.index(room.room) + 1 if room.room in ROOM_TYPES else 1
                self.room_id_map[room.x + dx, room.y + dy] = idx
        self.placed_rooms.append(room)

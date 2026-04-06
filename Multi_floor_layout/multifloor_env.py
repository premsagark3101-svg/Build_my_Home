"""
multifloor_env.py
=================
Multi-floor building RL environment implementing a hierarchical
planning system:

  Level 1 Agent (Ground Floor Planner)
    → Plans ground floor freely: living areas, kitchen, parking
    → After completion, extracts StructuralCore

  Level 2 Agent (Upper Floor Planner)
    → Receives structural constraints from Level 1
    → Replicates structural elements (stairs, elevator, wet walls)
    → Plans bedroom / study / bathroom layout with stack alignment
    → Runs once per upper floor

Both agents share the same FloorPlanEnv interface but with
different room lists and constraint maps injected.

Vertical alignment enforced:
  - Staircases: identical footprint every floor
  - Elevators : identical footprint every floor
  - Bathrooms : x-column within 2 cells of plumbing stack
  - Kitchens  : x-column within 2 cells of drain stack
  - Structural columns: load-bearing grid preserved
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from floor_plan_env import (
    PlacedRoom, GRID, NUM_ROOM_TYPES, ROOM_TYPES,
    DEFAULT_SIZES, ADJACENCY_PREFS, SEPARATION_PREFS,
)
from structural_constraints import (
    StructuralCore, FloorConstraints,
    StructuralCoreExtractor, build_floor_constraints,
    STAIRCASE_W, STAIRCASE_H, ELEVATOR_W, ELEVATOR_H,
)


# ─────────────────────────────────────────────────────────────────────────────
# Floor-specific room lists
# ─────────────────────────────────────────────────────────────────────────────

GROUND_FLOOR_ROOMS = [
    "living_room", "kitchen", "dining_room", "bathroom", "parking",
]

UPPER_FLOOR_ROOMS = [
    "master_bedroom", "bedroom", "bedroom", "bathroom", "study",
]

# Alignment reward bonus — given when structural alignment satisfied
ALIGNMENT_REWARD = 1.0
STACK_TOLERANCE  = 2   # cells allowed from plumbing stack


# ─────────────────────────────────────────────────────────────────────────────
# Single-floor environment (used by both L1 and L2 agents)
# ─────────────────────────────────────────────────────────────────────────────

class SingleFloorEnv:
    """
    One floor of the multi-floor building.
    Accepts FloorConstraints to enforce structural rules.
    """

    def __init__(
        self,
        floor_num:   int,
        rooms:       list[str],
        constraints: Optional[FloorConstraints] = None,
        seed:        int = 42,
    ):
        self.floor_num   = floor_num
        self.rooms_list  = list(rooms)
        self.constraints = constraints
        self.n_actions   = GRID * GRID
        self._rng        = random.Random(seed + floor_num * 17)
        self._np_rng     = np.random.default_rng(seed + floor_num * 17)
        self.reset()

    # ── Gymnasium-compatible API ──────────────────────────────────────

    def reset(self) -> tuple[dict, dict]:
        self.occupancy   = np.zeros((GRID, GRID), dtype=np.float32)
        self.room_id_map = np.zeros((GRID, GRID), dtype=np.float32)
        self.placed_rooms: list[PlacedRoom] = []
        self.remaining    = list(self.rooms_list)
        self._tried: set[int] = set()
        self.step_count   = 0
        self.done         = False

        # Pre-place structural elements from constraints
        if self.constraints:
            for pp in self.constraints.pre_placed:
                self._force_place(pp)

        self._rng.shuffle(self.remaining)
        return self._obs(), {"remaining": list(self.remaining), "floor": self.floor_num}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert not self.done
        self.step_count += 1

        x = int(action % GRID)
        y = int(action // GRID)

        if not self.remaining:
            return self._obs(), 0.0, True, False, self._info()

        room_name = self.remaining[0]
        w, h = DEFAULT_SIZES.get(room_name, [3, 3])

        # Stochastic rotation
        rotated = False
        if self._rng.random() < 0.25 and w != h:
            w, h   = h, w
            rotated = True

        reward, ok = self._reward(room_name, x, y, w, h, action)

        if ok:
            room = PlacedRoom(room_name, x, y, w, h, rotated)
            self._place(room)
            self.remaining.pop(0)
            self._tried.clear()
        else:
            self._tried.add(action)

        terminated = len(self.remaining) == 0
        truncated  = self.step_count >= GRID * GRID * 3

        if terminated:
            reward += self._final_bonus()
            self.done = True
        elif truncated:
            reward -= 0.5
            self.done = True

        return self._obs(), reward, terminated, truncated, self._info()

    def layout(self) -> list[dict]:
        result = []
        for r in self.placed_rooms:
            d = r.to_dict()
            d["floor"] = self.floor_num
            result.append(d)
        return result

    # ── Reward ───────────────────────────────────────────────────────

    def _reward(
        self, room: str, x: int, y: int, w: int, h: int, action: int
    ) -> tuple[float, bool]:

        # Boundary
        if x < 0 or y < 0 or x + w > GRID or y + h > GRID:
            return -1.5, False

        # Reserved structural cells
        test_cells = set((x+dx, y+dy) for dx in range(w) for dy in range(h))
        if self.constraints:
            if test_cells & self.constraints.reserved_cells:
                return -1.2, False

        # Overlap with existing rooms
        for pr in self.placed_rooms:
            if test_cells & set(pr.cells()):
                return -1.0, False

        # Repeated action
        if action in self._tried:
            return -0.2, False

        reward = 2.0  # valid placement base

        # ── Structural alignment rewards ──────────────────────────────
        if self.constraints:
            reward += self._alignment_reward(room, x, y)

        # ── Adjacency rewards ─────────────────────────────────────────
        test_room = PlacedRoom(room, x, y, w, h)
        for pr in self.placed_rooms:
            if test_room.is_adjacent_to(pr):
                pair  = (room, pr.room)
                rpair = (pr.room, room)
                if pair in ADJACENCY_PREFS or rpair in ADJACENCY_PREFS:
                    reward += 0.5
                if pair in SEPARATION_PREFS or rpair in SEPARATION_PREFS:
                    reward -= 0.4

        # ── Compactness ───────────────────────────────────────────────
        if self.placed_rooms and test_room.is_adjacent_to(self.placed_rooms[-1]):
            reward += 0.3

        # ── Centre-proximity ──────────────────────────────────────────
        cx = abs(x + w/2 - GRID/2) / GRID
        cy = abs(y + h/2 - GRID/2) / GRID
        reward += 0.15 * (1.0 - (cx+cy)/2)

        return reward, True

    def _alignment_reward(self, room: str, x: int, y: int) -> float:
        """Reward for placing wet rooms near plumbing stacks."""
        bonus = 0.0
        c = self.constraints

        if "bathroom" in room and c.bathroom_anchors:
            # Check if (x,y) is near a stack
            for (ax, ay) in c.core.plumbing_stacks:
                if abs(x - ax) <= STACK_TOLERANCE and abs(y - ay) <= STACK_TOLERANCE:
                    bonus += ALIGNMENT_REWARD
                    break

        if room == "kitchen" and c.kitchen_anchors:
            if c.core.kitchen_stack:
                kx, ky = c.core.kitchen_stack
                if abs(x - kx) <= STACK_TOLERANCE and abs(y - ky) <= STACK_TOLERANCE:
                    bonus += ALIGNMENT_REWARD * 0.7

        # Penalty for wet rooms far from stack
        if "bathroom" in room and c.core.plumbing_stacks:
            min_dist = min(
                abs(x - ax) + abs(y - ay)
                for (ax, ay) in c.core.plumbing_stacks
            )
            if min_dist > STACK_TOLERANCE * 2:
                bonus -= 0.5

        return bonus

    def _final_bonus(self) -> float:
        bonus = 1.0  # completion

        # Verify staircase is accessible from corridor
        if self.constraints and self.constraints.core.staircase:
            sx, sy = self.constraints.core.staircase
            # Any room touching the staircase zone?
            stair_cells = set(self.constraints.core.staircase_cells())
            for pr in self.placed_rooms:
                if pr.is_adjacent_to(PlacedRoom("_", sx, sy, STAIRCASE_W, STAIRCASE_H)):
                    bonus += 0.4
                    break

        # Compactness
        if self.placed_rooms:
            xs  = [r.x for r in self.placed_rooms]
            ys  = [r.y for r in self.placed_rooms]
            xe  = [r.x + r.w for r in self.placed_rooms]
            ye  = [r.y + r.h for r in self.placed_rooms]
            bb  = (max(xe)-min(xs)) * (max(ye)-min(ys))
            area= sum(r.w*r.h for r in self.placed_rooms)
            if bb > 0:
                bonus += (area/bb) * 0.5

        return bonus

    # ── Observation ───────────────────────────────────────────────────

    def _obs(self) -> dict:
        # Constraint map: show reserved cells as channel
        constraint_map = np.zeros((GRID, GRID), dtype=np.float32)
        if self.constraints:
            for (x, y) in self.constraints.reserved_cells:
                if 0 <= x < GRID and 0 <= y < GRID:
                    constraint_map[x, y] = 1.0
            # Mark plumbing stacks
            for (x, y) in self.constraints.core.plumbing_stacks:
                if 0 <= x < GRID and 0 <= y < GRID:
                    constraint_map[x, y] = 0.5  # softer signal

        dist_map = self._dist_map()
        grid_obs = np.stack([
            self.occupancy,
            self.room_id_map / (NUM_ROOM_TYPES + 1),
            dist_map,
            constraint_map,                  # 4th channel: structural map
        ], axis=0).astype(np.float32)        # (4, GRID, GRID)

        remaining_vec = np.zeros(NUM_ROOM_TYPES, dtype=np.float32)
        for r in self.remaining:
            if r in ROOM_TYPES:
                remaining_vec[ROOM_TYPES.index(r)] = min(
                    remaining_vec[ROOM_TYPES.index(r)] + 1, 1.0
                )

        progress = 1.0 - len(self.remaining) / max(len(self.rooms_list), 1)
        floor_enc = self.floor_num / 10.0   # floor encoding
        vec_obs   = np.append(remaining_vec, [progress, floor_enc]).astype(np.float32)

        return {"grid": grid_obs, "vec": vec_obs}

    def _dist_map(self) -> np.ndarray:
        if not self.placed_rooms:
            cx, cy = GRID//2, GRID//2
            xs = np.arange(GRID); ys = np.arange(GRID)
            xg, yg = np.meshgrid(xs, ys, indexing='ij')
            d = np.sqrt((xg-cx)**2 + (yg-cy)**2)
            return (1.0 - d/d.max()).astype(np.float32)
        d = np.ones((GRID, GRID), dtype=np.float32) * GRID
        for r in self.placed_rooms:
            x1,y1,x2,y2 = r.bbox()
            for x in range(GRID):
                for y in range(GRID):
                    dist = min(abs(x-x1), abs(x-x2), abs(y-y1), abs(y-y2))
                    d[x,y] = min(d[x,y], dist)
        return (1.0 - np.clip(d/GRID, 0, 1)).astype(np.float32)

    def _info(self) -> dict:
        return {
            "layout":    self.layout(),
            "remaining": list(self.remaining),
            "floor":     self.floor_num,
            "step":      self.step_count,
        }

    def _place(self, room: PlacedRoom):
        for dx in range(room.w):
            for dy in range(room.h):
                self.occupancy[room.x+dx, room.y+dy] = 1.0
                idx = ROOM_TYPES.index(room.room)+1 if room.room in ROOM_TYPES else 1
                self.room_id_map[room.x+dx, room.y+dy] = float(idx)
        self.placed_rooms.append(room)

    def _force_place(self, room_dict: dict):
        """Place a structural room without reward logic."""
        r = PlacedRoom(
            room    = room_dict["room"],
            x       = room_dict["x"],
            y       = room_dict["y"],
            w       = room_dict["w"],
            h       = room_dict["h"],
        )
        self._place(r)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Floor Environment  — orchestrates L1 and L2 agents
# ─────────────────────────────────────────────────────────────────────────────

class MultiFloorEnv:
    """
    Hierarchical multi-floor building environment.

    Usage
    -----
    env = MultiFloorEnv(n_floors=3)

    # Phase 1: Ground floor (Level-1 agent)
    obs, info = env.reset_floor(1)
    while not done:
        action = l1_agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)

    # After ground floor complete:
    env.finalise_ground_floor()

    # Phase 2: Upper floors (Level-2 agent, floor by floor)
    for floor in range(2, n_floors+1):
        obs, info = env.reset_floor(floor)
        while not done:
            action = l2_agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)

    building = env.get_building_layout()
    """

    OBS_GRID_CHANNELS = 4
    OBS_VEC_SIZE      = NUM_ROOM_TYPES + 2   # remaining + progress + floor_enc

    def __init__(
        self,
        n_floors:        int  = 3,
        ground_rooms:    Optional[list[str]] = None,
        upper_rooms:     Optional[list[str]] = None,
        place_elevator:  bool = True,
        seed:            int  = 42,
    ):
        self.n_floors       = n_floors
        self.ground_rooms   = ground_rooms or GROUND_FLOOR_ROOMS
        self.upper_rooms    = upper_rooms  or UPPER_FLOOR_ROOMS
        self.place_elevator = place_elevator and n_floors >= 3
        self.seed           = seed

        self.core:     Optional[StructuralCore]  = None
        self.floors:   dict[int, SingleFloorEnv] = {}
        self.layouts:  dict[int, list[dict]]     = {}

        self.current_floor: int = 1
        self.current_env:   Optional[SingleFloorEnv] = None
        self.n_actions = GRID * GRID

    def reset_floor(self, floor_num: int) -> tuple[dict, dict]:
        """Reset environment for a specific floor. Returns (obs, info)."""
        self.current_floor = floor_num

        if floor_num == 1:
            # Ground floor: no constraints yet
            env = SingleFloorEnv(
                floor_num   = 1,
                rooms       = self.ground_rooms,
                constraints = None,
                seed        = self.seed,
            )
        else:
            assert self.core is not None, \
                "Call finalise_ground_floor() before planning upper floors."
            fc = build_floor_constraints(
                core      = self.core,
                floor_num = floor_num,
                is_ground = False,
            )
            env = SingleFloorEnv(
                floor_num   = floor_num,
                rooms       = list(self.upper_rooms),
                constraints = fc,
                seed        = self.seed,
            )

        self.floors[floor_num] = env
        self.current_env       = env
        return env.reset()

    def step(self, action: int):
        """Proxy step to the current floor environment."""
        assert self.current_env is not None
        return self.current_env.step(action)

    def finalise_ground_floor(self):
        """
        Called after ground floor episode ends.
        Extracts StructuralCore from ground layout.
        """
        assert 1 in self.floors, "Ground floor not yet planned."
        ground_layout = self.floors[1].layout()
        # Strip structural pre-placed rooms
        room_only = [r for r in ground_layout if not r.get("structural", False)]
        self.layouts[1] = ground_layout

        extractor  = StructuralCoreExtractor()
        self.core  = extractor.extract(
            ground_layout  = room_only,
            n_floors       = self.n_floors,
            place_elevator = self.place_elevator,
        )
        return self.core

    def save_floor_layout(self, floor_num: int):
        """Call after completing an upper floor episode."""
        if floor_num in self.floors:
            self.layouts[floor_num] = self.floors[floor_num].layout()

    def get_building_layout(self) -> dict:
        """
        Return complete multi-floor layout in the specified output format:
        {
          "floor_1": [...],
          "floor_2": [...],
          ...
          "structural_core": {...}
        }
        """
        result = {}
        for fn in range(1, self.n_floors + 1):
            key = f"floor_{fn}"
            if fn in self.layouts:
                result[key] = self.layouts[fn]
            else:
                result[key] = []

        if self.core:
            result["structural_core"] = self.core.to_dict()

        return result

    @property
    def obs_dim(self) -> int:
        return self.OBS_GRID_CHANNELS * GRID * GRID + self.OBS_VEC_SIZE

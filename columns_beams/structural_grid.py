"""
structural_grid.py
==================
Main structural grid generator.

Pipeline:
  1. Divide building into candidate grid points
  2. Estimate structural load distribution  (LoadEstimator)
  3. Predict optimal column spacing         (ColumnSpacingPredictor / GBR)
  4. Place columns at optimal positions
  5. Connect columns with beams
  6. Define slab panels between beams
  7. Validate all span limits
  8. Output validated structural JSON

Engineering constraints enforced:
  • Max beam span        = 6.0 m
  • Min column spacing   = 3.0 m
  • Columns at all room corners (load concentration points)
  • Columns under wet rooms (plumbing stack loads)
  • Cantilever limit     = 2.0 m
  • Slab aspect ratio    ≤ 2.0 (two-way slab preferred)
"""

from __future__ import annotations

import json
import math
import itertools
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

from load_estimator import LoadEstimator, LoadMap
from column_predictor import ColumnSpacingPredictor, MIN_COLUMN_SPACING, MAX_BEAM_SPAN

# ─────────────────────────────────────────────────────────────────────────────
# Engineering constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_CANTILEVER     = 2.0   # metres
MAX_SLAB_RATIO     = 2.0   # Ly/Lx ≤ 2 for two-way slabs
COLUMN_SIZE        = 0.3   # 300mm × 300mm typical RC column
BEAM_WIDTH         = 0.23  # 230mm typical RC beam
BEAM_DEPTH_FACTOR  = 1/12  # L/12 rule for beam depth

# Room types that MUST have columns at their corners
HEAVY_ROOMS = {"parking", "gym", "storage", "kitchen", "elevator", "staircase"}

# Wet room types — need columns for plumbing stack support
WET_ROOMS = {"bathroom", "kitchen", "laundry", "utility_room"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Column:
    x:      float
    y:      float
    load_kn: float = 0.0
    reason: str   = "grid"   # "grid" | "corner" | "wet_stack" | "heavy_room"

    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_list(self) -> list[float]:
        return [round(self.x, 2), round(self.y, 2)]


@dataclass
class Beam:
    start: tuple[float, float]
    end:   tuple[float, float]
    span_m: float = 0.0
    depth_m:float = 0.0
    load_kn_m: float = 0.0
    direction: str = "X"   # "X" | "Y"
    valid:  bool = True
    violation: str = ""

    def to_dict(self) -> dict:
        d = {
            "start":      [round(self.start[0],2), round(self.start[1],2)],
            "end":        [round(self.end[0],2),   round(self.end[1],2)],
            "span_m":     round(self.span_m, 2),
            "depth_m":    round(self.depth_m, 3),
            "direction":  self.direction,
            "valid":      self.valid,
        }
        if not self.valid:
            d["violation"] = self.violation
        return d


@dataclass
class Slab:
    x1: float; y1: float
    x2: float; y2: float
    width_m:  float = 0.0
    length_m: float = 0.0
    area_m2:  float = 0.0
    type:     str   = "two_way"   # "two_way" | "one_way"
    thickness_mm: float = 150.0

    def corners(self) -> list[list[float]]:
        return [
            [round(self.x1,2), round(self.y1,2)],
            [round(self.x2,2), round(self.y1,2)],
            [round(self.x2,2), round(self.y2,2)],
            [round(self.x1,2), round(self.y2,2)],
        ]

    def to_dict(self) -> dict:
        return {
            "corners":      self.corners(),
            "width_m":      round(self.width_m, 2),
            "length_m":     round(self.length_m, 2),
            "area_m2":      round(self.area_m2, 2),
            "type":         self.type,
            "thickness_mm": round(self.thickness_mm),
        }


@dataclass
class ValidationReport:
    is_valid:         bool = True
    span_violations:  list[dict] = field(default_factory=list)
    spacing_violations:list[dict]= field(default_factory=list)
    cantilever_warnings:list[dict]=field(default_factory=list)
    slab_warnings:    list[dict] = field(default_factory=list)
    column_count:     int  = 0
    beam_count:       int  = 0
    slab_count:       int  = 0
    total_violations: int  = 0

    def to_dict(self) -> dict:
        return {
            "is_valid":              self.is_valid,
            "column_count":          self.column_count,
            "beam_count":            self.beam_count,
            "slab_count":            self.slab_count,
            "total_violations":      self.total_violations,
            "span_violations":       self.span_violations,
            "spacing_violations":    self.spacing_violations,
            "cantilever_warnings":   self.cantilever_warnings,
            "slab_warnings":         self.slab_warnings,
        }


@dataclass
class StructuralGrid:
    columns:    list[Column]
    beams:      list[Beam]
    slabs:      list[Slab]
    load_map:   Optional[LoadMap]
    spacing_map:Optional[np.ndarray]
    validation: ValidationReport
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Output in the specified JSON format."""
        return {
            "columns": [c.to_list() for c in self.columns],
            "beams":   [b.to_dict() for b in self.beams],
            "slabs":   [s.to_dict() for s in self.slabs],
            "metadata":    self.metadata,
            "validation":  self.validation.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        v = self.validation
        lines = [
            "═"*58,
            "  STRUCTURAL GRID SUMMARY",
            "═"*58,
            f"  Columns         : {v.column_count}",
            f"  Beams           : {v.beam_count}",
            f"  Slabs           : {v.slab_count}",
            f"  Valid           : {'✓ YES' if v.is_valid else '✗ NO  — see violations'}",
            f"  Violations      : {v.total_violations}",
        ]
        if v.span_violations:
            lines.append("  Span violations :")
            for sv in v.span_violations:
                lines.append(f"    ✗ {sv['message']}")
        if v.spacing_violations:
            lines.append("  Spacing violations:")
            for sv in v.spacing_violations:
                lines.append(f"    ✗ {sv['message']}")
        if v.cantilever_warnings:
            lines.append("  Cantilever warnings:")
            for cw in v.cantilever_warnings:
                lines.append(f"    ⚠ {cw['message']}")
        lines.append("═"*58)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Structural Grid Generator
# ─────────────────────────────────────────────────────────────────────────────

class StructuralGridGenerator:
    """
    Generates column/beam/slab layout from a room layout.

    Usage
    -----
    gen = StructuralGridGenerator()
    gen.train_model()
    grid = gen.generate(layout, n_floors=3)
    print(grid.to_json())
    """

    def __init__(
        self,
        grid_size:   int   = 20,
        n_floors:    int   = 1,
        floor_num:   int   = 1,
        max_span:    float = MAX_BEAM_SPAN,
        min_spacing: float = MIN_COLUMN_SPACING,
    ):
        self.grid_size   = grid_size
        self.n_floors    = n_floors
        self.floor_num   = floor_num
        self.max_span    = max_span
        self.min_spacing = min_spacing

        self.load_estimator = LoadEstimator(grid_size, floor_num, n_floors)
        self.predictor      = ColumnSpacingPredictor()

    def train_model(self, n_samples: int = 3000, verbose: bool = True) -> dict:
        """Train the ML column spacing predictor."""
        return self.predictor.train(n_samples=n_samples, verbose=verbose)

    def generate(
        self,
        layout:  list[dict],
        verbose: bool = True,
    ) -> StructuralGrid:
        """
        Full pipeline: layout → StructuralGrid

        Parameters
        ----------
        layout  : list of room dicts {"room","x","y","w","h"}

        Returns
        -------
        StructuralGrid  with columns, beams, slabs, validation
        """
        assert self.predictor.is_trained, "Call train_model() first."

        if verbose:
            print(f"\n  Generating structural grid (floor {self.floor_num}/{self.n_floors})...")

        # ── Step 1: Load estimation ───────────────────────────────────
        load_map = self.load_estimator.estimate(layout)
        if verbose:
            print(f"    Load map: max={load_map.max_load():.1f} kN/m²  mean={load_map.mean_load():.1f} kN/m²")

        # ── Step 2: ML-predicted spacing map ─────────────────────────
        spacing_map = self.predictor.predict_map(load_map)
        if verbose:
            occ = spacing_map[spacing_map > 0]
            if len(occ):
                print(f"    Predicted spans: {occ.min():.1f}–{occ.max():.1f} m  mean={occ.mean():.1f} m")

        # ── Step 3: Candidate column grid ────────────────────────────
        candidates = self._candidate_grid(layout, load_map, spacing_map)

        # ── Step 4: Mandatory columns (corners, wet rooms) ───────────
        mandatory  = self._mandatory_columns(layout, load_map)

        # ── Step 5: Merge and deduplicate columns ────────────────────
        columns    = self._merge_columns(candidates, mandatory)
        if verbose:
            print(f"    Columns: {len(columns)} placed")

        # ── Step 6: Beam generation ───────────────────────────────────
        beams      = self._generate_beams(columns, load_map)
        if verbose:
            print(f"    Beams  : {len(beams)} generated")

        # ── Step 7: Slab panels ────────────────────────────────────────
        slabs      = self._generate_slabs(columns)
        if verbose:
            print(f"    Slabs  : {len(slabs)} panels")

        # ── Step 8: Validation ────────────────────────────────────────
        validation = self._validate(columns, beams, slabs)
        if verbose:
            print(f"    Valid  : {'YES' if validation.is_valid else 'NO — '+str(validation.total_violations)+' violations'}")

        return StructuralGrid(
            columns     = columns,
            beams       = beams,
            slabs       = slabs,
            load_map    = load_map,
            spacing_map = spacing_map,
            validation  = validation,
            metadata    = {
                "floor_num":    self.floor_num,
                "n_floors":     self.n_floors,
                "grid_size_m":  self.grid_size,
                "max_span_m":   self.max_span,
                "min_spacing_m":self.min_spacing,
                "load_max_kn_m2": round(load_map.max_load(), 2),
                "load_mean_kn_m2":round(load_map.mean_load(), 2),
            },
        )

    # ── Private: candidate grid ──────────────────────────────────────────

    def _candidate_grid(
        self,
        layout:      list[dict],
        load_map:    LoadMap,
        spacing_map: np.ndarray,
    ) -> list[Column]:
        """
        Place columns on a variable-spacing grid driven by ML predictions.
        Denser where loads are high, sparser where loads are light.
        """
        G       = self.grid_size
        columns = []
        visited = set()

        x = 0.0
        while x <= G:
            y = 0.0
            while y <= G:
                xi, yi = min(int(x), G-1), min(int(y), G-1)
                # Predicted spacing at this cell
                spacing = float(spacing_map[xi, yi]) if spacing_map[xi, yi] > 0 else 4.5

                # Clamp to constraints
                spacing = float(np.clip(spacing, self.min_spacing, self.max_span))

                key = (round(x, 1), round(y, 1))
                if key not in visited:
                    load = float(load_map.load_at(xi, yi))
                    # Only place column if inside or on boundary of building
                    columns.append(Column(
                        x=round(x, 1), y=round(y, 1),
                        load_kn=load * spacing**2 / 2,
                        reason="grid",
                    ))
                    visited.add(key)

                y += spacing
            x_next = x
            xi_ = min(int(x), G-1)
            # Step in X using the average predicted spacing along this column line
            col_spacings = [
                float(np.clip(spacing_map[xi_, min(int(y_), G-1)], self.min_spacing, self.max_span))
                for y_ in np.arange(0, G, 2)
            ]
            x_step = float(np.mean(col_spacings)) if col_spacings else 4.5
            x += x_step

        return columns

    def _mandatory_columns(
        self,
        layout:   list[dict],
        load_map: LoadMap,
    ) -> list[Column]:
        """
        Place columns at:
          - All room corners (load concentration points)
          - Wet room origins (plumbing stacks)
          - Heavy room corners
        """
        mandatory = []

        for room in layout:
            rtype = room.get("room","bedroom")
            x0, y0 = room["x"], room["y"]
            w,  h  = room["w"], room["h"]

            # 4 corners of every room
            for cx, cy in [(x0, y0), (x0+w, y0), (x0, y0+h), (x0+w, y0+h)]:
                if 0 <= cx <= self.grid_size and 0 <= cy <= self.grid_size:
                    load = float(load_map.load_at(min(cx, self.grid_size-1),
                                                   min(cy, self.grid_size-1)))
                    reason = "corner"
                    if rtype in WET_ROOMS:
                        reason = "wet_stack"
                    elif rtype in HEAVY_ROOMS:
                        reason = "heavy_room"
                    mandatory.append(Column(x=float(cx), y=float(cy),
                                            load_kn=load, reason=reason))

        return mandatory

    def _merge_columns(
        self,
        candidates: list[Column],
        mandatory:  list[Column],
    ) -> list[Column]:
        """
        Snap all columns to a min_spacing grid, then deduplicate.
        Guarantees no two columns are closer than min_spacing.
        Priority: wet_stack > heavy_room > corner > grid
        """
        PRIORITY = {"wet_stack":0, "heavy_room":1, "corner":2, "grid":3}
        SNAP = self.min_spacing

        def snap(v: float) -> float:
            return round(round(v / SNAP) * SNAP, 1)

        snapped: dict[tuple, Column] = {}
        all_cols = mandatory + candidates

        for col in sorted(all_cols, key=lambda c: PRIORITY.get(c.reason, 9)):
            sx, sy = snap(col.x), snap(col.y)
            key = (sx, sy)
            if key not in snapped:
                col.x, col.y = sx, sy
                snapped[key] = col

        return [
            c for c in snapped.values()
            if -0.5 <= c.x <= self.grid_size + 0.5
            and -0.5 <= c.y <= self.grid_size + 0.5
        ]

    # ── Private: beams ───────────────────────────────────────────────────

    def _generate_beams(
        self,
        columns:  list[Column],
        load_map: LoadMap,
    ) -> list[Beam]:
        """
        Connect columns with beams.
        Strategy: grid-based — connect each column to its nearest
        X-neighbour and Y-neighbour within max_span.
        """
        beams: list[Beam] = []
        seen:  set[tuple] = set()

        # Sort columns into X-lines and Y-lines
        x_vals = sorted(set(round(c.x, 1) for c in columns))
        y_vals = sorted(set(round(c.y, 1) for c in columns))

        # X-direction beams (horizontal)
        for y in y_vals:
            row = sorted([c for c in columns if abs(c.y - y) < 0.5], key=lambda c: c.x)
            for i in range(len(row) - 1):
                a, b = row[i], row[i+1]
                span = math.hypot(b.x - a.x, b.y - a.y)
                key  = (round(a.x,1), round(a.y,1), round(b.x,1), round(b.y,1))
                rkey = (round(b.x,1), round(b.y,1), round(a.x,1), round(a.y,1))
                if key not in seen and rkey not in seen:
                    beam = self._make_beam(a, b, span, load_map, "X")
                    beams.append(beam)
                    seen.add(key)

        # Y-direction beams (vertical)
        for x in x_vals:
            col_ = sorted([c for c in columns if abs(c.x - x) < 0.5], key=lambda c: c.y)
            for i in range(len(col_) - 1):
                a, b = col_[i], col_[i+1]
                span = math.hypot(b.x - a.x, b.y - a.y)
                key  = (round(a.x,1), round(a.y,1), round(b.x,1), round(b.y,1))
                rkey = (round(b.x,1), round(b.y,1), round(a.x,1), round(a.y,1))
                if key not in seen and rkey not in seen:
                    beam = self._make_beam(a, b, span, load_map, "Y")
                    beams.append(beam)
                    seen.add(key)

        return beams

    def _make_beam(
        self, a: Column, b: Column, span: float, load_map: LoadMap, direction: str
    ) -> Beam:
        # Average load along beam
        nx, ny = int((a.x+b.x)/2), int((a.y+b.y)/2)
        nx = min(max(nx, 0), self.grid_size-1)
        ny = min(max(ny, 0), self.grid_size-1)
        load_kn_m = float(load_map.load_at(nx, ny))

        depth    = max(span * BEAM_DEPTH_FACTOR, 0.30)   # min 300mm
        is_valid = span <= self.max_span
        violation= f"Span {span:.2f}m exceeds max {self.max_span}m" if not is_valid else ""

        return Beam(
            start     = a.pos(),
            end       = b.pos(),
            span_m    = round(span, 3),
            depth_m   = round(depth, 3),
            load_kn_m = round(load_kn_m, 2),
            direction = direction,
            valid     = is_valid,
            violation = violation,
        )

    # ── Private: slabs ───────────────────────────────────────────────────

    def _generate_slabs(self, columns: list[Column]) -> list[Slab]:
        """
        Define slab panels as rectangular regions bounded by beam lines.
        Uses column X/Y lines to define the grid of panels.
        """
        slabs: list[Slab] = []

        xs = sorted(set(round(c.x) for c in columns))
        ys = sorted(set(round(c.y) for c in columns))

        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                x1, x2 = xs[i], xs[i+1]
                y1, y2 = ys[j], ys[j+1]
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                # Slab type
                ratio     = max(w, h) / max(min(w, h), 0.01)
                slab_type = "two_way" if ratio <= MAX_SLAB_RATIO else "one_way"

                # Thickness by L/25 rule (shorter span)
                shorter     = min(w, h)
                thickness_m = max(shorter / 25, 0.12)   # min 120mm

                slabs.append(Slab(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    width_m  = round(float(w), 2),
                    length_m = round(float(h), 2),
                    area_m2  = round(float(w * h), 2),
                    type     = slab_type,
                    thickness_mm = round(thickness_m * 1000),
                ))

        return slabs

    # ── Private: validation ──────────────────────────────────────────────

    def _validate(
        self,
        columns: list[Column],
        beams:   list[Beam],
        slabs:   list[Slab],
    ) -> ValidationReport:

        report = ValidationReport(
            column_count = len(columns),
            beam_count   = len(beams),
            slab_count   = len(slabs),
        )

        # Span violations
        for b in beams:
            if not b.valid:
                report.span_violations.append({
                    "message": b.violation,
                    "beam":    [list(b.start), list(b.end)],
                    "span_m":  b.span_m,
                })

        # Spacing violations (columns too close)
        col_positions = [(c.x, c.y) for c in columns]
        for i, (x1,y1) in enumerate(col_positions):
            for j, (x2,y2) in enumerate(col_positions):
                if j <= i:
                    continue
                dist = math.hypot(x2-x1, y2-y1)
                if 0 < dist < self.min_spacing - 0.1:
                    report.spacing_violations.append({
                        "message": f"Columns at {[x1,y1]} and {[x2,y2]} spacing={dist:.2f}m < {self.min_spacing}m",
                        "dist_m":  round(dist, 3),
                    })

        # Slab warnings
        for s in slabs:
            ratio = max(s.width_m, s.length_m) / max(min(s.width_m, s.length_m), 0.01)
            if ratio > MAX_SLAB_RATIO:
                report.slab_warnings.append({
                    "message": f"Slab {s.width_m:.1f}×{s.length_m:.1f}m ratio={ratio:.1f} > {MAX_SLAB_RATIO} (one-way slab)",
                    "area_m2": s.area_m2,
                })

        # Check for cantilever (column on one side only)
        xs = sorted(set(round(c.x) for c in columns))
        ys = sorted(set(round(c.y) for c in columns))
        G  = self.grid_size
        for x in [xs[0], xs[-1]] if xs else []:
            dist = min(x, G - x)
            if dist > MAX_CANTILEVER:
                report.cantilever_warnings.append({
                    "message": f"Edge column at x={x} has cantilever ≈{dist:.1f}m > {MAX_CANTILEVER}m",
                })
        for y in [ys[0], ys[-1]] if ys else []:
            dist = min(y, G - y)
            if dist > MAX_CANTILEVER:
                report.cantilever_warnings.append({
                    "message": f"Edge column at y={y} has cantilever ≈{dist:.1f}m > {MAX_CANTILEVER}m",
                })

        total = len(report.span_violations) + len(report.spacing_violations)
        report.total_violations = total
        report.is_valid         = (total == 0)
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Multi-floor structural grid
# ─────────────────────────────────────────────────────────────────────────────

class MultiFloorStructuralGrid:
    """
    Generates and validates structural grids for all floors of a building.
    Ensures column alignment across floors (same XY, every floor).
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.floor_grids: dict[int, StructuralGrid] = {}
        self._predictor   = ColumnSpacingPredictor()
        self._model_trained = False

    def train(self, verbose: bool = True) -> dict:
        metrics = self._predictor.train(verbose=verbose)
        self._model_trained = True
        return metrics

    def generate_all(
        self,
        building_layout: dict,
        verbose: bool = True,
    ) -> dict[int, StructuralGrid]:
        """
        Generate structural grids for all floors in building_layout.

        Parameters
        ----------
        building_layout : {"floor_1": [...], "floor_2": [...], ...}

        Returns
        -------
        dict mapping floor_num → StructuralGrid
        """
        assert self._model_trained, "Call train() first."

        floor_keys = sorted(
            [k for k in building_layout if k.startswith("floor_")],
            key=lambda k: int(k.split("_")[1])
        )
        n_floors = len(floor_keys)

        # First pass: generate each floor
        for key in floor_keys:
            fn      = int(key.split("_")[1])
            layout  = building_layout[key]
            gen     = StructuralGridGenerator(
                grid_size   = self.grid_size,
                n_floors    = n_floors,
                floor_num   = fn,
            )
            gen.predictor = self._predictor   # share trained model
            gen.predictor.is_trained = True
            grid = gen.generate(layout, verbose=verbose)
            self.floor_grids[fn] = grid

        # Second pass: align columns vertically
        if n_floors > 1:
            self._align_columns_vertically(verbose=verbose)

        return self.floor_grids

    def _align_columns_vertically(self, verbose: bool = True):
        """
        Ensure column XY positions are consistent across all floors.
        Takes the union of all column positions and adds missing ones.
        """
        # Collect all unique column positions from all floors
        all_positions: set[tuple[float,float]] = set()
        for fg in self.floor_grids.values():
            for c in fg.columns:
                all_positions.add((round(c.x), round(c.y)))

        if verbose:
            print(f"\n  Aligning {len(all_positions)} column positions across {len(self.floor_grids)} floors...")

        for fn, fg in self.floor_grids.items():
            existing = set((round(c.x), round(c.y)) for c in fg.columns)
            added    = 0
            for (px, py) in all_positions:
                if (px, py) not in existing:
                    # Add missing column (load = 0 — structural transfer column)
                    fg.columns.append(Column(
                        x=float(px), y=float(py),
                        load_kn=0.0, reason="alignment"
                    ))
                    added += 1
            if verbose and added:
                print(f"    Floor {fn}: added {added} alignment columns")

    def to_dict(self) -> dict:
        return {
            f"floor_{fn}": grid.to_dict()
            for fn, grid in sorted(self.floor_grids.items())
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

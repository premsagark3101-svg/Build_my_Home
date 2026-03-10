"""
constraint_validator.py
=======================
Rule engine that validates and enriches building layout constraints
extracted from the NLP module (building_nlp.py).

Responsibilities:
  - Calculate minimum room sizes (IRC / NBC residential standards)
  - Estimate recommended room sizes relative to plot
  - Validate total floor area vs plot capacity
  - Generate adjacency, separation & boundary placement constraints
  - Produce a solver-ready constraint package

Usage:
    from constraint_validator import ConstraintValidator
    validator = ConstraintValidator()
    result    = validator.validate(nlp_json_dict)
    print(result.to_json())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Building Standards  — [width_m, length_m]
# IRC 2021 / NBC India / common residential norms
# ─────────────────────────────────────────────────────────────────────────────

MINIMUM_SIZES: dict[str, list[float]] = {
    "bedroom":        [3.0, 3.0],   #  9.0 m²
    "master_bedroom": [3.5, 4.0],   # 14.0 m²
    "bathroom":       [1.5, 2.0],   #  3.0 m²
    "kitchen":        [2.4, 3.0],   #  7.2 m²
    "living_room":    [3.6, 4.5],   # 16.2 m²
    "dining_room":    [3.0, 3.6],   # 10.8 m²
    "study":          [2.4, 3.0],   #  7.2 m²
    "home_office":    [2.4, 3.0],   #  7.2 m²
    "guest_room":     [2.7, 3.0],   #  8.1 m²
    "garage":         [3.0, 5.5],   # 16.5 m²  (single car)
    "balcony":        [1.2, 2.0],   #  2.4 m²
    "terrace":        [2.0, 3.0],   #  6.0 m²
    "storage":        [1.5, 1.5],   #  2.25 m²
    "laundry":        [1.8, 2.0],   #  3.6 m²
    "gym":            [3.0, 4.0],   # 12.0 m²
    "utility_room":   [1.5, 2.0],   #  3.0 m²
}

RECOMMENDED_SIZES: dict[str, list[float]] = {
    "bedroom":        [3.5, 4.0],
    "master_bedroom": [4.0, 5.0],
    "bathroom":       [2.0, 2.5],
    "kitchen":        [3.0, 4.0],
    "living_room":    [4.5, 6.0],
    "dining_room":    [3.5, 4.5],
    "study":          [3.0, 3.5],
    "home_office":    [3.0, 3.5],
    "guest_room":     [3.0, 3.5],
    "garage":         [3.5, 6.0],
    "balcony":        [1.5, 3.0],
    "terrace":        [3.0, 4.0],
    "storage":        [2.0, 2.0],
    "laundry":        [2.0, 2.5],
    "gym":            [4.0, 5.0],
    "utility_room":   [2.0, 2.5],
}

PARKING_SLOT_SIZE  = [2.5, 5.0]   # m per bay
WALL_OVERHEAD      = 0.12          # 12 % for walls / structure
SETBACK_M          = 1.5           # minimum plot boundary setback
FLOOR_HEIGHT_M     = 3.0           # metres per storey
COVERAGE_RATIO     = 0.60          # buildable fraction of plot area per floor

# ─────────────────────────────────────────────────────────────────────────────
# Placement rules
# ─────────────────────────────────────────────────────────────────────────────

ADJACENCY_RULES = [
    {"rooms": ["bedroom",      "bathroom"],    "priority": "HIGH",   "reason": "Bedrooms require nearby bathroom access"},
    {"rooms": ["master_bedroom","bathroom"],   "priority": "HIGH",   "reason": "Master bedroom requires en-suite or adjacent bath"},
    {"rooms": ["kitchen",      "dining_room"], "priority": "HIGH",   "reason": "Kitchen and dining should be directly connected"},
    {"rooms": ["living_room",  "dining_room"], "priority": "MEDIUM", "reason": "Living and dining rooms share open-plan flow"},
    {"rooms": ["kitchen",      "living_room"], "priority": "MEDIUM", "reason": "Kitchen near living for open-plan connectivity"},
    {"rooms": ["parking",      "entrance"],    "priority": "HIGH",   "reason": "Parking must open onto the building entrance"},
    {"rooms": ["laundry",      "bathroom"],    "priority": "MEDIUM", "reason": "Laundry room near plumbing core saves pipe runs"},
    {"rooms": ["gym",          "bathroom"],    "priority": "MEDIUM", "reason": "Gym needs adjacent bathroom / changing area"},
    {"rooms": ["terrace",      "living_room"], "priority": "MEDIUM", "reason": "Terrace extends living space outdoors"},
    {"rooms": ["balcony",      "bedroom"],     "priority": "LOW",    "reason": "Balcony accessible from bedroom"},
    {"rooms": ["utility_room", "kitchen"],     "priority": "MEDIUM", "reason": "Utility room near kitchen for shared services"},
    {"rooms": ["storage",      "garage"],      "priority": "LOW",    "reason": "Storage conveniently co-located with garage"},
    {"rooms": ["study",        "bedroom"],     "priority": "LOW",    "reason": "Study near bedroom wing for quiet environment"},
]

SEPARATION_RULES = [
    {"rooms": ["bedroom",  "kitchen"],  "priority": "HIGH",   "reason": "Odour and noise must not penetrate sleeping areas"},
    {"rooms": ["bedroom",  "garage"],   "priority": "MEDIUM", "reason": "Noise and fumes must be isolated from bedrooms"},
    {"rooms": ["gym",      "bedroom"],  "priority": "LOW",    "reason": "Equipment noise should not disturb sleeping areas"},
    {"rooms": ["kitchen",  "bathroom"], "priority": "MEDIUM", "reason": "Cross-contamination risk between cooking and sanitary areas"},
]

FLOOR_HINTS: dict[str, str] = {
    "parking":        "GROUND",
    "garage":         "GROUND",
    "kitchen":        "GROUND",
    "living_room":    "GROUND",
    "dining_room":    "GROUND",
    "gym":            "GROUND_OR_BASEMENT",
    "utility_room":   "GROUND_OR_BASEMENT",
    "storage":        "ANY",
    "laundry":        "ANY",
    "bathroom":       "ANY",
    "home_office":    "ANY",
    "bedroom":        "UPPER",
    "master_bedroom": "UPPER",
    "guest_room":     "UPPER",
    "study":          "UPPER",
    "balcony":        "UPPER",
    "terrace":        "UPPER_OR_ROOF",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomSpec:
    name:             str
    count:            int
    min_size:         list[float]
    recommended_size: list[float]
    min_area_m2:      float
    recommended_area_m2: float
    total_min_area_m2:   float    # min_area × count
    floor_hint:       str
    notes:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "count":                 self.count,
            "min_size":              self.min_size,
            "recommended_size":      self.recommended_size,
            "min_area_m2":           round(self.min_area_m2, 2),
            "recommended_area_m2":   round(self.recommended_area_m2, 2),
            "total_min_area_m2":     round(self.total_min_area_m2, 2),
            "floor_hint":            self.floor_hint,
        }
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class AreaSummary:
    plot_area_m2:             float
    buildable_per_floor_m2:   float
    total_buildable_m2:       float
    min_required_m2:          float
    recommended_m2:           float
    utilisation_pct:          float
    headroom_m2:              float
    is_feasible:              bool
    verdict:                  str
    feasibility_notes:        list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plot_area_m2":           round(self.plot_area_m2, 2),
            "buildable_per_floor_m2": round(self.buildable_per_floor_m2, 2),
            "total_buildable_m2":     round(self.total_buildable_m2, 2),
            "min_required_m2":        round(self.min_required_m2, 2),
            "recommended_m2":         round(self.recommended_m2, 2),
            "utilisation_pct":        round(self.utilisation_pct, 1),
            "headroom_m2":            round(self.headroom_m2, 2),
            "is_feasible":            self.is_feasible,
            "verdict":                self.verdict,
            "feasibility_notes":      self.feasibility_notes,
        }


@dataclass
class ValidationResult:
    # ── Spec-required flat fields ──────────────────────────────────────
    plot_boundary: list[float]
    floors:        int

    # ── Detailed output ────────────────────────────────────────────────
    room_specs:             dict[str, RoomSpec]
    parking_spec:           Optional[dict]
    area_analysis:          AreaSummary
    adjacency_constraints:  list[dict]
    separation_constraints: list[dict]
    boundary_constraints:   dict
    floor_hints:            dict[str, str]

    # ── Status ─────────────────────────────────────────────────────────
    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)
    is_valid: bool = True

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # Flat min-size keys (matches spec example output)
        size_map = {
            f"{name}_min_size": spec.min_size
            for name, spec in self.room_specs.items()
        }

        d: dict = {
            **size_map,
            "plot_boundary": self.plot_boundary,
            "floors":        self.floors,
            "room_specs":    {k: v.to_dict() for k, v in self.room_specs.items()},
            "area_analysis": self.area_analysis.to_dict(),
            "placement_constraints": {
                "adjacency":   self.adjacency_constraints,
                "separation":  self.separation_constraints,
                "boundary":    self.boundary_constraints,
                "floor_hints": self.floor_hints,
            },
        }
        if self.parking_spec:
            d["parking_spec"] = self.parking_spec
        if self.warnings:
            d["warnings"] = self.warnings
        if self.errors:
            d["errors"] = self.errors
        d["is_valid"] = self.is_valid
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable one-page summary."""
        lines = [
            "═" * 58,
            "  BUILDING CONSTRAINT VALIDATION SUMMARY",
            "═" * 58,
            f"  Plot          : {self.plot_boundary[0]} × {self.plot_boundary[1]} m",
            f"  Floors        : {self.floors}",
            f"  Valid         : {'✓ YES' if self.is_valid else '✗ NO'}",
            "",
            "  ROOM MINIMUM SIZES",
            "  " + "─" * 40,
        ]
        for name, spec in self.room_specs.items():
            w, l = spec.min_size
            lines.append(
                f"  {name:<20} {spec.count}×  [{w} × {l} m]  "
                f"= {spec.total_min_area_m2:.1f} m²  |  floor: {spec.floor_hint}"
            )

        a = self.area_analysis
        lines += [
            "",
            "  AREA ANALYSIS",
            "  " + "─" * 40,
            f"  Plot area          : {a.plot_area_m2:.1f} m²",
            f"  Total buildable    : {a.total_buildable_m2:.1f} m²",
            f"  Min required       : {a.min_required_m2:.1f} m²",
            f"  Headroom           : {a.headroom_m2:.1f} m²",
            f"  Utilisation        : {a.utilisation_pct:.1f}%",
            f"  Verdict            : {a.verdict}",
        ]

        if self.adjacency_constraints:
            lines += ["", "  ADJACENCY RULES (must be near)"]
            lines.append("  " + "─" * 40)
            for r in self.adjacency_constraints:
                a_r, b_r = r["rooms"]
                lines.append(f"  [{r['priority']:<6}] {a_r} ↔ {b_r}")

        if self.separation_constraints:
            lines += ["", "  SEPARATION RULES (must NOT be adjacent)"]
            lines.append("  " + "─" * 40)
            for r in self.separation_constraints:
                a_r, b_r = r["rooms"]
                lines.append(f"  [{r['priority']:<6}] {a_r} ✗ {b_r}")

        if self.warnings:
            lines += ["", "  WARNINGS"]
            lines.append("  " + "─" * 40)
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        if self.errors:
            lines += ["", "  ERRORS"]
            lines.append("  " + "─" * 40)
            for e in self.errors:
                lines.append(f"  ✗  {e}")

        lines.append("═" * 58)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rule Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintValidator:
    """
    Validates NLP-extracted building JSON and produces a structured,
    solver-ready constraint package.

    Rules applied
    ─────────────
    R1  Bedrooms must be near bathrooms
    R2  Kitchen near living room
    R3  Parking near entrance
    R4  All rooms must stay within plot boundary (setback enforced)
    R5  Bedroom / kitchen separation
    R6  Plot capacity vs. total min required area
    R7  Floor assignment hints per room type
    """

    def validate(self, nlp_json: dict) -> ValidationResult:
        """
        Parameters
        ----------
        nlp_json : dict   Output of BuildingNLPParser.parse().to_dict()

        Returns
        -------
        ValidationResult  Fully populated constraint + validation object
        """
        errors:   list[str] = []
        warnings: list[str] = list(nlp_json.get("warnings", []))

        # ── Parse inputs ─────────────────────────────────────────────────
        plot_w   = nlp_json.get("plot_width")
        plot_l   = nlp_json.get("plot_length")
        floors   = int(nlp_json.get("floors", 1))
        rooms    = dict(nlp_json.get("rooms", {}))
        parking  = bool(nlp_json.get("parking", False))
        garden   = bool(nlp_json.get("garden",  False))
        pool     = bool(nlp_json.get("pool",     False))
        basement = bool(nlp_json.get("basement", False))

        logger.info("Validating: %s", json.dumps(nlp_json, indent=None))

        # ── Validate plot dimensions (R4) ─────────────────────────────────
        if plot_w is None or plot_l is None:
            errors.append("Plot dimensions are missing — area feasibility cannot be assessed.")
            plot_boundary = [0.0, 0.0]
            plot_area     = 0.0
        else:
            plot_w, plot_l = float(plot_w), float(plot_l)
            if plot_w <= 0 or plot_l <= 0:
                errors.append(f"Invalid plot dimensions: {plot_w} × {plot_l} m")
            plot_boundary = [plot_w, plot_l]
            plot_area     = plot_w * plot_l

        # ── Expand rooms (first bedroom → master_bedroom) ─────────────────
        expanded: dict[str, int] = {}
        for room, count in rooms.items():
            if room == "bedroom" and count > 0:
                expanded["master_bedroom"] = 1
                if count > 1:
                    expanded["bedroom"] = count - 1
            else:
                expanded[room] = count

        # ── Build room specs ──────────────────────────────────────────────
        room_specs:      dict[str, RoomSpec] = {}
        total_min_area   = 0.0
        total_rec_area   = 0.0

        for room, count in expanded.items():
            if count <= 0:
                continue
            spec = self._make_room_spec(room, count, floors, warnings)
            room_specs[room] = spec
            total_min_area  += spec.total_min_area_m2
            total_rec_area  += spec.recommended_area_m2 * count

        # Apply structural overhead
        total_min_area *= (1 + WALL_OVERHEAD)
        total_rec_area *= (1 + WALL_OVERHEAD)

        # ── Parking spec (R3) ─────────────────────────────────────────────
        parking_spec: Optional[dict] = None
        if parking:
            bays      = int(nlp_json.get("parking_bays", 1))
            bay_area  = PARKING_SLOT_SIZE[0] * PARKING_SLOT_SIZE[1]
            total_min_area += bay_area * bays
            total_rec_area += bay_area * bays * 1.2
            parking_spec = {
                "bays":               bays,
                "min_size_per_bay_m": PARKING_SLOT_SIZE,
                "total_area_m2":      round(bay_area * bays, 2),
                "floor_hint":         "GROUND",
                "placement_rule":     "MUST_BE_NEAR_ENTRANCE",
            }

        # ── Garden / pool / basement notes ────────────────────────────────
        if garden and plot_area:
            reserve = round(plot_area * 0.15, 1)
            warnings.append(
                f"Garden reserved: ~{reserve} m² (15% of plot) — reduces buildable footprint."
            )
        if pool:
            pool_area = 30.0
            total_min_area += pool_area
            warnings.append(f"Swimming pool adds ~{pool_area} m² to space requirements.")
        if basement:
            floors += 1   # basement counts as an extra buildable level
            warnings.append("Basement treated as an additional floor for area calculations.")

        # ── Area feasibility (R6) ─────────────────────────────────────────
        area_summary = self._assess_area(
            plot_area, floors, total_min_area, total_rec_area, warnings
        )

        # ── Placement constraints (R1, R2, R3) ────────────────────────────
        active = set(room_specs.keys())
        if parking:
            active.add("parking")

        adjacency  = self._adjacency(active)
        separation = self._separation(active)
        boundary   = self._boundary(plot_boundary, int(nlp_json.get("floors", 1)))
        hints      = self._floor_hints(active, int(nlp_json.get("floors", 1)))

        # ── Final validity ────────────────────────────────────────────────
        is_valid = (len(errors) == 0) and area_summary.is_feasible

        return ValidationResult(
            plot_boundary          = plot_boundary,
            floors                 = int(nlp_json.get("floors", 1)),
            room_specs             = room_specs,
            parking_spec           = parking_spec,
            area_analysis          = area_summary,
            adjacency_constraints  = adjacency,
            separation_constraints = separation,
            boundary_constraints   = boundary,
            floor_hints            = hints,
            warnings               = warnings,
            errors                 = errors,
            is_valid               = is_valid,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _make_room_spec(
        self, room: str, count: int, floors: int, warnings: list[str]
    ) -> RoomSpec:
        min_s = list(MINIMUM_SIZES.get(room,     [2.4, 3.0]))
        rec_s = list(RECOMMENDED_SIZES.get(room, [3.0, 3.5]))
        hint  = FLOOR_HINTS.get(room, "ANY")
        if hint == "UPPER" and floors == 1:
            hint = "GROUND"

        min_area = round(min_s[0] * min_s[1], 2)
        rec_area = round(rec_s[0] * rec_s[1], 2)
        notes    = []
        if count > 1:
            notes.append(f"Sizes are per unit; {count} units total.")

        return RoomSpec(
            name               = room,
            count              = count,
            min_size           = min_s,
            recommended_size   = rec_s,
            min_area_m2        = min_area,
            recommended_area_m2= rec_area,
            total_min_area_m2  = round(min_area * count, 2),
            floor_hint         = hint,
            notes              = notes,
        )

    def _assess_area(
        self,
        plot_area:   float,
        floors:      int,
        min_req:     float,
        rec:         float,
        warnings:    list[str],
    ) -> AreaSummary:

        if plot_area == 0:
            return AreaSummary(
                plot_area_m2=0, buildable_per_floor_m2=0,
                total_buildable_m2=0, min_required_m2=round(min_req,2),
                recommended_m2=round(rec,2), utilisation_pct=999,
                headroom_m2=0, is_feasible=False,
                verdict="UNKNOWN — plot dimensions missing",
                feasibility_notes=["Cannot assess without plot dimensions."],
            )

        per_floor   = plot_area * COVERAGE_RATIO
        total_build = per_floor * floors
        util        = (min_req / total_build * 100) if total_build else 999
        headroom    = total_build - min_req
        is_feasible = headroom >= 0
        notes: list[str] = []

        if not is_feasible:
            shortfall = abs(headroom)
            verdict   = f"INSUFFICIENT — {shortfall:.1f} m² short"
            notes.append(
                f"Need {min_req:.1f} m² but only {total_build:.1f} m² buildable "
                f"({floors} floor{'s' if floors>1 else ''} × {per_floor:.1f} m²/floor)."
            )
            notes.append("Options: add a floor, reduce rooms, or increase plot size.")
            warnings.append("⚠  Plot capacity INSUFFICIENT for the requested rooms.")
        elif util > 85:
            verdict = f"TIGHT — {util:.1f}% utilisation"
            notes.append(
                f"Only {headroom:.1f} m² headroom — layout will be compact. "
                "Consider adding a floor or reducing room count."
            )
        elif util > 65:
            verdict = f"COMFORTABLE — {util:.1f}% utilisation"
            notes.append(f"{headroom:.1f} m² of headroom for corridors and circulation.")
        else:
            verdict = f"SPACIOUS — {util:.1f}% utilisation"
            notes.append(f"Ample space: {headroom:.1f} m² headroom available.")

        return AreaSummary(
            plot_area_m2           = round(plot_area, 2),
            buildable_per_floor_m2 = round(per_floor, 2),
            total_buildable_m2     = round(total_build, 2),
            min_required_m2        = round(min_req, 2),
            recommended_m2         = round(rec, 2),
            utilisation_pct        = round(util, 1),
            headroom_m2            = round(headroom, 2),
            is_feasible            = is_feasible,
            verdict                = verdict,
            feasibility_notes      = notes,
        )

    def _adjacency(self, active: set[str]) -> list[dict]:
        out = []
        for rule in ADJACENCY_RULES:
            a, b = rule["rooms"]
            a_ok = a in active or f"master_{a}" in active
            b_ok = b in active or f"master_{b}" in active or b == "entrance"
            if a_ok and b_ok:
                out.append({
                    "constraint": "MUST_BE_ADJACENT",
                    "rooms":      rule["rooms"],
                    "priority":   rule["priority"],
                    "reason":     rule["reason"],
                })
        return out

    def _separation(self, active: set[str]) -> list[dict]:
        out = []
        for rule in SEPARATION_RULES:
            a, b = rule["rooms"]
            a_ok = a in active or f"master_{a}" in active
            b_ok = b in active or f"master_{b}" in active
            if a_ok and b_ok:
                out.append({
                    "constraint": "MUST_NOT_BE_ADJACENT",
                    "rooms":      rule["rooms"],
                    "priority":   rule["priority"],
                    "reason":     rule["reason"],
                })
        return out

    def _boundary(self, plot_boundary: list[float], floors: int) -> dict:
        w, l = (plot_boundary + [0, 0])[:2]
        uw = round(max(0.0, w - 2 * SETBACK_M), 2)
        ul = round(max(0.0, l - 2 * SETBACK_M), 2)
        return {
            "rule":                 "ALL_ROOMS_MUST_STAY_WITHIN_PLOT",
            "plot_width_m":         w,
            "plot_length_m":        l,
            "min_setback_m":        SETBACK_M,
            "usable_width_m":       uw,
            "usable_length_m":      ul,
            "max_building_height_m":round(floors * FLOOR_HEIGHT_M, 2),
            "constraints": [
                "No room may extend beyond plot boundary",
                f"Minimum {SETBACK_M} m setback from all plot edges",
                "Rooms may not span across plot corners",
                "Structural elements must remain within usable zone",
            ],
        }

    def _floor_hints(self, active: set[str], floors: int) -> dict[str, str]:
        hints = {}
        for room in active:
            hint = FLOOR_HINTS.get(room, "ANY")
            if hint == "UPPER" and floors == 1:
                hint = "GROUND"
            hints[room] = hint
        return hints


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    default_input = {
        "plot_width":  40,
        "plot_length": 60,
        "floors":      2,
        "rooms": {
            "bedroom":    3,
            "bathroom":   2,
            "kitchen":    1,
            "living_room":1,
        },
        "parking": True,
        "garden":  True,
    }

    if len(sys.argv) > 1:
        try:
            nlp_input = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            print(f"Invalid JSON argument: {e}")
            sys.exit(1)
    else:
        nlp_input = default_input

    validator = ConstraintValidator()
    result    = validator.validate(nlp_input)

    print(result.summary())
    print("\n── Full JSON output ──")
    print(result.to_json())

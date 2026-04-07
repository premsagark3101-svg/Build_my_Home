"""
cost_estimator.py
=================
ML-based construction cost estimation using GradientBoostingRegressor.

Three sub-models (one per cost component):
  - Material cost
  - Labor cost
  - MEP / specialist cost

Training data is generated from engineering cost norms
(RSMeans / CPWD Schedule of Rates India).

Output:
  {"material_cost": X, "labor_cost": Y, "total_cost": Z}
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Cost norms  (USD per m² — approximate residential, India-adjusted × 0.6)
# ─────────────────────────────────────────────────────────────────────────────

MAT_FOUNDATION_M2   = 80.0
MAT_STRUCTURAL_M2   = 120.0
MAT_MASONRY_M2      = 45.0
MAT_FINISHES_M2     = 60.0
MAT_MEP_M2          = 55.0
MAT_ROOFING_M2      = 40.0

LABOR_CIVIL_M2      = 35.0
LABOR_STRUCT_M2     = 50.0
LABOR_MEP_M2        = 40.0
LABOR_FINISHING_M2  = 30.0

ROOM_PREMIUM = {
    "bathroom":       1800,
    "kitchen":        2200,
    "master_bedroom": 500,
    "bedroom":        300,
    "living_room":    400,
    "dining_room":    300,
    "study":          250,
    "parking":        800,
    "gym":            600,
    "home_office":    300,
    "elevator":       35000,
    "staircase":      3000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    area_m2:    float,
    n_rooms:    int,
    n_floors:   int,
    n_bath:     int,
    n_kitchen:  int,
    n_parking:  int,
    has_elevator: bool,
    has_pool:   bool,
    n_columns:  int,
    n_beams:    int,
    plumb_len:  float,
    elec_len:   float,
) -> np.ndarray:
    return np.array([
        area_m2,
        n_rooms,
        n_floors,
        n_bath,
        n_kitchen,
        n_parking,
        int(has_elevator),
        int(has_pool),
        n_columns,
        n_beams,
        plumb_len,
        elec_len,
        area_m2 / max(n_floors, 1),          # per-floor area
        (n_bath + n_kitchen) / max(n_rooms,1),# wet room density
        n_columns / max(area_m2, 1) * 100,   # column density per 100m²
        math.log1p(area_m2),
        math.sqrt(area_m2),
        area_m2 * n_floors,                  # total built-up area
    ], dtype=np.float32)

FEATURE_NAMES = [
    "area_m2","n_rooms","n_floors","n_bath","n_kitchen","n_parking",
    "has_elevator","has_pool","n_columns","n_beams","plumb_len_m",
    "elec_len_m","area_per_floor","wet_room_density",
    "col_density","log_area","sqrt_area","total_bua",
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data
# ─────────────────────────────────────────────────────────────────────────────

def _first_principles_cost(feat: np.ndarray) -> tuple[float, float, float]:
    """
    Engineering cost formula from first principles.
    Returns (material_cost, labor_cost, mep_cost).
    """
    area       = feat[0]
    n_floors   = max(int(feat[2]), 1)
    n_bath     = int(feat[3])
    n_kitchen  = int(feat[4])
    n_parking  = int(feat[5])
    has_elev   = bool(feat[6])
    has_pool   = bool(feat[7])
    total_bua  = feat[17]

    mat = (
        MAT_FOUNDATION_M2  * area / n_floors +   # foundation on ground area
        MAT_STRUCTURAL_M2  * total_bua +
        MAT_MASONRY_M2     * total_bua +
        MAT_FINISHES_M2    * total_bua +
        MAT_ROOFING_M2     * area
    )

    lab = (
        LABOR_CIVIL_M2     * area / n_floors +
        LABOR_STRUCT_M2    * total_bua +
        LABOR_FINISHING_M2 * total_bua
    )

    mep = (
        MAT_MEP_M2  * total_bua +
        LABOR_MEP_M2 * total_bua +
        n_bath    * ROOM_PREMIUM["bathroom"] +
        n_kitchen * ROOM_PREMIUM["kitchen"]  +
        n_parking * ROOM_PREMIUM["parking"]  +
        (ROOM_PREMIUM["elevator"] if has_elev else 0) +
        (40000 if has_pool else 0)
    )

    # Complexity premium for multi-floor
    complexity = 1.0 + (n_floors - 1) * 0.08
    mat *= complexity
    lab *= complexity

    return float(mat), float(lab), float(mep)


def generate_training_data(n_samples: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y_mat, y_lab, y_mep = [], [], [], []

    for _ in range(n_samples):
        n_floors  = int(rng.integers(1, 5))
        area      = rng.uniform(60, 600) * n_floors
        n_rooms   = int(rng.integers(3, 20))
        n_bath    = int(rng.integers(1, 6))
        n_kitchen = int(rng.integers(1, 3))
        n_park    = int(rng.integers(0, 3))
        has_elev  = n_floors >= 3 and rng.random() > 0.4
        has_pool  = rng.random() > 0.85
        n_col     = int(rng.integers(8, 50))
        n_beam    = int(rng.integers(10, 80))
        p_len     = rng.uniform(10, 80)
        e_len     = rng.uniform(20, 120)

        feat = extract_features(area, n_rooms, n_floors, n_bath, n_kitchen,
                                n_park, has_elev, has_pool, n_col, n_beam, p_len, e_len)
        mat, lab, mep = _first_principles_cost(feat)

        # Add noise
        mat *= rng.uniform(0.90, 1.10)
        lab *= rng.uniform(0.88, 1.12)
        mep *= rng.uniform(0.85, 1.15)

        X.append(feat); y_mat.append(mat); y_lab.append(lab); y_mep.append(mep)

    return np.array(X, dtype=np.float32), np.array(y_mat), np.array(y_lab), np.array(y_mep)


# ─────────────────────────────────────────────────────────────────────────────
# Cost Estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CostEstimate:
    material_cost: float
    labor_cost:    float
    mep_cost:      float
    total_cost:    float
    contingency:   float   # 10%
    grand_total:   float
    cost_per_m2:   float
    area_m2:       float
    breakdown:     dict

    def to_dict(self) -> dict:
        return {
            "material_cost": round(self.material_cost),
            "labor_cost":    round(self.labor_cost),
            "mep_cost":      round(self.mep_cost),
            "total_cost":    round(self.total_cost),
            "contingency_10pct": round(self.contingency),
            "grand_total":   round(self.grand_total),
            "cost_per_m2":   round(self.cost_per_m2),
            "breakdown":     {k: round(v) for k, v in self.breakdown.items()},
        }


class CostEstimationModel:
    """
    GradientBoostingRegressor for building cost estimation.
    Three separate models for material, labor, and MEP costs.
    """

    def __init__(self):
        self._mat_model: Optional[Pipeline] = None
        self._lab_model: Optional[Pipeline] = None
        self._mep_model: Optional[Pipeline] = None
        self.is_trained = False
        self.metrics: dict = {}

    def _make_pipeline(self, seed: int = 42) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators  = 200,
                max_depth     = 4,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = seed,
            )),
        ])

    def train(self, n_samples: int = 2000, verbose: bool = True) -> dict:
        if verbose:
            print("  Training cost estimation models (GBR)...")

        X, y_mat, y_lab, y_mep = generate_training_data(n_samples)

        results = {}
        for name, y, attr in [
            ("material", y_mat, "_mat_model"),
            ("labor",    y_lab, "_lab_model"),
            ("mep",      y_mep, "_mep_model"),
        ]:
            model = self._make_pipeline()
            cv    = cross_val_score(model, X, y, cv=5,
                                    scoring="neg_mean_absolute_percentage_error")
            model.fit(X, y)
            setattr(self, attr, model)
            mape = -cv.mean() * 100
            results[f"{name}_mape_pct"] = round(mape, 2)
            if verbose:
                print(f"    {name:<10} MAPE = {mape:.1f}%")

        self.is_trained = True
        self.metrics    = results
        return results

    def estimate(
        self,
        area_m2:       float,
        n_rooms:       int,
        n_floors:      int,
        rooms:         list[dict],
        structural:    dict,
        mep:           Optional[dict] = None,
    ) -> CostEstimate:
        assert self.is_trained, "Call train() first."

        n_bath    = sum(1 for r in rooms if "bathroom" in r.get("room",""))
        n_kitchen = sum(1 for r in rooms if r.get("room","") == "kitchen")
        n_park    = sum(1 for r in rooms if r.get("room","") == "parking")
        has_elev  = any(r.get("room","") == "elevator" for r in rooms)
        has_pool  = any(r.get("room","") == "pool" for r in rooms)
        n_col     = len(structural.get("columns", []))
        n_beam    = len(structural.get("beams",   []))

        if mep:
            p_routes = mep.get("plumbing_routes",  [])
            e_routes = mep.get("electrical_routes",[])
            plumb_len = sum(
                sum(abs(p[j][0]-p[j-1][0])+abs(p[j][1]-p[j-1][1])
                    for j in range(1,len(p))) for p in p_routes if p
            )
            elec_len  = sum(
                sum(abs(p[j][0]-p[j-1][0])+abs(p[j][1]-p[j-1][1])
                    for j in range(1,len(p))) for p in e_routes if p
            )
        else:
            plumb_len, elec_len = 50.0, 80.0

        feat = extract_features(
            area_m2, n_rooms, n_floors, n_bath, n_kitchen,
            n_park, has_elev, has_pool, n_col, n_beam, plumb_len, elec_len
        ).reshape(1,-1)

        mat_cost = float(self._mat_model.predict(feat)[0])
        lab_cost = float(self._lab_model.predict(feat)[0])
        mep_cost = float(self._mep_model.predict(feat)[0])

        # Room-specific premiums
        room_premium = sum(
            ROOM_PREMIUM.get(r.get("room",""), 0) for r in rooms
        )
        mat_cost += room_premium * 0.4
        mep_cost += room_premium * 0.6

        total       = mat_cost + lab_cost + mep_cost
        contingency = total * 0.10
        grand       = total + contingency

        breakdown = {
            "foundation":   mat_cost * 0.15,
            "structure":    mat_cost * 0.35,
            "masonry":      mat_cost * 0.15,
            "finishes":     mat_cost * 0.20,
            "roofing":      mat_cost * 0.08,
            "mep_install":  mep_cost * 0.60,
            "fixtures":     mep_cost * 0.40,
            "civil_labor":  lab_cost * 0.35,
            "struct_labor": lab_cost * 0.35,
            "finish_labor": lab_cost * 0.30,
        }

        return CostEstimate(
            material_cost = mat_cost,
            labor_cost    = lab_cost,
            mep_cost      = mep_cost,
            total_cost    = total,
            contingency   = contingency,
            grand_total   = grand,
            cost_per_m2   = grand / max(area_m2, 1),
            area_m2       = area_m2,
            breakdown     = breakdown,
        )

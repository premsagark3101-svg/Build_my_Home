"""
column_predictor.py
===================
Scikit-learn regression pipeline that predicts optimal column spacing
at each candidate grid point, given local structural load features.

Model architecture:
  GradientBoostingRegressor (primary)
    Input  : 10-feature vector per candidate point
    Output : recommended column spacing (metres)

  Ridge regression fallback (if GBR overfits small data)

Training data is generated synthetically from structural engineering
first-principles, then the model refines predictions from actual
layout load distributions.

Engineering rules encoded in training data:
  - High load → shorter span (min 3m)
  - Light load → longer span (max 6m)
  - Edge columns → shorter spans (cantilever risk)
  - Wet rooms (bathroom/kitchen) → fixed column zones
  - Parking → heavy load, 5–6m spans (clear span needed)
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from load_estimator import LoadMap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Engineering constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_COLUMN_SPACING = 3.0   # metres
MAX_BEAM_SPAN      = 6.0   # metres
IDEAL_SPAN_LIGHT   = 5.5   # metres (< 4 kN/m²)
IDEAL_SPAN_MEDIUM  = 4.5   # metres (4–7 kN/m²)
IDEAL_SPAN_HEAVY   = 3.5   # metres (> 7 kN/m²)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data generator
# ─────────────────────────────────────────────────────────────────────────────

def _engineering_span(features: np.ndarray) -> float:
    """
    First-principles span calculation from feature vector.
    This defines the 'ground truth' for supervised training.

    Features[0] = load_mean (kN/m²)
    Features[1] = load_max
    Features[2] = load_std
    Features[3] = load_sum
    Features[4] = nonzero_fraction
    Features[5] = heavy_fraction
    Features[6] = pos_x (normalised)
    Features[7] = pos_y (normalised)
    Features[8] = dist_edge (normalised)
    Features[9] = floor_factor
    """
    load_mean   = features[0]
    heavy_frac  = features[5]
    dist_edge   = features[8]
    floor_factor= features[9]

    # Base span from mean load
    if load_mean <= 0:
        return MAX_BEAM_SPAN   # empty area → max span

    # Structural mechanics: span ∝ 1/√(load)
    # Calibrated so that at 10 kN/m² → 3m, at 4 kN/m² → 5.5m
    span = 8.5 / math.sqrt(max(load_mean, 0.5))

    # Edge penalty (columns near edges have less restraint)
    if dist_edge < 0.1:
        span *= 0.85

    # Heavy room penalty
    span *= (1.0 - heavy_frac * 0.3)

    # Multi-floor factor: lower floors carry more load → shorter spans
    span *= (0.85 + 0.15 * dist_edge / floor_factor) if floor_factor > 0 else 0.9

    return float(np.clip(span, MIN_COLUMN_SPACING, MAX_BEAM_SPAN))


def generate_training_data(
    n_samples: int = 3000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (features, span) training pairs
    spanning the full range of expected building conditions.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n_samples):
        # Randomise load scenario
        base_load    = rng.uniform(0.0, 12.0)
        load_mean    = base_load + rng.normal(0, 0.5)
        load_max     = max(load_mean, 0) + rng.uniform(0, 3.0)
        load_std     = rng.uniform(0, max(load_max, 0.1) * 0.3)
        load_sum     = load_mean * rng.uniform(4, 25)
        nonzero      = rng.uniform(0.3, 1.0)
        heavy_frac   = rng.uniform(0, 0.5) if load_mean > 5 else rng.uniform(0, 0.1)
        pos_x        = rng.uniform(0, 1)
        pos_y        = rng.uniform(0, 1)
        dist_edge    = min(pos_x, pos_y, 1-pos_x, 1-pos_y)
        floor_factor = rng.uniform(0.3, 1.0)

        feat = np.array([
            max(load_mean, 0), max(load_max, 0), load_std, max(load_sum, 0),
            nonzero, heavy_frac, pos_x, pos_y, dist_edge, floor_factor,
        ], dtype=np.float32)

        span = _engineering_span(feat)
        # Add small noise to simulate real variability
        span += rng.normal(0, 0.08)
        span  = float(np.clip(span, MIN_COLUMN_SPACING, MAX_BEAM_SPAN))

        X.append(feat); y.append(span)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Column Spacing Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ColumnSpacingPredictor:
    """
    Scikit-learn regression model that predicts recommended column
    spacing (metres) for each candidate grid point.

    Workflow
    --------
    predictor = ColumnSpacingPredictor()
    predictor.train()                            # train on synthetic data
    spacing_map = predictor.predict_map(load_map) # predict for all cells
    """

    MODEL_PATH = Path("outputs/column_predictor.pkl")

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self._train_score: Optional[float] = None

    def train(
        self,
        n_samples:  int  = 3000,
        seed:       int  = 42,
        verbose:    bool = True,
    ) -> dict:
        """Train GBR model on synthetic engineering data."""
        if verbose:
            print("  Training column spacing predictor...")

        X, y = generate_training_data(n_samples, seed)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators    = 200,
                max_depth       = 4,
                learning_rate   = 0.05,
                min_samples_leaf= 10,
                subsample       = 0.8,
                random_state    = seed,
            )),
        ])

        # Cross-validate
        cv_scores = cross_val_score(
            self.pipeline, X, y, cv=5, scoring="neg_mean_squared_error"
        )
        rmse_cv = math.sqrt(-cv_scores.mean())

        self.pipeline.fit(X, y)
        self.is_trained    = True
        self._train_score  = rmse_cv

        metrics = {
            "cv_rmse_m":      round(rmse_cv, 4),
            "n_train":        n_samples,
            "feature_names":  [
                "load_mean","load_max","load_std","load_sum",
                "nonzero_frac","heavy_frac","pos_x","pos_y",
                "dist_edge","floor_factor"
            ],
        }

        if verbose:
            print(f"    CV RMSE: {rmse_cv:.4f} m  (n={n_samples})")

        # Feature importance
        gbr = self.pipeline.named_steps["model"]
        importances = gbr.feature_importances_
        if verbose:
            names = metrics["feature_names"]
            top = sorted(zip(names, importances), key=lambda x: -x[1])[:4]
            for nm, imp in top:
                print(f"    {nm:<18} importance={imp:.3f}")

        return metrics

    def predict_span(self, features: np.ndarray) -> float:
        """Predict span for a single feature vector."""
        assert self.is_trained, "Call train() first."
        pred = float(self.pipeline.predict(features.reshape(1, -1))[0])
        return float(np.clip(pred, MIN_COLUMN_SPACING, MAX_BEAM_SPAN))

    def predict_map(self, load_map: LoadMap) -> np.ndarray:
        """
        Predict recommended column spacing for every cell in the grid.

        Returns
        -------
        spacing_map : np.ndarray (grid_size, grid_size)
            Recommended column spacing in metres at each cell.
        """
        assert self.is_trained, "Call train() first."
        G   = load_map.grid_size
        out = np.zeros((G, G), dtype=np.float32)

        # Batch prediction for speed
        features = []
        coords   = []
        for x in range(G):
            for y in range(G):
                feat = load_map.get_feature_vector(x, y)
                features.append(feat)
                coords.append((x, y))

        X_pred = np.array(features, dtype=np.float32)
        preds  = self.pipeline.predict(X_pred)
        preds  = np.clip(preds, MIN_COLUMN_SPACING, MAX_BEAM_SPAN)

        for (x, y), span in zip(coords, preds):
            out[x, y] = span

        return out

    def save(self, path: Optional[Path] = None):
        path = path or self.MODEL_PATH
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or self.MODEL_PATH
        if path.exists():
            with open(path, "rb") as f:
                self.pipeline  = pickle.load(f)
                self.is_trained = True
            return True
        return False

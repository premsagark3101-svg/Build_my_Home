"""
example_usage.py
================
Demonstrates ConstraintValidator with multiple building scenarios.
Run: python example_usage.py
"""

import json
from constraint_validator import ConstraintValidator

validator = ConstraintValidator()

SCENARIOS = [
    # ── 1. Spec example ──────────────────────────────────────────────────
    (
        "Spec Example — 2-floor house on 40×60",
        {
            "plot_width":  40, "plot_length": 60, "floors": 2,
            "rooms": {"bedroom": 3, "bathroom": 2, "kitchen": 1, "living_room": 1},
            "parking": True, "garden": True,
        }
    ),
    # ── 2. Small plot stress test ────────────────────────────────────────
    (
        "Overpacked — 5 beds on tiny 20×25 plot",
        {
            "plot_width": 20, "plot_length": 25, "floors": 1,
            "rooms": {"bedroom": 5, "bathroom": 3, "kitchen": 1,
                      "living_room": 1, "dining_room": 1},
            "parking": True,
        }
    ),
    # ── 3. Luxury villa ──────────────────────────────────────────────────
    (
        "Luxury villa on 80×100 — pool, gym, home office",
        {
            "plot_width": 80, "plot_length": 100, "floors": 3,
            "rooms": {"bedroom": 5, "bathroom": 4, "kitchen": 1,
                      "living_room": 1, "dining_room": 1, "gym": 1,
                      "home_office": 1, "study": 1, "terrace": 1},
            "parking": True, "garden": True, "pool": True,
        }
    ),
    # ── 4. Single-floor apartment ────────────────────────────────────────
    (
        "Single-floor apartment — no plot dims (area only)",
        {
            "plot_width": None, "plot_length": None, "floors": 1,
            "rooms": {"bedroom": 2, "bathroom": 1, "kitchen": 1,
                      "living_room": 1, "balcony": 1},
            "total_area_sqft": 900,
        }
    ),
    # ── 5. G+2 Indian residential ────────────────────────────────────────
    (
        "G+2 building on 30×50 — elevator, rooftop",
        {
            "plot_width": 30, "plot_length": 50, "floors": 3,
            "rooms": {"bedroom": 6, "bathroom": 4, "kitchen": 2,
                      "living_room": 2, "terrace": 1},
            "parking": True, "elevator": True, "rooftop": True,
        }
    ),
]


def divider(title: str):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


for label, nlp_json in SCENARIOS:
    divider(label)
    print(f"INPUT:\n{json.dumps(nlp_json, indent=2)}\n")

    result = validator.validate(nlp_json)

    # Human-readable summary
    print(result.summary())

    # Spec-compliant flat JSON
    flat = {k: v for k, v in result.to_dict().items()
            if k.endswith("_min_size") or k in ("plot_boundary", "floors")}
    print("\nSPEC OUTPUT (flat):")
    print(json.dumps(flat, indent=2))

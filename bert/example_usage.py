"""
example_usage.py
================
Demonstrates the BuildingNLPParser across a variety of real-world inputs.
Run with:  python example_usage.py
"""

import json
from building_nlp import BuildingNLPParser

parser = BuildingNLPParser()

TEST_CASES = [
    # ── Basic example from the spec ──────────────────────────────────────────
    (
        "Basic house",
        "I want a 2 floor house with 3 bedrooms, 2 bathrooms, kitchen, "
        "living room, parking and a garden on a 40x60 plot."
    ),

    # ── Word numbers + extra amenities ───────────────────────────────────────
    (
        "Word numbers & amenities",
        "Build me a three storey modern villa with four bedrooms, three bathrooms, "
        "a kitchen, dining room, study, home office and a swimming pool on a 60 by 80 plot."
    ),

    # ── Minimal input ────────────────────────────────────────────────────────
    (
        "Minimal description",
        "2 bed 1 bath bungalow, 30x40 plot."
    ),

    # ── Apartment style ──────────────────────────────────────────────────────
    (
        "Apartment / flat",
        "Single floor apartment: two bedrooms, one bathroom, an open kitchen, "
        "a living room and balcony. Total area 90 sqm."
    ),

    # ── G+N notation ─────────────────────────────────────────────────────────
    (
        "G+2 notation (Indian standard)",
        "G+2 residential building, 30x50 site, 6 bedrooms, 4 bathrooms, "
        "2 kitchens, parking, elevator, rooftop terrace."
    ),

    # ── Commercial / mixed ───────────────────────────────────────────────────
    (
        "Large mixed-use building",
        "I need a five-storey contemporary building: ground floor with gym, "
        "laundry room and utility room, upper floors with 12 guest rooms each "
        "with private bathrooms, a rooftop terrace, solar panels and underground parking."
    ),

    # ── Typos / informal ─────────────────────────────────────────────────────
    (
        "Informal / abbreviated",
        "3bed 2bath house, garden, garage, 25x45 plot, solar panels."
    ),
]


def separator(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


for label, text in TEST_CASES:
    separator(label)
    print(f"INPUT : {text}\n")

    result = parser.parse(text)
    output = result.to_dict()

    # Pretty-print JSON
    print("OUTPUT:")
    print(json.dumps(output, indent=2))

# Building NLP Parser

A production-grade Natural Language Processing module that converts free-form building requirement descriptions into validated, structured JSON constraints.

---

## Architecture

```
User Text Input
      │
      ▼
┌─────────────────────────────────────┐
│         BuildingNLPParser           │
│   (orchestrates backend selection)  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
Transformer    Rule-Based
 Extractor     Extractor  ◄─── Active (spaCy/transformers not installed)
(BERT/spaCy)  (regex NLP)
    │             │
    └──────┬──────┘
           ▼
  ConstraintValidator
  (cross-field checks)
           │
           ▼
  BuildingConstraints
  (structured dataclass)
           │
           ▼
   JSON Output
```

---

## Output JSON Schema

```json
{
  "plot_width":   40.0,         // Plot dimension (numeric, any unit)
  "plot_length":  60.0,         // Plot dimension
  "floors":       2,            // Number of storeys (int, ≥1)
  "rooms": {
    "bedroom":    3,            // Present only if count > 0
    "bathroom":   2,
    "kitchen":    1,
    "living_room":1,
    "dining_room":0,            // omitted when 0
    "study":      0,
    "garage":     0,
    "balcony":    0,
    "terrace":    0,
    "storage":    0,
    "laundry":    0,
    "gym":        0,
    "home_office":0,
    "guest_room": 0,
    "utility_room":0
  },
  "parking":      true,         // Boolean amenities
  "garden":       false,
  "pool":         false,        // Included only when true
  "basement":     false,
  "rooftop":      false,
  "elevator":     false,
  "solar_panels": false,
  "total_area_sqft": null,      // When plot dims not given but area is
  "style":        "modern",     // Architectural style if mentioned
  "warnings":     []            // Validation messages
}
```

---

## Supported Input Patterns

| Feature | Examples |
|---|---|
| Plot dimensions | `40x60`, `40 by 60`, `40×60` |
| Floor count | `2 floors`, `three storey`, `G+2`, `double floor` |
| Bedrooms | `3 bedrooms`, `three beds`, `3bed` |
| Bathrooms | `2 bathrooms`, `2 bath`, `two toilets` |
| Common rooms | kitchen, living room, dining room, study, lounge |
| Special rooms | home office, gym, laundry, utility room, guest room |
| Amenities | parking, garden, pool, basement, rooftop, elevator, solar panels |
| Total area | `90 sqm`, `1200 sqft` |
| Style | modern, contemporary, colonial, bungalow, villa, etc. |
| Word numbers | one, two, three … twenty, thirty, forty … |

---

## Validation Rules

- **Floors**: must be 1–200; defaults to 1 if invalid
- **Room counts**: must be 0–50; capped and warned
- **Plot dimensions**: must be positive; warned if >10,000 units
- **Cross-field**: warns on high bath:bed ratio, missing bedrooms, multi-floor without elevator

---

## Upgrading to Transformer Backend

```bash
pip install spacy transformers torch
python -m spacy download en_core_web_sm
```

Then implement `TransformerExtractor.extract()` in `building_nlp.py`. The parser will auto-detect and prefer it.

---

## Quick Start

```python
from building_nlp import BuildingNLPParser

parser = BuildingNLPParser()

result = parser.parse(
    "I want a 2 floor house with 3 bedrooms, 2 bathrooms, "
    "kitchen, living room, parking and a garden on a 40x60 plot."
)

print(result.to_json())
# → clean JSON output

# Or work with the dataclass directly
print(result.floors)          # 2
print(result.rooms.bedroom)   # 3
print(result.parking)         # True
```

---

## Files

| File | Description |
|---|---|
| `building_nlp.py` | Core NLP module |
| `example_usage.py` | 7 varied test cases with output |
| `README.md` | This file |

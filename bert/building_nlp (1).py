"""
building_nlp.py
===============
Natural Language Processing module that converts user building requirements
into structured JSON constraints.

Architecture:
  - Primary  : Rule-based NLP pipeline (regex + linguistic patterns)
  - Secondary: Transformer-ready interface (plug in spaCy / HuggingFace BERT)
  - Fallback : Safe defaults with validation warnings

Usage:
  from building_nlp import BuildingNLPParser
  parser = BuildingNLPParser()
  result = parser.parse("I want a 2 floor house with 3 bedrooms ...")
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RoomConstraints:
    bedroom: int = 0
    bathroom: int = 0
    kitchen: int = 0
    living_room: int = 0
    dining_room: int = 0
    study: int = 0
    garage: int = 0
    balcony: int = 0
    terrace: int = 0
    storage: int = 0
    laundry: int = 0
    gym: int = 0
    home_office: int = 0
    guest_room: int = 0
    utility_room: int = 0

    def to_dict(self) -> dict:
        """Return only rooms with count > 0."""
        return {k: v for k, v in asdict(self).items() if v > 0}


@dataclass
class BuildingConstraints:
    plot_width: Optional[float] = None
    plot_length: Optional[float] = None
    floors: int = 1
    rooms: RoomConstraints = field(default_factory=RoomConstraints)
    parking: bool = False
    garden: bool = False
    pool: bool = False
    basement: bool = False
    rooftop: bool = False
    elevator: bool = False
    solar_panels: bool = False
    total_area: Optional[float] = None
    style: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "plot_width": self.plot_width,
            "plot_length": self.plot_length,
            "floors": self.floors,
            "rooms": self.rooms.to_dict(),
            "parking": self.parking,
            "garden": self.garden,
        }
        # Include optional boolean features only if True
        for feat in ("pool", "basement", "rooftop", "elevator", "solar_panels"):
            if getattr(self, feat):
                d[feat] = True
        if self.total_area is not None:
            d["total_area_sqft"] = self.total_area
        if self.style:
            d["style"] = self.style
        if self.warnings:
            d["warnings"] = self.warnings
        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ConstraintValidator:
    """Validates and sanitises extracted numeric / boolean constraints."""

    MAX_FLOORS = 200
    MAX_ROOMS  = 50
    MAX_PLOT   = 10_000   # metres / feet

    @staticmethod
    def validate_floors(value: int, warnings: list[str]) -> int:
        if not isinstance(value, int) or value < 1:
            warnings.append(f"Invalid floor count '{value}', defaulting to 1.")
            return 1
        if value > ConstraintValidator.MAX_FLOORS:
            warnings.append(f"Floor count {value} seems unrealistic; capped at {ConstraintValidator.MAX_FLOORS}.")
            return ConstraintValidator.MAX_FLOORS
        return value

    @staticmethod
    def validate_room_count(name: str, value: int, warnings: list[str]) -> int:
        if not isinstance(value, int) or value < 0:
            warnings.append(f"Invalid count for '{name}': {value}. Set to 0.")
            return 0
        if value > ConstraintValidator.MAX_ROOMS:
            warnings.append(f"Room count for '{name}' ({value}) is unusually high; capped at {ConstraintValidator.MAX_ROOMS}.")
            return ConstraintValidator.MAX_ROOMS
        return value

    @staticmethod
    def validate_plot_dim(dim: str, value: float, warnings: list[str]) -> float:
        if value <= 0:
            warnings.append(f"Plot {dim} must be positive; got {value}.")
            return None
        if value > ConstraintValidator.MAX_PLOT:
            warnings.append(f"Plot {dim} of {value} is unusually large.")
        return value


# ---------------------------------------------------------------------------
# Word-to-number conversion
# ---------------------------------------------------------------------------

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "a": 1, "an": 1, "single": 1, "double": 2, "triple": 3, "quad": 4,
}

def word_to_int(text: str) -> Optional[int]:
    """Convert word or digit string to integer. Returns None on failure."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    if text in WORD_TO_NUM:
        return WORD_TO_NUM[text]
    # Handle compound words like "twenty-five"
    parts = re.split(r"[-\s]", text)
    total = 0
    for part in parts:
        n = WORD_TO_NUM.get(part)
        if n is None:
            return None
        total += n
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Core rule-based extractor
# ---------------------------------------------------------------------------

NUM_PAT = r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|" \
          r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|" \
          r"thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|single|double|triple)"

class RuleBasedExtractor:
    """
    Linguistic rule-based NLP pipeline using regex + contextual patterns.

    Designed to mirror the interface a transformer-based model would expose,
    making it straightforward to swap in a BERT/spaCy backend later.
    """

    # ------------------------------------------------------------------
    # Plot / dimensions
    # ------------------------------------------------------------------
    PLOT_PATTERNS = [
        # "40x60 plot"  |  "40 by 60"  |  "40*60"
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|by|\*|×)\s*(\d+(?:\.\d+)?)\s*(?:plot|land|site|lot|sqm|sq\.?m|sqft|sq\.?ft|m²|ft²)?", re.I),
    ]
    TOTAL_AREA_PATTERNS = [
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:sqm|sq\.?m|m²|square\s*met(?:re|er)s?)", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:sqft|sq\.?ft|ft²|square\s*fe?e?t)", re.I),
    ]

    # ------------------------------------------------------------------
    # Floors
    # ------------------------------------------------------------------
    FLOOR_PATTERNS = [
        re.compile(rf"({NUM_PAT})\s*(?:-\s*)?(?:floor|story|storey|level)s?", re.I),
        re.compile(rf"(?:floor|story|storey|level)s?\s*:\s*({NUM_PAT})", re.I),
        re.compile(r"\b(single|double|triple|one|two|three|four|five)\s*(?:-\s*)?(?:floor|story|storey|level)\b", re.I),
        re.compile(r"\b(g\+\d+)\b", re.I),    # e.g. G+2 → 3 floors
    ]

    # ------------------------------------------------------------------
    # Room patterns  →  (pattern, room_field_name)
    # ------------------------------------------------------------------
    ROOM_PATTERNS = [
        (re.compile(rf"({NUM_PAT})\s*bed(?:room)?s?", re.I),             "bedroom"),
        (re.compile(rf"({NUM_PAT})\s*bath(?:room)?s?", re.I),            "bathroom"),
        (re.compile(rf"({NUM_PAT})\s*toilet\b", re.I),                   "bathroom"),
        (re.compile(rf"({NUM_PAT})\s*(?:wc|washroom)s?\b", re.I),       "bathroom"),
        (re.compile(rf"({NUM_PAT})\s*kitchen", re.I),                    "kitchen"),
        (re.compile(r"(?:a\s+)?kitchen", re.I),                          "kitchen"),   # "a kitchen"
        (re.compile(rf"({NUM_PAT})\s*living\s*room", re.I),              "living_room"),
        (re.compile(r"(?:a\s+)?living\s*room", re.I),                    "living_room"),
        (re.compile(r"(?:a\s+)?lounge\b", re.I),                         "living_room"),
        (re.compile(rf"({NUM_PAT})\s*dining\s*room", re.I),              "dining_room"),
        (re.compile(r"(?:a\s+)?dining\s*(?:room|area)", re.I),          "dining_room"),
        (re.compile(rf"({NUM_PAT})\s*stud(?:y|ies)", re.I),              "study"),
        (re.compile(r"(?:a\s+)?study\b", re.I),                          "study"),
        (re.compile(rf"({NUM_PAT})\s*garage", re.I),                     "garage"),
        (re.compile(rf"({NUM_PAT})\s*balcon(?:y|ies)", re.I),            "balcony"),
        (re.compile(r"(?:a\s+)?balcon(?:y|ies)", re.I),                  "balcony"),
        (re.compile(rf"({NUM_PAT})\s*terrace", re.I),                    "terrace"),
        (re.compile(r"(?:a\s+)?terrace", re.I),                          "terrace"),
        (re.compile(rf"({NUM_PAT})\s*storage", re.I),                    "storage"),
        (re.compile(r"(?:a\s+)?(?:storage|store\s*room)", re.I),         "storage"),
        (re.compile(r"(?:a\s+)?laundry\s*(?:room)?", re.I),             "laundry"),
        (re.compile(r"(?:a\s+)?gym\b", re.I),                            "gym"),
        (re.compile(r"(?:a\s+)?home\s*office", re.I),                    "home_office"),
        (re.compile(rf"({NUM_PAT})\s*guest\s*(?:room|bedroom)", re.I),   "guest_room"),
        (re.compile(r"(?:a\s+)?utility\s*room", re.I),                   "utility_room"),
    ]

    # ------------------------------------------------------------------
    # Boolean amenity patterns
    # ------------------------------------------------------------------
    BOOL_PATTERNS = {
        "parking":      re.compile(r"\b(?:parking|car\s*park|driveway)\b", re.I),
        "garden":       re.compile(r"\b(?:garden|yard|lawn|backyard|front\s*yard)\b", re.I),
        "pool":         re.compile(r"\b(?:pool|swimming\s*pool)\b", re.I),
        "basement":     re.compile(r"\b(?:basement|cellar|underground)\b", re.I),
        "rooftop":      re.compile(r"\b(?:rooftop|roof\s*deck|roof\s*terrace)\b", re.I),
        "elevator":     re.compile(r"\b(?:elevator|lift)\b", re.I),
        "solar_panels": re.compile(r"\b(?:solar\s*panel|solar\s*energy|photovoltaic)\b", re.I),
    }

    # ------------------------------------------------------------------
    # Style patterns
    # ------------------------------------------------------------------
    STYLE_PATTERNS = re.compile(
        r"\b(modern|contemporary|traditional|colonial|victorian|mediterranean|"
        r"minimalist|industrial|craftsman|ranch|bungalow|villa|cottage|farmhouse)\b", re.I
    )

    # ------------------------------------------------------------------

    def extract(self, text: str) -> BuildingConstraints:
        constraints = BuildingConstraints()
        warnings    = constraints.warnings
        text_lower  = text.lower()

        # 1. Plot dimensions
        self._extract_plot(text, constraints, warnings)

        # 2. Total area (if no plot dims found)
        if constraints.plot_width is None and constraints.plot_length is None:
            self._extract_total_area(text, constraints, warnings)

        # 3. Floors
        self._extract_floors(text, text_lower, constraints, warnings)

        # 4. Rooms
        self._extract_rooms(text, constraints, warnings)

        # 5. Boolean amenities
        self._extract_booleans(text, constraints)

        # 6. Style
        style_match = self.STYLE_PATTERNS.search(text)
        if style_match:
            constraints.style = style_match.group(1).lower()

        return constraints

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _extract_plot(self, text: str, c: BuildingConstraints, warnings: list):
        for pat in self.PLOT_PATTERNS:
            m = pat.search(text)
            if m:
                w = float(m.group(1))
                l = float(m.group(2))
                c.plot_width  = ConstraintValidator.validate_plot_dim("width",  w, warnings)
                c.plot_length = ConstraintValidator.validate_plot_dim("length", l, warnings)
                return

    def _extract_total_area(self, text: str, c: BuildingConstraints, warnings: list):
        for pat in self.TOTAL_AREA_PATTERNS:
            m = pat.search(text)
            if m:
                c.total_area = float(m.group(1))
                return

    def _extract_floors(self, text: str, text_lower: str, c: BuildingConstraints, warnings: list):
        # G+N notation (e.g. "G+2" means ground + 2 upper = 3 floors)
        gplus = re.search(r"\bg\+(\d+)\b", text_lower)
        if gplus:
            c.floors = ConstraintValidator.validate_floors(int(gplus.group(1)) + 1, warnings)
            return

        for pat in self.FLOOR_PATTERNS:
            m = pat.search(text)
            if m:
                raw = m.group(1)
                val = word_to_int(raw)
                if val is not None:
                    c.floors = ConstraintValidator.validate_floors(val, warnings)
                    return

    def _extract_rooms(self, text: str, c: BuildingConstraints, warnings: list):
        rooms = c.rooms
        counted_rooms: set[str] = set()   # avoid double-counting

        for pat, room_name in self.ROOM_PATTERNS:
            m = pat.search(text)
            if not m:
                continue

            # Pattern may capture a number group or be a plain boolean-style match
            raw_count = None
            try:
                raw_count = m.group(1)
            except IndexError:
                pass

            if raw_count:
                count = word_to_int(raw_count)
                if count is None:
                    count = 1
            else:
                count = 1

            count = ConstraintValidator.validate_room_count(room_name, count, warnings)

            # Only set if not already set by a more specific pattern
            if room_name not in counted_rooms:
                current = getattr(rooms, room_name, 0)
                if count > current:
                    setattr(rooms, room_name, count)
                counted_rooms.add(room_name)

    def _extract_booleans(self, text: str, c: BuildingConstraints):
        for field_name, pat in self.BOOL_PATTERNS.items():
            if pat.search(text):
                setattr(c, field_name, True)


# ---------------------------------------------------------------------------
# Transformer-ready interface (stub — plug in BERT/spaCy here)
# ---------------------------------------------------------------------------

class TransformerExtractor:
    """
    Placeholder for a transformer-based extractor (BERT / spaCy NER).

    To activate:
      pip install transformers spacy
      python -m spacy download en_core_web_sm

    Then implement extract() using the loaded model.
    """

    def __init__(self):
        self._available = False
        try:
            import spacy                                          # noqa: F401
            import transformers                                   # noqa: F401
            self._available = True
            logger.info("Transformer backend detected (spaCy + HuggingFace).")
        except ImportError:
            logger.info("Transformer libraries not found — using rule-based fallback.")

    @property
    def is_available(self) -> bool:
        return self._available

    def extract(self, text: str) -> BuildingConstraints:
        """
        Production implementation would:
          1. Tokenize text with spaCy / BERT tokenizer.
          2. Run NER to detect quantities and entity types.
          3. Map NER labels → BuildingConstraints fields.
          4. Return structured constraints.
        """
        raise NotImplementedError("Transformer backend not yet implemented.")


# ---------------------------------------------------------------------------
# Main parser — orchestrates backends
# ---------------------------------------------------------------------------

class BuildingNLPParser:
    """
    High-level parser that routes between transformer and rule-based backends.

    Example
    -------
    >>> parser = BuildingNLPParser()
    >>> result = parser.parse("3 bed, 2 bath house on a 30x50 plot with a garden")
    >>> print(result.to_json())
    """

    def __init__(self, prefer_transformer: bool = True):
        self._transformer = TransformerExtractor()
        self._rule_based  = RuleBasedExtractor()
        self._use_transformer = prefer_transformer and self._transformer.is_available

    def parse(self, text: str) -> BuildingConstraints:
        """
        Parse a natural-language building requirement string.

        Parameters
        ----------
        text : str
            Free-form description of building requirements.

        Returns
        -------
        BuildingConstraints
            Structured constraint object with .to_dict() and .to_json() methods.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        text = self._preprocess(text)
        logger.info(f"Parsing: '{text[:80]}{'...' if len(text) > 80 else ''}'")

        if self._use_transformer:
            try:
                constraints = self._transformer.extract(text)
                logger.info("Extraction completed via transformer backend.")
                return constraints
            except Exception as e:
                logger.warning(f"Transformer failed ({e}); falling back to rule-based.")

        constraints = self._rule_based.extract(text)
        logger.info("Extraction completed via rule-based backend.")
        self._post_validate(constraints)
        return constraints

    @staticmethod
    def _preprocess(text: str) -> str:
        """Normalise text before extraction."""
        text = text.strip()
        # Normalise quotation marks
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r'["""]', '"', text)
        # Normalise separators
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _post_validate(c: BuildingConstraints):
        """Cross-field logical validation."""
        warnings = c.warnings

        # Multi-floor buildings should have stairs or elevator
        if c.floors > 2 and not c.elevator:
            warnings.append("Multi-floor building (>2): consider adding an elevator.")

        # Bedroom count consistency
        total_beds = c.rooms.bedroom + c.rooms.guest_room
        if total_beds == 0 and c.floors > 0:
            warnings.append("No bedrooms detected — is this a commercial or studio building?")

        # Bathroom-to-bedroom ratio
        if c.rooms.bathroom > 0 and c.rooms.bedroom > 0:
            ratio = c.rooms.bathroom / c.rooms.bedroom
            if ratio > 2:
                warnings.append(
                    f"Bathroom-to-bedroom ratio ({c.rooms.bathroom}:{c.rooms.bedroom}) is unusually high."
                )

        # Plot sanity
        if c.plot_width and c.plot_length:
            plot_area = c.plot_width * c.plot_length
            if plot_area < 50:
                warnings.append(f"Plot area ({plot_area} sq units) seems very small.")


# ---------------------------------------------------------------------------
# Convenience output methods (monkey-patched onto BuildingConstraints)
# ---------------------------------------------------------------------------

def _to_json(self: BuildingConstraints, indent: int = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent)

BuildingConstraints.to_json = _to_json


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    parser = BuildingNLPParser()

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = (
            "I want a 2 floor house with 3 bedrooms, 2 bathrooms, kitchen, "
            "living room, parking and a garden on a 40x60 plot."
        )

    result = parser.parse(user_input)
    print(result.to_json())

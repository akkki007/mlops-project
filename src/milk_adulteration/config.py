"""Shared constants and params loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# The six field-level readings the v1 model uses (PRD: "Model inputs (v1)").
FEATURES: list[str] = [
    "fat_pct",
    "snf_pct",
    "density",
    "ph",
    "freezing_point",
    "conductivity",
]

# Source column in the ommadav dataset -> our column name.
# Specific gravity is used as density (g/mL); at ~20 °C they differ by < 0.2%.
SOURCE_COLUMNS: dict[str, str] = {
    "Sample_ID": "sample_id",
    "Collection_Date": "collection_date",
    "Collection_Point": "collection_point",
    "State": "state",
    "Milk_Type": "milk_type",
    "Fat_percent": "fat_pct",
    "SNF_percent": "snf_pct",
    "Specific_Gravity": "density",
    "pH": "ph",
    "Freezing_Point_C": "freezing_point",
    "Conductivity_mS_cm": "conductivity",
    "Is_Adulterated": "is_adulterated",
    "Adulterant_Detected": "adulterant_raw",
}

PURE_CLASS = "none"
OTHER_CLASS = "other"

# v1 Stage 2 classes; adulterants with weak field-reading signal collapse to "other".
ADULTERANT_CLASSES: dict[str, str] = {
    "Water": "water",
    "Starch": "starch",
    "Urea": "urea",
    "Detergent": "detergent",
    "Sodium Bicarbonate": "sodium_bicarbonate",
    "Glucose": "glucose",
    "Skimmed Milk Powder": "skimmed_milk_powder",
    "Formalin": OTHER_CLASS,
    "Hydrogen Peroxide": OTHER_CLASS,
    "Vegetable Oil": OTHER_CLASS,
    "Melamine": OTHER_CLASS,
}

STAGE2_CLASSES: list[str] = sorted(set(ADULTERANT_CLASSES.values()))


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a params path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

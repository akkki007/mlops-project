"""FSSAI / dairy-literature rule baseline.

Thresholds are fixed from published standards, not fitted to the data:
- Fat and SNF minimums per milk type: FSSAI Food Products Standards, 2.1.2.
- Freezing point: normal milk sits around -0.512 to -0.550 °C.
- Density 1.026-1.034 g/mL, pH 6.4-6.8, conductivity 4.0-5.5 mS/cm: normal ranges
  for fresh cow/buffalo milk.

The score is the number of rules a sample breaks; one breach flags it.
"""

from __future__ import annotations

import pandas as pd

# milk_type -> (min fat %, min SNF %)
FSSAI_MINIMUMS: dict[str, tuple[float, float]] = {
    "Full Cream": (6.0, 9.0),
    "Standardised": (4.5, 8.5),
    "Toned": (3.0, 8.5),
    "Double Toned": (1.5, 9.0),
    "Skimmed": (0.0, 8.7),
}

RANGES: dict[str, tuple[float, float]] = {
    "freezing_point": (-0.550, -0.512),
    "density": (1.026, 1.034),
    "ph": (6.4, 6.8),
    "conductivity": (4.0, 5.5),
}

RULES: list[str] = ["fat_below_min", "snf_below_min", *(f"{c}_out_of_range" for c in RANGES)]


def breaches(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule. Unknown milk types only get the range rules."""
    mins = df["milk_type"].map(FSSAI_MINIMUMS)
    min_fat = mins.map(lambda m: m[0] if isinstance(m, tuple) else float("-inf"))
    min_snf = mins.map(lambda m: m[1] if isinstance(m, tuple) else float("-inf"))
    out = pd.DataFrame(index=df.index)
    out["fat_below_min"] = df["fat_pct"] < min_fat
    out["snf_below_min"] = df["snf_pct"] < min_snf
    for col, (lo, hi) in RANGES.items():
        out[f"{col}_out_of_range"] = (df[col] < lo) | (df[col] > hi)
    return out[RULES]


def score(df: pd.DataFrame) -> pd.Series:
    return breaches(df).sum(axis=1).astype(float)

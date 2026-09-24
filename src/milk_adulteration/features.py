"""Derived features that expose masked fraud (PRD: "Derived features")."""

from __future__ import annotations

import pandas as pd

from milk_adulteration.config import FEATURES

NORMAL_FREEZING_POINT = -0.52  # °C

DERIVED_FEATURES: list[str] = ["fat_snf_ratio", "density_residual", "fp_deviation"]


def expected_density(fat_pct: pd.Series, snf_pct: pd.Series) -> pd.Series:
    """Density implied by fat and SNF via Richmond's formula.

    SNF = 0.25 * CLR + 0.22 * fat + 0.72, with CLR = (density - 1) * 1000.
    """
    clr = (snf_pct - 0.22 * fat_pct - 0.72) / 0.25
    return 1 + clr / 1000


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived features added. Needs no fitting, so it is leak-free."""
    out = df.copy()
    out["fat_snf_ratio"] = out["fat_pct"] / out["snf_pct"]
    out["density_residual"] = out["density"] - expected_density(out["fat_pct"], out["snf_pct"])
    out["fp_deviation"] = out["freezing_point"] - NORMAL_FREEZING_POINT
    return out


def feature_matrix(df: pd.DataFrame, derived: bool = True) -> pd.DataFrame:
    cols = FEATURES + (DERIVED_FEATURES if derived else [])
    return (add_derived(df) if derived else df)[cols]

"""Pandera schemas for milk readings.

`FIELD_SCHEMA` covers the six model inputs and is shared by training and the API.
`CLEAN_SCHEMA` adds identifiers and labels for the training table.
"""

from __future__ import annotations

import pandera.pandas as pa

from milk_adulteration.config import PURE_CLASS, STAGE2_CLASSES

# Plausible physical ranges (PRD: "Data validation (Pandera)").
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "fat_pct": (0.0, 15.0),
    "snf_pct": (0.0, 15.0),
    "density": (1.000, 1.040),
    "ph": (5.5, 8.5),
    "freezing_point": (-1.0, 0.0),
    "conductivity": (2.0, 10.0),
}


def _feature_columns() -> dict[str, pa.Column]:
    return {
        name: pa.Column(
            float,
            checks=pa.Check.in_range(lo, hi),
            nullable=False,
            coerce=True,
        )
        for name, (lo, hi) in FEATURE_RANGES.items()
    }


FIELD_SCHEMA = pa.DataFrameSchema(_feature_columns(), strict=False, name="field_readings")

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "sample_id": pa.Column(str, nullable=False, unique=True),
        "collection_date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "collection_point": pa.Column(str, nullable=False),
        "state": pa.Column(str, nullable=False),
        "milk_type": pa.Column(str, nullable=False),
        **_feature_columns(),
        "is_adulterated": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True),
        "adulterant": pa.Column(str, pa.Check.isin([PURE_CLASS, *STAGE2_CLASSES]), nullable=False),
    },
    checks=[
        # Label consistency: pure rows have no adulterant, adulterated rows name one.
        pa.Check(
            lambda df: (df["is_adulterated"] == 1) == (df["adulterant"] != PURE_CLASS),
            element_wise=False,
            error="is_adulterated disagrees with adulterant",
        ),
    ],
    strict=True,
    ordered=False,
    name="milk_clean",
)

"""DVC stage `validate`: select v1 columns, map labels and enforce the schema.

Rows that fail validation are written to a rejected file with a reason;
they are never silently kept or scored.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import pandera.pandas as pa

from milk_adulteration.config import (
    ADULTERANT_CLASSES,
    FEATURES,
    PURE_CLASS,
    SOURCE_COLUMNS,
    load_params,
    resolve,
)
from milk_adulteration.data.schema import CLEAN_SCHEMA, FIELD_SCHEMA

log = logging.getLogger(__name__)


def map_adulterant(raw: pd.Series) -> pd.Series:
    """Map source adulterant names to v1 classes; missing means pure.

    Unknown names are kept as-is so schema validation rejects them loudly.
    """
    mapped = raw.map(lambda v: PURE_CLASS if pd.isna(v) else ADULTERANT_CLASSES.get(v, v))
    return mapped.astype(str)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(raw.columns)
    if missing:
        raise KeyError(f"Source data is missing columns: {sorted(missing)}")
    df = raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    df["adulterant"] = map_adulterant(df.pop("adulterant_raw"))
    return df.reset_index(drop=True)


def split_valid(
    df: pd.DataFrame, schema: pa.DataFrameSchema = CLEAN_SCHEMA
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid rows, rejected rows with a `reject_reason` column)."""
    try:
        return schema.validate(df, lazy=True), df.iloc[0:0].assign(reject_reason=[])
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        row_cases = cases[cases["index"].notna()]
        if len(row_cases) < len(cases):
            # Column-level failure (missing column, wrong dtype): not fixable per row.
            raise
        # str() per value: pandas 3 string dtype keeps NaN through astype(str).
        reason = [
            f"{'row' if pd.isna(col) else col}: {check} (got {case})"
            for col, check, case in zip(
                row_cases["column"], row_cases["check"], row_cases["failure_case"], strict=True
            )
        ]
        reasons = row_cases.assign(reason=reason).groupby("index")["reason"].agg("; ".join)
        reasons.index = reasons.index.astype(int)
        bad = df.index.isin(reasons.index)
        rejected = df[bad].assign(reject_reason=reasons)
        valid = schema.validate(df[~bad])
        return valid, rejected


def split_valid_readings(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate incoming readings (API, CSV batches) against `FIELD_SCHEMA`.

    Non-numeric values would make Pandera fail the whole column, so they are
    turned into row-level rejections first.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    df = df.copy()
    reasons: dict[int, list[str]] = {}
    for col in FEATURES:
        num = pd.to_numeric(df[col], errors="coerce")
        for i in df.index[num.isna() & df[col].notna()]:
            reasons.setdefault(i, []).append(f"{col}: not a number (got {df.at[i, col]!r})")
        df[col] = num.astype(float)
    bad = df.index.isin(list(reasons))
    not_numeric = df[bad].assign(reject_reason=["; ".join(reasons[i]) for i in df.index[bad]])
    valid, rejected = split_valid(df[~bad], FIELD_SCHEMA)
    return valid, pd.concat([rejected, not_numeric]).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = load_params()["validate"]

    raw = pd.read_csv(resolve(p["input"]))
    valid, rejected = split_valid(prepare(raw))

    out = resolve(p["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out, index=False)
    rejected.to_csv(resolve(p["rejected"]), index=False)
    log.info("valid=%d rejected=%d -> %s", len(valid), len(rejected), out)


if __name__ == "__main__":
    main()

"""DVC stage `augment`: grow each Stage 2 class to a target size from training data.

Method: "shift transplant". For each real adulterated training row we measure its
shift from the pure-milk mean of its milk type. A synthetic row is a random pure
training row plus one of those shifts, scaled by a random dose. This keeps each
adulterant's signature as the data shows it while varying the base milk and the
dose.

Why not a physical mixing model: in the source data the water rows' fat, SNF,
density and freezing-point drops imply about 40%, 21%, 33% and 12% dilution
respectively, so the generator does not follow mixing physics and a physics model
would teach shifts the data never shows. See reports/model.md for the with/without
augmentation comparison.

Combination rows (e.g. water + starch) add two shifts to one pure row; they model
masked fraud and are used for Stage 1 only.

Only the training split is augmented; validation and test stay real.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from milk_adulteration.config import FEATURES, PURE_CLASS, STAGE2_CLASSES, load_params, resolve
from milk_adulteration.data.schema import FEATURE_RANGES

log = logging.getLogger(__name__)


def pure_shifts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pure rows, adulterated rows' feature shifts from their milk type's pure mean)."""
    pure = df[df["adulterant"] == PURE_CLASS]
    base = pure.groupby("milk_type")[FEATURES].mean()
    pos = df[df["adulterant"] != PURE_CLASS]
    # A milk type with no pure rows falls back to the overall pure mean.
    ref = base.reindex(pos["milk_type"]).fillna(pure[FEATURES].mean())
    shifts = pd.DataFrame(pos[FEATURES].to_numpy() - ref.to_numpy(), columns=FEATURES)
    shifts["adulterant"] = pos["adulterant"].to_numpy()
    return pure, shifts


def _in_range(values: np.ndarray) -> np.ndarray:
    ok = np.ones(len(values), dtype=bool)
    for i, col in enumerate(FEATURES):
        lo, hi = FEATURE_RANGES[col]
        ok &= (values[:, i] >= lo) & (values[:, i] <= hi)
    return ok


def _synthesize(
    pure: pd.DataFrame,
    shift_sets: list[np.ndarray],
    n: int,
    dose_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """n rows = pure base + sum over shift_sets of (dose * random shift from that set)."""
    rows, doses = [], []
    while sum(len(r) for r in rows) < n:
        m = 2 * n
        base = pure.iloc[rng.integers(0, len(pure), m)]
        values = base[FEATURES].to_numpy().copy()
        dose = rng.uniform(*dose_range, size=(m, len(shift_sets)))
        for j, shifts in enumerate(shift_sets):
            values += dose[:, [j]] * shifts[rng.integers(0, len(shifts), m)]
        ok = _in_range(values)  # never create a row the schema would reject
        out = base.loc[:, ~base.columns.isin(FEATURES)].copy()
        out[FEATURES] = values
        rows.append(out[ok])
        doses.append(dose[ok].mean(axis=1))
    return pd.concat(rows).head(n), np.concatenate(doses)[:n]


def augment(df: pd.DataFrame, p: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Return real rows plus synthetic rows, tagged with `augmented`, `aug_combo`, `aug_dose`."""
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    dose_range = tuple(p["dose_range"])
    pure, shifts = pure_shifts(df)
    by_class = {c: g[FEATURES].to_numpy() for c, g in shifts.groupby("adulterant") if len(g) > 0}

    real = df.assign(augmented=False, aug_combo="", aug_dose=np.nan)
    parts = [real]
    for cls in STAGE2_CLASSES:
        need = p["target_per_class"] - int((df["adulterant"] == cls).sum())
        if need <= 0 or cls not in by_class:
            continue
        rows, dose = _synthesize(pure, [by_class[cls]], need, dose_range, rng)
        parts.append(
            rows.assign(
                adulterant=cls, is_adulterated=1, augmented=True, aug_combo="", aug_dose=dose
            )
        )
    for combo in p.get("combos", []):
        if not all(c in by_class for c in combo):
            continue
        name = "+".join(combo)
        rows, dose = _synthesize(
            pure, [by_class[c] for c in combo], p["rows_per_combo"], dose_range, rng
        )
        parts.append(
            rows.assign(
                adulterant=name, is_adulterated=1, augmented=True, aug_combo=name, aug_dose=dose
            )
        )

    out = pd.concat(parts, ignore_index=True)
    n_aug = int(out["augmented"].sum())
    out.loc[out["augmented"], "sample_id"] = [f"AUG{i:05d}" for i in range(n_aug)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = load_params()["augment"]
    out = augment(pd.read_csv(resolve(p["input"])), p)
    out.to_csv(resolve(p["output"]), index=False)
    counts = out[out["is_adulterated"] == 1].groupby(["adulterant", "augmented"]).size()
    log.info("Rows by class and augmented flag:\n%s", counts.unstack(fill_value=0))


if __name__ == "__main__":
    main()

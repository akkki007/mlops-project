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

Widened pure rows: the synthetic data's pure milk sits in narrow bands (freezing
point never below -0.540 C, for example) while real cow and buffalo milk varies
more. `widen_pure` stretches each reading of copies of the real pure rows, one
linear map per reading, from its training range onto the wider published range.
This keeps each row's relative position and the rows' ordering. These rows are
pure (Stage 1 only) and also serve as bases for the synthetic adulterated rows, so
adulterants are learned on the wider bases too.

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

MAX_ATTEMPTS = 50


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


def widen(pure: pd.DataFrame, ranges: dict[str, list[float]], n: int, rng) -> pd.DataFrame:
    """n copies of random pure rows, each reading stretched onto the union of its
    training range and `ranges[reading]`. Readings not in `ranges` are unchanged."""
    rows = pure.iloc[rng.integers(0, len(pure), n)].copy()
    for col, (lo, hi) in ranges.items():
        a, b = float(pure[col].min()), float(pure[col].max())
        lo, hi = min(lo, a), max(hi, b)
        if b > a:
            rows[col] = lo + (rows[col] - a) * (hi - lo) / (b - a)
    return rows


def _synthesize(
    pure: pd.DataFrame,
    shift_sets: list[np.ndarray],
    n: int,
    dose_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """n rows = pure base + sum over shift_sets of (dose * random shift from that set)."""
    rows, doses = [], []
    for _ in range(MAX_ATTEMPTS):
        if sum(len(r) for r in rows) >= n:
            break
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
    else:
        got = sum(len(r) for r in rows)
        if got < n:
            raise RuntimeError(
                f"only {got} of {n} synthetic rows fell inside the schema ranges after "
                f"{MAX_ATTEMPTS} attempts; the shifts or dose range are too large"
            )
    return pd.concat(rows).head(n), np.concatenate(doses)[:n]


def augment(df: pd.DataFrame, p: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Return real rows plus synthetic rows.

    Tags: `augmented` (bool), `aug_kind` ("", "class", "combo" or "widened_pure"),
    `aug_combo` (combination name, else "") and `aug_dose`.
    """
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    dose_range = tuple(p["dose_range"])
    # Shifts always come from real rows, measured against real pure milk.
    pure, shifts = pure_shifts(df)
    by_class = {c: g[FEATURES].to_numpy() for c, g in shifts.groupby("adulterant") if len(g) > 0}

    real = df.assign(augmented=False, aug_kind="", aug_combo="", aug_dose=np.nan)
    parts = [real]
    wp = p.get("widen_pure")
    if wp and wp.get("rows", 0) > 0:
        wide = widen(pure, wp["ranges"], wp["rows"], rng).assign(
            augmented=True, aug_kind="widened_pure", aug_combo="", aug_dose=np.nan
        )
        parts.append(wide)
        pure = pd.concat(
            [pure, wide.drop(columns=["augmented", "aug_kind", "aug_combo", "aug_dose"])]
        )
    for cls in STAGE2_CLASSES:
        need = p["target_per_class"] - int((df["adulterant"] == cls).sum())
        if need <= 0 or cls not in by_class:
            continue
        rows, dose = _synthesize(pure, [by_class[cls]], need, dose_range, rng)
        parts.append(
            rows.assign(
                adulterant=cls,
                is_adulterated=1,
                augmented=True,
                aug_kind="class",
                aug_combo="",
                aug_dose=dose,
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
                adulterant=name,
                is_adulterated=1,
                augmented=True,
                aug_kind="combo",
                aug_combo=name,
                aug_dose=dose,
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
    counts = out.groupby(["adulterant", "aug_kind"]).size()
    log.info("Rows by class and augmentation kind:\n%s", counts.unstack(fill_value=0))


if __name__ == "__main__":
    main()

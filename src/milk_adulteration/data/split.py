"""DVC stage `split`: stratified 70/15/15 train/validation/test split.

Stratified on the adulterant class (not just the binary label) so every rare
class lands in each split.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from milk_adulteration.config import load_params, resolve

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    rest, test = train_test_split(
        df, test_size=test_size, stratify=df["adulterant"], random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=rest["adulterant"],
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = load_params()["split"]

    df = pd.read_csv(resolve(p["input"]))
    out_dir = resolve(p["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in split(df, p["val_size"], p["test_size"], p["seed"]).items():
        part.sort_values("sample_id").to_csv(out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows, %d adulterated", name, len(part), part["is_adulterated"].sum())


if __name__ == "__main__":
    main()

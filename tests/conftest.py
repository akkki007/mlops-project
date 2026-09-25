import pandas as pd
import pytest


@pytest.fixture
def raw_rows() -> pd.DataFrame:
    """Three rows shaped like the source CSV: pure, water, melamine."""
    return pd.DataFrame(
        {
            "Sample_ID": ["S1", "S2", "S3"],
            "Collection_Date": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "Collection_Point": ["Farm Gate", "Processing Plant", "Retail Outlet"],
            "State": ["Karnataka", "Kerala", "Goa"],
            "Milk_Type": ["Toned", "Toned", "Skimmed"],
            "Fat_percent": [3.5, 2.1, 0.4],
            "SNF_percent": [8.9, 7.2, 9.0],
            "Specific_Gravity": [1.030, 1.020, 1.031],
            "pH": [6.7, 6.9, 6.6],
            "Freezing_Point_C": [-0.53, -0.45, -0.53],
            "Conductivity_mS_cm": [4.8, 4.2, 4.9],
            "Is_Adulterated": [0, 1, 1],
            "Adulterant_Detected": [None, "Water", "Melamine"],
            "Unused_Lab_Column": [1, 2, 3],
        }
    )


@pytest.fixture(scope="session")
def synthetic_clean() -> pd.DataFrame:
    """A clean-schema table: 300 pure rows plus 10 rows per Stage 2 class.

    Each adulterant shifts one reading so models have something to learn.
    Session-scoped for speed: tests must not modify it in place.
    """
    import numpy as np

    from milk_adulteration.config import STAGE2_CLASSES

    rng = np.random.default_rng(0)
    shifts = {
        "water": ("freezing_point", 0.04),
        "starch": ("snf_pct", 1.5),
        "urea": ("conductivity", 1.0),
        "detergent": ("ph", 1.0),
        "sodium_bicarbonate": ("ph", 0.6),
        "glucose": ("density", 0.004),
        "skimmed_milk_powder": ("fat_pct", -1.5),
        "other": ("snf_pct", 0.3),
    }
    classes = ["none"] * 300 + [c for c in STAGE2_CLASSES for _ in range(10)]
    n = len(classes)
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i:04d}" for i in range(n)],
            "collection_date": "2022-01-01",
            "collection_point": "Farm Gate",
            "state": "Kerala",
            "milk_type": "Toned",
            "fat_pct": rng.normal(3.4, 0.2, n),
            "snf_pct": rng.normal(8.8, 0.15, n),
            "density": rng.normal(1.030, 0.001, n),
            "ph": rng.normal(6.67, 0.05, n),
            "freezing_point": rng.normal(-0.53, 0.005, n),
            "conductivity": rng.normal(4.85, 0.15, n),
            "adulterant": classes,
        }
    )
    df["is_adulterated"] = (df["adulterant"] != "none").astype(int)
    for cls, (col, delta) in shifts.items():
        df.loc[df["adulterant"] == cls, col] += delta
    return df


AUG = {
    "seed": 0,
    "target_per_class": 30,
    "dose_range": [0.5, 1.5],
    "combos": [["water", "starch"]],
    "rows_per_combo": 10,
}
XGB = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}


@pytest.fixture(scope="session")
def trained(synthetic_clean):
    """A small cascade trained on `synthetic_clean`: (model, splits, folds)."""
    from sklearn.isotonic import IsotonicRegression

    from milk_adulteration.config import STAGE2_CLASSES
    from milk_adulteration.data.augment import augment
    from milk_adulteration.data.split import split
    from milk_adulteration.features import feature_matrix
    from milk_adulteration.models.cascade import Cascade
    from milk_adulteration.models.train import (
        FoldData,
        fit_stage1,
        fit_stage2,
        stage1_rows,
        stage2_rows,
    )

    parts = split(synthetic_clean, 0.2, 0.2, seed=0)
    folds = FoldData(parts["train"], AUG, k=3, seed=0)
    y, s, _ = folds.oof_stage1(XGB, exclude_other=True, seed=0)
    train_aug = augment(parts["train"], AUG)
    model = Cascade(
        stage1=fit_stage1(stage1_rows(train_aug, True), XGB, 0),
        calibrator=IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s, y),
        threshold=0.5,
        reject_risk=0.5,
        stage2=fit_stage2(stage2_rows(train_aug), XGB, 0),
        classes=list(STAGE2_CLASSES),
        features=list(feature_matrix(parts["train"].head(1)).columns),
    )
    return model, parts, folds

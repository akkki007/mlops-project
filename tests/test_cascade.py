import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from milk_adulteration.config import STAGE2_CLASSES
from milk_adulteration.data.augment import augment
from milk_adulteration.data.split import split
from milk_adulteration.evaluation import threshold_for_recall
from milk_adulteration.features import feature_matrix
from milk_adulteration.models.cascade import Cascade
from milk_adulteration.models.evaluate import evaluate, exit_criteria, render
from milk_adulteration.models.train import (
    FoldData,
    fit_stage1,
    fit_stage2,
    stage1_rows,
    stage2_rows,
)

AUG = {
    "seed": 0,
    "target_per_class": 30,
    "dose_range": [0.5, 1.5],
    "combos": [["water", "starch"]],
    "rows_per_combo": 10,
}
XGB = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}


@pytest.fixture(scope="module")
def trained(synthetic_clean):
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


def test_stage_row_filters(synthetic_clean):
    assert "other" not in set(stage1_rows(synthetic_clean, True)["adulterant"])
    assert "other" in set(stage1_rows(synthetic_clean, False)["adulterant"])
    aug = synthetic_clean.assign(aug_combo="")
    aug.loc[aug.index[-1], "aug_combo"] = "water+starch"
    assert len(stage2_rows(aug)) == int(synthetic_clean["is_adulterated"].sum()) - 1


def test_folds_never_score_synthetic_rows(trained):
    _, parts, folds = trained
    for real_tr, aug_tr, va in folds.folds:
        assert not va["sample_id"].str.startswith("AUG").any()
        assert set(va["sample_id"]).isdisjoint(real_tr["sample_id"])
        assert aug_tr["augmented"].any()


def test_predict_output(trained):
    model, parts, _ = trained
    out = model.predict(parts["test"])
    assert list(out.columns) == ["is_adulterated", "risk_score", "band", "adulterant", "confidence"]
    assert out["risk_score"].between(0, 1).all()
    flagged = out["is_adulterated"]
    assert (out.loc[~flagged, "band"] == "accept").all()
    assert set(out.loc[flagged, "band"]) <= {"retest", "reject"}
    assert out.loc[flagged, "adulterant"].isin(STAGE2_CLASSES).all()
    assert out.loc[~flagged, "adulterant"].isna().all()
    assert out.loc[flagged, "confidence"].between(0, 1).all()


def test_predict_rejects_invalid_readings(trained):
    model, parts, _ = trained
    with pytest.raises(Exception, match="ph"):
        model.predict(parts["test"].head(2).assign(ph=12.0))


def test_band_logic(trained):
    model, _, _ = trained
    bands = model.band(np.array([False, True, True]), np.array([0.9, 0.2, 0.7]))
    assert bands.tolist() == ["accept", "retest", "reject"]


def test_save_load_roundtrip(trained, tmp_path):
    model, parts, _ = trained
    model.save(tmp_path / "m.joblib")
    loaded = Cascade.load(tmp_path / "m.joblib")
    pd.testing.assert_frame_equal(loaded.predict(parts["test"]), model.predict(parts["test"]))


def test_evaluate_and_render(trained):
    model, parts, _ = trained
    r = evaluate(model, parts["test"], rounds=20, seed=0)
    assert r["end_to_end"]["all"]["positives"] == int(parts["test"]["is_adulterated"].sum())
    assert set(r["importance"]["stage1"]) == set(model.features)
    cv = {
        "stage1_cv_pr_auc": 1.0,
        "stage1_cv_pr_auc_no_aug": 0.9,
        "stage2_cv_macro_f1": 0.8,
        "stage2_cv_macro_f1_no_aug": 0.5,
    }
    assert len(exit_criteria(r, cv)) == 5
    md = render(r, cv, baselines=None)
    assert "exit criterion" in md and "Latency" in md


def test_threshold_midpoint_keeps_predictions_and_adds_margin():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.3, 0.9, 0.95])
    edge = threshold_for_recall(y, s, 1.0)
    mid = threshold_for_recall(y, s, 1.0, midpoint=True)
    assert edge == pytest.approx(0.9) and mid == pytest.approx(0.6)
    assert ((s >= edge) == (s >= mid)).all()

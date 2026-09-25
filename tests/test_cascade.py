import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from milk_adulteration.config import FEATURES, STAGE2_CLASSES
from milk_adulteration.evaluation import threshold_for_recall
from milk_adulteration.models.cascade import Cascade, raw_score_at_risk
from milk_adulteration.models.evaluate import evaluate, exit_criteria, render
from milk_adulteration.models.train import stage1_rows, stage2_rows


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
    assert list(out.columns) == [
        "is_adulterated",
        "risk_score",
        "band",
        "adulterant",
        "confidence",
        "unusual_readings",
    ]
    assert out["risk_score"].between(0, 1).all()
    flagged = out["is_adulterated"]
    usual = out["unusual_readings"].map(len) == 0
    assert (out.loc[~flagged & usual, "band"] == "accept").all()
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
    bands = model.band(
        np.array([False, True, True, False]),
        np.array([0.1, 0.2, 0.7, 0.1]),
        np.array([False, False, False, True]),
    )
    assert bands.tolist() == ["accept", "retest", "reject", "retest"]


def test_no_sample_is_accepted_with_high_risk(trained):
    """Regression: the flag threshold and the calibrator disagreed, so samples came
    back "accept" with risk 1.0. Blend pure and adulterated rows to probe the gap."""
    model, parts, _ = trained
    test = parts["test"]
    pure = test[test["is_adulterated"] == 0][FEATURES].to_numpy()
    bad = test[test["is_adulterated"] == 1][FEATURES].to_numpy()
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, (4000, 1))
    mix = (
        w * pure[rng.integers(0, len(pure), 4000)] + (1 - w) * bad[rng.integers(0, len(bad), 4000)]
    )
    out = model.predict(pd.DataFrame(mix, columns=FEATURES))
    accepted = out[out["band"] == "accept"]
    assert (accepted["risk_score"] < model.reject_risk).all()
    assert not (out["is_adulterated"] & out["adulterant"].isna()).any()


def test_raw_score_at_risk():
    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]
    )
    t = raw_score_at_risk(cal, 0.5)
    assert t == pytest.approx(0.25, abs=1e-6)
    assert cal.predict([t])[0] >= 0.5 > cal.predict([t - 1e-4])[0]
    assert raw_score_at_risk(cal, 1.5) == float("inf")


def test_unusual_readings_are_retested_not_accepted(trained):
    model, parts, _ = trained
    row = parts["test"][parts["test"]["is_adulterated"] == 0].head(1)
    lo = model.seen_ranges["snf_pct"][0]
    out = model.predict(row.assign(snf_pct=max(lo - 1.0, 3.0))).iloc[0]
    assert out["unusual_readings"] == ["snf_pct"]
    assert out["band"] in ("retest", "reject")


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

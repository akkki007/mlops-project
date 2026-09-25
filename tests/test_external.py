import pandas as pd
import pytest

from milk_adulteration.models.evaluate_external import (
    HORTVET_TO_CELSIUS,
    apply_units,
    evaluate_external,
    normalise_adulterant,
    parse_rename,
    predict_rows,
    render,
)
from milk_adulteration.models.train import check_device


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "none"),
        ("", "none"),
        ("Pure", "none"),
        ("Water", "water"),
        ("added water", "other"),  # unknown names are not guessed
        ("Sodium Bicarbonate", "sodium_bicarbonate"),
        ("sodium_bicarbonate", "sodium_bicarbonate"),
        ("Skimmed-Milk Powder", "skimmed_milk_powder"),
        ("Formalin", "other"),
    ],
)
def test_normalise_adulterant(raw, expected):
    assert normalise_adulterant(raw) == expected


def test_parse_rename():
    assert parse_rename("Fat=fat_pct, SNF = snf_pct") == {"Fat": "fat_pct", "SNF": "snf_pct"}
    assert parse_rename(None) == {}


def test_evaluate_external_with_labels(trained):
    model, parts, _ = trained
    df = parts["test"].drop(columns=["is_adulterated"]).copy()
    df.loc[df.index[0], "ph"] = 99  # one invalid row
    r = evaluate_external(model, df, label=None, adulterant="adulterant", rounds=20)
    assert r["rows"] == len(df) and r["rejected"] == 1
    assert set(r["stage1"]) == {"all", "detectable"}
    assert "macro_f1_present" in r["stage2"]
    assert "Stage 1" in render(r, "x.csv", model)


def test_evaluate_external_without_labels(trained):
    model, parts, _ = trained
    r = evaluate_external(model, parts["test"], label=None, adulterant=None)
    assert "stage1" not in r and 0 <= r["flag_rate"] <= 1
    assert "No labels" in render(r, "x.csv", model)


def test_evaluate_external_rejects_bad_label(trained):
    model, parts, _ = trained
    with pytest.raises(ValueError, match="0 and 1"):
        evaluate_external(model, parts["test"].assign(lab=2), label="lab", adulterant=None)


def test_check_device_cpu_is_noop():
    check_device("cpu")


def test_predict_rows_keeps_labels_and_order(trained):
    model, parts, _ = trained
    df = parts["test"].head(6).reset_index(drop=True).copy()
    df.loc[2, "ph"] = 99
    out = predict_rows(model, df)
    assert len(out) == 6 and out["sample_id"].tolist() == df["sample_id"].tolist()
    # The lab's own label columns are untouched; model answers are pred_*.
    assert out["is_adulterated"].tolist() == df["is_adulterated"].tolist()
    assert out["adulterant"].tolist() == df["adulterant"].tolist()
    assert out.loc[2, "status"] == "rejected" and "ph" in out.loc[2, "reject_reason"]
    scored = out[out["status"] == "scored"]
    assert scored["pred_band"].isin(["accept", "retest", "reject"]).all()
    assert out.loc[2, "pred_band"] is None or pd.isna(out.loc[2, "pred_band"])


def test_apply_units():
    df = pd.DataFrame({"freezing_point": [-0.55], "conductivity": [4800.0], "density": [28.0]})
    out = apply_units(df, freezing_point="H", conductivity="uS/cm", density="CLR")
    assert out["freezing_point"].iloc[0] == pytest.approx(-0.55 * HORTVET_TO_CELSIUS)
    assert out["conductivity"].iloc[0] == pytest.approx(4.8)
    assert out["density"].iloc[0] == pytest.approx(1.028)
    assert apply_units(df)["density"].iloc[0] == 28.0  # defaults change nothing


def test_template_csv_scores(trained):
    model, _, _ = trained
    df = pd.read_csv("examples/real_samples_template.csv", dtype={"sample_id": str})
    out = predict_rows(model, df)
    assert (out["status"] == "scored").all()
    assert out["sample_id"].tolist() == ["EX-001", "EX-002", "EX-003"]

import pytest

from milk_adulteration.models.evaluate_external import (
    evaluate_external,
    normalise_adulterant,
    parse_rename,
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

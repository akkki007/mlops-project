import pandas as pd
import pytest

from milk_adulteration.config import FEATURES, STAGE2_CLASSES
from milk_adulteration.data.schema import FIELD_SCHEMA
from milk_adulteration.data.validate import map_adulterant, prepare, split_valid


def test_prepare_keeps_only_v1_columns_and_maps_labels(raw_rows):
    df = prepare(raw_rows)
    assert "Unused_Lab_Column" not in df.columns
    assert set(FEATURES) <= set(df.columns)
    assert df["adulterant"].tolist() == ["none", "water", "other"]


def test_prepare_fails_on_missing_source_column(raw_rows):
    with pytest.raises(KeyError, match="Fat_percent"):
        prepare(raw_rows.drop(columns="Fat_percent"))


def test_map_adulterant_groups_weak_signal_as_other():
    out = map_adulterant(pd.Series(["Formalin", "Hydrogen Peroxide", "Vegetable Oil", None]))
    assert out.tolist() == ["other", "other", "other", "none"]


def test_stage2_classes():
    assert len(STAGE2_CLASSES) == 8
    assert "other" in STAGE2_CLASSES and "none" not in STAGE2_CLASSES


def test_clean_rows_pass(raw_rows):
    valid, rejected = split_valid(prepare(raw_rows))
    assert len(valid) == 3 and rejected.empty


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Fat_percent", 16.0),
        ("Specific_Gravity", 1.05),
        ("pH", 9.0),
        ("Freezing_Point_C", 0.1),
        ("Conductivity_mS_cm", 1.0),
        ("SNF_percent", None),
    ],
)
def test_out_of_range_or_null_row_is_rejected_with_reason(raw_rows, column, value):
    raw_rows.loc[1, column] = value
    valid, rejected = split_valid(prepare(raw_rows))
    assert valid["sample_id"].tolist() == ["S1", "S3"]
    assert rejected["sample_id"].tolist() == ["S2"]
    assert rejected["reject_reason"].iloc[0]


def test_label_disagreement_is_rejected(raw_rows):
    raw_rows.loc[0, "Is_Adulterated"] = 1  # says adulterated, but no adulterant named
    valid, rejected = split_valid(prepare(raw_rows))
    assert rejected["sample_id"].tolist() == ["S1"]
    assert "disagrees" in rejected["reject_reason"].iloc[0]


def test_unknown_adulterant_is_rejected(raw_rows):
    raw_rows.loc[1, "Adulterant_Detected"] = "Chalk"
    _, rejected = split_valid(prepare(raw_rows))
    assert rejected["sample_id"].tolist() == ["S2"]


def test_duplicate_sample_id_is_rejected(raw_rows):
    raw_rows.loc[2, "Sample_ID"] = "S1"
    _, rejected = split_valid(prepare(raw_rows))
    assert set(rejected["sample_id"]) == {"S1"}


def test_field_schema_validates_inference_payload():
    ok = pd.DataFrame([dict(zip(FEATURES, [3.5, 8.9, 1.03, 6.7, -0.53, 4.8], strict=True))])
    FIELD_SCHEMA.validate(ok)
    bad = ok.assign(ph=4.0)
    with pytest.raises(Exception, match="ph"):
        FIELD_SCHEMA.validate(bad)

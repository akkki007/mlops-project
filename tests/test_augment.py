import numpy as np
import pytest

from milk_adulteration.config import FEATURES, STAGE2_CLASSES
from milk_adulteration.data.augment import augment, pure_shifts
from milk_adulteration.data.schema import FIELD_SCHEMA

PARAMS = {
    "seed": 0,
    "target_per_class": 40,
    "dose_range": [0.5, 1.5],
    "combos": [["water", "starch"]],
    "rows_per_combo": 20,
}


@pytest.fixture
def aug(synthetic_clean):
    return augment(synthetic_clean, PARAMS)


def test_every_class_reaches_target(aug):
    single = aug[(aug["is_adulterated"] == 1) & (aug["aug_combo"] == "")]
    counts = single["adulterant"].value_counts()
    assert all(counts[c] == PARAMS["target_per_class"] for c in STAGE2_CLASSES)


def test_real_rows_kept_unchanged_and_tagged(synthetic_clean, aug):
    real = aug[~aug["augmented"]]
    assert len(real) == len(synthetic_clean)
    assert real[FEATURES].equals(synthetic_clean[FEATURES])


def test_synthetic_rows_are_valid_and_uniquely_named(aug):
    synth = aug[aug["augmented"]]
    FIELD_SCHEMA.validate(synth)
    assert synth["sample_id"].str.startswith("AUG").all()
    assert aug["sample_id"].is_unique
    assert synth["aug_dose"].between(0.5, 1.5).all()


def test_combos_tagged_for_stage1_only(aug):
    combo = aug[aug["aug_combo"] != ""]
    assert len(combo) == 20
    assert (combo["adulterant"] == "water+starch").all()
    assert (combo["is_adulterated"] == 1).all()


def test_synthetic_rows_carry_the_class_signature(synthetic_clean, aug):
    # The fixture's water rows have freezing point +0.04 over pure.
    pure_fp = synthetic_clean.loc[synthetic_clean["adulterant"] == "none", "freezing_point"].mean()
    water = aug[aug["augmented"] & (aug["adulterant"] == "water")]
    assert water["freezing_point"].mean() - pure_fp == pytest.approx(0.04, abs=0.01)


def test_deterministic(synthetic_clean):
    a, b = augment(synthetic_clean, PARAMS), augment(synthetic_clean, PARAMS)
    assert np.allclose(a[FEATURES], b[FEATURES])
    c = augment(synthetic_clean, PARAMS, seed=1)
    assert not np.allclose(a[FEATURES], c[FEATURES])


def test_pure_shifts_are_relative_to_milk_type(synthetic_clean):
    _, shifts = pure_shifts(synthetic_clean)
    assert len(shifts) == int(synthetic_clean["is_adulterated"].sum())
    assert shifts.loc[shifts["adulterant"] == "starch", "snf_pct"].mean() == pytest.approx(
        1.5, abs=0.15
    )

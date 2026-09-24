import numpy as np
import pandas as pd
import pytest

from milk_adulteration.config import FEATURES
from milk_adulteration.features import DERIVED_FEATURES, expected_density, feature_matrix
from milk_adulteration.models import rules


def test_expected_density_matches_richmond():
    # SNF 8.5, fat 3.0 -> CLR = (8.5 - 0.66 - 0.72) / 0.25 = 28.48
    assert expected_density(pd.Series([3.0]), pd.Series([8.5])).iloc[0] == pytest.approx(1.02848)


def test_feature_matrix_columns(synthetic_clean):
    assert list(feature_matrix(synthetic_clean, derived=False).columns) == FEATURES
    X = feature_matrix(synthetic_clean)
    assert list(X.columns) == FEATURES + DERIVED_FEATURES
    assert not X.isna().any().any()
    assert "fp_deviation" not in synthetic_clean.columns  # input left untouched


def _sample(**overrides):
    base = dict(
        milk_type="Toned",
        fat_pct=3.4,
        snf_pct=8.8,
        density=1.030,
        ph=6.67,
        freezing_point=-0.53,
        conductivity=4.85,
    )
    return pd.DataFrame([{**base, **overrides}])


def test_normal_milk_breaks_no_rule():
    assert rules.score(_sample()).iloc[0] == 0


@pytest.mark.parametrize(
    ("overrides", "rule"),
    [
        ({"fat_pct": 2.9}, "fat_below_min"),
        ({"milk_type": "Full Cream", "fat_pct": 5.5, "snf_pct": 9.2}, "fat_below_min"),
        ({"snf_pct": 8.4}, "snf_below_min"),
        ({"freezing_point": -0.50}, "freezing_point_out_of_range"),
        ({"density": 1.024}, "density_out_of_range"),
        ({"ph": 7.2}, "ph_out_of_range"),
        ({"conductivity": 6.0}, "conductivity_out_of_range"),
    ],
)
def test_each_rule_fires(overrides, rule):
    b = rules.breaches(_sample(**overrides)).iloc[0]
    assert b[rule] and b.sum() == 1


def test_unknown_milk_type_only_gets_range_rules():
    b = rules.breaches(_sample(milk_type="Camel", fat_pct=0.1, snf_pct=1.0)).iloc[0]
    assert not b["fat_below_min"] and not b["snf_below_min"]


def test_score_is_breach_count():
    s = rules.score(_sample(ph=7.5, conductivity=6.5))
    assert s.dtype == np.float64 and s.iloc[0] == 2

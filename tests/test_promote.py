import pandas as pd
import pytest

from milk_adulteration.data.validate import split_valid_readings
from milk_adulteration.models.promote import decide

GATES = {"min_recall": 0.95, "min_precision": 0.80}
GOOD = {"recall": 1.0, "precision": 0.9, "pr_auc": 0.97}


def test_first_model_promotes_if_it_passes_gates():
    assert decide(GOOD, None, GATES) == (True, "no production model yet")


@pytest.mark.parametrize(("metric", "value"), [("recall", 0.9), ("precision", 0.7)])
def test_gates_block_promotion(metric, value):
    ok, reason = decide({**GOOD, metric: value}, None, GATES)
    assert not ok and metric in reason


def test_regression_blocks_and_ties_promote():
    current = {"recall": 1.0, "precision": 0.95, "pr_auc": 0.98}
    ok, reason = decide(GOOD, current, GATES)
    assert not ok and "pr_auc" in reason
    assert decide({**GOOD, "pr_auc": 0.98}, current, GATES)[0]
    # Precision is a gate, not a comparison.
    assert decide({**GOOD, "pr_auc": 0.99}, current, GATES)[0]


def test_split_valid_readings_row_level_reasons():
    df = pd.DataFrame(
        {
            "fat_pct": [3.4, "abc", 3.1, 99],
            "snf_pct": [8.7, 8.7, None, 8.7],
            "density": [1.03] * 4,
            "ph": [6.7] * 4,
            "freezing_point": [-0.53] * 4,
            "conductivity": [4.8] * 4,
        }
    )
    valid, rejected = split_valid_readings(df)
    assert valid.index.tolist() == [0]
    reasons = rejected["reject_reason"].tolist()
    assert "not a number" in reasons[0] and "snf_pct" in reasons[1] and "in_range" in reasons[2]


def test_split_valid_readings_missing_column():
    with pytest.raises(KeyError, match="ph"):
        split_valid_readings(pd.DataFrame({"fat_pct": [3.4]}))

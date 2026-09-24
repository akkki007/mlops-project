import numpy as np
import pytest

from milk_adulteration.evaluation import (
    binary_metrics,
    bootstrap_ci,
    multiclass_metrics,
    threshold_for_recall,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])


@pytest.mark.parametrize(("target", "expected"), [(1.0, 0.4), (0.75, 0.7), (0.5, 0.8)])
def test_threshold_for_recall_picks_highest_meeting_target(target, expected):
    thr = threshold_for_recall(Y, S, target)
    assert thr == pytest.approx(expected)
    assert binary_metrics(Y, S, thr)["recall"] >= target


def test_binary_metrics_counts():
    m = binary_metrics(Y, S, 0.4)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (4, 1, 0, 3)
    assert m["precision"] == pytest.approx(0.8)
    assert m["flag_rate"] == pytest.approx(5 / 8)


def test_bootstrap_ci_brackets_point_estimate():
    ci = bootstrap_ci(Y, S, 0.4, rounds=200, seed=0)
    lo, hi = ci["precision"]
    assert 0 <= lo <= 0.8 <= hi <= 1


def test_multiclass_metrics_keeps_absent_labels():
    m = multiclass_metrics(["a", "b"], ["a", "a"], labels=["a", "b", "c"])
    assert m["per_class_f1"]["c"] == 0.0
    assert len(m["confusion"]) == 3

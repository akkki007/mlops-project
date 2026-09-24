import pytest

from milk_adulteration.config import STAGE2_CLASSES
from milk_adulteration.data.split import split
from milk_adulteration.models.baselines import render, run_stage1, run_stage2, summarise

PARAMS = {
    "seed": 0,
    "target_recall": 0.95,
    "bootstrap_rounds": 50,
    "stage2_cv_folds": 3,
    "stage2_cv_repeats": 2,
    "mlflow": {"experiment": "test"},
    "logreg": {"C": 1.0, "max_iter": 500},
    "random_forest": {"n_estimators": 20, "min_samples_leaf": 1},
}


@pytest.fixture
def splits(synthetic_clean):
    return split(synthetic_clean, val_size=0.2, test_size=0.2, seed=0)


def test_split_is_disjoint_stratified_and_complete(synthetic_clean, splits):
    ids = [set(s["sample_id"]) for s in splits.values()]
    assert sum(map(len, ids)) == len(synthetic_clean)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for part in splits.values():
        assert set(STAGE2_CLASSES) <= set(part["adulterant"])


def test_split_is_deterministic(synthetic_clean):
    a = split(synthetic_clean, 0.2, 0.2, seed=1)["test"]["sample_id"].tolist()
    b = split(synthetic_clean, 0.2, 0.2, seed=1)["test"]["sample_id"].tolist()
    assert a == b


def test_baselines_end_to_end(splits):
    s1, s2 = run_stage1(splits, PARAMS), run_stage2(splits, PARAMS)
    assert set(s1) == {
        "fssai_rules",
        "logreg",
        "logreg+derived",
        "random_forest",
        "random_forest+derived",
    }
    for res in s1.values():
        for view in ("all", "detectable"):
            assert res[view]["val"]["recall"] >= 0.95 or res["model"] is None
    # The shifts are large, so the forest should separate most adulterated rows.
    assert s1["random_forest+derived"]["all"]["test"]["pr_auc"] > 0.8
    assert 0 <= s2["random_forest"]["cv_macro_f1_mean"] <= 1

    md = render(summarise(s1, s2), PARAMS)
    assert "Detectable adulterants only" in md and "| fssai_rules |" in md

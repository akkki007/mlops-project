import numpy as np
import pytest

from milk_adulteration.models.stress_test import add_water, pure_samples, render, run

P = {
    "seed": 0,
    "samples": 50,
    "pure_ranges": {
        "fat_pct": [3.0, 5.0],
        "snf_pct": [8.3, 9.2],
        "density": [1.027, 1.033],
        "ph": [6.6, 6.8],
        "freezing_point": [-0.55, -0.52],
        "conductivity": [4.0, 5.5],
    },
    "water": {
        "fat_pct": 0.0,
        "snf_pct": 0.0,
        "density": 0.998,
        "ph": 7.0,
        "freezing_point": 0.0,
        "conductivity": 0.5,
    },
    "water_fractions": [0.1, 0.3],
    "sweep_points": 3,
}


def test_add_water_follows_mixing_law():
    df = pure_samples(P["pure_ranges"], 5, np.random.default_rng(0))
    mixed = add_water(df, P["water"], 0.2)
    assert mixed["fat_pct"].to_numpy() == pytest.approx(0.8 * df["fat_pct"].to_numpy())
    assert mixed["freezing_point"].to_numpy() == pytest.approx(
        0.8 * df["freezing_point"].to_numpy()
    )
    assert (mixed["density"] < df["density"]).all()


def test_run_and_render(trained):
    model, parts, _ = trained
    train = parts["train"]
    r = run(model, train[train["adulterant"] == "none"], P)
    assert 0 <= r["pure_false_alarm_rate"] <= 1
    assert set(r["water"]) == {"0.1", "0.3"}
    assert len(r["sweep"]["ph"]) == 3
    assert "not real samples" in render(r, P)

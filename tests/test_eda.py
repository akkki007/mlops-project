from milk_adulteration.data.validate import prepare, split_valid
from milk_adulteration.eda import render, summarise


def test_summarise_and_render(raw_rows):
    df, _ = split_valid(prepare(raw_rows))
    # The source CSV round-trip stores dates as strings; EDA must not care.
    s = summarise(df)
    assert s["rows"] == 3 and s["positives"] == 2
    assert set(s["shift_z"]) == {"water", "other"}
    md = render(s)
    assert "synthetic" in md and "| water |" in md

# Milk Adulteration Detection

An MLOps service that flags adulterated milk from six routine readings (fat %, SNF %,
density, pH, freezing point, conductivity) using a two-stage XGBoost cascade.
Stage 1 scores whether a sample is adulterated; Stage 2 names the likely adulterant.

## Status

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | Data in: download, EDA, Pandera schema, DVC | Done |
| 2 | Baselines: FSSAI rules, logistic regression, Random Forest | Next |
| 3 | Two-stage XGBoost, augmentation, calibration | |
| 4 | FastAPI service, Docker, GitHub Actions | |
| 5 | React dashboard, PostgreSQL, Prometheus + Grafana | |
| 6 | Validation on real data, SHAP report, write-up | |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
dvc repro          # ingest -> validate -> eda
pytest -q
```

## Data pipeline

| Stage | Module | Output |
| --- | --- | --- |
| `ingest` | `milk_adulteration.data.ingest` | `data/raw/milk_combined_full_dataset.csv`, checked against the SHA-256 in `params.yaml` |
| `validate` | `milk_adulteration.data.validate` | `data/interim/milk_clean.csv` (six features, IDs, labels) and `milk_rejected.csv` (failed rows with a reason) |
| `eda` | `milk_adulteration.eda` | [`reports/eda.md`](reports/eda.md), `reports/eda_summary.json` (DVC metrics) |

The source is the CC0 [Indian Milk Adulteration Detection Dataset](https://github.com/ommadav/Indian-Milk-Adulteration-Detection-Dataset)
(2,500 rows, **synthetic**). Only the six field-level readings are kept; the lab-only
columns (HPLC, GC-MS, strip tests) are dropped so the model matches what a collection
center can measure. `Specific_Gravity` is used as density.

**Labels.** `adulterant` maps the 11 source adulterants to 8 v1 classes: water, starch,
urea, detergent, sodium_bicarbonate, glucose, skimmed_milk_powder, and `other`
(formalin, hydrogen peroxide, vegetable oil, melamine). Pure rows are `none`.

**Schema** (`milk_adulteration.data.schema`). `FIELD_SCHEMA` checks the six inputs
(no nulls, plausible ranges: fat and SNF 0–15 %, density 1.000–1.040, pH 5.5–8.5,
freezing point −1.0–0 °C, conductivity 2–10 mS/cm) and will be reused by the API.
`CLEAN_SCHEMA` adds unique sample IDs, dates and label consistency.

No DVC remote is configured yet; add one with `dvc remote add -d storage <url>` and
`dvc push` to share cached data.

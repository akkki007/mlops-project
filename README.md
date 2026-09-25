# Milk Adulteration Detection

An MLOps service that flags adulterated milk from six routine readings (fat %, SNF %,
density, pH, freezing point, conductivity) using a two-stage XGBoost cascade.
Stage 1 scores whether a sample is adulterated; Stage 2 names the likely adulterant.

> **Not yet tested on real milk.** All results are on a synthetic dataset. A
> physics stress test ([`reports/stress_test.md`](reports/stress_test.md)) found the
> first model flagged 39% of pure cow milk across published normal ranges; training
> now widens pure milk to those ranges, which brings that to 0.6% (buffalo milk: 25%).
> To score real samples on your own computer, follow [RUN_LOCALLY.md](RUN_LOCALLY.md).
> [GUIDE.md](GUIDE.md) has the details and how to train on a GPU (faster, not more
> accurate).

## Status

| Week | Milestone | Status |
| --- | --- | --- |
| 1 | Data in: download, EDA, Pandera schema, DVC | Done |
| 2 | Baselines: FSSAI rules, logistic regression, Random Forest | Done |
| 3 | Two-stage XGBoost, augmentation, calibration | Done: exit criterion met |
| 4 | FastAPI service, Docker, GitHub Actions | Done |
| 5 | React dashboard, PostgreSQL, Prometheus + Grafana | Next |
| 6 | Validation on real data, SHAP report, write-up | |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
dvc repro          # ingest -> validate -> eda -> split -> baselines -> augment -> train -> evaluate -> promote
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs
pytest -q
uvicorn milk_adulteration.api.app:app --reload   # API on :8000, docs at /docs
```

Install only what you need: `pip install -e .` (load and run the model),
`.[serve]` (API), `.[train]` (pipeline), `.[dev]` (everything plus tests).

## Data pipeline

| Stage | Module | Output |
| --- | --- | --- |
| `ingest` | `milk_adulteration.data.ingest` | `data/raw/milk_combined_full_dataset.csv`, checked against the SHA-256 in `params.yaml` |
| `validate` | `milk_adulteration.data.validate` | `data/interim/milk_clean.csv` (six features, IDs, labels) and `milk_rejected.csv` (failed rows with a reason) |
| `eda` | `milk_adulteration.eda` | [`reports/eda.md`](reports/eda.md), `reports/eda_summary.json` (DVC metrics) |
| `split` | `milk_adulteration.data.split` | `data/processed/{train,val,test}.csv`: 70/15/15, stratified on adulterant class, seed 42 |
| `baselines` | `milk_adulteration.models.baselines` | [`reports/baselines.md`](reports/baselines.md), `reports/baselines.json`, one MLflow run per model |
| `augment` | `milk_adulteration.data.augment` | `data/processed/train_aug.csv`: every Stage 2 class grown to 300 rows, plus water+starch and water+urea rows |
| `train` | `milk_adulteration.models.train` | `models/cascade.joblib`, `reports/train_cv.json`; Optuna trials as nested MLflow runs |
| `evaluate` | `milk_adulteration.models.evaluate` | [`reports/model.md`](reports/model.md), `reports/model_metrics.json` |
| `promote` | `milk_adulteration.models.promote` | Registers the model in MLflow; moves the `production` alias if it earns it. Writes `models/model_info.json` and [`reports/promotion.json`](reports/promotion.json) |

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

## Baselines (Week 2)

Full tables with 95% bootstrap CIs are in [`reports/baselines.md`](reports/baselines.md).
Results are on synthetic data.

- **FSSAI rules** (`models/rules.py`, fixed published thresholds): precision 1.00 but
  recall 0.53 on test.
- **Logistic regression** fails: the adulterant signals point in opposite directions
  (water lowers density, glucose raises it), which a linear model can't separate.
- **Random Forest + derived features**: test PR-AUC 0.93. Formalin and H₂O₂ (class
  `other`) look like pure milk, so reaching recall ≥ 0.95 on *all* adulterants means
  flagging most samples. On the detectable adulterants (`other` removed) it reaches
  recall 1.00 and precision 1.00 (12 of 12, no false alarms).
- **Stage 2** (which adulterant): best CV macro-F1 is 0.77 (Random Forest), under the
  0.80 target. Starch and skimmed milk powder both raise SNF and get confused.

Derived features (`features.py`): fat/SNF ratio, density residual versus Richmond's
formula, and freezing-point deviation from −0.52 °C.

MLflow runs are tagged with the SHA-256 of each split file and the git commit.
Tracking is a local SQLite store (`mlflow.db`, not committed).

## Two-stage cascade (Week 3)

Full results are in [`reports/model.md`](reports/model.md). Results are on synthetic data.

```python
from milk_adulteration.models.cascade import Cascade

model = Cascade.load("models/cascade.joblib")
model.predict(df)  # is_adulterated, risk_score, band, adulterant, confidence
```

- **Recall target scope.** The recall target covers the *detectable* adulterants.
  `other` (formalin, H₂O₂, vegetable oil, melamine) looks like pure milk in the six
  readings and goes to a confirmatory lab test. Stage 1 does not train on it
  (`train.stage1_exclude_other`).
- **Augmentation** (`data/augment.py`) uses "shift transplant": a random pure
  training row plus the measured shift of a real adulterated training row, scaled
  by a dose of 0.5–1.5×. A physical mixing model was not used because the source
  data does not follow one: its water rows imply 12–40% dilution depending on
  which reading you use.
- **Tuning.** Optuna runs 30 trials per stage on 5-fold CV over real training
  rows. Each fold is augmented from its own training part and scored only on real
  rows.
- **Calibration and threshold.** Isotonic calibration is fitted on out-of-fold
  scores of real rows. Samples are flagged on the raw Stage 1 score, at a
  threshold picked on validation for recall ≥ 0.95 and moved halfway to the next
  lower score for a safety margin. It is lowered further if needed so that no
  sample with calibrated risk ≥ 0.5 is ever accepted. Flagged samples with risk
  ≥ 0.5 are `reject`, the rest `retest`. Unflagged samples with readings outside
  the training ranges are also `retest`.
- **Pure-milk widening.** 2,000 extra pure rows stretched onto typical published
  cow and buffalo ranges (`augment.widen_pure`). See GUIDE.md for why and what it
  costs.

| Test metric (detectable adulterants) | Result | Target |
| --- | --- | --- |
| Stage 1 recall | 1.00 (12/12) | ≥ 0.95 |
| Stage 1 precision | 1.00 (0 false alarms) | ≥ 0.80 |
| Stage 2 macro-F1 (test / train CV) | 0.62 (19 samples) / 0.80 | ≥ 0.80 |
| Single-sample p95 latency | ~14 ms | < 100 ms |

On all adulterants, including `other`, Stage 1 catches 16 of 19 with no false
alarms. The Random Forest baseline needed 213 false alarms to catch all 19.

## Prediction API (Week 4)

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | One sample → `is_adulterated`, `risk_score`, `band`, `adulterant`, `confidence`, `unusual_readings`, `model_version`. Out-of-range or non-numeric readings get a 422 with the reason. Readings outside what the model saw in training are listed in `unusual_readings` and never accepted. |
| `POST /predict/batch` | Up to 10,000 samples as JSON, a CSV body (`text/csv`) or a CSV upload (multipart field `file`). Per-row results plus a summary; invalid rows are returned as `rejected` with a reason, never scored. |
| `GET /model/info` | Registry version, MLflow run ID, data hashes, threshold, test metrics |
| `GET /health` | Liveness |
| `GET /metrics` | Prometheus: request counts and latency by route, predictions by band, adulterant and collection point, rejected rows, risk-score histogram |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "sample_id": "S-1", "collection_point": "Farm Gate", "fat_pct": 2.2, "snf_pct": 6.9,
  "density": 1.019, "ph": 6.9, "freezing_point": -0.46, "conductivity": 4.2}'
curl -s localhost:8000/predict/batch -F file=@samples.csv
```

**Docker.** Build after `dvc repro`, because the image bundles `models/cascade.joblib`:

```bash
docker compose up --build        # or: docker build -t milk-adulteration-api .
python scripts/smoke_test.py     # checks every endpoint against a running API
```

The image runs as a non-root user with a health check. It uses `xgboost-cpu`
(same library, without the ~300 MB CUDA dependency), with runtime pins in
`requirements/serve.txt` that must match the training versions because the
model is a pickle. Measured in the container: `/predict` p95 ≈ 17 ms; a
10,000-row CSV batch takes ≈ 1.1 s.

**Promotion.** A model gets the MLflow `production` alias only if, on detectable
test adulterants, it meets recall ≥ 0.95 and precision ≥ 0.80, and is no worse
than the current production model on PR-AUC and recall. Ties promote, so a
retrain on new data can replace the old model. Set `MLFLOW_TRACKING_URI` to use a
shared MLflow server instead of the local `mlflow.db`.

**CI** (`.github/workflows/ci.yml`): lint and tests (including schema and API
tests) → full `dvc repro` → build the image, smoke-test the container → push to
`ghcr.io/<owner>/<repo>/api` from `main` or a `v*` tag, only if promoted.

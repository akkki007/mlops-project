# Guide: training on a GPU and testing on real milk samples

This guide is for anyone retraining the model on their own machine, with or
without an NVIDIA GPU. Read the first section before spending GPU time.

## 1. Where the model stands today

**The model has not been tested on real milk samples.** Every number in
[`reports/model.md`](reports/model.md) comes from the *synthetic* ommadav dataset:
2,500 rows simulated from FSSAI/BIS standards, with only 124 adulterated rows.

| What we measured (synthetic test set) | Result |
| --- | --- |
| Stage 1 recall / precision, detectable adulterants | 1.00 / 1.00 (12 of 12 caught, 0 false alarms) |
| Stage 1 recall, all adulterants | 0.84 (16 of 19; formalin and H₂O₂ look like pure milk) |
| Stage 2 macro-F1 (which adulterant) | 0.62 on test (19 samples), 0.80 in tuning CV, about 0.77 across CV seeds |

Treat these as "the pipeline works", not "the model works on milk":

- The synthetic data is far cleaner than real milk. Its pure samples sit exactly
  inside the FSSAI limits, so the classes separate almost perfectly.
- The test set is tiny: 12 detectable adulterated samples. The 95% confidence
  intervals in the report are wide for Stage 2.
- Augmentation copies the adulterant patterns of the synthetic generator. If real
  adulterated milk behaves differently, the model has never seen it.

### A physics stress test found a real-world problem, now partly fixed

No real labelled samples with all six readings were reachable: the Kaggle
Lactoscanner data needs a login, the MADS repo doesn't ship its CSV, and the other
public set is synthetic. So [`reports/stress_test.md`](reports/stress_test.md)
(`dvc repro stress_test`) probes the model with *generated* samples based on dairy
physics instead.

**What it found.** The first model flagged **39% of pure cow milk** and **83% of
pure buffalo milk** spread across published normal ranges. Synthetic pure milk never
freezes below −0.540 °C, so the model flagged anything colder, and real milk often
is colder.

**The fix.** Training now adds 2,000 pure rows stretched onto typical published
cow and buffalo ranges (`augment.widen_pure` in `params.yaml`), and the synthetic
adulterated rows are built on those wider bases too.

| Stress test (generated samples) | Before | After |
|---|---|---|
| Pure cow milk flagged | 38.7% | 0.6% |
| Pure buffalo milk flagged | 83% | 25% (about 3% when fat ≥ 6%) |
| 3% / 5% added water flagged | 81% / 100% | 40% / 73% |
| 10% added water or more flagged | 100% | 100% |
| Synthetic test, detectable adulterants | 12/12, 0 false alarms | unchanged |

Read these carefully:

- **Buffalo milk with low fat and high SNF** (for example 5% fat, 9.8% SNF) is still
  often flagged. Its readings look like starch or milk powder, which raise SNF with
  fat unchanged. Without knowing whether the milk is cow or buffalo, the six readings
  can't tell them apart. A milk-type input would fix this; it's not in v1.
- **Very light watering (under 5%) is caught less often.** Wider pure-milk ranges
  overlap with lightly watered milk. That's the price of fewer false alarms.
- **The widening and the stress test use similar published ranges,** so the
  "after" column shows the fix took effect. It is not proof of accuracy on real milk.
- The ranges in `params.yaml` are approximate literature values. **Replace them
  with your own lab's pure-milk data** when you have it.

Readings outside anything the model saw in training (for example SNF below 4.9%)
are never accepted. They come back as `retest` with `unusual_readings` listing them,
because a tree model can't extrapolate.

The only way to know how accurate the model is on real milk is to score it on real,
lab-confirmed samples. Section 5 shows how.

## 2. Will a GPU make it more accurate?

**No.** A GPU makes XGBoost *train faster*; it does not make the model more accurate.
Accuracy here is limited by the data, not by compute:

- The whole training set is about 1,750 real rows plus about 2,700 augmented rows.
  On a laptop CPU, the full `train` stage (60 Optuna trials × 5 folds) takes
  about 2 minutes.
- On data this small, a GPU is often *no faster* than a CPU, because the fixed
  cost of moving data to the GPU dominates.
- Results on GPU differ slightly from CPU: GPU histogram building is not
  bit-for-bit identical. Expect small metric changes, not improvements.

Where a GPU does help:

- **A much larger hyperparameter search**, for example 500+ Optuna trials per
  stage, or repeated CV over many seeds to get tighter confidence intervals.
- **Much larger data.** If real datasets grow to hundreds of thousands of rows.
- **Future models on spectra.** The PRD mentions a 1D-CNN on NIR spectra; that's
  where a GPU really matters.

What would actually improve accuracy, in order:

1. **Real labelled samples** from a dairy or college lab (even 200), used first to
   *test* the model (section 5), then to retrain.
2. **More real adulterated samples per class.** Today some classes have 5–9 real
   training rows.
3. **Better augmentation** calibrated on real adulterated milk instead of the
   synthetic generator.

## 3. Set up a GPU machine

Requirements: an NVIDIA GPU with a recent driver, Linux or WSL2, Python 3.11, git.

```bash
nvidia-smi                      # must list your GPU and a driver version

git clone https://github.com/akkki007/mlops-project.git
cd mlops-project
git checkout claude/work-prioritization-5tfpx6   # or main, once merged

python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` installs the full `xgboost` package, which includes CUDA
support on Linux x86-64. (The Docker image uses `xgboost-cpu`, which has no CUDA
support; don't install that one for GPU training.)

Check that XGBoost can see the GPU:

```bash
python - <<'EOF'
from milk_adulteration.models.train import check_device
check_device("cuda")
print("GPU OK")
EOF
```

If this raises "no CUDA support", run `pip uninstall -y xgboost-cpu && pip install xgboost`.
If it raises "found no usable GPU", fix the driver first (`nvidia-smi` must work).

## 4. Train on the GPU

The device is a pipeline parameter, `train.device` in [`params.yaml`](params.yaml).
Training refuses to start if you ask for `cuda` and no GPU is usable. XGBoost itself
would otherwise fall back to CPU silently.

**Option A: a one-off experiment** (does not change `params.yaml`):

```bash
dvc exp run -S train.device=cuda
dvc exp show --only-changed          # compare with the committed CPU run
```

**Option B: make it the default.** Edit `params.yaml`:

```yaml
train:
  device: cuda
```

then run `dvc repro`. Either way the pipeline downloads the data, validates it,
trains, evaluates and writes:

| File | What it is |
| --- | --- |
| `models/cascade.joblib` | The trained two-stage model |
| `reports/model.md` | Test results, Week 3 exit criterion, augmentation comparison |
| `reports/train_cv.json` | Best hyperparameters and CV scores |
| `reports/promotion.json` | Whether it was promoted to `production` in MLflow |

Browse all runs and Optuna trials with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. The training run is tagged
`train_device=cuda`.

**A bigger search**, which is the main reason to use a GPU here:

```bash
dvc exp run -S train.device=cuda -S train.n_trials_stage1=300 -S train.n_trials_stage2=300
```

Only keep a bigger search if `reports/model.md` improves on the **cross-validation**
numbers. The test set is too small to separate small improvements from luck, and
choosing among many runs by test score overfits the test set.

### Serving a GPU-trained model

The saved model is always switched back to CPU before it's written, so the API and
Docker image (which have no GPU) load it normally. Two things must match between
the training machine and the serving image, because the model file is a pickle:

- `numpy`, `scikit-learn` and `xgboost` versions must equal the pins in
  [`requirements/serve.txt`](requirements/serve.txt). Check with
  `pip list | grep -Ei "numpy|scikit-learn|xgboost"`.
- Build the image on the machine that trained the model (`docker compose up --build`),
  or copy `models/cascade.joblib` and `models/model_info.json` to where you build it.

`models/` is not in git. To share trained models between machines, configure a DVC
remote (`dvc remote add -d storage <s3/gdrive/ssh url>`, then `dvc push`) or copy the
two files.

## 5. Test the model on real samples

This is the most important step, and it needs no GPU. **[RUN_LOCALLY.md](RUN_LOCALLY.md)
is the step-by-step version for a lab** (install on any laptop, CSV template, unit
conversions, per-sample results). The short version:

**What the CSV needs:** one row per sample with all six readings in these units:

| Column | Unit | Allowed range |
| --- | --- | --- |
| `fat_pct` | % | 0–15 |
| `snf_pct` | % | 3–15 |
| `density` | g/mL (specific gravity is fine) | 1.000–1.040 |
| `ph` | pH | 5.5–8.5 |
| `freezing_point` | °C (negative, about −0.52 for pure milk) | −1.0 to 0 |
| `conductivity` | mS/cm | 2–10 |

For accuracy figures, add the lab result: a 0/1 column (1 = adulterated) and/or a
column naming the adulterant (`Water`, `Starch`, `Urea`, `Detergent`,
`Sodium Bicarbonate`, `Glucose`, `Skimmed Milk Powder`; anything else counts as
`other`, and blank or `pure` means pure).

```bash
python -m milk_adulteration.models.evaluate_external real_samples.csv \
    --label is_adulterated --adulterant adulterant \
    --rename "Fat=fat_pct,SNF=snf_pct,Density=density,pH=ph,FP=freezing_point,EC=conductivity" \
    --out reports/external_real.md --predictions reports/external_real_predictions.csv
```

It writes `reports/external_real.md` (and `.json`), and with `--predictions` a CSV
with the model's answer for every sample. The report contains:

- recall and precision with confidence intervals, for all adulterants and for
  detectable ones;
- per-adulterant F1;
- how many rows failed validation and why. Rows outside the ranges above are
  never scored.

It only scores the file; it never trains on it, so it's a fair check. Without
labels, it reports the flag rate and bands only.

**How to read the results:**

- **Recall well below 0.95 on detectable adulterants** means real adulterated milk
  doesn't look like the synthetic data. Retrain with real samples before relying
  on the model.
- **Many false alarms on pure samples** usually means real pure milk varies more
  than the synthetic data. Real milk varies by breed, season and region.
- **Fewer than about 30 adulterated samples** gives wide confidence intervals.
  Treat the numbers as a sanity check, not a verdict.

**If your dataset lacks some readings** (many Lactoscanner exports have fat, SNF,
density, protein, lactose and added water, but no pH or conductivity), this model
can't score it. All six readings are required. Retraining on a different feature
set is a code change; open an issue with the column list.

**Before retraining on real samples**, keep a portion aside that training never
sees, and report results on it separately from the synthetic numbers.

## 6. Troubleshooting

| Problem | Fix |
| --- | --- |
| `train.device='cuda' but this XGBoost build has no CUDA support` | `pip uninstall -y xgboost-cpu && pip install xgboost` |
| `found no usable GPU` | `nvidia-smi` must work; on WSL2, install the Windows NVIDIA driver with WSL support |
| `Checksum mismatch` in `ingest` | The upstream CSV changed. Review it, then update `ingest.sha256` in `params.yaml` |
| API fails to load the model (`AttributeError`/`ModuleNotFoundError` when unpickling) | Version mismatch: align numpy, scikit-learn and xgboost with `requirements/serve.txt` and retrain |
| `dvc repro` says everything is up to date | Nothing changed. Use `dvc repro --force train` or change a parameter |
| Metrics differ slightly between GPU and CPU runs | Expected; GPU training is not bit-identical to CPU |

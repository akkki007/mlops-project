# Run the model on real milk samples on your own computer

This guide is for a lab or collection center that has real milk readings and wants
the model's verdict on them. It needs no GPU and no cloud account: an ordinary
laptop (Windows, macOS or Linux) is enough.

> **Read this first.** The model was trained on a *synthetic* dataset and has
> never been checked against real, lab-confirmed milk. Use it to decide which
> samples to **retest**, not to reject milk on its own. Your lab results are what
> will show how accurate it really is, and they're the most useful thing you can
> send back (step 7).

## 1. What you need

- Python 3.11 ([python.org](https://www.python.org/downloads/); on Windows, tick
  "Add python.exe to PATH" during install).
- Git ([git-scm.com](https://git-scm.com/downloads)).
- About 2 GB of free disk space and an internet connection for the first setup.
- Readings for each sample: **fat %, SNF %, density, pH, freezing point and
  conductivity**. The model needs all six. If your instrument doesn't measure one
  of them (many Lactoscanner exports have no pH or conductivity), the model
  can't score those samples.

## 2. Install (once, about 5 minutes)

**macOS / Linux** (Terminal):

```bash
git clone https://github.com/akkki007/mlops-project.git
cd mlops-project
git checkout claude/work-prioritization-5tfpx6    # or main, once merged
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On macOS, XGBoost also needs OpenMP: `brew install libomp`.

**Windows** (Command Prompt, `cmd.exe`):

```bat
git clone https://github.com/akkki007/mlops-project.git
cd mlops-project
git checkout claude/work-prioritization-5tfpx6
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

In PowerShell, the activate line is `.venv\Scripts\Activate.ps1` instead. If
PowerShell refuses to run it, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, then try again.

Every later session starts with `cd mlops-project` and the activate line. Your
prompt then begins with `(.venv)`.

> **Windows: type each command on one line.** Commands in this guide are written on
> one line so they paste into Command Prompt and PowerShell as they are. The `\` at
> the end of a line that you may see in Linux guides doesn't work on Windows:
> Command Prompt uses `^` instead, and PowerShell uses a backtick (`` ` ``).

## 3. Get the trained model (once, about 6 minutes)

The model file is not stored in git, so build it on your machine:

```bash
dvc repro
```

This downloads the training data (16 MB), checks it, trains the model and writes
`models/cascade.joblib`. You'll see each step's name as it runs. When it finishes,
`reports/model.md` holds the test results. Rerunning it later does nothing unless
something changed.

You only need to redo this after pulling a newer version of the project.

## 4. Put your readings in a CSV file

Start from [`examples/real_samples_template.csv`](examples/real_samples_template.csv).
You can open it in Excel or LibreOffice and save as CSV. Its three rows are
made-up examples; replace them with your samples. One row per sample:

| Column | Required | Unit expected | Typical pure milk |
| --- | --- | --- | --- |
| `sample_id` | no | any text | — |
| `collection_point` | no | any text | — |
| `fat_pct` | **yes** | % | cow 3–5, buffalo 5–8 |
| `snf_pct` | **yes** | % | cow 8.3–9.2, buffalo 9–10 |
| `density` | **yes** | g/mL (specific gravity is fine) | 1.027–1.034 |
| `ph` | **yes** | pH | 6.6–6.8 |
| `freezing_point` | **yes** | °C, negative | −0.520 to −0.560 |
| `conductivity` | **yes** | mS/cm | 3.5–5.5 |
| `is_adulterated` | only for accuracy | 0 = pure, 1 = adulterated (lab-confirmed) | — |
| `adulterant` | only for accuracy | e.g. `Water`, `Starch`, `Urea`; blank = pure | — |

Any other columns (for example `milk_type`) are kept in the output and otherwise
ignored.

**Different column names?** Keep your headers and map them with `--rename`, for
example `--rename "Fat=fat_pct,SNF=snf_pct,SG=density,pH=ph,FPD=freezing_point,EC=conductivity"`.

**Different units?** Convert on the fly:

| Your instrument reports | Add this option |
| --- | --- |
| Freezing point in °H (Hortvet), e.g. −0.540 | `--freezing-point-unit H` (converted with °C = 0.9658 × °H) |
| Conductivity in µS/cm, e.g. 4800 | `--conductivity-unit uS/cm` |
| Lactometer reading (CLR), e.g. 28 | `--density-unit CLR` (28 means 1.028 g/mL) |

Check your instrument's manual if you're unsure which unit it uses. A wrong unit
is the most common reason for every sample coming back rejected or flagged.

## 5. Score your samples

Put your file in the project folder (here `my_samples.csv`) and run:

```bash
python -m milk_adulteration.models.evaluate_external my_samples.csv --predictions results/my_results.csv --out results/my_report.md
```

The `results` folder is created for you. To try it before your own file is ready,
score the template: replace `my_samples.csv` with `examples/real_samples_template.csv`.

**`results/my_results.csv`** has one row per sample: all your columns, then

| Column | Meaning |
| --- | --- |
| `status` | `scored`, or `rejected` if the readings failed validation (see `reject_reason`) |
| `pred_band` | **`accept`** (looks pure), **`retest`** (uncertain: test again in the lab) or **`reject`** (likely adulterated) |
| `pred_is_adulterated` | the model's yes/no |
| `pred_risk_score` | 0–1; higher means more likely adulterated |
| `pred_adulterant` | the most likely adulterant, only for flagged samples |
| `pred_confidence` | how sure it is of that adulterant (0–1) |
| `pred_unusual_readings` | readings outside anything the model has seen; such samples are never `accept` |
| `reject_reason` | why a row wasn't scored, e.g. `ph: in_range(5.5, 8.5) (got 12.0)` |

**`results/my_report.md`** summarises how many samples were scored, rejected and
flagged.

### If you have lab-confirmed results: measure accuracy

Fill in `is_adulterated` (and `adulterant` if known) for each sample, then add
`--label is_adulterated --adulterant adulterant`:

```bash
python -m milk_adulteration.models.evaluate_external my_samples.csv --label is_adulterated --adulterant adulterant --predictions results/my_results.csv --out results/my_report.md
```

The report then shows recall (share of adulterated samples caught), precision
(share of flags that were right), per-adulterant scores and 95% confidence
intervals. The model never learns from this file, so it's a fair test. With fewer
than about 30 adulterated samples the intervals are wide; treat the numbers as a
first look.

## 6. Check single samples in a browser (optional)

```bash
uvicorn milk_adulteration.api.app:app
```

Open <http://127.0.0.1:8000/docs>, choose **POST /predict**, then **Try it out**.
Edit the example readings and press **Execute**. **POST /predict/batch** accepts
the same CSV file as step 5. Stop the server with Ctrl+C.

With Docker installed, `docker compose up --build` runs the same service in a
container, after step 3.

## 7. How far to trust the results

What the model is expected to handle, from tests on synthetic and generated
samples (not real milk):

- **Added water**: caught reliably from about 10% up; often missed below 5%.
- **Detergent, sodium bicarbonate, urea, glucose, starch, skimmed milk powder**:
  caught on the synthetic data. Naming the right one is weaker, and starch and milk
  powder get confused.
- **Formalin, hydrogen peroxide, vegetable oil, melamine**: these barely change
  the six readings. **The model will usually miss them**; only lab tests find them.
- **Buffalo milk with low fat and high SNF** (for example 5% fat, 9.8% SNF) is
  often flagged even when pure, because it looks like milk-powder or starch
  adulteration. Record whether each sample is cow or buffalo milk.

**Please send back** your CSV with lab results: all six readings, the lab verdict,
the adulterant if known, cow or buffalo, and the instrument used. Even 50–200
samples would show the model's real accuracy and let it be retrained on real milk.
[GUIDE.md](GUIDE.md) covers retraining (with or without a GPU).

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `error: unrecognized arguments: \` | You pasted a multi-line Linux command into Windows. Put the whole command on one line |
| `python3.11: command not found` / `py -3.11` fails | Install Python 3.11, or use the `python3`/`python` you have if `python --version` says 3.11 or later |
| `No such file or directory: my_samples.csv` | Put the CSV in the `mlops-project` folder, or give its full path in quotes, e.g. `"D:\data\my samples.csv"` |
| `No such file or directory: models/cascade.joblib` | Run step 3 (`dvc repro`) first |
| `is missing columns [...]` | Your headers differ: use `--rename` (step 4) |
| Every row `rejected`, reasons mention `freezing_point` | Freezing point is probably in °H or positive; use `--freezing-point-unit H` and make sure values are negative |
| Every row `rejected`, reasons mention `conductivity` | Probably µS/cm: add `--conductivity-unit uS/cm` |
| Every row `rejected`, reasons mention `density` | Probably a lactometer reading: add `--density-unit CLR` |
| `retest` with `pred_unusual_readings` filled | That reading is outside what the model knows; check the value, then lab-test the sample |
| macOS: `Library not loaded: libomp.dylib` | `brew install libomp` |
| `Checksum mismatch` during `dvc repro` | The public training file changed upstream. Report it; don't edit the checksum unless you've reviewed the new file |

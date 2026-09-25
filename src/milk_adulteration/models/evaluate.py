"""DVC stage `evaluate`: score the cascade once on the untouched test set.

Reports Stage 1 on all adulterants and on detectable ones (the recall target
agreed for v1), Stage 2 on adulterated test rows, the end-to-end cascade,
calibration, single-sample latency and the Week 3 exit criterion.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from milk_adulteration.config import (
    OTHER_CLASS,
    PURE_CLASS,
    STAGE2_CLASSES,
    load_params,
    resolve,
    tracking_uri,
)
from milk_adulteration.evaluation import binary_metrics, bootstrap_ci, multiclass_metrics
from milk_adulteration.models.cascade import BANDS, Cascade

log = logging.getLogger(__name__)

TARGETS = {"recall": 0.95, "precision": 0.80, "pr_auc": 0.90, "macro_f1": 0.80}


def stage1_view(
    df: pd.DataFrame, raw: np.ndarray, risk: np.ndarray, thr: float, rounds: int, seed: int
) -> dict[str, Any]:
    y = df["is_adulterated"].to_numpy()
    return {
        **binary_metrics(y, raw, thr),
        "ci95": bootstrap_ci(y, raw, thr, rounds, seed),
        "brier": float(brier_score_loss(y, risk)),
        "mean_risk": float(risk.mean()),
        "positive_rate": float(y.mean()),
    }


def end_to_end(df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    pos = df["is_adulterated"].to_numpy() == 1
    caught = pred["is_adulterated"].to_numpy() & pos
    named = caught & (pred["adulterant"].to_numpy() == df["adulterant"].to_numpy())
    return {
        "positives": int(pos.sum()),
        "caught": int(caught.sum()),
        "caught_and_named": int(named.sum()),
        "caught_and_named_rate": float(named.sum() / max(pos.sum(), 1)),
    }


def latency_ms(model: Cascade, df: pd.DataFrame, n: int = 200) -> dict[str, float]:
    rows = [df.iloc[[i % len(df)]] for i in range(n)]
    model.predict(rows[0])  # warm-up
    times = []
    for row in rows:
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def importance(model: Any, features: list[str]) -> dict[str, float]:
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    # get_score keys are the training column names; missing means never used.
    return {f: round(gain.get(f, 0.0) / total, 4) for f in features}


def evaluate(model: Cascade, test: pd.DataFrame, rounds: int, seed: int) -> dict[str, Any]:
    pred = model.predict(test)
    raw, risk = model.raw_score(test), pred["risk_score"].to_numpy()
    det = (test["adulterant"] != OTHER_CLASS).to_numpy()

    pos = test[test["is_adulterated"] == 1]
    s2_pred = np.asarray(model.classes)[model.adulterant_proba(pos).argmax(axis=1)]
    bands = pd.crosstab(
        np.where(test["is_adulterated"] == 1, "adulterated", "pure"), pred["band"]
    ).reindex(columns=list(BANDS), fill_value=0)

    return {
        "threshold": model.threshold,
        "stage1": {
            "all": stage1_view(test, raw, risk, model.threshold, rounds, seed),
            "detectable": stage1_view(
                test[det], raw[det], risk[det], model.threshold, rounds, seed
            ),
        },
        "stage2": multiclass_metrics(pos["adulterant"], s2_pred, STAGE2_CLASSES),
        "end_to_end": {
            "all": end_to_end(test, pred),
            "detectable": end_to_end(test[det], pred[det]),
        },
        "other_flagged": int(pred.loc[test["adulterant"] == OTHER_CLASS, "is_adulterated"].sum()),
        "other_total": int((test["adulterant"] == OTHER_CLASS).sum()),
        "pure_flagged": int(pred.loc[test["adulterant"] == PURE_CLASS, "is_adulterated"].sum()),
        "bands": bands.to_dict(orient="index"),
        "latency_ms": latency_ms(model, test),
        "importance": {
            "stage1": importance(model.stage1, model.features),
            "stage2": importance(model.stage2, model.features),
        },
    }


def exit_criteria(r: dict[str, Any], cv: dict[str, Any]) -> list[tuple[str, float, float, bool]]:
    d = r["stage1"]["detectable"]
    rows = [
        ("Stage 1 recall, detectable (test)", d["recall"], TARGETS["recall"]),
        ("Stage 1 precision, detectable (test)", d["precision"], TARGETS["precision"]),
        ("Stage 1 PR-AUC, detectable (test)", d["pr_auc"], TARGETS["pr_auc"]),
        ("Stage 2 macro-F1 (test)", r["stage2"]["macro_f1"], TARGETS["macro_f1"]),
        ("Stage 2 macro-F1 (train CV)", cv["stage2_cv_macro_f1"], TARGETS["macro_f1"]),
    ]
    return [(name, val, tgt, val >= tgt) for name, val, tgt in rows]


def render(r: dict[str, Any], cv: dict[str, Any], baselines: dict[str, Any] | None) -> str:
    def s1_row(label: str, m: dict[str, Any]) -> str:
        ci = m["ci95"]
        return (
            f"| {label} | {m['recall']:.3f} ({ci['recall'][0]:.2f}–{ci['recall'][1]:.2f}) | "
            f"{m['precision']:.3f} ({ci['precision'][0]:.2f}–{ci['precision'][1]:.2f}) | "
            f"{m['pr_auc']:.3f} ({ci['pr_auc'][0]:.2f}–{ci['pr_auc'][1]:.2f}) | "
            f"{m['tp']} | {m['fp']} | {m['fn']} |"
        )

    s1 = r["stage1"]
    rows = [s1_row("XGBoost cascade, detectable", s1["detectable"])]
    rows.append(s1_row("XGBoost cascade, all", s1["all"]))
    if baselines:
        rf = baselines["stage1"]["random_forest+derived"]
        for view in ("detectable", "all"):
            m = {**rf[view]["test"], "ci95": rf[view]["test_ci95"]}
            rows.append(s1_row(f"Baseline RF + derived, {view}", m))

    crit = exit_criteria(r, cv)
    crit_rows = [
        f"| {n} | {v:.3f} | ≥ {t:.2f} | {'pass' if ok else '**miss**'} |" for n, v, t, ok in crit
    ]
    s1_pass = all(ok for n, *_, ok in crit if n.startswith("Stage 1") and "PR-AUC" not in n)

    pc = r["stage2"]["per_class_f1"]
    labels = r["stage2"]["labels"]
    conf = r["stage2"]["confusion"]
    conf_rows = [
        f"| {labels[i]} | " + " | ".join(map(str, row)) + " |" for i, row in enumerate(conf)
    ]
    bands = r["bands"]
    band_rows = [
        f"| {k} | " + " | ".join(str(bands[k].get(b, 0)) for b in BANDS) + " |" for k in bands
    ]
    imp = r["importance"]
    imp_rows = [f"| {f} | {imp['stage1'][f]:.3f} | {imp['stage2'][f]:.3f} |" for f in imp["stage1"]]
    e2e_rows = [
        f"| {view.title()} | {m['positives']} | {m['caught']} | {m['caught_and_named']} |"
        for view, m in r["end_to_end"].items()
    ]
    lat = r["latency_ms"]
    rf2 = baselines["stage2"]["random_forest"]["cv_macro_f1_mean"] if baselines else None
    bl_rf_f1 = f"{rf2:.3f}" if rf2 is not None else "n/a"
    gain = f"{cv['stage2_cv_macro_f1'] - rf2:+.2f}" if rf2 is not None else "n/a"
    nl = "\n"
    return f"""# Week 3: two-stage XGBoost cascade

Generated by `dvc repro evaluate`. All numbers are on the **synthetic** ommadav
dataset; the test set (375 rows, 19 adulterated) was used once, here.

**Week 3 exit criterion (Stage 1 recall ≥ 0.95 and precision ≥ 0.80 on test,
detectable adulterants): {"MET" if s1_pass else "NOT MET"}.**

| Check | Value | Target | Result |
|---|---|---|---|
{nl.join(crit_rows)}

"Detectable" excludes the `other` class (formalin, H₂O₂, vegetable oil, melamine),
as agreed for v1: those need a confirmatory lab test.

## Stage 1: pure vs adulterated (test, 95% bootstrap CI)

Threshold {r["threshold"]:.4f} on the raw Stage 1 score, chosen on validation.

| Model | Recall | Precision | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|
{nl.join(rows)}

Stage 1 never trained on `other`, yet flags {r["other_flagged"]} of {r["other_total"]}
`other` test samples; {r["pure_flagged"]} pure samples are flagged.

**Calibration.** Brier score {s1["all"]["brier"]:.4f} (all test rows). Mean risk
{s1["all"]["mean_risk"]:.3f} against an actual rate of {s1["all"]["positive_rate"]:.3f}.
Because training CV separates the classes almost perfectly, isotonic calibration is
close to a 0/1 step, so the risk score is near 0 or 1 for most samples. Expect a
smoother score on real data.

### Bands (test)

| True class | accept | retest | reject |
|---|---|---|---|
{nl.join(band_rows)}

## Stage 2: which adulterant (adulterated test rows)

Test macro-F1 **{r["stage2"]["macro_f1"]:.3f}** over {sum(map(sum, conf))} samples;
train CV macro-F1 {cv["stage2_cv_macro_f1"]:.3f}. Classes with one test sample swing
the test figure by about 0.1, so read it together with the CV number.

| Class | Test F1 |
|---|---|
{nl.join(f"| {c} | {v:.2f} |" for c, v in pc.items())}

Confusion matrix (rows = true, columns = predicted, order as rows):

| True \\ Pred | {" | ".join(labels)} |
|---|{"---|" * len(labels)}
{nl.join(conf_rows)}

## End to end (Stage 1 flag, then Stage 2 name)

| View | Adulterated | Caught | Caught and named correctly |
|---|---|---|---|
{nl.join(e2e_rows)}

## Augmentation: does it help? (train CV, same tuned params)

| Metric | With augmentation | Without |
|---|---|---|
| Stage 1 PR-AUC (detectable) | {cv["stage1_cv_pr_auc"]:.3f} | {cv["stage1_cv_pr_auc_no_aug"]:.3f} |
| Stage 2 macro-F1 | {cv["stage2_cv_macro_f1"]:.3f} | {cv["stage2_cv_macro_f1_no_aug"]:.3f} |

Augmented rows come only from each fold's training part and are never scored.
The "without" column reuses parameters tuned with augmentation, which flatters
augmentation; the Week 2 Random Forest reached Stage 2 CV macro-F1
{bl_rf_f1} without it; the cascade's Stage 2 (XGBoost with augmentation) is
{gain} above that.

## Feature importance (share of total gain)

| Feature | Stage 1 | Stage 2 |
|---|---|---|
{nl.join(imp_rows)}

## Latency

Single-sample `Cascade.predict`, including Pandera validation: p50
{lat["p50"]:.1f} ms, p95 {lat["p95"]:.1f} ms (API target: p95 < 100 ms).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    p, tp = params["evaluate"], params["train"]
    split_dir = resolve(params["split"]["output_dir"])

    model = Cascade.load(resolve(tp["model"]))
    test = pd.read_csv(split_dir / "test.csv")
    cv = json.loads(resolve(tp["cv_report"]).read_text())
    bl_path = resolve(params["baselines"]["metrics"])
    baselines = json.loads(bl_path.read_text()) if bl_path.exists() else None

    r = evaluate(model, test, p["bootstrap_rounds"], tp["seed"])
    report = render(r, cv, baselines)
    resolve(p["report"]).write_text(report)

    metrics = {
        "threshold": r["threshold"],
        "stage1": {
            v: {k: m[k] for k in ("recall", "precision", "pr_auc", "brier", "tp", "fp", "fn")}
            for v, m in r["stage1"].items()
        },
        "stage2_macro_f1": r["stage2"]["macro_f1"],
        "stage2_per_class_f1": r["stage2"]["per_class_f1"],
        "end_to_end": r["end_to_end"],
        "exit_criteria": {n: ok for n, _, _, ok in exit_criteria(r, cv)},
    }
    resolve(p["metrics"]).write_text(json.dumps(metrics, indent=2) + "\n")

    mlflow.set_tracking_uri(tracking_uri(tp))
    with mlflow.start_run(run_id=model.metadata["mlflow_run_id"]):
        for view, m in r["stage1"].items():
            mlflow.log_metrics(
                {f"test_{view}_{k}": m[k] for k in ("recall", "precision", "pr_auc", "brier")}
            )
        mlflow.log_metric("test_stage2_macro_f1", r["stage2"]["macro_f1"])
        mlflow.log_metric("latency_p95_ms", r["latency_ms"]["p95"])
        mlflow.log_text(report, "model_report.md")
    log.info("Wrote %s", resolve(p["report"]))


if __name__ == "__main__":
    main()

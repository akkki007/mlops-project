"""DVC stage `baselines`: FSSAI rules, logistic regression and Random Forest.

Stage 1 (pure vs adulterated): fit on train, pick the threshold on validation
for the target recall, report once on test with bootstrap CIs.
Stage 2 (which adulterant): fit on adulterated train rows, report on
adulterated test rows (as if Stage 1 were perfect), plus repeated stratified CV
on train+val positives because the test set holds only ~19 of them.

Every model is logged as an MLflow run tagged with the data version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from milk_adulteration.config import OTHER_CLASS, STAGE2_CLASSES, load_params, resolve
from milk_adulteration.data.split import SPLITS
from milk_adulteration.evaluation import (
    binary_metrics,
    bootstrap_ci,
    multiclass_metrics,
    threshold_for_recall,
)
from milk_adulteration.features import feature_matrix
from milk_adulteration.models import rules

log = logging.getLogger(__name__)

RULES_THRESHOLD = 1.0  # one breached rule flags the sample


def make_models(p: dict[str, Any]) -> dict[str, Callable[[], ClassifierMixin]]:
    lr, rf = p["logreg"], p["random_forest"]
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], class_weight="balanced"),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=rf["n_estimators"],
            min_samples_leaf=rf["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=p["seed"],
            n_jobs=-1,
        ),
    }


def model_params(model: ClassifierMixin) -> dict[str, Any]:
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return {k: v for k, v in est.get_params().items() if np.isscalar(v) or v is None}


def data_version(split_dir: Path) -> dict[str, str]:
    return {
        f"data_sha256_{name}": hashlib.sha256((split_dir / f"{name}.csv").read_bytes()).hexdigest()[
            :12
        ]
        for name in SPLITS
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate_stage1(
    splits: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    p: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate on all rows, and on "detectable" rows with the `other` class removed.

    `other` (formalin, H2O2, vegetable oil, melamine) barely moves field readings,
    so it dominates the misses; the detectable view shows how the model does on the
    adulterants field readings can see. Each view picks its own threshold.
    """
    out: dict[str, Any] = {}
    for view in ("all", "detectable"):
        keep = {
            k: (splits[k]["adulterant"] != OTHER_CLASS).to_numpy()
            if view == "detectable"
            else np.ones(len(splits[k]), dtype=bool)
            for k in ("val", "test")
        }
        y = {k: splits[k]["is_adulterated"].to_numpy()[keep[k]] for k in keep}
        sc = {k: scores[k][keep[k]] for k in keep}
        thr = threshold
        if thr is None:
            thr = threshold_for_recall(y["val"], sc["val"], p["target_recall"])
        out[view] = {
            "threshold": thr,
            "val": binary_metrics(y["val"], sc["val"], thr),
            "test": binary_metrics(y["test"], sc["test"], thr),
            "test_ci95": bootstrap_ci(y["test"], sc["test"], thr, p["bootstrap_rounds"], p["seed"]),
        }
    return out


def run_stage1(splits: dict[str, pd.DataFrame], p: dict[str, Any]) -> dict[str, Any]:
    y_train = splits["train"]["is_adulterated"].to_numpy()
    rule_scores = {k: rules.score(splits[k]).to_numpy() for k in ("val", "test")}
    results: dict[str, Any] = {
        "fssai_rules": {
            "model": None,
            **evaluate_stage1(splits, rule_scores, p, threshold=RULES_THRESHOLD),
        }
    }
    for name, factory in make_models(p).items():
        for derived in (False, True):
            X = {k: feature_matrix(v, derived) for k, v in splits.items()}
            model = factory().fit(X["train"], y_train)
            scores = {k: model.predict_proba(X[k])[:, 1] for k in ("val", "test")}
            results[f"{name}{'+derived' if derived else ''}"] = {
                "model": model,
                "features": list(X["train"].columns),
                **evaluate_stage1(splits, scores, p),
            }
    return results


def run_stage2(splits: dict[str, pd.DataFrame], p: dict[str, Any]) -> dict[str, Any]:
    pos = {k: v[v["is_adulterated"] == 1] for k, v in splits.items()}
    dev = pd.concat([pos["train"], pos["val"]])
    cv = RepeatedStratifiedKFold(
        n_splits=p["stage2_cv_folds"], n_repeats=p["stage2_cv_repeats"], random_state=p["seed"]
    )
    results: dict[str, Any] = {}
    for name, factory in make_models(p).items():
        for derived in (False, True):
            X_train = feature_matrix(pos["train"], derived)
            model = factory().fit(X_train, pos["train"]["adulterant"])
            test = multiclass_metrics(
                pos["test"]["adulterant"],
                model.predict(feature_matrix(pos["test"], derived)),
                STAGE2_CLASSES,
            )
            # Repeated CV: cross_val_predict needs a partition, so score each repeat.
            X_dev, y_dev = feature_matrix(dev, derived), dev["adulterant"].to_numpy()
            cv_f1 = []
            splits_iter = list(cv.split(X_dev, y_dev))
            for r in range(p["stage2_cv_repeats"]):
                folds = splits_iter[r * p["stage2_cv_folds"] : (r + 1) * p["stage2_cv_folds"]]
                pred = cross_val_predict(factory(), X_dev, y_dev, cv=folds)
                cv_f1.append(multiclass_metrics(y_dev, pred, STAGE2_CLASSES)["macro_f1"])
            results[f"{name}{'+derived' if derived else ''}"] = {
                "model": model,
                "features": list(X_train.columns),
                "test": test,
                "cv_macro_f1_mean": float(np.mean(cv_f1)),
                "cv_macro_f1_std": float(np.std(cv_f1)),
            }
    return results


def log_to_mlflow(
    stage: str, key: str, res: dict[str, Any], tags: dict[str, str], example: pd.DataFrame | None
) -> str:
    with mlflow.start_run(run_name=f"{stage}/{key}") as run:
        mlflow.set_tags({**tags, "stage": stage, "model": key})
        model = res.get("model")
        if model is not None:
            mlflow.log_params(model_params(model))
            mlflow.log_param("features", ",".join(res["features"]))
        if stage == "stage1":
            for view, prefix in (("all", ""), ("detectable", "det_")):
                r = res[view]
                mlflow.log_param(f"{prefix}threshold", r["threshold"])
                for split in ("val", "test"):
                    mlflow.log_metrics({f"{prefix}{split}_{k}": v for k, v in r[split].items()})
                for k, (lo, hi) in r["test_ci95"].items():
                    mlflow.log_metrics(
                        {f"{prefix}test_{k}_ci_lo": lo, f"{prefix}test_{k}_ci_hi": hi}
                    )
        else:
            mlflow.log_metrics(
                {
                    "test_macro_f1": res["test"]["macro_f1"],
                    "cv_macro_f1_mean": res["cv_macro_f1_mean"],
                    "cv_macro_f1_std": res["cv_macro_f1_std"],
                    **{f"test_f1_{c}": v for c, v in res["test"]["per_class_f1"].items()},
                }
            )
            mlflow.log_dict(res["test"], "test_confusion.json")
        if model is not None and example is not None:
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=example[res["features"]].head(3),
                # Our own freshly trained trees, so safe to load back via skops.
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        return run.info.run_id


def summarise(stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    def strip(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k not in ("model", "features")}

    return {
        "stage1": {k: strip(v) for k, v in stage1.items()},
        "stage2": {k: strip(v) for k, v in stage2.items()},
    }


def render(summary: dict[str, Any], p: dict[str, Any]) -> str:
    def s1_rows(view: str) -> str:
        return "\n".join(s1_row(key, r[view]) for key, r in summary["stage1"].items())

    def s1_row(key: str, r: dict[str, Any]) -> str:
        t, ci = r["test"], r["test_ci95"]
        return (
            f"| {key} | {r['threshold']:.3f} | {t['recall']:.3f} "
            f"({ci['recall'][0]:.2f}–{ci['recall'][1]:.2f}) | {t['precision']:.3f} "
            f"({ci['precision'][0]:.2f}–{ci['precision'][1]:.2f}) | {t['pr_auc']:.3f} "
            f"({ci['pr_auc'][0]:.2f}–{ci['pr_auc'][1]:.2f}) | {t['tp']} | {t['fp']} | {t['fn']} |"
        )

    header = (
        "| Model | Threshold | Recall | Precision | PR-AUC | TP | FP | FN |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    s2_rows = [
        f"| {key} | {r['test']['macro_f1']:.3f} | "
        f"{r['cv_macro_f1_mean']:.3f} ± {r['cv_macro_f1_std']:.3f} |"
        for key, r in summary["stage2"].items()
    ]
    best2 = max(summary["stage2"], key=lambda k: summary["stage2"][k]["cv_macro_f1_mean"])
    per_class = summary["stage2"][best2]["test"]["per_class_f1"]
    pc_rows = [f"| {c} | {v:.2f} |" for c, v in per_class.items()]
    nl = "\n"
    return f"""# Week 2 baselines

Generated by `dvc repro baselines`. All numbers are on the **synthetic** ommadav
dataset. Each model is an MLflow run in experiment `{p["mlflow"]["experiment"]}`.

## Stage 1: pure vs adulterated (test set, 95% bootstrap CI)

ML thresholds are the highest that reach recall ≥ {p["target_recall"]} on
validation; the rule baseline flags any sample breaking one or more rules.
Targets: recall ≥ 0.95, precision ≥ 0.80, PR-AUC ≥ 0.90.

### All adulterants

{header}
{s1_rows("all")}

### Detectable adulterants only (`other` rows removed)

Formalin and hydrogen peroxide look like pure milk in all six readings, so a
recall target that includes them forces the threshold down to near zero.

{header}
{s1_rows("detectable")}

## Stage 2: which adulterant (adulterated rows only)

Test macro-F1 covers ~19 samples, so the repeated {p["stage2_cv_folds"]}-fold CV on
train+val positives ({p["stage2_cv_repeats"]} repeats) is the steadier number.
Target: macro-F1 ≥ 0.80.

| Model | Test macro-F1 | CV macro-F1 |
|---|---|---|
{nl.join(s2_rows)}

### Per-class test F1 for the best CV model (`{best2}`)

| Class | F1 |
|---|---|
{nl.join(pc_rows)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    p = params["baselines"]
    split_dir = resolve(params["split"]["output_dir"])

    splits = {k: pd.read_csv(split_dir / f"{k}.csv") for k in SPLITS}
    stage1, stage2 = run_stage1(splits, p), run_stage2(splits, p)
    summary = summarise(stage1, stage2)

    if not args.no_mlflow:
        mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
        mlflow.set_experiment(p["mlflow"]["experiment"])
        tags = {**data_version(split_dir), "git_commit": git_commit()}
        example = feature_matrix(splits["train"], derived=True)
        for stage, results in (("stage1", stage1), ("stage2", stage2)):
            for key, res in results.items():
                summary[stage][key]["mlflow_run_id"] = log_to_mlflow(stage, key, res, tags, example)

    report, metrics = resolve(p["report"]), resolve(p["metrics"])
    report.write_text(render(summary, p))
    # MLflow run IDs change every run; keep them out of the DVC-tracked metrics.
    metrics_only = json.loads(json.dumps(summary))
    for stage in metrics_only.values():
        for res in stage.values():
            res.pop("mlflow_run_id", None)
            res.get("test", {}).pop("confusion", None)
    metrics.write_text(json.dumps(metrics_only, indent=2) + "\n")
    log.info("Wrote %s and %s", report, metrics)


if __name__ == "__main__":
    main()

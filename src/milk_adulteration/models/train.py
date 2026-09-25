"""DVC stage `train`: tune, fit and calibrate the two-stage XGBoost cascade.

Tuning uses Optuna on stratified k-fold CV over the real training rows. Each
fold's training part is augmented from that fold alone and scoring uses only
real rows, so synthetic rows never leak into a score.

- Stage 1 objective: PR-AUC on detectable adulterants (see `stage1_exclude_other`).
- Stage 2 objective: macro-F1 over the eight classes on real adulterated rows.
- Calibration: isotonic regression on the out-of-fold Stage 1 scores of real
  training rows, so the risk score reflects the real ~5% prevalence rather than
  the augmented one.
- Threshold: on the raw Stage 1 score, the highest reaching the recall target on
  detectable validation rows, moved halfway to the next lower validation score
  for a safety margin.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from milk_adulteration.config import (
    FEATURES,
    OTHER_CLASS,
    STAGE2_CLASSES,
    load_params,
    resolve,
    tracking_uri,
)
from milk_adulteration.data.augment import augment
from milk_adulteration.evaluation import threshold_for_recall
from milk_adulteration.features import feature_matrix
from milk_adulteration.models.baselines import data_version, git_commit
from milk_adulteration.models.cascade import Cascade, raw_score_at_risk

log = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def stage1_rows(df: pd.DataFrame, exclude_other: bool) -> pd.DataFrame:
    return df[df["adulterant"] != OTHER_CLASS] if exclude_other else df


def stage2_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Adulterated single-adulterant rows (combination rows are Stage 1 only)."""
    pos = df[df["is_adulterated"] == 1]
    return pos[pos["aug_combo"] == ""] if "aug_combo" in pos else pos


# Set once in main() from params: where XGBoost trains ("cpu" or "cuda") and threads.
RUNTIME: dict[str, Any] = {"device": "cpu", "n_jobs": 4}


def check_device(device: str) -> None:
    """Fail loudly if a GPU was asked for but XGBoost cannot use one.

    XGBoost otherwise falls back to CPU with only a log warning.
    """
    if device == "cpu":
        return
    import xgboost

    if not xgboost.build_info().get("USE_CUDA"):
        raise RuntimeError(
            f"train.device={device!r} but this XGBoost build has no CUDA support. "
            "Install the full package: pip uninstall -y xgboost-cpu && pip install xgboost"
        )
    X = np.random.default_rng(0).random((64, 2))
    booster = XGBClassifier(n_estimators=2, device=device).fit(X, X[:, 0] > 0.5).get_booster()
    if f'"device":"{device}' not in booster.save_config().replace(" ", ""):
        raise RuntimeError(f"train.device={device!r} but XGBoost found no usable GPU")


def _runtime() -> dict[str, Any]:
    return {"device": RUNTIME["device"], "n_jobs": RUNTIME["n_jobs"]}


def fit_stage1(train: pd.DataFrame, params: dict[str, Any], seed: int) -> XGBClassifier:
    y = train["is_adulterated"].to_numpy()
    model = XGBClassifier(
        **params,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric="aucpr",
        random_state=seed,
        **_runtime(),
    )
    return model.fit(feature_matrix(train), y)


def fit_stage2(train: pd.DataFrame, params: dict[str, Any], seed: int) -> XGBClassifier:
    y = train["adulterant"].map({c: i for i, c in enumerate(STAGE2_CLASSES)}).to_numpy()
    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=len(STAGE2_CLASSES),
        eval_metric="mlogloss",
        random_state=seed,
        **_runtime(),
    )
    return model.fit(feature_matrix(train), y, sample_weight=compute_sample_weight("balanced", y))


class FoldData:
    """Real CV folds over the training rows, each with its own augmented training part."""

    def __init__(self, train: pd.DataFrame, aug_params: dict[str, Any], k: int, seed: int):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        self.train = train.reset_index(drop=True)
        self.folds = []
        for i, (tr, va) in enumerate(skf.split(self.train, self.train["adulterant"])):
            real_tr, real_va = self.train.iloc[tr], self.train.iloc[va]
            aug_tr = augment(real_tr, aug_params, seed=seed + i)
            self.folds.append((real_tr.assign(aug_combo=""), aug_tr, real_va))

    def oof_stage1(self, params: dict, exclude_other: bool, seed: int, augmented: bool = True):
        """Out-of-fold Stage 1 scores for the real training rows Stage 1 is scored on."""
        idx, scores = [], []
        for real_tr, aug_tr, va in self.folds:
            model = fit_stage1(
                stage1_rows(aug_tr if augmented else real_tr, exclude_other), params, seed
            )
            va = stage1_rows(va, exclude_other)
            idx.append(va.index.to_numpy())
            scores.append(model.predict_proba(feature_matrix(va))[:, 1])
        idx = np.concatenate(idx)
        return self.train.loc[idx, "is_adulterated"].to_numpy(), np.concatenate(scores), idx

    def oof_stage2(self, params: dict, seed: int, augmented: bool = True):
        true, pred = [], []
        for real_tr, aug_tr, va in self.folds:
            model = fit_stage2(stage2_rows(aug_tr if augmented else real_tr), params, seed)
            va = stage2_rows(va)
            true.append(va["adulterant"].to_numpy())
            pred.append(
                np.asarray(STAGE2_CLASSES)[model.predict_proba(feature_matrix(va)).argmax(1)]
            )
        return np.concatenate(true), np.concatenate(pred)


def macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(true, pred, labels=STAGE2_CLASSES, average="macro", zero_division=0))


def tune(
    name: str, objective: Callable[[dict[str, Any]], float], n_trials: int, seed: int
) -> optuna.Study:
    def wrapped(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        score = objective(params)
        with mlflow.start_run(run_name=f"{name}/trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_score", score)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), study_name=name
    )
    study.optimize(wrapped, n_trials=n_trials)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = load_params()
    p, aug_p = params["train"], params["augment"]
    seed, excl = p["seed"], p["stage1_exclude_other"]
    RUNTIME.update(device=p.get("device", "cpu"), n_jobs=p.get("n_jobs", 4))
    check_device(RUNTIME["device"])
    log.info("Training on %s", RUNTIME["device"])
    split_dir = resolve(params["split"]["output_dir"])

    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    train_aug = pd.read_csv(resolve(aug_p["output"]), keep_default_na=False, na_values=[""])
    train_aug["aug_combo"] = train_aug["aug_combo"].fillna("")
    folds = FoldData(train, aug_p, p["cv_folds"], seed)

    mlflow.set_tracking_uri(tracking_uri(p))
    mlflow.set_experiment(p["mlflow"]["experiment"])
    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags({**data_version(split_dir), "git_commit": git_commit()})
        mlflow.log_params(
            {f"augment.{k}": v for k, v in aug_p.items() if k not in ("input", "output")}
        )
        mlflow.log_params(
            {
                "stage1_exclude_other": excl,
                "target_recall": p["target_recall"],
                "train_device": RUNTIME["device"],
            }
        )

        def s1_objective(prm: dict[str, Any]) -> float:
            y, s, _ = folds.oof_stage1(prm, excl, seed)
            return float(average_precision_score(y, s))

        def s2_objective(prm: dict[str, Any]) -> float:
            return macro_f1(*folds.oof_stage2(prm, seed))

        log.info("Tuning Stage 1 (%d trials)", p["n_trials_stage1"])
        s1 = tune("stage1", s1_objective, p["n_trials_stage1"], seed)
        log.info("Stage 1 best CV PR-AUC %.4f", s1.best_value)
        log.info("Tuning Stage 2 (%d trials)", p["n_trials_stage2"])
        s2 = tune("stage2", s2_objective, p["n_trials_stage2"], seed)
        log.info("Stage 2 best CV macro-F1 %.4f", s2.best_value)

        # Calibrate on out-of-fold scores of real rows with the chosen params.
        y_oof, s_oof, _ = folds.oof_stage1(s1.best_params, excl, seed)
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            s_oof, y_oof
        )

        # Ablation: same params, no augmentation.
        y_na, s_na, _ = folds.oof_stage1(s1.best_params, excl, seed, augmented=False)
        cv = {
            "stage1_cv_pr_auc": s1.best_value,
            "stage1_cv_pr_auc_no_aug": float(average_precision_score(y_na, s_na)),
            "stage2_cv_macro_f1": s2.best_value,
            "stage2_cv_macro_f1_no_aug": macro_f1(*folds.oof_stage2(s2.best_params, seed, False)),
        }

        stage1 = fit_stage1(stage1_rows(train_aug, excl), s1.best_params, seed)
        stage2 = fit_stage2(stage2_rows(train_aug), s2.best_params, seed)
        # Serve on CPU whatever we trained on: the API and Docker image have no GPU.
        for m in (stage1, stage2):
            m.set_params(device="cpu")
        model = Cascade(
            stage1=stage1,
            calibrator=calibrator,
            threshold=0.0,
            reject_risk=p["reject_risk"],
            stage2=stage2,
            classes=list(STAGE2_CLASSES),
            features=list(feature_matrix(train.head(1)).columns),
            seen_ranges={
                f: (float(train_aug[f].min()), float(train_aug[f].max())) for f in FEATURES
            },
            metadata={
                "mlflow_run_id": run.info.run_id,
                "stage1_params": s1.best_params,
                "stage2_params": s2.best_params,
                "stage1_exclude_other": excl,
                "train_device": RUNTIME["device"],
                **data_version(split_dir),
                "git_commit": git_commit(),
            },
        )
        val_det = stage1_rows(val, excl)
        recall_threshold = threshold_for_recall(
            val_det["is_adulterated"].to_numpy(),
            model.raw_score(val_det),
            p["target_recall"],
            midpoint=True,
        )
        # Also flag anything the calibrated risk already calls likely adulterated, so
        # no sample is accepted with risk >= reject_risk. (The calibrator is fitted on
        # CV fold models, the recall threshold on the final model: their score gaps
        # need not line up.)
        risk_threshold = raw_score_at_risk(calibrator, p["reject_risk"])
        model.threshold = min(recall_threshold, risk_threshold)
        model.metadata.update(
            threshold=model.threshold,
            recall_threshold=recall_threshold,
            risk_threshold=risk_threshold,
        )

        mlflow.log_params({f"stage1.{k}": v for k, v in s1.best_params.items()})
        mlflow.log_params({f"stage2.{k}": v for k, v in s2.best_params.items()})
        mlflow.log_param("threshold", model.threshold)
        mlflow.log_metrics(cv)
        model_path = resolve(p["model"])
        model.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

    report = {
        **cv,
        "threshold": model.threshold,
        "stage1_params": s1.best_params,
        "stage2_params": s2.best_params,
        "calibration_rows": int(len(y_oof)),
    }
    resolve(p["cv_report"]).write_text(json.dumps(report, indent=2) + "\n")
    log.info("Saved %s (threshold %.4f): %s", model_path, model.threshold, cv)


if __name__ == "__main__":
    main()

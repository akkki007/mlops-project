"""Metrics shared by baselines and the main model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_for_recall(y_true: np.ndarray, score: np.ndarray, target: float) -> float:
    """Highest threshold whose recall (predicting `score >= t`) meets `target`.

    The highest such threshold gives the fewest false alarms at that recall.
    """
    _, recall, thresholds = precision_recall_curve(y_true, score)
    # recall[i] belongs to thresholds[i]; the last recall value has no threshold.
    ok = np.flatnonzero(recall[:-1] >= target)
    if ok.size == 0:
        return float(np.min(score))
    return float(thresholds[ok.max()])


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "flag_rate": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    rounds: int,
    seed: int,
    metrics: tuple[str, ...] = ("recall", "precision", "pr_auc"),
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for binary metrics."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    n = len(y_true)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # a resample with one class has no recall or PR-AUC
        m = binary_metrics(y_true[idx], score[idx], threshold)
        for k in metrics:
            samples[k].append(m[k])
    return {
        k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
        for k, v in samples.items()
    }


def multiclass_metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": dict(
            zip(
                labels,
                map(float, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)),
                strict=True,
            )
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

"""The two-stage model: Stage 1 risk score, Stage 2 adulterant name."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from milk_adulteration.data.schema import FIELD_SCHEMA
from milk_adulteration.features import feature_matrix

BANDS = ("accept", "retest", "reject")


def raw_score_at_risk(calibrator: IsotonicRegression, level: float) -> float:
    """Lowest raw score whose calibrated risk reaches `level` (inf if it never does).

    Isotonic predictions are monotone and piecewise linear between the fitted
    points, so bisection on that interval finds it.
    """
    xs = calibrator.X_thresholds_
    lo, hi = float(xs.min()), float(xs.max())
    if calibrator.predict([hi])[0] < level:
        return float("inf")
    if calibrator.predict([lo])[0] >= level:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if calibrator.predict([mid])[0] >= level else (mid, hi)
    return hi


@dataclass
class Cascade:
    stage1: XGBClassifier
    calibrator: IsotonicRegression
    threshold: float  # raw Stage 1 score at or above this flags the sample
    reject_risk: float  # flagged samples with calibrated risk at or above this: "reject"
    stage2: XGBClassifier
    classes: list[str]  # Stage 2 label index -> adulterant class
    features: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Per reading, the (min, max) the model saw in training. Tree models cannot
    # extrapolate, so readings outside these are marked unusual and never "accept".
    seen_ranges: dict[str, tuple[float, float]] | None = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return feature_matrix(df)[self.features]

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.stage1.predict_proba(self._X(df))[:, 1]

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.raw_score(df))

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        # Flag on the raw score: isotonic calibration is a step function and loses
        # the resolution the threshold needs.
        return self.raw_score(df) >= self.threshold

    def band(
        self, flagged: np.ndarray, risk: np.ndarray, unusual: np.ndarray | None = None
    ) -> np.ndarray:
        """reject: flagged with risk >= reject_risk. retest: flagged with lower risk, or
        not flagged but with readings outside what the model saw. accept: the rest."""
        unusual = np.zeros(len(flagged), dtype=bool) if unusual is None else unusual
        return np.select(
            [flagged & (risk >= self.reject_risk), flagged | unusual],
            ["reject", "retest"],
            "accept",
        )

    def unusual_readings(self, df: pd.DataFrame) -> list[list[str]]:
        """Per row, the readings outside the training ranges (empty when unknown)."""
        if not self.seen_ranges:
            return [[] for _ in range(len(df))]
        outside = pd.DataFrame(
            {f: ~df[f].between(lo, hi) for f, (lo, hi) in self.seen_ranges.items()}
        )
        return [list(outside.columns[row]) for row in outside.to_numpy()]

    def adulterant_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.stage2.predict_proba(self._X(df))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate readings, score them, and name the adulterant for flagged samples."""
        df = FIELD_SCHEMA.validate(df)
        raw = self.raw_score(df)
        risk = self.calibrator.predict(raw)
        flagged = raw >= self.threshold
        unusual = self.unusual_readings(df)
        out = pd.DataFrame(
            {
                "is_adulterated": flagged,
                "risk_score": risk,
                "band": self.band(flagged, risk, np.array([bool(u) for u in unusual])),
                "adulterant": None,
                "confidence": np.nan,
                "unusual_readings": unusual,
            },
            index=df.index,
        )
        if flagged.any():
            proba = self.adulterant_proba(df[flagged])
            out.loc[flagged, "adulterant"] = np.asarray(self.classes)[proba.argmax(axis=1)]
            out.loc[flagged, "confidence"] = proba.max(axis=1)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Cascade:
        model = joblib.load(path)
        if not isinstance(model, Cascade):
            raise TypeError(f"{path} does not hold a Cascade")
        return model

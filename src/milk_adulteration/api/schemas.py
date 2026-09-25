"""Request and response models for the prediction API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from milk_adulteration.data.schema import FEATURE_RANGES


def _not_bool(value: object) -> object:
    # Pydantic would otherwise read JSON true/false as 1.0/0.0.
    if isinstance(value, bool):
        raise ValueError("a reading must be a number, not true/false")
    return value


Reading = Annotated[float, BeforeValidator(_not_bool)]


def _reading(name: str, description: str) -> object:
    lo, hi = FEATURE_RANGES[name]
    return Field(
        ge=lo,
        le=hi,
        allow_inf_nan=False,
        description=f"{description} (plausible range {lo} to {hi})",
    )


class Sample(BaseModel):
    sample_id: str | None = Field(None, max_length=64)
    collection_point: str | None = Field(None, max_length=64)
    fat_pct: Reading = _reading("fat_pct", "Fat, %")
    snf_pct: Reading = _reading("snf_pct", "Solids-not-fat, %")
    density: Reading = _reading("density", "Density, g/mL")
    ph: Reading = _reading("ph", "pH")
    freezing_point: Reading = _reading("freezing_point", "Freezing point, °C")
    conductivity: Reading = _reading("conductivity", "Electrical conductivity, mS/cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sample_id": "S-0001",
                    "collection_point": "Village Collection Center",
                    "fat_pct": 3.4,
                    "snf_pct": 8.7,
                    "density": 1.029,
                    "ph": 6.68,
                    "freezing_point": -0.53,
                    "conductivity": 4.85,
                }
            ]
        }
    }


Band = Literal["accept", "retest", "reject"]


class Prediction(BaseModel):
    sample_id: str | None = None
    is_adulterated: bool
    risk_score: float = Field(description="Calibrated probability of a detectable adulterant")
    band: Band
    adulterant: str | None = Field(None, description="Likely adulterant, only when flagged")
    confidence: float | None = Field(None, description="Stage 2 probability of `adulterant`")
    unusual_readings: list[str] = Field(
        default_factory=list,
        description="Readings outside what the model saw in training; such samples are "
        "never accepted, only retested",
    )
    model_version: str


class BatchRow(BaseModel):
    row: int = Field(description="0-based row number in the request")
    sample_id: str | None = None
    status: Literal["scored", "rejected"]
    is_adulterated: bool | None = None
    risk_score: float | None = None
    band: Band | None = None
    adulterant: str | None = None
    confidence: float | None = None
    unusual_readings: list[str] = Field(default_factory=list)
    reject_reason: str | None = None


class BatchSummary(BaseModel):
    rows: int
    scored: int
    rejected: int
    flagged: int
    flag_rate: float = Field(description="Flagged share of scored rows")
    bands: dict[str, int]
    adulterants: dict[str, int]


class BatchResponse(BaseModel):
    model_version: str
    summary: BatchSummary
    results: list[BatchRow]


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None

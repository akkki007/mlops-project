"""FastAPI prediction service.

Run: `uvicorn milk_adulteration.api.app:app`. The model loads at startup from
`MODEL_PATH` (default models/cascade.joblib) and its registry/metrics info from
`MODEL_INFO_PATH` (default model_info.json next to the model).
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from milk_adulteration import __version__
from milk_adulteration.api.metrics import Metrics
from milk_adulteration.api.schemas import BatchResponse, Health, Prediction, Sample
from milk_adulteration.config import FEATURES
from milk_adulteration.data.validate import split_valid_readings
from milk_adulteration.models.cascade import BANDS, Cascade

log = logging.getLogger(__name__)

MAX_BATCH_ROWS = 10_000
# Read identifiers as text so CSV IDs like "007" keep their leading zeros.
TEXT_COLUMNS = {"sample_id": str, "collection_point": str}
BATCH_BODY_DOC = {
    "requestBody": {
        "required": True,
        "description": f"Up to {MAX_BATCH_ROWS} samples as JSON (a list, or {{'samples': [...]}}),"
        " a CSV body (text/csv), or a CSV upload (multipart field `file`)."
        f" Required columns: {', '.join(FEATURES)}; optional: sample_id, collection_point.",
        "content": {
            "application/json": {"schema": {"type": "array", "items": {"type": "object"}}},
            "text/csv": {"schema": {"type": "string"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            },
        },
    }
}


def _none_if_nan(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def model_version(model: Cascade, info: dict[str, Any]) -> str:
    if info.get("model_version"):
        return info["model_version"]
    run = model.metadata.get("mlflow_run_id")
    return f"run-{run[:8]}" if run else "unversioned"


def create_app(model: Cascade | None = None, model_info: dict[str, Any] | None = None) -> FastAPI:
    """Build the app. Tests pass a model; production loads it from disk at startup."""
    state: dict[str, Any] = {"model": model, "info": model_info or {}}
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["model"] is None:
            path = Path(os.environ.get("MODEL_PATH", "models/cascade.joblib"))
            state["model"] = Cascade.load(path)
            info_path = Path(os.environ.get("MODEL_INFO_PATH", path.with_name("model_info.json")))
            if info_path.exists():
                state["info"] = json.loads(info_path.read_text())
            log.info("Loaded %s from %s", model_version(state["model"], state["info"]), path)
        metrics.model_info.labels(version=model_version(state["model"], state["info"])).set(1)
        yield

    app = FastAPI(
        title="Milk Adulteration Detection",
        version=__version__,
        description="Two-stage model: Stage 1 flags adulterated milk from six field "
        "readings; Stage 2 names the likely adulterant. It triages; a lab confirms.",
        lifespan=lifespan,
    )

    def get_model() -> Cascade:
        return state["model"]

    def version() -> str:
        return model_version(state["model"], state["info"])

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # template, keeps cardinality bounded
            if path != "/metrics":
                metrics.requests.labels(request.method, path, str(status)).inc()
                metrics.latency.labels(request.method, path).observe(time.perf_counter() - start)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", model_loaded=state["model"] is not None, model_version=version())

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model/info")
    def info() -> dict[str, Any]:
        m = get_model()
        return {
            **state["info"],
            "model_version": version(),
            "mlflow_run_id": state["info"].get("mlflow_run_id", m.metadata.get("mlflow_run_id")),
            "threshold": m.threshold,
            "reject_risk": m.reject_risk,
            "features": FEATURES,
            "derived_features": [f for f in m.features if f not in FEATURES],
            "adulterant_classes": m.classes,
            "stage1_exclude_other": m.metadata.get("stage1_exclude_other"),
        }

    @app.post("/predict", response_model=Prediction)
    def predict(sample: Sample) -> Prediction:
        pred = get_model().predict(pd.DataFrame([sample.model_dump()])).iloc[0]
        metrics.observe_predictions(pred.to_frame().T, [sample.collection_point])
        return Prediction(
            sample_id=sample.sample_id,
            is_adulterated=bool(pred["is_adulterated"]),
            risk_score=float(pred["risk_score"]),
            band=pred["band"],
            adulterant=pred["adulterant"],
            confidence=_none_if_nan(pred["confidence"]),
            model_version=version(),
        )

    async def read_batch(request: Request) -> pd.DataFrame:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        try:
            if ctype == "application/json":
                body = await request.json()
                records = body.get("samples") if isinstance(body, dict) else body
                if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                    raise HTTPException(
                        422, "JSON body must be a list of objects or {'samples': [...]}"
                    )
                df = pd.DataFrame.from_records(records)
            elif ctype in ("text/csv", "application/csv"):
                df = pd.read_csv(io.BytesIO(await request.body()), dtype=TEXT_COLUMNS)
            elif ctype == "multipart/form-data":
                upload = (await request.form()).get("file")
                if upload is None or isinstance(upload, str):
                    raise HTTPException(422, "multipart upload needs a CSV in field 'file'")
                df = pd.read_csv(io.BytesIO(await upload.read()), dtype=TEXT_COLUMNS)
            else:
                raise HTTPException(415, "use application/json, text/csv or multipart/form-data")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise HTTPException(400, f"could not parse body: {exc}") from exc
        if df.empty:
            raise HTTPException(422, "batch is empty")
        if len(df) > MAX_BATCH_ROWS:
            raise HTTPException(413, f"batch has {len(df)} rows; the limit is {MAX_BATCH_ROWS}")
        return df.reset_index(drop=True)

    def score_batch(df: pd.DataFrame) -> dict[str, Any]:
        try:
            valid, rejected = split_valid_readings(df)
        except KeyError as exc:
            raise HTTPException(422, str(exc.args[0])) from exc
        preds = get_model().predict(valid) if len(valid) else pd.DataFrame()
        ids = df["sample_id"] if "sample_id" in df else pd.Series([None] * len(df))
        cps = df["collection_point"] if "collection_point" in df else pd.Series([None] * len(df))
        if len(valid):
            metrics.observe_predictions(preds, cps.loc[valid.index].astype(object))
        metrics.rejected.labels("/predict/batch").inc(len(rejected))

        rows = []
        for i in df.index:
            sid = _none_if_nan(ids.iloc[i])
            base = {"row": int(i), "sample_id": None if sid is None else str(sid)}
            if i in preds.index:
                p = preds.loc[i]
                rows.append(
                    {
                        **base,
                        "status": "scored",
                        "is_adulterated": bool(p["is_adulterated"]),
                        "risk_score": float(p["risk_score"]),
                        "band": p["band"],
                        "adulterant": p["adulterant"],
                        "confidence": _none_if_nan(p["confidence"]),
                    }
                )
            else:
                reason = rejected.loc[i, "reject_reason"]
                rows.append({**base, "status": "rejected", "reject_reason": reason})

        flagged = int(preds["is_adulterated"].sum()) if len(preds) else 0
        bands = preds["band"].value_counts() if len(preds) else pd.Series(dtype=int)
        adulterants = preds["adulterant"].dropna().value_counts() if len(preds) else bands
        return {
            "model_version": version(),
            "summary": {
                "rows": len(df),
                "scored": len(preds),
                "rejected": len(rejected),
                "flagged": flagged,
                "flag_rate": flagged / len(preds) if len(preds) else 0.0,
                "bands": {b: int(bands.get(b, 0)) for b in BANDS},
                "adulterants": {k: int(v) for k, v in adulterants.items()},
            },
            "results": rows,
        }

    @app.post("/predict/batch", response_model=BatchResponse, openapi_extra=BATCH_BODY_DOC)
    async def predict_batch(request: Request) -> dict[str, Any]:
        df = await read_batch(request)
        return await run_in_threadpool(score_batch, df)

    return app


app = create_app()

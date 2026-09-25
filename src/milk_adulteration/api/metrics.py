"""Prometheus metrics for the API. One registry per app keeps tests isolated."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

MAX_LABEL_LEN = 64


def label(value: str | None) -> str:
    """Bound label values supplied by clients."""
    return (value or "unknown").strip()[:MAX_LABEL_LEN] or "unknown"


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "path"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )
        self.predictions = Counter(
            "milk_predictions_total",
            "Scored samples",
            ["band", "adulterant", "collection_point"],
            registry=self.registry,
        )
        self.rejected = Counter(
            "milk_rejected_rows_total",
            "Rows rejected by validation",
            ["endpoint"],
            registry=self.registry,
        )
        self.risk = Histogram(
            "milk_risk_score",
            "Calibrated risk score of scored samples",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "milk_model_info",
            "Loaded model (value is always 1)",
            ["version"],
            registry=self.registry,
        )

    def observe_predictions(self, preds, collection_points) -> None:
        for (_, p), cp in zip(preds.iterrows(), collection_points, strict=True):
            self.predictions.labels(
                band=p["band"], adulterant=p["adulterant"] or "none", collection_point=label(cp)
            ).inc()
            self.risk.observe(float(p["risk_score"]))

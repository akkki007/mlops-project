"""Smoke-test a running API: python scripts/smoke_test.py [base_url].

Uses only the standard library so CI can run it against the container.
Checks each endpoint, and that obviously watered milk is flagged while typical
pure milk is accepted.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

PURE = {
    "sample_id": "smoke-pure",
    "collection_point": "smoke-test",
    "fat_pct": 3.4,
    "snf_pct": 8.7,
    "density": 1.029,
    "ph": 6.68,
    "freezing_point": -0.53,
    "conductivity": 4.85,
}
# About 25% added water: every reading diluted toward water's values.
WATERED = {
    **PURE,
    "sample_id": "smoke-water",
    "fat_pct": 2.2,
    "snf_pct": 6.9,
    "density": 1.019,
    "ph": 6.9,
    "freezing_point": -0.46,
    "conductivity": 4.2,
}


def call(base: str, path: str, body: object | None = None, ctype: str = "application/json"):
    data = (
        None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    )
    req = urllib.request.Request(base + path, data=data, headers={"content-type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
            return r.status, (
                json.loads(raw) if "json" in r.headers.get("content-type", "") else raw
            )
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_ready(base: str, seconds: int = 60) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if call(base, "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f"API at {base} not healthy after {seconds}s")


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    wait_ready(base)
    failures = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail if not ok else ''}")
        if not ok:
            failures.append(name)

    status, health = call(base, "/health")
    check("health", status == 200 and health["model_loaded"], health)

    status, info = call(base, "/model/info")
    check("model info", status == 200 and info.get("mlflow_run_id"), info)

    status, pure = call(base, "/predict", PURE)
    check("pure milk accepted", status == 200 and pure["band"] == "accept", pure)

    status, water = call(base, "/predict", WATERED)
    check(
        "watered milk flagged as water",
        status == 200 and water["is_adulterated"] and water["adulterant"] == "water",
        water,
    )

    status, bad = call(base, "/predict", {**PURE, "ph": 12})
    check("out-of-range reading rejected", status == 422, bad)

    csv = "fat_pct,snf_pct,density,ph,freezing_point,conductivity\n"
    csv += (
        "3.4,8.7,1.029,6.68,-0.53,4.85\n2.2,6.9,1.019,6.9,-0.46,4.2\nabc,8.7,1.03,6.7,-0.53,4.8\n"
    )
    status, batch = call(base, "/predict/batch", csv.encode(), "text/csv")
    counts = ("rows", "scored", "rejected", "flagged")
    ok = status == 200 and tuple(batch["summary"][k] for k in counts) == (3, 2, 1, 1)
    check("batch CSV", ok, batch)

    status, metrics = call(base, "/metrics")
    check("metrics", status == 200 and "milk_predictions_total" in metrics, str(metrics)[:200])

    if failures:
        raise SystemExit(f"{len(failures)} smoke check(s) failed: {failures}")
    print("All smoke checks passed")


if __name__ == "__main__":
    main()

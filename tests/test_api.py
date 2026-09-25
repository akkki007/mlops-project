import io

import pytest
from fastapi.testclient import TestClient

from milk_adulteration.api.app import MAX_BATCH_ROWS, create_app
from milk_adulteration.config import FEATURES

INFO = {"model_version": "milk-adulteration-cascade/v7", "mlflow_run_id": "abc123"}
PURE = {
    "sample_id": "s1",
    "collection_point": "Farm Gate",
    "fat_pct": 3.4,
    "snf_pct": 8.8,
    "density": 1.030,
    "ph": 6.67,
    "freezing_point": -0.53,
    "conductivity": 4.85,
}


@pytest.fixture(scope="module")
def client(trained):
    model, _, _ = trained
    with TestClient(create_app(model, INFO)) as c:
        yield c


@pytest.fixture
def test_rows(trained):
    return trained[1]["test"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": INFO["model_version"],
    }


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["model_version"] == INFO["model_version"]
    assert body["mlflow_run_id"] == "abc123"
    assert body["features"] == FEATURES
    assert "fp_deviation" in body["derived_features"]
    assert "water" in body["adulterant_classes"]


def test_predict_matches_model(client, trained, test_rows):
    model = trained[0]
    row = test_rows.iloc[0]
    body = client.post("/predict", json={f: float(row[f]) for f in FEATURES}).json()
    expected = model.predict(test_rows.iloc[[0]]).iloc[0]
    assert body["is_adulterated"] == bool(expected["is_adulterated"])
    assert body["risk_score"] == pytest.approx(expected["risk_score"])
    assert body["band"] == expected["band"]
    assert body["model_version"] == INFO["model_version"]


def test_predict_response_shape(client):
    body = client.post("/predict", json=PURE).json()
    assert set(body) == {
        "sample_id",
        "is_adulterated",
        "risk_score",
        "band",
        "adulterant",
        "confidence",
        "model_version",
    }
    assert body["sample_id"] == "s1"
    if not body["is_adulterated"]:
        assert body["adulterant"] is None and body["confidence"] is None


@pytest.mark.parametrize(
    "change", [{"ph": 9.0}, {"density": 0.99}, {"fat_pct": -1}, {"conductivity": "high"}]
)
def test_predict_rejects_invalid_reading(client, change):
    r = client.post("/predict", json={**PURE, **change})
    assert r.status_code == 422
    assert list(change)[0] in r.text


def test_predict_requires_all_readings(client):
    r = client.post("/predict", json={k: v for k, v in PURE.items() if k != "snf_pct"})
    assert r.status_code == 422 and "snf_pct" in r.text


def _batch_rows(test_rows, n=5):
    return test_rows[["sample_id", "collection_point", *FEATURES]].head(n)


def test_batch_json_list_and_wrapped(client, test_rows):
    records = _batch_rows(test_rows).to_dict(orient="records")
    for body in (records, {"samples": records}):
        r = client.post("/predict/batch", json=body)
        assert r.status_code == 200
        s = r.json()["summary"]
        assert (s["rows"], s["scored"], s["rejected"]) == (5, 5, 0)
        assert sum(s["bands"].values()) == 5


def test_batch_csv_body_and_upload_with_bad_rows(client, test_rows):
    df = _batch_rows(test_rows, 4).astype({"fat_pct": object})
    df.loc[df.index[1], "fat_pct"] = "abc"
    df.loc[df.index[2], "ph"] = 11.0
    csv = df.to_csv(index=False).encode()
    responses = [
        client.post("/predict/batch", content=csv, headers={"content-type": "text/csv"}),
        client.post("/predict/batch", files={"file": ("batch.csv", io.BytesIO(csv), "text/csv")}),
    ]
    for r in responses:
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["summary"]["scored"] == 2 and body["summary"]["rejected"] == 2
        results = body["results"]
        assert [x["row"] for x in results] == [0, 1, 2, 3]
        assert results[1]["status"] == "rejected" and "not a number" in results[1]["reject_reason"]
        assert results[2]["status"] == "rejected" and "ph" in results[2]["reject_reason"]
        assert (
            results[0]["status"] == "scored" and results[0]["sample_id"] == df.iloc[0]["sample_id"]
        )


def test_batch_errors(client, test_rows):
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422
    assert client.post("/predict/batch", json=[]).status_code == 422
    missing = _batch_rows(test_rows).drop(columns="ph").to_dict(orient="records")
    r = client.post("/predict/batch", json=missing)
    assert r.status_code == 422 and "ph" in r.text
    r = client.post("/predict/batch", content=b"x", headers={"content-type": "text/plain"})
    assert r.status_code == 415
    too_many = [PURE] * (MAX_BATCH_ROWS + 1)
    assert client.post("/predict/batch", json=too_many).status_code == 413


def test_metrics_exposed(client):
    client.post("/predict", json=PURE)
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "milk_predictions_total" in text and 'collection_point="Farm Gate"' in text
    assert f'milk_model_info{{version="{INFO["model_version"]}"}} 1.0' in text
    assert "milk_rejected_rows_total" in text
    # Unknown paths share one label value.
    client.get("/nope")
    assert 'path="unmatched"' in client.get("/metrics").text

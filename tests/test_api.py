"""
Minimal smoke tests for the API. Run with: pytest -q

Not a full test suite -- just enough to prove /health and /predict work,
that missing fields don't crash the service, and that malformed input
returns a 4xx instead of a 500.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_request.json").read_text())
SAMPLE_ANOMALY = json.loads((ROOT / "sample_anomaly_request.json").read_text())


def test_health() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_predict_normal_window() -> None:
    with TestClient(app) as client:
        resp = client.post("/predict", json=SAMPLE)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["anomaly_score"] <= 1.0
        assert isinstance(body["is_anomaly"], bool)


def test_predict_flags_real_failure_window_as_anomalous() -> None:
    with TestClient(app) as client:
        resp = client.post("/predict", json=SAMPLE_ANOMALY)
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_anomaly"] is True
        assert body["anomaly_score"] > 0.5


def test_predict_handles_missing_fields() -> None:
    with TestClient(app) as client:
        payload = {"readings": [{"air_temperature": 298.1}, {"rotational_speed": 1400}]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["anomaly_score"] <= 1.0


def test_predict_rejects_empty_window() -> None:
    with TestClient(app) as client:
        resp = client.post("/predict", json={"readings": []})
        assert resp.status_code == 422  # pydantic min_length violation

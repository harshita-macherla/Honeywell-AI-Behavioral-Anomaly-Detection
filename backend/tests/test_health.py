"""Tests for GET /health and GET /api/v1/status."""


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_reports_all_models_loaded(client):
    resp = client.get("/api/v1/status")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["dataset_rows_loaded"] == 60007
    assert body["entities_indexed"] == 685
    assert body["uptime_seconds"] >= 0

    expected_models = {
        "Isolation Forest v2",
        "Feature Scaler v2",
        "XGBoost Attack Classifier v2",
        "Label Encoder v2",
        "LSTM Autoencoder v2",
    }
    reported_models = {m["name"] for m in body["models"]}
    assert reported_models == expected_models
    assert all(m["loaded"] is True for m in body["models"])

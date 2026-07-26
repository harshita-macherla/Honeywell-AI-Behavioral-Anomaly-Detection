"""Tests for POST /api/v1/predict (live Stage 1 -> Stage 2 -> Risk scoring)."""


def test_predict_known_attack_event_scores_high(client, attack_predict_payload):
    resp = client.post("/api/v1/predict", json=attack_predict_payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["entity_id"] == attack_predict_payload["entity_id"]
    assert body["cold_start"] is False
    assert 0.0 <= body["isolation_forest_score"] <= 1.0
    assert 0.0 <= body["lstm_reconstruction_score"] <= 1.0
    assert 0.0 <= body["fused_anomaly_score"] <= 1.0
    assert 0.0 <= body["fused_anomaly_score_percentile"] <= 1.0
    assert body["risk_level"] in {"Critical", "High", "Medium", "Low"}
    assert 0 <= body["risk_score"] <= 100
    assert isinstance(body["reasons"], list)


def test_predict_known_benign_event_returns_valid_response(client, benign_predict_payload):
    resp = client.post("/api/v1/predict", json=benign_predict_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_level"] in {"Critical", "High", "Medium", "Low"}
    assert 0 <= body["risk_score"] <= 100


def test_predict_cold_start_entity_still_scores(client, cold_start_predict_payload):
    resp = client.post("/api/v1/predict", json=cold_start_predict_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cold_start"] is True
    assert body["history_events_used"] == 0
    # A cold-start entity must still produce a complete, valid response --
    # this is the exact path build_sequences()'s zero-padding convention
    # covers (see scripts/train_stage1_anomaly_detection_v2.py).
    assert body["risk_level"] in {"Critical", "High", "Medium", "Low"}


def test_predict_rejects_missing_features(client, attack_predict_payload):
    bad_payload = dict(attack_predict_payload)
    bad_features = dict(bad_payload["features"])
    del bad_features["historical_event_count"]
    bad_payload["features"] = bad_features

    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422
    assert "historical_event_count" in resp.json()["detail"]["missing_features"]


def test_predict_rejects_unexpected_extra_features(client, attack_predict_payload):
    bad_payload = dict(attack_predict_payload)
    bad_features = dict(bad_payload["features"])
    bad_features["some_made_up_feature"] = 1.0
    bad_payload["features"] = bad_features

    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422
    assert "some_made_up_feature" in resp.json()["detail"]["unexpected_features"]


def test_predict_rejects_missing_required_top_level_field(client, attack_predict_payload):
    bad_payload = dict(attack_predict_payload)
    del bad_payload["resource_sensitivity"]
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422  # Pydantic schema validation, before scoring_service even runs


def test_predict_mfa_rule_is_entity_type_aware(client, attack_predict_payload):
    """
    Regression test for the entity-type-aware MFA rule fix made during the
    Risk Scoring Engine v2 milestone: the "MFA Not Used" rule must fire for
    a human user with mfa_used=False, but must NOT fire for a non-human
    entity with mfa_used=False (device auth doesn't use MFA in this
    dataset's design -- see risk_scoring_engine_v2.py's RULES comment).
    """
    user_payload = dict(attack_predict_payload)
    user_payload["entity_type"] = "user"
    user_payload["mfa_used"] = False
    resp_user = client.post("/api/v1/predict", json=user_payload)
    assert resp_user.status_code == 200
    assert "MFA Not Used" in resp_user.json()["triggered_rules"]

    device_payload = dict(attack_predict_payload)
    device_payload["entity_type"] = "iot_device"
    device_payload["mfa_used"] = False
    resp_device = client.post("/api/v1/predict", json=device_payload)
    assert resp_device.status_code == 200
    assert "MFA Not Used" not in resp_device.json()["triggered_rules"]

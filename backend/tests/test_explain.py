"""Tests for GET /api/v1/explain/{log_id}."""


def test_explain_known_critical_alert(client, known_critical_alert):
    resp = client.get(f"/api/v1/explain/{known_critical_alert['log_id']}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["log_id"] == known_critical_alert["log_id"]
    assert body["risk_level"] == "Critical"
    assert body["predicted_attack_type"] is not None
    # This fixture is specifically a correctly-classified alert.
    assert body["predicted_attack_type"] == body["actual_attack_type"]

    assert isinstance(body["rule_reasons"], list)
    assert isinstance(body["shap_reasons"], list)
    assert isinstance(body["merged_reasons"], list)
    assert 0 < len(body["merged_reasons"]) <= 5

    # SHAP contributions: real signed floats, ranked by magnitude, capped at 10.
    contributions = body["top_shap_contributions"]
    assert 0 < len(contributions) <= 10
    magnitudes = [abs(c["shap_value"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for c in contributions:
        assert set(c.keys()) == {"feature", "readable_name", "shap_value"}
        assert isinstance(c["shap_value"], float)


def test_explain_404_for_unknown_log_id(client):
    resp = client.get("/api/v1/explain/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]

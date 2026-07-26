"""
Tests for the analyst dashboard endpoints: GET /api/v1/alerts,
GET /api/v1/alerts/{log_id}, GET /api/v1/entities/{entity_id}/history,
GET /api/v1/stats/overview.

Several assertions below hardcode exact counts (e.g. risk_level_counts).
This is intentional, not brittleness: the underlying dataset
(dataset/processed/risk_scores_v2.csv) is frozen -- this milestone's
instructions explicitly say to reuse it exactly as-is, not regenerate it
-- so its true risk-level distribution is a fixed, known quantity. Asserting
the exact values is precisely the kind of regression test that would have
caught the risk_level_for() boundary-gap bug found and fixed during the
Backend API milestone (55 rows were silently mislabeled "Low"); a looser
assertion (e.g. "count > 0") would not have caught it.
"""


def test_list_alerts_default_pagination(client):
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 60007
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["alerts"]) == 50
    # Default sort is risk_score descending -- the ranked alert queue.
    scores = [a["risk_score"] for a in body["alerts"]]
    assert scores == sorted(scores, reverse=True)


def test_list_alerts_filter_by_risk_level_matches_known_counts(client):
    resp = client.get("/api/v1/alerts", params={"risk_level": "Critical", "limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1263
    assert body["alerts"][0]["risk_level"] == "Critical"


def test_list_alerts_filter_by_high_risk_level_matches_fixed_count(client):
    # Regression test for the risk_level_for() boundary-gap bug: before the
    # fix, this count was 222 (51 rows incorrectly fell into "Low").
    resp = client.get("/api/v1/alerts", params={"risk_level": "High", "limit": 1})
    assert resp.status_code == 200
    assert resp.json()["total"] == 273


def test_list_alerts_filter_by_entity_type(client):
    resp = client.get("/api/v1/alerts", params={"entity_type": "iot_device", "limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4535


def test_list_alerts_combined_filters(client):
    resp = client.get("/api/v1/alerts", params={"risk_level": "Critical", "entity_type": "iot_device", "limit": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 61
    assert all(a["risk_level"] == "Critical" and a["entity_type"] == "iot_device" for a in body["alerts"])


def test_list_alerts_pagination_offset(client):
    page0 = client.get("/api/v1/alerts", params={"limit": 10, "offset": 0}).json()
    page1 = client.get("/api/v1/alerts", params={"limit": 10, "offset": 10}).json()
    assert page0["alerts"][0]["log_id"] != page1["alerts"][0]["log_id"]


def test_get_alert_detail(client, known_critical_alert):
    resp = client.get(f"/api/v1/alerts/{known_critical_alert['log_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["log_id"] == known_critical_alert["log_id"]
    assert body["entity_id"] == known_critical_alert["entity_id"]
    assert body["risk_level"] == "Critical"


def test_get_alert_detail_404_for_unknown_log_id(client):
    resp = client.get("/api/v1/alerts/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_entity_history_returns_events(client, known_entity_id):
    resp = client.get(f"/api/v1/entities/{known_entity_id}/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_id"] == known_entity_id
    assert body["total_events"] > 1
    # Default limit is 100 -- events returned is capped at that even if the
    # entity has more total events than that (this dataset's entities have
    # up to ~145 events each, so this must not assume total_events <= 100).
    assert len(body["events"]) == min(body["total_events"], 100)
    # Events are returned most-recent-first.
    timestamps = [e["timestamp"] for e in body["events"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_entity_history_respects_limit(client, known_entity_id):
    resp = client.get(f"/api/v1/entities/{known_entity_id}/history", params={"limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()["events"]) == 1


def test_entity_history_404_for_unknown_entity(client):
    resp = client.get("/api/v1/entities/DOES_NOT_EXIST_999/history")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_stats_overview_matches_known_corrected_distribution(client):
    resp = client.get("/api/v1/stats/overview")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_events"] == 60007
    assert body["total_entities"] == 685
    assert body["risk_level_counts"] == {
        "Critical": 1263,
        "High": 273,
        "Medium": 16821,
        "Low": 41650,
    }
    assert body["critical_alert_precision"] == 0.9873

    entity_types_covered = {row["entity_type"] for row in body["entity_type_breakdown"]}
    assert entity_types_covered == {
        "user", "service_account", "edge_device", "iot_device", "industrial_controller", "server",
    }

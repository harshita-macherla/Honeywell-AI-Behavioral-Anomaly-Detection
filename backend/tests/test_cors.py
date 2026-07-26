"""
Regression tests for the CORS bug found and fixed during the React
Analyst Dashboard milestone: the backend originally had no CORS
configuration, which silently blocked every browser request from the
frontend's origin despite the API itself working correctly.
"""


def test_cors_preflight_allows_localhost_origin(client):
    resp = client.options(
        "/api/v1/status",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_cors_preflight_allows_localhost_hostname(client):
    resp = client.options(
        "/api/v1/status",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_actual_get_request_includes_cors_header(client):
    resp = client.get("/api/v1/status", headers={"Origin": "http://127.0.0.1:5173"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

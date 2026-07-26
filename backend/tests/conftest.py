"""
conftest.py
===========
Shared pytest fixtures for the backend API test suite.

The `client` fixture is SESSION-scoped and wraps FastAPI's TestClient as a
context manager, which triggers the app's real `lifespan` handler
(`load_state()` in backend/app/dependencies.py) exactly once for the whole
test run -- the same model/dataset loading path used in production, not a
mocked substitute. This is deliberate: loading the v2 models + building
the SHAP TreeExplainer + computing historical LSTM-AE baselines takes
~60-90s (dominated by TensorFlow import + inference over 60,007 rows), so
re-running it per test would make the suite impractically slow AND would
not actually test anything the session-scoped load doesn't already cover.

No backend or ML code is imported, patched, or reimplemented here --
fixtures only read the SAME frozen dataset/processed/risk_scores_v2.csv
the app itself loads, to build realistic request payloads and to assert
against known-correct values.
"""

import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Make `backend` importable as a package regardless of the cwd pytest is
# invoked from (consistent with how backend/app/dependencies.py adds
# scripts/ to sys.path for the same reason).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.main import app  # noqa: E402
from backend.app import config  # noqa: E402
from backend.app.dependencies import stage1_v2  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient -- triggers real app startup/shutdown once."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def dataset_df():
    """
    The same frozen dataset the app loads, read directly here so fixtures
    below can build realistic /predict payloads and other tests can assert
    against ground truth without depending on the app's in-memory state.
    """
    return pd.read_csv(config.RISK_SCORES_V2_PATH, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def known_critical_alert(dataset_df):
    """A real, known Critical-tier, correctly-classified alert (log_id) to test /alerts/{id} and /explain/{id} against."""
    row = dataset_df[
        (dataset_df["risk_level"] == "Critical")
        & (dataset_df["predicted_attack_type"] == dataset_df["attack_type"])
    ].iloc[0]
    return row


@pytest.fixture(scope="session")
def known_entity_id(dataset_df):
    """A real entity_id with more than one recorded event, for /entities/{id}/history tests."""
    counts = dataset_df["entity_id"].value_counts()
    return counts[counts > 1].index[0]


def _row_to_predict_payload(row, dataset_df):
    """
    Builds a valid POST /api/v1/predict payload from a real dataset row --
    reuses stage1_v2.NUMERIC_FEATURES (the same 86-feature list the app
    itself validates against) rather than hardcoding a 7th copy of it.
    """
    features = {f: float(row[f]) for f in stage1_v2.NUMERIC_FEATURES}
    return {
        "entity_id": row["entity_id"],
        "entity_type": row["entity_type"],
        "timestamp": str(row["timestamp"]),
        "resource_sensitivity": int(row["resource_sensitivity"]),
        "failed_login_count": int(row["failed_login_count"]),
        "mfa_used": bool(row["mfa_used"]),
        "features": features,
    }


@pytest.fixture(scope="session")
def attack_predict_payload(dataset_df):
    """A real, known attack event's features, as a /predict request body."""
    row = dataset_df[dataset_df["label_is_attack"] == 1].iloc[0]
    return _row_to_predict_payload(row, dataset_df)


@pytest.fixture(scope="session")
def benign_predict_payload(dataset_df):
    """A real, known benign event's features, as a /predict request body."""
    row = dataset_df[dataset_df["label_is_attack"] == 0].iloc[0]
    return _row_to_predict_payload(row, dataset_df)


@pytest.fixture(scope="session")
def cold_start_predict_payload(attack_predict_payload):
    """Same real feature values, but with an entity_id the app has never seen -- exercises the cold-start path."""
    payload = dict(attack_predict_payload)
    payload["entity_id"] = "TEST_COLD_START_ENTITY_ZZZ"
    return payload

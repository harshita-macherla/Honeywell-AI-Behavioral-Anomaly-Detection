"""
dependencies.py
================
Loads every v2 artifact ONCE at application startup and reuses the
existing pipeline scripts (scripts/train_stage1_anomaly_detection_v2.py,
scripts/train_stage2_classification_v2.py, scripts/risk_scoring_engine_v2.py)
as ordinary Python modules -- their NUMERIC_FEATURES / CLASSIFIER_FEATURES /
RULES / compute_risk_score() / get_shap_reasons() / etc. are imported and
called directly, not re-typed a fourth time. This is the "build the backend
around the existing artifacts" requirement: no model is retrained, no
feature list is duplicated, no rule is redefined.

Verified safe to import: all three scripts guard their actual pipeline
execution (data loading, training, CSV writes) behind
`if __name__ == "__main__":`, so importing them only defines constants and
functions -- confirmed during integration testing (see milestone notes).
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from xgboost import XGBClassifier

from . import config

# Make scripts/ importable as ordinary modules (train_stage1_anomaly_detection_v2,
# train_stage2_classification_v2, risk_scoring_engine_v2) without moving or
# modifying those files.
if config.SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, config.SCRIPTS_DIR)

import train_stage1_anomaly_detection_v2 as stage1_v2   # noqa: E402
import train_stage2_classification_v2 as stage2_v2       # noqa: E402
import risk_scoring_engine_v2 as risk_v2                 # noqa: E402


class AppState:
    """
    Holds every loaded model, the historical scored dataset, and a
    per-entity index for LSTM sequence lookups. Instantiated once by
    `load_state()` in the FastAPI lifespan handler in main.py.
    """

    def __init__(self):
        self.start_time = time.time()

        self.isolation_forest = None
        self.feature_scaler = None
        self.lstm_autoencoder = None
        self.xgb_classifier = None
        self.label_encoder = None

        self.dataset: pd.DataFrame = None          # full risk_scores_v2.csv, in memory
        self.entity_index: dict = {}                # entity_id -> sorted list of positional row indices
        self.model_status: list = []

        # Historical baselines needed to normalize a LIVE (n=1) event the
        # same way the batch scripts normalized the full historical
        # dataset -- see the module docstring in services/scoring_service.py
        # for why this is necessary rather than optional.
        self.historical_scaled_features: np.ndarray = None  # (n_rows, 86), row-aligned with self.dataset
        self.if_raw_min: float = None
        self.if_raw_max: float = None
        self.lstm_mse_min: float = None
        self.lstm_mse_max: float = None
        self.shap_explainer = None

    def load(self):
        # --- Load v2 models (trained by prior milestones; never retrained here) ---
        self.isolation_forest = self._load_joblib("Isolation Forest v2", config.ISOLATION_FOREST_V2_PATH)
        self.feature_scaler = self._load_joblib("Feature Scaler v2", config.FEATURE_SCALER_V2_PATH)
        self.label_encoder = self._load_joblib("Label Encoder v2", config.LABEL_ENCODER_V2_PATH)

        # CHANGED: loaded via XGBoost's native JSON format instead of
        # joblib -- the joblib artifact repeatedly failed to load after
        # download ("XGBoostError: input stream corrupted") despite
        # loading correctly server-side; XGBoost's own save_model/
        # load_model round-trip was verified to produce byte-for-byte
        # identical predict()/predict_proba() output before this switch.
        try:
            self.xgb_classifier = XGBClassifier()
            self.xgb_classifier.load_model(config.XGB_CLASSIFIER_V2_PATH)
            self.model_status.append({"name": "XGBoost Attack Classifier v2", "loaded": True, "path": config.XGB_CLASSIFIER_V2_PATH})
        except Exception as exc:
            self.model_status.append({"name": "XGBoost Attack Classifier v2", "loaded": False, "path": str(exc)})
            raise

        try:
            self.lstm_autoencoder = tf.keras.models.load_model(config.LSTM_AUTOENCODER_V2_PATH)
            self.model_status.append({"name": "LSTM Autoencoder v2", "loaded": True, "path": config.LSTM_AUTOENCODER_V2_PATH})
        except Exception as exc:
            self.model_status.append({"name": "LSTM Autoencoder v2", "loaded": False, "path": str(exc)})
            raise

        # --- Load the fully risk-scored v2 dataset (analyst endpoints + LSTM history lookups) ---
        self.dataset = pd.read_csv(config.RISK_SCORES_V2_PATH, parse_dates=["timestamp"])
        # attack_type / predicted_attack_type use the real-NaN-for-benign
        # convention established in train_stage2_classification_v2.py /
        # risk_scoring_engine_v2.py -- keep that convention in memory too.

        # Build a per-entity, time-sorted index of positional row numbers so
        # a live /predict call can cheaply fetch an entity's most recent
        # events for LSTM sequence construction, without re-scanning the
        # full 60k-row dataset per request.
        self.dataset = self.dataset.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        for entity_id, group in self.dataset.groupby("entity_id"):
            self.entity_index[entity_id] = group.index.tolist()

        # --- Historical baselines for online (n=1) score normalization ---
        # Reuses stage1_v2.NUMERIC_FEATURES (feature order) and
        # stage1_v2.build_sequences (exact same sequencing logic used at
        # training time) rather than reimplementing either.
        raw_matrix = self.dataset[stage1_v2.NUMERIC_FEATURES].values.astype(np.float64)
        self.historical_scaled_features = self.feature_scaler.transform(raw_matrix).astype(np.float32)

        if_raw_hist = -self.isolation_forest.decision_function(self.historical_scaled_features)
        self.if_raw_min = float(if_raw_hist.min())
        self.if_raw_max = float(if_raw_hist.max())

        historical_sequences = stage1_v2.build_sequences(
            self.dataset, self.historical_scaled_features, stage1_v2.SEQUENCE_LENGTH
        )
        reconstructed_hist = self.lstm_autoencoder.predict(historical_sequences, batch_size=512, verbose=0)
        mse_hist = np.mean(np.square(historical_sequences - reconstructed_hist), axis=(1, 2))
        self.lstm_mse_min = float(mse_hist.min())
        self.lstm_mse_max = float(mse_hist.max())

        # --- SHAP explainer built once (TreeExplainer construction parses
        # the whole tree ensemble; reusing one instance across requests is
        # what makes /predict and /explain fast per-call) ---
        self.shap_explainer = shap.TreeExplainer(self.xgb_classifier)

    def _load_joblib(self, name, path):
        try:
            obj = joblib.load(path)
            self.model_status.append({"name": name, "loaded": True, "path": path})
            return obj
        except Exception as exc:
            self.model_status.append({"name": name, "loaded": False, "path": str(exc)})
            raise

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


_state: AppState = None


def load_state() -> AppState:
    global _state
    _state = AppState()
    _state.load()
    return _state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("Application state not loaded yet -- load_state() must run at startup.")
    return _state

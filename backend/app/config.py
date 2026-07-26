"""
config.py
=========
Central path configuration for the backend. Points at the SAME artifacts
the ML pipeline milestones already produced -- no paths here create new
data locations; they all resolve into the existing project/dataset and
project/models directories built by:
    scripts/train_stage1_anomaly_detection_v2.py
    scripts/train_stage2_classification_v2.py
    scripts/risk_scoring_engine_v2.py

PROJECT_ROOT / SCRIPTS_DIR is also added to sys.path by dependencies.py so
the backend can import those scripts as ordinary Python modules and reuse
their feature lists, rule definitions, and helper functions directly,
instead of maintaining a fourth duplicate copy of the same constants.
"""

import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../project/backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                                 # .../project

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

# v2 model artifacts (produced by prior milestones -- loaded, never retrained)
ISOLATION_FOREST_V2_PATH = os.path.join(MODELS_DIR, "isolation_forest_v2.joblib")
FEATURE_SCALER_V2_PATH = os.path.join(MODELS_DIR, "feature_scaler_v2.joblib")
LSTM_AUTOENCODER_V2_PATH = os.path.join(MODELS_DIR, "lstm_autoencoder_v2.keras")
# CHANGED: xgb_attack_classifier_v2.joblib repeatedly failed to load after
# download ("XGBoostError: input stream corrupted") despite loading
# correctly server-side and passing MD5-stable regeneration -- consistent
# with joblib's pickle-based container being sensitive to transfer-layer
# corruption in a way XGBoost's own serialization format is not. Switched
# to XGBoost's native JSON format (produced via model.save_model(...),
# loaded via XGBClassifier().load_model(...) in dependencies.py) as a
# transfer-robust, self-describing alternative. Verified byte-for-byte
# identical predict()/predict_proba() output against the original
# joblib-loaded model before switching.
XGB_CLASSIFIER_V2_PATH = os.path.join(MODELS_DIR, "xgb_attack_classifier_v2.json")
LABEL_ENCODER_V2_PATH = os.path.join(MODELS_DIR, "label_encoder_v2.joblib")

# v2 scored dataset (produced by risk_scoring_engine_v2.py) -- the single
# source of truth for both (a) analyst/dashboard endpoints and (b) the
# historical-sequence lookups the LSTM-AE needs for live scoring.
RISK_SCORES_V2_PATH = os.path.join(DATASET_PROCESSED_DIR, "risk_scores_v2.csv")

API_TITLE = "Behavioral Anomaly Detection API (v2)"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Backend for the Honeywell AI Hackathon Behavioral Anomaly Detection "
    "project. Serves the trained v2 pipeline (Isolation Forest + LSTM "
    "Autoencoder -> XGBoost attack classifier -> rule+SHAP risk scoring "
    "engine) for live event scoring, explanation, and analyst dashboard "
    "queries. Models are loaded once at startup; none are retrained here."
)

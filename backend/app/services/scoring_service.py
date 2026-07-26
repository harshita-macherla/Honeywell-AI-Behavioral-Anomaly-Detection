"""
scoring_service.py
====================
Live, single-event scoring: Stage 1 v2 (Isolation Forest + LSTM-AE fusion)
-> Stage 2 v2 (XGBoost attack classification) -> Risk Scoring v2 (rules +
SHAP), reusing the SAME functions, constants, and trained model objects the
batch pipeline milestones already built and validated. Nothing here is a
reimplementation of that logic -- it is the same logic, invoked per-event
instead of per-DataFrame.

WHY A NEW MODULE INSTEAD OF EDITING THE SCRIPTS
----------------------------------------------------------------------------
The existing scripts (train_stage1_anomaly_detection_v2.py,
train_stage2_classification_v2.py, risk_scoring_engine_v2.py) are written
as BATCH jobs: they normalize Isolation Forest / LSTM-AE scores via
min-max over the WHOLE input array they're given
(`(raw - raw.min()) / (raw.max() - raw.min())`), and risk_scoring_engine_v2.py
percentile-ranks fused_anomaly_score via `.rank(pct=True)` over the whole
dataset. Both of those normalizations are mathematically undefined (or
trivially 0) for a single-row batch of size 1 -- there is no "min" and
"max" across one value. This is a genuine architectural fact about batch
vs. online normalization, not a bug in the batch scripts (which were never
meant to run on n=1). The fix applied here is to normalize each new live
event against the HISTORICAL min/max/percentile distribution (computed
once at startup from the same trained models on the same historical data)
instead of against itself -- this is the standard way batch-trained
anomaly-score normalizers are served online, and it keeps a live score
directly comparable to the historical risk_scores_v2.csv scores.
"""

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .. import config
from ..dependencies import AppState, stage1_v2, stage2_v2, risk_v2
from ..schemas import PredictRequest


def _validate_features(features: dict) -> None:
    expected = set(stage1_v2.NUMERIC_FEATURES)
    got = set(features.keys())
    missing = expected - got
    extra = got - expected
    if missing or extra:
        detail = {}
        if missing:
            detail["missing_features"] = sorted(missing)
        if extra:
            detail["unexpected_features"] = sorted(extra)
        raise HTTPException(status_code=422, detail=detail)


def _historical_lookup(state: AppState, entity_id: str, seq_len: int):
    """
    Returns (prior_scaled_vectors, n_found) -- the entity's most recent up
    to (seq_len - 1) historical SCALED feature vectors, oldest-first, ready
    to be prepended to the new event for LSTM-AE sequencing. Empty for a
    genuinely new (cold-start) entity_id.
    """
    idxs = state.entity_index.get(entity_id, [])
    recent_idxs = idxs[-(seq_len - 1):]
    if not recent_idxs:
        return np.zeros((0, len(stage1_v2.NUMERIC_FEATURES)), dtype=np.float32), 0
    return state.historical_scaled_features[recent_idxs], len(recent_idxs)


def score_event(state: AppState, req: PredictRequest) -> dict:
    _validate_features(req.features)

    seq_len = stage1_v2.SEQUENCE_LENGTH
    x_raw_row = np.array([[req.features[f] for f in stage1_v2.NUMERIC_FEATURES]], dtype=np.float64)
    x_scaled_row = state.feature_scaler.transform(x_raw_row).astype(np.float32)  # (1, 86)

    # ---------------- Stage 1: Isolation Forest (point anomaly) ----------------
    if_raw = -state.isolation_forest.decision_function(x_scaled_row)[0]
    if_score = float(
        np.clip((if_raw - state.if_raw_min) / (state.if_raw_max - state.if_raw_min + 1e-9), 0.0, 1.0)
    )

    # ---------------- Stage 1: LSTM Autoencoder (sequence anomaly) ----------------
    prior_vectors, n_found = _historical_lookup(state, req.entity_id, seq_len)
    window = np.vstack([prior_vectors, x_scaled_row]) if n_found else x_scaled_row
    pad_len = seq_len - window.shape[0]
    if pad_len > 0:
        window = np.vstack([np.zeros((pad_len, window.shape[1]), dtype=np.float32), window])
    sequence = window[np.newaxis, :, :]  # (1, seq_len, 86)

    reconstructed = state.lstm_autoencoder.predict(sequence, verbose=0)
    lstm_raw = float(np.mean(np.square(sequence - reconstructed)))
    lstm_score = float(
        np.clip((lstm_raw - state.lstm_mse_min) / (state.lstm_mse_max - state.lstm_mse_min + 1e-9), 0.0, 1.0)
    )

    # ---------------- Stage 1 fusion (reuses the v2-corrected weights) ----------------
    fused_score = stage1_v2.FUSION_WEIGHT_IF * if_score + stage1_v2.FUSION_WEIGHT_LSTM * lstm_score
    fused_percentile = float((state.dataset["fused_anomaly_score"] < fused_score).mean())

    # ---------------- Stage 2: XGBoost attack classification ----------------
    classifier_row = {**req.features, "isolation_forest_score": if_score,
                       "lstm_reconstruction_score": lstm_score, "fused_anomaly_score": fused_score}
    x_classifier = np.array([[classifier_row[f] for f in stage2_v2.CLASSIFIER_FEATURES]], dtype=np.float64)

    pred_idx = int(state.xgb_classifier.predict(x_classifier)[0])
    pred_proba = state.xgb_classifier.predict_proba(x_classifier)[0]
    confidence = float(pred_proba.max())
    predicted_attack_type = state.label_encoder.inverse_transform([pred_idx])[0]

    shap_values = state.shap_explainer.shap_values(x_classifier)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    row_shap = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]
    shap_reasons = risk_v2.get_shap_reasons(row_shap, stage2_v2.CLASSIFIER_FEATURES)

    # ---------------- Risk scoring: rules + SHAP merge ----------------
    rule_row = pd.Series({**req.features,
                           "entity_type": req.entity_type,
                           "resource_sensitivity": req.resource_sensitivity,
                           "failed_login_count": req.failed_login_count,
                           "mfa_used": req.mfa_used})
    rule_score, rule_reasons = risk_v2.compute_rule_score(rule_row)

    risk_score = risk_v2.compute_risk_score(
        fused_anomaly_score_scaled=fused_percentile,
        prediction_confidence=confidence,
        predicted_attack_type=predicted_attack_type,
        rule_score=rule_score,
    )
    risk_level = risk_v2.risk_level_for(risk_score)
    merged_reasons = risk_v2.merge_reasons(rule_reasons, shap_reasons)

    return {
        "entity_id": req.entity_id,
        "entity_type": req.entity_type,
        "timestamp": req.timestamp,
        "cold_start": n_found < (seq_len - 1),
        "history_events_used": n_found,
        "isolation_forest_score": round(if_score, 4),
        "lstm_reconstruction_score": round(lstm_score, 4),
        "fused_anomaly_score": round(fused_score, 4),
        "fused_anomaly_score_percentile": round(fused_percentile, 4),
        "predicted_attack_type": None if predicted_attack_type == "None" else predicted_attack_type,
        "prediction_confidence": round(confidence, 4),
        "rule_based_score": rule_score,
        "triggered_rules": rule_reasons,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": merged_reasons,
    }

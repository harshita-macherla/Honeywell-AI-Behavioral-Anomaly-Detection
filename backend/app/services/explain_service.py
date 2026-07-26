"""
explain_service.py
====================
Explanation lookups for events already scored by the batch pipeline
(risk_scoring_engine_v2.py's output, held in memory as `state.dataset`).
Re-derives per-feature SHAP contributions live (cheap for a single row,
using the SAME TreeExplainer instance built once at startup) rather than
persisting raw SHAP arrays to disk -- risk_scores_v2.csv already stores
the merged reasons_str, but not per-feature signed SHAP values, so this
recomputation is what surfaces the full "why" behind reasons_str for a
single alert on demand.
"""

import numpy as np
import pandas as pd
from fastapi import HTTPException

from ..dependencies import AppState, stage2_v2, risk_v2


def explain_log_id(state: AppState, log_id: str) -> dict:
    matches = state.dataset[state.dataset["log_id"] == log_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"log_id '{log_id}' not found")
    row = matches.iloc[0]

    x_row = np.array([[row[f] for f in stage2_v2.CLASSIFIER_FEATURES]], dtype=np.float64)
    predicted_attack_type = row["predicted_attack_type"] if pd.notna(row["predicted_attack_type"]) else "None"
    pred_idx = state.label_encoder.transform([predicted_attack_type])[0]

    shap_values = state.shap_explainer.shap_values(x_row)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    row_shap = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

    shap_reasons = risk_v2.get_shap_reasons(row_shap, stage2_v2.CLASSIFIER_FEATURES)
    rule_score, rule_reasons = risk_v2.compute_rule_score(row)
    merged_reasons = risk_v2.merge_reasons(rule_reasons, shap_reasons)

    contributions = sorted(
        zip(stage2_v2.CLASSIFIER_FEATURES, row_shap.tolist()),
        key=lambda x: abs(x[1]), reverse=True,
    )[:10]
    top_shap_contributions = [
        {
            "feature": feat,
            "readable_name": risk_v2.SHAP_READABLE_NAMES.get(feat, feat),
            "shap_value": round(float(val), 5),
        }
        for feat, val in contributions
    ]

    return {
        "log_id": row["log_id"],
        "entity_id": row["entity_id"],
        "entity_type": row["entity_type"],
        "timestamp": str(row["timestamp"]),
        "predicted_attack_type": None if predicted_attack_type == "None" else predicted_attack_type,
        "actual_attack_type": row["attack_type"] if pd.notna(row["attack_type"]) else None,
        "prediction_confidence": round(float(row["prediction_confidence"]), 4),
        "risk_score": float(row["risk_score"]),
        "risk_level": row["risk_level"],
        "rule_reasons": rule_reasons,
        "shap_reasons": shap_reasons,
        "merged_reasons": merged_reasons,
        "top_shap_contributions": top_shap_contributions,
    }

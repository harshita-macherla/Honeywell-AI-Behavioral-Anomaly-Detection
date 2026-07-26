"""
risk_scoring_engine.py
========================
Combines THREE independent signals into one 0-100 risk score per event,
and generates a ranked, human-readable list of reasons for it:

    1. Stage 1 fused anomaly score   (unsupervised "how weird is this")
    2. Stage 2 classification confidence  (supervised "how sure are we
       this is a specific attack type")
    3. Rule-based factors             (explicit, auditable security rules
       that a SOC analyst would recognize immediately, independent of any
       model -- this is deliberate defense-in-depth: if the ML pipeline
       ever mis-scores something, blatant rule violations like "sensitive
       resource + no MFA + midnight login" still surface as risk)

Explainability layers (per the problem statement's requirement to use
SHAP + feature importance + rule-based explanations together):
    - SHAP (TreeExplainer on the Stage 2 XGBoost model) -> "why did the
      model think this was attack type X"
    - Rule-based flags -> "which explicit security rules were violated"
    - The two are merged into one ranked reasons list per event, capped at
      the 5 most significant reasons (matches the dashboard's card format).

Output: dataset/processed/risk_scores.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

STAGE2_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "stage2_predictions.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "risk_scores.csv")

CLASSIFIER_FEATURES = [
    "distance_from_prev_km", "implied_speed_kmh", "impossible_travel_flag",
    "is_new_device", "is_new_os", "is_new_browser",
    "login_hour_deviation", "is_odd_hour_login", "cold_start_flag",
    "is_cross_department_access", "is_sensitive_resource",
    "failed_login_count", "rolling_avg_failed_logins", "failed_login_spike",
    "vpn_used", "mfa_used", "file_download_size_mb", "session_duration_min",
    "isolation_forest_score", "lstm_reconstruction_score", "fused_anomaly_score",
]

# ----------------------------------------------------------------------------
# Rule-based factors: explicit, auditable, independent of any ML model.
# Each rule contributes a fixed number of points (capped at 100 total),
# mirroring how a real SOC rule engine (e.g. Splunk correlation rules)
# would score known-bad patterns regardless of what the ML models say.
# ----------------------------------------------------------------------------
RULES = [
    # (condition_fn, points, human-readable label)
    (lambda r: r["impossible_travel_flag"] == 1, 25, "Impossible Travel"),
    (lambda r: r["is_new_device"] == 1, 15, "New Device"),
    (lambda r: r["failed_login_count"] >= 5, 15, "High Failed Login Count"),
    (lambda r: r["is_sensitive_resource"] == 1, 15, "Sensitive Resource Access"),
    (lambda r: r["is_odd_hour_login"] == 1, 10, "Midnight Login"),
    (lambda r: r["is_cross_department_access"] == 1, 10, "Cross-Department Access"),
    (lambda r: r["mfa_used"] == 0, 10, "MFA Not Used"),
]

SHAP_READABLE_NAMES = {
    "distance_from_prev_km": "Large Geographic Jump",
    "implied_speed_kmh": "Impossible Travel Speed",
    "impossible_travel_flag": "Impossible Travel",
    "is_new_device": "New Device",
    "is_new_os": "Unrecognized OS",
    "is_new_browser": "Unrecognized Browser",
    "login_hour_deviation": "Login Time Deviation",
    "is_odd_hour_login": "Midnight Login",
    "cold_start_flag": "Limited User History",
    "is_cross_department_access": "Cross-Department Access",
    "is_sensitive_resource": "Sensitive Resource Access",
    "failed_login_count": "High Failed Login Count",
    "rolling_avg_failed_logins": "Failed Logins Above Personal Norm",
    "failed_login_spike": "Sudden Failed Login Spike",
    "vpn_used": "VPN Usage Pattern",
    "mfa_used": "MFA Status Anomaly",
    "file_download_size_mb": "Unusual Download Volume",
    "session_duration_min": "Abnormal Session Duration",
    "isolation_forest_score": "Point-Anomaly Signal",
    "lstm_reconstruction_score": "Sequence-Anomaly Signal",
    "fused_anomaly_score": "High Overall Anomaly Score",
}

RISK_LEVEL_BINS = [
    (80, 100, "Critical"),
    (60, 79, "High"),
    (35, 59, "Medium"),
    (0, 34, "Low"),
]


def compute_rule_score(row: pd.Series):
    """Returns (rule_score 0-100, list of triggered rule labels)."""
    score = 0
    triggered = []
    for condition, points, label in RULES:
        if condition(row):
            score += points
            triggered.append(label)
    return min(score, 100), triggered


def compute_risk_score(fused_anomaly_score: float, prediction_confidence: float,
                        predicted_attack_type: str, rule_score: int) -> float:
    """
    Weighted fusion of the three signals. Weights reflect how much we trust
    each signal:
      - 45% fused anomaly score: the most general, always-available signal
      - 35%/10% classifier confidence: weighted heavily when the model
        actually predicts an attack type, but only lightly trusted when it
        predicts "None" -- we don't want classifier confidence in "benign"
        to suppress a risk score that other signals say is high (defense
        in depth against Stage 2 false negatives).
      - 20% rule-based score: explicit, auditable factors
    """
    if predicted_attack_type != "None":
        confidence_component = prediction_confidence * 100 * 0.35
        weight_rule = 0.20
    else:
        confidence_component = (1 - prediction_confidence) * 100 * 0.10
        weight_rule = 0.20

    score = (fused_anomaly_score * 100 * 0.45) + confidence_component + (rule_score * weight_rule)
    return round(min(max(score, 0), 100), 1)


def risk_level_for(score: float) -> str:
    for low, high, label in RISK_LEVEL_BINS:
        if low <= score <= high:
            return label
    return "Low"


def get_shap_reasons(shap_row: np.ndarray, feature_names: list, top_k: int = 5) -> list:
    contributions = list(zip(feature_names, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feat, val in contributions[:top_k]:
        if abs(val) < 1e-4:
            continue
        reasons.append(SHAP_READABLE_NAMES.get(feat, feat))
    return reasons


def merge_reasons(rule_reasons: list, shap_reasons: list, max_reasons: int = 5) -> list:
    """
    Merges rule-based and SHAP-based reasons, de-duplicated, rule-based
    first (since they're the most auditable/certain), then SHAP fills any
    remaining slots with model-driven insights not already covered.
    """
    merged = list(rule_reasons)
    for r in shap_reasons:
        if r not in merged:
            merged.append(r)
    return merged[:max_reasons]


if __name__ == "__main__":
    print("[1/5] Loading Stage 2 predictions...")
    df = pd.read_csv(STAGE2_PATH, parse_dates=["timestamp"], keep_default_na=False, na_values=[""])
    for col in CLASSIFIER_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print("[2/5] Loading Stage 2 classifier for SHAP explainability...")
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_attack_classifier.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    X = df[CLASSIFIER_FEATURES].values

    print("[3/5] Computing SHAP values for all events (TreeExplainer, exact + fast)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    predicted_class_indices = label_encoder.transform(df["predicted_attack_type"])

    print("[4/5] Computing rule-based scores, SHAP reasons, and fused risk scores...")
    rule_scores, rule_reasons_list, shap_reasons_list, risk_scores = [], [], [], []

    for i, row in df.iterrows():
        rule_score, rule_reasons = compute_rule_score(row)
        pred_idx = predicted_class_indices[i]
        row_shap = shap_values[i, :, pred_idx] if shap_values.ndim == 3 else shap_values[i]
        shap_reasons = get_shap_reasons(row_shap, CLASSIFIER_FEATURES)

        risk_score = compute_risk_score(
            fused_anomaly_score=row["fused_anomaly_score"],
            prediction_confidence=row["prediction_confidence"],
            predicted_attack_type=row["predicted_attack_type"],
            rule_score=rule_score,
        )

        rule_scores.append(rule_score)
        rule_reasons_list.append(rule_reasons)
        shap_reasons_list.append(shap_reasons)
        risk_scores.append(risk_score)

    df["rule_based_score"] = rule_scores
    df["risk_score"] = risk_scores
    df["risk_level"] = [risk_level_for(s) for s in risk_scores]
    df["reasons"] = [
        merge_reasons(rr, sr) for rr, sr in zip(rule_reasons_list, shap_reasons_list)
    ]
    df["reasons_str"] = df["reasons"].apply(lambda lst: ", ".join(lst) if lst else "No significant risk factors")

    print("[5/5] Saving final risk-scored dataset...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Risk Score Distribution ===")
    print(df["risk_level"].value_counts())

    print("\n=== Sample High-Risk Alerts (Critical tier) ===")
    top_alerts = df[df["risk_level"] == "Critical"].sort_values("risk_score", ascending=False).head(3)
    for _, r in top_alerts.iterrows():
        print(f"\nUser: {r['user_id']}  |  Predicted Attack: {r['predicted_attack_type']}  |  Actual Label: {r['attack_type']}")
        print(f"Risk Score: {r['risk_score']:.0f}")
        print(f"Reasons:")
        for reason in r["reasons"]:
            print(f"  - {reason}")

    print(f"\nSaved risk-scored dataset to: {OUTPUT_PATH}")

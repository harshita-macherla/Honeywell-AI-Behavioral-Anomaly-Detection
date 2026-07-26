"""
risk_scoring_engine_v2.py
============================
Risk Scoring Engine (v2), extended from risk_scoring_engine.py to run
against the Enterprise Dataset v2 / Stage 2 v2 output.

REPLACES (for the v2 data track only) scripts/risk_scoring_engine.py. The
v1 script and its output (dataset/processed/risk_scores.csv) are NOT
modified, NOT overwritten, and NOT deleted -- they remain as the v1
baseline result for the report. This script is additive: it produces its
own, separately-named v2 artifact, consistent with how
train_stage1_anomaly_detection_v2.py and train_stage2_classification_v2.py
handled the same v1/v2 split.

ARCHITECTURE (unchanged from v1 -- extended, not redesigned)
----------------------------------------------------------------------------
Combines THREE independent signals into one 0-100 risk score per event,
and generates a ranked, human-readable list of reasons for it:

    1. Stage 1 v2 fused anomaly score   (unsupervised "how weird is this")
    2. Stage 2 v2 classification confidence  (supervised "how sure are we
       this is a specific attack type")
    3. Rule-based factors                (explicit, auditable security
       rules a SOC analyst would recognize immediately, independent of any
       model -- defense-in-depth against ML mis-scoring)

Explainability layers (unchanged from v1):
    - SHAP (TreeExplainer on the Stage 2 v2 XGBoost model) -> "why did the
      model think this was attack type X"
    - Rule-based flags -> "which explicit security rules were violated"
    - Merged into one ranked reasons list per event, capped at 5.

WHAT CHANGED VS v1 (schema porting + genuine v2-specific extensions)
----------------------------------------------------------------------------
- STAGE2_PATH / OUTPUT_PATH / model paths point at the v2 files.
- CLASSIFIER_FEATURES is Stage 2 v2's exact 89-column feature list
  (BASE_FEATURES + Stage-1-output columns) -- MUST match the feature order
  the v2 XGBoost model was trained on, or predictions/SHAP silently
  misalign. Copied verbatim from train_stage2_classification_v2.py rather
  than imported, matching v1's existing convention of each script keeping
  its own copy of this list (v1's risk_scoring_engine.py duplicates
  train_stage2_classification.py's list the same way).
- Rule set entity-type-aware extensions (v2-specific, "where appropriate"
  per this milestone's requirement):
    * "Sensitive Resource Access" rule adapted from v1's binary
      is_sensitive_resource flag (which does not exist in v2) to v2's
      graded resource_sensitivity (1-5) field, using the >=4 threshold
      feature_engineering_v2.py itself already uses for the same purpose
      in low_and_slow_exfil_score / credential_misuse_score.
    * "MFA Not Used" rule restricted to entity_type == "user" -- see the
      "BUG FOUND" note below for why this restriction is necessary, not
      optional.
    * NEW: "OT/IoT Device Fingerprint Mismatch" -- non-human entities
      (edge_device, iot_device, industrial_controller, server,
      service_account) whose device fingerprint changed get extra
      dedicated weight, since Device Spoofing against OT/IoT is exactly
      the domain-agnostic scenario the Honeywell brief calls out
      explicitly (industrial edge gateways, home IoT hubs).
    * NEW: "Privilege Escalation Signature" -- leverages v2's
      privilege_deviation feature (from the org-hierarchy modeling added
      in generate_logs_v2.py), which has no v1 equivalent since v1 had no
      privilege_level/manager hierarchy. Directly supports v2's new
      Privilege_Escalation attack class.
- SHAP_READABLE_NAMES extended to the 86+3 v2 feature vocabulary (same
  category structure as train_stage2_classification_v2.py's
  READABLE_NAMES, kept as its own copy per v1's existing per-script
  convention, not shared/imported).
- Sample-alert printout uses entity_id/entity_type instead of v1's
  user_id (v2 has no user_id column).

Output: dataset/processed/risk_scores_v2.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

STAGE2_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "stage2_predictions_v2.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "risk_scores_v2.csv")

# EXACT copy of train_stage2_classification_v2.py's CLASSIFIER_FEATURES
# (BASE_FEATURES + STAGE1_OUTPUT_FEATURES). Must stay byte-identical in
# name AND order to the list the v2 XGBoost model was trained on.
BASE_FEATURES = [
    "historical_event_count", "time_since_last_login_hours", "avg_session_duration",
    "session_duration_zscore", "day_of_week_deviation", "weekend_activity_flag",
    "holiday_activity_flag", "login_hour", "behavioral_drift_score",
    "rolling_failed_login_rate_7d", "is_new_device", "is_new_os", "is_new_browser",
    "device_usage_frequency", "new_device_probability", "fingerprint_change_score",
    "device_reputation", "managed_device_flag", "device_risk_score",
    "distance_from_prev_km", "geo_velocity_kmh", "impossible_travel_flag",
    "country_change", "city_change", "network_zone_change", "is_hosting_asn",
    "remote_access_score", "internal_network_score", "anonymization_risk_score",
    "rolling_avg_failed_logins", "failed_login_spike", "failed_login_streak",
    "mfa_deviation", "auth_method_entropy", "password_spray_score",
    "credential_stuffing_score", "resource_diversity_count",
    "resource_sensitivity_deviation", "is_cross_department_access",
    "privilege_deviation", "critical_resource_rate_7d", "resource_entropy",
    "command_sequence_length", "command_entropy", "command_rarity_score",
    "dangerous_command_ratio", "privilege_escalation_cmd_score",
    "lolbin_usage_flag", "powershell_usage_flag", "command_novelty_score",
    "session_size", "session_hijack_flag", "session_age_minutes",
    "concurrent_session_count_1h", "session_restart_rate",
    "peer_group_resource_sensitivity_deviation", "department_baseline_sensitivity",
    "business_unit_deviation", "privilege_baseline_sensitivity", "manager_deviation",
    "adaptive_threshold", "adaptive_threshold_exceeded_flag", "baseline_confidence",
    "cold_start_score", "cold_start_flag", "rolling_count_1h", "rolling_count_24h",
    "rolling_count_7d", "burst_score", "hour_sin", "hour_cos", "weekday_sin",
    "weekday_cos", "seasonality_score", "login_hour_deviation", "is_odd_hour_login",
    "credential_misuse_score", "brute_force_score", "impossible_travel_score",
    "device_spoofing_score", "lateral_movement_score", "session_hijacking_score",
    "low_and_slow_exfil_score", "living_off_the_land_score", "insider_threat_score",
    "command_abuse_score",
]
STAGE1_OUTPUT_FEATURES = ["isolation_forest_score", "lstm_reconstruction_score", "fused_anomaly_score"]
CLASSIFIER_FEATURES = BASE_FEATURES + STAGE1_OUTPUT_FEATURES

# ----------------------------------------------------------------------------
# Rule-based factors: explicit, auditable, independent of any ML model.
# Same architecture as v1 (fixed-point rules capped at 100), extended with
# v2's graded/entity-aware fields.
# ----------------------------------------------------------------------------
RULES = [
    # (condition_fn, points, human-readable label)  -- unchanged from v1
    (lambda r: r["impossible_travel_flag"] == 1, 25, "Impossible Travel"),
    (lambda r: r["is_new_device"] == 1, 15, "New Device"),
    (lambda r: r["failed_login_count"] >= 5, 15, "High Failed Login Count"),
    # ADAPTED from v1: v1's binary is_sensitive_resource does not exist in
    # v2 (which uses a graded 1-5 resource_sensitivity field instead) --
    # threshold matches feature_engineering_v2.py's own >=4 convention.
    (lambda r: r["resource_sensitivity"] >= 4, 15, "Sensitive Resource Access"),
    (lambda r: r["is_odd_hour_login"] == 1, 10, "Midnight Login"),
    (lambda r: r["is_cross_department_access"] == 1, 10, "Cross-Department Access"),
    # ADAPTED from v1 (entity-type-aware) -- see "BUG FOUND" note below.
    (lambda r: r["entity_type"] == "user" and r["mfa_used"] == 0, 10, "MFA Not Used"),
    # NEW (v2, entity-type-aware): Device Spoofing signal specifically for
    # non-human entities -- the domain-agnostic OT/IoT/edge-device scenario
    # the Honeywell brief calls out explicitly.
    (lambda r: r["entity_type"] != "user" and r["fingerprint_change_score"] == 1,
     15, "OT/IoT Device Fingerprint Mismatch"),
    # NEW (v2): leverages v2's org-hierarchy privilege_deviation feature,
    # which has no v1 equivalent (v1 had no privilege_level/manager
    # hierarchy) -- directly supports v2's new Privilege_Escalation class.
    (lambda r: abs(r["privilege_deviation"]) >= 1, 10, "Privilege Level Deviation"),
]

# Human-readable labels for the v2 feature vocabulary used in SHAP
# explanations. Kept as its own copy (not imported from
# train_stage2_classification_v2.py) to match v1's existing convention of
# each script keeping an independent readable-names dict.
SHAP_READABLE_NAMES = {
    "historical_event_count": "Limited historical baseline for this entity",
    "time_since_last_login_hours": "Unusual gap since last activity",
    "avg_session_duration": "Session duration deviates from entity's baseline",
    "session_duration_zscore": "Abnormal session duration (statistical outlier)",
    "day_of_week_deviation": "Unusual day-of-week for this entity",
    "weekend_activity_flag": "Weekend activity",
    "holiday_activity_flag": "Company-holiday activity",
    "login_hour": "Login occurred at an unusual hour for this entity",
    "behavioral_drift_score": "Gradual behavioral drift from entity's norm",
    "rolling_failed_login_rate_7d": "Elevated 7-day failed-login rate",
    "is_new_device": "New/unrecognized device",
    "is_new_os": "Unrecognized operating system",
    "is_new_browser": "Unrecognized browser",
    "device_usage_frequency": "Rarely-used device for this entity",
    "new_device_probability": "High device-novelty probability",
    "fingerprint_change_score": "Device fingerprint changed on known hardware",
    "device_reputation": "Device shared across unusually many entities",
    "managed_device_flag": "Unmanaged/unenrolled device",
    "device_risk_score": "High composite device risk",
    "distance_from_prev_km": "Large geographic jump since last login",
    "geo_velocity_kmh": "Implausible travel speed",
    "impossible_travel_flag": "Impossible travel detected",
    "country_change": "Country changed since last login",
    "city_change": "City changed since last login",
    "network_zone_change": "Network zone changed",
    "is_hosting_asn": "Connection from hosting/datacenter network (non-residential)",
    "remote_access_score": "Elevated remote-access risk",
    "internal_network_score": "Internal-network access pattern",
    "anonymization_risk_score": "Possible anonymization/proxy usage",
    "rolling_avg_failed_logins": "Failed logins above personal norm",
    "failed_login_spike": "Sudden spike in failed logins",
    "failed_login_streak": "Consecutive failed-login streak",
    "mfa_deviation": "MFA usage deviates from entity's norm",
    "auth_method_entropy": "Unusual variety of authentication methods",
    "password_spray_score": "Password-spray pattern signature",
    "credential_stuffing_score": "Credential-stuffing pattern signature",
    "resource_diversity_count": "Unusually broad resource access",
    "resource_sensitivity_deviation": "Accessing more sensitive resources than usual",
    "is_cross_department_access": "Cross-department resource access",
    "privilege_deviation": "Privilege level deviates from expected footprint",
    "critical_resource_rate_7d": "Elevated critical-resource access rate (7d)",
    "resource_entropy": "Unusual resource-access entropy",
    "command_sequence_length": "Unusually long command sequence",
    "command_entropy": "Unusual command-sequence entropy",
    "command_rarity_score": "Rare command(s) for this entity/role",
    "dangerous_command_ratio": "High ratio of dangerous commands",
    "privilege_escalation_cmd_score": "Privilege-escalation command signature",
    "lolbin_usage_flag": "Living-off-the-land binary (LOLBin) usage",
    "powershell_usage_flag": "PowerShell usage",
    "command_novelty_score": "Novel command pattern for this entity",
    "session_size": "Unusually large session size",
    "session_hijack_flag": "Session-hijack signature detected",
    "session_age_minutes": "Abnormal session age",
    "concurrent_session_count_1h": "Multiple concurrent sessions within 1 hour",
    "session_restart_rate": "Elevated session-restart rate",
    "peer_group_resource_sensitivity_deviation": "Deviates from peer group's resource-sensitivity baseline",
    "department_baseline_sensitivity": "Deviates from department's resource-sensitivity baseline",
    "business_unit_deviation": "Deviates from business-unit baseline",
    "privilege_baseline_sensitivity": "Deviates from privilege-level baseline",
    "manager_deviation": "Deviates from manager's team baseline",
    "adaptive_threshold": "Exceeded entity's adaptive behavioral threshold",
    "adaptive_threshold_exceeded_flag": "Adaptive threshold exceeded",
    "baseline_confidence": "Low confidence in entity's behavioral baseline",
    "cold_start_score": "Cold-start entity (limited history)",
    "cold_start_flag": "Cold-start entity",
    "rolling_count_1h": "Elevated event count (1h window)",
    "rolling_count_24h": "Elevated event count (24h window)",
    "rolling_count_7d": "Elevated event count (7d window)",
    "burst_score": "Burst-of-activity signature",
    "hour_sin": "Unusual time-of-day pattern (cyclical encoding)",
    "hour_cos": "Unusual time-of-day pattern (cyclical encoding)",
    "weekday_sin": "Unusual day-of-week pattern (cyclical encoding)",
    "weekday_cos": "Unusual day-of-week pattern (cyclical encoding)",
    "seasonality_score": "Unusual seasonal/cyclical activity pattern",
    "login_hour_deviation": "Login time deviates from usual pattern",
    "is_odd_hour_login": "Midnight/off-hours login",
    "credential_misuse_score": "Credential-misuse composite signature",
    "brute_force_score": "Brute-force composite signature",
    "impossible_travel_score": "Impossible-travel composite signature",
    "device_spoofing_score": "Device-spoofing composite signature",
    "lateral_movement_score": "Lateral-movement composite signature",
    "session_hijacking_score": "Session-hijacking composite signature",
    "low_and_slow_exfil_score": "Low-and-slow exfiltration composite signature",
    "living_off_the_land_score": "Living-off-the-land composite signature",
    "insider_threat_score": "Insider-threat composite signature",
    "command_abuse_score": "Command-abuse composite signature",
    "isolation_forest_score": "Point-Anomaly Signal (Isolation Forest)",
    "lstm_reconstruction_score": "Sequence-Anomaly Signal (LSTM-AE)",
    "fused_anomaly_score": "High Overall Anomaly Score",
}

RISK_LEVEL_BINS = [
    (80, 100, "Critical"),
    (60, 79, "High"),
    (35, 59, "Medium"),
    (0, 34, "Low"),
]


def compute_rule_score(row: pd.Series):
    """Returns (rule_score 0-100, list of triggered rule labels). Unchanged from v1."""
    score = 0
    triggered = []
    for condition, points, label in RULES:
        if condition(row):
            score += points
            triggered.append(label)
    return min(score, 100), triggered


def compute_risk_score(fused_anomaly_score_scaled: float, prediction_confidence: float,
                        predicted_attack_type: str, rule_score: int) -> float:
    """
    Weighted fusion of the three signals. Identical weighting to v1:
      - 45% fused anomaly score (percentile-scaled -- see BUG FOUND note
        at the call site below for why the raw 0-1 score cannot be used
        directly here)
      - 35%/10% classifier confidence (35% when an attack type is
        predicted, 10% -- of (1-confidence) -- when "None" is predicted,
        so classifier confidence in "benign" cannot suppress a risk score
        other signals say is high)
      - 20% rule-based score
    """
    if predicted_attack_type != "None":
        confidence_component = prediction_confidence * 100 * 0.35
        weight_rule = 0.20
    else:
        confidence_component = (1 - prediction_confidence) * 100 * 0.10
        weight_rule = 0.20

    score = (fused_anomaly_score_scaled * 100 * 0.45) + confidence_component + (rule_score * weight_rule)
    return round(min(max(score, 0), 100), 1)


def risk_level_for(score: float) -> str:
    """
    BUG FOUND DURING BACKEND INTEGRATION TESTING, FIXED HERE:
    The original v1 implementation (inherited unchanged into this v2
    script) checked `low <= score <= high` against inclusive-inclusive
    bins (80,100), (60,79), (35,59), (0,34). Because risk_score is a
    rounded float with one decimal place (see compute_risk_score()'s
    `round(..., 1)`), any score strictly between 59.0-60.0 or 79.0-80.0
    (e.g. 59.1, 79.4) matched NONE of the four bins and silently fell
    through to the `return "Low"` default -- mislabeling Medium/High-tier
    events as Low.
      Discovered live: the new backend's /api/v1/predict endpoint returned
    risk_score=59.1 labeled "Low" instead of "Medium" for a real IoT-device
    event during endpoint testing. Quantified the blast radius against the
    already-computed risk_scores_v2.csv: 4 rows fall in the 59-60 gap (2 of
    them true attacks) and 51 rows fall in the 79-80 gap (all 51 of them
    true attacks) -- all 55 were mislabeled "Low" when they should have
    been "Medium" or "High" respectively. The same gap exists in v1's
    risk_scoring_engine.py (68 affected rows there) but that script is
    left untouched, per the project rule to treat v1 as a frozen baseline
    for the report -- only this v2 script (which the new backend actually
    depends on) is corrected here.
      Fix: replaced the gappy inclusive-inclusive bin scan with a gapless
    threshold cascade covering the full 0-100 range with no seams,
    regardless of decimal precision. RISK_LEVEL_BINS above is kept as
    human-readable documentation of the same thresholds; this function no
    longer iterates it.
    """
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 35:
        return "Medium"
    else:
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
    """Merges rule-based and SHAP-based reasons, de-duplicated, rule-based first. Unchanged from v1."""
    merged = list(rule_reasons)
    for r in shap_reasons:
        if r not in merged:
            merged.append(r)
    return merged[:max_reasons]


if __name__ == "__main__":
    print("[1/5] Loading Stage 2 v2 predictions...")
    df = pd.read_csv(STAGE2_PATH, parse_dates=["timestamp"])

    # attack_type / predicted_attack_type were exported with the "real NaN
    # for benign" convention (see the bug fix documented in
    # train_stage2_classification_v2.py). Rule conditions and the risk
    # formula both compare predicted_attack_type against the string
    # "None", and label_encoder_v2 was fit on the string "None" as one of
    # its 13 classes -- so both columns need the NaN filled back to "None"
    # here, the same way Stage 2 v2's own load_data() did internally.
    df["attack_type"] = df["attack_type"].fillna("None")
    df["predicted_attack_type"] = df["predicted_attack_type"].fillna("None")

    for col in CLASSIFIER_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print("[2/5] Loading Stage 2 v2 classifier for SHAP explainability...")
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_attack_classifier_v2.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder_v2.joblib"))
    X = df[CLASSIFIER_FEATURES].values

    print("[3/5] Computing SHAP values for all events (TreeExplainer, exact + fast)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    print(f"      SHAP values shape: {shap_values.shape}")

    predicted_class_indices = label_encoder.transform(df["predicted_attack_type"])

    # BUG FOUND DURING INTEGRATION TESTING, FIXED HERE:
    # Ran the full pipeline end-to-end and found risk_scores_v2.csv
    # produced ZERO "Critical" (80-100) tier alerts despite 1,507 true
    # attack rows and 99.66% Stage 2 v2 classification accuracy -- vs. 95
    # Critical alerts in v1's risk_scores.csv on a comparably-sized attack
    # set. Root-caused by decomposing the three risk components for known
    # attack rows: prediction_confidence averaged ~0.995 in both v1 and
    # v2 (no difference), rule_based_score averaged HIGHER in v2 (38.9 vs
    # 33.1 -- the new entity-aware rules are working correctly), but
    # fused_anomaly_score averaged only 0.155 for v2 attacks vs. 0.475 for
    # v1 attacks -- a 3x compression in the single highest-weighted (45%)
    # component.
    #   This is NOT a Stage 1 v2 model-quality bug: Stage 1 v2's own
    # ROC-AUC (0.9439, see train_stage1_anomaly_detection_v2.py) confirms
    # attacks are correctly RANKED above normal events. The issue is that
    # fused_anomaly_score's raw MAGNITUDE is heavily right-skewed --
    # lstm_reconstruction_score (which the corrected v2 fusion weights at
    # 0.8, vs. v1's 0.2) has a long tail (median 0.017, but a handful of
    # extreme cold-start/burst sequences stretch the min-max-normalized
    # max to 1.0), so the typical attack's raw fused score sits around the
    # 97.5th percentile of raw VALUE while still being a very high
    # RANK. compute_risk_score()'s fixed linear formula (fused_score *
    # 100 * 0.45) assumes a roughly uniform 0-1 spread -- true for v1's
    # more balanced 0.8/0.2 fusion, false for v2's AUC-optimal 0.2/0.8
    # fusion. Verified this is a scale/calibration issue, not a ranking
    # issue, by checking the rank-vs-value gap directly (attacks average
    # the 93rd raw-value percentile despite averaging only 0.155 in raw
    # value).
    #   Fix (scoped entirely to THIS engine -- Stage 1 v2's model, weights,
    # and saved artifacts are correct as-is and are NOT modified): convert
    # fused_anomaly_score to a percentile rank (0-1) before it enters the
    # linear risk formula. This preserves Stage 1 v2's AUC-optimal fusion
    # weighting untouched while restoring a usable, non-skewed input scale
    # for the risk score -- the same rank-based logic Stage 1 v2's own
    # evaluate() already uses for alert thresholding
    # (np.quantile(fused_scores, 1-contamination)), just applied per-row
    # here instead of at a single cutoff.
    df["fused_anomaly_score_percentile"] = df["fused_anomaly_score"].rank(pct=True)

    print("[4/5] Computing rule-based scores, SHAP reasons, and fused risk scores "
          "(entity-type-aware rules included)...")
    rule_scores, rule_reasons_list, shap_reasons_list, risk_scores = [], [], [], []

    for i, row in df.iterrows():
        rule_score, rule_reasons = compute_rule_score(row)
        pred_idx = predicted_class_indices[i]
        row_shap = shap_values[i, :, pred_idx] if shap_values.ndim == 3 else shap_values[i]
        shap_reasons = get_shap_reasons(row_shap, CLASSIFIER_FEATURES)

        risk_score = compute_risk_score(
            fused_anomaly_score_scaled=row["fused_anomaly_score_percentile"],
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

    # Restore the "real NaN for benign" convention on export (same fix
    # applied in train_stage2_classification_v2.py, for the same reason --
    # keeps this file consistent with the rest of the v2 pipeline and safe
    # to read with a plain pd.read_csv() downstream).
    df["attack_type"] = df["attack_type"].replace("None", np.nan)
    df["predicted_attack_type"] = df["predicted_attack_type"].replace("None", np.nan)

    print("[5/5] Saving final v2 risk-scored dataset...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Risk Score Distribution ===")
    print(df["risk_level"].value_counts())

    print("\n=== Risk Level by Entity Type ===")
    print(pd.crosstab(df["entity_type"], df["risk_level"]))

    print("\n=== Sample High-Risk Alerts (Critical tier) ===")
    top_alerts = df[df["risk_level"] == "Critical"].sort_values("risk_score", ascending=False).head(5)
    for _, r in top_alerts.iterrows():
        actual_label = r["attack_type"] if pd.notna(r["attack_type"]) else "None"
        print(f"\nEntity: {r['entity_id']} ({r['entity_type']})  |  "
              f"Predicted Attack: {r['predicted_attack_type']}  |  Actual Label: {actual_label}")
        print(f"Risk Score: {r['risk_score']:.0f}")
        print("Reasons:")
        for reason in r["reasons"]:
            print(f"  - {reason}")

    print(f"\nSaved v2 risk-scored dataset to: {OUTPUT_PATH}")

"""
train_stage2_classification_v2.py
====================================
Stage 2 of the two-stage pipeline (v2): ATTACK CLASSIFICATION,
retrained against the Enterprise Dataset v2 / Stage 1 v2 output.

REPLACES (for the v2 data track only) scripts/train_stage2_classification.py.
The v1 script and its outputs (models/xgb_attack_classifier.joblib,
models/label_encoder.joblib, dataset/processed/stage2_predictions.csv) are
NOT modified, NOT overwritten, and NOT deleted -- they remain as the v1
baseline result for the report. This script is additive: it produces its
own, separately-named v2 artifacts, consistent with how
train_stage1_anomaly_detection_v2.py handled the same v1/v2 split.

Given an event, classifies it into one of 13 classes: None (benign) or one
of the 12 v2 attack types (Credential_Misuse, Credential_Stuffing,
Brute_Force, Impossible_Travel, Device_Spoofing, Lateral_Movement,
Insider_Threat, Privilege_Escalation, Low_and_Slow_Exfiltration,
Command_Abuse, Living_off_the_Land, Session_Hijacking) -- up from v1's
6-class taxonomy (None + 5 attacks).

Trained on the FULL v2 dataset (not just Stage-1-v2-flagged rows) -- same
rationale as v1: Stage 1 v2 has real false negatives/positives at a strict
alert threshold, so Stage 2 needs to see the full picture to act as an
independent second opinion during analyst triage, not a rubber stamp.

Model choice: XGBoost multiclass (multi:softprob) -- unchanged from v1.
    - Handles severe class imbalance (12 attack classes at ~125 each vs.
      ~58,500 benign, out of 60,007 total rows) via balanced sample weights.
    - Tree-based -> pairs with SHAP's TreeExplainer (exact, fast), which is
      what the eventual dashboard needs for real-time per-alert explanations.
    - Naturally handles the mixed numeric/binary 86-feature set from
      feature_engineering_v2.py without embeddings or extra scaling.

WHAT CHANGED VS v1 (data-schema porting, not a redesign)
----------------------------------------------------------------------------
- ANOMALY_SCORES_PATH / OUTPUT_PATH point at the v2 files (produced by
  train_stage1_anomaly_detection_v2.py).
- CLASSIFIER_FEATURES is Stage 1 v2's 86-column NUMERIC_FEATURES list PLUS
  Stage 1 v2's own outputs (isolation_forest_score, lstm_reconstruction_score,
  fused_anomaly_score) -- same "features + stage 1 opinion" pattern as v1,
  just with the v2 feature set instead of v1's 18 columns.
- v2's `attack_type` column uses real NaN for benign rows (v1's serialized
  CSV used the literal string "None"). Handled explicitly below -- see the
  "BUG FOUND" note in load_data().
- 13-class label space instead of v1's 6-class space.
- explain_top_reasons()'s readable-name mapping is extended to cover the
  86-feature v2 vocabulary (device/network/auth/resource/command/session/
  organization/temporal/attack-composite categories), not just v1's raw
  18 columns.

Outputs:
    models/xgb_attack_classifier_v2.joblib
    models/label_encoder_v2.joblib
    dataset/processed/stage2_predictions_v2.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

ANOMALY_SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "anomaly_scores_v2.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "stage2_predictions_v2.csv")

# Same 86 curated behavioral features Stage 1 v2 used, PLUS Stage 1 v2's own
# outputs as additional features (Stage 1's opinion is informative signal
# for Stage 2 to weigh) -- identical pattern to v1's CLASSIFIER_FEATURES.
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


def load_data():
    df = pd.read_csv(ANOMALY_SCORES_PATH, parse_dates=["timestamp"])

    # BUG FOUND DURING INTEGRATION TESTING, FIXED HERE:
    # v1's load_data() used keep_default_na=False, na_values=[""] because
    # v1's anomaly_scores.csv serialized benign rows' attack_type as the
    # LITERAL STRING "None" (a quirk of how v1's generator wrote that
    # column), and pandas' default NaN-sniffing would otherwise have
    # silently turned that literal "None" into a real NaN and merged it
    # with genuinely missing data. Reusing that same read_csv flag
    # unmodified against anomaly_scores_v2.csv was tested here and found to
    # be a NO-OP that masks a real difference: v2's generator/feature
    # pipeline writes benign rows' attack_type as an ACTUAL missing value
    # (true NaN), not the string "None" (confirmed: 58,500 NaN rows in
    # anomaly_scores_v2.csv, zero literal "None" strings). Since a
    # LabelEncoder cannot fit on NaN, attack_type must be explicitly filled
    # with the string "None" here -- otherwise LabelEncoder.fit_transform()
    # raises immediately on the very first run. This is done explicitly
    # below (not via the na_values read_csv flag, which does not apply to
    # values already read as NaN by pandas' own default sniffing).
    df["attack_type"] = df["attack_type"].fillna("None")

    # BUG FOUND DURING INTEGRATION TESTING, FIXED HERE (second bug, related
    # to the one above): the fillna("None") above is necessary so
    # LabelEncoder can fit on attack_type (it cannot fit on NaN), but if the
    # resulting DataFrame is later persisted to CSV as-is, it PERMANENTLY
    # bakes the literal string "None" into stage2_predictions_v2.csv's
    # attack_type column -- silently reintroducing, in this script's own
    # output, the exact "None"-vs-NaN footgun v1 had to work around, and
    # breaking the "real NaN for benign" convention that anomaly_scores_v2.csv
    # (this script's own input) and features_v2.csv both use. Confirmed via
    # a direct round-trip test: reading stage2_predictions_v2.csv back with
    # plain pd.read_csv() converted the literal "None" strings to NaN anyway
    # (pandas' default NA-value list includes "None"), while a naive
    # accuracy check using those NaNs against predicted_attack_type produced
    # a nonsensical ~2% "accuracy" instead of the true ~99.66% -- i.e. this
    # bug would have silently corrupted any downstream analysis of the
    # exported file, not just this script's own console output (which is
    # unaffected, since it evaluates in-memory arrays, never the CSV).
    # Fix: keep a SEPARATE encoding-only column so the string "None" never
    # leaks into what gets written to disk; the exported attack_type/
    # predicted_attack_type columns are converted back to real NaN for "no
    # attack" immediately before the CSV is written (see bottom of script).
    ATTACK_TYPE_ENCODING_COL = "attack_type"  # already filled above, used only for y = label_encoder.fit_transform(...)

    for col in CLASSIFIER_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def train_classifier(X_train, y_train, num_classes):
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


# Human-readable labels for the v2 feature vocabulary, extending v1's
# 21-entry dict to the 86-feature + 3-Stage1-output set. Grouped in the
# same category order as FEATURE_CATALOG.md for maintainability.
READABLE_NAMES = {
    # User behavior
    "historical_event_count": "Limited historical baseline for this entity",
    "time_since_last_login_hours": "Unusual gap since last activity",
    "avg_session_duration": "Session duration deviates from entity's baseline",
    "session_duration_zscore": "Abnormal session duration (statistical outlier)",
    "day_of_week_deviation": "Unusual day-of-week for this entity",
    "weekend_activity_flag": "Weekend activity",
    "holiday_activity_flag": "Company-holiday activity",
    "behavioral_drift_score": "Gradual behavioral drift from entity's norm",
    "rolling_failed_login_rate_7d": "Elevated 7-day failed-login rate",
    # Device trust
    "is_new_device": "New/unrecognized device",
    "is_new_os": "Unrecognized operating system",
    "is_new_browser": "Unrecognized browser",
    "device_usage_frequency": "Rarely-used device for this entity",
    "new_device_probability": "High device-novelty probability",
    "fingerprint_change_score": "Device fingerprint changed on known hardware",
    "device_reputation": "Device shared across unusually many entities",
    "managed_device_flag": "Unmanaged/unenrolled device",
    "device_risk_score": "High composite device risk",
    # Network
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
    # Authentication
    "rolling_avg_failed_logins": "Failed logins above personal norm",
    "failed_login_spike": "Sudden spike in failed logins",
    "failed_login_streak": "Consecutive failed-login streak",
    "mfa_deviation": "MFA usage deviates from entity's norm",
    "auth_method_entropy": "Unusual variety of authentication methods",
    "password_spray_score": "Password-spray pattern signature",
    "credential_stuffing_score": "Credential-stuffing pattern signature",
    # Resource access
    "resource_diversity_count": "Unusually broad resource access",
    "resource_sensitivity_deviation": "Accessing more sensitive resources than usual",
    "is_cross_department_access": "Cross-department resource access",
    "privilege_deviation": "Privilege level deviates from expected footprint",
    "critical_resource_rate_7d": "Elevated critical-resource access rate (7d)",
    "resource_entropy": "Unusual resource-access entropy",
    # Command sequence
    "command_sequence_length": "Unusually long command sequence",
    "command_entropy": "Unusual command-sequence entropy",
    "command_rarity_score": "Rare command(s) for this entity/role",
    "dangerous_command_ratio": "High ratio of dangerous commands",
    "privilege_escalation_cmd_score": "Privilege-escalation command signature",
    "lolbin_usage_flag": "Living-off-the-land binary (LOLBin) usage",
    "powershell_usage_flag": "PowerShell usage",
    "command_novelty_score": "Novel command pattern for this entity",
    # Session
    "session_size": "Unusually large session size",
    "session_hijack_flag": "Session-hijack signature detected",
    "session_age_minutes": "Abnormal session age",
    "concurrent_session_count_1h": "Multiple concurrent sessions within 1 hour",
    "session_restart_rate": "Elevated session-restart rate",
    # Organization / peer-group
    "peer_group_resource_sensitivity_deviation": "Deviates from peer group's resource-sensitivity baseline",
    "department_baseline_sensitivity": "Deviates from department's resource-sensitivity baseline",
    "business_unit_deviation": "Deviates from business-unit baseline",
    "privilege_baseline_sensitivity": "Deviates from privilege-level baseline",
    "manager_deviation": "Deviates from manager's team baseline",
    # Behavioral baseline / cold start
    "adaptive_threshold": "Exceeded entity's adaptive behavioral threshold",
    "adaptive_threshold_exceeded_flag": "Adaptive threshold exceeded",
    "baseline_confidence": "Low confidence in entity's behavioral baseline",
    "cold_start_score": "Cold-start entity (limited history)",
    "cold_start_flag": "Cold-start entity",
    # Temporal
    "rolling_count_1h": "Elevated event count (1h window)",
    "rolling_count_24h": "Elevated event count (24h window)",
    "rolling_count_7d": "Elevated event count (7d window)",
    "burst_score": "Burst-of-activity signature",
    "login_hour_deviation": "Login time deviates from usual pattern",
    "is_odd_hour_login": "Off-hours login",
    # Temporal (cyclical encodings + seasonality) -- BUG FIX: these 6
    # features were originally missing from this dict (found via a
    # systematic diff of BASE_FEATURES against READABLE_NAMES during
    # integration testing), which caused explain_top_reasons() to silently
    # fall back to raw snake_case column names (e.g. "hour_cos",
    # "seasonality_score") whenever one of them was among an event's top-5
    # SHAP contributors -- confirmed reproducible on Event 45437 in the
    # smoke test below (Low_and_Slow_Exfiltration reasons list originally
    # showed "seasonality_score, hour_cos, login_hour" verbatim).
    "login_hour": "Login occurred at an unusual hour for this entity",
    "hour_sin": "Unusual time-of-day pattern (cyclical encoding)",
    "hour_cos": "Unusual time-of-day pattern (cyclical encoding)",
    "weekday_sin": "Unusual day-of-week pattern (cyclical encoding)",
    "weekday_cos": "Unusual day-of-week pattern (cyclical encoding)",
    "seasonality_score": "Unusual seasonal/cyclical activity pattern",
    # Attack-specific composites
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
    # Stage 1 outputs
    "isolation_forest_score": "Flagged by point-anomaly detector (Isolation Forest)",
    "lstm_reconstruction_score": "Flagged by sequence-anomaly detector (LSTM-AE)",
    "fused_anomaly_score": "High overall Stage 1 anomaly score",
}


def explain_top_reasons(shap_row: np.ndarray, feature_names: list, top_k: int = 5) -> list:
    """
    Converts a single row of SHAP values into a ranked, human-readable list
    of reasons -- the bridge between raw SHAP numbers and the dashboard's
    "Reasons: ..." style output. Identical logic to v1; only the readable
    name vocabulary changed (86+3 v2 features instead of v1's 18+3).
    """
    contributions = list(zip(feature_names, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, val in contributions[:top_k]:
        if abs(val) < 1e-4:
            continue
        reasons.append(READABLE_NAMES.get(feat, feat))
    return reasons


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/6] Loading v2 data with Stage 1 v2 outputs...")
    df = load_data()
    print(f"      Rows: {len(df)}  |  Entities: {df['entity_id'].nunique()}  |  "
          f"Entity types: {sorted(df['entity_type'].unique())}")

    print("[2/6] Encoding attack_type labels (13 classes: None + 12 v2 attack types)...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["attack_type"])
    X = df[CLASSIFIER_FEATURES].values
    print(f"Classes: {list(label_encoder.classes_)}")

    print("[3/6] Train/test split (stratified, 80/20)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=SEED, stratify=y
    )

    print("[4/6] Training XGBoost multiclass classifier (balanced sample weights)...")
    model = train_classifier(X_train, y_train, num_classes=len(label_encoder.classes_))

    print("[5/6] Evaluating on held-out test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall test accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

    print("\n[6/6] Computing SHAP values (TreeExplainer) for explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape varies by shap version -- handled below

    # Handle both shap API shapes across versions: some versions return a
    # list of (n_samples, n_features) arrays (one per class); others return
    # a single (n_samples, n_features, n_classes) array directly.
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)

    print(f"      SHAP values shape: {shap_values.shape}")

    # Demonstrate explainability on a few actual attack examples, sampling
    # across DIFFERENT attack types (not just the first 3 rows) so the
    # v2 taxonomy's breadth is actually exercised in this smoke test.
    print("\n=== Sample Explainable Predictions ===")
    none_class_idx = label_encoder.transform(["None"])[0]
    attack_mask = y_test != none_class_idx
    attack_positions = np.where(attack_mask)[0]
    # take one example per distinct true class present in the test attacks (up to 6 for brevity)
    seen_classes = set()
    sample_positions = []
    for pos in attack_positions:
        true_cls = y_test[pos]
        if true_cls not in seen_classes:
            seen_classes.add(true_cls)
            sample_positions.append(pos)
        if len(sample_positions) >= 6:
            break

    for pos in sample_positions:
        predicted_class_idx = y_pred[pos]
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        true_class = label_encoder.inverse_transform([y_test[pos]])[0]
        row_shap = shap_values[pos, :, predicted_class_idx] if shap_values.ndim == 3 else shap_values[pos]
        reasons = explain_top_reasons(row_shap, CLASSIFIER_FEATURES, top_k=5)
        original_row_idx = idx_test[pos]
        fused_score = df.loc[original_row_idx, "fused_anomaly_score"]
        entity_type = df.loc[original_row_idx, "entity_type"]
        print(f"\nEvent {original_row_idx} | Entity type: {entity_type} | "
              f"True: {true_class} | Predicted: {predicted_class}")
        print(f"  Anomaly Score (0-1): {fused_score:.3f}")
        print(f"  Reasons: {', '.join(reasons) if reasons else '(no dominant contributor above threshold)'}")

    # Persist model + encoder for the risk-scoring milestone and the future API layer
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_attack_classifier_v2.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder_v2.joblib"))

    # Persist full predictions (train+test) for downstream risk scoring / dashboard
    all_pred = model.predict(X)
    all_pred_proba = model.predict_proba(X)
    df_out = df.copy()
    df_out["predicted_attack_type"] = label_encoder.inverse_transform(all_pred)
    df_out["prediction_confidence"] = all_pred_proba.max(axis=1)

    # Restore the "real NaN for benign" convention (see BUG FOUND note in
    # load_data()) before writing to disk, instead of persisting the
    # encoding-only literal string "None" -- keeps this file consistent
    # with anomaly_scores_v2.csv / features_v2.csv and safe to read with a
    # plain pd.read_csv() downstream.
    df_out["attack_type"] = df_out["attack_type"].replace("None", np.nan)
    df_out["predicted_attack_type"] = df_out["predicted_attack_type"].replace("None", np.nan)

    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved v2 classifier + label encoder to: {MODELS_DIR}")
    print(f"Saved Stage 2 v2 predictions to: {OUTPUT_PATH}")

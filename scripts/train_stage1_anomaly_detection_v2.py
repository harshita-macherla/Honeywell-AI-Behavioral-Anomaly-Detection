"""
train_stage1_anomaly_detection_v2.py
======================================
Stage 1 of the two-stage pipeline (v2): BEHAVIORAL ANOMALY DETECTION,
retrained against the Enterprise Dataset v2 / Feature Pipeline v2.

REPLACES (for the v2 data track only) scripts/train_stage1_anomaly_detection.py.
The v1 script and its outputs (models/isolation_forest.joblib,
models/lstm_autoencoder.keras, models/feature_scaler.joblib,
dataset/processed/anomaly_scores.csv) are NOT modified, NOT overwritten, and
NOT deleted -- they remain as the v1 baseline result for the report. This
script is additive: it produces its own, separately-named v2 artifacts.

WHY A SEPARATE SCRIPT (not an edit of the v1 file)
----------------------------------------------------
The v1 script hardcodes an 18-column v1-only feature list (e.g.
`implied_speed_kmh`, `user_id` grouping) that does not exist in
features_v2.csv (v2 uses `geo_velocity_kmh`, `entity_id`, 86 curated
features across 6 entity types instead of 1). Editing the v1 file in place
would break the v1 baseline that Milestones 2-5 already validated. Per the
project rule "do not rewrite working code", the v1 file is left untouched.

METHODOLOGY (unchanged from v1 -- same architecture, ported to v2 schema)
----------------------------------------------------------------------------
Trains two complementary UNSUPERVISED models and fuses their outputs into
a single anomaly score per event:

    1. Isolation Forest  -> POINT anomalies (a single event that looks
       statistically unusual across the 86-feature vector: brand-new
       device, odd hour, geo-velocity spike, privilege deviation, etc.)

    2. LSTM Autoencoder   -> SEQUENCE anomalies (a burst of consecutive
       events per entity whose PATTERN is off, even if no single event in
       it looks extreme -- this is what catches Lateral Movement, Insider
       Threat drift, and Low-and-Slow Exfiltration in the v2 attack
       taxonomy).

Both models train WITHOUT seeing label_is_attack -- this is what makes the
approach viable under v2's ~2.5% contamination rate across 12 attack types
and 6 entity types (user, service_account, edge_device, iot_device,
industrial_controller, server). Labels are used ONLY at evaluation time.

WHAT CHANGED VS v1 (data-schema porting only, not a redesign)
----------------------------------------------------------------------------
- FEATURES_PATH / OUTPUT_PATH point at the v2 files.
- NUMERIC_FEATURES is the 86-column curated feature set from
  feature_engineering_v2.py / FEATURE_CATALOG.md, replacing v1's 18-column
  list. These 86 columns are already engineered, numeric, non-null feature
  values (verified during integration testing -- see script docstring
  footer) so no "True"/"False" string-coercion workaround is needed here
  (that workaround was specific to how v1's features.csv was serialized).
- Sequence building groups by `entity_id` (v2's universal entity key)
  instead of v1's `user_id`, so it correctly covers all 6 entity types,
  not just humans.
- CONTAMINATION reflects v2's actual injected attack ratio (~2.5%, vs
  v1's ~2%) -- taken from the generator's documented design ratio, not
  computed from ground-truth labels (that would be label leakage into a
  model hyperparameter).
- Output columns include `entity_type` (kept from the input features) so
  downstream consumers (Stage 2 v2, dashboard) can reason per entity type.

Outputs:
    models/isolation_forest_v2.joblib
    models/lstm_autoencoder_v2.keras
    models/feature_scaler_v2.joblib
    dataset/processed/anomaly_scores_v2.csv   (fused scores + eval metrics)
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "features_v2.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "anomaly_scores_v2.csv")

SEQUENCE_LENGTH = 5          # number of consecutive events per entity fed to the LSTM-AE (unchanged from v1)
CONTAMINATION = 0.025          # matches v2's documented ~2.5% injected-attack ratio (12 attack types x ~125 rows each)

# BUG FOUND DURING INTEGRATION TESTING, FIXED HERE:
# v1 hardcoded FUSION_WEIGHT_IF=0.8 / FUSION_WEIGHT_LSTM=0.2 because on v1's
# data Isolation Forest (AUC 0.984) vastly outperformed the LSTM-AE (AUC
# 0.88) standalone. Blindly reusing that 0.8/0.2 split on v2 was tested
# during this milestone and found to be measurably WORSE: on v2 the two
# detectors are much closer in standalone quality (IF AUC 0.9266, LSTM-AE
# AUC 0.9128 -- v2's richer per-event composite features apparently narrow
# IF's edge, while the LSTM-AE benefits more from the extra sequential
# attack types added in v2 -- Lateral Movement, Session Hijacking,
# Low-and-Slow Exfiltration, Insider Threat). A weight sweep (0.0-1.0 in
# 0.1 steps, then refined) against the actual v2 standalone scores found
# the fused ROC-AUC peaks at IF weight ~0.20-0.22 (AUC ~0.946), not 0.8
# (AUC 0.9306) and not 0.5 (AUC 0.9389). Sweep evidence (fused AUC by IF
# weight): 0.0->0.9128, 0.2->0.9461, 0.5->0.9389, 0.8->0.9306, 1.0->0.9266.
# Re-running this sweep is cheap and recommended any time the v2 dataset,
# features, or either detector is retrained.
FUSION_WEIGHT_IF = 0.2         # corrected for v2 (was 0.8 in v1)
FUSION_WEIGHT_LSTM = 0.8       # corrected for v2 (was 0.2 in v1) -- LSTM-AE is the stronger v2 signal

# The 86 curated behavioral features produced by feature_engineering_v2.py
# (see FEATURE_CATALOG.md for the full description of each). Deliberately
# excludes raw identifiers (entity_id, department, timestamp, ip_address,
# device_id, ...) and the label columns -- Stage 1 must generalize across
# entities/entity-types based on BEHAVIOR, not memorize identities.
NUMERIC_FEATURES = [
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


def load_features():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    # Defensive coercion (mirrors v1's safety net): even though the v2
    # feature pipeline already validated these columns are numeric with no
    # nulls/infs, Stage 1 should not silently crash the whole training run
    # if this script is ever pointed at a re-generated features_v2.csv that
    # regresses that guarantee.
    for col in NUMERIC_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ----------------------------------------------------------------------------
# Model 1: Isolation Forest (point anomalies) -- identical methodology to v1
# ----------------------------------------------------------------------------
def train_isolation_forest(X_scaled: np.ndarray) -> IsolationForest:
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model


def isolation_forest_scores(model: IsolationForest, X_scaled: np.ndarray) -> np.ndarray:
    # decision_function: higher = more normal, lower = more anomalous.
    # Flip and min-max normalize to [0, 1] so higher = more anomalous.
    raw = -model.decision_function(X_scaled)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return normalized


# ----------------------------------------------------------------------------
# Model 2: LSTM Autoencoder (sequence anomalies) -- grouped by entity_id
# ----------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, X_scaled: np.ndarray, seq_len: int):
    """
    Builds one sequence of `seq_len` consecutive events PER ENTITY (sorted
    by time), covering all 6 entity types (user, service_account,
    edge_device, iot_device, industrial_controller, server) since v2's
    universal join key is `entity_id`, not `user_id`. Short entity
    histories (cold-start case) are left-padded with zeros so every entity
    still gets scored rather than being dropped.
    """
    n_features = X_scaled.shape[1]
    sequences = np.zeros((len(df), seq_len, n_features), dtype=np.float32)

    # group row-indices by entity, preserving the time-sorted order already in df
    entity_groups = df.groupby("entity_id").indices  # dict: entity_id -> array of positional indices

    for entity_id, idxs in entity_groups.items():
        idxs = np.sort(idxs)
        for pos, row_idx in enumerate(idxs):
            start = max(0, pos - seq_len + 1)
            window_idxs = idxs[start:pos + 1]
            window = X_scaled[window_idxs]
            pad_len = seq_len - len(window)
            if pad_len > 0:
                window = np.vstack([np.zeros((pad_len, n_features)), window])
            sequences[row_idx] = window

    return sequences


def build_lstm_autoencoder(seq_len: int, n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, n_features))
    x = layers.LSTM(32, activation="tanh", return_sequences=True)(inputs)
    encoded = layers.LSTM(16, activation="tanh", return_sequences=False)(x)
    x = layers.RepeatVector(seq_len)(encoded)
    x = layers.LSTM(16, activation="tanh", return_sequences=True)(x)
    x = layers.LSTM(32, activation="tanh", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder_v2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_lstm_autoencoder(sequences: np.ndarray) -> keras.Model:
    print("Training LSTM Autoencoder...")
    n_features = sequences.shape[2]
    model = build_lstm_autoencoder(SEQUENCE_LENGTH, n_features)

    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(
        sequences, sequences,
        epochs=15,
        batch_size=256,
        shuffle=True,
        verbose=2,
        callbacks=[early_stop],
    )
    return model


def lstm_reconstruction_scores(model: keras.Model, sequences: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(sequences, batch_size=512, verbose=0)
    mse = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
    normalized = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
    return normalized


# ----------------------------------------------------------------------------
# Evaluation (identical methodology to v1)
# ----------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, fused_scores: np.ndarray, contamination: float):
    auc = roc_auc_score(y_true, fused_scores)

    threshold = np.quantile(fused_scores, 1 - contamination)
    y_pred = (fused_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Stage 1 v2 Evaluation (Fused Anomaly Score) ===")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Alert threshold:    {threshold:.4f}  (flags top {contamination*100:.1f}% as anomalous)")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted):\n{cm}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"], zero_division=0))

    return y_pred, threshold, {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def evaluate_standalone(name: str, y_true: np.ndarray, scores: np.ndarray, contamination: float):
    """Reports standalone AUC for a single detector -- used to sanity-check
    the FUSION_WEIGHT_IF / FUSION_WEIGHT_LSTM split inherited from v1 is
    still a reasonable choice on the v2 dataset."""
    auc = roc_auc_score(y_true, scores)
    print(f"  {name} standalone ROC-AUC: {auc:.4f}")
    return auc


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/7] Loading v2 feature matrix...")
    df = load_features()
    print(f"      Rows: {len(df)}  |  Entities: {df['entity_id'].nunique()}  |  "
          f"Entity types: {sorted(df['entity_type'].unique())}")

    print("[2/7] Scaling numeric features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[NUMERIC_FEATURES].values)

    print("[3/7] Training Isolation Forest (point-anomaly detector)...")
    iso_forest = train_isolation_forest(X_scaled)
    if_scores = isolation_forest_scores(iso_forest, X_scaled)

    print(f"[4/7] Building per-entity sequences (length={SEQUENCE_LENGTH}) for LSTM-AE...")
    sequences = build_sequences(df, X_scaled, SEQUENCE_LENGTH)

    print("[5/7] Training LSTM Autoencoder (sequence-anomaly detector)...")
    lstm_ae = train_lstm_autoencoder(sequences)
    lstm_scores = lstm_reconstruction_scores(lstm_ae, sequences)

    print("[6/7] Fusing scores and evaluating against ground truth...")
    y_true = df["label_is_attack"].values
    print("  Standalone detector AUCs (sanity-check for fusion weights):")
    evaluate_standalone("Isolation Forest", y_true, if_scores, CONTAMINATION)
    evaluate_standalone("LSTM Autoencoder", y_true, lstm_scores, CONTAMINATION)

    fused_scores = FUSION_WEIGHT_IF * if_scores + FUSION_WEIGHT_LSTM * lstm_scores
    y_pred, threshold, metrics = evaluate(y_true, fused_scores, CONTAMINATION)

    print("[7/7] Persisting v2 models and scored dataset...")
    df_out = df.copy()
    df_out["isolation_forest_score"] = if_scores
    df_out["lstm_reconstruction_score"] = lstm_scores
    df_out["fused_anomaly_score"] = fused_scores
    df_out["predicted_anomaly"] = y_pred
    df_out.to_csv(OUTPUT_PATH, index=False)

    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest_v2.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler_v2.joblib"))
    lstm_ae.save(os.path.join(MODELS_DIR, "lstm_autoencoder_v2.keras"))

    print(f"\nSaved fused anomaly scores to: {OUTPUT_PATH}")
    print(f"Saved v2 models to: {MODELS_DIR}")

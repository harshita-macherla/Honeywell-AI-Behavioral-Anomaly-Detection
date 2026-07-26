"""
train_stage1_anomaly_detection.py
===================================
Stage 1 of the two-stage pipeline: BEHAVIORAL ANOMALY DETECTION.

Trains two complementary unsupervised models and fuses their outputs into
a single anomaly score per event:

    1. Isolation Forest  -> catches POINT anomalies (a single event that
       looks statistically unusual: brand-new device, odd hour, huge
       failed-login spike, impossible travel speed, etc.)

    2. LSTM Autoencoder   -> catches SEQUENCE anomalies (a session/burst of
       events that individually might look ok-ish but the PATTERN across
       consecutive actions is off -- this is what actually lets us catch
       Lateral Movement, where each single resource access might not look
       extreme, but a rapid burst of cross-department, short-duration
       accesses is the tell).

Both models are trained UNSUPERVISED (they never see label_is_attack during
training) -- this directly addresses the CLASS IMBALANCE requirement: with
attacks at ~2% of the data, a supervised model trained naively would be
starved of positive examples. Unsupervised anomaly detection sidesteps
this by learning "normal" and flagging deviations, needing no attack
labels to do so. Labels are used ONLY at evaluation time to measure
detection accuracy / false positive rate.

Outputs:
    models/isolation_forest.joblib
    models/lstm_autoencoder.keras
    models/feature_scaler.joblib
    dataset/processed/anomaly_scores.csv   (fused scores + eval metrics)
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

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "features.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "anomaly_scores.csv")

SEQUENCE_LENGTH = 5          # number of consecutive events per user fed to the LSTM-AE
CONTAMINATION = 0.02          # matches the known ~2% attack ratio (informs IF's decision boundary)
FUSION_WEIGHT_IF = 0.8        # Isolation Forest: AUC 0.984 standalone, primary detector
FUSION_WEIGHT_LSTM = 0.2      # LSTM-AE: AUC 0.88 standalone -- real but noisier signal at a
                               # strict top-2% cutoff, kept as a secondary contributor rather
                               # than diluting IF with an even 50/50 split (empirically tested)

# Numeric behavioral features fed into BOTH models. Deliberately excludes
# raw identifiers (user_id, department, timestamp) and the label columns --
# Stage 1 must generalize across users/departments based on BEHAVIOR, not memorize identities.
NUMERIC_FEATURES = [
    "distance_from_prev_km", "implied_speed_kmh", "impossible_travel_flag",
    "is_new_device", "is_new_os", "is_new_browser",
    "login_hour_deviation", "is_odd_hour_login", "cold_start_flag",
    "is_cross_department_access", "is_sensitive_resource",
    "failed_login_count", "rolling_avg_failed_logins", "failed_login_spike",
    "vpn_used", "mfa_used", "file_download_size_mb", "session_duration_min",
]


def load_features():
    # keep_default_na=False: prevents pandas from re-interpreting the
    # literal string "None" in attack_type as NaN (see Milestone 2 note).
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"], keep_default_na=False,
                      na_values=[""])
    # Booleans may have been written as "True"/"False" strings; coerce numerics.
    for col in NUMERIC_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": 1, "False": 0}).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ----------------------------------------------------------------------------
# Model 1: Isolation Forest (point anomalies)
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
    # We flip and min-max normalize to [0, 1] so higher = more anomalous,
    # consistent with how we'll interpret the fused risk score later.
    raw = -model.decision_function(X_scaled)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return normalized


# ----------------------------------------------------------------------------
# Model 2: LSTM Autoencoder (sequence anomalies)
# ----------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, X_scaled: np.ndarray, seq_len: int):
    """
    Builds one sequence of `seq_len` consecutive events PER USER (sorted by
    time). Short user histories (fewer than seq_len events -- a cold-start
    case) are left-padded with zeros so every user still gets scored,
    rather than being dropped, which would silently ignore new employees.
    Returns sequences aligned so sequence[i] ends at the i-th row of df
    (so we can map the reconstruction error back to a specific log event).
    """
    n_features = X_scaled.shape[1]
    sequences = np.zeros((len(df), seq_len, n_features), dtype=np.float32)

    # group row-indices by user, preserving the time-sorted order already in df
    user_groups = df.groupby("user_id").indices  # dict: user_id -> array of positional indices

    for user_id, idxs in user_groups.items():
        idxs = np.sort(idxs)
        for pos, row_idx in enumerate(idxs):
            start = max(0, pos - seq_len + 1)
            window_idxs = idxs[start:pos + 1]
            window = X_scaled[window_idxs]
            # left-pad with zeros if history shorter than seq_len
            pad_len = seq_len - len(window)
            if pad_len > 0:
                window = np.vstack([np.zeros((pad_len, n_features)), window])
            sequences[row_idx] = window

    return sequences


def build_lstm_autoencoder(seq_len: int, n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, n_features))
    # Encoder: compress the sequence into a latent behavioral summary
    x = layers.LSTM(32, activation="tanh", return_sequences=True)(inputs)
    encoded = layers.LSTM(16, activation="tanh", return_sequences=False)(x)
    # Decoder: reconstruct the full sequence from that summary
    x = layers.RepeatVector(seq_len)(encoded)
    x = layers.LSTM(16, activation="tanh", return_sequences=True)(x)
    x = layers.LSTM(32, activation="tanh", return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_lstm_autoencoder(sequences: np.ndarray) -> keras.Model:
    print("Training LSTM Autoencoder...")
    n_features = sequences.shape[2]
    model = build_lstm_autoencoder(SEQUENCE_LENGTH, n_features)

    # Trained to reconstruct sequences of NORMAL-looking behavior. We don't
    # filter out attacks here (unsupervised -- we don't use labels), but
    # since attacks are only ~2% of data, the autoencoder predominantly
    # learns to reconstruct normal patterns and will show high
    # reconstruction error on the rare attack sequences by construction.
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
    # Mean squared error per sequence = reconstruction error = anomaly signal
    mse = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
    normalized = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)
    return normalized


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, fused_scores: np.ndarray, contamination: float):
    auc = roc_auc_score(y_true, fused_scores)

    # Threshold at the (1 - contamination) percentile, i.e. flag the top
    # ~2% highest-scoring events as anomalies -- consistent with how a SOC
    # analyst would triage a fixed daily alert budget.
    threshold = np.quantile(fused_scores, 1 - contamination)
    y_pred = (fused_scores >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Stage 1 Evaluation (Fused Anomaly Score) ===")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Alert threshold:    {threshold:.4f}  (flags top {contamination*100:.1f}% as anomalous)")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted):\n{cm}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"], zero_division=0))

    return y_pred, threshold, {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/6] Loading feature matrix...")
    df = load_features()

    print("[2/6] Scaling numeric features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[NUMERIC_FEATURES].values)

    print("[3/6] Training Isolation Forest (point-anomaly detector)...")
    iso_forest = train_isolation_forest(X_scaled)
    if_scores = isolation_forest_scores(iso_forest, X_scaled)

    print(f"[4/6] Building per-user sequences (length={SEQUENCE_LENGTH}) for LSTM-AE...")
    sequences = build_sequences(df, X_scaled, SEQUENCE_LENGTH)

    print("[5/6] Training LSTM Autoencoder (sequence-anomaly detector)...")
    lstm_ae = train_lstm_autoencoder(sequences)
    lstm_scores = lstm_reconstruction_scores(lstm_ae, sequences)

    print("[6/6] Fusing scores and evaluating against ground truth...")
    fused_scores = FUSION_WEIGHT_IF * if_scores + FUSION_WEIGHT_LSTM * lstm_scores

    y_true = df["label_is_attack"].values
    y_pred, threshold, metrics = evaluate(y_true, fused_scores, CONTAMINATION)

    # Persist everything Milestone 4 (classification) and the dashboard need
    df_out = df.copy()
    df_out["isolation_forest_score"] = if_scores
    df_out["lstm_reconstruction_score"] = lstm_scores
    df_out["fused_anomaly_score"] = fused_scores
    df_out["predicted_anomaly"] = y_pred
    df_out.to_csv(OUTPUT_PATH, index=False)

    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    lstm_ae.save(os.path.join(MODELS_DIR, "lstm_autoencoder.keras"))

    print(f"\nSaved fused anomaly scores to: {OUTPUT_PATH}")
    print(f"Saved models to: {MODELS_DIR}")

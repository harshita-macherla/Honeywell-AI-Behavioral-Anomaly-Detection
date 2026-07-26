"""
train_stage2_classification.py
================================
Stage 2 of the two-stage pipeline: ATTACK CLASSIFICATION.

Given an event (whether or not Stage 1 already flagged it), classify it
into one of 6 classes: None (benign) or one of the 5 attack types.

Trained on the FULL dataset rather than only Stage-1-flagged rows -- this
is deliberate: Stage 1 has ~33% false negatives on some attack types at a
strict alert threshold, and ~330 false positives. If Stage 2 only ever saw
Stage-1-flagged rows, it would never learn to say "this looks anomalous
but it's actually benign", which is exactly what an analyst needs when
triaging Stage-1 alerts. Training on the full set lets Stage 2 act as an
independent second opinion, not just a rubber stamp on Stage 1's output.

Model choice: XGBoost multiclass (multi:softprob).
    - Handles the severe class imbalance (5 attack classes at ~200 each vs.
      ~49,000 benign) via per-sample balanced weighting.
    - Tree-based -> pairs with SHAP's TreeExplainer, which is EXACT and fast
      (unlike KernelSHAP needed for neural nets) -- critical since the
      dashboard needs real-time, per-alert explanations, not just global
      feature importance computed once offline.
    - Naturally handles the mixed numeric/binary feature set we engineered
      in Milestone 2 without needing embeddings or extensive scaling.

Outputs:
    models/xgb_attack_classifier.joblib
    models/label_encoder.joblib
    dataset/processed/stage2_predictions.csv
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

ANOMALY_SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "anomaly_scores.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "stage2_predictions.csv")

# Stage 2 gets the same behavioral features as Stage 1, PLUS Stage 1's own
# outputs (fused score, IF score, LSTM score) as additional features --
# Stage 1's opinion is itself informative signal for Stage 2 to weigh.
CLASSIFIER_FEATURES = [
    "distance_from_prev_km", "implied_speed_kmh", "impossible_travel_flag",
    "is_new_device", "is_new_os", "is_new_browser",
    "login_hour_deviation", "is_odd_hour_login", "cold_start_flag",
    "is_cross_department_access", "is_sensitive_resource",
    "failed_login_count", "rolling_avg_failed_logins", "failed_login_spike",
    "vpn_used", "mfa_used", "file_download_size_mb", "session_duration_min",
    "isolation_forest_score", "lstm_reconstruction_score", "fused_anomaly_score",
]


def load_data():
    df = pd.read_csv(ANOMALY_SCORES_PATH, parse_dates=["timestamp"],
                      keep_default_na=False, na_values=[""])
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


def explain_top_reasons(shap_row: np.ndarray, feature_names: list, top_k: int = 5) -> list:
    """
    Converts a single row of SHAP values into a ranked, human-readable list
    of reasons -- this is the bridge between raw SHAP numbers and the
    dashboard's "Reasons: New Device, Impossible Travel, ..." style output.
    """
    contributions = list(zip(feature_names, shap_row))
    # Rank by absolute contribution to the predicted class (most influential first)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    readable_names = {
        "distance_from_prev_km": "Large geographic jump since last login",
        "implied_speed_kmh": "Impossible travel speed",
        "impossible_travel_flag": "Impossible travel detected",
        "is_new_device": "New/unrecognized device",
        "is_new_os": "Unrecognized operating system",
        "is_new_browser": "Unrecognized browser",
        "login_hour_deviation": "Login time deviates from usual pattern",
        "is_odd_hour_login": "Midnight/off-hours login",
        "cold_start_flag": "Limited historical baseline for this user",
        "is_cross_department_access": "Cross-department resource access",
        "is_sensitive_resource": "Sensitive resource access",
        "failed_login_count": "High failed login count",
        "rolling_avg_failed_logins": "Failed logins above personal norm",
        "failed_login_spike": "Sudden spike in failed logins",
        "vpn_used": "VPN usage pattern",
        "mfa_used": "MFA status anomaly",
        "file_download_size_mb": "Unusual file download volume",
        "session_duration_min": "Abnormal session duration",
        "isolation_forest_score": "Flagged by point-anomaly detector",
        "lstm_reconstruction_score": "Flagged by sequence-anomaly detector",
        "fused_anomaly_score": "High overall anomaly score",
    }

    reasons = []
    for feat, val in contributions[:top_k]:
        if abs(val) < 1e-4:
            continue  # skip negligible contributions
        reasons.append(readable_names.get(feat, feat))
    return reasons


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/6] Loading data with Stage 1 outputs...")
    df = load_data()

    print("[2/6] Encoding attack_type labels...")
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
    # Compute SHAP for the test set (fast, exact for tree models)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features, n_classes) or list per class

    # Handle both shap API shapes across versions
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)

    # Demonstrate explainability on a few actual attack examples
    print("\n=== Sample Explainable Predictions ===")
    attack_mask = y_test != label_encoder.transform(["None"])[0]
    sample_positions = np.where(attack_mask)[0][:3]
    for pos in sample_positions:
        predicted_class_idx = y_pred[pos]
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        true_class = label_encoder.inverse_transform([y_test[pos]])[0]
        row_shap = shap_values[pos, :, predicted_class_idx] if shap_values.ndim == 3 else shap_values[pos]
        reasons = explain_top_reasons(row_shap, CLASSIFIER_FEATURES, top_k=5)
        original_row_idx = idx_test[pos]
        fused_score = df.loc[original_row_idx, "fused_anomaly_score"]
        print(f"\nEvent {original_row_idx} | True: {true_class} | Predicted: {predicted_class}")
        print(f"  Anomaly Score (0-1): {fused_score:.3f}")
        print(f"  Reasons: {', '.join(reasons)}")

    # Persist model + encoder for Milestone 5 (risk scoring) and the API layer
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_attack_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    # Save full predictions (train+test) for downstream risk scoring / dashboard
    all_pred = model.predict(X)
    all_pred_proba = model.predict_proba(X)
    df_out = df.copy()
    df_out["predicted_attack_type"] = label_encoder.inverse_transform(all_pred)
    df_out["prediction_confidence"] = all_pred_proba.max(axis=1)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved classifier + label encoder to: {MODELS_DIR}")
    print(f"Saved Stage 2 predictions to: {OUTPUT_PATH}")

"""
schemas.py
==========
Pydantic request/response models for the API.

The 86 engineered features consumed by Stage 1/2 v2 are accepted as a
single `features` dict rather than 86 individual Pydantic fields -- this
mirrors how a real feature-store-fed model server receives input (feature
engineering already happened upstream in scripts/feature_engineering_v2.py;
this backend serves MODEL INFERENCE on already-engineered features, it does
not reimplement the stateful, historical-rolling-window feature pipeline).
Field-level validation against the exact 86-name schema happens in
services/scoring_service.py, reusing scripts/train_stage1_anomaly_detection_v2.py's
NUMERIC_FEATURES list as the single source of truth -- not duplicated here.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# Health / status
# ----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    path: str


class StatusResponse(BaseModel):
    status: str
    api_version: str
    models: List[ModelStatus]
    dataset_rows_loaded: int
    entities_indexed: int
    uptime_seconds: float


# ----------------------------------------------------------------------------
# Prediction (live scoring of a new event through Stage 1 -> Stage 2 -> Risk)
# ----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    entity_id: str = Field(..., example="U0131")
    entity_type: str = Field(..., example="user")
    timestamp: Optional[str] = Field(
        None, description="ISO timestamp of the new event; used only for the response payload."
    )
    resource_sensitivity: int = Field(
        ..., ge=1, le=5, description="Graded resource sensitivity (1-5), used by the rule engine."
    )
    failed_login_count: int = Field(0, ge=0, description="Used by the rule engine.")
    mfa_used: bool = Field(True, description="Used by the entity-type-aware MFA rule.")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "The 86 engineered behavioral features produced by "
            "feature_engineering_v2.py (see FEATURE_CATALOG.md). All 86 keys "
            "are required; unknown/extra keys are rejected."
        ),
    )


class PredictResponse(BaseModel):
    entity_id: str
    entity_type: str
    timestamp: Optional[str]
    cold_start: bool = Field(..., description="True if fewer than SEQUENCE_LENGTH-1 prior events were found for this entity.")
    history_events_used: int = Field(..., description="Number of prior events actually available for the LSTM-AE sequence.")
    isolation_forest_score: float
    lstm_reconstruction_score: float
    fused_anomaly_score: float
    fused_anomaly_score_percentile: float = Field(
        ..., description="Percentile rank of fused_anomaly_score against the historical v2 dataset."
    )
    predicted_attack_type: Optional[str]
    prediction_confidence: float
    rule_based_score: float
    triggered_rules: List[str]
    risk_score: float
    risk_level: str
    reasons: List[str]


# ----------------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    feature: str
    readable_name: str
    shap_value: float


class ExplainResponse(BaseModel):
    log_id: str
    entity_id: str
    entity_type: str
    timestamp: str
    predicted_attack_type: Optional[str]
    actual_attack_type: Optional[str]
    prediction_confidence: float
    risk_score: float
    risk_level: str
    rule_reasons: List[str]
    shap_reasons: List[str]
    merged_reasons: List[str]
    top_shap_contributions: List[FeatureContribution]


# ----------------------------------------------------------------------------
# Analyst dashboard endpoints
# ----------------------------------------------------------------------------
class AlertSummary(BaseModel):
    log_id: str
    entity_id: str
    entity_type: str
    department: Optional[str]
    timestamp: str
    predicted_attack_type: Optional[str]
    actual_attack_type: Optional[str]
    prediction_confidence: float
    risk_score: float
    risk_level: str
    reasons: List[str]


class AlertListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    alerts: List[AlertSummary]


class EntityHistoryEvent(BaseModel):
    log_id: str
    timestamp: str
    resource_accessed: Optional[str]
    predicted_attack_type: Optional[str]
    actual_attack_type: Optional[str]
    risk_score: float
    risk_level: str


class EntityHistoryResponse(BaseModel):
    entity_id: str
    entity_type: str
    department: Optional[str]
    role: Optional[str]
    total_events: int
    max_risk_score: float
    attack_event_count: int
    events: List[EntityHistoryEvent]


class RiskLevelCounts(BaseModel):
    Critical: int
    High: int
    Medium: int
    Low: int


class EntityTypeBreakdown(BaseModel):
    entity_type: str
    total_events: int
    critical_alerts: int
    high_alerts: int


class StatsOverviewResponse(BaseModel):
    total_events: int
    total_entities: int
    risk_level_counts: RiskLevelCounts
    entity_type_breakdown: List[EntityTypeBreakdown]
    top_predicted_attack_types: Dict[str, int]
    critical_alert_precision: Optional[float] = Field(
        None, description="Fraction of Critical-tier alerts that are true positives (ground truth). "
                           "Included for report/demo purposes; a live SOC would not have this label."
    )

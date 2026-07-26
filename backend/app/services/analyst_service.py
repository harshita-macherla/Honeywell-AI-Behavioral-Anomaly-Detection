"""
analyst_service.py
=====================
Serves the analyst-facing dashboard queries (PDF deliverable #6: "ranked
alert queue, risk score, contributing factors, entity history view") from
the in-memory copy of risk_scoring_engine_v2.py's output. No scoring
happens here -- this module only filters/sorts/aggregates the already
fully-scored dataset held in AppState.
"""

from typing import Optional
import pandas as pd
from fastapi import HTTPException

from ..dependencies import AppState

RISK_LEVEL_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}


def _row_to_alert_summary(row: pd.Series) -> dict:
    reasons = row["reasons_str"].split(", ") if row["reasons_str"] != "No significant risk factors" else []
    return {
        "log_id": row["log_id"],
        "entity_id": row["entity_id"],
        "entity_type": row["entity_type"],
        "department": row["department"] if pd.notna(row["department"]) else None,
        "timestamp": str(row["timestamp"]),
        "predicted_attack_type": row["predicted_attack_type"] if pd.notna(row["predicted_attack_type"]) else None,
        "actual_attack_type": row["attack_type"] if pd.notna(row["attack_type"]) else None,
        "prediction_confidence": round(float(row["prediction_confidence"]), 4),
        "risk_score": float(row["risk_score"]),
        "risk_level": row["risk_level"],
        "reasons": reasons,
    }


def list_alerts(
    state: AppState,
    risk_level: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    min_risk_score: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    df = state.dataset

    if risk_level is not None:
        df = df[df["risk_level"] == risk_level]
    if entity_type is not None:
        df = df[df["entity_type"] == entity_type]
    if entity_id is not None:
        df = df[df["entity_id"] == entity_id]
    if min_risk_score is not None:
        df = df[df["risk_score"] >= min_risk_score]

    total = len(df)
    df = df.sort_values("risk_score", ascending=False)
    page = df.iloc[offset: offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "alerts": [_row_to_alert_summary(row) for _, row in page.iterrows()],
    }


def get_alert(state: AppState, log_id: str) -> dict:
    matches = state.dataset[state.dataset["log_id"] == log_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"log_id '{log_id}' not found")
    return _row_to_alert_summary(matches.iloc[0])


def get_entity_history(state: AppState, entity_id: str, limit: int = 100) -> dict:
    idxs = state.entity_index.get(entity_id)
    if not idxs:
        raise HTTPException(status_code=404, detail=f"entity_id '{entity_id}' not found")

    df = state.dataset.loc[idxs].sort_values("timestamp", ascending=False)
    first_row = state.dataset.loc[idxs[0]]

    events = [
        {
            "log_id": row["log_id"],
            "timestamp": str(row["timestamp"]),
            "resource_accessed": row["resource_accessed"] if pd.notna(row["resource_accessed"]) else None,
            "predicted_attack_type": row["predicted_attack_type"] if pd.notna(row["predicted_attack_type"]) else None,
            "actual_attack_type": row["attack_type"] if pd.notna(row["attack_type"]) else None,
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
        }
        for _, row in df.head(limit).iterrows()
    ]

    return {
        "entity_id": entity_id,
        "entity_type": first_row["entity_type"],
        "department": first_row["department"] if pd.notna(first_row["department"]) else None,
        "role": first_row["role"] if pd.notna(first_row["role"]) else None,
        "total_events": len(idxs),
        "max_risk_score": float(df["risk_score"].max()),
        "attack_event_count": int(df["attack_type"].notna().sum()),
        "events": events,
    }


def get_stats_overview(state: AppState) -> dict:
    df = state.dataset

    risk_counts = df["risk_level"].value_counts()
    risk_level_counts = {level: int(risk_counts.get(level, 0)) for level in ["Critical", "High", "Medium", "Low"]}

    entity_type_breakdown = []
    for entity_type, group in df.groupby("entity_type"):
        entity_type_breakdown.append({
            "entity_type": entity_type,
            "total_events": int(len(group)),
            "critical_alerts": int((group["risk_level"] == "Critical").sum()),
            "high_alerts": int((group["risk_level"] == "High").sum()),
        })
    entity_type_breakdown.sort(key=lambda d: d["total_events"], reverse=True)

    top_attack_types = (
        df["predicted_attack_type"].dropna().value_counts().head(10).to_dict()
    )
    top_attack_types = {k: int(v) for k, v in top_attack_types.items()}

    critical = df[df["risk_level"] == "Critical"]
    critical_precision = (
        float((critical["label_is_attack"] == 1).mean()) if len(critical) > 0 else None
    )

    return {
        "total_events": int(len(df)),
        "total_entities": int(df["entity_id"].nunique()),
        "risk_level_counts": risk_level_counts,
        "entity_type_breakdown": entity_type_breakdown,
        "top_predicted_attack_types": top_attack_types,
        "critical_alert_precision": round(critical_precision, 4) if critical_precision is not None else None,
    }

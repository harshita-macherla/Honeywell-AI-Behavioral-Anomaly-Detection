"""
analyst.py
==========
Analyst-facing dashboard endpoints (PDF deliverable #6): a ranked alert
queue with filtering, per-alert detail, entity history view, and a
dashboard summary/stats endpoint. All served from the in-memory copy of
risk_scoring_engine_v2.py's output -- see services/analyst_service.py.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from ..dependencies import AppState, get_state
from ..schemas import AlertListResponse, AlertSummary, EntityHistoryResponse, StatsOverviewResponse
from ..services import analyst_service

router = APIRouter(prefix="/api/v1", tags=["analyst"])


@router.get("/alerts", response_model=AlertListResponse)
def list_alerts(
    risk_level: Optional[str] = Query(None, description="Filter: Critical | High | Medium | Low"),
    entity_type: Optional[str] = Query(None, description="Filter: user | service_account | edge_device | iot_device | industrial_controller | server"),
    entity_id: Optional[str] = Query(None),
    min_risk_score: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    state: AppState = Depends(get_state),
):
    return analyst_service.list_alerts(
        state, risk_level=risk_level, entity_type=entity_type, entity_id=entity_id,
        min_risk_score=min_risk_score, limit=limit, offset=offset,
    )


@router.get("/alerts/{log_id}", response_model=AlertSummary)
def get_alert(log_id: str, state: AppState = Depends(get_state)):
    return analyst_service.get_alert(state, log_id)


@router.get("/entities/{entity_id}/history", response_model=EntityHistoryResponse)
def entity_history(
    entity_id: str,
    limit: int = Query(100, ge=1, le=1000),
    state: AppState = Depends(get_state),
):
    return analyst_service.get_entity_history(state, entity_id, limit=limit)


@router.get("/stats/overview", response_model=StatsOverviewResponse)
def stats_overview(state: AppState = Depends(get_state)):
    return analyst_service.get_stats_overview(state)

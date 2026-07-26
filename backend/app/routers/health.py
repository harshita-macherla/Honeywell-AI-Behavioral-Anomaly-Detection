"""
health.py
=========
Health and status endpoints. /health is a plain liveness check (no model
access, always fast); /status reports which v2 models are loaded and basic
dataset stats, useful for confirming a deployment actually has the trained
artifacts available before analysts start relying on it.
"""

from fastapi import APIRouter, Depends

from ..dependencies import AppState, get_state
from ..schemas import HealthResponse, StatusResponse, ModelStatus
from .. import config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@router.get("/api/v1/status", response_model=StatusResponse)
def status(state: AppState = Depends(get_state)):
    return {
        "status": "ok",
        "api_version": config.API_VERSION,
        "models": [ModelStatus(**m) for m in state.model_status],
        "dataset_rows_loaded": int(len(state.dataset)),
        "entities_indexed": len(state.entity_index),
        "uptime_seconds": round(state.uptime_seconds, 1),
    }

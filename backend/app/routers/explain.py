"""
explain.py
==========
Explanation endpoint for events already scored by the batch pipeline.
Given a log_id, returns the merged rule + SHAP reasons plus the top-10
signed SHAP feature contributions behind the Stage 2 v2 classification.
"""

from fastapi import APIRouter, Depends

from ..dependencies import AppState, get_state
from ..schemas import ExplainResponse
from ..services.explain_service import explain_log_id

router = APIRouter(prefix="/api/v1", tags=["explain"])


@router.get("/explain/{log_id}", response_model=ExplainResponse)
def explain(log_id: str, state: AppState = Depends(get_state)):
    return explain_log_id(state, log_id)

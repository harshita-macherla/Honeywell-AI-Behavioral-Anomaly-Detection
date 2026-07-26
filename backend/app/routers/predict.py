"""
predict.py
==========
Live scoring endpoint: runs a new event through Stage 1 v2 -> Stage 2 v2 ->
Risk Scoring v2 using the trained models loaded at startup. See
services/scoring_service.py for the full pipeline reuse details.
"""

from fastapi import APIRouter, Depends

from ..dependencies import AppState, get_state
from ..schemas import PredictRequest, PredictResponse
from ..services.scoring_service import score_event

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, state: AppState = Depends(get_state)):
    return score_event(state, req)

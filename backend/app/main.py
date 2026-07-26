"""
main.py
=======
FastAPI application entrypoint. Loads all v2 models and the risk-scored
v2 dataset ONCE at startup (via the lifespan handler below), then serves
health, prediction, explanation, and analyst-dashboard endpoints against
that already-loaded state for every request.

Run with:
    uvicorn backend.app.main:app --reload --port 8000
(from the project root, so the `backend` package resolves correctly)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config
from .dependencies import load_state
from .routers import health, predict, explain, analyst


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load every v2 model + the scored dataset once. Nothing here
    # trains or regenerates any artifact -- it only reads what the prior
    # milestones already produced.
    load_state()
    yield
    # Shutdown: nothing to clean up (no open connections/files held open).


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(analyst.router)

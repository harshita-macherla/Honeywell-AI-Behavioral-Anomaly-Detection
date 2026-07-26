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
from fastapi.middleware.cors import CORSMiddleware

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

# BUG FOUND DURING FRONTEND INTEGRATION TESTING, FIXED HERE:
# The backend had no CORS configuration. Every page of the React dashboard
# (served from the Vite dev server on a different origin, e.g.
# http://127.0.0.1:5173) rendered correctly -- routing, layout, and login
# all worked with zero React errors -- but every single API call was
# silently blocked by the browser's CORS preflight check
# ("No 'Access-Control-Allow-Origin' header is present"), confirmed via a
# full Playwright run across every route: each page showed "Could not
# reach the API" despite curl-based backend testing (previous milestone)
# proving the endpoints themselves work correctly. This made the entire
# dashboard non-functional end-to-end despite every individual layer
# (backend logic, frontend logic) being correct in isolation.
# Fix: added FastAPI's CORSMiddleware, scoped to local dev origins (the
# Vite dev server's default ports) plus a permissive fallback for any
# localhost/127.0.0.1 port, since this is a demo/hackathon deployment, not
# a public-internet service with untrusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(analyst.router)

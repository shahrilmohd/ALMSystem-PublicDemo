"""
FastAPI application entry point.

Creates the app, configures CORS, mounts all routers, and handles
table creation on startup.

Running locally
---------------
    uv run uvicorn api.main:app --reload --port 8000

Interactive docs (auto-generated from schemas):
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc

CORS
----
Configured to allow all origins in development (allow_origins=["*"]).
In production, replace with the exact frontend domain:
    allow_origins=["https://app.yourcompany.com"]

The desktop app (PyQt6) calls the API via Python's requests library,
not a browser, so CORS does not affect it.  The configuration is for
the Phase 3 web frontend.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env file from the project root (two levels up from api/main.py).
# Using an explicit path avoids any ambiguity about the working directory
# when the server is started.  Safe to call if .env does not exist.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import _engine
from api.routers import ai, batches, config, data, results, runs, workers
from storage.db import create_all_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:  create all DB tables (safe to call repeatedly — uses
              CREATE TABLE IF NOT EXISTS semantics).
    Shutdown: nothing required for SQLite; connection pool is cleaned
              up by SQLAlchemy when the process exits.
    """
    create_all_tables(_engine)
    yield


app = FastAPI(
    title="ALM System API",
    description=(
        "HTTP interface for the segregated fund Asset and Liability Model. "
        "Submit projection runs, poll status, and retrieve results."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict to specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Router mounting
# ---------------------------------------------------------------------------
# Each router handles all paths under its prefix.
# Example: runs.router handles POST / and GET /{run_id}
#          → mounted at /runs → outside world sees POST /runs, GET /runs/{run_id}
app.include_router(runs.router,    prefix="/runs",        tags=["Runs"])
app.include_router(config.router,  prefix="/config",      tags=["Config"])
app.include_router(results.router, prefix="/results",     tags=["Results"])
app.include_router(batches.router, prefix="/batches",     tags=["Batches"])
app.include_router(workers.router, prefix="/workers",     tags=["Workers"])
app.include_router(ai.router,      prefix="/ai",          tags=["AI Assistant"])
app.include_router(data.router,    prefix="/assumptions", tags=["Data"])


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Liveness probe — returns 200 if the API process is running."""
    return {"status": "ok"}

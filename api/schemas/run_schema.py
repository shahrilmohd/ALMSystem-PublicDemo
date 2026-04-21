"""
Pydantic schemas for the /runs endpoints.

Request schema:   SubmitRunRequest  — what the caller sends to POST /runs.
Response schemas: RunStatusResponse — returned by POST /runs and GET /runs/{run_id}.
                  RunListResponse   — returned by GET /runs.

Why separate schemas from ORM models?
    RunRecord (storage/models/run_record.py) is a SQLAlchemy ORM model —
    it maps to a database table and carries SQLAlchemy-specific state.
    Exposing ORM objects directly over HTTP couples the API contract to the
    database schema, meaning a DB column rename silently breaks API clients.
    These Pydantic schemas are the stable public contract. ORM objects are
    converted to schemas inside the router before being returned.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SubmitRunRequest(BaseModel):
    """
    Body for POST /runs.

    config_json:
        The full RunConfig serialised as a JSON string.
        The API stores it as-is; the worker deserialises and validates it
        at execution time.  This allows the API to accept and queue the job
        immediately without importing the engine.

        Use POST /config/validate first to check the config is well-formed
        before submitting.
    """
    config_json: str = Field(
        ...,
        description="RunConfig serialised as a JSON string (use Pydantic model_dump_json()).",
    )


class RunStatusResponse(BaseModel):
    """
    Returned by POST /runs (on submission) and GET /runs/{run_id} (on poll).

    status lifecycle:
        PENDING   → job accepted, not yet picked up by a worker
        RUNNING   → worker has started execution
        COMPLETED → run finished successfully; results are in the DB
        FAILED    → run ended with an error; see error_message
    """
    run_id:           str               = Field(..., description="UUID identifying this run.")
    run_name:         Optional[str]     = Field(None, description="Human-readable label from the submitted RunConfig.")
    run_type:         str               = Field(..., description="LIABILITY_ONLY, DETERMINISTIC, or STOCHASTIC.")
    status:           str               = Field(..., description="PENDING | RUNNING | COMPLETED | FAILED.")
    created_at:       datetime          = Field(..., description="When the run record was created.")
    started_at:       Optional[datetime] = Field(None, description="When the worker began execution.")
    completed_at:     Optional[datetime] = Field(None, description="When execution finished (success or failure).")
    duration_seconds: Optional[float]   = Field(None, description="Wall-clock execution time in seconds.")
    error_message:    Optional[str]     = Field(None, description="First-line error on FAILED runs.")
    n_scenarios:      Optional[int]     = Field(None, description="Number of ESG scenarios (STOCHASTIC only).")
    n_timesteps:      Optional[int]     = Field(None, description="Number of monthly projection steps.")

    model_config = {"from_attributes": True}


class RunListResponse(BaseModel):
    """Returned by GET /runs."""
    runs: list[RunStatusResponse]
    total: int = Field(..., description="Total number of run records in the database.")

"""Pydantic schemas for the /batches endpoints.

Request schema:   SubmitBatchRequest   — what the caller sends to POST /batches.
Response schemas: BatchStatusResponse  — returned by POST /batches and GET /batches/{id}.
                  BatchListResponse    — returned by GET /batches.

Batch status is derived at query time from member run statuses rather than
stored directly, which avoids synchronisation complexity between the batch
record and its constituent run records.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from api.schemas.run_schema import RunStatusResponse


class SubmitBatchRequest(BaseModel):
    """
    Body for POST /batches.

    label:
        Optional human-readable name for the batch (e.g. "Q4 2025 BEL runs").
        Stored as-is; not validated beyond being a string.

    configs:
        List of RunConfig JSON strings, one per run.  Each string must be
        parseable JSON; the worker deserialises and fully validates each
        config at execution time.  Minimum 1 config, maximum 50 per batch.
        Use POST /config/validate first to check each config is well-formed.
    """
    label: Optional[str] = Field(
        None,
        description="Optional name for this batch.",
    )
    configs: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of config_json strings, one per run.",
    )


class BatchStatusResponse(BaseModel):
    """
    Returned by POST /batches (on submission) and GET /batches/{batch_id}.

    status lifecycle (derived from member runs):
        PENDING   → all runs queued, none started yet
        RUNNING   → at least one run is executing
        COMPLETED → every run finished successfully
        FAILED    → at least one run ended with an error

    counts:
        completed_runs — number of COMPLETED member runs
        failed_runs    — number of FAILED member runs
        pending_runs   — number of PENDING member runs (excludes RUNNING)
    """
    batch_id:       str                    = Field(..., description="UUID identifying this batch.")
    label:          Optional[str]          = Field(None, description="Optional user-provided batch name.")
    status:         str                    = Field(..., description="PENDING | RUNNING | COMPLETED | FAILED (derived).")
    created_at:     datetime               = Field(..., description="When the batch was submitted.")
    total_runs:     int                    = Field(..., description="Number of runs in this batch.")
    completed_runs: int                    = Field(..., description="Number of runs with status COMPLETED.")
    failed_runs:    int                    = Field(..., description="Number of runs with status FAILED.")
    pending_runs:   int                    = Field(..., description="Number of runs with status PENDING.")
    runs:           list[RunStatusResponse] = Field(..., description="Status of each member run.")

    model_config = {"from_attributes": True}


class BatchListResponse(BaseModel):
    """Returned by GET /batches."""
    batches: list[BatchStatusResponse]
    total:   int = Field(..., description="Total number of batches in the database.")

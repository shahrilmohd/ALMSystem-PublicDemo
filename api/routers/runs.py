"""
/runs router — run submission and status polling.

Endpoints
---------
POST /runs
    Accept a run config, write a PENDING RunRecord to the DB, return the run_id.
    Step 13 (worker) will add the RQ enqueue call here.

GET /runs
    List all run records ordered by created_at descending.

GET /runs/{run_id}
    Return the current status of a single run.

Architectural rule:
    This router does NOT import anything from engine/.
    It is a thin HTTP wrapper over the storage layer.
    All execution logic lives in the worker (Step 13).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import redis as redis_lib
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.run_schema import RunListResponse, RunStatusResponse, SubmitRunRequest
from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository

router = APIRouter()


@router.post("/", response_model=RunStatusResponse, status_code=201)
def submit_run(body: SubmitRunRequest, db: Session = Depends(get_db)) -> RunStatusResponse:
    """
    Submit a new projection run.

    Validates that config_json is parseable JSON, writes a PENDING RunRecord
    to the database, and returns the run_id immediately.

    The run is NOT executed here — the worker picks it up from the queue
    (Step 13).  Poll GET /runs/{run_id} to track status.

    Returns 400 if config_json is not valid JSON.
    """
    # Validate config_json is at least parseable JSON.
    # Full RunConfig validation happens in the worker at execution time.
    try:
        config_dict = json.loads(body.config_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"config_json is not valid JSON: {exc}") from exc

    # Derive fields from the config dict (best-effort; worker re-validates).
    run_type = str(config_dict.get("run_type", "unknown")).upper()
    run_name = config_dict.get("run_name") or None

    run_id = str(uuid.uuid4())
    record = RunRecord(
        run_id=run_id,
        run_name=run_name,
        run_type=run_type,
        status="PENDING",
        created_at=datetime.now(timezone.utc),
        config_json=body.config_json,
    )

    repo = RunRepository(db)
    repo.save(record)
    db.commit()

    # Enqueue the job.  If Redis is unavailable the run record already exists
    # as PENDING — the user can retry or a health check will surface the issue.
    try:
        from worker.job_queue import get_queue
        queue = get_queue()
        queue.enqueue("worker.tasks.run_alm_job", run_id)
    except redis_lib.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable — Redis is not reachable. The run record has been saved as PENDING.",
        )

    return RunStatusResponse.model_validate(record)


@router.get("/", response_model=RunListResponse)
def list_runs(db: Session = Depends(get_db)) -> RunListResponse:
    """
    Return all run records ordered by created_at descending (most recent first).
    """
    repo = RunRepository(db)
    records = repo.list_all()
    return RunListResponse(
        runs=[RunStatusResponse.model_validate(r) for r in records],
        total=len(records),
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunStatusResponse:
    """
    Return the current status of a single run.

    Returns 404 if no run with the given run_id exists.
    """
    repo = RunRepository(db)
    try:
        record = repo.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")
    return RunStatusResponse.model_validate(record)

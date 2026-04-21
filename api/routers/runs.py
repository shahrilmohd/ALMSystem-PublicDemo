"""
/runs router — run submission, status polling, and model point access.

Endpoints
---------
POST /runs
    Accept a run config, write a PENDING RunRecord to the DB, return the run_id.
    Step 13 (worker) will add the RQ enqueue call here.

GET /runs
    List all run records ordered by created_at descending.

GET /runs/{run_id}
    Return the current status of a single run.

GET /runs/{run_id}/model_points?population_type=...
    Return up to 500 rows of BPA model point data for a run, optionally
    filtered to one population type.  The path is read from the run's
    stored config_json; the CSV is loaded on-demand (not persisted in the DB).
    Returns 404 if the run does not exist or is not a BPA run.
    Returns 422 if population_type is not a recognised value.

Architectural rule:
    This router does NOT import anything from engine/.
    It is a thin HTTP wrapper over the storage layer.
    All execution logic lives in the worker (Step 13).
"""
from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import redis as redis_lib
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.data_schema import ModelPointsResponse, _MAX_ROWS
from api.schemas.run_schema import RunListResponse, RunStatusResponse, SubmitRunRequest
from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository

_VALID_POPULATION_TYPES = {"in_payment", "deferred", "dependant", "enhanced"}

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


@router.get("/{run_id}/model_points", response_model=ModelPointsResponse)
def get_model_points(
    run_id: str,
    population_type: Optional[str] = Query(
        default=None,
        description=(
            "Filter by population type: 'in_payment', 'deferred', "
            "'dependant', or 'enhanced'. Omit to return all rows."
        ),
    ),
    db: Session = Depends(get_db),
) -> ModelPointsResponse:
    """
    Return BPA model point rows for a run.

    Reads the CSV path stored in the run's config_json, loads the file,
    optionally filters by population_type, and returns up to 500 rows.

    Returns **404** if the run does not exist or is not a BPA run.
    Returns **422** if population_type is not a recognised value.
    Returns **503** if the model point file cannot be read.
    """
    # Validate population_type before touching the DB.
    if population_type is not None and population_type not in _VALID_POPULATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown population_type {population_type!r}. "
                f"Valid values: {sorted(_VALID_POPULATION_TYPES)}"
            ),
        )

    # Fetch run record.
    repo = RunRepository(db)
    try:
        record = repo.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")

    # Parse config_json — no engine import required (plain dict navigation).
    try:
        config = json.loads(record.config_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Run config_json is not valid JSON: {exc}",
        ) from exc

    bpa_inputs = (config.get("input_sources") or {}).get("bpa_inputs")
    if not bpa_inputs:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Run {run_id!r} has no BPA inputs configured (run_type is not 'bpa' "
                "or bpa_inputs block is absent)."
            ),
        )

    mp_path_str = bpa_inputs.get("bpa_model_points_path")
    if not mp_path_str:
        raise HTTPException(
            status_code=503,
            detail=f"Run {run_id!r}: bpa_model_points_path is missing from config.",
        )

    mp_path = Path(mp_path_str)
    if not mp_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model point file not found on server: {mp_path}",
        )

    # Load CSV — cap at _MAX_ROWS + 1 rows so we never materialise a large
    # file fully into memory.  The +1 lets us detect truncation correctly.
    try:
        df = pd.read_csv(mp_path, nrows=_MAX_ROWS + 1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read model point file: {exc}",
        ) from exc

    # Filter by population_type if requested.
    if population_type is not None:
        if "population_type" not in df.columns:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The model point file has no 'population_type' column; "
                    "cannot filter."
                ),
            )
        df = df[df["population_type"] == population_type]

    total_rows = len(df)
    truncated = total_rows > _MAX_ROWS
    df = df.head(_MAX_ROWS)

    return ModelPointsResponse(
        run_id=run_id,
        population_type=population_type,
        row_count=len(df),
        truncated=truncated,
        columns=list(df.columns),
        data=df.where(pd.notna(df), other=None).to_dict(orient="records"),
    )

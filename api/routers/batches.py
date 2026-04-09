"""
/batches router — batch run submission and status polling.

Endpoints
---------
POST /batches
    Accept a list of run configs, write a BatchRecord and one PENDING
    RunRecord per config to the DB, enqueue all runs, return batch status.

GET /batches
    List all batch records ordered by created_at descending.

GET /batches/{batch_id}
    Return the current status of a single batch (and its member runs).

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
from api.schemas.batch_schema import (
    BatchListResponse,
    BatchStatusResponse,
    SubmitBatchRequest,
)
from api.schemas.run_schema import RunStatusResponse
from storage.batch_repository import BatchRepository
from storage.models.batch_record import BatchRecord
from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_batch_response(
    batch_repo: BatchRepository,
    batch_id: str,
) -> BatchStatusResponse:
    """
    Assemble a BatchStatusResponse from repository data.

    Fetches the BatchRecord, member RunRecords, and derived status from the
    repository layer and maps them into the API schema.

    Parameters
    ----------
    batch_repo : BatchRepository
        Repository instance bound to the current DB session.
    batch_id : str
        UUID of the batch to describe.
    """
    batch = batch_repo.get(batch_id)
    member_runs = batch_repo.get_member_runs(batch_id)
    status = batch_repo.derive_status(batch_id)

    completed_runs = sum(1 for r in member_runs if r.status == "COMPLETED")
    failed_runs    = sum(1 for r in member_runs if r.status == "FAILED")
    pending_runs   = sum(1 for r in member_runs if r.status == "PENDING")

    run_responses = [RunStatusResponse.model_validate(r) for r in member_runs]

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        label=batch.label,
        status=status,
        created_at=batch.created_at,
        total_runs=batch.total_runs,
        completed_runs=completed_runs,
        failed_runs=failed_runs,
        pending_runs=pending_runs,
        runs=run_responses,
    )


# ---------------------------------------------------------------------------
# POST /batches
# ---------------------------------------------------------------------------

@router.post("/", response_model=BatchStatusResponse, status_code=201)
def submit_batch(body: SubmitBatchRequest, db: Session = Depends(get_db)) -> BatchStatusResponse:
    """
    Submit a batch of projection runs.

    Validates that every config_json string in the request body is parseable
    JSON, writes a BatchRecord and one PENDING RunRecord per config to the
    database, enqueues all runs, and returns the full batch status immediately.

    The runs are NOT executed here — workers pick them up from the queue.
    Poll GET /batches/{batch_id} to track batch progress, or GET /runs/{run_id}
    for individual run status.

    Returns 400 if any config_json string is not valid JSON (specifying which
    index failed).  Returns 503 if Redis is not reachable.
    """
    # ------------------------------------------------------------------
    # Validate all config strings before writing anything to the DB.
    # This keeps the operation atomic from the user's perspective: either
    # all configs are accepted or none are.
    # ------------------------------------------------------------------
    config_dicts: list[dict] = []
    for idx, config_str in enumerate(body.configs):
        try:
            config_dicts.append(json.loads(config_str))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"configs[{idx}] is not valid JSON: {exc}",
            ) from exc

    batch_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Build RunRecords.
    # ------------------------------------------------------------------
    run_ids: list[str] = []
    run_records: list[RunRecord] = []
    for config_str, config_dict in zip(body.configs, config_dicts):
        run_id = str(uuid.uuid4())
        run_type = str(config_dict.get("run_type", "unknown")).upper()
        run_records.append(
            RunRecord(
                run_id=run_id,
                run_type=run_type,
                status="PENDING",
                created_at=now,
                config_json=config_str,
                batch_id=batch_id,
            )
        )
        run_ids.append(run_id)

    # ------------------------------------------------------------------
    # Persist batch and all run records, then commit once.
    # ------------------------------------------------------------------
    batch_record = BatchRecord(
        batch_id=batch_id,
        label=body.label,
        created_at=now,
        total_runs=len(body.configs),
    )

    batch_repo = BatchRepository(db)
    run_repo   = RunRepository(db)

    batch_repo.save(batch_record)
    for record in run_records:
        run_repo.save(record)

    db.commit()

    # ------------------------------------------------------------------
    # Enqueue all runs.  If Redis is unavailable the records already exist
    # as PENDING — a health check will surface the issue.
    # ------------------------------------------------------------------
    try:
        from worker.job_queue import get_queue
        queue = get_queue()
        for run_id in run_ids:
            queue.enqueue("worker.tasks.run_alm_job", run_id)
    except redis_lib.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Job queue unavailable — Redis is not reachable. "
                "All run records have been saved as PENDING."
            ),
        )

    return _build_batch_response(batch_repo, batch_id)


# ---------------------------------------------------------------------------
# GET /batches
# ---------------------------------------------------------------------------

@router.get("/", response_model=BatchListResponse)
def list_batches(db: Session = Depends(get_db)) -> BatchListResponse:
    """
    Return all batch records ordered by created_at descending (most recent first).

    Each entry in the response includes the full member-run list and derived
    batch status.
    """
    batch_repo = BatchRepository(db)
    batches = batch_repo.list_all()
    batch_responses = [
        _build_batch_response(batch_repo, b.batch_id) for b in batches
    ]
    return BatchListResponse(batches=batch_responses, total=len(batch_responses))


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}
# ---------------------------------------------------------------------------

@router.get("/{batch_id}", response_model=BatchStatusResponse)
def get_batch(batch_id: str, db: Session = Depends(get_db)) -> BatchStatusResponse:
    """
    Return the current status of a single batch and all of its member runs.

    Returns 404 if no batch with the given batch_id exists.
    """
    batch_repo = BatchRepository(db)
    try:
        return _build_batch_response(batch_repo, batch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id!r}")

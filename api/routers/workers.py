"""
/workers router — read-only worker status.

Endpoint
--------
GET /workers
    Return the count and state of all currently registered RQ worker processes.
    Workers self-register in Redis when they start and deregister when they stop,
    so this endpoint reflects the live state at the moment of the request.

Implementation note — Windows compatibility
-------------------------------------------
RQ 2.x imports `multiprocessing.get_context('fork')` at module level in
`rq/scheduler.py`.  Windows only supports the `spawn` start method so
`from rq import Worker` raises ValueError on Windows at import time.

This router therefore queries the RQ worker registry **directly from Redis**
using redis-py, with no import of the `rq` package.  The Redis data layout
used by RQ 2.x is:

    Set  key  "rq:workers"          → set of worker-key strings
                                       e.g. "rq:worker:hostname.12345"
    Hash key  "rq:worker:<name>"    → fields: name, state, queues,
                                       current_job, last_heartbeat, …

Phase 2 scope
-------------
This endpoint is intentionally read-only.  Starting and stopping workers is
handled by the desktop app (Step 14), which spawns and kills local worker
subprocesses directly via Python's subprocess module.
"""
from __future__ import annotations

import redis as redis_lib
from fastapi import APIRouter, HTTPException

from api.schemas.worker_schema import WorkerListResponse, WorkerResponse
from worker.job_queue import get_redis_connection

router = APIRouter()

# Redis keys used by RQ 2.x worker registry.
_WORKERS_SET_KEY   = "rq:workers"
_WORKER_KEY_PREFIX = "rq:worker:"


def _decode(value: bytes | str | None, default: str = "") -> str:
    """Decode a Redis bytes value to str, returning default for None."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@router.get("/", response_model=WorkerListResponse)
def list_workers() -> WorkerListResponse:
    """
    Return status of all currently registered RQ worker processes.

    Workers appear in this list as soon as they start and are removed
    automatically when they stop or their heartbeat expires.

    Returns 503 if Redis is unreachable.
    """
    try:
        conn = get_redis_connection()
        # RQ stores worker key names in a plain Redis set.
        raw_worker_keys: set[bytes] = conn.smembers(_WORKERS_SET_KEY)
    except redis_lib.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Worker registry unavailable — Redis is not reachable.",
        )

    worker_responses: list[WorkerResponse] = []

    for raw_key in raw_worker_keys:
        worker_key = _decode(raw_key)
        if not worker_key:
            continue

        try:
            data: dict[bytes, bytes] = conn.hgetall(worker_key)
        except redis_lib.exceptions.ConnectionError:
            continue  # worker key vanished between SMEMBERS and HGETALL

        if not data:
            continue

        name  = _decode(data.get(b"name"), default=worker_key)
        state = _decode(data.get(b"state"), default="idle")

        # RQ stores the current job id under "current_job" (RQ 2.x).
        current_job_id: str | None = _decode(data.get(b"current_job")) or None

        # Queues are stored as a comma-separated string: "alm,default"
        queues_raw = _decode(data.get(b"queues"), default="")
        queues = [q.strip() for q in queues_raw.split(",") if q.strip()]

        worker_responses.append(
            WorkerResponse(
                name=name,
                state=state,
                current_job_id=current_job_id,
                queues=queues,
            )
        )

    idle_count = sum(1 for w in worker_responses if w.state == "idle")
    busy_count = sum(1 for w in worker_responses if w.state == "busy")

    return WorkerListResponse(
        total_workers=len(worker_responses),
        idle_workers=idle_count,
        busy_workers=busy_count,
        workers=worker_responses,
    )

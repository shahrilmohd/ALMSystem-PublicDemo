"""
Pydantic schemas for the /workers endpoint.

Response schemas only — there is no write endpoint in Phase 2.
Worker processes are managed by the desktop app (Step 14), which spawns
and kills local subprocesses directly.  This endpoint is read-only status.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class WorkerResponse(BaseModel):
    """
    Status snapshot of a single RQ worker process.

    state values (from rq.worker.base.WorkerStatus):
        "idle"      — worker is running but has no job at this moment
        "busy"      — worker is actively executing a job
        "started"   — worker has just started, not yet picked up a job
        "suspended" — worker is paused (e.g. during a deployment)
    """
    name:            str            = Field(..., description="Unique worker identifier (set by RQ on startup).")
    state:           str            = Field(..., description="idle | busy | started | suspended.")
    current_job_id:  Optional[str]  = Field(None, description="run_id of the job currently executing, or None if idle.")
    queues:          list[str]      = Field(..., description="Names of the queues this worker is listening to.")


class WorkerListResponse(BaseModel):
    """Returned by GET /workers."""
    total_workers: int = Field(..., description="Total number of registered worker processes.")
    idle_workers:  int = Field(..., description="Workers with no current job.")
    busy_workers:  int = Field(..., description="Workers actively executing a job.")
    workers:       list[WorkerResponse]

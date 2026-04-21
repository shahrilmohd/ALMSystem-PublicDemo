"""
Progress reporting for long-running ALM projection jobs.

How progress works
------------------
RQ jobs run inside a worker process that has no direct HTTP connection back
to the frontend.  Progress is communicated via RQ's own job metadata store,
which lives in Redis alongside the job ticket.

Each call to report() writes two values into the job's metadata:
    progress   — a float from 0.0 to 1.0
    message    — a short human-readable description of the current stage

The frontend (or API) can read this at any time by calling:
    rq.job.Job.fetch(run_id, connection=redis_conn).meta

Why RQ job metadata rather than a separate Redis key?
    RQ already maintains a metadata dict for every job in Redis.  Writing
    into it keeps all job state in one place and avoids managing a separate
    key namespace.  The metadata is cleaned up automatically when RQ
    expires the job record.

Usage inside tasks.py
---------------------
    from worker.progress import report

    def run_alm_job(run_id: str) -> None:
        report(0.0, "Starting projection")
        ...
        report(0.5, "Halfway through scenarios")
        ...
        report(1.0, "Complete")

Reading progress from outside (e.g. the API or desktop app)
------------------------------------------------------------
    from rq.job import Job
    from worker.job_queue import get_redis_connection

    conn = get_redis_connection()
    job  = Job.fetch(run_id, connection=conn)
    progress = job.meta.get("progress", 0.0)
    message  = job.meta.get("message", "")
"""
from __future__ import annotations

from rq import get_current_job


def report(fraction: float, message: str) -> None:
    """
    Write progress into the current RQ job's metadata.

    Must be called from inside an RQ job function (i.e. inside tasks.py).
    If called outside a job context — e.g. in tests running synchronously
    without RQ — get_current_job() returns None and this is a no-op.

    Parameters
    ----------
    fraction : float
        Progress from 0.0 (not started) to 1.0 (complete).
    message : str
        Short description of the current stage, e.g. "Running scenario 42/100".
    """
    job = get_current_job()
    if job is None:
        return

    job.meta["progress"] = max(0.0, min(1.0, fraction))
    job.meta["message"]  = message
    job.save_meta()

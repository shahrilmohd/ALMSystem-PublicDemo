"""
RQ worker entry point.

Starts a single worker process that listens on the 'alm_jobs' queue and
executes ALM projection jobs dispatched by the API.

Usage
-----
    uv run python -m worker.main
    uv run python -m worker.main --burst   # process all queued jobs then exit

On Windows the RQ SpawnWorker is used (fork is unavailable).
On Linux/Mac the standard Worker is used.

Prerequisites
-------------
    Redis must be running on localhost:6379 (or ALM_REDIS_URL must be set).
    The FastAPI server must be running to receive run submissions.
"""
from __future__ import annotations

# job_queue applies the Windows multiprocessing patch before any rq import.
from worker.job_queue import _QUEUE_NAME, get_redis_connection  # noqa: F401 — patch applied on import

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ALM RQ Worker")
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Exit after all currently queued jobs have been processed.",
    )
    args = parser.parse_args()

    conn = get_redis_connection()

    if sys.platform == "win32":
        # SpawnWorker is the intended Windows worker but it uses os.wait4
        # internally (Unix-only) and crashes when monitoring the work horse.
        # SimpleWorker runs each job in the same process — no fork/spawn, no
        # os.wait4 — which is the only reliable option on Windows with RQ 2.x.
        from rq import SimpleWorker
        worker = SimpleWorker([_QUEUE_NAME], connection=conn)
    else:
        from rq import Worker
        worker = Worker([_QUEUE_NAME], connection=conn)

    print(f"Worker starting — queue: {_QUEUE_NAME!r}  burst={args.burst}")
    worker.work(burst=args.burst)


if __name__ == "__main__":
    main()

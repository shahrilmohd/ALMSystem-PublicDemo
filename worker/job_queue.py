"""
Redis connection and RQ queue setup.

Responsibilities
----------------
- Build the Redis connection from environment config.
- Expose get_queue() — the single entry point for enqueuing jobs.
- Expose get_redis_connection() — used by progress.py and tests.

Redis URL priority
------------------
1. ALM_REDIS_URL environment variable  (set this in production / Docker)
2. Default: redis://localhost:6379/0   (standard local Redis install)

Why functions rather than module-level objects?
    Instantiating Redis at import time would crash the API process if Redis
    is not running — even when the API is only serving GET requests that
    never touch the queue.  A function defers the connection attempt until
    the first actual enqueue or progress check, keeping the API resilient
    to a temporarily unavailable Redis.

    In plain terms: a module-level object dials the phone the moment you
    pick up the phone book.  A function only dials when you actually want
    to make a call.

Redis URL anatomy
-----------------
    redis://localhost:6379/0
    │        │         │   └── database slot (Redis has 16 slots; we use slot 0)
    │        │         └─────── port (Redis default)
    │        └───────────────── host (localhost = same machine)
    └────────────────────────── protocol
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Windows compatibility — must run before any `rq` import anywhere in the
# process.  rq/scheduler.py does `get_context('fork')` at import time, which
# raises ValueError on Windows (only 'spawn' is supported).  Redirecting
# 'fork' to 'spawn' lets the import succeed; plain Queue.enqueue() calls
# never touch the scheduler, so the substitution is harmless for our use.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import multiprocessing as _mp
    _orig_get_context = _mp.get_context

    def _win_get_context(method=None):
        return _orig_get_context("spawn" if method == "fork" else method)

    _mp.get_context = _win_get_context

import redis

_REDIS_URL  = os.getenv("ALM_REDIS_URL", "redis://localhost:6379/0")
_QUEUE_NAME = "alm_jobs"


def get_redis_connection() -> redis.Redis:
    """
    Return a Redis client connected to the configured URL.

    Raises redis.exceptions.ConnectionError if Redis is unreachable.
    The caller is responsible for handling this — the API catches it and
    returns 503 Service Unavailable rather than a 500 Internal Server Error.
    """
    return redis.Redis.from_url(_REDIS_URL)


def get_queue(*, is_async: bool = True):
    """
    Return the RQ Queue used for all ALM projection jobs.

    All jobs are placed on a single queue named 'alm_jobs'.  A single queue
    is sufficient for Phase 2 — the team is small and runs are sequential.
    Multiple queues (e.g. fast / slow) can be introduced later if needed.

    Parameters
    ----------
    is_async : bool
        True  (default) — jobs are executed by a separate worker process.
        False           — jobs execute synchronously in the calling process.
                          Used in tests to run jobs without a real Redis instance
                          or a running worker.
    """
    from rq import Queue  # deferred — rq crashes on Windows at module level (scheduler.py fork context)
    conn = get_redis_connection()
    return Queue(_QUEUE_NAME, connection=conn, is_async=is_async)

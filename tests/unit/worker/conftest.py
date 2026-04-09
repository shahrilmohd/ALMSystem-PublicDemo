"""
Shared fixtures for worker unit tests.

fakeredis
---------
Tests must not require a real Redis instance.  fakeredis is an in-process
Redis implementation that behaves identically to redis-py without any network
connection.  All RQ operations (enqueue, job metadata, job fetch) work
against the fake server transparently.

db_session
----------
The same in-memory SQLite pattern used in the API tests, re-used here so
run_alm_job() can read and write RunRecords without touching the filesystem.
"""
from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

import fakeredis
import pytest
from rq import Queue
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from storage.db import create_all_tables, get_engine, get_session_factory
from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository


@pytest.fixture
def fake_redis():
    """An in-process fakeredis server — no real Redis needed."""
    return fakeredis.FakeRedis()


@pytest.fixture
def fake_queue(fake_redis) -> Queue:
    """
    An RQ Queue backed by fakeredis.

    is_async=False means jobs execute synchronously in the test process
    when queue.enqueue() is called — no worker subprocess needed.
    """
    return Queue("alm_jobs", connection=fake_redis, is_async=False)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Fresh in-memory SQLite session for each test."""
    engine = get_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        yield session


def make_pending_run(
    db: Session,
    run_id: str = "run-001",
    run_type: str = "LIABILITY_ONLY",
    config_json: str = '{"run_type": "liability_only"}',
) -> RunRecord:
    """Insert a PENDING RunRecord and return it."""
    record = RunRecord(
        run_id=run_id,
        run_type=run_type,
        status="PENDING",
        created_at=datetime.now(timezone.utc),
        config_json=config_json,
    )
    RunRepository(db).save(record)
    db.commit()
    return record

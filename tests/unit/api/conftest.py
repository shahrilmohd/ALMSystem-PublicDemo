"""
Shared fixtures for API unit tests.

Why StaticPool + check_same_thread=False?
-----------------------------------------
Two issues arise when testing FastAPI with in-memory SQLite:

1. Thread isolation (SingletonThreadPool default):
   SQLAlchemy's default pool for in-memory SQLite is SingletonThreadPool,
   which gives EACH THREAD its own connection — meaning its own empty
   database.  FastAPI's TestClient runs synchronous endpoints in a worker
   thread (via anyio.to_thread.run_sync).  Without StaticPool, the worker
   thread connects to a completely different in-memory DB and sees no data.

   Fix: StaticPool — all sessions share ONE underlying connection, so the
   in-memory database is the same object regardless of which thread connects.

2. SQLite thread check:
   Python's sqlite3 module refuses by default to let a connection created
   in one thread be used in another.  Even with StaticPool the connection
   is created in the fixture thread but used in the worker thread.

   Fix: connect_args={"check_same_thread": False} — disables this check.

Fixture layout:
    _test_engine  ← in-memory SQLite engine, scoped per test
    db_session    ← Session for direct test-level DB operations (pre-populate)
    client        ← TestClient wired to the same engine via get_db override
"""
from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.dependencies import get_db
from api.main import app
from storage.db import create_all_tables, get_engine, get_session_factory
from storage.models.run_record import RunRecord
from storage.run_repository import RunRepository


@pytest.fixture
def _test_engine():
    """
    A fresh in-memory SQLite engine for each test.

    StaticPool + check_same_thread=False ensures that every session —
    whether created in the test thread or the FastAPI worker thread —
    connects to the same in-memory database.
    """
    engine = get_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)
    return engine


@pytest.fixture
def db_session(_test_engine) -> Generator[Session, None, None]:
    """
    A Session for direct test-level DB operations (pre-populating records).

    Uses the same engine as the TestClient so data committed here is
    immediately visible to endpoint sessions.
    """
    SessionFactory = get_session_factory(_test_engine)
    with SessionFactory() as session:
        yield session


@pytest.fixture
def client(_test_engine, db_session) -> Generator[TestClient, None, None]:
    """
    TestClient with get_db overridden to use the shared test engine.

    Each HTTP request gets a fresh Session from the same engine, so it
    sees all data committed by db_session during the test.
    """
    TestSessionFactory = get_session_factory(_test_engine)

    def override_get_db():
        with TestSessionFactory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_job_queue():
    """
    Patch get_queue for all API unit tests — no real Redis required.

    The runs router imports get_queue() lazily (inside the POST handler).
    We patch the function in its source module so all calls within the
    request are intercepted regardless of when the import occurs.
    The mock queue's enqueue() is a no-op, which is all the API tests need.
    """
    mock_queue = MagicMock()
    with patch("worker.job_queue.get_queue", return_value=mock_queue):
        yield mock_queue


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_run_record(
    db: Session,
    run_id: str = "run-001",
    run_type: str = "DETERMINISTIC",
    status: str = "COMPLETED",
    config_json: str = '{"run_type": "deterministic"}',
) -> RunRecord:
    """Insert a RunRecord into the test DB and return it."""
    record = RunRecord(
        run_id=run_id,
        run_type=run_type,
        status=status,
        created_at=datetime.now(timezone.utc),
        config_json=config_json,
        n_timesteps=12,
    )
    repo = RunRepository(db)
    repo.save(record)
    db.commit()
    return record

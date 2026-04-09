"""
Shared fixtures for storage unit tests.

All tests use an in-memory SQLite database — fast, isolated, and torn down
automatically after each test via the function-scoped session fixture.
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from storage.db import create_all_tables, get_engine, get_session_factory


@pytest.fixture
def session() -> Session:
    """
    Provide a fresh in-memory SQLite session for each test.

    Tables are created before the test and the session is closed after.
    Each test gets a completely isolated database — no shared state.
    """
    engine = get_engine("sqlite:///:memory:")
    create_all_tables(engine)
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as s:
        yield s

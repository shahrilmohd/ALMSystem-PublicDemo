"""
API dependency injection.

All shared resources that endpoint functions need — database sessions,
configuration — are defined here as FastAPI dependency functions.

Why dependency injection?
    Each endpoint that needs a DB session declares it as a parameter.
    FastAPI calls get_db() automatically and injects the result.
    This ensures:
      - Sessions are always closed after each request, even on error.
      - Tests can swap get_db() for an in-memory SQLite session without
        touching any router code (via app.dependency_overrides).
      - No boilerplate session management scattered across endpoint files.

DB URL priority:
    1. ALM_DB_URL environment variable (set this in production / Docker).
    2. Default: "sqlite:///alm.db" — file in the working directory.
"""
from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy.orm import Session

from storage.db import get_engine, get_session_factory

_DB_URL = os.getenv("ALM_DB_URL", "sqlite:///alm.db")
_engine = get_engine(_DB_URL)
_SessionFactory = get_session_factory(_engine)


def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session for the current request.

    Used as a FastAPI dependency:

        @router.get("/")
        def my_endpoint(db: Session = Depends(get_db)):
            ...

    The session is closed when the request finishes, whether the handler
    succeeded or raised an exception.  Commit is the caller's responsibility
    — write endpoints must call session.commit() explicitly.
    """
    with _SessionFactory() as session:
        yield session

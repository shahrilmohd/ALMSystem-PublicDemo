"""
Database engine and session factory.

Usage
-----
    from storage.db import get_engine, get_session_factory, create_all_tables

    engine  = get_engine()                  # default: SQLite file "alm.db"
    Session = get_session_factory(engine)

    with Session() as session:
        run_repo = RunRepository(session)
        run_repo.save(record)
        session.commit()

Switching to PostgreSQL (production)
-------------------------------------
Change the `url` parameter in `get_engine()` — everything else is unchanged:

    engine = get_engine("postgresql://user:pass@host/dbname")

No other file references the database URL (DECISIONS.md §29).
"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from storage.models.run_record import Base  # Base is shared by all ORM models
# Import all ORM models so Base.metadata knows about their tables before create_all_tables().
import storage.models.result_record   # noqa: F401
import storage.models.batch_record    # noqa: F401
import storage.models.ifrs17_record   # noqa: F401


_DEFAULT_URL = "sqlite:///alm.db"


def get_engine(url: str = _DEFAULT_URL, **kwargs) -> Engine:
    """
    Create a SQLAlchemy engine.

    Parameters
    ----------
    url : str
        SQLAlchemy connection string.
        Default: "sqlite:///alm.db" (file in the working directory).
        For in-memory SQLite (testing): "sqlite:///:memory:".
        For PostgreSQL: "postgresql://user:pass@host/dbname".
    **kwargs
        Passed directly to `create_engine()`.  Useful overrides:
            echo=True      — log all SQL statements (debugging).
            pool_size=N    — connection pool size (PostgreSQL).
    """
    return create_engine(url, **kwargs)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    Return a session factory bound to the given engine.

    The caller owns session lifecycle (open/commit/rollback/close).
    Use as a context manager:

        Session = get_session_factory(engine)
        with Session() as session:
            ...
            session.commit()
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_all_tables(engine: Engine) -> None:
    """
    Create all ORM-defined tables if they do not already exist.

    Safe to call multiple times (CREATE TABLE IF NOT EXISTS semantics).
    Call once at application startup before any repository operations.
    """
    Base.metadata.create_all(engine)

"""
RunRepository — persistence for projection run records.

Architectural rule (DECISIONS.md §29):
    Nothing outside storage/ calls SQLAlchemy directly.
    All run record access goes through this class.

Usage
-----
    with Session() as session:
        repo = RunRepository(session)
        repo.save(RunRecord(run_id="r1", run_type="DETERMINISTIC",
                            status="PENDING", created_at=datetime.utcnow(),
                            config_json="{}"))
        session.commit()

        record = repo.get("r1")
        repo.update_status("r1", "COMPLETED",
                           completed_at=datetime.utcnow(),
                           duration_seconds=12.3)
        session.commit()
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models.run_record import RunRecord


class RunRepository:
    """
    CRUD operations for RunRecord (run_records table).

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Active database session.  The caller owns the session lifecycle
        (commit, rollback, close).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, record: RunRecord) -> None:
        """
        Insert or update a RunRecord.

        Uses merge() so the call is idempotent: if a record with the same
        run_id already exists it is updated; otherwise a new row is inserted.
        The caller must call session.commit() to persist the change.
        """
        self._session.merge(record)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunRecord:
        """
        Retrieve a run by its run_id.

        Raises
        ------
        KeyError
            If no run with the given run_id exists.
        """
        record = self._session.get(RunRecord, run_id)
        if record is None:
            raise KeyError(f"No run found with run_id={run_id!r}.")
        return record

    def list_all(self) -> list[RunRecord]:
        """
        All run records ordered by created_at descending (most recent first).
        """
        stmt = select(RunRecord).order_by(RunRecord.created_at.desc())
        return list(self._session.scalars(stmt))

    def exists(self, run_id: str) -> bool:
        """Return True if a run with the given run_id exists."""
        return self._session.get(RunRecord, run_id) is not None

    # ------------------------------------------------------------------
    # Partial update
    # ------------------------------------------------------------------

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the status (and optional timing/error fields) of an existing run.

        Only non-None keyword arguments are written — other fields are left
        unchanged.  The caller must call session.commit() to persist.

        Raises
        ------
        KeyError
            If no run with the given run_id exists.
        """
        record = self.get(run_id)
        record.status = status
        if started_at is not None:
            record.started_at = started_at
        if completed_at is not None:
            record.completed_at = completed_at
        if duration_seconds is not None:
            record.duration_seconds = duration_seconds
        if error_message is not None:
            record.error_message = error_message

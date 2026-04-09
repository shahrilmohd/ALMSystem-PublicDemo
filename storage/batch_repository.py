"""
BatchRepository — persistence for batch records.

Architectural rule (DECISIONS.md §29):
    Nothing outside storage/ calls SQLAlchemy directly.
    All batch record access goes through this class.

Usage
-----
    with Session() as session:
        repo = BatchRepository(session)
        repo.save(BatchRecord(batch_id="b1", created_at=datetime.utcnow(),
                              total_runs=3))
        session.commit()

        batch = repo.get("b1")
        runs  = repo.get_member_runs("b1")
        status = repo.derive_status("b1")
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models.batch_record import BatchRecord
from storage.models.run_record import RunRecord


class BatchRepository:
    """
    CRUD operations for BatchRecord (batch_records table).

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

    def save(self, batch: BatchRecord) -> None:
        """
        Insert or update a BatchRecord.

        Uses merge() so the call is idempotent: if a record with the same
        batch_id already exists it is updated; otherwise a new row is inserted.
        The caller must call session.commit() to persist the change.
        """
        self._session.merge(batch)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, batch_id: str) -> BatchRecord:
        """
        Retrieve a batch by its batch_id.

        Raises
        ------
        KeyError
            If no batch with the given batch_id exists.
        """
        record = self._session.get(BatchRecord, batch_id)
        if record is None:
            raise KeyError(f"No batch found with batch_id={batch_id!r}.")
        return record

    def list_all(self) -> list[BatchRecord]:
        """
        All batch records ordered by created_at descending (most recent first).
        """
        stmt = select(BatchRecord).order_by(BatchRecord.created_at.desc())
        return list(self._session.scalars(stmt))

    def get_member_runs(self, batch_id: str) -> list[RunRecord]:
        """
        Return all RunRecords that belong to the given batch.

        Results are ordered by created_at ascending so the caller sees runs
        in submission order.

        Parameters
        ----------
        batch_id : str
            UUID of the parent batch.
        """
        stmt = (
            select(RunRecord)
            .where(RunRecord.batch_id == batch_id)
            .order_by(RunRecord.created_at.asc())
        )
        return list(self._session.scalars(stmt))

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def derive_status(self, batch_id: str) -> str:
        """
        Compute overall batch status from the statuses of member runs.

        Rules (applied in priority order):
            1. Any run FAILED     → "FAILED"
            2. Any run RUNNING    → "RUNNING"
            3. All runs COMPLETED → "COMPLETED"
            4. Otherwise          → "PENDING"

        Parameters
        ----------
        batch_id : str
            UUID of the batch to inspect.
        """
        runs = self.get_member_runs(batch_id)
        if not runs:
            return "PENDING"

        statuses = {r.status for r in runs}

        if "FAILED" in statuses:
            return "FAILED"
        if "RUNNING" in statuses:
            return "RUNNING"
        if all(r.status == "COMPLETED" for r in runs):
            return "COMPLETED"
        return "PENDING"

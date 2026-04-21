"""
BatchRecord — SQLAlchemy ORM model for a batch of projection runs.

A batch groups multiple RunRecords submitted together for parallel execution.
Each member RunRecord carries the batch_id as a foreign key (batch_id column
on run_records).  Batch status is derived at query time from the member runs
rather than stored, which avoids synchronisation complexity.

Fields
------
batch_id   : VARCHAR PK   — UUID string.
label      : TEXT NULL    — Optional user-provided name (e.g. "Q4 2025 BEL runs").
created_at : DATETIME     — When the batch was submitted.
total_runs : INTEGER      — Number of runs in the batch (fixed at submission time).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.run_record import Base


class BatchRecord(Base):
    """ORM model for the batch_records table."""

    __tablename__ = "batch_records"

    batch_id:   Mapped[str]       = mapped_column(String,  primary_key=True)
    label:      Mapped[str | None] = mapped_column(Text,    nullable=True)
    created_at: Mapped[datetime]  = mapped_column(DateTime, nullable=False)
    total_runs: Mapped[int]       = mapped_column(Integer,  nullable=False)

    def __repr__(self) -> str:
        return (
            f"BatchRecord(batch_id={self.batch_id!r}, label={self.label!r}, "
            f"total_runs={self.total_runs!r})"
        )

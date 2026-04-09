"""
RunRecord — SQLAlchemy ORM model for a projection run.

Maps to the `run_records` table (DECISIONS.md §29).

Fields
------
run_id          : VARCHAR PK   — UUID string from the run mode orchestrator.
run_name        : VARCHAR NULL — Human-readable label from the submitted RunConfig.
run_type        : VARCHAR      — "LIABILITY_ONLY", "DETERMINISTIC", "STOCHASTIC".
status          : VARCHAR      — "PENDING", "RUNNING", "COMPLETED", "FAILED".
created_at      : DATETIME     — Populated on insert.
started_at      : DATETIME NULL — Set when execution begins.
completed_at    : DATETIME NULL — Set on COMPLETED or FAILED.
duration_seconds: FLOAT NULL   — Wall-clock seconds for the full run.
error_message   : TEXT NULL    — First-line error string on FAILED runs.
config_json     : TEXT         — Full RunConfig as JSON (Pydantic model_dump_json()).
n_scenarios     : INTEGER NULL — ESG scenario count (STOCHASTIC only; NULL otherwise).
n_timesteps     : INTEGER NULL — Monthly projection steps.
batch_id        : VARCHAR NULL — UUID of the parent BatchRecord (NULL for standalone runs).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    """ORM model for the run_records table."""

    __tablename__ = "run_records"

    run_id:           Mapped[str]             = mapped_column(String,  primary_key=True)
    run_name:         Mapped[str | None]      = mapped_column(String,  nullable=True)
    run_type:         Mapped[str]             = mapped_column(String,  nullable=False)
    status:           Mapped[str]             = mapped_column(String,  nullable=False)
    created_at:       Mapped[datetime]        = mapped_column(DateTime, nullable=False)
    started_at:       Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at:     Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None]    = mapped_column(Float,   nullable=True)
    error_message:    Mapped[str | None]      = mapped_column(Text,    nullable=True)
    config_json:      Mapped[str]             = mapped_column(Text,    nullable=False)
    batch_id:         Mapped[str | None]      = mapped_column(String,  nullable=True, index=True)
    n_scenarios:      Mapped[int | None]      = mapped_column(Integer, nullable=True)
    n_timesteps:      Mapped[int | None]      = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"RunRecord(run_id={self.run_id!r}, run_type={self.run_type!r}, "
            f"status={self.status!r})"
        )

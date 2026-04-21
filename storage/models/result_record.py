"""
ResultRecord — SQLAlchemy ORM model for a single projection timestep result.

Maps to the `result_records` table (DECISIONS.md §29).

One row per (run_id, scenario_id, timestep, cohort_id).  This mirrors the
schema of ResultStore.as_dataframe() / RESULT_COLUMNS exactly.

All liability fields are always populated.
Asset fields (total_market_value … mv_fvoci) are NULL for LIABILITY_ONLY runs.
cohort_id is NULL for all non-BPA runs; populated in Phase 3 (DECISIONS.md §17).
"""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.run_record import Base


class ResultRecord(Base):
    """ORM model for the result_records table."""

    __tablename__ = "result_records"

    # ------------------------------------------------------------------
    # Primary key — surrogate integer, never exposed outside storage/
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ------------------------------------------------------------------
    # Index dimensions
    # ------------------------------------------------------------------
    run_id:      Mapped[str]      = mapped_column(String,  ForeignKey("run_records.run_id"), nullable=False)
    scenario_id: Mapped[int]      = mapped_column(Integer, nullable=False)
    timestep:    Mapped[int]      = mapped_column(Integer, nullable=False)
    cohort_id:   Mapped[str | None] = mapped_column(String, nullable=True)

    # ------------------------------------------------------------------
    # Liability cash flows (always populated)
    # ------------------------------------------------------------------
    premiums:           Mapped[float] = mapped_column(Float, nullable=False)
    death_claims:       Mapped[float] = mapped_column(Float, nullable=False)
    surrender_payments: Mapped[float] = mapped_column(Float, nullable=False)
    maturity_payments:  Mapped[float] = mapped_column(Float, nullable=False)
    expenses:           Mapped[float] = mapped_column(Float, nullable=False)
    net_outgo:          Mapped[float] = mapped_column(Float, nullable=False)

    # ------------------------------------------------------------------
    # Decrements (always populated)
    # ------------------------------------------------------------------
    in_force_start: Mapped[float] = mapped_column(Float, nullable=False)
    deaths:         Mapped[float] = mapped_column(Float, nullable=False)
    lapses:         Mapped[float] = mapped_column(Float, nullable=False)
    maturities:     Mapped[float] = mapped_column(Float, nullable=False)
    in_force_end:   Mapped[float] = mapped_column(Float, nullable=False)

    # ------------------------------------------------------------------
    # Valuation (always populated)
    # ------------------------------------------------------------------
    bel:     Mapped[float] = mapped_column(Float, nullable=False)
    reserve: Mapped[float] = mapped_column(Float, nullable=False)

    # ------------------------------------------------------------------
    # Asset fields (NULL for LIABILITY_ONLY runs)
    # ------------------------------------------------------------------
    total_market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_book_value:   Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_balance:       Mapped[float | None] = mapped_column(Float, nullable=True)
    eir_income:         Mapped[float | None] = mapped_column(Float, nullable=True)
    coupon_income:      Mapped[float | None] = mapped_column(Float, nullable=True)
    dividend_income:    Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealised_gl:      Mapped[float | None] = mapped_column(Float, nullable=True)
    realised_gl:        Mapped[float | None] = mapped_column(Float, nullable=True)
    oci_reserve:        Mapped[float | None] = mapped_column(Float, nullable=True)
    mv_ac:              Mapped[float | None] = mapped_column(Float, nullable=True)
    mv_fvtpl:           Mapped[float | None] = mapped_column(Float, nullable=True)
    mv_fvoci:           Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------
    # BPA MA attribution (NULL for all non-BPA runs — DECISIONS.md §21)
    # ------------------------------------------------------------------
    bel_pre_ma:  Mapped[float | None] = mapped_column(Float, nullable=True)
    bel_post_ma: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------
    # Constraints and indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        # Mirrors the ResultStore duplicate-key rule
        UniqueConstraint("run_id", "scenario_id", "timestep", "cohort_id",
                         name="uq_result_key"),
        # Fast retrieval by run + scenario
        Index("ix_result_run_scenario", "run_id", "scenario_id"),
    )

    def __repr__(self) -> str:
        return (
            f"ResultRecord(run_id={self.run_id!r}, scenario_id={self.scenario_id}, "
            f"timestep={self.timestep}, cohort_id={self.cohort_id!r})"
        )

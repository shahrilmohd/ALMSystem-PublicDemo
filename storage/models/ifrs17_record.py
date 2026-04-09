"""
storage/models/ifrs17_record.py — ORM (Object-Relational Mapping) models for IFRS 17 persistent state.

Two tables (DECISIONS.md §37):

  ifrs17_state     — one row per (cohort_id, valuation_date).
                     Cross-period rolling balances. Written at run end;
                     read at next run start as opening state.

  ifrs17_movements — one row per (cohort_id, valuation_date, period_index).
                     Full movement attribution for each projection period.
                     Required for IFRS 17 note disclosures and audit.
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from storage.models.run_record import Base


class Ifrs17StateRecord(Base):
    """
    Maps to the `ifrs17_state` table.

    Primary key: (cohort_id, valuation_date).
    A rerun overwrites the existing row (upsert) — latest state wins.
    """

    __tablename__ = "ifrs17_state"

    # Composite primary key
    cohort_id:      Mapped[str]  = mapped_column(String, primary_key=True)
    valuation_date: Mapped[date] = mapped_column(Date,   primary_key=True)

    # Rolling balances
    csm_balance:               Mapped[float] = mapped_column(Float, nullable=False)
    loss_component:            Mapped[float] = mapped_column(Float, nullable=False)
    remaining_coverage_units:  Mapped[float] = mapped_column(Float, nullable=False)
    total_coverage_units:      Mapped[float] = mapped_column(Float, nullable=False)
    locked_in_rate:            Mapped[float] = mapped_column(Float, nullable=False)
    inception_date:            Mapped[date]  = mapped_column(Date,  nullable=False)


class Ifrs17MovementRecord(Base):
    """
    Maps to the `ifrs17_movements` table.

    One row per (cohort_id, valuation_date, period_index).
    Unique constraint enforces no duplicates; reruns overwrite via upsert.
    """

    __tablename__ = "ifrs17_movements"
    __table_args__ = (
        UniqueConstraint(
            "cohort_id", "valuation_date", "period_index",
            name="uq_ifrs17_movements",
        ),
    )

    id:             Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    cohort_id:      Mapped[str]  = mapped_column(String,  nullable=False, index=True)
    valuation_date: Mapped[date] = mapped_column(Date,    nullable=False, index=True)
    period_index:   Mapped[int]  = mapped_column(Integer, nullable=False)

    # CSM movements
    csm_opening:                   Mapped[float] = mapped_column(Float, nullable=False)
    csm_accretion:                 Mapped[float] = mapped_column(Float, nullable=False)
    csm_adjustment_non_financial:  Mapped[float] = mapped_column(Float, nullable=False)
    csm_release:                   Mapped[float] = mapped_column(Float, nullable=False)
    csm_closing:                   Mapped[float] = mapped_column(Float, nullable=False)

    # Loss component
    loss_component_opening:  Mapped[float] = mapped_column(Float, nullable=False)
    loss_component_addition: Mapped[float] = mapped_column(Float, nullable=False)
    loss_component_release:  Mapped[float] = mapped_column(Float, nullable=False)
    loss_component_closing:  Mapped[float] = mapped_column(Float, nullable=False)

    # Finance income/expense
    insurance_finance_pl:  Mapped[float] = mapped_column(Float, nullable=False)
    insurance_finance_oci: Mapped[float] = mapped_column(Float, nullable=False)

    # Balance sheet
    lrc:            Mapped[float] = mapped_column(Float, nullable=False)
    lic:            Mapped[float] = mapped_column(Float, nullable=False)
    bel_current:    Mapped[float] = mapped_column(Float, nullable=False)
    bel_locked:     Mapped[float] = mapped_column(Float, nullable=False)
    risk_adjustment:Mapped[float] = mapped_column(Float, nullable=False)

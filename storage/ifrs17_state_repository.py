"""
storage/ifrs17_state_repository.py — Ifrs17StateStore

Persistence for IFRS 17 rolling state and period movements.

Architectural rules (DECISIONS.md §37):
    - ifrs17_state    : one row per (cohort_id, valuation_date); upsert on rerun.
    - ifrs17_movements: one row per (cohort_id, valuation_date, period_index); upsert on rerun.
    - All DB access goes through this class only — nothing calls SQLAlchemy directly.

Usage
-----
    with Session() as session:
        store = Ifrs17StateStore(session)
        store.save_state(closing_state)
        store.save_movements(cohort_id, valuation_date, step_results)
        session.commit()

        state = store.load_state("pensioner", date(2024, 12, 31))
        movements = store.get_movements("pensioner", date(2024, 12, 31))
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from engine.ifrs17.gmm import GmmStepResult
from engine.ifrs17.state import Ifrs17State
from storage.models.ifrs17_record import Ifrs17MovementRecord, Ifrs17StateRecord


class Ifrs17StateStore:
    """
    CRUD operations for IFRS 17 state and movements tables.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Active database session.  The caller owns the session lifecycle
        (commit, rollback, close).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # State (ifrs17_state table)
    # ------------------------------------------------------------------

    def load_state(self, cohort_id: str, valuation_date: date) -> Optional[Ifrs17State]:
        """
        Return the Ifrs17State for (cohort_id, valuation_date), or None if absent.

        Returns None for first-run inception where no prior state exists.
        """
        stmt = select(Ifrs17StateRecord).where(
            Ifrs17StateRecord.cohort_id == cohort_id,
            Ifrs17StateRecord.valuation_date == valuation_date,
        )
        record = self._session.execute(stmt).scalar_one_or_none()
        if record is None:
            return None
        return Ifrs17State(
            cohort_id                = record.cohort_id,
            valuation_date           = record.valuation_date,
            csm_balance              = record.csm_balance,
            loss_component           = record.loss_component,
            remaining_coverage_units = record.remaining_coverage_units,
            total_coverage_units     = record.total_coverage_units,
            locked_in_rate           = record.locked_in_rate,
            inception_date           = record.inception_date,
        )

    def save_state(self, state: Ifrs17State) -> None:
        """
        Upsert an Ifrs17State row — insert or replace on (cohort_id, valuation_date).

        The caller must call session.commit() to persist the change.
        """
        stmt = (
            sqlite_insert(Ifrs17StateRecord)
            .values(
                cohort_id                = state.cohort_id,
                valuation_date           = state.valuation_date,
                csm_balance              = state.csm_balance,
                loss_component           = state.loss_component,
                remaining_coverage_units = state.remaining_coverage_units,
                total_coverage_units     = state.total_coverage_units,
                locked_in_rate           = state.locked_in_rate,
                inception_date           = state.inception_date,
            )
            .on_conflict_do_update(
                index_elements=["cohort_id", "valuation_date"],
                set_={
                    "csm_balance":              state.csm_balance,
                    "loss_component":           state.loss_component,
                    "remaining_coverage_units": state.remaining_coverage_units,
                    "total_coverage_units":     state.total_coverage_units,
                    "locked_in_rate":           state.locked_in_rate,
                    "inception_date":           state.inception_date,
                },
            )
        )
        self._session.execute(stmt)

    # ------------------------------------------------------------------
    # Movements (ifrs17_movements table)
    # ------------------------------------------------------------------

    def save_movements(
        self,
        cohort_id:      str,
        valuation_date: date,
        movements:      list[GmmStepResult],
    ) -> None:
        """
        Upsert all period movements for (cohort_id, valuation_date).

        Reruns overwrite existing rows via the unique constraint on
        (cohort_id, valuation_date, period_index).

        The caller must call session.commit() to persist the change.
        """
        for period_index, r in enumerate(movements):
            stmt = (
                sqlite_insert(Ifrs17MovementRecord)
                .values(
                    cohort_id                      = cohort_id,
                    valuation_date                 = valuation_date,
                    period_index                   = period_index,
                    csm_opening                    = r.csm_opening,
                    csm_accretion                  = r.csm_accretion,
                    csm_adjustment_non_financial   = r.csm_adjustment_non_financial,
                    csm_release                    = r.csm_release,
                    csm_closing                    = r.csm_closing,
                    loss_component_opening         = r.loss_component_opening,
                    loss_component_addition        = r.loss_component_addition,
                    loss_component_release         = r.loss_component_release,
                    loss_component_closing         = r.loss_component_closing,
                    insurance_finance_pl           = r.insurance_finance_pl,
                    insurance_finance_oci          = r.insurance_finance_oci,
                    lrc                            = r.lrc,
                    lic                            = r.lic,
                    bel_current                    = r.bel_current,
                    bel_locked                     = r.bel_locked,
                    risk_adjustment                = r.risk_adjustment,
                )
                .on_conflict_do_update(
                    index_elements=["cohort_id", "valuation_date", "period_index"],
                    set_={
                        "csm_opening":                  r.csm_opening,
                        "csm_accretion":                r.csm_accretion,
                        "csm_adjustment_non_financial": r.csm_adjustment_non_financial,
                        "csm_release":                  r.csm_release,
                        "csm_closing":                  r.csm_closing,
                        "loss_component_opening":       r.loss_component_opening,
                        "loss_component_addition":      r.loss_component_addition,
                        "loss_component_release":       r.loss_component_release,
                        "loss_component_closing":       r.loss_component_closing,
                        "insurance_finance_pl":         r.insurance_finance_pl,
                        "insurance_finance_oci":        r.insurance_finance_oci,
                        "lrc":                          r.lrc,
                        "lic":                          r.lic,
                        "bel_current":                  r.bel_current,
                        "bel_locked":                   r.bel_locked,
                        "risk_adjustment":              r.risk_adjustment,
                    },
                )
            )
            self._session.execute(stmt)

    def get_movements(
        self,
        cohort_id:      str,
        valuation_date: date,
    ) -> list[GmmStepResult]:
        """
        Return all period movements for (cohort_id, valuation_date) in period order.

        Returns an empty list if no movements exist for this key.
        """
        stmt = (
            select(Ifrs17MovementRecord)
            .where(
                Ifrs17MovementRecord.cohort_id == cohort_id,
                Ifrs17MovementRecord.valuation_date == valuation_date,
            )
            .order_by(Ifrs17MovementRecord.period_index)
        )
        records = self._session.execute(stmt).scalars().all()
        return [
            GmmStepResult(
                csm_opening                  = rec.csm_opening,
                csm_accretion                = rec.csm_accretion,
                csm_adjustment_non_financial = rec.csm_adjustment_non_financial,
                csm_release                  = rec.csm_release,
                csm_closing                  = rec.csm_closing,
                loss_component_opening       = rec.loss_component_opening,
                loss_component_addition      = rec.loss_component_addition,
                loss_component_release       = rec.loss_component_release,
                loss_component_closing       = rec.loss_component_closing,
                insurance_finance_pl         = rec.insurance_finance_pl,
                insurance_finance_oci        = rec.insurance_finance_oci,
                lrc                          = rec.lrc,
                lic                          = rec.lic,
                p_and_l_csm_release          = rec.csm_release,
                p_and_l_loss_component       = rec.loss_component_release - rec.loss_component_addition,
                p_and_l_insurance_finance    = rec.insurance_finance_pl,
                bel_current                  = rec.bel_current,
                bel_locked                   = rec.bel_locked,
                risk_adjustment              = rec.risk_adjustment,
            )
            for rec in records
        ]

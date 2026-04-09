"""
DeferredLiability — BPA deferred member liability class.

Design (DECISIONS.md §25)
--------------------------
Models deferred pension scheme members who have not yet retired.
Four competing decrements operate simultaneously each period:

  1. Death (pre-retirement)   → death_claims
  2. Ill-health retirement    → early retirement at the current pension rate
  3. Retirement (age-driven)  → conversion to in-payment annuity
  4. Transfer Value (TV)      → member exits; lump sum outgo

BEL uses a two-phase forward projection:
  Phase 1 — project surviving deferred population forward
  Phase 2 — for each retirement/TV event, accumulate the PV contribution

The in-payment annuity BEL at retirement delegates to InPaymentLiability.get_bel()
to avoid re-implementing validated annuity logic.

Model point schema (required DataFrame columns):
    mp_id                : any
    sex                  : str    "M" or "F"
    age                  : float
    weight               : float
    deferred_pension_pa  : float  Pension at date of leaving (pre-revaluation, £ p.a.)
    era                  : float  Earliest retirement age
    nra                  : float  Normal retirement age
    revaluation_type     : str    "CPI", "RPI", or "fixed"
    revaluation_cap      : float
    revaluation_floor    : float
    deferment_years      : float  Years already deferred at valuation date

NOTE: Computation logic is not included in this public demo.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.core.projection_calendar import ProjectionCalendar
from engine.liability.base_liability import (
    Decrements,
    LiabilityCashflows,
)
from engine.liability.multi_decrement import MultiDecrementLiability
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.in_payment import InPaymentLiability


REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "mp_id", "sex", "age", "weight",
    "deferred_pension_pa", "era", "nra",
    "revaluation_type", "revaluation_cap", "revaluation_floor",
    "deferment_years",
})


class DeferredLiability(MultiDecrementLiability):
    """
    Deferred member liability for BPA.

    Parameters
    ----------
    calendar : ProjectionCalendar
    tv_annuity_factor : float
        Placeholder annuity factor for TV amount (Step 18).
        Replaced by actuarial PV in Step 20.
    """

    def __init__(
        self,
        calendar: ProjectionCalendar,
        tv_annuity_factor: float = 20.0,
    ) -> None:
        self._calendar = calendar
        self._tv_annuity_factor = tv_annuity_factor
        self._in_payment = InPaymentLiability(calendar)

    @staticmethod
    def _validate(model_points: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(model_points.columns)
        if missing:
            raise ValueError(f"DeferredLiability: missing columns {missing}")

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        """ABC stub — returns zeros."""
        self._validate(model_points)
        n = len(model_points)
        return pd.DataFrame({
            "mp_id":      model_points["mp_id"].values,
            "q_death":    np.zeros(n),
            "q_retire":   np.zeros(n),
            "q_transfer": np.zeros(n),
            "q_commute":  np.zeros(n),
        })

    def project_cashflows(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> LiabilityCashflows:
        """
        Project deferred member cashflows (TV outgo + expenses) for one period.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

    def get_bel(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> float:
        """
        BEL via two-phase forward projection.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

    def get_reserve(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> float:
        """Reserve = BEL (no risk margin in Phase 3 Step 18)."""
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

    def get_decrements(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> Decrements:
        """
        Aggregate decrement counts: deaths, TV lapses, retirements.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

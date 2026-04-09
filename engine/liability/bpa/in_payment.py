"""
InPaymentLiability — BPA in-payment pensioner liability class.

Design (DECISIONS.md §15, §18, §19, §27)
------------------------------------------
Models a block of in-payment pensioners with LPI-linked annual pension.
One primary decrement: death.

Cashflow mapping onto LiabilityCashflows:
    maturity_payments  — pension outgo this period
    expenses           — per-policy expense loading
    premiums           — 0
    death_claims       — 0
    surrender_payments — 0

Model point schema (required DataFrame columns):
    mp_id       : any
    sex         : str    "M" or "F"
    age         : float  Age at valuation date
    weight      : float  Number of lives in group (>= 0)
    pension_pa  : float  Annual pension at valuation date (£)
    lpi_cap     : float  LPI cap, e.g. 0.05
    lpi_floor   : float  LPI floor, typically 0.0
    gmp_pa      : float  GMP component of pension_pa; 0.0 if none

dt is sourced from ProjectionCalendar via period.year_fraction.
BEL discounting uses spot discount factors from RiskFreeRateCurve.
In Step 20 the discount curve is replaced by the post-MA adjusted curve.

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


# ---------------------------------------------------------------------------
# Required model point columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "mp_id", "sex", "age", "weight",
    "pension_pa", "lpi_cap", "lpi_floor", "gmp_pa",
})


# ---------------------------------------------------------------------------
# InPaymentLiability
# ---------------------------------------------------------------------------

class InPaymentLiability(MultiDecrementLiability):
    """
    In-payment pensioner liability for BPA.

    Parameters
    ----------
    calendar : ProjectionCalendar
        Hybrid timestep calendar for this run.
    """

    def __init__(self, calendar: ProjectionCalendar) -> None:
        self._calendar = calendar

    @staticmethod
    def _validate(model_points: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(model_points.columns)
        if missing:
            raise ValueError(f"InPaymentLiability: missing columns {missing}")

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        """ABC stub — returns zeros (no assumptions at ABC level)."""
        self._validate(model_points)
        n = len(model_points)
        return pd.DataFrame({
            "mp_id":      model_points["mp_id"].values,
            "q_death":    np.zeros(n),
            "q_retire":   np.zeros(n),
            "q_transfer": np.zeros(n),
            "q_commute":  np.zeros(n),
        })

    def get_decrement_rates_with_assumptions(
        self,
        t: int,
        model_points: pd.DataFrame,
        assumptions: BPAAssumptions,
    ) -> pd.DataFrame:
        """
        Per-model-point period mortality rates using the mortality basis.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

    def project_cashflows(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> LiabilityCashflows:
        """
        Project pension outgo for one projection period.

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
        Best Estimate Liability — present value of all future pension cashflows.

        Projects forward over all future periods, applying survival probabilities
        from the mortality basis and LPI inflation, discounted using spot DFs.

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
        Aggregate decrement counts for this timestep.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

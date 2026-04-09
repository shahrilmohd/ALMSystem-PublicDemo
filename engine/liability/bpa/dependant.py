"""
DependantLiability — BPA contingent dependant pension liability class.

Design (DECISIONS.md §26)
--------------------------
Models contingent dependant pensions triggered by the death of a main member.
Each model point represents one (member, potential-dependant) pair.

Weight convention: model point weight = member_weight × dependant_proportion,
applied by BPADataLoader. This class receives weight-adjusted model points.

BEL is computed as a convolution over trigger periods:

  BEL = Σ_s  f_death(s) × V_dep(s)

where f_death(s) is the member death probability density at period s and
V_dep(s) is the PV of the dependant annuity if triggered at the end of period s.
NumPy array operations are used to avoid Python per-period overhead.

Model point schema:
    mp_id          : any
    member_sex     : str    "M" or "F"
    member_age     : float
    dependant_sex  : str    "M" or "F"
    dependant_age  : float
    weight         : float  member_weight × dependant_proportion (>= 0)
    pension_pa     : float  Dependant annual pension (£)
    lpi_cap        : float
    lpi_floor      : float

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


REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "mp_id", "member_sex", "member_age", "dependant_sex", "dependant_age",
    "weight", "pension_pa", "lpi_cap", "lpi_floor",
})


class DependantLiability(MultiDecrementLiability):
    """
    Contingent dependant pension liability for BPA.

    Parameters
    ----------
    calendar : ProjectionCalendar
    """

    def __init__(self, calendar: ProjectionCalendar) -> None:
        self._calendar = calendar

    @staticmethod
    def _validate(model_points: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(model_points.columns)
        if missing:
            raise ValueError(f"DependantLiability: missing columns {missing}")

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
        Expected new dependant pension outgo triggered in period t.

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
        BEL via convolution over future trigger periods.

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
        Aggregate member decrement counts (deaths triggering dependant pensions).

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "BPA computation logic is not included in the public demo."
        )

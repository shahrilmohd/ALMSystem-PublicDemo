"""
EnhancedLiability — BPA impaired / enhanced life liability class.

Design (DECISIONS.md §19)
--------------------------
Enhanced lives are in-payment pensioners whose mortality has been individually
medically underwritten, expressed as a positive age rating (additional years).
The age rating shifts the lookup into the same S3 ultimate table — not a
separate select table.

EnhancedLiability subclasses InPaymentLiability and overrides only the
mortality-related methods to apply the rated age. All cashflow, BEL, reserve,
and decrement logic is inherited unchanged.

Additional model point column (beyond InPaymentLiability.REQUIRED_COLUMNS):
    rating_years : float  Age shift for mortality lookup. Must be >= 0.
                          rating_years = 0 produces standard InPaymentLiability results.

NOTE: Computation logic is not included in this public demo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.core.projection_calendar import ProjectionCalendar
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.in_payment import InPaymentLiability, REQUIRED_COLUMNS


ENHANCED_REQUIRED_COLUMNS: frozenset[str] = REQUIRED_COLUMNS | frozenset({"rating_years"})


class EnhancedLiability(InPaymentLiability):
    """
    Impaired / enhanced life liability for BPA.

    Identical to InPaymentLiability except mortality uses effective_age =
    actual_age + elapsed_years + rating_years.

    Parameters
    ----------
    calendar : ProjectionCalendar
    """

    def __init__(self, calendar: ProjectionCalendar) -> None:
        super().__init__(calendar)

    @staticmethod
    def _validate(model_points: pd.DataFrame) -> None:
        missing = ENHANCED_REQUIRED_COLUMNS - set(model_points.columns)
        if missing:
            raise ValueError(f"EnhancedLiability: missing columns {missing}")

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        """ABC stub — returns zeros; use get_decrement_rates_with_assumptions()."""
        self._validate(model_points)
        n = len(model_points)
        return pd.DataFrame({
            "mp_id":      model_points["mp_id"].values,
            "q_death":    np.zeros(n),
            "q_retire":   np.zeros(n),
            "q_transfer": np.zeros(n),
            "q_commute":  np.zeros(n),
        })

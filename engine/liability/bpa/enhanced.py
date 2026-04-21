"""
engine/liability/bpa/enhanced.py — BPA enhanced (impaired life) annuity liability.

Proprietary implementation — stubbed in public demo.
Subclasses InPaymentLiability with age-rating adjustments for impaired
lives. The effective age is derived from the rated mortality assumption
rather than the actual age.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from engine.liability.bpa.in_payment import InPaymentLiability
from engine.core.projection_calendar import ProjectionCalendar


class EnhancedLiability(InPaymentLiability):
    """
    BPA enhanced annuity liability — impaired life age-rating.

    Subclasses InPaymentLiability; overrides effective age computation
    to apply the rated age offset for impaired lives.
    """

    def __init__(self, calendar: ProjectionCalendar) -> None:
        self._calendar = calendar

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

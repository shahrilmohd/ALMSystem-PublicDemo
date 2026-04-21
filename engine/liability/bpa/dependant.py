"""
engine/liability/bpa/dependant.py — BPA contingent dependant liability.

Proprietary implementation — stubbed in public demo.
Models contingent spouse/dependant benefits that arise on member death.
Cash flows are computed by convolving member mortality with dependant
survival probabilities.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from engine.core.projection_calendar import ProjectionCalendar
from engine.liability.multi_decrement import MultiDecrementLiability


class DependantLiability(MultiDecrementLiability):
    """
    BPA contingent dependant liability.

    Cash flows: dependant pension payments conditional on member death,
    scaled by dependant proportion and joint-life survival.
    """

    def __init__(self, calendar: ProjectionCalendar) -> None:
        self._calendar = calendar

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def get_decrement_rates_with_assumptions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def project_cashflows(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def get_bel(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def get_reserve(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def get_decrements(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def step_time(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

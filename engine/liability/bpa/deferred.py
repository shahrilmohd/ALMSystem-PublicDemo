"""
engine/liability/bpa/deferred.py — BPA deferred member liability.

Proprietary implementation — stubbed in public demo.
Models deferred pension scheme members under a 4-decrement framework
(mortality, retirement, withdrawal, ill-health retirement). At retirement,
deferred members convert to in-payment annuitants.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from engine.core.projection_calendar import ProjectionCalendar
from engine.liability.multi_decrement import MultiDecrementLiability


class DeferredLiability(MultiDecrementLiability):
    """
    BPA deferred member liability — not yet in receipt of pension.

    Decrements: mortality, retirement, withdrawal, ill-health.
    At retirement: spawns InPaymentLiability for resulting in-payment exposure.
    """

    def __init__(self, calendar: ProjectionCalendar, *args: Any, **kwargs: Any) -> None:
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

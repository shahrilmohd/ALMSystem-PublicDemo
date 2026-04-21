"""
engine/ifrs17/risk_adjustment.py — IFRS 17 Risk Adjustment (Cost of Capital method).

Proprietary implementation — stubbed in public demo.
Computes the RA as the present value of the cost of holding risk capital
over the remaining coverage period, using the SII longevity SCR as a proxy.
"""
from __future__ import annotations
from typing import Any

_DEFAULT_COC_RATE: float = 0.06


class CostOfCapitalRA:
    """Risk Adjustment via the Cost of Capital method (IFRS 17 B91)."""

    def __init__(self, coc_rate: float = _DEFAULT_COC_RATE) -> None:
        self._coc_rate = coc_rate

    @property
    def coc_rate(self) -> float:
        return self._coc_rate

    def compute(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

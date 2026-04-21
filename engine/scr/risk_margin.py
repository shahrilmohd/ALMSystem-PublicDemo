"""
engine/scr/risk_margin.py — SII Risk Margin (Cost of Capital run-off method).

Proprietary implementation — stubbed in public demo.
Projects the SCR run-off under best estimate assumptions and discounts
the projected cost of capital at the risk-free rate.
"""
from __future__ import annotations
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


class RiskMarginCalculator:
    """Computes the IFRS 17 / SII Risk Margin via cost-of-capital run-off."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

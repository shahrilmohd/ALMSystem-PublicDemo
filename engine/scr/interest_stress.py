"""
engine/scr/interest_stress.py — SII interest rate stress engine.

Proprietary implementation — stubbed in public demo.
Computes the interest SCR as the change in net asset value under
relative up and down shifts to the risk-free rate curve.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class InterestStressResult:
    scr_interest_up: float = 0.0
    scr_interest_down: float = 0.0
    scr_interest: float = 0.0


class InterestStressEngine:
    """SII Standard Formula interest rate stress — SCR_interest module."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> InterestStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

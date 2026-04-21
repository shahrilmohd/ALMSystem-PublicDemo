"""
engine/scr/spread_stress.py — SII credit spread stress engine.

Proprietary implementation — stubbed in public demo.
Computes the spread SCR as the change in net asset value under a credit
spread widening shock, with optional Matching Adjustment offset.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class SpreadStressResult:
    scr_spread_up: float = 0.0
    scr_spread_down: float = 0.0
    scr_spread: float = 0.0
    ma_offset: float = 0.0
    net_scr_spread: float = 0.0


class SpreadStressEngine:
    """SII Standard Formula spread stress — SCR_spread module."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> SpreadStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

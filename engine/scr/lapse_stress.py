"""
engine/scr/lapse_stress.py — SII lapse stress engine.

Proprietary implementation — stubbed in public demo.
Computes SCR_lapse as the worst of permanent up, permanent down,
and mass lapse scenarios.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class LapseStressResult:
    scr_lapse_up: float = 0.0
    scr_lapse_down: float = 0.0
    scr_lapse_mass: float = 0.0
    scr_lapse: float = 0.0


class LapseStressEngine:
    """SII Standard Formula lapse stress — SCR_lapse module."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> LapseStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

"""
engine/scr/longevity_stress.py — SII longevity stress engine.

Proprietary implementation — stubbed in public demo.
Applies a permanent improvement shock to base mortality rates and
computes the resulting increase in BEL as the longevity SCR.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class LongevityStressResult:
    base_bel: float = 0.0
    stressed_bel: float = 0.0
    scr_longevity: float = 0.0
    mortality_stress_factor: float = 0.0


class LongevityStressEngine:
    """SII Standard Formula longevity stress — SCR_longevity module."""

    def __init__(self, mortality_stress_factor: float = 0.20) -> None:
        self._mortality_stress_factor = mortality_stress_factor

    @property
    def mortality_stress_factor(self) -> float:
        return self._mortality_stress_factor

    def compute(self, *args: Any, **kwargs: Any) -> LongevityStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

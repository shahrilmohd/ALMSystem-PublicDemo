"""
engine/scr/currency_stress.py — SII currency stress engine.

Proprietary implementation — stubbed in public demo.
Computes SCR_currency as the worst-case currency shock across
foreign currency asset exposures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class CurrencyStressResult:
    scr_currency: float = 0.0


class CurrencyStressEngine:
    """SII Standard Formula currency stress — SCR_currency module."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> CurrencyStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

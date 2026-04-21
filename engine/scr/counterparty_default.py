"""
engine/scr/counterparty_default.py — SII counterparty default risk engine.

Proprietary implementation — stubbed in public demo.
Computes SCR_default based on loss-given-default and probability of default
for type 1 and type 2 exposures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class CounterpartyExposure:
    """Represents a single counterparty credit exposure."""
    name: str
    exposure: float
    probability_of_default: float
    loss_given_default: float
    exposure_type: int = 1

    def __post_init__(self) -> None:
        if self.exposure_type not in (1, 2):
            raise ValueError("exposure_type must be 1 or 2")


@dataclass(frozen=True)
class CounterpartyDefaultResult:
    scr_default: float = 0.0
    type1_scr: float = 0.0
    type2_scr: float = 0.0


class CounterpartyDefaultEngine:
    """SII Standard Formula counterparty default risk — SCR_default module."""

    def compute(self, *args: Any, **kwargs: Any) -> CounterpartyDefaultResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

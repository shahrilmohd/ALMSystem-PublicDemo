"""
engine/ifrs17/csm.py — CSM tracker for IFRS 17 GMM.

Proprietary implementation — stubbed in public demo.
Tracks the Contractual Service Margin balance, applies accretion,
experience variance adjustments, and unlocking for assumption changes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CsmStepResult:
    """Per-period CSM movement output. Fields represent the CSM waterfall."""
    opening_csm: float = 0.0
    accretion: float = 0.0
    experience_variance: float = 0.0
    assumption_change: float = 0.0
    release: float = 0.0
    closing_csm: float = 0.0


class CsmTracker:
    """Accumulates the CSM balance across IFRS 17 reporting periods."""

    def __init__(self, opening_csm: float, locked_in_rate: float) -> None:
        self._csm = opening_csm
        self._locked_in_rate = locked_in_rate

    @property
    def balance(self) -> float:
        return self._csm

    def step(self, *args: Any, **kwargs: Any) -> CsmStepResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

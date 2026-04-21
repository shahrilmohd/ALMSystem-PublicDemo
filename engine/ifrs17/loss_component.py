"""
engine/ifrs17/loss_component.py — IFRS 17 Loss Component tracker.

Proprietary implementation — stubbed in public demo.
Monitors onerous contracts, tracks the loss component balance,
and reverses the loss component as the deficit unwinds.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LossComponentStepResult:
    opening_balance: float = 0.0
    onerous_excess_added: float = 0.0
    release: float = 0.0
    closing_balance: float = 0.0
    is_onerous: bool = False


class LossComponentTracker:
    """Tracks the IFRS 17 loss component for onerous contract groups."""

    def __init__(self, opening_loss_component: float = 0.0) -> None:
        self._balance = opening_loss_component

    @property
    def balance(self) -> float:
        return self._balance

    def add_onerous_excess(self, excess: float) -> None:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def step(self, *args: Any, **kwargs: Any) -> LossComponentStepResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

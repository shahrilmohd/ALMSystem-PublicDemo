"""
engine/scr/expense_stress.py — SII expense stress engine.

Proprietary implementation — stubbed in public demo.
Applies an increase in expense levels and expense inflation rate
to compute SCR_expense.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class ExpenseStressResult:
    scr_expense: float = 0.0
    stressed_expense_increase: float = 0.0
    stressed_inflation_increase: float = 0.0


class ExpenseStressEngine:
    """SII Standard Formula expense stress — SCR_expense module."""

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions

    def compute(self, *args: Any, **kwargs: Any) -> ExpenseStressResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

"""
engine/scr/
===========
SII standard formula SCR stress engines.

Step 22 exports
---------------
SCRResult           — frozen dataclass holding spread/interest/longevity outputs
SpreadStressEngine  — credit spread stress with optional MA offset (DECISIONS.md §5, §51)
InterestStressEngine — interest rate curve stress (DECISIONS.md §51)
LongevityStressEngine — 20% mortality improvement stress (DECISIONS.md §49, §51)
SCRCalculator       — orchestrates all three engines → SCRResult

Step 26 exports (full BSCR)
----------------------------
SCRStressAssumptions — central assumption table for all SII stress parameters
LapseStressEngine / LapseStressResult
ExpenseStressEngine / ExpenseStressResult
CurrencyStressEngine / CurrencyStressResult
CounterpartyExposure / CounterpartyDefaultEngine / CounterpartyDefaultResult
BSCRResult          — full SII BSCR output dataclass
BSCRAggregator      — quadratic form correlation aggregation
RiskMarginCalculator — CoC run-off risk margin
BSCRCalculator      — single-arg full BSCR orchestrator
"""
from engine.scr.scr_assumptions import SCRStressAssumptions
from engine.scr.scr_result import SCRResult
from engine.scr.spread_stress import SpreadStressEngine, SpreadStressResult
from engine.scr.interest_stress import InterestStressEngine, InterestStressResult
from engine.scr.longevity_stress import LongevityStressEngine, LongevityStressResult
from engine.scr.scr_calculator import SCRCalculator
from engine.scr.lapse_stress import LapseStressEngine, LapseStressResult
from engine.scr.expense_stress import ExpenseStressEngine, ExpenseStressResult
from engine.scr.currency_stress import CurrencyStressEngine, CurrencyStressResult
from engine.scr.counterparty_default import (
    CounterpartyExposure,
    CounterpartyDefaultEngine,
    CounterpartyDefaultResult,
)
from engine.scr.bscr_result import BSCRResult
from engine.scr.bscr_aggregator import BSCRAggregator
from engine.scr.risk_margin import RiskMarginCalculator
from engine.scr.bscr_calculator import BSCRCalculator

__all__ = [
    # Step 22
    "SCRResult",
    "SpreadStressEngine",
    "SpreadStressResult",
    "InterestStressEngine",
    "InterestStressResult",
    "LongevityStressEngine",
    "LongevityStressResult",
    "SCRCalculator",
    # Step 26
    "SCRStressAssumptions",
    "LapseStressEngine",
    "LapseStressResult",
    "ExpenseStressEngine",
    "ExpenseStressResult",
    "CurrencyStressEngine",
    "CurrencyStressResult",
    "CounterpartyExposure",
    "CounterpartyDefaultEngine",
    "CounterpartyDefaultResult",
    "BSCRResult",
    "BSCRAggregator",
    "RiskMarginCalculator",
    "BSCRCalculator",
]

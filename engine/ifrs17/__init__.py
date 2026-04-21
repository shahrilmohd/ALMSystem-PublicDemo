"""
engine/ifrs17 — IFRS 17 General Measurement Model (GMM) engine.

Public API
----------
GmmEngine            Orchestrates all GMM components per contract group.
GmmStepResult        Per-period movement output from GmmEngine.
Ifrs17State          Cross-period rolling balances (CSM, loss component, CU).
CoverageUnitProvider Protocol — injected per product (BPA in Step 21).
CostOfCapitalRA      Risk Adjustment via Cost of Capital method.
LockedInAssumptions  Frozen-at-inception assumptions.
CurrentAssumptions   Current-period assumptions.
AssumptionProvider   Protocol for supplying assumptions to the run mode.

See DECISIONS.md §28, §33–§37 for architecture and design rationale.
"""
from engine.ifrs17.gmm import GmmEngine, GmmStepResult
from engine.ifrs17.state import Ifrs17State
from engine.ifrs17.coverage_units import CoverageUnitProvider
from engine.ifrs17.risk_adjustment import CostOfCapitalRA
from engine.ifrs17.assumptions import (
    AssumptionProvider,
    CurrentAssumptions,
    LockedInAssumptions,
)

__all__ = [
    "GmmEngine",
    "GmmStepResult",
    "Ifrs17State",
    "CoverageUnitProvider",
    "CostOfCapitalRA",
    "AssumptionProvider",
    "CurrentAssumptions",
    "LockedInAssumptions",
]

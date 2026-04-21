"""
engine/ifrs17/gmm.py — IFRS 17 Gross Margin Model engine.

Proprietary implementation — stubbed in public demo.
GmmEngine orchestrates the per-period GMM step: CSM accretion and unlocking,
experience variance attribution, RA release, LRC/LIC split, loss component
monitoring, and coverage unit allocation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GmmStepResult:
    """Per-period IFRS 17 GMM movement output."""
    period_start: Any = None
    period_end: Any = None
    opening_csm: float = 0.0
    closing_csm: float = 0.0
    csm_release: float = 0.0
    opening_ra: float = 0.0
    closing_ra: float = 0.0
    ra_release: float = 0.0
    insurance_revenue: float = 0.0
    insurance_service_result: float = 0.0
    opening_lrc: float = 0.0
    closing_lrc: float = 0.0
    opening_lic: float = 0.0
    closing_lic: float = 0.0
    loss_component: float = 0.0
    is_onerous: bool = False


class GmmEngine:
    """
    IFRS 17 General Measurement Model engine.

    Orchestrates GmmStepResult per reporting period for a single contract group.
    Injected into BPARun; called at each accounting period boundary.
    State is persisted via Ifrs17StateStore between valuation dates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def step(self, *args: Any, **kwargs: Any) -> GmmStepResult:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

    def closing_state(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Proprietary implementation — not available in public demo.")

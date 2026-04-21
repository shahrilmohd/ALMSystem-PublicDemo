"""
engine/ifrs17/assumptions.py — IFRS 17 assumption objects

Separates locked-in assumptions (frozen at contract group inception) from
current-period assumptions (updated each reporting period). This split is
required for correct IFRS 17 movement attribution:

  - Non-financial assumption changes (mortality, inflation) for future service
    → adjust the CSM (IFRS 17 para 44)
  - Financial assumption changes (discount rate) for future service
    → OCI if OCI option elected (IFRS 17 para 88b)

See DECISIONS.md §34 for the full rationale including the dual BEL requirement
and the identity: fcf_change_financial = bel_current - bel_locked.

The AssumptionProvider protocol is injected into the run mode orchestrator.
GmmEngine itself never calls AssumptionProvider directly — the orchestrator
resolves assumptions and passes bel_current, bel_locked, and
fcf_change_non_financial to GmmEngine.step().
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class LockedInAssumptions:
    """
    Assumptions frozen at contract group inception.

    The locked-in rate is used for:
      1. CSM accretion (IFRS 17 para 44 — must not use current rate)
      2. BEL_locked calculation (to split finance income between P&L and OCI)

    The rate is stored as a single parallel shift (not a full yield curve),
    consistent with the MA benefit design in §21. See DECISIONS.md §34.

    Attributes
    ----------
    cohort_id : str
        Contract group this applies to.
    inception_date : date
        Date of first recognition. The locked-in rate was determined on this date.
    locked_in_rate : float
        Annual discount rate as a decimal (e.g. 0.05 for 5%).
        For BPA: post-MA parallel shift rate at the transaction date (§21).
    """

    cohort_id:      str
    inception_date: date
    locked_in_rate: float

    def __post_init__(self) -> None:
        if self.locked_in_rate <= -1.0:
            raise ValueError(
                f"locked_in_rate must be > -1, got {self.locked_in_rate}"
            )


@dataclass(frozen=True)
class CurrentAssumptions:
    """
    Current-period assumptions supplied by the run mode at each timestep.

    Used to compute bel_current (BEL at the current discount rate).
    Changes in these assumptions relative to the prior period drive the
    fcf_change attribution inputs to GmmEngine.step().

    Attributes
    ----------
    t : int
        0-based projection period index.
    current_rate : float
        Current-period annual discount rate as a decimal.
    mortality_table : str
        Identifier of the mortality table in use (e.g. "S3PMA_CMI2023").
        Used by the orchestrator to detect table changes between periods.
    inflation_index : float
        Current CPI or RPI annual rate from the ESG scenario (decimal).
    """

    t:               int
    current_rate:    float
    mortality_table: str
    inflation_index: float

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"t must be >= 0, got {self.t}")
        if self.current_rate <= -1.0:
            raise ValueError(
                f"current_rate must be > -1, got {self.current_rate}"
            )


class AssumptionProvider(Protocol):
    """
    Protocol for supplying IFRS 17 assumptions to the run mode orchestrator.

    The orchestrator calls get_locked_in() once per contract group at run
    start (to construct GmmEngine) and get_current(t) at each timestep
    (to resolve the current-period discount rate and mortality table).

    Implementations are product-specific and injected by the run mode.
    GmmEngine never calls this protocol directly.
    """

    def get_locked_in(self, cohort_id: str) -> LockedInAssumptions:
        """Return the locked-in assumptions for the given contract group."""
        ...

    def get_current(self, t: int) -> CurrentAssumptions:
        """Return the current-period assumptions for projection period t."""
        ...

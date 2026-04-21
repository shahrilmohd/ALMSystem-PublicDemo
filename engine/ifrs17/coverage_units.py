"""
engine/ifrs17/coverage_units.py — CoverageUnitProvider protocol

Coverage units measure the insurance service provided to policyholders each
period. For BPA the definition is the PV of expected future annuity payments
at the locked-in rate — a decreasing stock measure. See DECISIONS.md §35.

This module defines the protocol only. Product-specific implementations are:
  - BPACoverageUnitProvider (Step 21, engine/liability/bpa/)
  - Conventional implementations (future)

GmmEngine accepts any object that satisfies this protocol.
"""
from __future__ import annotations

from typing import Protocol


class CoverageUnitProvider(Protocol):
    """
    Protocol for supplying coverage unit values to GmmEngine.

    Coverage units for period t represent the insurance service consumed
    in that period. GmmEngine uses them to determine what fraction of the
    opening CSM is released to P&L.

    For BPA (DECISIONS.md §35):
      units_consumed(t)  = PV of outgo in period t, discounted at locked-in rate
      units_remaining(t) = PV of all future outgo from period t onwards,
                           discounted at locked-in rate (a decreasing balance)

    The release fraction at period t is:
      release_fraction = units_consumed(t) / units_remaining_opening(t)

    where units_remaining_opening(t) is the remaining balance at the START
    of period t (before consuming period t's units).

    Implementations must be provided by the product layer and injected into
    GmmEngine at construction. GmmEngine never computes coverage units itself.
    """

    def units_consumed(self, t: int) -> float:
        """
        Coverage units consumed (insurance service delivered) in period t.

        For BPA: PV of expected pension outgo in period t at the locked-in rate.

        Parameters
        ----------
        t : int
            0-based projection period index.

        Returns
        -------
        float
            Units consumed. Must be >= 0.
        """
        ...

    def units_remaining(self, t: int) -> float:
        """
        Remaining coverage units at the START of period t (before consuming
        period t's units).

        For BPA: PV of all expected pension outgo from period t to end of
        projection, discounted at the locked-in rate.

        Parameters
        ----------
        t : int
            0-based projection period index.

        Returns
        -------
        float
            Remaining units. Must be >= 0.
            Returns 0 after the final projection period.
        """
        ...

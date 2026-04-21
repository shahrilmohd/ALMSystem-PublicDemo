"""
MultiDecrementLiability — intermediate abstract class for BPA liability models.

Position in the hierarchy (DECISIONS.md §15)
--------------------------------------------
    BaseLiability (abstract)
    ├── ConventionalLiability          ← unchanged
    ├── ULLiability                    ← unchanged
    └── MultiDecrementLiability (abstract, this module)
        ├── InPaymentLiability         ← Phase 3
        ├── DeferredLiability          ← Phase 3
        ├── DependantLiability         ← Phase 3
        └── EnhancedLiability          ← Phase 3

Purpose
-------
BPA liabilities require simultaneous projection of multiple competing decrements
at each timestep (mortality, retirement, transfer value, commutation, ill-health).
Conventional and UL products operate on a single primary decrement and must not
be required to implement multi-decrement logic they have no use for.

This class adds one abstract method — get_decrement_rates() — without modifying
BaseLiability or any of its existing subclasses.

Naming note
-----------
DECISIONS.md §15 names this method `get_decrements`. That name is already taken
by BaseLiability.get_decrements(), which returns an aggregate `Decrements` count
object (in_force_start, deaths, lapses, maturities, in_force_end) used by Fund
and run modes to track in-force counts.

The new method returns per-model-point *probabilities* (rates), not aggregate
counts — a fundamentally different thing. It is named `get_decrement_rates` here
to make that distinction unambiguous and avoid a Liskov Substitution violation.
BPA subclasses must implement both: BaseLiability.get_decrements() for the
aggregate count interface, and get_decrement_rates() for the per-MP rate interface.

Interface contract
------------------
Subclasses must implement:
  1. All abstract methods inherited from BaseLiability
     (project_cashflows, get_bel, get_reserve, get_decrements).
  2. get_decrement_rates(t, model_points) — returns a per-model-point DataFrame
     of decrement probabilities for timestep t.

See DECISIONS.md §15 for the full design rationale.
"""
from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from engine.liability.base_liability import BaseLiability


class MultiDecrementLiability(BaseLiability):
    """
    Abstract intermediate class for liability models with multiple competing
    decrements operating simultaneously each period.

    Required by all BPA liability types (InPaymentLiability, DeferredLiability,
    DependantLiability, EnhancedLiability). Not required by conventional or
    unit-linked products.

    Adds one abstract method to the BaseLiability contract:

        get_decrements(t, model_points) → pd.DataFrame

    All other BaseLiability abstract methods (project, get_bel, get_reserve)
    must still be implemented by concrete subclasses.
    """

    @abstractmethod
    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of decrement *probabilities* for each model point at
        timestep t.

        This method is distinct from BaseLiability.get_decrements(), which returns
        aggregate Decrements counts used by Fund/run modes for in-force tracking.
        get_decrement_rates() returns per-model-point probability rates consumed
        internally by BPA subclasses inside their own project_cashflows() logic.

        Parameters
        ----------
        t : int
            0-based projection timestep (month index for monthly periods;
            period index for hybrid-timestep BPA runs — see DECISIONS.md §27).
        model_points : pd.DataFrame
            Current in-force model point DataFrame. Must not be modified.

        Returns
        -------
        pd.DataFrame
            One row per model point. Required columns:

            mp_id      : any   Model point identifier.
            q_death    : float Probability of death this period. [0, 1].
            q_retire   : float Probability of retirement this period. [0, 1].
                                0.0 for in-payment members (already retired).
            q_transfer : float Probability of transfer value election. [0, 1].
                                0.0 for in-payment members and full-buyout schemes
                                (DECISIONS.md §25).
            q_commute  : float Probability of commutation at retirement. [0, 1].
                                Typically applied at the retirement event only.

        Constraints
        -----------
        - All probability columns are floats in [0.0, 1.0].
        - The sum of all decrement probabilities for a single model point must
          not exceed 1.0 at any timestep.
        - Irrelevant columns for a given subclass must still be present, set to 0.0.
        """

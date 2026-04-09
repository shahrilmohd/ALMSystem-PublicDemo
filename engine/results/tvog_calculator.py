"""
TvogCalculator — computes the Time Value of Options and Guarantees (TVOG).

Definition
----------
TVOG = mean(BEL₀ᵢ over all scenarios i) − deterministic BEL₀

where BEL₀ᵢ is the Best Estimate Liability at timestep 0 for ESG scenario i.

This captures the option value embedded in policyholder guarantees such as
minimum maturity benefits, bonus floors, and minimum death benefits. Under
adverse economic paths these guarantees bite, inflating the BEL above the
central estimate. The mean across all paths is higher than the deterministic
BEL precisely because guarantee payoffs are asymmetric — favourable scenarios
reduce the BEL somewhat, but adverse scenarios increase it substantially.

Inputs
------
  result_store       : ResultStore populated by a StochasticRun (N scenarios).
                       BEL at timestep 0 is read for each scenario.
  deterministic_bel  : BEL at t=0 from a DeterministicRun (the central estimate).
                       This is the liability value computed under best-estimate
                       assumptions without stochastic uncertainty.

Output
------
  TvogResult — see dataclass below.

Numerical anchor
----------------
Given 3 scenarios with BEL₀ = [1_000, 1_200, 1_400] and deterministic BEL₀ = 1_100:
  mean_stochastic_bel = 1_200.0
  tvog                = 100.0

Given 3 scenarios with BEL₀ all equal to 1_000 and deterministic BEL₀ = 1_000:
  tvog = 0.0   (no rate uncertainty → no option value, as expected for flat ESG paths)

References
----------
  ALM_Architecture.md §6.3  (scenario loop and TVOG)
  DECISIONS.md §14           (StochasticRun design, BEL discounting)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.results.result_store import ResultStore


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TvogResult:
    """
    All outputs from a single TVOG calculation.

    Attributes
    ----------
    tvog : float
        Time Value of Options and Guarantees.
        = mean_stochastic_bel − deterministic_bel.
        Positive → stochastic mean BEL exceeds deterministic BEL (guarantee
        exposure present). Zero or negative → guarantees are out of the money
        on average across the scenario distribution.

    mean_stochastic_bel : float
        Arithmetic mean of BEL₀ across all N stochastic scenarios.

    deterministic_bel : float
        BEL₀ from the deterministic / central-estimate run. Supplied by the
        caller; not read from the store.

    n_scenarios : int
        Number of ESG scenarios used in the calculation.

    scenario_bels : list[float]
        BEL at t=0 for each scenario, ordered by scenario ID ascending.
        Retained for distribution analysis, charting, and tail attribution.

    percentile_bels : dict[int, float]
        BEL percentile distribution. Keys are the requested percentile integers
        (e.g. {5: 980.0, 50: 1050.0, 95: 1340.0}). Computed over scenario_bels
        using linear interpolation (NumPy default).
    """
    tvog:               float
    mean_stochastic_bel: float
    deterministic_bel:  float
    n_scenarios:        int
    scenario_bels:      list[float]
    percentile_bels:    dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class TvogCalculator:
    """
    Computes TVOG from a stochastic ResultStore and a deterministic BEL.

    Parameters
    ----------
    result_store : ResultStore
        Populated by StochasticRun.execute(). Must contain at least one scenario
        with a result at timestep 0. The store may contain any number of timesteps
        per scenario — only t=0 BEL is used for the TVOG calculation.
    deterministic_bel : float
        BEL at t=0 from a DeterministicRun. This is the central-estimate
        liability the TVOG is measured against.
    percentiles : list[int] | None
        Percentile points to compute over the scenario BEL distribution.
        Default: [5, 10, 25, 50, 75, 90, 95].
        Each value must be in [0, 100]. Duplicates are computed once.

    Usage
    -----
        calc   = TvogCalculator(store, deterministic_bel=12_345_678.0)
        result = calc.calculate()
        print(f"TVOG = {result.tvog:,.0f}")
        print(f"50th pct BEL = {result.percentile_bels[50]:,.0f}")
    """

    _DEFAULT_PERCENTILES: list[int] = [5, 10, 25, 50, 75, 90, 95]

    def __init__(
        self,
        result_store:      ResultStore,
        deterministic_bel: float,
        percentiles:       list[int] | None = None,
    ) -> None:
        self._store             = result_store
        self._deterministic_bel = deterministic_bel
        self._percentiles       = (
            percentiles if percentiles is not None else self._DEFAULT_PERCENTILES
        )

    # -----------------------------------------------------------------------
    # calculate
    # -----------------------------------------------------------------------

    def calculate(self) -> TvogResult:
        """
        Compute TVOG from the BEL distribution at t=0 across all scenarios.

        Algorithm
        ---------
        1. Collect BEL at timestep 0 for every scenario in the store, ordered
           by scenario ID ascending.
        2. Compute mean_stochastic_bel = arithmetic mean of those BELs.
        3. tvog = mean_stochastic_bel − deterministic_bel.
        4. Compute requested percentile points over the distribution.

        Returns
        -------
        TvogResult

        Raises
        ------
        ValueError
            If the ResultStore contains no scenarios.
        KeyError
            If any scenario is missing a result at timestep 0 (propagated from
            ResultStore.get()).
        """
        scenario_ids = self._store.all_scenarios()
        if not scenario_ids:
            raise ValueError(
                "ResultStore is empty — cannot compute TVOG. "
                "Run StochasticRun.execute() before calling TvogCalculator.calculate()."
            )

        # Read BEL at t=0 for each scenario, preserving scenario ID order
        scenario_bels: list[float] = [
            self._store.get(sid, timestep=0).bel
            for sid in scenario_ids
        ]

        bels = np.array(scenario_bels, dtype=np.float64)
        mean_bel = float(np.mean(bels))
        tvog     = mean_bel - self._deterministic_bel

        percentile_bels: dict[int, float] = {
            p: float(np.percentile(bels, p))
            for p in self._percentiles
        }

        return TvogResult(
            tvog=tvog,
            mean_stochastic_bel=mean_bel,
            deterministic_bel=self._deterministic_bel,
            n_scenarios=len(scenario_ids),
            scenario_bels=scenario_bels,
            percentile_bels=percentile_bels,
        )

"""
ResultStore — central collector for all model outputs.

Design
------
Every output produced during a projection run is written here. Models are
stateless: they compute and return values, but never retain them. The run
mode orchestrator calls store() after each timestep.

Index
-----
Results are keyed by (scenario_id, timestep, cohort_id).  The run_id is set
once on the store and validated on every write — a result from a different run
cannot be accidentally mixed in.

    scenario_id = 0       for liability-only and deterministic runs (one scenario)
    scenario_id = 1…N     for stochastic runs (N ESG scenarios)
    cohort_id   = None    for all non-BPA runs (conventional / unit-linked)
    cohort_id   = str     for BPA runs — identifies the contract group cohort
                          (DECISIONS.md §17)

The store makes no assumption about the number of scenarios or timesteps in
advance.  It grows dynamically as results arrive.

Retrieval
---------
    store.get(scenario_id, timestep)                 → single TimestepResult (cohort_id=None)
    store.get(scenario_id, timestep, cohort_id)      → single TimestepResult for BPA cohort
    store.all_timesteps(scenario_id)                 → list, ordered by timestep (cohort_id=None)
    store.all_timesteps(scenario_id, cohort_id)      → list for a specific BPA cohort
    store.all_scenarios()                            → sorted list of scenario IDs
    store.cohort_ids()                               → sorted list of distinct non-None cohort_ids
    store.as_dataframe()                             → flat pandas DataFrame
    store.as_cohort_pivot(scenario_id)               → wide DataFrame pivoted by cohort_id

Architectural rules (from ALM_Architecture.md):
  * engine/ has zero imports from frontend/, api/, or worker/
  * Results are never stored inside models — all outputs go here
  * Models are stateless between time steps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from engine.liability.base_liability import Decrements, LiabilityCashflows


# ---------------------------------------------------------------------------
# TimestepResult — one row in the result ledger
# ---------------------------------------------------------------------------

@dataclass
class TimestepResult:
    """
    All outputs for a single (scenario_id, timestep, cohort_id) combination.

    Liability fields (always populated):
        run_id, scenario_id, timestep, cohort_id, cashflows, decrements, bel, reserve.

    cohort_id (optional, DECISIONS.md §17):
        None  — all non-BPA runs (conventional / unit-linked).
        str   — BPA runs: identifies the contract group cohort within a run.
                Enables per-cohort reporting without a separate ResultStore per cohort.

    Asset fields (populated by DeterministicRun and StochasticRun; None for
    LiabilityOnlyRun which has no asset model):
        total_market_value, total_book_value, cash_balance,
        eir_income, coupon_income, dividend_income,
        unrealised_gl, realised_gl, oci_reserve,
        mv_ac, mv_fvtpl, mv_fvoci.

    BPA MA attribution fields (populated by BPARun only; None for all other runs):
        bel_pre_ma   — BEL discounted at plain EIOPA RFR (pre-MA).
                       Retained for attribution, stress testing, IFRS 17 inputs.
        bel_post_ma  — BEL discounted at EIOPA RFR + MA benefit (post-MA).
                       This is the regulatory Solvency II BEL (= bel for BPA runs).
        See DECISIONS.md §21.
    """

    run_id:      str
    scenario_id: int
    timestep:    int
    cashflows:   LiabilityCashflows
    decrements:  Decrements
    bel:         float
    reserve:     float
    cohort_id:   Optional[str] = field(default=None)

    # Asset fields — None for LIABILITY_ONLY runs
    total_market_value: Optional[float] = field(default=None)
    total_book_value:   Optional[float] = field(default=None)
    cash_balance:       Optional[float] = field(default=None)
    eir_income:         Optional[float] = field(default=None)
    coupon_income:      Optional[float] = field(default=None)
    dividend_income:    Optional[float] = field(default=None)
    unrealised_gl:      Optional[float] = field(default=None)
    realised_gl:        Optional[float] = field(default=None)
    oci_reserve:        Optional[float] = field(default=None)
    mv_ac:              Optional[float] = field(default=None)
    mv_fvtpl:           Optional[float] = field(default=None)
    mv_fvoci:           Optional[float] = field(default=None)

    # BPA MA attribution fields — None for all non-BPA runs (DECISIONS.md §21)
    bel_pre_ma:  Optional[float] = field(default=None)
    bel_post_ma: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

# Column order for as_dataframe() — kept as a module constant so callers can
# reference it (e.g. to build empty DataFrames with matching schema).
RESULT_COLUMNS: tuple[str, ...] = (
    "run_id", "scenario_id", "timestep", "cohort_id",
    # Cash flows
    "premiums", "death_claims", "surrender_payments",
    "maturity_payments", "expenses", "net_outgo",
    # Decrements
    "in_force_start", "deaths", "lapses", "maturities", "in_force_end",
    # Valuation
    "bel", "reserve",
    # Asset fields (None / NaN for LIABILITY_ONLY runs)
    "total_market_value", "total_book_value", "cash_balance",
    "eir_income", "coupon_income", "dividend_income",
    "unrealised_gl", "realised_gl", "oci_reserve",
    "mv_ac", "mv_fvtpl", "mv_fvoci",
    # BPA MA attribution (None / NaN for non-BPA runs — DECISIONS.md §21)
    "bel_pre_ma", "bel_post_ma",
)


class ResultStore:
    """
    Central collector for all projection outputs.

    Parameters
    ----------
    run_id : str
        Unique identifier for the run this store belongs to.  All results
        written to this store must carry the same run_id.

    Usage
    -----
        store = ResultStore(run_id="run_001")
        store.store(TimestepResult(...))
        df = store.as_dataframe()
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._results: dict[tuple[int, int, Optional[str]], TimestepResult] = {}

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Unique identifier for this run."""
        return self._run_id

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def store(self, result: TimestepResult) -> None:
        """
        Write a single timestep result.

        Raises
        ------
        ValueError
            If a result for (scenario_id, timestep) already exists, or if
            result.run_id does not match this store's run_id.
        """
        if result.run_id != self._run_id:
            raise ValueError(
                f"Result run_id '{result.run_id}' does not match "
                f"store run_id '{self._run_id}'."
            )
        key = (result.scenario_id, result.timestep, result.cohort_id)
        if key in self._results:
            raise ValueError(
                f"Result for scenario_id={result.scenario_id}, "
                f"timestep={result.timestep}, cohort_id={result.cohort_id!r} "
                f"has already been stored."
            )
        self._results[key] = result

    # -----------------------------------------------------------------------
    # Read — single result
    # -----------------------------------------------------------------------

    def get(
        self,
        scenario_id: int,
        timestep: int,
        cohort_id: Optional[str] = None,
    ) -> TimestepResult:
        """
        Retrieve the result for a specific (scenario_id, timestep, cohort_id).

        Parameters
        ----------
        scenario_id : int
        timestep    : int
        cohort_id   : str | None
            None (default) for all non-BPA runs.  Pass the cohort string for
            BPA runs where results are stored per contract group.

        Raises
        ------
        KeyError
            If no result exists for the given key.
        """
        key = (scenario_id, timestep, cohort_id)
        if key not in self._results:
            raise KeyError(
                f"No result stored for scenario_id={scenario_id}, "
                f"timestep={timestep}, cohort_id={cohort_id!r}."
            )
        return self._results[key]

    # -----------------------------------------------------------------------
    # Read — collections
    # -----------------------------------------------------------------------

    def all_timesteps(
        self,
        scenario_id: int,
        cohort_id: Optional[str] = None,
    ) -> list[TimestepResult]:
        """
        All results for a given (scenario_id, cohort_id), sorted by timestep.

        Parameters
        ----------
        scenario_id : int
        cohort_id   : str | None
            None (default) for non-BPA runs.  Pass the cohort string for
            BPA per-cohort retrieval.

        Returns an empty list if no matching results are found.
        """
        return sorted(
            (
                r for r in self._results.values()
                if r.scenario_id == scenario_id and r.cohort_id == cohort_id
            ),
            key=lambda r: r.timestep,
        )

    def all_scenarios(self) -> list[int]:
        """Sorted list of unique scenario IDs present in the store."""
        return sorted({r.scenario_id for r in self._results.values()})

    def cohort_ids(self) -> list[str]:
        """
        Sorted list of unique non-None cohort_ids stored.

        Returns an empty list for non-BPA runs (where every result has
        cohort_id=None).  BPA runs return the full sorted list of contract
        group identifiers.  See DECISIONS.md §17.
        """
        return sorted({
            r.cohort_id
            for r in self._results.values()
            if r.cohort_id is not None
        })

    # -----------------------------------------------------------------------
    # Counts
    # -----------------------------------------------------------------------

    def result_count(self) -> int:
        """Total number of (scenario_id, timestep) results stored."""
        return len(self._results)

    def scenario_count(self) -> int:
        """Number of distinct scenarios in the store."""
        return len(self.all_scenarios())

    def timestep_count(self, scenario_id: int) -> int:
        """Number of timesteps stored for a given scenario."""
        return len(self.all_timesteps(scenario_id))

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def as_dataframe(self) -> pd.DataFrame:
        """
        Export all results as a flat DataFrame.

        Rows are sorted by (scenario_id, timestep).  Columns follow
        RESULT_COLUMNS.  Returns an empty DataFrame (with correct columns)
        if the store is empty.
        """
        if not self._results:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        rows = []
        for r in sorted(self._results.values(),
                        key=lambda x: (x.scenario_id, x.cohort_id or "", x.timestep)):
            rows.append({
                "run_id":              r.run_id,
                "scenario_id":        r.scenario_id,
                "timestep":           r.timestep,
                "cohort_id":          r.cohort_id,
                "premiums":           r.cashflows.premiums,
                "death_claims":       r.cashflows.death_claims,
                "surrender_payments": r.cashflows.surrender_payments,
                "maturity_payments":  r.cashflows.maturity_payments,
                "expenses":           r.cashflows.expenses,
                "net_outgo":          r.cashflows.net_outgo,
                "in_force_start":     r.decrements.in_force_start,
                "deaths":             r.decrements.deaths,
                "lapses":             r.decrements.lapses,
                "maturities":         r.decrements.maturities,
                "in_force_end":       r.decrements.in_force_end,
                "bel":                r.bel,
                "reserve":            r.reserve,
                # Asset fields
                "total_market_value": r.total_market_value,
                "total_book_value":   r.total_book_value,
                "cash_balance":       r.cash_balance,
                "eir_income":         r.eir_income,
                "coupon_income":      r.coupon_income,
                "dividend_income":    r.dividend_income,
                "unrealised_gl":      r.unrealised_gl,
                "realised_gl":        r.realised_gl,
                "oci_reserve":        r.oci_reserve,
                "mv_ac":              r.mv_ac,
                "mv_fvtpl":           r.mv_fvtpl,
                "mv_fvoci":           r.mv_fvoci,
                # BPA MA attribution — NaN for non-BPA runs
                "bel_pre_ma":         r.bel_pre_ma,
                "bel_post_ma":        r.bel_post_ma,
            })
        return pd.DataFrame(rows)

    def as_cohort_pivot(self, scenario_id: int = 0) -> pd.DataFrame:
        """
        Return a wide DataFrame pivoted by cohort_id for a single scenario.

        Rows     = timestep (sorted ascending, used as index during construction)
        Columns  = MultiIndex of (cohort_id, field) where field is a numeric
                   field from the stored results.

        Useful for comparing BEL trajectories and cashflows across cohorts
        in a single view.  Returns an empty DataFrame if no cohort results
        exist for the given scenario_id.

        Parameters
        ----------
        scenario_id : int, default 0
            Which scenario to pivot.  Use 0 for BPARun (deterministic scenario).
        """
        ids = self.cohort_ids()
        if not ids:
            return pd.DataFrame()

        _FIELDS = [
            "bel", "bel_pre_ma", "bel_post_ma", "net_outgo",
            "in_force_start", "deaths", "lapses", "maturities", "in_force_end",
            "total_market_value", "cash_balance",
        ]

        indexed_frames: list[pd.DataFrame] = []
        for cid in ids:
            rows = self.all_timesteps(scenario_id, cohort_id=cid)
            if not rows:
                continue
            df_cid = pd.DataFrame([{
                "timestep":           r.timestep,
                "bel":                r.bel,
                "bel_pre_ma":         r.bel_pre_ma,
                "bel_post_ma":        r.bel_post_ma,
                "net_outgo":          r.cashflows.net_outgo,
                "in_force_start":     r.decrements.in_force_start,
                "deaths":             r.decrements.deaths,
                "lapses":             r.decrements.lapses,
                "maturities":         r.decrements.maturities,
                "in_force_end":       r.decrements.in_force_end,
                "total_market_value": r.total_market_value,
                "cash_balance":       r.cash_balance,
            } for r in rows]).set_index("timestep")
            # Give columns a MultiIndex so they appear as (cohort_id, field)
            df_cid.columns = pd.MultiIndex.from_tuples(
                [(cid, col) for col in df_cid.columns]
            )
            indexed_frames.append(df_cid)

        if not indexed_frames:
            return pd.DataFrame()

        result = pd.concat(indexed_frames, axis=1).sort_index()
        result.index.name = "timestep"
        return result.reset_index()

    def summary(self) -> dict:
        """
        Aggregate summary across all stored results.

        Returns a dict with:
            run_id            : str
            scenario_count    : int
            result_count      : int
            total_premiums    : float
            total_net_outgo   : float
            total_bel         : float  (sum across all scenarios and timesteps)
            mean_bel          : float  (mean across all scenarios and timesteps)
        """
        if not self._results:
            return {
                "run_id":         self._run_id,
                "scenario_count": 0,
                "result_count":   0,
                "total_premiums": 0.0,
                "total_net_outgo": 0.0,
                "total_bel":      0.0,
                "mean_bel":       0.0,
            }
        results = list(self._results.values())
        total_premiums  = sum(r.cashflows.premiums  for r in results)
        total_net_outgo = sum(r.cashflows.net_outgo for r in results)
        total_bel       = sum(r.bel                 for r in results)
        return {
            "run_id":          self._run_id,
            "scenario_count":  self.scenario_count(),
            "result_count":    self.result_count(),
            "total_premiums":  total_premiums,
            "total_net_outgo": total_net_outgo,
            "total_bel":       total_bel,
            "mean_bel":        total_bel / len(results),
        }

"""
LiabilityOnlyRun — projects liability cash flows with no asset model.

Purpose
-------
The liability-only run is the first end-to-end projection mode.  It runs a
monthly loop over the full projection term, calling the Conventional liability
model at each timestep and writing results to ResultStore.

Used for:
    - BEL validation against independently-calculated values
    - Liability cash flow quality assurance
    - Standalone reserve runs before the full ALM loop is available

Architecture notes
------------------
Data is INJECTED into the constructor.  model_points and assumptions are
passed in as parameters — this class never reads files or databases.  The
data loader (Step 6, liability_data_loader.py) is the layer that reads from
files and then calls this constructor.  Separating reading from projecting
keeps each concern isolated and makes this class fully testable without
touching the filesystem.

Time loop
---------
The monthly loop lives in execute() for this run mode.  Since LiabilityOnlyRun
has no asset model and does not go through Fund, it does not use
engine/core/top.py.  Run modes that coordinate assets and liabilities
(DeterministicRun, StochasticRun) will use top.py when it is built (Step 8).

MP state advancement
--------------------
After storing results for timestep t, the model point DataFrame is advanced
by one calendar month before the next iteration:

    policy_duration_mths += 1  (all rows, vectorised)
    attained_age += 1          (rows where new duration is a whole year)
    accrued_bonus_per_policy   (advanced monthly for ENDOW_PAR rows only)

The original self._model_points is never mutated.  execute() works on a copy.

Stateless model design
----------------------
The Conventional model is stateless — it receives inputs, computes outputs, and
returns them without retaining anything internally.  All computed values (BEL,
cash flows, decrements) are immediately written to ResultStore.  Nothing
accumulates inside self._model between timesteps.  This is the architectural
rule: results are never stored inside models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.results.result_store import ResultStore, TimestepResult
from engine.run_modes.base_run import BaseRun, ProgressCallback


class LiabilityOnlyRun(BaseRun):
    """
    Liability-only projection: monthly loop, no asset model.

    Parameters
    ----------
    config : RunConfig
        Validated master run configuration.  run_type must be LIABILITY_ONLY.
    fund_config : FundConfig
        Validated fund configuration.
    model_points : pd.DataFrame
        Pre-loaded, validated model point DataFrame.  Must contain all columns
        required by Conventional (see Conventional.REQUIRED_COLUMNS).
        Data is injected here — this class never loads from files.
    assumptions : ConventionalAssumptions
        Pre-built assumption object including rate_curve, mortality rates,
        lapse rates, expense loadings, and surrender value factors.
    progress_callback : ProgressCallback, optional
        Called at each timestep with (fraction, message).  Fraction is in
        [0.0, 1.0].  Passes through to the Worker / UI progress panel.
    """

    def __init__(
        self,
        config:            RunConfig,
        fund_config:       FundConfig,
        model_points:      pd.DataFrame,
        assumptions:       ConventionalAssumptions,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        super().__init__(config, fund_config, progress_callback)
        self._model_points = model_points
        self._assumptions  = assumptions
        self._model: Conventional           = Conventional()
        self._store: Optional[ResultStore]  = None

    # -----------------------------------------------------------------------
    # validate_config — pre-flight check before any I/O
    # -----------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Verify run_type is LIABILITY_ONLY before any I/O starts.

        Raises
        ------
        ValueError
            If config.run_type is not RunType.LIABILITY_ONLY.
        """
        if self._config.run_type != RunType.LIABILITY_ONLY:
            raise ValueError(
                f"LiabilityOnlyRun requires run_type='liability_only', "
                f"got '{self._config.run_type.value}'."
            )

    # -----------------------------------------------------------------------
    # setup — instantiate model components
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Instantiate the Conventional liability model and ResultStore.

        Called once by run() before execute().  Both are held as instance
        attributes so execute() and teardown() can access them.
        """
        self._model = Conventional()
        self._store = ResultStore(run_id=self._config.run_id)
        self._logger.info(
            "Setup complete: model=%s, store run_id=%s",
            self._model.__class__.__name__,
            self._config.run_id,
        )

    # -----------------------------------------------------------------------
    # execute — monthly projection loop
    # -----------------------------------------------------------------------

    def execute(self) -> None:
        """
        Run the monthly projection loop.

        Two-pass design:

        Pass 1 — Forward projection (O(N) pandas operations):
            Advance model points month by month.  At each step compute
            decrements and cash flows and store them.  No BEL calculation
            here — that avoids running a nested inner projection at every
            timestep (which would be O(N²) pandas operations).

        Pass 2 — BEL backward summation (O(N²) pure-float arithmetic):
            BEL at timestep t = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
            where DF(k) is the k-month discount factor from rate_curve and
            net_outgo[t+s] is the value already stored in Pass 1.
            All operands are plain Python floats — no pandas overhead.
            For a 120-month projection this is 7,260 float multiplications,
            which runs in microseconds versus hours for the nested approach.

        Reserve = BEL (Phase 1 rule), set directly from the BEL value.

        NOTE: The time loop lives here — not in top.py — because this run mode
        has no asset model and does not go through Fund.  DeterministicRun and
        StochasticRun will delegate the loop to engine/core/top.py (Step 8).
        """
        total_months = self._config.projection.projection_term_years * 12
        mp = self._model_points.copy()

        # ------------------------------------------------------------------
        # Pass 1 — forward: collect decrements and cashflows
        # ------------------------------------------------------------------
        forward: list[tuple] = []
        for t in range(total_months):
            decrements = self._model.get_decrements(mp, self._assumptions, t)
            cashflows  = self._model.project_cashflows(mp, self._assumptions, t)
            forward.append((t, decrements, cashflows))
            mp = self._advance_model_points(mp)
            self.report_progress(
                (t + 1) / (2 * total_months),
                f"Month {t + 1}/{total_months} (forward pass)",
            )

        # ------------------------------------------------------------------
        # Pass 2 — BEL: sum future net_outgo values with discount factors
        # BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
        # ------------------------------------------------------------------
        net_outgos = [cf.net_outgo for _, _, cf in forward]
        discount_factors = [
            self._assumptions.rate_curve.discount_factor(s + 1)
            for s in range(total_months)
        ]
        bels: list[float] = [
            sum(
                net_outgos[t + s] * discount_factors[s]
                for s in range(total_months - t)
            )
            for t in range(total_months)
        ]

        # ------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------
        for (t, decrements, cashflows), bel in zip(forward, bels):
            self._store.store(TimestepResult(
                run_id=self._config.run_id,
                scenario_id=0,
                timestep=t,
                cashflows=cashflows,
                decrements=decrements,
                bel=bel,
                reserve=bel,
            ))
            self.report_progress(
                0.5 + (t + 1) / (2 * total_months),
                f"Month {t + 1}/{total_months} (BEL)",
            )

        self._logger.info(
            "Projection complete: %d months, %d results stored",
            total_months,
            self._store.result_count(),
        )

    # -----------------------------------------------------------------------
    # teardown — write output files to disk
    # -----------------------------------------------------------------------

    def teardown(self) -> None:
        """
        Write ResultStore output to disk, applying output filters.

        Filters applied in order:
          1. output_horizon_years — clip to first N years if set.
          2. output_timestep — MONTHLY keeps all rows; ANNUAL keeps year-end
             snapshots (t = 11, 23, 35, ...); QUARTERLY keeps quarter-end
             snapshots (t = 2, 5, 8, 11, ...).

        Output file:
            {output_dir}/{run_id}_liability_results.csv     (default)
            {output_dir}/{run_id}_liability_results.parquet

        The output directory is created automatically if it does not exist.
        """
        output_dir = Path(self._config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._store.as_dataframe()

        # --- Horizon filter ---
        if self._config.output.output_horizon_years is not None:
            max_t = self._config.output.output_horizon_years * 12 - 1
            df = df[df["timestep"] <= max_t].reset_index(drop=True)

        # --- Timestep filter (point-in-time snapshots) ---
        ts = self._config.output.output_timestep.value
        if ts == "annual":
            df = df[((df["timestep"] + 1) % 12) == 0].reset_index(drop=True)
        elif ts == "quarterly":
            df = df[((df["timestep"] + 1) % 3) == 0].reset_index(drop=True)
        # monthly: keep all rows unchanged

        # --- Write file ---
        stem = self._output_stem("results")
        fmt  = self._config.output.result_format.value
        if fmt == "parquet":
            path = output_dir / f"{stem}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_dir / f"{stem}.csv"
            df.to_csv(path, index=False)

        self._logger.info("Results written to %s (%d rows)", path, len(df))

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _advance_model_points(self, mp: pd.DataFrame) -> pd.DataFrame:
        """
        Advance all model point rows by one calendar month.

        Delegates to Conventional._advance_model_points, which applies the
        full decrement cascade (deaths → lapses → maturities) via
        _apply_decrements before advancing duration, age, and bonus accrual.

        Returns a new DataFrame.  The input mp is not mutated.
        """
        return self._model._advance_model_points(mp, self._assumptions)

        return mp

    # -----------------------------------------------------------------------
    # Public read-only accessors
    # -----------------------------------------------------------------------

    @property
    def store(self) -> Optional[ResultStore]:
        """The ResultStore populated by execute().  None before setup()."""
        return self._store

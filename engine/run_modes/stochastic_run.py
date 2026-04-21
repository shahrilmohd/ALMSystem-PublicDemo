"""
StochasticRun — full ALM projection over N ESG scenarios.

Purpose
-------
StochasticRun repeats the same two-pass Fund projection loop as DeterministicRun,
but runs it N times — once per scenario. The result distribution across scenarios
feeds the TVOG calculator in Step 10.

Key design points (DECISIONS.md §14)
--------------------------------------
Asset model isolation:
    The opening asset_model is deep-copied before each scenario. Every scenario
    therefore starts from the identical valuation-date portfolio state. Bonds
    are stateful objects — without isolation, scenario 2 would inherit scenario 1's
    end-of-projection book values instead of the opening ones.

Scenario selection:
    scenario_ids=None  → run every scenario in the store (full stochastic run)
    scenario_ids=[650] → run only scenario 650 (tail analysis)
    scenario_ids=[1, 500, 999] → run three specific scenarios (spot check)
    See DECISIONS.md §13 for the design rationale.

Group MP enforcement:
    RunConfig.validate_run_type_consistency() already rejects seriatim input for
    stochastic runs at config construction time. StochasticRun.validate_config()
    checks run_type only.

Two-pass BEL:
    Identical to DeterministicRun. Pass 1 collects FundTimestepResult for each
    month. Pass 2 computes BEL from discounted net_outgos. BEL discounting uses
    assumptions.rate_curve (the deterministic base curve) for Phase 1.

Parallelism:
    StochasticConfig.parallel_scenarios is read but not implemented in Phase 1.
    Scenarios run serially. The per-scenario loop is parallelisation-ready:
    each iteration is fully independent.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import numpy as np

from engine.asset.asset_model import AssetModel
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.core.fund import Fund, FundTimestepResult
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.liability.liability_state import conventional_state_from_mps
from engine.results.result_store import ResultStore, TimestepResult
from engine.run_modes.base_run import BaseRun, ProgressCallback
from engine.scenarios.scenario_store import ScenarioStore
from engine.strategy.bonus_strategy import BonusStrategy
from engine.strategy.investment_strategy import InvestmentStrategy


class StochasticRun(BaseRun):
    """
    Full ALM projection over N ESG scenarios.

    Parameters
    ----------
    config : RunConfig
        Validated master run configuration. run_type must be STOCHASTIC.
    fund_config : FundConfig
        Validated fund configuration.
    model_points : pd.DataFrame
        Pre-loaded, validated group model point DataFrame.
        Seriatim data is not permitted for stochastic runs (DECISIONS.md §9).
    assumptions : ConventionalAssumptions
        Pre-built assumption object. assumptions.rate_curve is used for BEL
        discounting (base/deterministic curve) across all scenarios (Phase 1).
    asset_model : AssetModel
        Opening portfolio at the valuation date. Deep-copied before each scenario
        so all scenarios start from identical opening state (DECISIONS.md §14).
    investment_strategy : InvestmentStrategy
        Pre-built strategy. Injected per CLAUDE.md Rule 5.
    scenario_store : ScenarioStore
        Pre-loaded collection of EsgScenarios. Each scenario supplies a full
        time-series of rate curves and equity returns (DECISIONS.md §13).
    scenario_ids : list[int] | None
        If None, run all scenarios in the store.
        If a list, run only the specified scenario IDs.
        IDs not present in the store raise KeyError at execute() time.
    initial_cash : float
        Opening cash balance at the start of the projection (£). Default: 0.0.
    progress_callback : ProgressCallback, optional
        Called at the end of each completed scenario with (fraction, message).
    bonus_strategy : BonusStrategy | None, optional
        If supplied, drives per-scenario reversionary and terminal bonus rates
        for with-profits (PAR) policies via the smoothed declared bonus
        mechanism (Step 27).  If None, all scenarios use the flat
        ``assumptions.bonus_rate_yr`` with zero terminal bonus (original behaviour).
    """

    def __init__(
        self,
        config:              RunConfig,
        fund_config:         FundConfig,
        model_points:        pd.DataFrame,
        assumptions:         ConventionalAssumptions,
        asset_model:         AssetModel,
        investment_strategy: InvestmentStrategy,
        scenario_store:      ScenarioStore,
        scenario_ids:        Optional[list[int]] = None,
        initial_cash:        float = 0.0,
        progress_callback:   Optional[ProgressCallback] = None,
        bonus_strategy:      Optional[BonusStrategy] = None,
    ) -> None:
        super().__init__(config, fund_config, progress_callback)
        self._model_points        = model_points
        self._assumptions         = assumptions
        self._asset_model         = asset_model
        self._investment_strategy = investment_strategy
        self._scenario_store      = scenario_store
        self._scenario_ids        = scenario_ids
        self._initial_cash        = initial_cash
        self._bonus_strategy      = bonus_strategy
        self._store: Optional[ResultStore] = None

    # -----------------------------------------------------------------------
    # validate_config
    # -----------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Verify run_type is STOCHASTIC.

        Group MP enforcement is already handled by RunConfig cross-field
        validation. Only run_type is checked here.

        Raises
        ------
        ValueError
            If config.run_type is not RunType.STOCHASTIC.
        """
        if self._config.run_type != RunType.STOCHASTIC:
            raise ValueError(
                f"StochasticRun requires run_type='stochastic', "
                f"got '{self._config.run_type.value}'."
            )

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Determine which scenario IDs to run and initialise ResultStore.

        Resolves scenario_ids: if None, uses all IDs from the store sorted
        ascending. Stores the resolved list as self._ids_to_run.
        """
        if self._scenario_ids is None:
            self._ids_to_run: list[int] = self._scenario_store.scenario_ids()
        else:
            self._ids_to_run = sorted(self._scenario_ids)

        self._store = ResultStore(run_id=self._config.run_id)
        self._logger.info(
            "Setup complete: %d scenarios to run, ResultStore run_id=%s",
            len(self._ids_to_run),
            self._config.run_id,
        )

    # -----------------------------------------------------------------------
    # execute — N × two-pass projection loop
    # -----------------------------------------------------------------------

    def execute(self) -> None:
        """
        Run the two-pass Fund projection loop for each selected scenario.

        Branches on ``StochasticConfig.use_vectorised`` (DECISIONS.md §32):
          - ``False`` (default): serial loop, one scenario at a time.
          - ``True``: vectorised path — liability cashflows computed for all N
            scenarios in one ``batch_step`` call per timestep; asset steps
            remain serial.
        """
        assert self._config.stochastic is not None
        assert self._store is not None
        if self._config.stochastic.use_vectorised:
            self._execute_vectorised()
        else:
            self._execute_serial()

    def _execute_serial(self) -> None:
        """
        Serial two-pass Fund projection loop (original implementation).

        For each scenario_id in self._ids_to_run:
          1. Deep-copy the opening asset_model (DECISIONS.md §14).
          2. Construct a fresh Fund with the copied portfolio.
          3. Pass 1 — Forward: advance Fund month by month using the scenario's
             time-series of AssetScenarioPoints.
          4. Pass 2 — Backward: compute BEL from discounted net_outgos.
          5. Store TimestepResults with scenario_id set to the ESG scenario ID.
          6. Report progress.
        """
        assert self._store is not None
        total_months  = self._config.projection.projection_term_years * 12
        n_scenarios   = len(self._ids_to_run)

        # Pre-compute discount factors once (same base curve for all scenarios)
        discount_factors = [
            self._assumptions.rate_curve.discount_factor(s + 1)
            for s in range(total_months)
        ]

        for completed, scenario_id in enumerate(self._ids_to_run):
            esg_scenario = self._scenario_store.get(scenario_id)

            # ------------------------------------------------------------------
            # Validate scenario has enough timesteps
            # ------------------------------------------------------------------
            if esg_scenario.n_months < total_months:
                raise ValueError(
                    f"Scenario {scenario_id} has {esg_scenario.n_months} timesteps "
                    f"but projection requires {total_months}."
                )

            # ------------------------------------------------------------------
            # Isolate asset state: fresh opening portfolio for this scenario
            # ------------------------------------------------------------------
            asset_model_copy = copy.deepcopy(self._asset_model)
            liability_model  = Conventional()
            fund = Fund(
                asset_model=asset_model_copy,
                liability_model=liability_model,
                investment_strategy=self._investment_strategy,
                initial_cash=self._initial_cash,
            )

            mp = self._model_points.copy()

            # ------------------------------------------------------------------
            # Pass 1 — Forward: advance Fund using scenario's rate path
            # ------------------------------------------------------------------
            forward: list[FundTimestepResult] = []

            # Bonus strategy path: per-scenario vectorised state (n=1).
            # Smoothed return EMA is reset per scenario (independent paths).
            if self._bonus_strategy is not None:
                bs_states = conventional_state_from_mps(mp, n_scenarios=1)
                smoothed_return = np.zeros(1)

            for t in range(total_months):
                if self._bonus_strategy is not None:
                    earned_ret_yr      = np.array([esg_scenario.get_timestep(t).equity_total_return_yr])
                    earned_ret_monthly = (1.0 + earned_ret_yr) ** (1.0 / 12.0) - 1.0
                    smoothed_return    = self._bonus_strategy.update_smoothed_returns(
                        smoothed_return, earned_ret_yr
                    )
                    guaranteed_benefits = (
                        mp["sum_assured"].to_numpy(dtype=float)
                        + np.asarray(bs_states.accrued_bonus)
                    )
                    bonus_rates          = self._bonus_strategy.declare_reversionary(smoothed_return)
                    terminal_bonus_rates = self._bonus_strategy.compute_terminal_bonus_rate(
                        np.asarray(bs_states.asset_share), guaranteed_benefits
                    )
                    bs_states, step_cashflows, step_decrements = liability_model.batch_step(
                        bs_states, mp, bonus_rates, self._assumptions, t,
                        terminal_bonus_rates=terminal_bonus_rates,
                        earned_returns_monthly=earned_ret_monthly,
                    )
                    scenario_point = esg_scenario.get_timestep(t)
                    result = fund.step_time_with_liability(
                        scenario_point, step_cashflows[0], step_decrements[0]
                    )
                    mp = self._advance_model_points_no_bonus(mp)
                else:
                    scenario_point = esg_scenario.get_timestep(t)
                    result = fund.step_time(scenario_point, mp, self._assumptions)
                    mp = self._advance_model_points(mp)

                forward.append(result)

            # ------------------------------------------------------------------
            # Pass 2 — BEL: backward summation of discounted future net_outgos
            # BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
            # ------------------------------------------------------------------
            net_outgos = [r.cashflows.net_outgo for r in forward]
            bels: list[float] = [
                sum(
                    net_outgos[t + s] * discount_factors[s]
                    for s in range(total_months - t)
                )
                for t in range(total_months)
            ]

            # ------------------------------------------------------------------
            # Store results for this scenario
            # ------------------------------------------------------------------
            for t, (fund_result, bel) in enumerate(zip(forward, bels)):
                ar = fund_result.asset
                self._store.store(TimestepResult(
                    run_id=self._config.run_id,
                    scenario_id=scenario_id,
                    timestep=t,
                    cashflows=fund_result.cashflows,
                    decrements=fund_result.decrements,
                    bel=bel,
                    reserve=bel,
                    total_market_value=ar.total_market_value,
                    total_book_value=ar.total_book_value,
                    cash_balance=ar.cash_balance,
                    eir_income=ar.eir_income,
                    coupon_income=ar.coupon_income,
                    dividend_income=ar.dividend_income,
                    unrealised_gl=ar.unrealised_gl,
                    realised_gl=ar.realised_gl,
                    oci_reserve=ar.oci_reserve,
                    mv_ac=ar.mv_ac,
                    mv_fvtpl=ar.mv_fvtpl,
                    mv_fvoci=ar.mv_fvoci,
                ))

            self.report_progress(
                0.05 + 0.90 * (completed + 1) / n_scenarios,
                f"Scenario {scenario_id} complete ({completed + 1}/{n_scenarios})",
            )
            self._logger.debug(
                "Scenario %d complete (%d/%d)", scenario_id, completed + 1, n_scenarios
            )

        self._logger.info(
            "Stochastic projection complete: %d scenarios, %d total results stored",
            n_scenarios,
            self._store.result_count(),
        )

    def _execute_vectorised(self) -> None:
        """
        Vectorised two-pass projection loop (DECISIONS.md §32, Step 5).

        Differences from ``_execute_serial``:
          - Liability cashflows and decrements for ALL N scenarios are computed
            in one ``liability.batch_step()`` call per timestep instead of N
            separate ``project_cashflows`` calls.
          - Asset steps remain serial (assets are stateful; vectorising them
            is deferred to Phase 4).
          - ``Fund.step_time_with_liability`` is used to feed pre-computed
            liability results into the fund without recomputing them.

        Produces numerically identical results to ``_execute_serial`` for the
        conventional product with a flat bonus rate.  When scenario-specific
        bonus rates are introduced (Phase 4), this path will also correctly
        handle diverging accrued bonuses across scenarios.
        """
        assert self._store is not None
        total_months = self._config.projection.projection_term_years * 12
        n_scenarios  = len(self._ids_to_run)

        # Validate all scenarios have enough timesteps upfront
        for scenario_id in self._ids_to_run:
            esg = self._scenario_store.get(scenario_id)
            if esg.n_months < total_months:
                raise ValueError(
                    f"Scenario {scenario_id} has {esg.n_months} timesteps "
                    f"but projection requires {total_months}."
                )

        # Pre-compute discount factors (base curve, same for all scenarios)
        discount_factors = [
            self._assumptions.rate_curve.discount_factor(s + 1)
            for s in range(total_months)
        ]

        # Smoothed return EMA state — maintained here (outside JAX) because it
        # is fund-level time-series state, not policy-level group state.
        # See DECISIONS.md §64 for rationale.
        smoothed_returns = np.zeros(n_scenarios)

        # Build initial vectorised liability state
        liability_model = Conventional()
        states = conventional_state_from_mps(self._model_points, n_scenarios)

        # Isolate one asset Fund per scenario (same as serial path)
        funds: list[Fund] = [
            Fund(
                asset_model=copy.deepcopy(self._asset_model),
                liability_model=Conventional(),
                investment_strategy=self._investment_strategy,
                initial_cash=self._initial_cash,
            )
            for _ in range(n_scenarios)
        ]

        # Storage: forward pass results indexed [scenario_index][timestep]
        all_cashflows:  list[list] = [[] for _ in range(n_scenarios)]
        all_decrements: list[list] = [[] for _ in range(n_scenarios)]
        all_asset:      list[list] = [[] for _ in range(n_scenarios)]

        mp = self._model_points.copy()  # single model points (deterministic, shared)

        # ------------------------------------------------------------------
        # Pass 1 — Forward (vectorised liability, serial asset)
        # ------------------------------------------------------------------
        for t in range(total_months):
            # Gather per-scenario annual equity returns for this timestep
            earned_rets_yr = np.array([
                self._scenario_store.get(sid).get_timestep(t).equity_total_return_yr
                for sid in self._ids_to_run
            ])
            earned_rets_monthly = (1.0 + earned_rets_yr) ** (1.0 / 12.0) - 1.0

            if self._bonus_strategy is not None:
                # Update smoothed return EMA (encapsulated in strategy)
                smoothed_returns = self._bonus_strategy.update_smoothed_returns(
                    smoothed_returns, earned_rets_yr
                )
                # Per-scenario guaranteed benefit: (n_scenarios, n_groups).
                # Each scenario tracks its own accrued bonus history independently.
                guaranteed_benefits = (
                    mp["sum_assured"].to_numpy(dtype=float)
                    + np.asarray(states.accrued_bonus)
                )
                bonus_rates = self._bonus_strategy.declare_reversionary(smoothed_returns)
                terminal_bonus_rates = self._bonus_strategy.compute_terminal_bonus_rate(
                    np.asarray(states.asset_share), guaranteed_benefits
                )
            else:
                bonus_rates = np.full(n_scenarios, self._assumptions.bonus_rate_yr)
                terminal_bonus_rates = np.zeros(n_scenarios)

            # One batch_step call computes cashflows for all N scenarios
            states, step_cashflows, step_decrements = liability_model.batch_step(
                states, mp, bonus_rates, self._assumptions, t,
                terminal_bonus_rates=terminal_bonus_rates,
                earned_returns_monthly=earned_rets_monthly,
            )

            # Asset step still serial: each Fund is stateful
            for i, scenario_id in enumerate(self._ids_to_run):
                scenario_point = self._scenario_store.get(scenario_id).get_timestep(t)
                fund_result = funds[i].step_time_with_liability(
                    scenario_point,
                    step_cashflows[i],
                    step_decrements[i],
                )
                all_cashflows[i].append(step_cashflows[i])
                all_decrements[i].append(step_decrements[i])
                all_asset[i].append(fund_result.asset)

            # Advance model points once (deterministic — same for all scenarios)
            mp = self._advance_model_points(mp)

        # ------------------------------------------------------------------
        # Pass 2 — BEL backward summation (identical to serial path)
        # ------------------------------------------------------------------
        for completed, (i, scenario_id) in enumerate(enumerate(self._ids_to_run)):
            net_outgos = [cf.net_outgo for cf in all_cashflows[i]]
            bels: list[float] = [
                sum(
                    net_outgos[t + s] * discount_factors[s]
                    for s in range(total_months - t)
                )
                for t in range(total_months)
            ]

            for t, (cashflows, decrements, asset, bel) in enumerate(
                zip(all_cashflows[i], all_decrements[i], all_asset[i], bels)
            ):
                self._store.store(TimestepResult(
                    run_id=self._config.run_id,
                    scenario_id=scenario_id,
                    timestep=t,
                    cashflows=cashflows,
                    decrements=decrements,
                    bel=bel,
                    reserve=bel,
                    total_market_value=asset.total_market_value,
                    total_book_value=asset.total_book_value,
                    cash_balance=asset.cash_balance,
                    eir_income=asset.eir_income,
                    coupon_income=asset.coupon_income,
                    dividend_income=asset.dividend_income,
                    unrealised_gl=asset.unrealised_gl,
                    realised_gl=asset.realised_gl,
                    oci_reserve=asset.oci_reserve,
                    mv_ac=asset.mv_ac,
                    mv_fvtpl=asset.mv_fvtpl,
                    mv_fvoci=asset.mv_fvoci,
                ))

            self.report_progress(
                0.05 + 0.90 * (completed + 1) / n_scenarios,
                f"Scenario {scenario_id} complete ({completed + 1}/{n_scenarios})",
            )

        self._logger.info(
            "Vectorised projection complete: %d scenarios, %d total results stored",
            n_scenarios,
            self._store.result_count(),
        )

    # -----------------------------------------------------------------------
    # teardown — write outputs to disk
    # -----------------------------------------------------------------------

    def teardown(self) -> None:
        """
        Write ResultStore output to disk with output filters applied.

        Applies the same horizon and timestep filters as DeterministicRun.
        For stochastic runs, Parquet is recommended (result_format=parquet in
        RunConfig) as output volume is N × T rows.

        Output file:
            {output_dir}/{run_id}_stochastic_results.csv
            {output_dir}/{run_id}_stochastic_results.parquet
        """
        assert self._store is not None
        output_dir = Path(self._config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._store.as_dataframe()

        # Horizon filter
        if self._config.output.output_horizon_years is not None:
            max_t = self._config.output.output_horizon_years * 12 - 1
            df = df[df["timestep"] <= max_t].reset_index(drop=True)

        # Timestep filter
        ts = self._config.output.output_timestep.value
        if ts == "annual":
            df = df[((df["timestep"] + 1) % 12) == 0].reset_index(drop=True)
        elif ts == "quarterly":
            df = df[((df["timestep"] + 1) % 3) == 0].reset_index(drop=True)

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
    # Public read-only accessors
    # -----------------------------------------------------------------------

    @property
    def store(self) -> Optional[ResultStore]:
        """The ResultStore populated by execute(). None before setup()."""
        return self._store

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _advance_model_points(self, mp: pd.DataFrame) -> pd.DataFrame:
        """
        Advance all model point rows by one calendar month.

        Identical to DeterministicRun._advance_model_points(). Duplicated
        intentionally — stochastic and deterministic run modes may diverge
        in future (e.g. unit-linked MPs need different advancement logic).
        """
        mp = mp.copy()

        mp["policy_duration_mths"] = mp["policy_duration_mths"] + 1

        remaining_term = mp["policy_term_yr"] * 12 - mp["policy_duration_mths"]
        mp.loc[remaining_term <= 0, "in_force_count"] = 0.0

        anniversary_mask = (mp["policy_duration_mths"] % 12) == 0
        mp.loc[anniversary_mask, "attained_age"] = (
            mp.loc[anniversary_mask, "attained_age"] + 1
        )

        par_mask = mp["policy_code"] == "ENDOW_PAR"
        if par_mask.any():
            mp.loc[par_mask, "accrued_bonus_per_policy"] = (
                mp.loc[par_mask, "accrued_bonus_per_policy"]
                + self._assumptions.bonus_rate_yr
                * mp.loc[par_mask, "sum_assured"]
                / 12.0
            )

        return mp

    def _advance_model_points_no_bonus(self, mp: pd.DataFrame) -> pd.DataFrame:
        """Advance MP by one month: duration, age, in_force — but NOT PAR bonus accrual.

        Used when batch_step owns accrued_bonus updates to prevent double-counting.
        """
        mp = mp.copy()

        mp["policy_duration_mths"] = mp["policy_duration_mths"] + 1

        remaining_term = mp["policy_term_yr"] * 12 - mp["policy_duration_mths"]
        mp.loc[remaining_term <= 0, "in_force_count"] = 0.0

        anniversary_mask = (mp["policy_duration_mths"] % 12) == 0
        mp.loc[anniversary_mask, "attained_age"] = (
            mp.loc[anniversary_mask, "attained_age"] + 1
        )

        return mp

"""
Unit tests for StochasticRun.

Rules under test
----------------
validate_config:
  1.  Passes silently for run_type = STOCHASTIC.
  2.  Raises ValueError for run_type = DETERMINISTIC.
  3.  Raises ValueError for run_type = LIABILITY_ONLY.

setup:
  4.  self._store is a ResultStore instance after setup().
  5.  store.run_id matches config.run_id.
  6.  scenario_ids=None → _ids_to_run contains all IDs from the store, sorted.
  7.  scenario_ids=[2, 1] → _ids_to_run is [1, 2] (sorted).

execute:
  8.  Result count equals n_scenarios × projection_term_months.
  9.  Results for scenario 1 all have scenario_id = 1.
  10. Results for scenario 2 all have scenario_id = 2.
  11. Timesteps within each scenario are 0, 1, …, T-1.
  12. BEL > 0 for a non-trivial liability.
  13. Asset columns (total_market_value, cash_balance) are populated.
  14. Original model_points DataFrame is not mutated during execute().
  15. Running scenario_ids=[2] produces results only for scenario 2.
  16. Requesting a scenario_id absent from the store raises KeyError at execute().
  17. Scenario with too few timesteps raises ValueError at execute().
  18. Each scenario starts with the same opening asset state (deep copy isolation).

teardown:
  19. CSV file is written to output_dir.
  20. Parquet file is written when result_format = PARQUET.
  21. Output file contains the correct column names.
  22. Annual output_timestep writes only year-end rows.

progress callback:
  23. Callback is called at least once per completed scenario.
  24. Fractions are non-decreasing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.config.fund_config import AssetClassWeights, FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.results.result_store import ResultStore
from engine.run_modes.stochastic_run import StochasticRun
from engine.scenarios.scenario_engine import ScenarioLoader
from engine.scenarios.scenario_store import ScenarioStore
from engine.strategy.investment_strategy import InvestmentStrategy
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECTION_TERM_YEARS = 1   # 12 months — fast tests
N_SCENARIOS           = 2   # small stochastic run


def make_stoch_config(
    tmp_path: Path,
    projection_term_years: int = PROJECTION_TERM_YEARS,
    result_format: str = "csv",
    output_timestep: str = "monthly",
) -> RunConfig:
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    asset_file       = tmp_path / "assets.csv"
    asset_file.write_text("placeholder\n")
    scenario_file    = tmp_path / "scenarios.csv"
    scenario_file.write_text("scenario_id\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        run_type="stochastic",
        input_mode="group_mp",
        projection_term_years=projection_term_years,
        output_timestep=output_timestep,
        asset_data_path=asset_file,
        scenario_file_path=scenario_file,
        stochastic={"num_scenarios": N_SCENARIOS},
    )
    data["output"]["output_dir"]    = str(tmp_path / "outputs")
    data["output"]["result_format"] = result_format
    return RunConfig.from_dict(data)


def make_fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id":   "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 1.0, "equities": 0.0, "derivatives": 0.0, "cash": 0.0},
        "crediting_groups": [
            {"group_id": "GRP_A", "group_name": "Group A", "product_codes": ["P1"]},
        ],
    })


def make_assumptions(rate: float = 0.03) -> ConventionalAssumptions:
    return ConventionalAssumptions(
        mortality_rates={},
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(rate),
    )


def make_model_points() -> pd.DataFrame:
    """100 ENDOW_NONPAR policies, 1-year term just started."""
    return pd.DataFrame([{
        "group_id":                "GRP_A",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          1,
        "policy_duration_mths":    0,
        "accrued_bonus_per_policy": 0.0,
    }])


def make_asset_model(face: float = 1_000_000.0, coupon: float = 0.05) -> AssetModel:
    bond = Bond("corp_1", face, coupon, 36, "FVTPL", face)
    return AssetModel([bond])


def make_no_rebalance_strategy() -> InvestmentStrategy:
    weights = AssetClassWeights(bonds=1.0, equities=0.0, derivatives=0.0, cash=0.0)
    return InvestmentStrategy(weights, rebalancing_tolerance=1.0)


def make_scenario_store(n_scenarios: int = N_SCENARIOS, n_months: int = 12) -> ScenarioStore:
    """Flat 3% scenarios — all identical, N scenarios, n_months each."""
    return ScenarioLoader.flat(
        n_scenarios=n_scenarios,
        rate=0.03,
        equity_return_yr=0.07,
        n_months=n_months,
    )


def build_run(
    tmp_path: Path,
    projection_term_years: int = PROJECTION_TERM_YEARS,
    n_scenarios: int = N_SCENARIOS,
    scenario_ids: list[int] | None = None,
    result_format: str = "csv",
    output_timestep: str = "monthly",
    initial_cash: float = 0.0,
    progress_callback=None,
) -> StochasticRun:
    n_months = projection_term_years * 12
    return StochasticRun(
        config=make_stoch_config(tmp_path, projection_term_years, result_format, output_timestep),
        fund_config=make_fund_config(),
        model_points=make_model_points(),
        assumptions=make_assumptions(rate=0.03),
        asset_model=make_asset_model(),
        investment_strategy=make_no_rebalance_strategy(),
        scenario_store=make_scenario_store(n_scenarios=n_scenarios, n_months=n_months),
        scenario_ids=scenario_ids,
        initial_cash=initial_cash,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# 1-3: validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:

    def test_passes_for_stochastic(self, tmp_path):
        run = build_run(tmp_path)
        run.validate_config()   # no exception

    def test_raises_for_deterministic(self, tmp_path):
        run = build_run(tmp_path)
        run._config.__dict__["run_type"] = RunType.DETERMINISTIC
        with pytest.raises(ValueError, match="stochastic"):
            run.validate_config()

    def test_raises_for_liability_only(self, tmp_path):
        run = build_run(tmp_path)
        run._config.__dict__["run_type"] = RunType.LIABILITY_ONLY
        with pytest.raises(ValueError, match="stochastic"):
            run.validate_config()


# ---------------------------------------------------------------------------
# 4-7: setup
# ---------------------------------------------------------------------------

class TestSetup:

    def test_store_created_after_setup(self, tmp_path):
        run = build_run(tmp_path)
        run.validate_config()
        run.setup()
        assert isinstance(run.store, ResultStore)

    def test_store_run_id_matches_config(self, tmp_path):
        run = build_run(tmp_path)
        run.validate_config()
        run.setup()
        assert run.store.run_id == run.config.run_id

    def test_scenario_ids_none_uses_all_sorted(self, tmp_path):
        """None → run all IDs from store in ascending order."""
        run = build_run(tmp_path, n_scenarios=3, scenario_ids=None)
        run.validate_config()
        run.setup()
        assert run._ids_to_run == [1, 2, 3]

    def test_scenario_ids_list_sorted(self, tmp_path):
        """Explicit list [2, 1] → _ids_to_run sorted to [1, 2]."""
        run = build_run(tmp_path, n_scenarios=2, scenario_ids=[2, 1])
        run.validate_config()
        run.setup()
        assert run._ids_to_run == [1, 2]


# ---------------------------------------------------------------------------
# 8-18: execute
# ---------------------------------------------------------------------------

class TestExecute:

    def _full_run(
        self,
        tmp_path,
        term_years=PROJECTION_TERM_YEARS,
        n_scenarios=N_SCENARIOS,
        scenario_ids=None,
    ) -> StochasticRun:
        run = build_run(
            tmp_path,
            projection_term_years=term_years,
            n_scenarios=n_scenarios,
            scenario_ids=scenario_ids,
        )
        run.run()
        return run

    def test_result_count_equals_scenarios_times_months(self, tmp_path):
        run = self._full_run(tmp_path)
        assert run.store.result_count() == N_SCENARIOS * PROJECTION_TERM_YEARS * 12

    def test_scenario_1_results_have_correct_scenario_id(self, tmp_path):
        run = self._full_run(tmp_path)
        for r in run.store.all_timesteps(1):
            assert r.scenario_id == 1

    def test_scenario_2_results_have_correct_scenario_id(self, tmp_path):
        run = self._full_run(tmp_path)
        for r in run.store.all_timesteps(2):
            assert r.scenario_id == 2

    def test_timesteps_ascending_within_scenario(self, tmp_path):
        run = self._full_run(tmp_path)
        ts = [r.timestep for r in run.store.all_timesteps(1)]
        assert ts == list(range(PROJECTION_TERM_YEARS * 12))

    def test_bel_positive_for_nontrivial_liability(self, tmp_path):
        run = self._full_run(tmp_path)
        bel_t0 = run.store.get(1, 0).bel
        assert bel_t0 > 0.0

    def test_asset_columns_populated(self, tmp_path):
        run = self._full_run(tmp_path)
        r   = run.store.get(1, 0)
        assert r.total_market_value is not None
        assert r.cash_balance is not None
        assert r.coupon_income is not None

    def test_model_points_not_mutated(self, tmp_path):
        original_mp       = make_model_points()
        original_duration = original_mp["policy_duration_mths"].iloc[0]
        run = StochasticRun(
            config=make_stoch_config(tmp_path),
            fund_config=make_fund_config(),
            model_points=original_mp,
            assumptions=make_assumptions(rate=0.03),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
            scenario_store=make_scenario_store(),
        )
        run.run()
        assert original_mp["policy_duration_mths"].iloc[0] == original_duration

    def test_scenario_ids_filter_limits_results(self, tmp_path):
        """Run only scenario 2: result store has results for scenario 2 only."""
        run = self._full_run(tmp_path, scenario_ids=[2])
        assert run.store.result_count() == PROJECTION_TERM_YEARS * 12
        # Only scenario_id=2 in the store
        df = run.store.as_dataframe()
        assert set(df["scenario_id"].unique()) == {2}

    def test_unknown_scenario_id_raises_keyerror(self, tmp_path):
        """Scenario 999 not in store → KeyError at execute time."""
        run = build_run(tmp_path, scenario_ids=[999])
        run.validate_config()
        run.setup()
        with pytest.raises(KeyError):
            run.execute()

    def test_insufficient_scenario_timesteps_raises(self, tmp_path):
        """Scenario has 6 months but projection needs 12 → ValueError."""
        run = StochasticRun(
            config=make_stoch_config(tmp_path, projection_term_years=1),
            fund_config=make_fund_config(),
            model_points=make_model_points(),
            assumptions=make_assumptions(rate=0.03),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
            scenario_store=make_scenario_store(n_scenarios=1, n_months=6),
        )
        run.validate_config()
        run.setup()
        with pytest.raises(ValueError, match="timesteps"):
            run.execute()

    def test_deep_copy_isolation(self, tmp_path):
        """
        Each scenario should start from the same opening asset state.

        We run 2 scenarios with the same flat rate path but 3-year bonds.
        After a full 1-year projection the bond book values are updated.
        Deep copy means scenario 2 starts from the *original* opening BV,
        not from the end-state of scenario 1.

        We verify this indirectly: the total_book_value reported at t=0 is
        the same for both scenarios (both started from the same opening portfolio).
        """
        run = self._full_run(tmp_path, n_scenarios=2)
        bv_scen1_t0 = run.store.get(1, 0).total_book_value
        bv_scen2_t0 = run.store.get(2, 0).total_book_value
        assert bv_scen1_t0 == pytest.approx(bv_scen2_t0)


# ---------------------------------------------------------------------------
# 19-22: teardown
# ---------------------------------------------------------------------------

class TestTeardown:

    def test_csv_file_written(self, tmp_path):
        run = build_run(tmp_path, result_format="csv")
        run.run()
        output_dir = tmp_path / "outputs"
        assert len(list(output_dir.glob("*.csv"))) == 1

    def test_parquet_file_written(self, tmp_path):
        run = build_run(tmp_path, result_format="parquet")
        run.run()
        output_dir = tmp_path / "outputs"
        assert len(list(output_dir.glob("*.parquet"))) == 1

    def test_output_has_result_columns(self, tmp_path):
        run = build_run(tmp_path)
        run.run()
        output_dir = tmp_path / "outputs"
        csv_file   = list(output_dir.glob("*.csv"))[0]
        df = pd.read_csv(csv_file)
        for col in ["run_id", "scenario_id", "timestep", "bel", "total_market_value"]:
            assert col in df.columns

    def test_annual_output_has_year_end_rows_only(self, tmp_path):
        """For a 1-year projection with 2 scenarios, annual → 2 rows (t=11 each)."""
        run = build_run(tmp_path, output_timestep="annual")
        run.run()
        output_dir = tmp_path / "outputs"
        csv_file   = list(output_dir.glob("*.csv"))[0]
        df = pd.read_csv(csv_file)
        assert len(df) == N_SCENARIOS
        assert (df["timestep"] == 11).all()


# ---------------------------------------------------------------------------
# 23-24: progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_callback_called_at_least_once_per_scenario(self, tmp_path):
        calls: List[Tuple[float, str]] = []
        run = build_run(
            tmp_path,
            n_scenarios=N_SCENARIOS,
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        # At minimum one call per scenario from execute() plus one from setup()
        assert len(calls) >= N_SCENARIOS

    def test_fractions_non_decreasing(self, tmp_path):
        fractions: List[float] = []
        run = build_run(
            tmp_path,
            n_scenarios=N_SCENARIOS,
            progress_callback=lambda f, m: fractions.append(f),
        )
        run.run()
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1] - 1e-9

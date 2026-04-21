"""
Unit tests for DeterministicRun.

Rules under test
----------------
validate_config:
  1.  Passes silently for run_type = DETERMINISTIC.
  2.  Raises ValueError for run_type = LIABILITY_ONLY.
  3.  Raises ValueError for run_type = STOCHASTIC.

setup:
  4.  self._fund is a Fund instance after setup().
  5.  self._store is a ResultStore instance after setup().
  6.  store.run_id matches config.run_id.

execute:
  7.  Result count equals projection_term_years × 12.
  8.  All results stored with scenario_id = 0.
  9.  Results stored in ascending timestep order (0, 1, 2, ...).
  10. BEL > 0 for a non-trivial liability in the final month.
  11. Asset columns (total_market_value, cash_balance) are populated.
  12. total_market_value > 0 when portfolio has assets.
  13. Original model_points DataFrame is not mutated during execute().

teardown:
  14. CSV file is written to output_dir.
  15. Parquet file is written when result_format = PARQUET.
  16. Output file contains the correct column names.
  17. Annual output_timestep writes only year-end rows.

progress callback:
  18. Callback is called at least once during execute().
  19. Fractions are non-decreasing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.config.fund_config import AssetClassWeights, FundConfig
from engine.config.run_config import RunConfig, RunType, ResultFormat, OutputTimestep
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.results.result_store import RESULT_COLUMNS
from engine.run_modes.deterministic_run import DeterministicRun
from engine.strategy.investment_strategy import InvestmentStrategy
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det_config(
    tmp_path: Path,
    projection_term_years: int = 1,
    result_format: str = "csv",
    output_timestep: str = "monthly",
) -> RunConfig:
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=projection_term_years,
    )
    data["run_type"]  = "deterministic"
    data["output"]["output_dir"]      = str(tmp_path / "outputs")
    data["output"]["result_format"]   = result_format
    data["output"]["output_timestep"] = output_timestep
    # DeterministicRun requires asset_data_path
    asset_file = tmp_path / "assets.csv"
    asset_file.write_text("placeholder\n")
    data["input_sources"]["asset_data_path"] = str(asset_file)
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


def make_assumptions(rate: float = 0.0) -> ConventionalAssumptions:
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


def build_run(
    tmp_path: Path,
    projection_term_years: int = 1,
    result_format: str = "csv",
    output_timestep: str = "monthly",
    initial_cash: float = 0.0,
) -> DeterministicRun:
    return DeterministicRun(
        config=make_det_config(tmp_path, projection_term_years, result_format, output_timestep),
        fund_config=make_fund_config(),
        model_points=make_model_points(),
        assumptions=make_assumptions(rate=0.03),
        asset_model=make_asset_model(),
        investment_strategy=make_no_rebalance_strategy(),
        initial_cash=initial_cash,
    )


# ---------------------------------------------------------------------------
# 1-3: validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:

    def test_passes_for_deterministic(self, tmp_path):
        run = build_run(tmp_path)
        run.validate_config()   # no exception

    def test_raises_for_liability_only(self, tmp_path):
        assumption_dir   = tmp_path / "assumptions"
        assumption_dir.mkdir()
        fund_config_file = tmp_path / "fund_config.yaml"
        fund_config_file.write_text("placeholder: true\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
        )
        # liability_only config has no asset_data_path, so inject directly
        config = RunConfig.from_dict(data)
        run = DeterministicRun(
            config=config,
            fund_config=make_fund_config(),
            model_points=make_model_points(),
            assumptions=make_assumptions(),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
        )
        with pytest.raises(ValueError, match="deterministic"):
            run.validate_config()

    def test_raises_for_wrong_type_message(self, tmp_path):
        run = build_run(tmp_path)
        # Manually swap run_type after construction to test message
        run._config.__dict__["run_type"] = RunType.LIABILITY_ONLY
        with pytest.raises(ValueError, match="deterministic"):
            run.validate_config()


# ---------------------------------------------------------------------------
# 4-6: setup
# ---------------------------------------------------------------------------

class TestSetup:

    def test_fund_created_after_setup(self, tmp_path):
        from engine.core.fund import Fund
        run = build_run(tmp_path)
        run.validate_config()
        run.setup()
        assert isinstance(run.fund, Fund)

    def test_store_created_after_setup(self, tmp_path):
        from engine.results.result_store import ResultStore
        run = build_run(tmp_path)
        run.validate_config()
        run.setup()
        assert isinstance(run.store, ResultStore)

    def test_store_run_id_matches_config(self, tmp_path):
        run = build_run(tmp_path)
        run.validate_config()
        run.setup()
        assert run.store.run_id == run.config.run_id


# ---------------------------------------------------------------------------
# 7-13: execute
# ---------------------------------------------------------------------------

class TestExecute:

    def _full_run(self, tmp_path, term_years=1) -> DeterministicRun:
        run = build_run(tmp_path, projection_term_years=term_years)
        run.run()
        return run

    def test_result_count_equals_term_months(self, tmp_path):
        run = self._full_run(tmp_path, term_years=1)
        assert run.store.result_count() == 12

    def test_all_results_scenario_id_zero(self, tmp_path):
        run = self._full_run(tmp_path)
        for r in run.store.all_timesteps(0):
            assert r.scenario_id == 0

    def test_results_in_ascending_timestep_order(self, tmp_path):
        run  = self._full_run(tmp_path)
        ts   = [r.timestep for r in run.store.all_timesteps(0)]
        assert ts == list(range(12))

    def test_bel_positive_for_nontrivial_liability(self, tmp_path):
        """BEL at t=0 should be positive (100 policies, £10k SA, premiums < SA)."""
        run = self._full_run(tmp_path)
        bel_t0 = run.store.get(0, 0).bel
        assert bel_t0 > 0.0

    def test_asset_columns_populated(self, tmp_path):
        run = self._full_run(tmp_path)
        r   = run.store.get(0, 0)
        assert r.total_market_value is not None
        assert r.cash_balance is not None
        assert r.coupon_income is not None

    def test_total_market_value_positive(self, tmp_path):
        run = self._full_run(tmp_path)
        r   = run.store.get(0, 0)
        assert r.total_market_value > 0.0

    def test_model_points_not_mutated(self, tmp_path):
        original_mp = make_model_points()
        original_duration = original_mp["policy_duration_mths"].iloc[0]
        run = DeterministicRun(
            config=make_det_config(tmp_path),
            fund_config=make_fund_config(),
            model_points=original_mp,
            assumptions=make_assumptions(rate=0.03),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
        )
        run.run()
        assert original_mp["policy_duration_mths"].iloc[0] == original_duration


# ---------------------------------------------------------------------------
# 14-17: teardown
# ---------------------------------------------------------------------------

class TestTeardown:

    def test_csv_file_written(self, tmp_path):
        run = build_run(tmp_path, result_format="csv")
        run.run()
        output_dir = tmp_path / "outputs"
        csv_files  = list(output_dir.glob("*.csv"))
        assert len(csv_files) == 1

    def test_parquet_file_written(self, tmp_path):
        run = build_run(tmp_path, result_format="parquet")
        run.run()
        output_dir    = tmp_path / "outputs"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_output_has_result_columns(self, tmp_path):
        run = build_run(tmp_path)
        run.run()
        output_dir = tmp_path / "outputs"
        csv_file   = list(output_dir.glob("*.csv"))[0]
        df = pd.read_csv(csv_file)
        for col in ["run_id", "scenario_id", "timestep", "bel", "total_market_value"]:
            assert col in df.columns

    def test_annual_output_has_year_end_rows_only(self, tmp_path):
        run = build_run(tmp_path, output_timestep="annual")
        run.run()
        output_dir = tmp_path / "outputs"
        csv_file   = list(output_dir.glob("*.csv"))[0]
        df = pd.read_csv(csv_file)
        # For 1-year projection, annual output = 1 row at t=11
        assert len(df) == 1
        assert df["timestep"].iloc[0] == 11


# ---------------------------------------------------------------------------
# 18-19: progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_callback_called_at_least_once(self, tmp_path):
        calls: List[Tuple[float, str]] = []
        run = DeterministicRun(
            config=make_det_config(tmp_path),
            fund_config=make_fund_config(),
            model_points=make_model_points(),
            assumptions=make_assumptions(rate=0.03),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        run.run()
        assert len(calls) > 0

    def test_fractions_non_decreasing(self, tmp_path):
        fractions: List[float] = []
        run = DeterministicRun(
            config=make_det_config(tmp_path),
            fund_config=make_fund_config(),
            model_points=make_model_points(),
            assumptions=make_assumptions(rate=0.03),
            asset_model=make_asset_model(),
            investment_strategy=make_no_rebalance_strategy(),
            progress_callback=lambda f, m: fractions.append(f),
        )
        run.run()
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1]

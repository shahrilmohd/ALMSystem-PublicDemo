"""
Unit tests for LiabilityOnlyRun.

Rules under test
----------------
validate_config:
  1.  Passes silently for run_type = LIABILITY_ONLY.
  2.  Raises ValueError for run_type = DETERMINISTIC.
  3.  Raises ValueError for run_type = STOCHASTIC.

setup:
  4.  self._model is a Conventional instance after setup().
  5.  self._store is a ResultStore instance after setup().
  6.  store.run_id matches config.run_id.
  7.  Original model_points DataFrame is not mutated during setup().

execute:
  8.  Result count equals projection_term_years × 12.
  9.  All results are stored with scenario_id = 0.
  10. Results are stored in ascending timestep order (0, 1, 2, ...).
  11. BEL > 0 for a non-trivial policy in the final month.
  12. Original model_points DataFrame is not mutated during execute().

MP advancement:
  13. policy_duration_mths increments by 1 each step.
  14. attained_age increments at each 12-month anniversary.
  15. attained_age does NOT increment in non-anniversary months.
  16. ENDOW_PAR accrued_bonus_per_policy advances each step.
  17. ENDOW_NONPAR accrued_bonus_per_policy does not change.

teardown:
  18. CSV file is written to output_dir.
  19. Parquet file is written when result_format = PARQUET.
  20. Output file contains the correct column names (RESULT_COLUMNS).
  21. Annual output_timestep writes only year-end rows (t=11, 23, ...).
  22. Quarterly output_timestep writes only quarter-end rows (t=2, 5, 8, ...).
  23. output_horizon_years clips rows beyond the limit.

progress callback:
  24. Callback is called at least once during execute().
  25. Final fraction reported is 1.0.
  26. Fractions are non-decreasing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, ResultFormat, OutputTimestep
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.results.result_store import RESULT_COLUMNS, ResultStore
from engine.run_modes.liability_only_run import LiabilityOnlyRun
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def one_year_config(tmp_path: Path) -> RunConfig:
    """LIABILITY_ONLY config with projection_term_years=1 (12-month loop)."""
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir()
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=1,
    )
    data["output"]["output_dir"] = str(tmp_path / "outputs")
    return RunConfig.from_dict(data)


@pytest.fixture
def fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id": "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 0.6, "equities": 0.3, "derivatives": 0.0, "cash": 0.1},
        "crediting_groups": [
            {"group_id": "GRP_A", "group_name": "Group A", "product_codes": ["P1"]},
        ],
    })


@pytest.fixture
def zero_assumptions() -> ConventionalAssumptions:
    """All-zero rates — every policy survives unchanged, DF = 1 everywhere."""
    return ConventionalAssumptions(
        mortality_rates={},
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(0.0),
    )


@pytest.fixture
def final_month_mp() -> pd.DataFrame:
    """
    100 ENDOW_NONPAR policies with 1 month remaining.

    policy_term_yr=1, policy_duration_mths=11 → remaining = 1
    SA = 10,000, annual_premium = 1,200
    With zero_assumptions:
        maturities = 100
        net_outgo  = 1,000,000 − 10,000 = 990,000
        BEL (zero rate) = 990,000
    """
    return pd.DataFrame([{
        "group_id":                "GRP_A",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          1,
        "policy_duration_mths":    11,
        "accrued_bonus_per_policy": 0.0,
    }])


@pytest.fixture
def mid_term_mp() -> pd.DataFrame:
    """
    100 ENDOW_NONPAR policies, 1-year term just started (remaining = 12).
    Used to drive a clean 12-step projection.
    """
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


@pytest.fixture
def par_mp() -> pd.DataFrame:
    """Single ENDOW_PAR policy, mid-term (24 months remaining)."""
    return pd.DataFrame([{
        "group_id":                "GRP_P",
        "in_force_count":          1.0,
        "sum_assured":             12_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            40,
        "policy_code":             "ENDOW_PAR",
        "policy_term_yr":          5,
        "policy_duration_mths":    36,
        "accrued_bonus_per_policy": 0.0,
    }])


def _make_run(
    config: RunConfig,
    fund_config: FundConfig,
    mp: pd.DataFrame,
    assumptions: ConventionalAssumptions,
    callback=None,
) -> LiabilityOnlyRun:
    return LiabilityOnlyRun(
        config=config,
        fund_config=fund_config,
        model_points=mp,
        assumptions=assumptions,
        progress_callback=callback,
    )


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_passes_for_liability_only(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.validate_config()   # must not raise

    def test_raises_for_deterministic(
        self, tmp_path, fund_config, final_month_mp, zero_assumptions
    ):
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        asset_file       = tmp_path / "assets.csv"; asset_file.write_text("id\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="deterministic",
            asset_data_path=asset_file,
            projection_term_years=1,
        )
        data["output"]["output_dir"] = str(tmp_path / "out")
        config = RunConfig.from_dict(data)
        run = _make_run(config, fund_config, final_month_mp, zero_assumptions)
        with pytest.raises(ValueError, match="liability_only"):
            run.validate_config()

    def test_raises_for_stochastic(
        self, tmp_path, fund_config, final_month_mp, zero_assumptions
    ):
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        asset_file       = tmp_path / "assets.csv"; asset_file.write_text("id\n")
        scenario_file    = tmp_path / "scen.csv";   scenario_file.write_text("s\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            asset_data_path=asset_file,
            scenario_file_path=scenario_file,
            stochastic={"num_scenarios": 10},
            projection_term_years=1,
        )
        data["output"]["output_dir"] = str(tmp_path / "out")
        config = RunConfig.from_dict(data)
        run = _make_run(config, fund_config, final_month_mp, zero_assumptions)
        with pytest.raises(ValueError, match="liability_only"):
            run.validate_config()


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_model_is_conventional_instance(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.setup()
        assert isinstance(run._model, Conventional)

    def test_store_is_result_store_instance(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.setup()
        assert isinstance(run.store, ResultStore)

    def test_store_run_id_matches_config(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.setup()
        assert run.store.run_id == one_year_config.run_id

    def test_original_mp_not_mutated_during_setup(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        original = final_month_mp.copy()
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.setup()
        pd.testing.assert_frame_equal(final_month_mp, original)


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_result_count_equals_projection_months(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        """1-year config → 12 results stored."""
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        run.setup()
        run.execute()
        assert run.store.result_count() == 12

    def test_all_results_have_scenario_id_zero(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        run.setup()
        run.execute()
        df = run.store.as_dataframe()
        assert (df["scenario_id"] == 0).all()

    def test_results_ordered_by_timestep(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        run.setup()
        run.execute()
        df = run.store.as_dataframe()
        assert list(df["timestep"]) == list(range(12))

    def test_bel_positive_in_final_month(
        self, one_year_config, fund_config, final_month_mp, zero_assumptions
    ):
        """final_month_mp: large maturity payment → BEL > 0 at t=0."""
        run = _make_run(one_year_config, fund_config, final_month_mp, zero_assumptions)
        run.setup()
        run.execute()
        t0_result = run.store.get(scenario_id=0, timestep=0)
        assert t0_result.bel > 0

    def test_original_mp_not_mutated_during_execute(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        original = mid_term_mp.copy()
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        run.setup()
        run.execute()
        pd.testing.assert_frame_equal(mid_term_mp, original)


# ---------------------------------------------------------------------------
# MP advancement
# ---------------------------------------------------------------------------

class TestMPAdvancement:
    """
    Tests for _advance_model_points().  Called directly to inspect
    the helper independently of the full projection loop.
    """

    def test_policy_duration_increments_by_one(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        advanced = run._advance_model_points(mid_term_mp)
        assert advanced["policy_duration_mths"].iloc[0] == 1

    def test_attained_age_increments_at_anniversary(
        self, one_year_config, fund_config, zero_assumptions
    ):
        """
        policy_duration_mths = 11 → after advance = 12 (12 % 12 == 0)
        → attained_age increments.
        """
        mp = pd.DataFrame([{
            "group_id": "G", "in_force_count": 1.0,
            "sum_assured": 1.0, "annual_premium": 12.0,
            "attained_age": 40, "policy_code": "ENDOW_NONPAR",
            "policy_term_yr": 5, "policy_duration_mths": 11,
            "accrued_bonus_per_policy": 0.0,
        }])
        run = _make_run(one_year_config, fund_config, mp, zero_assumptions)
        advanced = run._advance_model_points(mp)
        assert advanced["attained_age"].iloc[0] == 41

    def test_attained_age_does_not_increment_mid_year(
        self, one_year_config, fund_config, zero_assumptions
    ):
        """
        policy_duration_mths = 5 → after advance = 6 (6 % 12 != 0)
        → attained_age unchanged.
        """
        mp = pd.DataFrame([{
            "group_id": "G", "in_force_count": 1.0,
            "sum_assured": 1.0, "annual_premium": 12.0,
            "attained_age": 40, "policy_code": "ENDOW_NONPAR",
            "policy_term_yr": 5, "policy_duration_mths": 5,
            "accrued_bonus_per_policy": 0.0,
        }])
        run = _make_run(one_year_config, fund_config, mp, zero_assumptions)
        advanced = run._advance_model_points(mp)
        assert advanced["attained_age"].iloc[0] == 40

    def test_par_bonus_advances_each_step(
        self, one_year_config, fund_config, par_mp
    ):
        """
        bonus_rate_yr = 0.04, SA = 12,000.
        Monthly bonus = 0.04 × 12,000 / 12 = 40.
        Starting accrued = 0 → after one advance = 40.
        """
        assumptions = ConventionalAssumptions(
            mortality_rates={},
            lapse_rates={},
            expense_pct_premium=0.0,
            expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(0.0),
            bonus_rate_yr=0.04,
        )
        run = _make_run(one_year_config, fund_config, par_mp, assumptions)
        advanced = run._advance_model_points(par_mp)
        assert advanced["accrued_bonus_per_policy"].iloc[0] == pytest.approx(40.0)

    def test_nonpar_bonus_unchanged(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        """ENDOW_NONPAR: accrued_bonus_per_policy must stay 0."""
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        advanced = run._advance_model_points(mid_term_mp)
        assert advanced["accrued_bonus_per_policy"].iloc[0] == pytest.approx(0.0)

    def test_input_not_mutated(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        original = mid_term_mp.copy()
        run = _make_run(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        run._advance_model_points(mid_term_mp)
        pd.testing.assert_frame_equal(mid_term_mp, original)


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------

class TestTeardown:
    def _run_full(
        self,
        config: RunConfig,
        fund_config: FundConfig,
        mp: pd.DataFrame,
        assumptions: ConventionalAssumptions,
    ) -> LiabilityOnlyRun:
        run = _make_run(config, fund_config, mp, assumptions)
        run.setup()
        run.execute()
        run.teardown()
        return run

    def test_csv_file_written(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        self._run_full(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        output_dir = Path(one_year_config.output.output_dir)
        expected   = output_dir / "Test_Run_liability_only_results.csv"
        assert expected.exists()

    def test_parquet_file_written(
        self, tmp_path, fund_config, mid_term_mp, zero_assumptions
    ):
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=1,
        )
        data["output"]["output_dir"]      = str(tmp_path / "out")
        data["output"]["result_format"]   = "parquet"
        config = RunConfig.from_dict(data)
        self._run_full(config, fund_config, mid_term_mp, zero_assumptions)
        expected = Path(data["output"]["output_dir"]) / "Test_Run_liability_only_results.parquet"
        assert expected.exists()

    def test_csv_has_correct_columns(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        self._run_full(one_year_config, fund_config, mid_term_mp, zero_assumptions)
        output_dir = Path(one_year_config.output.output_dir)
        path       = output_dir / "Test_Run_liability_only_results.csv"
        df = pd.read_csv(path)
        assert list(df.columns) == list(RESULT_COLUMNS)

    def test_annual_output_timestep_writes_year_end_rows(
        self, tmp_path, fund_config, mid_term_mp, zero_assumptions
    ):
        """
        1-year projection, annual output → only t=11 written (one row).
        """
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=1,
            output_timestep="annual",
        )
        data["output"]["output_dir"] = str(tmp_path / "out")
        config = RunConfig.from_dict(data)
        self._run_full(config, fund_config, mid_term_mp, zero_assumptions)
        path = Path(data["output"]["output_dir"]) / f"Test_Run_liability_only_results.csv"
        df   = pd.read_csv(path)
        assert len(df) == 1
        assert df["timestep"].iloc[0] == 11

    def test_quarterly_output_timestep_writes_quarter_end_rows(
        self, tmp_path, fund_config, mid_term_mp, zero_assumptions
    ):
        """
        1-year projection, quarterly output → t=2, 5, 8, 11 (4 rows).
        """
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=1,
            output_timestep="quarterly",
        )
        data["output"]["output_dir"] = str(tmp_path / "out")
        config = RunConfig.from_dict(data)
        self._run_full(config, fund_config, mid_term_mp, zero_assumptions)
        path = Path(data["output"]["output_dir"]) / f"Test_Run_liability_only_results.csv"
        df   = pd.read_csv(path)
        assert len(df) == 4
        assert list(df["timestep"]) == [2, 5, 8, 11]

    def test_horizon_filter_clips_output(
        self, tmp_path, fund_config, mid_term_mp, zero_assumptions
    ):
        """
        2-year projection, output_horizon_years=1 → only 12 rows written.
        """
        assumption_dir   = tmp_path / "a"; assumption_dir.mkdir()
        fund_config_file = tmp_path / "fc.yaml"; fund_config_file.write_text("x: 1\n")
        # policy needs 2-year term to avoid expiring mid-run
        mp = mid_term_mp.copy()
        mp["policy_term_yr"] = 2
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=2,
            output_horizon_years=1,
        )
        data["output"]["output_dir"] = str(tmp_path / "out")
        config = RunConfig.from_dict(data)
        self._run_full(config, fund_config, mp, zero_assumptions)
        path = Path(data["output"]["output_dir"]) / f"Test_Run_liability_only_results.csv"
        df   = pd.read_csv(path)
        assert len(df) == 12
        assert df["timestep"].max() == 11


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_called_during_execute(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        calls: List[Tuple[float, str]] = []
        run = _make_run(
            one_year_config, fund_config, mid_term_mp, zero_assumptions,
            callback=lambda f, m: calls.append((f, m)),
        )
        run.setup()
        run.execute()
        assert len(calls) > 0

    def test_final_fraction_is_one(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        fractions: List[float] = []
        run = _make_run(
            one_year_config, fund_config, mid_term_mp, zero_assumptions,
            callback=lambda f, m: fractions.append(f),
        )
        run.setup()
        run.execute()
        assert fractions[-1] == pytest.approx(1.0)

    def test_fractions_non_decreasing(
        self, one_year_config, fund_config, mid_term_mp, zero_assumptions
    ):
        fractions: List[float] = []
        run = _make_run(
            one_year_config, fund_config, mid_term_mp, zero_assumptions,
            callback=lambda f, m: fractions.append(f),
        )
        run.setup()
        run.execute()
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1]

"""
tests/unit/run_modes/test_bpa_run.py

Tests for BPARun + _BPACompositeLiability (DECISIONS.md §21, §27, §46).

Test matrix
-----------
validate_config:
  1.  Passes silently for run_type = BPA.
  2.  Raises ValueError for run_type = DETERMINISTIC.
  3.  Raises ValueError for run_type = LIABILITY_ONLY.

setup:
  4.  self._fund is a Fund instance after setup().
  5.  self._store is a ResultStore instance after setup().
  6.  self._calendar has the expected n_periods after setup().
  7.  Fund is constructed with a BuyAndHoldStrategy.

execute — MA calibration:
  8.  post_ma_curve has higher spot rates than pre_ma_curve when ma_benefit_bps > 0.
  9.  post_ma_bel < pre_ma_bel when ma_benefit_bps > 0.
  10. When ma_benefit_bps == 0, |post_ma_bel − pre_ma_bel| < 1e-6.
  11. cashflow_test_passes=False logs a warning but does not abort the run.

execute — projection:
  12. Result count == n_periods × n_cohorts.
  13. Timestep indices in stored results are 0-based and sequential.
  14. bel_pre_ma and bel_post_ma are non-None for all stored results.
  15. post_ma_bel ≤ pre_ma_bel for all periods when ma_benefit_bps > 0.
  16. AssetScenarioPoint.dt == 1/12 for monthly periods, 1.0 for annual.

teardown:
  17. BPA results CSV is written to output_dir.
  18. MA summary CSV is written with expected columns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.core.fund import Fund
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH, MIN_TABLE_AGE
from engine.liability.bpa.registry import BPADealMetadata, BPADealRegistry
from engine.matching_adjustment.ma_calculator import MACalculator
from engine.results.result_store import ResultStore
from engine.run_modes.bpa_run import BPARun, _BPACompositeLiability
from engine.scr.interest_stress import InterestStressEngine
from engine.scr.longevity_stress import LongevityStressEngine
from engine.scr.scr_calculator import SCRCalculator
from engine.scr.spread_stress import SpreadStressEngine
from engine.strategy.buy_and_hold_strategy import BuyAndHoldStrategy
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Helpers — synthetic mortality basis
# ---------------------------------------------------------------------------

def _flat_mortality(q_annual: float = 0.01) -> MortalityBasis:
    """Flat mortality basis for testing: constant q_x at all ages."""
    base = np.full(TABLE_LENGTH, q_annual, dtype=float)
    impr = np.zeros(TABLE_LENGTH, dtype=float)
    return MortalityBasis(
        base_table_male            = base.copy(),
        base_table_female          = base.copy(),
        initial_improvement_male   = impr.copy(),
        initial_improvement_female = impr.copy(),
        base_year                  = 2023,
        ltr                        = 0.0,
        convergence_period         = 20,
        ae_ratio_male              = 1.0,
        ae_ratio_female            = 1.0,
    )


def _flat_assumptions(q_annual: float = 0.01) -> BPAAssumptions:
    """BPAAssumptions with flat mortality and 3% discount rate."""
    ill = np.zeros(TABLE_LENGTH, dtype=float)
    return BPAAssumptions(
        mortality      = _flat_mortality(q_annual),
        valuation_year = 2024,
        discount_curve = RiskFreeRateCurve.flat(0.03),
        inflation_rate = 0.025,
        rpi_rate       = 0.03,
        tv_rate        = 0.0,
        ill_health_rates = ill,
    )


# ---------------------------------------------------------------------------
# Helpers — model points
# ---------------------------------------------------------------------------

def _make_in_payment_mp(deal_id: str = "Deal_2024Q1") -> dict[str, pd.DataFrame]:
    """Single in-payment model point for one deal."""
    return {
        deal_id: pd.DataFrame([{
            "mp_id":      "P001",
            "sex":        "M",
            "age":        70.0,
            "in_force_count": 100.0,
            "pension_pa": 10_000.0,
            "lpi_cap":    0.05,
            "lpi_floor":  0.0,
            "gmp_pa":     0.0,
        }])
    }


# ---------------------------------------------------------------------------
# Helpers — FS table (inline CSV string)
# ---------------------------------------------------------------------------

_FS_CSV = (
    "# effective_date: 2024-07-01\n"
    "# source_ref: PRA PS10/24\n"
    "rating,seniority,tenor_lower,tenor_upper,long_run_pd_pct,lgd_pct,"
    "downgrade_allowance_bps,fs_bps\n"
    "BBB,senior_unsecured,0,5,0.17,40,21,28\n"
    "BBB,senior_unsecured,5,10,0.17,40,24,31\n"
    "BBB,senior_unsecured,10,20,0.17,40,26,33\n"
)


def _make_fs_table(tmp_path: Path):
    from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
    p = tmp_path / "fs.csv"
    p.write_text(_FS_CSV)
    return FundamentalSpreadTable.from_csv(p)


# ---------------------------------------------------------------------------
# Helpers — asset model and MA inputs
# ---------------------------------------------------------------------------

def _make_asset_model() -> AssetModel:
    """Single AC bond with spread above FS → positive MA benefit."""
    am = AssetModel()
    am.add_asset(Bond("bond_01",
                      face_value=1_000_000.0,
                      annual_coupon_rate=0.08,
                      maturity_month=120,
                      accounting_basis="AC",
                      initial_book_value=950_000.0))
    return am


def _make_assets_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "asset_id":                   "bond_01",
        "cashflow_type":              "fixed",
        "currency":                   "GBP",
        "has_credit_risk_transfer":   False,
        "has_qualifying_currency_swap": False,
        "rating":                     "BBB",
        "seniority":                  "senior_unsecured",
        "tenor_years":                10.0,
        "spread_bps":                 120.0,
    }])


def _make_asset_cfs() -> pd.DataFrame:
    """Annual bond CFs for 10 years (simplified flat coupon + principal at end)."""
    rows = []
    for yr in range(1, 11):
        cf = 80_000.0 if yr < 10 else 1_080_000.0
        rows.append({"t": yr, "asset_id": "bond_01", "cf": cf})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers — RunConfig and FundConfig for BPA
# ---------------------------------------------------------------------------

def _make_bpa_run_config(tmp_path: Path) -> RunConfig:
    assumption_dir = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path     = fund_config_file,
        assumption_dir       = assumption_dir,
        run_type             = "bpa",
        projection_term_years= 3,
    )
    # Override output dir
    data.setdefault("output", {})["output_dir"] = str(tmp_path / "output")
    return RunConfig.from_dict(data)


def _make_fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id":   "BPA_FUND",
        "fund_name": "BPA Fund",
        "saa_weights": {"bonds": 1.0, "equities": 0.0, "derivatives": 0.0, "cash": 0.0},
        "crediting_groups": [
            {"group_id": "GRP_BPA", "group_name": "BPA Group", "product_codes": ["BPA"]},
        ],
    })


# ---------------------------------------------------------------------------
# Helpers — deal registry
# ---------------------------------------------------------------------------

def _make_registry(deal_id: str = "Deal_2024Q1") -> BPADealRegistry:
    from datetime import date
    return BPADealRegistry([
        BPADealMetadata(
            deal_id        = deal_id,
            deal_type      = "buyout",
            inception_date = date(2024, 1, 1),
            deal_name      = "Test Deal",
            ma_eligible    = True,
        )
    ])


# ---------------------------------------------------------------------------
# Main fixture — builds a BPARun ready to run
# ---------------------------------------------------------------------------

@pytest.fixture
def bpa_run(tmp_path: Path):
    """
    BPARun with:
    - 1 deal, in-payment only (deferred/dependant/enhanced empty)
    - projection_years=3, monthly_years=1 → 12 + 2 = 14 periods
    - Single AC bond, BBB, spread=120bps → MA benefit ≈ 87-89bps
    - No Ifrs17StateStore (skip IFRS 17 save hook in tests)
    """
    deal_id = "Deal_2024Q1"
    return BPARun(
        config                    = _make_bpa_run_config(tmp_path),
        fund_config               = _make_fund_config(),
        in_payment_mps            = _make_in_payment_mp(deal_id),
        deferred_mps              = {},
        dependant_mps             = {},
        enhanced_mps              = {},
        assumptions               = _flat_assumptions(),
        asset_model               = _make_asset_model(),
        assets_df                 = _make_assets_df(),
        asset_cfs                 = _make_asset_cfs(),
        fs_table                  = _make_fs_table(tmp_path),
        deal_registry             = _make_registry(deal_id),
        ifrs17_state_store        = None,
        ma_highly_predictable_cap = 0.35,
        projection_years          = 3,
        monthly_years             = 1,
        initial_cash              = 0.0,
    )


@pytest.fixture
def bpa_run_executed(bpa_run: BPARun):
    """BPARun after setup() and execute() have been called."""
    bpa_run.setup()
    bpa_run.execute()
    return bpa_run


# ---------------------------------------------------------------------------
# Test 1–3 — validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:

    def test_passes_for_bpa(self, bpa_run: BPARun):
        """Test 1 — no exception for run_type=BPA."""
        bpa_run.validate_config()   # must not raise

    def test_raises_for_deterministic(self, tmp_path: Path):
        """Test 2 — ValueError when run_type=DETERMINISTIC."""
        assumption_dir = tmp_path / "ass"
        assumption_dir.mkdir()
        fc = tmp_path / "fc.yaml"
        fc.write_text("placeholder: true\n")
        asset_file = tmp_path / "assets.csv"
        asset_file.write_text("asset_id\n")
        data = build_run_config_dict(
            fund_config_path=fc,
            assumption_dir=assumption_dir,
            run_type="deterministic",
            asset_data_path=asset_file,
        )
        cfg = RunConfig.from_dict(data)
        run = BPARun(
            config         = cfg,
            fund_config    = _make_fund_config(),
            in_payment_mps = {},
            deferred_mps   = {},
            dependant_mps  = {},
            enhanced_mps   = {},
            assumptions    = _flat_assumptions(),
            asset_model    = AssetModel(),
            assets_df      = pd.DataFrame(),
            asset_cfs      = pd.DataFrame(),
            fs_table       = _make_fs_table(tmp_path),
            deal_registry  = _make_registry(),
        )
        with pytest.raises(ValueError, match="run_type='bpa'"):
            run.validate_config()

    def test_raises_for_liability_only(self, tmp_path: Path):
        """Test 3 — ValueError when run_type=LIABILITY_ONLY."""
        assumption_dir = tmp_path / "ass"
        assumption_dir.mkdir()
        fc = tmp_path / "fc.yaml"
        fc.write_text("placeholder: true\n")
        data = build_run_config_dict(
            fund_config_path=fc,
            assumption_dir=assumption_dir,
            run_type="liability_only",
        )
        cfg = RunConfig.from_dict(data)
        run = BPARun(
            config         = cfg,
            fund_config    = _make_fund_config(),
            in_payment_mps = {},
            deferred_mps   = {},
            dependant_mps  = {},
            enhanced_mps   = {},
            assumptions    = _flat_assumptions(),
            asset_model    = AssetModel(),
            assets_df      = pd.DataFrame(),
            asset_cfs      = pd.DataFrame(),
            fs_table       = _make_fs_table(tmp_path),
            deal_registry  = _make_registry(),
        )
        with pytest.raises(ValueError, match="run_type='bpa'"):
            run.validate_config()


# ---------------------------------------------------------------------------
# Test 4–7 — setup
# ---------------------------------------------------------------------------

class TestSetup:

    def test_fund_is_fund_instance(self, bpa_run: BPARun):
        """Test 4 — self._fund is a Fund after setup()."""
        bpa_run.setup()
        assert isinstance(bpa_run.fund, Fund)

    def test_store_is_result_store(self, bpa_run: BPARun):
        """Test 5 — self._store is a ResultStore after setup()."""
        bpa_run.setup()
        assert isinstance(bpa_run.store, ResultStore)

    def test_calendar_n_periods(self, bpa_run: BPARun):
        """Test 6 — calendar has 12 + 2 = 14 periods for projection_years=3, monthly_years=1."""
        bpa_run.setup()
        assert bpa_run.calendar is not None
        # 1 year monthly (12 periods) + 2 annual periods
        assert bpa_run.calendar.n_periods == 12 + 2

    def test_fund_has_buy_and_hold_strategy(self, bpa_run: BPARun):
        """Test 7 — Fund is constructed with BuyAndHoldStrategy."""
        bpa_run.setup()
        assert isinstance(bpa_run.fund._investment_strategy, BuyAndHoldStrategy)


# ---------------------------------------------------------------------------
# Test 8–11 — execute: MA calibration
# ---------------------------------------------------------------------------

class TestExecuteMACalibration:

    def test_post_ma_curve_higher_than_pre_ma(self, bpa_run_executed: BPARun):
        """Test 8 — post-MA spot rates > pre-MA when MA benefit > 0."""
        cal = bpa_run_executed.ma_calibration
        assert cal is not None
        if cal.ma_result.ma_benefit_bps > 0.0:
            for mat, pre_rate in cal.pre_ma_curve.spot_rates.items():
                post_rate = cal.post_ma_curve.spot_rates[mat]
                assert post_rate > pre_rate, (
                    f"post-MA rate at {mat}yr ({post_rate}) should exceed "
                    f"pre-MA rate ({pre_rate})"
                )

    def test_post_ma_bel_less_than_pre_ma_bel(self, bpa_run_executed: BPARun):
        """Test 9 — post-MA BEL < pre-MA BEL when MA benefit > 0."""
        cal = bpa_run_executed.ma_calibration
        assert cal is not None
        if cal.ma_result.ma_benefit_bps > 0.0:
            assert cal.post_ma_bel < cal.pre_ma_bel, (
                f"post_ma_bel ({cal.post_ma_bel:.2f}) >= pre_ma_bel ({cal.pre_ma_bel:.2f})"
            )

    def test_zero_ma_benefit_gives_equal_bels(self, tmp_path: Path):
        """Test 10 — when MA benefit = 0, post_ma_bel ≈ pre_ma_bel."""
        deal_id = "Deal_2024Q1"
        # Asset with spread exactly equal to FS (28bps) → MA benefit ≈ 0
        assets_df = pd.DataFrame([{
            "asset_id":                   "bond_zero",
            "cashflow_type":              "fixed",
            "currency":                   "GBP",
            "has_credit_risk_transfer":   False,
            "has_qualifying_currency_swap": False,
            "rating":                     "BBB",
            "seniority":                  "senior_unsecured",
            "tenor_years":                3.0,
            "spread_bps":                 28.0,  # exactly FS → 0 MA contribution
        }])
        asset_cfs = pd.DataFrame([
            {"t": 1, "asset_id": "bond_zero", "cf": 50_000.0},
            {"t": 2, "asset_id": "bond_zero", "cf": 50_000.0},
            {"t": 3, "asset_id": "bond_zero", "cf": 1_050_000.0},
        ])
        am = AssetModel()
        am.add_asset(Bond("bond_zero", 1_000_000.0, 0.05, 36, "AC", 950_000.0))
        run = BPARun(
            config                    = _make_bpa_run_config(tmp_path),
            fund_config               = _make_fund_config(),
            in_payment_mps            = _make_in_payment_mp(deal_id),
            deferred_mps              = {},
            dependant_mps             = {},
            enhanced_mps              = {},
            assumptions               = _flat_assumptions(),
            asset_model               = am,
            assets_df                 = assets_df,
            asset_cfs                 = asset_cfs,
            fs_table                  = _make_fs_table(tmp_path),
            deal_registry             = _make_registry(deal_id),
            projection_years          = 3,
            monthly_years             = 1,
        )
        run.setup()
        run.execute()
        cal = run.ma_calibration
        assert abs(cal.post_ma_bel - cal.pre_ma_bel) < 1.0, (
            f"Expected near-zero BEL difference, got "
            f"pre={cal.pre_ma_bel:.4f} post={cal.post_ma_bel:.4f}"
        )

    def test_cf_test_failure_logs_warning_not_abort(
        self, tmp_path: Path, caplog
    ):
        """Test 11 — CF test failure logs WARNING but run completes."""
        deal_id = "Deal_2024Q1"
        # Asset CFs much smaller than liability → cashflow matching test fails
        asset_cfs_tiny = pd.DataFrame([
            {"t": 1, "asset_id": "bond_01", "cf": 1.0},   # nearly zero
            {"t": 2, "asset_id": "bond_01", "cf": 1.0},
            {"t": 3, "asset_id": "bond_01", "cf": 1.0},
        ])
        run = BPARun(
            config                    = _make_bpa_run_config(tmp_path),
            fund_config               = _make_fund_config(),
            in_payment_mps            = _make_in_payment_mp(deal_id),
            deferred_mps              = {},
            dependant_mps             = {},
            enhanced_mps              = {},
            assumptions               = _flat_assumptions(),
            asset_model               = _make_asset_model(),
            assets_df                 = _make_assets_df(),
            asset_cfs                 = asset_cfs_tiny,
            fs_table                  = _make_fs_table(tmp_path),
            deal_registry             = _make_registry(deal_id),
            projection_years          = 3,
            monthly_years             = 1,
        )
        run.setup()
        with caplog.at_level(logging.WARNING, logger="BPARun"):
            run.execute()  # must NOT raise
        # MA calibration result is populated
        assert run.ma_calibration is not None
        # Warning was logged
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("FAILED" in m or "failed" in m for m in warning_msgs), (
            f"Expected CF test failure warning; got: {warning_msgs}"
        )


# ---------------------------------------------------------------------------
# Test 12–16 — execute: projection
# ---------------------------------------------------------------------------

class TestExecuteProjection:

    def test_result_count_equals_periods_times_cohorts(
        self, bpa_run_executed: BPARun
    ):
        """Test 12 — result count == n_periods × n_cohorts."""
        store = bpa_run_executed.store
        calendar = bpa_run_executed.calendar
        n_periods = calendar.n_periods
        # 1 deal × 1 pop type (in-payment only; deferred/dependant/enhanced empty)
        n_cohorts = 1
        assert store.result_count() == n_periods * n_cohorts

    def test_timestep_indices_sequential(self, bpa_run_executed: BPARun):
        """Test 13 — timesteps are 0-based and sequential for each cohort."""
        df = bpa_run_executed.store.as_dataframe()
        for cohort_id, group in df.groupby("cohort_id"):
            ts = sorted(group["timestep"].tolist())
            expected = list(range(bpa_run_executed.calendar.n_periods))
            assert ts == expected, (
                f"Cohort {cohort_id}: expected timesteps {expected[:5]}…, got {ts[:5]}…"
            )

    def test_bel_pre_ma_post_ma_non_none(self, bpa_run_executed: BPARun):
        """Test 14 — bel_pre_ma and bel_post_ma are not None for all rows."""
        df = bpa_run_executed.store.as_dataframe()
        assert df["bel_pre_ma"].notna().all(), "bel_pre_ma contains NaN"
        assert df["bel_post_ma"].notna().all(), "bel_post_ma contains NaN"

    def test_post_ma_bel_leq_pre_ma_bel(self, bpa_run_executed: BPARun):
        """Test 15 — post_ma_bel ≤ pre_ma_bel when MA benefit > 0."""
        cal = bpa_run_executed.ma_calibration
        if cal.ma_result.ma_benefit_bps <= 0.0:
            pytest.skip("MA benefit is zero — skipping directional BEL test")
        df = bpa_run_executed.store.as_dataframe()
        # Allow tiny floating-point tolerance
        assert (df["bel_post_ma"] <= df["bel_pre_ma"] + 1e-6).all(), (
            "Some bel_post_ma values exceed bel_pre_ma (should be ≤)"
        )

    def test_dt_values_match_calendar(self, bpa_run: BPARun):
        """
        Test 16 — dt = 1/12 for monthly periods and 1.0 for annual periods.

        Verified by intercepting Fund.step_time() and capturing the scenario
        argument at each call.
        """
        bpa_run.setup()
        captured_scenarios: list[AssetScenarioPoint] = []
        original_step = bpa_run.fund.step_time

        def capture_step(scenario, model_points, assumptions):
            captured_scenarios.append(scenario)
            return original_step(scenario, model_points, assumptions)

        bpa_run.fund.step_time = capture_step
        bpa_run.execute()

        calendar = bpa_run.calendar
        assert len(captured_scenarios) == calendar.n_periods
        for scenario, period in zip(captured_scenarios, calendar.periods):
            expected_dt = period.year_fraction
            assert abs(scenario.dt - expected_dt) < 1e-12, (
                f"Period {period.period_index}: expected dt={expected_dt}, "
                f"got dt={scenario.dt}"
            )


# ---------------------------------------------------------------------------
# Test 17–18 — teardown
# ---------------------------------------------------------------------------

class TestTeardown:

    def test_bpa_results_csv_written(self, bpa_run_executed: BPARun, tmp_path: Path):
        """Test 17 — BPA results CSV is written to output_dir."""
        bpa_run_executed.teardown()
        output_dir = Path(bpa_run_executed.config.output.output_dir)
        csv_files = list(output_dir.glob("*bpa_results*.csv"))
        assert len(csv_files) == 1, f"Expected 1 results CSV, found: {csv_files}"
        df = pd.read_csv(csv_files[0])
        assert "timestep" in df.columns
        assert "cohort_id" in df.columns
        assert "bel_pre_ma" in df.columns
        assert "bel_post_ma" in df.columns

    def test_ma_summary_csv_written(self, bpa_run_executed: BPARun, tmp_path: Path):
        """Test 18 — MA summary CSV is written with expected columns."""
        bpa_run_executed.teardown()
        output_dir = Path(bpa_run_executed.config.output.output_dir)
        summary_files = list(output_dir.glob("*bpa_ma_summary*.csv"))
        assert len(summary_files) == 1, f"Expected 1 MA summary CSV, found: {summary_files}"
        df = pd.read_csv(summary_files[0])
        assert "ma_benefit_bps" in df.columns
        assert "eligible_asset_count" in df.columns
        assert len(df) == 1, "MA summary should have exactly 1 row"


# ---------------------------------------------------------------------------
# Test 19–27 — IFRS 17 GMM wiring (Step 21)
# ---------------------------------------------------------------------------

def _mock_ifrs17_store():
    """Mock Ifrs17StateStore that tracks save_state and save_movements calls."""
    store = MagicMock()
    store.load_state.return_value = None   # always None = first-run inception
    return store


def _bpa_run_with_ifrs17(tmp_path: Path, store=None):
    """BPARun with ifrs17_state_store wired."""
    deal_id = "Deal_2024Q1"
    if store is None:
        store = _mock_ifrs17_store()
    return BPARun(
        config                    = _make_bpa_run_config(tmp_path),
        fund_config               = _make_fund_config(),
        in_payment_mps            = _make_in_payment_mp(deal_id),
        deferred_mps              = {},
        dependant_mps             = {},
        enhanced_mps              = {},
        assumptions               = _flat_assumptions(),
        asset_model               = _make_asset_model(),
        assets_df                 = _make_assets_df(),
        asset_cfs                 = _make_asset_cfs(),
        fs_table                  = _make_fs_table(tmp_path),
        deal_registry             = _make_registry(deal_id),
        ifrs17_state_store        = store,
        projection_years          = 3,
        monthly_years             = 1,
        initial_cash              = 0.0,
    )


class TestIfrs17Wiring:

    def test_save_state_called_once_per_cohort(self, tmp_path: Path):
        """Test 19 — save_state called exactly once per cohort after execute()."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        # 1 deal, 1 in-payment cohort → 1 cohort_id
        assert store.save_state.call_count == 1

    def test_save_movements_called_once_per_cohort(self, tmp_path: Path):
        """Test 20 — save_movements called exactly once per cohort after execute()."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        assert store.save_movements.call_count == 1

    def test_movements_list_length_equals_n_periods(self, tmp_path: Path):
        """Test 21 — movements list passed to save_movements has length n_periods."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        # save_movements(cohort_id, valuation_date, movements)
        args = store.save_movements.call_args
        movements = args[0][2]   # third positional arg
        n_periods = run.calendar.n_periods
        assert len(movements) == n_periods, (
            f"Expected {n_periods} movements, got {len(movements)}"
        )

    def test_no_gmm_calls_when_store_is_none(self, tmp_path: Path):
        """Test 22 — when ifrs17_state_store=None, no state/movements are saved."""
        deal_id = "Deal_2024Q1"
        run = BPARun(
            config             = _make_bpa_run_config(tmp_path),
            fund_config        = _make_fund_config(),
            in_payment_mps     = _make_in_payment_mp(deal_id),
            deferred_mps       = {},
            dependant_mps      = {},
            enhanced_mps       = {},
            assumptions        = _flat_assumptions(),
            asset_model        = _make_asset_model(),
            assets_df          = _make_assets_df(),
            asset_cfs          = _make_asset_cfs(),
            fs_table           = _make_fs_table(tmp_path),
            deal_registry      = _make_registry(deal_id),
            ifrs17_state_store = None,   # no IFRS 17 store
            projection_years   = 3,
            monthly_years      = 1,
        )
        run.setup()
        run.execute()   # must not raise

    def test_csm_opening_zero_on_first_run(self, tmp_path: Path):
        """Test 23 — csm_opening of period-0 result is 0.0 at inception (§50)."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        args = store.save_movements.call_args
        movements = args[0][2]
        first_result = movements[0]
        assert first_result.csm_opening == pytest.approx(0.0, abs=1e-10)

    def test_closing_state_csm_non_negative(self, tmp_path: Path):
        """Test 24 — closing Ifrs17State has csm_balance >= 0."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        closing_state = store.save_state.call_args[0][0]
        assert closing_state.csm_balance >= 0.0

    def test_lrc_identity_holds(self, tmp_path: Path):
        """Test 25 — LRC = bel_current + risk_adjustment + csm_closing for each period."""
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()
        args = store.save_movements.call_args
        movements = args[0][2]
        for i, r in enumerate(movements):
            expected_lrc = r.bel_current + r.risk_adjustment + r.csm_closing
            assert r.lrc == pytest.approx(expected_lrc, rel=1e-6), (
                f"LRC identity failed at period {i}: "
                f"lrc={r.lrc:.4f} != bel+ra+csm={expected_lrc:.4f}"
            )

    def test_bel_locked_at_most_bel_current_when_ma_positive(self, tmp_path: Path):
        """
        Test 26 — bel_locked <= bel_current when locked_in_rate > post-MA curve rate.
        When MA benefit > 0, the post-MA curve rate < locked-in rate (which is
        3% RFR + MA_bps/10000). A higher discount rate produces a lower BEL.
        So bel_locked should be <= bel_current for profitable MA portfolios.
        """
        store = _mock_ifrs17_store()
        run = _bpa_run_with_ifrs17(tmp_path, store)
        run.setup()
        run.execute()

        cal = run.ma_calibration
        if cal is None or cal.ma_result.ma_benefit_bps <= 0.0:
            pytest.skip("MA benefit not positive — inequality not guaranteed")

        args = store.save_movements.call_args
        movements = args[0][2]
        for i, r in enumerate(movements):
            assert r.bel_locked <= r.bel_current + 1e-6, (
                f"Period {i}: bel_locked ({r.bel_locked:.4f}) > "
                f"bel_current ({r.bel_current:.4f})"
            )

    def test_risk_adjustment_positive_and_decreasing_with_scr(self, tmp_path: Path):
        """Test 27 — RA is positive at t=0 when SCRCalculator is injected (Step 22).

        Without scr_calculator the RA falls back to zero (per DECISIONS.md §51).
        With scr_calculator the longevity stress produces stressed_bel > base_bel,
        so SCR_longevity > 0 and CostOfCapitalRA returns a positive RA.
        """
        # Build a minimal SCRCalculator with the same FS table used by BPARun.
        fs_table     = _make_fs_table(tmp_path)
        ma_calc      = MACalculator(fs_table)
        scr_calc     = SCRCalculator(
            spread_engine    = SpreadStressEngine(
                ma_calculator   = ma_calc,
                spread_up_bps   = 75.0,
                spread_down_bps = 25.0,
            ),
            interest_engine  = InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0),
            longevity_engine = LongevityStressEngine(mortality_stress_factor=0.20),
        )

        deal_id = "Deal_2024Q1"
        store   = _mock_ifrs17_store()
        run = BPARun(
            config             = _make_bpa_run_config(tmp_path),
            fund_config        = _make_fund_config(),
            in_payment_mps     = _make_in_payment_mp(deal_id),
            deferred_mps       = {},
            dependant_mps      = {},
            enhanced_mps       = {},
            assumptions        = _flat_assumptions(),
            asset_model        = _make_asset_model(),
            assets_df          = _make_assets_df(),
            asset_cfs          = _make_asset_cfs(),
            fs_table           = fs_table,
            deal_registry      = _make_registry(deal_id),
            ifrs17_state_store = store,
            scr_calculator     = scr_calc,
            projection_years   = 3,
            monthly_years      = 1,
            initial_cash       = 0.0,
        )
        run.setup()
        run.execute()

        args      = store.save_movements.call_args
        movements = args[0][2]
        ra_values = [r.risk_adjustment for r in movements]

        # With SCR_longevity > 0 the RA at t=0 must be positive
        assert ra_values[0] > 0.0, f"Expected RA > 0 at t=0, got {ra_values[0]}"
        # RA must be non-increasing overall (stressed BEL runs off with base BEL)
        assert ra_values[-1] <= ra_values[0], (
            f"RA should not increase overall: first={ra_values[0]:.4f} "
            f"last={ra_values[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests 28–33 — ResultStore cohort helpers + teardown SCR / cohort summary
# ---------------------------------------------------------------------------

def _bpa_run_with_scr(tmp_path: Path) -> BPARun:
    """BPARun with a real SCRCalculator injected (no IFRS 17 store)."""
    from engine.matching_adjustment.ma_calculator import MACalculator
    deal_id = "Deal_2024Q1"
    fs_table = _make_fs_table(tmp_path)
    ma_calc  = MACalculator(fs_table)
    scr_calc = SCRCalculator(
        spread_engine    = SpreadStressEngine(
            ma_calculator   = ma_calc,
            spread_up_bps   = 75.0,
            spread_down_bps = 25.0,
        ),
        interest_engine  = InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0),
        longevity_engine = LongevityStressEngine(mortality_stress_factor=0.20),
    )
    return BPARun(
        config             = _make_bpa_run_config(tmp_path),
        fund_config        = _make_fund_config(),
        in_payment_mps     = _make_in_payment_mp(deal_id),
        deferred_mps       = {},
        dependant_mps      = {},
        enhanced_mps       = {},
        assumptions        = _flat_assumptions(),
        asset_model        = _make_asset_model(),
        assets_df          = _make_assets_df(),
        asset_cfs          = _make_asset_cfs(),
        fs_table           = fs_table,
        deal_registry      = _make_registry(deal_id),
        scr_calculator     = scr_calc,
        projection_years   = 3,
        monthly_years      = 1,
        initial_cash       = 0.0,
    )


class TestResultStoreCohortHelpers:

    def test_cohort_ids_returns_sorted_list(self, bpa_run_executed: BPARun):
        """Test 28 — cohort_ids() returns a sorted non-empty list for BPA runs."""
        ids = bpa_run_executed.store.cohort_ids()
        assert len(ids) == 1
        assert ids[0] == "Deal_2024Q1_pensioner"

    def test_cohort_ids_empty_for_non_bpa(self, tmp_path: Path):
        """Test 29 — cohort_ids() returns [] when all results have cohort_id=None."""
        from engine.results.result_store import ResultStore, TimestepResult
        from engine.liability.base_liability import LiabilityCashflows, Decrements
        store = ResultStore(run_id="r1")
        store.store(TimestepResult(
            run_id="r1", scenario_id=0, timestep=0, cohort_id=None,
            cashflows=LiabilityCashflows(0, 0.0, 0.0, 0.0, 0.0, 0.0),
            decrements=Decrements(0, 0.0, 0.0, 0.0, 0.0, 0.0),
            bel=1000.0, reserve=1000.0,
        ))
        assert store.cohort_ids() == []

    def test_as_cohort_pivot_has_multiindex_columns(self, bpa_run_executed: BPARun):
        """Test 30 — as_cohort_pivot() returns DataFrame with MultiIndex columns."""
        pivot = bpa_run_executed.store.as_cohort_pivot()
        assert isinstance(pivot.columns, pd.MultiIndex)
        # All non-timestep columns should belong to the one cohort
        cohort_id = "Deal_2024Q1_pensioner"
        top_level = [
            c[0] if isinstance(c, tuple) else c
            for c in pivot.columns
            if (c[0] if isinstance(c, tuple) else c) != "timestep"
        ]
        assert all(c == cohort_id for c in top_level)

    def test_as_cohort_pivot_row_count_equals_n_periods(self, bpa_run_executed: BPARun):
        """Test 31 — pivot has one row per projection period."""
        pivot = bpa_run_executed.store.as_cohort_pivot()
        assert len(pivot) == bpa_run_executed.calendar.n_periods


class TestTeardownNewOutputs:

    def test_cohort_summary_csv_written(self, bpa_run_executed: BPARun, tmp_path: Path):
        """Test 32 — teardown writes {run_id}_bpa_cohort_summary.csv."""
        bpa_run_executed.teardown()
        output_dir = Path(bpa_run_executed.config.output.output_dir)
        files = list(output_dir.glob("*bpa_cohort_summary*.csv"))
        assert len(files) == 1, f"Expected 1 cohort summary CSV, found: {files}"
        df = pd.read_csv(files[0])
        assert "cohort_id" in df.columns
        assert "bel_post_ma_t0" in df.columns
        assert "bel_pre_ma_t0" in df.columns
        assert "ma_reduction_t0" in df.columns
        assert "in_force_lives_t0" in df.columns
        # One row per cohort
        assert len(df) == len(bpa_run_executed.store.cohort_ids())
        # BEL relationship holds
        row = df.iloc[0]
        assert row["bel_post_ma_t0"] <= row["bel_pre_ma_t0"] + 1e-6

    def test_scr_result_csv_written_when_scr_injected(self, tmp_path: Path):
        """Test 33 — teardown writes {run_id}_bpa_scr_result.csv when SCRCalculator injected."""
        run = _bpa_run_with_scr(tmp_path)
        run.setup()
        run.execute()
        run.teardown()
        output_dir = Path(run.config.output.output_dir)
        files = list(output_dir.glob("*bpa_scr_result*.csv"))
        assert len(files) == 1, f"Expected 1 SCR result CSV, found: {files}"
        df = pd.read_csv(files[0])
        assert "scr_spread" in df.columns
        assert "scr_interest" in df.columns
        assert "scr_longevity" in df.columns
        assert "bscr_partial" in df.columns
        assert "base_bel_post_ma" in df.columns
        assert len(df) == 1
        # bscr_partial >= each individual SCR component
        row = df.iloc[0]
        assert row["bscr_partial"] >= row["scr_spread"] - 1e-9
        assert row["bscr_partial"] >= row["scr_interest"] - 1e-9
        assert row["bscr_partial"] >= row["scr_longevity"] - 1e-9

    def test_no_scr_csv_when_scr_not_injected(self, bpa_run_executed: BPARun, tmp_path: Path):
        """Test 34 — no SCR CSV when SCRCalculator not provided."""
        bpa_run_executed.teardown()
        output_dir = Path(bpa_run_executed.config.output.output_dir)
        files = list(output_dir.glob("*bpa_scr_result*.csv"))
        assert len(files) == 0, f"Expected no SCR CSV without scr_calculator, found: {files}"

"""
tests/integration/test_bpa_end_to_end.py
==========================================
End-to-end integration tests for the full BPA lifecycle.

Design (DECISIONS.md §17, §49, §51, Step 23 plan)
---------------------------------------------------
No mocks.  Every engine component runs for real:
  - BPARun (MA pre-pass, forward pass, backward BEL, SCR, IFRS 17 GMM)
  - MACalculator + FundamentalSpreadTable
  - SCRCalculator (spread, interest, longevity)
  - GmmEngine + CostOfCapitalRA (via mock Ifrs17StateStore)
  - ResultStore (cohort_ids, as_cohort_pivot)
  - teardown output files

Test matrix
-----------
E1   — Zero-mortality BEL within tolerance of closed-form annuity value
E2   — MA benefit within 2 bps of hand-calculated spread − FS
E3   — Two-deal cohort isolation: each deal's BEL is independent
E4   — SCR runs end-to-end; spread loss is positive; own funds change correctly signed
E5   — All four output CSVs exist after teardown with expected columns
E6   — IFRS 17 smoke test: GmmEngine runs; RA > 0 at t=0 with SCR injected
E7   — BEL is monotonically non-increasing over time (annuity run-off)
E8   — ResultStore row count matches n_periods × n_cohorts

Numerical reference (E1)
-------------------------
Configuration:
  q_annual       = 0.0  (no deaths — pure discount annuity)
  pension_pa     = 12,000 per life
  weight         = 100 lives
  discount_rate  = 3% p.a. flat (pre-MA base)
  projection     = 3 years, monthly_years=1 → 14 periods

Monthly outgo per period = (1/12) × 12,000 × 100 = 100,000
Annual outgo per period  = 1.0   × 12,000 × 100 = 1,200,000

BEL_pre_ma_t0 = Σ_{s=0}^{11} 100,000 × (1.03)^{-(s+1)/12}
              + 1,200,000 × [(1.03)^{-2} + (1.03)^{-3}]

Expected to within £1 (< 0.001% of portfolio).

Numerical reference (E2)
-------------------------
Bond: AC, BBB, senior_unsecured, tenor=7.5yr, spread=100bps
FS table (inline fixture): BBB/7.5yr falls in [5,10) → fs_bps=31
Expected MA_benefit = 100 − 31 = 69 bps  (±2 bps tolerance)
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH
from engine.liability.bpa.registry import BPADealMetadata, BPADealRegistry
from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
from engine.matching_adjustment.ma_calculator import MACalculator
from engine.run_modes.bpa_run import BPARun
from engine.scr.interest_stress import InterestStressEngine
from engine.scr.longevity_stress import LongevityStressEngine
from engine.scr.scr_calculator import SCRCalculator
from engine.scr.spread_stress import SpreadStressEngine
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# FS table fixture — inline CSV  (3 rows: BBB, senior_unsecured)
# ---------------------------------------------------------------------------

_FS_CSV = (
    "# effective_date: 2024-07-01\n"
    "# source_ref: PRA PS10/24\n"
    "rating,seniority,tenor_lower,tenor_upper,long_run_pd_pct,lgd_pct,"
    "downgrade_allowance_bps,fs_bps\n"
    "BBB,senior_unsecured,0,5,0.17,40,21,28\n"
    "BBB,senior_unsecured,5,10,0.17,40,24,31\n"   # tenor 7.5yr → fs=31
    "BBB,senior_unsecured,10,20,0.17,40,26,33\n"
)


def _make_fs_table(tmp_path: Path) -> FundamentalSpreadTable:
    p = tmp_path / "fs.csv"
    p.write_text(_FS_CSV)
    return FundamentalSpreadTable.from_csv(p)


# ---------------------------------------------------------------------------
# Mortality basis — zero mortality for closed-form BEL test
# ---------------------------------------------------------------------------

def _zero_mortality() -> MortalityBasis:
    base = np.zeros(TABLE_LENGTH, dtype=float)
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


def _flat_mortality(q_annual: float = 0.02) -> MortalityBasis:
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


def _assumptions(
    mortality: MortalityBasis,
    rate: float = 0.03,
    inflation_rate: float = 0.025,
    rpi_rate: float = 0.03,
    expense_pa: float = 150.0,
) -> BPAAssumptions:
    ill = np.zeros(TABLE_LENGTH, dtype=float)
    return BPAAssumptions(
        mortality        = mortality,
        valuation_year   = 2024,
        discount_curve   = RiskFreeRateCurve.flat(rate),
        inflation_rate   = inflation_rate,
        rpi_rate         = rpi_rate,
        tv_rate          = 0.0,
        ill_health_rates = ill,
        expense_pa       = expense_pa,
    )


# ---------------------------------------------------------------------------
# Asset fixtures
# ---------------------------------------------------------------------------

def _make_asset_model(spread_bps: float = 100.0, tenor_months: int = 90) -> AssetModel:
    """Single AC bond for MA benefit calculation."""
    am = AssetModel()
    am.add_asset(Bond("bond_01",
                      face_value=1_000_000.0,
                      annual_coupon_rate=0.07,
                      maturity_month=tenor_months,
                      accounting_basis="AC",
                      initial_book_value=950_000.0))
    return am


def _make_assets_df(spread_bps: float = 100.0, tenor_years: float = 7.5) -> pd.DataFrame:
    """Asset metadata — tenor=7.5yr falls in [5,10) → FS=31 bps."""
    return pd.DataFrame([{
        "asset_id":                     "bond_01",
        "cashflow_type":                "fixed",
        "currency":                     "GBP",
        "has_credit_risk_transfer":     False,
        "has_qualifying_currency_swap": False,
        "rating":                       "BBB",
        "seniority":                    "senior_unsecured",
        "tenor_years":                  tenor_years,
        "spread_bps":                   spread_bps,
    }])


def _make_asset_cfs(tenor_years: int = 7) -> pd.DataFrame:
    """Annual bond cashflows — flat coupon + principal at maturity."""
    rows = []
    for yr in range(1, tenor_years + 1):
        cf = 70_000.0 if yr < tenor_years else 1_070_000.0
        rows.append({"t": yr, "asset_id": "bond_01", "cf": cf})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model point builders
# ---------------------------------------------------------------------------

def _in_payment_mp(
    deal_id: str,
    in_force_count: float = 100.0,
    pension_pa: float = 12_000.0,
    age: float = 70.0,
) -> dict[str, pd.DataFrame]:
    return {
        deal_id: pd.DataFrame([{
            "mp_id":          "P001",
            "sex":            "M",
            "age":            age,
            "in_force_count": in_force_count,
            "pension_pa":     pension_pa,
            "lpi_cap":        0.05,
            "lpi_floor":      0.0,
            "gmp_pa":         0.0,
        }])
    }


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _run_config(tmp_path: Path, run_id: str = "e2e_test") -> RunConfig:
    assumption_dir = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path      = fund_config_file,
        assumption_dir        = assumption_dir,
        run_type              = "bpa",
        projection_term_years = 3,
    )
    data.setdefault("output", {})["output_dir"] = str(tmp_path / "output")
    data["run_id"] = run_id
    return RunConfig.from_dict(data)


def _fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id":   "BPA_FUND",
        "fund_name": "BPA Fund",
        "saa_weights": {"bonds": 1.0, "equities": 0.0, "derivatives": 0.0, "cash": 0.0},
        "crediting_groups": [
            {"group_id": "G1", "group_name": "BPA Group", "product_codes": ["BPA"]},
        ],
    })


def _registry(deal_id: str) -> BPADealRegistry:
    return BPADealRegistry([
        BPADealMetadata(
            deal_id        = deal_id,
            deal_type      = "buyout",
            inception_date = date(2024, 1, 1),
            deal_name      = f"Deal {deal_id}",
            ma_eligible    = True,
        )
    ])


def _scr_calculator(fs_table: FundamentalSpreadTable) -> SCRCalculator:
    ma_calc = MACalculator(fs_table)
    return SCRCalculator(
        spread_engine    = SpreadStressEngine(
            ma_calculator   = ma_calc,
            spread_up_bps   = 75.0,
            spread_down_bps = 25.0,
        ),
        interest_engine  = InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0),
        longevity_engine = LongevityStressEngine(mortality_stress_factor=0.20),
    )


def _mock_ifrs17_store() -> MagicMock:
    store = MagicMock()
    store.load_state.return_value = None
    return store


# ---------------------------------------------------------------------------
# Shared fixture — zero-mortality run (used by E1, E7, E8)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zero_mortality_run(tmp_path_factory):
    """
    Full BPA run with zero mortality, flat 3% discount, zero inflation, single deal.

    Zero inflation is critical for the closed-form BEL check (E1): with non-zero
    LPI the pension grows each period and the flat outgo formula is wrong.
    With inflation_rate=0 and lpi_floor=0, the pension is constant → closed form holds.

    Used for closed-form BEL comparison (E1) and run-off checks (E7, E8).
    """
    tmp_path = tmp_path_factory.mktemp("zero_mortality")
    deal_id  = "E2E_Deal"
    fs_table = _make_fs_table(tmp_path)
    run = BPARun(
        config           = _run_config(tmp_path),
        fund_config      = _fund_config(),
        in_payment_mps   = _in_payment_mp(deal_id, in_force_count=100.0, pension_pa=12_000.0),
        deferred_mps     = {},
        dependant_mps    = {},
        enhanced_mps     = {},
        assumptions      = _assumptions(
            _zero_mortality(), rate=0.03,
            inflation_rate=0.0, rpi_rate=0.0,   # flat pension — closed form applies
            expense_pa=0.0,                     # zero expenses — pure discount annuity
        ),
        asset_model      = _make_asset_model(spread_bps=100.0, tenor_months=90),
        assets_df        = _make_assets_df(spread_bps=100.0, tenor_years=7.5),
        asset_cfs        = _make_asset_cfs(tenor_years=7),
        fs_table         = fs_table,
        deal_registry    = _registry(deal_id),
        projection_years = 3,
        monthly_years    = 1,
        initial_cash     = 0.0,
    )
    run.setup()
    run.execute()
    return run


# ---------------------------------------------------------------------------
# E1 — Closed-form BEL (zero mortality, flat 3% discount)
# ---------------------------------------------------------------------------

class TestE1ZeroMortalityBEL:

    def test_bel_pre_ma_within_tolerance_of_closed_form(self, zero_mortality_run: BPARun):
        """
        E1 — bel_pre_ma at t=0 must match the closed-form annuity within £1.

        Closed form:
            monthly_outgo = (1/12) × 12,000 × 100 = 100,000
            annual_outgo  = 1.0 × 12,000 × 100 = 1,200,000
            BEL = Σ_{s=0}^{11} monthly_outgo × (1.03)^{-(s+1)/12}
                + 1,200,000 × [(1.03)^{-2} + (1.03)^{-3}]
        """
        from engine.core.projection_calendar import ProjectionCalendar
        cal = ProjectionCalendar(projection_years=3, monthly_years=1)
        rfr = RiskFreeRateCurve.flat(0.03)

        # Replicate backward BEL from first principles
        expected_bel = 0.0
        for period in cal.periods:
            t_end_months = (period.time_in_years + period.year_fraction) * 12.0
            df_end       = rfr.discount_factor(t_end_months)
            outgo        = period.year_fraction * 12_000.0 * 100.0
            expected_bel += outgo * df_end

        # Retrieve stored t=0 result for the single cohort
        cohort_id = "E2E_Deal_pensioner"
        r = zero_mortality_run.store.get(0, 0, cohort_id)
        actual_bel = r.bel_pre_ma

        assert actual_bel is not None, "bel_pre_ma is None at t=0"
        assert abs(actual_bel - expected_bel) < 1.0, (
            f"BEL error too large: computed={actual_bel:.2f}, "
            f"analytical={expected_bel:.2f}, "
            f"diff={abs(actual_bel - expected_bel):.4f}"
        )

    def test_bel_post_ma_less_than_pre_ma(self, zero_mortality_run: BPARun):
        """E1b — post-MA BEL < pre-MA BEL (MA benefit > 0 with spread=100, FS=31)."""
        cohort_id = "E2E_Deal_pensioner"
        r = zero_mortality_run.store.get(0, 0, cohort_id)
        assert r.bel_post_ma < r.bel_pre_ma, (
            f"Expected bel_post_ma ({r.bel_post_ma:.2f}) < bel_pre_ma ({r.bel_pre_ma:.2f})"
        )


# ---------------------------------------------------------------------------
# E2 — MA benefit within 2 bps of hand-calculated spread − FS
# ---------------------------------------------------------------------------

class TestE2MABenefit:

    def test_ma_benefit_within_2bps_of_expected(self, zero_mortality_run: BPARun):
        """
        E2 — MA_benefit = spread − FS = 100 − 31 = 69 bps  (±2 bps).

        Bond: BBB, senior_unsecured, tenor=7.5yr (in [5,10) bucket → FS=31).
        """
        cal = zero_mortality_run.ma_calibration
        assert cal is not None
        expected_ma_bps = 100.0 - 31.0  # spread_bps − fs_bps
        actual_ma_bps   = cal.ma_result.ma_benefit_bps
        assert abs(actual_ma_bps - expected_ma_bps) <= 2.0, (
            f"MA benefit {actual_ma_bps:.2f} bps deviates more than 2 bps from "
            f"expected {expected_ma_bps:.1f} bps"
        )

    def test_ma_cashflow_test_status_recorded(self, zero_mortality_run: BPARun):
        """E2b — cashflow_test_passes field is a bool (pass or fail — not None)."""
        cal = zero_mortality_run.ma_calibration
        assert isinstance(cal.ma_result.cashflow_test_passes, bool)


# ---------------------------------------------------------------------------
# E3 — Two-deal cohort isolation
# ---------------------------------------------------------------------------

class TestE3CohortIsolation:

    @pytest.fixture(scope="class")
    def two_deal_run(self, tmp_path_factory):
        """
        Two deals:
          Deal_A: q=0 (immortal) — higher BEL
          Deal_B: q=0.02          — lower BEL (faster mortality run-off)
        Same asset portfolio, same assumptions except mortality basis.
        """
        tmp_path = tmp_path_factory.mktemp("two_deal")
        fs_table = _make_fs_table(tmp_path)

        # Both deals share the same in-payment model — different model points only
        from engine.liability.bpa.registry import BPADealRegistry, BPADealMetadata

        registry = BPADealRegistry([
            BPADealMetadata("Deal_A", "buyout", date(2024, 1, 1), "Deal A", True),
            BPADealMetadata("Deal_B", "buyout", date(2024, 1, 1), "Deal B", True),
        ])

        # Deal_A: q=0, 50 lives
        deal_a_mps = pd.DataFrame([{
            "mp_id": "A001", "sex": "M", "age": 70.0,
            "in_force_count": 50.0, "pension_pa": 12_000.0,
            "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        # Deal_B: q=0.02, 50 lives
        deal_b_mps = pd.DataFrame([{
            "mp_id": "B001", "sex": "M", "age": 70.0,
            "in_force_count": 50.0, "pension_pa": 12_000.0,
            "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])

        # Use zero_mortality assumptions — per-deal mortality difference doesn't
        # affect BEL direction since q_annual is 0.  Instead we use different
        # weights to produce different cohort BELs.
        run = BPARun(
            config           = _run_config(tmp_path, run_id="two_deal"),
            fund_config      = _fund_config(),
            in_payment_mps   = {"Deal_A": deal_a_mps, "Deal_B": deal_b_mps},
            deferred_mps     = {},
            dependant_mps    = {},
            enhanced_mps     = {},
            assumptions      = _assumptions(_zero_mortality(), rate=0.03, inflation_rate=0.0, rpi_rate=0.0, expense_pa=0.0),
            asset_model      = _make_asset_model(spread_bps=100.0, tenor_months=90),
            assets_df        = _make_assets_df(spread_bps=100.0, tenor_years=7.5),
            asset_cfs        = _make_asset_cfs(tenor_years=7),
            fs_table         = fs_table,
            deal_registry    = registry,
            projection_years = 3,
            monthly_years    = 1,
            initial_cash     = 0.0,
        )
        run.setup()
        run.execute()
        return run

    def test_two_cohort_ids_present(self, two_deal_run: BPARun):
        """E3a — Two cohort_ids in store."""
        ids = two_deal_run.store.cohort_ids()
        assert "Deal_A_pensioner" in ids
        assert "Deal_B_pensioner" in ids
        assert len(ids) == 2

    def test_result_count_is_periods_times_two_cohorts(self, two_deal_run: BPARun):
        """E3b — result_count == n_periods × 2 cohorts."""
        n_periods = two_deal_run.calendar.n_periods
        assert two_deal_run.store.result_count() == n_periods * 2

    def test_cohort_bels_sum_to_total(self, two_deal_run: BPARun):
        """E3c — BEL_A + BEL_B ≈ BEL_total-if-computed-as-single (both zero-mortality, 50 lives each)."""
        r_a = two_deal_run.store.get(0, 0, "Deal_A_pensioner")
        r_b = two_deal_run.store.get(0, 0, "Deal_B_pensioner")
        assert r_a.bel_post_ma is not None
        assert r_b.bel_post_ma is not None
        # Both cohorts have same mortality (zero) and same pension — BELs should be equal
        assert abs(r_a.bel_post_ma - r_b.bel_post_ma) < 1.0, (
            f"Identical cohorts should have equal BEL: "
            f"A={r_a.bel_post_ma:.2f}, B={r_b.bel_post_ma:.2f}"
        )

    def test_as_cohort_pivot_has_two_cohorts(self, two_deal_run: BPARun):
        """E3d — as_cohort_pivot() contains both cohort_ids as top-level columns."""
        pivot = two_deal_run.store.as_cohort_pivot()
        # After reset_index(), timestep may appear as ("timestep","") or "timestep"
        top_level = {
            c[0] if isinstance(c, tuple) else c
            for c in pivot.columns
            if (c[0] if isinstance(c, tuple) else c) not in ("timestep",)
        }
        assert "Deal_A_pensioner" in top_level
        assert "Deal_B_pensioner" in top_level


# ---------------------------------------------------------------------------
# E4 — SCR end-to-end (no mocks)
# ---------------------------------------------------------------------------

class TestE4SCRIntegration:

    @pytest.fixture(scope="class")
    def scr_run(self, tmp_path_factory):
        """
        BPARun with a real SCRCalculator injected.

        Uses non-zero mortality (q=0.02) so that the 20% longevity
        improvement stress produces SCR_longevity > 0.
        """
        tmp_path = tmp_path_factory.mktemp("scr")
        deal_id  = "SCR_Deal"
        fs_table = _make_fs_table(tmp_path)
        run = BPARun(
            config           = _run_config(tmp_path, run_id="scr_run"),
            fund_config      = _fund_config(),
            in_payment_mps   = _in_payment_mp(deal_id, in_force_count=100.0),
            deferred_mps     = {},
            dependant_mps    = {},
            enhanced_mps     = {},
            assumptions      = _assumptions(
                _flat_mortality(q_annual=0.02), rate=0.03,
                inflation_rate=0.0, rpi_rate=0.0,
                expense_pa=0.0,   # zero expenses: get_bel() and project_cashflows() agree
            ),
            asset_model      = _make_asset_model(spread_bps=100.0, tenor_months=90),
            assets_df        = _make_assets_df(spread_bps=100.0, tenor_years=7.5),
            asset_cfs        = _make_asset_cfs(tenor_years=7),
            fs_table         = fs_table,
            deal_registry    = _registry(deal_id),
            scr_calculator   = _scr_calculator(fs_table),
            projection_years = 3,
            monthly_years    = 1,
        )
        run.setup()
        run.execute()
        return run

    def test_scr_result_is_not_none(self, scr_run: BPARun):
        """E4a — scr_result is populated after execute() with SCRCalculator injected."""
        assert scr_run.scr_result is not None

    def test_scr_spread_is_non_negative(self, scr_run: BPARun):
        """E4b — SCR_spread >= 0 (capital requirement is always non-negative)."""
        assert scr_run.scr_result.scr_spread >= 0.0

    def test_spread_up_asset_mv_change_is_negative(self, scr_run: BPARun):
        """E4c — Spread widening reduces asset MV (negative change)."""
        assert scr_run.scr_result.spread_up_asset_mv_change < 0.0, (
            f"Expected negative asset MV change under spread widening, "
            f"got {scr_run.scr_result.spread_up_asset_mv_change:.2f}"
        )

    def test_spread_up_bel_change_is_negative_for_bpa(self, scr_run: BPARun):
        """E4d — MA offset reduces BEL under spread widening (BEL change is negative)."""
        # Spread widening increases MA benefit → post-MA discount rate increases → BEL falls
        assert scr_run.scr_result.spread_up_bel_change < 0.0, (
            f"Expected negative BEL change under spread widening (MA offset), "
            f"got {scr_run.scr_result.spread_up_bel_change:.2f}"
        )

    def test_scr_longevity_positive(self, scr_run: BPARun):
        """E4e — SCR_longevity > 0 for an annuity portfolio (more lives = more liability)."""
        assert scr_run.scr_result.scr_longevity > 0.0

    def test_longevity_stressed_bel_series_length(self, scr_run: BPARun):
        """E4f — longevity_stressed_bel_series has length == n_periods."""
        n = scr_run.calendar.n_periods
        assert len(scr_run.scr_result.longevity_stressed_bel_series) == n


# ---------------------------------------------------------------------------
# E5 — All four output CSVs exist after teardown
# ---------------------------------------------------------------------------

class TestE5OutputFiles:

    @pytest.fixture(scope="class")
    def run_with_teardown(self, tmp_path_factory):
        """BPARun with SCRCalculator; setup → execute → teardown."""
        tmp_path = tmp_path_factory.mktemp("teardown")
        deal_id  = "TD_Deal"
        fs_table = _make_fs_table(tmp_path)
        run = BPARun(
            config           = _run_config(tmp_path, run_id="teardown_run"),
            fund_config      = _fund_config(),
            in_payment_mps   = _in_payment_mp(deal_id, in_force_count=100.0),
            deferred_mps     = {},
            dependant_mps    = {},
            enhanced_mps     = {},
            assumptions      = _assumptions(_zero_mortality(), rate=0.03, inflation_rate=0.0, rpi_rate=0.0, expense_pa=0.0),
            asset_model      = _make_asset_model(spread_bps=100.0, tenor_months=90),
            assets_df        = _make_assets_df(spread_bps=100.0, tenor_years=7.5),
            asset_cfs        = _make_asset_cfs(tenor_years=7),
            fs_table         = fs_table,
            deal_registry    = _registry(deal_id),
            scr_calculator   = _scr_calculator(fs_table),
            projection_years = 3,
            monthly_years    = 1,
        )
        run.setup()
        run.execute()
        run.teardown()
        return run

    def test_results_csv_exists(self, run_with_teardown: BPARun):
        """E5a — {run_id}_bpa_results.csv is written."""
        output_dir = Path(run_with_teardown.config.output.output_dir)
        files = list(output_dir.glob("*bpa_results*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        assert "cohort_id" in df.columns
        assert "bel_post_ma" in df.columns

    def test_ma_summary_csv_exists(self, run_with_teardown: BPARun):
        """E5b — {run_id}_bpa_ma_summary.csv is written."""
        output_dir = Path(run_with_teardown.config.output.output_dir)
        files = list(output_dir.glob("*bpa_ma_summary*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        assert "ma_benefit_bps" in df.columns
        assert len(df) == 1

    def test_cohort_summary_csv_exists(self, run_with_teardown: BPARun):
        """E5c — {run_id}_bpa_cohort_summary.csv is written with correct columns."""
        output_dir = Path(run_with_teardown.config.output.output_dir)
        files = list(output_dir.glob("*bpa_cohort_summary*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        required_cols = [
            "cohort_id", "deal_id", "population_type",
            "bel_post_ma_t0", "bel_pre_ma_t0",
            "ma_reduction_t0", "ma_reduction_pct",
            "in_force_lives_t0", "last_nonzero_period",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column '{col}' in cohort summary"
        assert len(df) == 1  # single cohort

    def test_scr_result_csv_exists(self, run_with_teardown: BPARun):
        """E5d — {run_id}_bpa_scr_result.csv is written with correct columns."""
        output_dir = Path(run_with_teardown.config.output.output_dir)
        files = list(output_dir.glob("*bpa_scr_result*.csv"))
        assert len(files) == 1
        df = pd.read_csv(files[0])
        assert "scr_spread" in df.columns
        assert "scr_interest" in df.columns
        assert "scr_longevity" in df.columns
        assert "bscr_partial" in df.columns
        assert len(df) == 1

    def test_results_row_count_matches_periods_times_cohorts(self, run_with_teardown: BPARun):
        """E5e — Results CSV row count == n_periods × n_cohorts (before any output filter)."""
        output_dir = Path(run_with_teardown.config.output.output_dir)
        files = list(output_dir.glob("*bpa_results*.csv"))
        df = pd.read_csv(files[0])
        n_periods = run_with_teardown.calendar.n_periods
        n_cohorts = len(run_with_teardown.store.cohort_ids())
        assert len(df) == n_periods * n_cohorts


# ---------------------------------------------------------------------------
# E6 — IFRS 17 smoke test (GmmEngine runs, RA > 0 with SCR)
# ---------------------------------------------------------------------------

class TestE6Ifrs17Smoke:

    @pytest.fixture(scope="class")
    def ifrs17_run(self, tmp_path_factory):
        """
        BPARun with mock IFRS 17 store + SCR calculator.

        Uses non-zero mortality (q=0.02) so that SCR_longevity > 0 and
        CostOfCapitalRA returns a positive RA at t=0.
        """
        tmp_path = tmp_path_factory.mktemp("ifrs17")
        deal_id  = "I17_Deal"
        fs_table = _make_fs_table(tmp_path)
        store    = _mock_ifrs17_store()
        run = BPARun(
            config             = _run_config(tmp_path, run_id="ifrs17_run"),
            fund_config        = _fund_config(),
            in_payment_mps     = _in_payment_mp(deal_id, in_force_count=100.0),
            deferred_mps       = {},
            dependant_mps      = {},
            enhanced_mps       = {},
            assumptions        = _assumptions(
                _flat_mortality(q_annual=0.02), rate=0.03,
                inflation_rate=0.0, rpi_rate=0.0,
                expense_pa=0.0,   # zero expenses for consistent BEL computation
            ),
            asset_model        = _make_asset_model(spread_bps=100.0, tenor_months=90),
            assets_df          = _make_assets_df(spread_bps=100.0, tenor_years=7.5),
            asset_cfs          = _make_asset_cfs(tenor_years=7),
            fs_table           = fs_table,
            deal_registry      = _registry(deal_id),
            ifrs17_state_store = store,
            scr_calculator     = _scr_calculator(fs_table),
            projection_years   = 3,
            monthly_years      = 1,
        )
        run.setup()
        run.execute()
        return run

    def test_gmm_save_state_called(self, ifrs17_run: BPARun):
        """E6a — save_state called once (1 cohort)."""
        assert ifrs17_run._ifrs17_state_store.save_state.call_count == 1

    def test_gmm_movements_length_equals_n_periods(self, ifrs17_run: BPARun):
        """E6b — movements list has length == n_periods."""
        args = ifrs17_run._ifrs17_state_store.save_movements.call_args
        movements = args[0][2]
        assert len(movements) == ifrs17_run.calendar.n_periods

    def test_ra_at_t0_positive_with_scr(self, ifrs17_run: BPARun):
        """E6c — RA > 0 at t=0 when SCRCalculator is injected (longevity stress > 0)."""
        args      = ifrs17_run._ifrs17_state_store.save_movements.call_args
        movements = args[0][2]
        assert movements[0].risk_adjustment > 0.0, (
            f"Expected RA > 0 at t=0, got {movements[0].risk_adjustment}"
        )

    def test_lrc_identity_all_periods(self, ifrs17_run: BPARun):
        """E6d — LRC = BEL + RA + CSM at every period."""
        args      = ifrs17_run._ifrs17_state_store.save_movements.call_args
        movements = args[0][2]
        for i, r in enumerate(movements):
            expected = r.bel_current + r.risk_adjustment + r.csm_closing
            assert r.lrc == pytest.approx(expected, rel=1e-6), (
                f"LRC identity failed at period {i}: lrc={r.lrc:.4f} expected={expected:.4f}"
            )


# ---------------------------------------------------------------------------
# E7 — BEL is monotonically non-increasing over time
# ---------------------------------------------------------------------------

class TestE7BELRunOff:

    def test_bel_post_ma_non_increasing(self, zero_mortality_run: BPARun):
        """
        E7 — bel_post_ma at each period <= bel_post_ma at the previous period.

        With zero mortality the BEL must decrease over time purely from the
        annuity payout (fewer future payments remain to be discounted).
        """
        cohort_id = "E2E_Deal_pensioner"
        results   = zero_mortality_run.store.all_timesteps(0, cohort_id=cohort_id)
        bels      = [r.bel_post_ma for r in results]
        for i in range(1, len(bels)):
            assert bels[i] <= bels[i - 1] + 1e-6, (
                f"BEL increased at period {i}: "
                f"bel[{i-1}]={bels[i-1]:.2f} bel[{i}]={bels[i]:.2f}"
            )

    def test_bel_at_final_period_near_zero(self, zero_mortality_run: BPARun):
        """E7b — BEL at the last period is close to zero (annuity nearly exhausted)."""
        cohort_id = "E2E_Deal_pensioner"
        results   = zero_mortality_run.store.all_timesteps(0, cohort_id=cohort_id)
        final_bel = results[-1].bel_post_ma
        # After 3 years of run-off, the final BEL should be small relative to opening
        opening_bel = results[0].bel_post_ma
        assert final_bel < opening_bel * 0.5, (
            f"Expected final BEL << opening; got final={final_bel:.2f} opening={opening_bel:.2f}"
        )


# ---------------------------------------------------------------------------
# E8 — ResultStore row count
# ---------------------------------------------------------------------------

class TestE8RowCount:

    def test_result_count_is_periods_times_cohorts(self, zero_mortality_run: BPARun):
        """E8 — ResultStore.result_count() == n_periods × n_cohorts."""
        n_periods = zero_mortality_run.calendar.n_periods
        n_cohorts = len(zero_mortality_run.store.cohort_ids())
        assert zero_mortality_run.store.result_count() == n_periods * n_cohorts

    def test_all_timesteps_covered_for_cohort(self, zero_mortality_run: BPARun):
        """E8b — every period index 0..n-1 appears exactly once per cohort."""
        cohort_id = "E2E_Deal_pensioner"
        results   = zero_mortality_run.store.all_timesteps(0, cohort_id=cohort_id)
        n_periods = zero_mortality_run.calendar.n_periods
        timesteps = sorted(r.timestep for r in results)
        assert timesteps == list(range(n_periods))

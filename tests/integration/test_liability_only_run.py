"""
Integration tests for LiabilityOnlyRun.

These tests run the full projection loop end-to-end with hand-calculated
expected values.  They are the primary numerical validation for Step 5.

Test scenarios
--------------
Scenario A — Single group, 1-year projection, zero assumptions
    100 ENDOW_NONPAR policies, SA=10,000, annual_premium=1,200.
    policy_term_yr=1, policy_duration_mths=0 → remaining = 12 months.
    Zero mortality, zero lapses, zero expenses, zero discount rate.

    Expected values (hand-calculated):
        Months t=0..10:  no maturities
            premiums     = 1,200/12 × 100 = 10,000
            net_outgo    = −10,000  (premium income, no outgo)

        Month t=11:  final month, all 100 policies mature
            maturities   = 100
            premiums     = 10,000
            maturity_pmt = 10,000 × 100 = 1,000,000
            net_outgo    = 1,000,000 − 10,000 = 990,000

        BEL at t=0 (zero discount, 12 months remaining):
            = Σ net_outgo × DF(t)
            = 11 × (−10,000) × 1 + 990,000 × 1
            = −110,000 + 990,000 = 880,000

        BEL at t=11 (final month only):
            = 990,000 × 1 = 990,000

        Total premiums stored (all 12 months):
            = 10,000 × 12 = 120,000

Scenario B — Seriatim heterogeneous, 5-year policies at final month
    3 individual policies (in_force_count=1 each), all at their final month.
    Different SA, premium, and attained_age per policy — tests vectorised
    computation across heterogeneous rows.

    Policy data:
        P001: SA=10,000,  annual_prem=1,200, age=45, dur_mths=59, term_yr=5
        P002: SA=20,000,  annual_prem=2,400, age=50, dur_mths=59, term_yr=5
        P003: SA=5,000,   annual_prem=600,   age=55, dur_mths=59, term_yr=5

    Zero assumptions, 5-year projection term.

    Expected values at t=0 (all at final month):
        Maturities    = 1 + 1 + 1 = 3
        Maturity_pmts = 10,000 + 20,000 + 5,000 = 35,000
        Premiums      = (1,200 + 2,400 + 600) / 12 = 350
        Net outgo     = 35,000 − 350 = 34,650
        BEL (zero discount) = 34,650

    t=1 onwards: all policies expired → zero cash flows, zero BEL.

Scenario C — Mixed new-business portfolio, 10-year full-term projection
    Three groups all starting at duration=0 (new business), projected for
    10 years (120 months).  Each group is a different policy type with a
    different term to maturity, so they expire at different points.

    Group data:
        GRP_A: 100 ENDOW_NONPAR, SA=10,000, prem=1,200,  5yr term, dur=0
        GRP_B:  50 TERM,          SA=20,000, prem=2,400,  3yr term, dur=0
        GRP_C: 200 ENDOW_PAR,     SA= 5,000, prem=  600, 10yr term, dur=0

    Monthly premium per group: each = 10,000 (by design, for easy arithmetic).
    Zero mortality, lapses, expenses, discount rate, and PAR bonus rate.

    Lifecycle (zero decrements throughout):
        t= 0..35  (36 months): all three groups active → premiums = 30,000
        t=35:     GRP_B final month (TERM) → no maturity payment
        t=36..58  (23 months): GRP_A + GRP_C active → premiums = 20,000
        t=59:     GRP_A final month → maturity = 100 × 10,000 = 1,000,000
                  net_outgo = 1,000,000 − 20,000 = 980,000
        t=60..118 (59 months): GRP_C only → premiums = 10,000
        t=119:    GRP_C final month → maturity = 200 × 5,000 = 1,000,000
                  net_outgo = 1,000,000 − 10,000 = 990,000

    BEL at t=0 (zero discount, all 120 months summed):
        = 36×(−30,000) + 23×(−20,000) + 980,000 + 59×(−10,000) + 990,000
        = −1,080,000 − 460,000 + 980,000 − 590,000 + 990,000
        = −160,000

    Total premiums (by group):
        GRP_A: 60 months × 10,000 =   600,000
        GRP_B: 36 months × 10,000 =   360,000
        GRP_C: 120 months × 10,000 = 1,200,000
        Total                      = 2,160,000
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.run_modes.liability_only_run import LiabilityOnlyRun
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, *, projection_term_years: int = 1) -> RunConfig:
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=projection_term_years,
    )
    data["output"]["output_dir"] = str(tmp_path / "outputs")
    return RunConfig.from_dict(data)


def _make_fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id": "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 0.6, "equities": 0.3, "derivatives": 0.0, "cash": 0.1},
        "crediting_groups": [
            {"group_id": "GRP_A", "group_name": "Group A", "product_codes": ["P1"]},
        ],
    })


def _zero_assumptions() -> ConventionalAssumptions:
    return ConventionalAssumptions(
        mortality_rates={},
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(0.0),
    )


def _run_full(config, fund_config, mp, assumptions) -> LiabilityOnlyRun:
    run = LiabilityOnlyRun(
        config=config,
        fund_config=fund_config,
        model_points=mp,
        assumptions=assumptions,
    )
    run.run()
    return run


# ---------------------------------------------------------------------------
# Scenario A — single group, 1-year projection
# ---------------------------------------------------------------------------

class TestScenarioA:
    """
    100 ENDOW_NONPAR policies, SA=10,000, annual_premium=1,200.
    1-year term (12 months), starting from policy_duration_mths=0.
    Zero mortality, zero lapses, zero expenses, zero discount rate.
    """

    @pytest.fixture
    def mp_a(self) -> pd.DataFrame:
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
    def run_a(self, tmp_path, mp_a) -> LiabilityOnlyRun:
        config      = _make_config(tmp_path, projection_term_years=1)
        fund_config = _make_fund_config()
        return _run_full(config, fund_config, mp_a, _zero_assumptions())

    def test_result_count_is_12(self, run_a):
        assert run_a.store.result_count() == 12

    def test_bel_at_t0_matches_manual_calculation(self, run_a):
        """
        BEL at t=0 (zero discount, 12 months remaining):
            11 × (−10,000) + 990,000 = 880,000
        """
        r = run_a.store.get(scenario_id=0, timestep=0)
        assert r.bel == pytest.approx(880_000.0, rel=1e-6)

    def test_bel_at_t11_is_990000(self, run_a):
        """
        BEL at t=11 (final month, zero discount):
            net_outgo = 1,000,000 − 10,000 = 990,000
            BEL = 990,000 × DF(1) = 990,000 × 1.0 = 990,000
        """
        r = run_a.store.get(scenario_id=0, timestep=11)
        assert r.bel == pytest.approx(990_000.0, rel=1e-6)

    def test_bel_increases_as_maturity_approaches(self, run_a):
        """
        With zero discount and pure premium income before final month,
        BEL must increase as the large maturity payment gets closer.
        """
        bels = [run_a.store.get(0, t).bel for t in range(12)]
        for i in range(1, len(bels)):
            assert bels[i] >= bels[i - 1]

    def test_total_premiums_across_all_months(self, run_a):
        """
        100 policies × (1,200/12) per month × 12 months = 120,000.
        """
        df    = run_a.store.as_dataframe()
        total = df["premiums"].sum()
        assert total == pytest.approx(120_000.0, rel=1e-6)

    def test_premiums_constant_each_month(self, run_a):
        """
        With zero decrements, premiums are constant at 10,000 per month.
        Uses max absolute difference — pytest.approx does not work with Series.
        """
        df = run_a.store.as_dataframe()
        assert (df["premiums"] - 10_000.0).abs().max() < 1e-6

    def test_maturity_only_in_final_month(self, run_a):
        """
        Months t=0..10: maturity_payments = 0.
        Month t=11:     maturity_payments = 1,000,000.
        """
        df        = run_a.store.as_dataframe()
        non_final = df[df["timestep"] < 11]
        final     = df[df["timestep"] == 11]
        assert non_final["maturity_payments"].abs().max() < 1e-6
        assert final["maturity_payments"].iloc[0] == pytest.approx(1_000_000.0, rel=1e-6)

    def test_in_force_stays_100_with_zero_decrements(self, run_a):
        """Zero mortality and lapses → IF_start = 100 throughout."""
        df = run_a.store.as_dataframe()
        assert (df["in_force_start"] - 100.0).abs().max() < 1e-6

    def test_reserve_equals_bel(self, run_a):
        """Phase 1: reserve must equal BEL at every timestep."""
        for t in range(12):
            r = run_a.store.get(0, t)
            assert r.reserve == pytest.approx(r.bel)

    def test_output_csv_row_count_matches_store(self, tmp_path, mp_a):
        """Teardown CSV must have the same number of rows as the store."""
        config      = _make_config(tmp_path, projection_term_years=1)
        fund_config = _make_fund_config()
        run         = _run_full(config, fund_config, mp_a, _zero_assumptions())
        path = (
            Path(config.output.output_dir)
            / "Test_Run_liability_only_results.csv"
        )
        df = pd.read_csv(path)
        assert len(df) == run.store.result_count()


# ---------------------------------------------------------------------------
# Scenario B — seriatim heterogeneous policies at final month
# ---------------------------------------------------------------------------

class TestScenarioBSeriatim:
    """
    3 individual policies (in_force_count=1), each at their own final month.
    All different: different SA, premium, and attained_age.

    This test validates that vectorised calculation handles heterogeneous
    rows correctly — each row uses its own SA and premium in the formula.
    """

    @pytest.fixture
    def mp_b(self) -> pd.DataFrame:
        """
        3 seriatim-style rows.  All ENDOW_NONPAR, all at final month.

        P001: SA=10,000,  prem=1,200, age=45, dur=59, term=5 → remaining=1
        P002: SA=20,000,  prem=2,400, age=50, dur=59, term=5 → remaining=1
        P003: SA= 5,000,  prem=  600, age=55, dur=59, term=5 → remaining=1
        """
        return pd.DataFrame([
            {
                "group_id": "P001", "in_force_count": 1.0,
                "sum_assured": 10_000.0, "annual_premium": 1_200.0,
                "attained_age": 45, "policy_code": "ENDOW_NONPAR",
                "policy_term_yr": 5, "policy_duration_mths": 59,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "P002", "in_force_count": 1.0,
                "sum_assured": 20_000.0, "annual_premium": 2_400.0,
                "attained_age": 50, "policy_code": "ENDOW_NONPAR",
                "policy_term_yr": 5, "policy_duration_mths": 59,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "P003", "in_force_count": 1.0,
                "sum_assured": 5_000.0, "annual_premium": 600.0,
                "attained_age": 55, "policy_code": "ENDOW_NONPAR",
                "policy_term_yr": 5, "policy_duration_mths": 59,
                "accrued_bonus_per_policy": 0.0,
            },
        ])

    @pytest.fixture
    def run_b(self, tmp_path, mp_b) -> LiabilityOnlyRun:
        config      = _make_config(tmp_path, projection_term_years=5)
        fund_config = _make_fund_config()
        return _run_full(config, fund_config, mp_b, _zero_assumptions())

    def test_result_count_is_60(self, run_b):
        """5-year projection → 60 results."""
        assert run_b.store.result_count() == 60

    def test_total_maturity_payments_at_t0(self, run_b):
        """
        All 3 policies mature at t=0.
        Total maturity = 10,000 + 20,000 + 5,000 = 35,000.
        """
        r = run_b.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.maturity_payments == pytest.approx(35_000.0, rel=1e-6)

    def test_total_premiums_at_t0(self, run_b):
        """
        Monthly premiums = (1,200 + 2,400 + 600) / 12 = 350.
        """
        r = run_b.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.premiums == pytest.approx(350.0, rel=1e-6)

    def test_net_outgo_at_t0(self, run_b):
        """
        net_outgo = 35,000 − 350 = 34,650.
        """
        r = run_b.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.net_outgo == pytest.approx(34_650.0, rel=1e-6)

    def test_bel_at_t0_equals_net_outgo(self, run_b):
        """
        1 month remaining, zero discount → BEL = net_outgo × 1.0 = 34,650.
        """
        r = run_b.store.get(scenario_id=0, timestep=0)
        assert r.bel == pytest.approx(34_650.0, rel=1e-6)

    def test_cashflows_zero_after_policies_expire(self, run_b):
        """
        All 3 policies expire at t=0.  From t=1 onwards all cash flows = 0.
        """
        for t in range(1, 60):
            r = run_b.store.get(0, t)
            assert r.cashflows.net_outgo == pytest.approx(0.0, abs=1e-9)

    def test_bel_zero_after_policies_expire(self, run_b):
        """BEL = 0 for t ≥ 1 — no future cash flows remain."""
        for t in range(1, 60):
            r = run_b.store.get(0, t)
            assert r.bel == pytest.approx(0.0, abs=1e-9)

    def test_total_maturities_at_t0(self, run_b):
        """
        3 in-force-count-1 policies all mature → total maturities = 3.0.
        """
        r = run_b.store.get(0, 0)
        assert r.decrements.maturities == pytest.approx(3.0, rel=1e-6)

    def test_each_policy_contributes_its_own_sa(self, run_b):
        """
        Maturity payments = Σ SA_i × maturities_i.
        P001: 10,000×1 + P002: 20,000×1 + P003: 5,000×1 = 35,000.
        Confirms vectorised row-wise SA lookup is correct.
        """
        r = run_b.store.get(0, 0)
        assert r.cashflows.maturity_payments == pytest.approx(35_000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Scenario C — mixed new-business portfolio, 10-year full-term projection
# ---------------------------------------------------------------------------

class TestScenarioC:
    """
    Three policy groups, all new business (duration=0), different types and
    term lengths.  Projected for 10 years so every group reaches its own
    natural expiry within the projection window.

    GRP_A: 100 ENDOW_NONPAR, SA=10,000, prem=1,200,  5yr term, dur=0
    GRP_B:  50 TERM,          SA=20,000, prem=2,400,  3yr term, dur=0
    GRP_C: 200 ENDOW_PAR,     SA= 5,000, prem=  600, 10yr term, dur=0

    Monthly premium per group = 10,000 (by design).
    Zero mortality, lapses, expenses, discount rate, and PAR bonus rate.
    """

    @pytest.fixture
    def mp_c(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "group_id": "GRP_A", "in_force_count": 100.0,
                "sum_assured": 10_000.0, "annual_premium": 1_200.0,
                "attained_age": 40, "policy_code": "ENDOW_NONPAR",
                "policy_term_yr": 5, "policy_duration_mths": 0,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "GRP_B", "in_force_count": 50.0,
                "sum_assured": 20_000.0, "annual_premium": 2_400.0,
                "attained_age": 35, "policy_code": "TERM",
                "policy_term_yr": 3, "policy_duration_mths": 0,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "GRP_C", "in_force_count": 200.0,
                "sum_assured": 5_000.0, "annual_premium": 600.0,
                "attained_age": 30, "policy_code": "ENDOW_PAR",
                "policy_term_yr": 10, "policy_duration_mths": 0,
                "accrued_bonus_per_policy": 0.0,
            },
        ])

    @pytest.fixture
    def run_c(self, tmp_path, mp_c) -> LiabilityOnlyRun:
        config      = _make_config(tmp_path, projection_term_years=10)
        fund_config = _make_fund_config()
        return _run_full(config, fund_config, mp_c, _zero_assumptions())

    def test_result_count_is_120(self, run_c):
        """10-year projection → 120 monthly results."""
        assert run_c.store.result_count() == 120

    def test_premiums_at_t0_all_groups_active(self, run_c):
        """
        All three groups active at t=0:
            GRP_A: 100 × 1,200/12 = 10,000
            GRP_B:  50 × 2,400/12 = 10,000
            GRP_C: 200 ×   600/12 = 10,000
            Total                  = 30,000
        """
        r = run_c.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.premiums == pytest.approx(30_000.0, rel=1e-6)

    def test_premiums_at_t36_grp_b_expired(self, run_c):
        """
        GRP_B (3yr TERM) is zeroed after t=35.  At t=36 only GRP_A and GRP_C
        contribute: 10,000 + 10,000 = 20,000.
        """
        r = run_c.store.get(scenario_id=0, timestep=36)
        assert r.cashflows.premiums == pytest.approx(20_000.0, rel=1e-6)

    def test_premiums_at_t60_grp_a_expired(self, run_c):
        """
        GRP_A (5yr ENDOW_NONPAR) is zeroed after t=59.  At t=60 only GRP_C
        contributes: 10,000.
        """
        r = run_c.store.get(scenario_id=0, timestep=60)
        assert r.cashflows.premiums == pytest.approx(10_000.0, rel=1e-6)

    def test_term_policy_has_no_maturity_payment(self, run_c):
        """
        GRP_B is TERM — no maturity benefit is ever paid.
        At t=35 (GRP_B's final month) maturity_payments must reflect only
        any ENDOW groups that happen to mature that same month (none do).
        """
        r = run_c.store.get(scenario_id=0, timestep=35)
        # Only GRP_B is at its final month; it is TERM → no maturity
        assert r.cashflows.maturity_payments == pytest.approx(0.0, abs=1e-9)

    def test_grp_a_matures_at_t59(self, run_c):
        """
        GRP_A: 5yr ENDOW_NONPAR, duration=0 → final month at t=59.
        maturity_payments = 100 × 10,000 = 1,000,000.
        """
        r = run_c.store.get(scenario_id=0, timestep=59)
        assert r.cashflows.maturity_payments == pytest.approx(1_000_000.0, rel=1e-6)

    def test_grp_c_matures_at_t119(self, run_c):
        """
        GRP_C: 10yr ENDOW_PAR (bonus_rate=0), duration=0 → final month at t=119.
        maturity_payments = 200 × 5,000 = 1,000,000.
        """
        r = run_c.store.get(scenario_id=0, timestep=119)
        assert r.cashflows.maturity_payments == pytest.approx(1_000_000.0, rel=1e-6)

    def test_no_maturity_before_t59(self, run_c):
        """No group reaches its final month before t=59."""
        df = run_c.store.as_dataframe()
        early = df[df["timestep"] < 59]
        assert early["maturity_payments"].abs().max() < 1e-6

    def test_bel_at_t0_matches_hand_calculation(self, run_c):
        """
        BEL at t=0 (zero discount, all 120 months summed):
            36 × (−30,000)  =  −1,080,000
            23 × (−20,000)  =    −460,000
              1 ×   980,000  =     980,000
            59 × (−10,000)  =    −590,000
              1 ×   990,000  =     990,000
            Total            =    −160,000
        """
        r = run_c.store.get(scenario_id=0, timestep=0)
        assert r.bel == pytest.approx(-160_000.0, rel=1e-6)

    def test_bel_at_t119_is_final_net_outgo(self, run_c):
        """
        At t=119 (GRP_C final month, zero discount):
            BEL = net_outgo × DF(1) = 990,000 × 1.0 = 990,000.
        """
        r = run_c.store.get(scenario_id=0, timestep=119)
        assert r.bel == pytest.approx(990_000.0, rel=1e-6)

    def test_bel_increases_in_final_60_months(self, run_c):
        """
        From t=60 to t=119, only GRP_C remains and approaches maturity.
        With zero discount, BEL must strictly increase each month as one
        premium payment drops off and the maturity payment gets closer.

        BEL at t=k (60 ≤ k ≤ 119) = (119−k) × (−10,000) + 990,000.
        """
        bels = [run_c.store.get(0, t).bel for t in range(60, 120)]
        for i in range(1, len(bels)):
            assert bels[i] > bels[i - 1]

    def test_total_premiums_match_full_lifecycle(self, run_c):
        """
        GRP_A: 60 months × 10,000 =   600,000
        GRP_B: 36 months × 10,000 =   360,000
        GRP_C: 120 months × 10,000 = 1,200,000
        Total                      = 2,160,000
        """
        df    = run_c.store.as_dataframe()
        total = df["premiums"].sum()
        assert total == pytest.approx(2_160_000.0, rel=1e-6)

    def test_total_maturity_payments_two_groups(self, run_c):
        """
        GRP_A: 1,000,000 at t=59
        GRP_C: 1,000,000 at t=119
        GRP_B: 0 (TERM policy)
        Total = 2,000,000
        """
        df    = run_c.store.as_dataframe()
        total = df["maturity_payments"].sum()
        assert total == pytest.approx(2_000_000.0, rel=1e-6)

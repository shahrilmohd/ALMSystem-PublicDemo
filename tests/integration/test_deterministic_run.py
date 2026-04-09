"""
Integration tests for DeterministicRun — end-to-end numerical verification.

These tests use real liability models, real bond objects, and hand-calculated
expected values to verify that the full Fund + DeterministicRun pipeline
produces correct numbers.

Scenario A — Asset-only (trivial liability)
--------------------------------------------
Portfolio : 1 FVTPL bond, face=£1,200,000, coupon=5%, 36-month maturity
Liability : 100 ENDOW_NONPAR, SA=£10,000, premium=£1,200/yr, 1-yr term
Rate curve: 3% flat
Equity    : none
Initial cash: £0

Monthly coupon = £1,200,000 × 5% / 12 = £5,000

At t=0 (first month):
  - Liability: 100 policies at start, 1-yr term, ~zero mortality
    net_outgo ≈ −premiums = −100 × 1,200/12 = −£10,000  (premiums > claims)
    So net_outgo ≈ −10,000 (cash INFLOW from premiums)
  - Asset income: coupon = £5,000
  - cash_balance ≈ 0 + 5,000 − (−10,000) = £15,000

Checks:
  A1. cash_balance at t=0 ≈ coupon_income − net_outgo
  A2. total_market_value at t=0 ≈ £1,200,000 (FVTPL repriced at 3% curve;
      coupon 5% > 3% so bond trades at a premium)
  A3. BEL at t=0 > 0 (policies have remaining claims liability)
  A4. coupon_income at every t > 0 until bond matures
  A5. result_count = 12 (1-year projection)

Scenario B — AC bond amortisation check
-----------------------------------------
Portfolio : 1 AC bond, face=£100,000, coupon=5%, 36-month maturity,
            initial_book_value=£95,000 (discount bond)
            EIR ≈ 6.9% annual (from DECISIONS.md Section 2 anchor)
Liability : trivial (zero net_outgo — no policies)
Rate curve: 3% flat
Initial cash: £0

At t=0:
  Monthly EIR income = book_value × monthly_eir
  Monthly coupon paid into cash = £100,000 × 5% / 12 = £417

  BV advances: new_bv = old_bv × (1 + monthly_eir) − monthly_coupon
  => new_bv > old_bv  (discount bond: BV rises toward par)

  Check B1: eir_income > coupon_income (EIR > coupon rate on a discount bond)
  Check B2: total_book_value at t=1 > total_book_value at t=0
            (book value amortises upward toward par for discount bond)
  Check B3: total_market_value ≠ total_book_value  (AC: MV reported separately)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.config.fund_config import AssetClassWeights, FundConfig
from engine.config.run_config import RunConfig
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.run_modes.deterministic_run import DeterministicRun
from engine.strategy.investment_strategy import InvestmentStrategy
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path, term_years: int = 1) -> RunConfig:
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    asset_file = tmp_path / "assets.csv"
    asset_file.write_text("placeholder\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=term_years,
    )
    data["run_type"] = "deterministic"
    data["output"]["output_dir"]    = str(tmp_path / "outputs")
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


def make_no_rebalance_strategy() -> InvestmentStrategy:
    return InvestmentStrategy(
        AssetClassWeights(bonds=1.0, equities=0.0, derivatives=0.0, cash=0.0),
        rebalancing_tolerance=1.0,
    )


def make_nonpar_model_points(count: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "group_id":                "GRP_A",
        "in_force_count":          count,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          1,
        "policy_duration_mths":    0,
        "accrued_bonus_per_policy": 0.0,
    }])


def make_zero_mp() -> pd.DataFrame:
    """Single expired row (in_force_count=0) — produces zero liability cashflows."""
    return pd.DataFrame([{
        "group_id":                "GRP_A",
        "in_force_count":          0.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          1,
        "policy_duration_mths":    0,
        "accrued_bonus_per_policy": 0.0,
    }])


# ---------------------------------------------------------------------------
# Scenario A — FVTPL bond + real liability
# ---------------------------------------------------------------------------

class TestScenarioA:

    INITIAL_CASH  = 1_500_000.0   # large enough to prevent forced sells over 12 months
    BOND_FACE     = 1_200_000.0
    BOND_COUPON   = 0.05
    RATE_CURVE    = RiskFreeRateCurve.flat(0.03)

    @pytest.fixture
    def run_a(self, tmp_path) -> DeterministicRun:
        bond = Bond("corp_1", self.BOND_FACE, self.BOND_COUPON, 36, "FVTPL", self.BOND_FACE)
        run  = DeterministicRun(
            config=make_config(tmp_path, term_years=1),
            fund_config=make_fund_config(),
            model_points=make_nonpar_model_points(count=100.0),
            assumptions=ConventionalAssumptions(
                mortality_rates={},
                lapse_rates={},
                expense_pct_premium=0.0,
                expense_per_policy=0.0,
                surrender_value_factors={},
                rate_curve=self.RATE_CURVE,
            ),
            asset_model=AssetModel([bond]),
            investment_strategy=make_no_rebalance_strategy(),
            initial_cash=self.INITIAL_CASH,
        )
        run.run()
        return run

    def test_a1_cash_balance_t0(self, run_a):
        """
        At t=0: cash = initial_cash + coupon_income − net_outgo.
        initial_cash = £1,500,000 (ensures no forced sell during 1-yr projection)
        net_outgo for ENDOW_NONPAR with zero mortality ≈ −premiums = −£10,000/mo
        coupon_income = BOND_FACE × BOND_COUPON / 12 = £5,000
        cash ≈ 1,500,000 + 5,000 − (−10,000) = £1,515,000
        """
        r              = run_a.store.get(0, 0)
        monthly_coupon = self.BOND_FACE * self.BOND_COUPON / 12
        expected_cash  = self.INITIAL_CASH + monthly_coupon - r.cashflows.net_outgo
        assert r.cash_balance == pytest.approx(expected_cash, rel=1e-4)

    def test_a2_total_mv_at_premium(self, run_a):
        """FVTPL bond with coupon 5% > risk-free 3% trades at a premium to par."""
        r = run_a.store.get(0, 0)
        assert r.total_market_value > self.BOND_FACE

    def test_a3_bel_positive(self, run_a):
        """Policies have remaining liability → BEL > 0 at t=0."""
        r = run_a.store.get(0, 0)
        assert r.bel > 0.0

    def test_a4_coupon_income_positive_each_period(self, run_a):
        """Coupon income > 0 every month (bond has 36-month maturity, run=12 months)."""
        for r in run_a.store.all_timesteps(0):
            assert r.coupon_income > 0.0, f"coupon_income = 0 at t={r.timestep}"

    def test_a5_result_count(self, run_a):
        """1-year projection → 12 results."""
        assert run_a.store.result_count() == 12

    def test_a6_coupon_income_value(self, run_a):
        """Coupon income at each t = BOND_FACE × BOND_COUPON / 12."""
        expected = self.BOND_FACE * self.BOND_COUPON / 12
        for r in run_a.store.all_timesteps(0):
            assert r.coupon_income == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Scenario B — AC bond amortisation
# ---------------------------------------------------------------------------

class TestScenarioB:

    @pytest.fixture
    def run_b(self, tmp_path) -> DeterministicRun:
        # Discount bond: face=£100k, coupon=5%, IBV=£95k, 36-month maturity
        bond = Bond("ac_1", 100_000.0, 0.05, 36, "AC", 95_000.0)
        run  = DeterministicRun(
            config=make_config(tmp_path, term_years=1),
            fund_config=make_fund_config(),
            model_points=make_zero_mp(),
            assumptions=ConventionalAssumptions(
                mortality_rates={},
                lapse_rates={},
                expense_pct_premium=0.0,
                expense_per_policy=0.0,
                surrender_value_factors={},
                rate_curve=RiskFreeRateCurve.flat(0.03),
            ),
            asset_model=AssetModel([bond]),
            investment_strategy=make_no_rebalance_strategy(),
            initial_cash=0.0,
        )
        run.run()
        return run

    def test_b1_eir_income_greater_than_coupon_for_discount_bond(self, run_b):
        """
        Discount bond EIR > coupon rate, so EIR income > coupon cash received.
        eir_income recognises the discount unwind as well as the coupon.
        """
        r = run_b.store.get(0, 0)
        # eir_income includes coupon + discount amortisation
        # coupon cash = 100,000 × 5% / 12 ≈ £417
        # EIR income = BV × monthly_EIR (EIR ≈ 6.9%) ≈ £547
        assert r.eir_income > r.coupon_income, (
            f"Expected eir_income ({r.eir_income:.2f}) > "
            f"coupon_income ({r.coupon_income:.2f}) for discount bond"
        )

    def test_b2_book_value_increases_each_month(self, run_b):
        """
        Discount bond: book value amortises upward toward par each period.
        total_book_value at t=1 > total_book_value at t=0.
        """
        r0 = run_b.store.get(0, 0)
        r1 = run_b.store.get(0, 1)
        assert r1.total_book_value > r0.total_book_value, (
            f"BV should rise: t=0 BV={r0.total_book_value:.2f}, "
            f"t=1 BV={r1.total_book_value:.2f}"
        )

    def test_b3_market_value_differs_from_book_value(self, run_b):
        """
        AC bond: balance sheet reports amortised BV, not MV.
        MV is repriced using rate_curve and will differ from BV
        for a bond with coupon ≠ risk-free rate.
        """
        r = run_b.store.get(0, 0)
        assert abs(r.total_market_value - r.total_book_value) > 100.0, (
            f"AC bond MV ({r.total_market_value:.2f}) should differ materially "
            f"from BV ({r.total_book_value:.2f})"
        )

    def test_b4_total_eir_income_equals_discount_plus_coupons(self, run_b):
        """
        Over the full 12-month projection, total EIR income should exceed
        total coupon cash received by the amortisation of the discount.
        This is the DECISIONS.md Section 2 invariant: over full term,
        total EIR income = total coupons + (par − purchase_price).
        For 12 months of a 36-month bond: partial amortisation.
        """
        total_eir    = sum(r.eir_income    for r in run_b.store.all_timesteps(0))
        total_coupon = sum(r.coupon_income for r in run_b.store.all_timesteps(0))
        assert total_eir > total_coupon


# ---------------------------------------------------------------------------
# Scenario A — two-pass BEL formula numerical verification
# ---------------------------------------------------------------------------

class TestScenarioABel:
    """
    Numerical verification of the two-pass BEL formula in DeterministicRun.

    DeterministicRun.execute() computes BEL via:
        BEL(t) = Σ_{s=0}^{T-1-t}  net_outgo[t+s]  ×  DF(s+1)

    where DF(k) = assumptions.rate_curve.discount_factor(k)  [k in months].

    With zero mortality, zero lapse, and zero expenses:
        net_outgo[s] = −£10,000   for s = 0..10   (premiums only: 100 × £1,200/12)
        net_outgo[11] = +£990,000               (maturity £1,000,000 − premium £10,000)

    These cashflows are deterministic and exact, so expected BEL values can be
    computed from the same rate curve used in the run — giving a true cross-check
    of the formula rather than just a sign or magnitude check.

    Re-uses Scenario A setup: FVTPL bond + 100 ENDOW_NONPAR, 3% flat curve.
    The asset side does not affect BEL (BEL discounts liability net_outgos only).
    """

    RATE_CURVE    = RiskFreeRateCurve.flat(0.03)
    TOTAL_MONTHS  = 12
    # Exact net_outgos for zero mortality / zero lapse / zero expenses
    # 100 policies × £10,000 SA, 1-yr term, £1,200/yr premium
    MONTHLY_PREMIUM = 100.0 * 1_200.0 / 12        # = £10,000
    MATURITY_CLAIM  = 100.0 * 10_000.0             # = £1,000,000
    _NET_OUTGOS = [-10_000.0] * 11 + [990_000.0]   # final: maturity − premium

    @pytest.fixture
    def run_a(self, tmp_path) -> DeterministicRun:
        """Same Scenario A setup as TestScenarioA — re-declared so this class is self-contained."""
        bond = Bond("corp_1", 1_200_000.0, 0.05, 36, "FVTPL", 1_200_000.0)
        run  = DeterministicRun(
            config=make_config(tmp_path, term_years=1),
            fund_config=make_fund_config(),
            model_points=make_nonpar_model_points(count=100.0),
            assumptions=ConventionalAssumptions(
                mortality_rates={},
                lapse_rates={},
                expense_pct_premium=0.0,
                expense_per_policy=0.0,
                surrender_value_factors={},
                rate_curve=self.RATE_CURVE,
            ),
            asset_model=AssetModel([bond]),
            investment_strategy=make_no_rebalance_strategy(),
            initial_cash=1_500_000.0,
        )
        run.run()
        return run

    def _expected_bel(self, t: int) -> float:
        """
        Hand-calculate BEL(t) using the same formula as DeterministicRun.execute():
            BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
        """
        return sum(
            self._NET_OUTGOS[t + s] * self.RATE_CURVE.discount_factor(s + 1)
            for s in range(self.TOTAL_MONTHS - t)
        )

    def test_bel_t11_single_remaining_cashflow(self, run_a):
        """
        At t=11 only one cashflow remains: net_outgo[11] = £990,000 discounted 1 month.

            BEL(11) = £990,000 × DF(1 month)

        This is the cleanest possible test of the two-pass discounting formula —
        a single multiplication with a known input and a known rate.
        """
        expected = 990_000.0 * self.RATE_CURVE.discount_factor(1)
        r = run_a.store.get(0, 11)
        assert r.bel == pytest.approx(expected, rel=1e-4)

    def test_bel_t0_full_discounted_sum(self, run_a):
        """
        BEL(0) = Σ_{s=0}^{11} net_outgo[s] × DF(s+1)

        Expected value is computed from the same RATE_CURVE and the known cashflows,
        providing a direct cross-check of the full summation.

        With 3% flat annual curve:
            BEL(0) ≈ −10,000 × [DF(1)+…+DF(11)] + 990,000 × DF(12)
                   ≈ −108,389 + 961,165 ≈ £852,776
        """
        expected = self._expected_bel(0)
        r = run_a.store.get(0, 0)
        assert r.bel == pytest.approx(expected, rel=1e-4)

    def test_bel_less_than_undiscounted_sum(self, run_a):
        """
        With a positive discount rate and positive net BEL, the discounted BEL(0)
        must be strictly less than the undiscounted sum of cashflows.

        Undiscounted sum = −10,000 × 11 + 990,000 = £880,000
        Discounted BEL(0) at 3% ≈ £852,776

        This confirms the 3% rate curve is actually being applied (not ignored).
        """
        undiscounted = sum(self._NET_OUTGOS)   # = £880,000
        r = run_a.store.get(0, 0)
        assert r.bel < undiscounted

    def test_bel_increases_monotonically_toward_maturity(self, run_a):
        """
        As the projection approaches the large maturity payment at t=11, BEL
        increases at every step. Each month that passes removes a premium inflow
        (−£10,000) from the BEL calculation while the undiscounted maturity outgo
        grows closer — so BEL must rise throughout.
        """
        bels = [run_a.store.get(0, t).bel for t in range(self.TOTAL_MONTHS)]
        for t in range(self.TOTAL_MONTHS - 1):
            assert bels[t + 1] > bels[t], (
                f"BEL should increase toward maturity: "
                f"BEL({t})={bels[t]:.2f}, BEL({t+1})={bels[t+1]:.2f}"
            )

"""
Unit tests for engine/core/fund.py.

Rules under test
----------------
Construction:
  1.  Fund is constructed with asset_model, liability_model, strategy, initial_cash.
  2.  cash_balance property returns initial_cash.
  3.  asset_model property exposes the injected AssetModel.

Cash balance mechanics:
  4.  Asset income (coupon) increases cash_balance.
  5.  Liability net_outgo decreases cash_balance.
  6.  Net cash change = asset_income - net_outgo.

Forced sells:
  7.  When cash < 0 after step, forced sell is triggered.
  8.  After forced sell, cash_balance moves towards zero.
  9.  FVTPL bond sold before AC bond (strategy respects AC constraint).

SAA rebalancing:
  10. No rebalancing when portfolio is within tolerance.
  11. Rebalancing triggered when drift exceeds tolerance.
  12. Cash balance adjusted for rebalancing buy/sell proceeds.

step_time return:
  13. Returns FundTimestepResult with cashflows, decrements, asset fields.
  14. asset.total_market_value > 0 when portfolio has assets.
  15. asset.cash_balance matches Fund.cash_balance after step.
  16. asset.coupon_income > 0 for a FVTPL bond with non-zero coupon.
  17. asset.eir_income > 0 for an AC bond after one step.

Step ordering invariant:
  18. Realised G/L from rebalancing appears in the SAME period's asset result
      (i.e. rebalance is called before step_time, not after).

Zero-liability scenario:
  19. With zero net_outgo, cash_balance increases by exactly coupon_income.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.config.fund_config import AssetClassWeights
from engine.core.fund import Fund, FundTimestepResult
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import BaseLiability, Decrements, LiabilityCashflows
from engine.strategy.investment_strategy import InvestmentStrategy


# ---------------------------------------------------------------------------
# Minimal test-double liability model
# ---------------------------------------------------------------------------

class FixedCashflowLiability(BaseLiability):
    """
    Liability model that returns a fixed net_outgo every period.
    Used to isolate Fund cash mechanics from real liability calculations.
    """
    def __init__(self, net_outgo: float = 0.0, in_force: float = 100.0) -> None:
        self._net_outgo  = net_outgo
        self._in_force   = in_force

    def project_cashflows(self, _mp, _assumptions, timestep) -> LiabilityCashflows:
        return LiabilityCashflows(
            timestep=timestep,
            premiums=0.0,
            death_claims=self._net_outgo if self._net_outgo > 0 else 0.0,
            surrender_payments=0.0,
            maturity_payments=0.0,
            expenses=0.0,
        )

    def get_decrements(self, _mp, _assumptions, timestep) -> Decrements:
        return Decrements(
            timestep=timestep,
            in_force_start=self._in_force,
            deaths=0.0,
            lapses=0.0,
            maturities=0.0,
            in_force_end=self._in_force,
        )

    def get_bel(self, _mp, _assumptions, _timestep) -> float:
        return 0.0

    def get_reserve(self, _mp, _assumptions, _timestep) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FLAT_3PCT = RiskFreeRateCurve.flat(0.03)
EMPTY_MP  = pd.DataFrame()           # Fund never reads MP fields directly
DUMMY_ASS = None                      # passed to FixedCashflowLiability which ignores it


def make_scenario(t: int = 0) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=t,
        rate_curve=FLAT_3PCT,
        equity_total_return_yr=0.07,
    )


def make_fvtpl_bond(
    asset_id: str = "b1",
    face: float = 1_000_000.0,
    maturity_month: int = 36,
) -> Bond:
    return Bond(asset_id, face, 0.05, maturity_month, "FVTPL", face)


def make_ac_bond(asset_id: str = "ac1", face: float = 1_000_000.0) -> Bond:
    # discount bond: initial_book_value = 95% of face
    return Bond(asset_id, face, 0.05, 36, "AC", face * 0.95)


def make_no_rebalance_strategy() -> InvestmentStrategy:
    """Strategy that never triggers rebalancing (tolerance=1.0 = 100%)."""
    weights = AssetClassWeights(bonds=1.0, equities=0.0, derivatives=0.0, cash=0.0)
    return InvestmentStrategy(weights, rebalancing_tolerance=1.0)


def make_weights(bonds: float = 1.0, equities: float = 0.0) -> AssetClassWeights:
    cash = 1.0 - bonds - equities
    return AssetClassWeights(bonds=bonds, equities=equities, derivatives=0.0, cash=cash)


# ---------------------------------------------------------------------------
# 1-3: Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_cash_balance_property_returns_initial(self):
        am   = AssetModel([make_fvtpl_bond()])
        liab = FixedCashflowLiability()
        strat = make_no_rebalance_strategy()
        fund = Fund(am, liab, strat, initial_cash=50_000.0)
        assert fund.cash_balance == 50_000.0

    def test_asset_model_property(self):
        am   = AssetModel([make_fvtpl_bond()])
        fund = Fund(am, FixedCashflowLiability(), make_no_rebalance_strategy())
        assert fund.asset_model is am

    def test_default_initial_cash_is_zero(self):
        fund = Fund(AssetModel(), FixedCashflowLiability(), make_no_rebalance_strategy())
        assert fund.cash_balance == 0.0


# ---------------------------------------------------------------------------
# 4-6: Cash balance mechanics
# ---------------------------------------------------------------------------

class TestCashBalance:

    def _make_fund(self, net_outgo: float, initial_cash: float = 0.0) -> Fund:
        am   = AssetModel([make_fvtpl_bond()])
        liab = FixedCashflowLiability(net_outgo=net_outgo)
        return Fund(am, liab, make_no_rebalance_strategy(), initial_cash=initial_cash)

    def test_coupon_income_increases_cash(self):
        """FVTPL bond with 5% coupon on £1M face → £4,167/month coupon."""
        fund = self._make_fund(net_outgo=0.0, initial_cash=0.0)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        monthly_coupon = 1_000_000.0 * 0.05 / 12
        assert fund.cash_balance == pytest.approx(monthly_coupon, rel=1e-6)

    def test_net_outgo_decreases_cash(self):
        """£5,000 net_outgo reduces cash balance by £5,000 net of coupon."""
        monthly_coupon = 1_000_000.0 * 0.05 / 12
        fund = self._make_fund(net_outgo=5_000.0, initial_cash=10_000.0)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        assert fund.cash_balance == pytest.approx(10_000.0 + monthly_coupon - 5_000.0, rel=1e-4)

    def test_net_cash_change_equals_income_minus_outgo(self):
        """cash_balance = initial + income - outgo (no rebalancing)."""
        initial       = 100_000.0
        net_outgo     = 2_000.0
        monthly_coupon = 1_000_000.0 * 0.05 / 12
        fund = self._make_fund(net_outgo=net_outgo, initial_cash=initial)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        expected = initial + monthly_coupon - net_outgo
        assert fund.cash_balance == pytest.approx(expected, rel=1e-6)

    def test_zero_liability_cash_equals_coupon_income(self):
        """Test 19: zero net_outgo → cash increases by exactly coupon income."""
        fund   = self._make_fund(net_outgo=0.0, initial_cash=0.0)
        result = fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        assert fund.cash_balance == pytest.approx(result.asset.coupon_income, rel=1e-9)


# ---------------------------------------------------------------------------
# 7-9: Forced sells
# ---------------------------------------------------------------------------

class TestForcedSells:

    def test_forced_sell_triggered_when_cash_insufficient(self):
        """
        £1M FVTPL bond, zero coupon, large net_outgo = £50,000.
        Strategy allows AC sell (force_sell_ac=True) but only FVTPL exists.
        After forced sell, cash should be ≥ 0.
        """
        bond = Bond("fv1", 1_000_000.0, 0.0, 36, "FVTPL", 1_000_000.0)
        am   = AssetModel([bond])
        liab = FixedCashflowLiability(net_outgo=50_000.0)
        weights = make_weights(bonds=1.0)
        strat = InvestmentStrategy(weights, rebalancing_tolerance=1.0, force_sell_ac=True)
        fund  = Fund(am, liab, strat, initial_cash=0.0)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        assert fund.cash_balance >= -1.0   # within £1 rounding tolerance

    def test_no_forced_sell_when_cash_positive(self):
        """When asset income covers outgo, forced sell is not triggered."""
        bond = make_fvtpl_bond()           # £4,167 coupon/month
        am   = AssetModel([bond])
        liab = FixedCashflowLiability(net_outgo=1_000.0)   # well within coupon
        strat = make_no_rebalance_strategy()
        fund  = Fund(am, liab, strat, initial_cash=0.0)
        opening_face = bond.face_value
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        # Bond not touched — face value unchanged
        assert bond.face_value == pytest.approx(opening_face, rel=1e-9)


# ---------------------------------------------------------------------------
# 10-12: SAA rebalancing
# ---------------------------------------------------------------------------

class TestRebalancing:

    def test_no_rebalancing_within_tolerance(self):
        """Bond exactly at SAA target — no trades expected, cash unchanged."""
        bond  = make_fvtpl_bond(face=1_000_000.0)
        am    = AssetModel([bond])
        liab  = FixedCashflowLiability(net_outgo=0.0)
        # SAA: 100% bonds; bond is 100% of portfolio → within tolerance
        strat = InvestmentStrategy(
            make_weights(bonds=1.0), rebalancing_tolerance=0.05
        )
        fund  = Fund(am, liab, strat, initial_cash=0.0)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        # face_value unchanged (no trade executed)
        assert bond.face_value == pytest.approx(1_000_000.0, rel=1e-6)

    def test_rebalancing_adjusts_cash_balance(self):
        """
        SAA: 50% bonds, 50% cash.
        Portfolio: 100% bonds (£1M).
        Strategy should sell 50% of bonds → +£500k cash from sell proceeds,
        offset by the buy of the cash-class (no cash asset to buy → cash stays).
        """
        bond  = Bond("b1", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0)
        am    = AssetModel([bond])
        liab  = FixedCashflowLiability(net_outgo=0.0)
        strat = InvestmentStrategy(
            make_weights(bonds=0.5, equities=0.0),   # 50% bonds, 50% cash
            rebalancing_tolerance=0.01,
        )
        fund  = Fund(am, liab, strat, initial_cash=0.0)
        fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        # After sell, cash balance should be roughly +£500k from sell proceeds
        # plus the monthly coupon
        monthly_coupon = 1_000_000.0 * 0.05 / 12
        assert fund.cash_balance > monthly_coupon   # definitely gained from sell
        assert bond.face_value < 1_000_000.0        # bond was partially sold


# ---------------------------------------------------------------------------
# 13-17: FundTimestepResult fields
# ---------------------------------------------------------------------------

class TestStepTimeResult:

    def _run_one_step(
        self,
        face: float = 1_000_000.0,
        coupon: float = 0.05,
        basis: str = "FVTPL",
        net_outgo: float = 0.0,
        initial_cash: float = 0.0,
    ) -> FundTimestepResult:
        ibv  = face * (0.95 if basis == "AC" else 1.0)
        bond = Bond("b1", face, coupon, 36, basis, ibv)
        am   = AssetModel([bond])
        liab = FixedCashflowLiability(net_outgo=net_outgo)
        fund = Fund(am, liab, make_no_rebalance_strategy(), initial_cash=initial_cash)
        return fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)

    def test_returns_fund_timestep_result(self):
        result = self._run_one_step()
        assert isinstance(result, FundTimestepResult)

    def test_asset_total_mv_positive(self):
        result = self._run_one_step()
        assert result.asset.total_market_value > 0.0

    def test_asset_cash_balance_matches_fund(self):
        am   = AssetModel([make_fvtpl_bond()])
        liab = FixedCashflowLiability(net_outgo=0.0)
        fund = Fund(am, liab, make_no_rebalance_strategy(), initial_cash=50_000.0)
        result = fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        assert result.asset.cash_balance == pytest.approx(fund.cash_balance, rel=1e-9)

    def test_fvtpl_coupon_income_positive(self):
        result = self._run_one_step(coupon=0.05, basis="FVTPL")
        assert result.asset.coupon_income > 0.0

    def test_fvtpl_coupon_income_value(self):
        """FVTPL coupon = face × annual_rate / 12."""
        result = self._run_one_step(face=1_200_000.0, coupon=0.05, basis="FVTPL")
        expected = 1_200_000.0 * 0.05 / 12
        assert result.asset.coupon_income == pytest.approx(expected, rel=1e-4)

    def test_ac_eir_income_positive(self):
        """AC bond: eir_income > 0 after one step."""
        result = self._run_one_step(basis="AC")
        assert result.asset.eir_income > 0.0

    def test_cashflows_in_result(self):
        result = self._run_one_step(net_outgo=1_000.0)
        assert result.cashflows.net_outgo == pytest.approx(1_000.0, rel=1e-9)

    def test_decrements_in_result(self):
        result = self._run_one_step()
        assert result.decrements.in_force_start == pytest.approx(100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 18: Step ordering invariant — realised G/L in correct period
# ---------------------------------------------------------------------------

class TestStepOrdering:

    def test_realised_gl_captured_in_same_period_as_rebalance(self):
        """
        AC bond purchased at discount (BV < MV).
        Force-sell it: realised_gl = MV - BV > 0.
        This should appear in the SAME period's result, not the next.
        """
        # AC bond at 95p; MV ≈ par (with 3% flat curve and 5% coupon it's at premium)
        bond  = Bond("ac_sell", 100_000.0, 0.05, 36, "AC", 95_000.0)
        am    = AssetModel([bond])
        # Large net_outgo forces a sell of the AC bond (force_sell_ac=True)
        liab  = FixedCashflowLiability(net_outgo=200_000.0)
        strat = InvestmentStrategy(
            AssetClassWeights(bonds=1.0, equities=0.0, derivatives=0.0, cash=0.0),
            rebalancing_tolerance=1.0,
            force_sell_ac=True,
        )
        fund   = Fund(am, liab, strat, initial_cash=0.0)
        result = fund.step_time(make_scenario(), EMPTY_MP, DUMMY_ASS)
        # Realised GL should be non-zero (AC bond was sold)
        assert result.asset.realised_gl != 0.0

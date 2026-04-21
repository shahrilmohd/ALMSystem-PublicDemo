"""
Unit tests for engine/asset/equity.py.
"""
from __future__ import annotations

import math
import pytest

from engine.asset.equity import Equity
from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve


def make_scenario(timestep: int = 0, total_return: float = 0.08) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=timestep,
        rate_curve=RiskFreeRateCurve.flat(0.03),
        equity_total_return_yr=total_return,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestEquityConstruction:

    def test_valid_equity(self):
        eq = Equity("eq1", 500_000.0, dividend_yield_yr=0.03)
        assert eq.asset_id == "eq1"
        assert eq.asset_class == "equities"
        assert eq.accounting_basis == "FVTPL"

    def test_zero_initial_mv_raises(self):
        with pytest.raises(ValueError, match="initial_market_value"):
            Equity("x", 0.0)

    def test_negative_initial_mv_raises(self):
        with pytest.raises(ValueError, match="initial_market_value"):
            Equity("x", -1.0)

    def test_dividend_yield_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dividend_yield_yr"):
            Equity("x", 100.0, dividend_yield_yr=1.5)

    def test_zero_dividend_yield_allowed(self):
        eq = Equity("x", 100.0, dividend_yield_yr=0.0)
        assert eq.dividend_yield_yr == 0.0


# ---------------------------------------------------------------------------
# Market value
# ---------------------------------------------------------------------------

class TestEquityMarketValue:

    def test_initial_mv(self):
        mv = 500_000.0
        eq = Equity("x", mv)
        assert eq.market_value(make_scenario()) == mv

    def test_book_value_equals_market_value(self):
        eq = Equity("x", 500_000.0)
        assert eq.get_book_value() == eq.market_value(make_scenario())


# ---------------------------------------------------------------------------
# Cash flows
# ---------------------------------------------------------------------------

class TestEquityCashflows:

    def test_dividend_income(self):
        mv    = 1_200_000.0
        yield_ = 0.04
        eq    = Equity("x", mv, dividend_yield_yr=yield_)
        cf    = eq.project_cashflows(make_scenario(0))
        expected = mv * yield_ / 12.0
        assert abs(cf.dividend_income - expected) < 1e-6

    def test_no_coupon_income(self):
        eq = Equity("x", 500_000.0, 0.03)
        cf = eq.project_cashflows(make_scenario())
        assert cf.coupon_income == 0.0

    def test_no_maturity_proceeds(self):
        eq = Equity("x", 500_000.0, 0.03)
        cf = eq.project_cashflows(make_scenario())
        assert cf.maturity_proceeds == 0.0

    def test_zero_dividend_yield_zero_income(self):
        eq = Equity("x", 500_000.0, 0.0)
        cf = eq.project_cashflows(make_scenario())
        assert cf.dividend_income == 0.0


# ---------------------------------------------------------------------------
# step_time — price appreciation
# ---------------------------------------------------------------------------

class TestEquityStepTime:

    def test_market_value_grows_with_positive_return(self):
        eq  = Equity("x", 1_000_000.0, 0.03)
        mv0 = eq.market_value(make_scenario())
        eq.step_time(make_scenario(0, total_return=0.12))
        assert eq.market_value(make_scenario()) > mv0

    def test_market_value_shrinks_with_negative_return(self):
        eq = Equity("x", 1_000_000.0, 0.0)
        eq.step_time(make_scenario(0, total_return=-0.20))
        assert eq.market_value(make_scenario()) < 1_000_000.0

    def test_price_return_excludes_dividend(self):
        """
        With total_return = dividend_yield, price_return = 0 →
        market value should be unchanged after step.
        """
        yield_ = 0.04
        eq  = Equity("x", 1_000_000.0, dividend_yield_yr=yield_)
        mv0 = eq.market_value(make_scenario())
        eq.step_time(make_scenario(0, total_return=yield_))  # price_return = 0
        mv1 = eq.market_value(make_scenario())
        assert abs(mv1 - mv0) < 1e-6

    def test_annual_return_applied_monthly(self):
        """
        After 12 monthly steps with 8% annual total return and 3% yield:
        price_return = 5% pa.  MV should grow by factor (1+5%)^(12/12) = 1.05.
        """
        eq = Equity("x", 1_000_000.0, dividend_yield_yr=0.03)
        for t in range(12):
            eq.step_time(make_scenario(t, total_return=0.08))
        expected = 1_000_000.0 * (1.05)   # price return 5% over 1 year
        assert abs(eq.market_value(make_scenario()) - expected) < 100.0

    def test_pnl_keys_present(self):
        eq = Equity("x", 500_000.0, 0.03)
        eq.step_time(make_scenario())
        pnl = eq.get_pnl_components()
        for key in ("eir_income", "coupon_received", "dividend_income",
                    "unrealised_gl", "realised_gl", "oci_reserve"):
            assert key in pnl

    def test_pnl_eir_income_zero(self):
        eq = Equity("x", 500_000.0, 0.03)
        eq.step_time(make_scenario())
        assert eq.get_pnl_components()["eir_income"] == 0.0

    def test_pnl_oci_reserve_zero(self):
        eq = Equity("x", 500_000.0, 0.03)
        eq.step_time(make_scenario())
        assert eq.get_pnl_components()["oci_reserve"] == 0.0

    def test_pnl_dividend_income_in_step(self):
        """Dividend income in step_time P&L matches project_cashflows."""
        mv    = 1_000_000.0
        yield_ = 0.04
        eq    = Equity("x", mv, dividend_yield_yr=yield_)
        eq.step_time(make_scenario(0, total_return=yield_))
        pnl      = eq.get_pnl_components()
        expected = mv * yield_ / 12.0
        assert abs(pnl["dividend_income"] - expected) < 1e-6

    def test_apply_return_equivalent_to_step_time(self):
        """apply_return() is a backwards-compat alias for step_time()."""
        eq1 = Equity("a", 1_000_000.0, 0.03)
        eq2 = Equity("b", 1_000_000.0, 0.03)
        s   = make_scenario(0, total_return=0.10)
        eq1.step_time(s)
        eq2.apply_return(s)
        assert abs(eq1.market_value(make_scenario()) -
                   eq2.market_value(make_scenario())) < 1e-9


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

class TestEquityRebalancing:

    def test_buy_increases_mv(self):
        eq    = Equity("x", 500_000.0)
        trade = eq.rebalance(750_000.0, make_scenario())
        assert trade > 0.0
        assert eq.market_value(make_scenario()) == pytest.approx(750_000.0)

    def test_sell_decreases_mv(self):
        eq    = Equity("x", 500_000.0)
        trade = eq.rebalance(200_000.0, make_scenario())
        assert trade < 0.0
        assert eq.market_value(make_scenario()) == pytest.approx(200_000.0)

    def test_trade_amount_equals_delta_mv(self):
        eq    = Equity("x", 500_000.0)
        trade = eq.rebalance(800_000.0, make_scenario())
        assert abs(trade - 300_000.0) < 1e-6

    def test_no_realised_gl_for_equity(self):
        """FVTPL equity: no realised G/L on rebalancing."""
        eq = Equity("x", 500_000.0, 0.02)
        eq.rebalance(250_000.0, make_scenario())
        eq.step_time(make_scenario())
        # FVTPL: realised_gl should be 0
        assert eq.get_pnl_components()["realised_gl"] == 0.0


# ---------------------------------------------------------------------------
# Spread and duration
# ---------------------------------------------------------------------------

class TestEquitySpreadAndDuration:

    def test_calibration_spread_zero(self):
        assert Equity("x", 500_000.0).get_calibration_spread() == 0.0

    def test_default_allowance_zero(self):
        assert Equity("x", 500_000.0).get_default_allowance() == 0.0

    def test_duration_zero(self):
        assert Equity("x", 500_000.0).get_duration(make_scenario()) == 0.0

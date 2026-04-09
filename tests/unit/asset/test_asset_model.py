"""
Unit tests for engine/asset/asset_model.py.
"""
from __future__ import annotations

import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.asset.equity import Equity
from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve


def make_scenario(timestep: int = 0, rate: float = 0.03) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=timestep,
        rate_curve=RiskFreeRateCurve.flat(rate),
        equity_total_return_yr=0.08,
    )


def make_ac_bond(asset_id: str = "b_ac", face: float = 1_000_000.0) -> Bond:
    return Bond(asset_id, face, 0.05, 36, "AC", face * 0.95)


def make_fvtpl_bond(asset_id: str = "b_fv", face: float = 1_000_000.0,
                    maturity_month: int = 36) -> Bond:
    return Bond(asset_id, face, 0.05, maturity_month, "FVTPL", face)


def make_equity(asset_id: str = "eq1", mv: float = 500_000.0) -> Equity:
    return Equity(asset_id, mv, dividend_yield_yr=0.03)


# ---------------------------------------------------------------------------
# Portfolio management
# ---------------------------------------------------------------------------

class TestPortfolioManagement:

    def test_empty_model(self):
        am = AssetModel()
        assert len(am) == 0

    def test_add_and_retrieve_asset(self):
        am = AssetModel()
        b  = make_ac_bond()
        am.add_asset(b)
        assert am.has_asset("b_ac")
        assert am.get_asset("b_ac") is b

    def test_add_duplicate_raises(self):
        am = AssetModel()
        am.add_asset(make_ac_bond())
        with pytest.raises(ValueError, match="already exists"):
            am.add_asset(make_ac_bond())

    def test_remove_asset(self):
        am = AssetModel()
        am.add_asset(make_ac_bond())
        removed = am.remove_asset("b_ac")
        assert not am.has_asset("b_ac")
        assert removed.asset_id == "b_ac"

    def test_remove_nonexistent_raises(self):
        am = AssetModel()
        with pytest.raises(KeyError):
            am.remove_asset("nonexistent")

    def test_get_nonexistent_raises(self):
        am = AssetModel()
        with pytest.raises(KeyError):
            am.get_asset("nonexistent")

    def test_init_with_list(self):
        assets = [make_ac_bond("b1"), make_fvtpl_bond("b2"), make_equity("e1")]
        am = AssetModel(assets)
        assert len(am) == 3

    def test_iteration(self):
        assets = [make_ac_bond("b1"), make_equity("e1")]
        am = AssetModel(assets)
        ids = {a.asset_id for a in am}
        assert ids == {"b1", "e1"}

    def test_asset_ids(self):
        assets = [make_ac_bond("b1"), make_fvtpl_bond("b2")]
        am = AssetModel(assets)
        assert set(am.asset_ids()) == {"b1", "b2"}

    def test_assets_by_class(self):
        am = AssetModel([make_ac_bond("b1"), make_fvtpl_bond("b2"), make_equity("e1")])
        bonds    = am.assets_by_class("bonds")
        equities = am.assets_by_class("equities")
        assert len(bonds) == 2
        assert len(equities) == 1

    def test_assets_by_basis(self):
        am = AssetModel([make_ac_bond("b_ac"), make_fvtpl_bond("b_fv")])
        ac_assets = am.assets_by_basis("AC")
        fv_assets = am.assets_by_basis("FVTPL")
        assert len(ac_assets) == 1
        assert len(fv_assets) == 1


# ---------------------------------------------------------------------------
# Market value aggregation
# ---------------------------------------------------------------------------

class TestMarketValueAggregation:

    def test_total_market_value_sum(self):
        b1 = make_ac_bond("b1", 1_000_000.0)
        b2 = make_fvtpl_bond("b2", 2_000_000.0)
        am = AssetModel([b1, b2])
        s  = make_scenario()
        expected = b1.market_value(s) + b2.market_value(s)
        assert abs(am.total_market_value(s) - expected) < 1.0

    def test_total_book_value_sum(self):
        b1 = make_ac_bond("b1")     # BV = 950_000
        b2 = make_fvtpl_bond("b2")  # BV = 1_000_000
        am = AssetModel([b1, b2])
        assert abs(am.total_book_value() - (b1.get_book_value() + b2.get_book_value())) < 1e-6

    def test_market_value_by_class(self):
        b1 = make_ac_bond("b1")
        eq = make_equity("e1", 300_000.0)
        am = AssetModel([b1, eq])
        s  = make_scenario()
        mv = am.market_value_by_class(s)
        assert "bonds" in mv
        assert "equities" in mv
        assert abs(mv["equities"] - 300_000.0) < 1e-6

    def test_market_value_by_basis_all_keys_present(self):
        am = AssetModel([make_ac_bond("b1")])
        s  = make_scenario()
        mv = am.market_value_by_basis(s)
        assert set(mv.keys()) == {"AC", "FVTPL", "FVOCI"}

    def test_market_value_by_basis_correct_values(self):
        b_ac = make_ac_bond("b_ac", 1_000_000.0)
        b_fv = make_fvtpl_bond("b_fv", 2_000_000.0)
        am   = AssetModel([b_ac, b_fv])
        s    = make_scenario()
        mv   = am.market_value_by_basis(s)
        assert abs(mv["AC"]    - b_ac.market_value(s)) < 1.0
        assert abs(mv["FVTPL"] - b_fv.market_value(s)) < 1.0
        assert mv["FVOCI"] == 0.0

    def test_empty_portfolio_zero_mv(self):
        am = AssetModel()
        assert am.total_market_value(make_scenario()) == 0.0


# ---------------------------------------------------------------------------
# Cash flow collection
# ---------------------------------------------------------------------------

class TestCashflowCollection:

    def test_aggregate_coupons(self):
        b1 = make_ac_bond("b1", 1_000_000.0)
        b2 = make_fvtpl_bond("b2", 2_000_000.0)
        am = AssetModel([b1, b2])
        s  = make_scenario(0)
        cf = am.collect_cashflows(s)
        expected = (b1.annual_coupon_rate * 1_000_000.0 / 12.0 +
                    b2.annual_coupon_rate * 2_000_000.0 / 12.0)
        assert abs(cf.coupon_income - expected) < 1e-6

    def test_dividend_income_from_equity(self):
        eq = make_equity("e1", 600_000.0)
        am = AssetModel([eq])
        cf = am.collect_cashflows(make_scenario())
        expected = 600_000.0 * 0.03 / 12.0
        assert abs(cf.dividend_income - expected) < 1e-6

    def test_matured_bond_removed_from_portfolio(self):
        """A bond that matures this period should be removed after cashflow collection."""
        b = Bond("short", 100_000.0, 0.05, 1, "FVTPL", 100_000.0)
        am = AssetModel([b])
        am.collect_cashflows(make_scenario(0))   # remaining = 1 → matures
        assert not am.has_asset("short")

    def test_non_matured_bond_stays_in_portfolio(self):
        b  = make_fvtpl_bond("b1", maturity_month=36)
        am = AssetModel([b])
        am.collect_cashflows(make_scenario(0))
        assert am.has_asset("b1")


# ---------------------------------------------------------------------------
# P&L aggregation
# ---------------------------------------------------------------------------

class TestPnLAggregation:

    def test_aggregate_pnl_sums_eir_income(self):
        b1 = make_ac_bond("b1", 1_000_000.0)
        b2 = make_ac_bond("b2", 500_000.0)
        am = AssetModel([b1, b2])
        s  = make_scenario(0)
        am.step_time(s)
        pnl = am.aggregate_pnl()
        assert pnl["eir_income"] == pytest.approx(
            b1.get_pnl_components()["eir_income"] +
            b2.get_pnl_components()["eir_income"],
            abs=1e-6,
        )

    def test_pnl_by_basis_separates_ac_and_fvtpl(self):
        b_ac = make_ac_bond("b_ac")
        b_fv = make_fvtpl_bond("b_fv")
        am   = AssetModel([b_ac, b_fv])
        s    = make_scenario(0)
        am.step_time(s)
        by_basis = am.pnl_by_basis()
        # AC: should have non-zero eir_income
        assert by_basis["AC"]["eir_income"] > 0.0
        # FVTPL: eir_income should be 0
        assert by_basis["FVTPL"]["eir_income"] == 0.0
        # FVTPL: coupon income should be present
        assert by_basis["FVTPL"]["coupon_received"] > 0.0


# ---------------------------------------------------------------------------
# step_time
# ---------------------------------------------------------------------------

class TestStepTime:

    def test_step_time_advances_all_assets(self):
        b_ac = make_ac_bond("b_ac")
        eq   = make_equity("eq1")
        am   = AssetModel([b_ac, eq])
        s    = make_scenario(0, rate=0.03)

        bv_before = b_ac.get_book_value()
        mv_before = eq.market_value(s)

        am.step_time(s)

        # AC bond: BV should have amortised slightly upward (discount bond)
        assert b_ac.get_book_value() > bv_before

        # Equity: MV should have changed based on equity return in scenario
        # (scenario uses 8% total return, 0% dividend → price return ≈ 8%)
        # After 1 month the change should be small but present
        assert eq.market_value(make_scenario(0)) != mv_before


# ---------------------------------------------------------------------------
# Default allowance
# ---------------------------------------------------------------------------

class TestDefaultAllowance:

    def test_total_default_allowance_sums_bonds(self):
        b1 = Bond("b1", 1_000_000.0, 0.05, 36, "AC", 950_000.0,
                  calibration_spread=0.01)
        b2 = Bond("b2", 500_000.0, 0.05, 36, "FVTPL", 500_000.0,
                  calibration_spread=0.005)
        am = AssetModel([b1, b2])
        total = am.total_default_allowance(lgd_rate=0.40)
        expected = b1.get_default_allowance(0.40) + b2.get_default_allowance(0.40)
        assert abs(total - expected) < 1e-9

    def test_zero_allowance_for_zero_spread_portfolio(self):
        am = AssetModel([make_ac_bond("b1"), make_fvtpl_bond("b2")])
        assert am.total_default_allowance() == 0.0

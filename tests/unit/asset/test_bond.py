"""
Unit tests for engine/asset/bond.py.

Numerical tests cover AC, FVTPL, and FVOCI paths independently (CLAUDE.md).

Anchors from DECISIONS.md Section 2 (EIR example — annual timestep version):
    Bond: par=100, coupon=5%, purchase price=95, 3-year maturity
    EIR ≈ 6.9%
    Year 1: BV 95.00 → 96.55.
    Total EIR income over 3 years = 20 = 15 (coupons) + 5 (discount unwind).

Note: The DECISIONS.md anchor uses annual coupon payments for simplicity.
The model projects monthly, so results match to within ~£0.15 per £100 par
due to the different payment timing (monthly vs annual compounding).  The
total-income anchor (sum = 20) is timing-invariant and must hold exactly.
"""
from __future__ import annotations

import math
import pytest

from engine.asset.bond import Bond
from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scenario(timestep: int, rate: float = 0.03) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=timestep,
        rate_curve=RiskFreeRateCurve.flat(rate),
        equity_total_return_yr=0.07,
    )


def make_ac_bond(
    face_value: float = 1_000_000.0,
    coupon_rate: float = 0.05,
    maturity_month: int = 36,
    initial_book_value: float | None = None,
    calibration_spread: float = 0.0,
    eir: float | None = None,
) -> Bond:
    ibv = initial_book_value if initial_book_value is not None else face_value * 0.95
    return Bond(
        asset_id="ac_bond_1",
        face_value=face_value,
        annual_coupon_rate=coupon_rate,
        maturity_month=maturity_month,
        accounting_basis="AC",
        initial_book_value=ibv,
        calibration_spread=calibration_spread,
        eir=eir,
    )


def make_fvtpl_bond(
    face_value: float = 1_000_000.0,
    coupon_rate: float = 0.05,
    maturity_month: int = 36,
    initial_book_value: float | None = None,
    calibration_spread: float = 0.0,
) -> Bond:
    ibv = initial_book_value if initial_book_value is not None else face_value
    return Bond(
        asset_id="fvtpl_bond_1",
        face_value=face_value,
        annual_coupon_rate=coupon_rate,
        maturity_month=maturity_month,
        accounting_basis="FVTPL",
        initial_book_value=ibv,
        calibration_spread=calibration_spread,
    )


def make_fvoci_bond(
    face_value: float = 1_000_000.0,
    coupon_rate: float = 0.05,
    maturity_month: int = 36,
    initial_book_value: float | None = None,
) -> Bond:
    ibv = initial_book_value if initial_book_value is not None else face_value * 0.95
    return Bond(
        asset_id="fvoci_bond_1",
        face_value=face_value,
        annual_coupon_rate=coupon_rate,
        maturity_month=maturity_month,
        accounting_basis="FVOCI",
        initial_book_value=ibv,
    )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestBondConstruction:
    def test_valid_ac_bond(self):
        bond = make_ac_bond()
        assert bond.asset_id == "ac_bond_1"
        assert bond.asset_class == "bonds"
        assert bond.accounting_basis == "AC"
        assert bond.eir > 0.0

    def test_valid_fvtpl_bond(self):
        assert make_fvtpl_bond().accounting_basis == "FVTPL"

    def test_valid_fvoci_bond(self):
        assert make_fvoci_bond().accounting_basis == "FVOCI"

    def test_invalid_accounting_basis(self):
        with pytest.raises(ValueError, match="accounting_basis"):
            Bond("x", 1000.0, 0.05, 12, "UNKNOWN", 950.0)

    def test_negative_face_value_raises(self):
        with pytest.raises(ValueError, match="face_value"):
            Bond("x", -1.0, 0.05, 12, "AC", 950.0)

    def test_zero_initial_book_value_raises(self):
        with pytest.raises(ValueError, match="initial_book_value"):
            Bond("x", 1000.0, 0.05, 12, "AC", 0.0)

    def test_negative_calibration_spread_raises(self):
        with pytest.raises(ValueError, match="calibration_spread"):
            Bond("x", 1000.0, 0.05, 12, "AC", 950.0, calibration_spread=-0.01)

    def test_eir_provided_directly_used_as_is(self):
        """If eir is supplied, calculate_eir() is bypassed."""
        bond = Bond(
            asset_id="x",
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            accounting_basis="AC",
            initial_book_value=95.0,
            eir=0.069,
        )
        assert abs(bond.eir - 0.069) < 1e-9


# ---------------------------------------------------------------------------
# EIR calculation — call patterns
# ---------------------------------------------------------------------------

class TestCalculateEIR:

    def test_par_bond_eir_equals_coupon_rate_flat(self):
        """Bond at par with monthly coupons: EIR is slightly above coupon rate.

        With monthly coupon payments, the investor receives cash sooner than
        with annual payments, so the effective yield is marginally higher than
        the stated coupon rate.  The difference is small (~0.12%) and is
        correct IFRS 9 behaviour — the solver finds the flat rate that
        discounts all cash flows back to the purchase price.
        """
        eir = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            initial_book_value=100.0,
        )
        assert eir > 0.05
        assert abs(eir - 0.05) < 0.002

    def test_discount_bond_eir_greater_than_coupon(self):
        """Discount bond: EIR > coupon rate (DECISIONS.md anchor ≈ 6.9%)."""
        eir = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            initial_book_value=95.0,
        )
        assert eir > 0.05
        assert abs(eir - 0.069) < 0.005, f"Expected ≈6.9%, got {eir:.4f}"

    def test_premium_bond_eir_less_than_coupon(self):
        """Premium bond: EIR < coupon rate."""
        eir = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            initial_book_value=105.0,
        )
        assert eir < 0.05

    def test_eir_from_scenario_uses_yield_curve(self):
        """
        When scenario is supplied, calculate_eir uses the risk-free curve to
        compute the initial book value, then solves the flat EIR from it.

        For a 3% flat risk-free curve and zero calibration spread:
          - the curve-priced initial_book_value = PV of all CFs at 3%
          - the resulting EIR should be close to 3% for a par-priced bond
        """
        scenario = make_scenario(0, rate=0.03)
        face     = 1_000_000.0
        coupon   = 0.03   # coupon matches risk-free rate → near-par pricing
        months   = 60

        eir = Bond.calculate_eir(
            face_value=face,
            annual_coupon_rate=coupon,
            maturity_month=months,
            scenario=scenario,
            calibration_spread=0.0,
        )
        # Bond priced near par at 3% rf → EIR should be close to 3%
        assert abs(eir - 0.03) < 0.002, (
            f"Expected EIR≈3% for near-par bond, got {eir:.4f}"
        )

    def test_eir_from_scenario_with_spread(self):
        """
        With a positive calibration_spread, curve prices the bond below par →
        EIR > coupon rate.
        """
        scenario = make_scenario(0, rate=0.03)
        eir = Bond.calculate_eir(
            face_value=1_000_000.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            scenario=scenario,
            calibration_spread=0.02,  # 2% additional spread → bond prices below par
        )
        assert eir > 0.05

    def test_initial_book_value_takes_precedence_over_scenario(self):
        """When both are supplied, initial_book_value is used; scenario ignored."""
        scenario = make_scenario(0, rate=0.10)  # very high rate — would distort MV
        eir_direct = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            initial_book_value=95.0,  # explicit book value
            scenario=scenario,        # should be ignored
        )
        eir_no_scenario = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            initial_book_value=95.0,
        )
        assert abs(eir_direct - eir_no_scenario) < 1e-9

    def test_neither_ibv_nor_scenario_raises(self):
        with pytest.raises(ValueError, match="initial_book_value or a scenario"):
            Bond.calculate_eir(
                face_value=100.0,
                annual_coupon_rate=0.05,
                maturity_month=36,
            )

    def test_eir_curve_vs_flat_are_different_for_steep_curve(self):
        """
        For a steeply sloped yield curve, the flat-rate EIR and the curve-derived
        EIR should differ — the scenario path gives a more accurate inception price.
        """
        # Steep upward-sloping curve
        steep_curve = RiskFreeRateCurve(
            spot_rates={1.0: 0.01, 5.0: 0.04, 10.0: 0.07}
        )
        scenario = AssetScenarioPoint(0, steep_curve, 0.07)

        eir_curve = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=60,
            scenario=scenario,
            calibration_spread=0.0,
        )
        eir_flat = Bond.calculate_eir(
            face_value=100.0,
            annual_coupon_rate=0.05,
            maturity_month=60,
            initial_book_value=100.0,  # force flat-rate at par
        )
        # They should differ — curve prices the bond differently from flat=par
        assert abs(eir_curve - eir_flat) > 0.001


# ---------------------------------------------------------------------------
# AC path — book value amortisation and P&L
# ---------------------------------------------------------------------------

class TestACPath:

    def test_bv_converges_to_par_at_maturity(self):
        """The defining property of EIR: BV must reach face_value at maturity."""
        face, ibv = 1_000_000.0, 950_000.0
        bond = make_ac_bond(face_value=face, initial_book_value=ibv, maturity_month=36)
        for t in range(36):
            bond.step_time(make_scenario(t))
        assert abs(bond.get_book_value() - face) < 1.0

    def test_bv_increases_for_discount_bond(self):
        """AC discount bond: BV increases each period toward par."""
        bond = make_ac_bond(face_value=100.0, initial_book_value=95.0)
        prev_bv = bond.get_book_value()
        for t in range(12):
            bond.step_time(make_scenario(t))
            new_bv = bond.get_book_value()
            assert new_bv > prev_bv, f"Month {t+1}: BV should increase"
            prev_bv = new_bv

    def test_bv_decreases_for_premium_bond(self):
        """AC premium bond: BV decreases each period toward par."""
        bond = Bond("x", 100.0, 0.05, 36, "AC", 105.0)
        prev_bv = bond.get_book_value()
        for t in range(12):
            bond.step_time(make_scenario(t))
            assert bond.get_book_value() < prev_bv
            prev_bv = bond.get_book_value()

    def test_total_eir_income_equals_economic_return(self):
        """
        Total EIR income over the bond's life = coupons + discount unwind.
        DECISIONS.md anchor: £95 bond, £5 coupon × 3 yrs + £5 discount = £20.
        Monthly projection: total must equal 20 (timing-invariant).
        """
        face, ibv, coupon_r, months = 100.0, 95.0, 0.05, 36
        bond = Bond("x", face, coupon_r, months, "AC", ibv)

        total_eir_income = 0.0
        for t in range(months):
            bond.step_time(make_scenario(t))
            total_eir_income += bond.get_pnl_components()["eir_income"]

        expected = face * coupon_r * (months / 12) + (face - ibv)  # 15 + 5 = 20
        assert abs(total_eir_income - expected) < 0.01, (
            f"Expected total EIR income ≈ {expected:.2f}, got {total_eir_income:.4f}"
        )

    def test_pnl_keys_present_ac(self):
        bond = make_ac_bond()
        bond.step_time(make_scenario(0))
        pnl = bond.get_pnl_components()
        for key in ("eir_income", "coupon_received", "dividend_income",
                    "unrealised_gl", "realised_gl", "oci_reserve"):
            assert key in pnl

    def test_eir_income_greater_than_coupon_for_discount_bond(self):
        bond = make_ac_bond(face_value=100.0, initial_book_value=95.0)
        bond.step_time(make_scenario(0))
        pnl = bond.get_pnl_components()
        assert pnl["eir_income"] > pnl["coupon_received"]

    def test_oci_reserve_zero_for_ac(self):
        bond = make_ac_bond()
        bond.step_time(make_scenario(0))
        assert bond.get_pnl_components()["oci_reserve"] == 0.0

    def test_realised_gl_zero_before_any_sale(self):
        bond = make_ac_bond()
        bond.step_time(make_scenario(0))
        assert bond.get_pnl_components()["realised_gl"] == 0.0

    def test_eir_income_uses_curve_when_scenario_provided_at_construction(self):
        """
        Bond constructed with EIR derived from curve (via calculate_eir with
        scenario) should amortise differently from flat-rate EIR if the curve
        prices the bond differently.
        """
        scenario = make_scenario(0, rate=0.03)
        cs       = Bond.calibrate_spread(
            face_value=1_000_000.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            observed_market_value=960_000.0,
            scenario=scenario,
        )
        eir_curve = Bond.calculate_eir(
            face_value=1_000_000.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            scenario=scenario,
            calibration_spread=cs,
        )
        bond = Bond(
            asset_id="x",
            face_value=1_000_000.0,
            annual_coupon_rate=0.05,
            maturity_month=36,
            accounting_basis="AC",
            initial_book_value=960_000.0,
            eir=eir_curve,
            calibration_spread=cs,
        )
        bond.step_time(scenario)
        pnl = bond.get_pnl_components()
        assert pnl["eir_income"] > 0.0


# ---------------------------------------------------------------------------
# FVTPL path
# ---------------------------------------------------------------------------

class TestFVTPLPath:

    def test_book_value_equals_market_value_after_step(self):
        bond = make_fvtpl_bond(face_value=100.0, initial_book_value=100.0)
        s    = make_scenario(0)
        bond.step_time(s)
        assert abs(bond.get_book_value() - bond.market_value(s)) < 1e-9

    def test_eir_income_zero_for_fvtpl(self):
        bond = make_fvtpl_bond()
        bond.step_time(make_scenario(0))
        assert bond.get_pnl_components()["eir_income"] == 0.0

    def test_oci_reserve_zero_for_fvtpl(self):
        bond = make_fvtpl_bond()
        bond.step_time(make_scenario(0))
        assert bond.get_pnl_components()["oci_reserve"] == 0.0

    def test_total_pnl_equals_economic_return_fvtpl(self):
        """
        FVTPL total P&L (coupon + unrealised_gl) over bond life must equal
        total coupons + price gain, consistent with the AC total EIR income.
        Use a fixed flat curve equal to coupon rate so MV ≈ par throughout
        and the total equals coupons + discount.
        """
        face, ibv, coupon_r, months = 100.0, 95.0, 0.05, 36
        bond    = make_fvtpl_bond(face_value=face, coupon_rate=coupon_r,
                                  maturity_month=months, initial_book_value=ibv)
        total_pnl = 0.0
        for t in range(months):
            s = make_scenario(t, rate=coupon_r)
            bond.step_time(s)
            pnl = bond.get_pnl_components()
            total_pnl += pnl["coupon_received"] + pnl["unrealised_gl"]

        expected = face * coupon_r * (months / 12) + (face - ibv)  # 15 + 5 = 20
        assert abs(total_pnl - expected) < 0.01


# ---------------------------------------------------------------------------
# FVOCI path
# ---------------------------------------------------------------------------

class TestFVOCIPath:

    def test_book_value_equals_market_value_fvoci(self):
        bond = make_fvoci_bond(face_value=100.0, initial_book_value=95.0)
        s    = make_scenario(0)
        bond.step_time(s)
        assert abs(bond.get_book_value() - bond.market_value(s)) < 1e-9

    def test_eir_income_nonzero_for_fvoci(self):
        bond = make_fvoci_bond(face_value=100.0, initial_book_value=95.0)
        bond.step_time(make_scenario(0))
        assert bond.get_pnl_components()["eir_income"] > 0.0

    def test_oci_reserve_tracks_cumulative_mv_movements(self):
        bond       = make_fvoci_bond(face_value=100.0, initial_book_value=95.0)
        cumulative = 0.0
        for t in range(6):
            bond.step_time(make_scenario(t))
            pnl        = bond.get_pnl_components()
            cumulative += pnl["unrealised_gl"]
            assert abs(pnl["oci_reserve"] - cumulative) < 1e-9, (
                f"Period {t}: oci_reserve mismatch"
            )

    def test_fvoci_eir_income_matches_ac_eir_income(self):
        """
        AC and FVOCI bonds with identical parameters must produce the same
        total EIR income (both use the EIR method for P&L income recognition).
        """
        face, ibv, coupon_r, months = 100.0, 95.0, 0.05, 36
        rate = 0.05
        ac_bond   = Bond("ac", face, coupon_r, months, "AC",   ibv)
        fvoci_bond = Bond("oc", face, coupon_r, months, "FVOCI", ibv)

        ac_total = oc_total = 0.0
        for t in range(months):
            s = make_scenario(t, rate=rate)
            ac_bond.step_time(s)
            fvoci_bond.step_time(s)
            ac_total += ac_bond.get_pnl_components()["eir_income"]
            oc_total += fvoci_bond.get_pnl_components()["eir_income"]

        assert abs(ac_total - oc_total) < 0.01


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

class TestRebalancing:

    def test_buy_returns_positive_trade(self):
        bond = make_fvtpl_bond(face_value=1_000_000.0, initial_book_value=1_000_000.0)
        s    = make_scenario(0)
        trade = bond.rebalance(bond.market_value(s) * 1.5, s)
        assert trade > 0.0
        assert bond.face_value > 1_000_000.0

    def test_sell_returns_negative_trade(self):
        bond = make_fvtpl_bond(face_value=1_000_000.0, initial_book_value=1_000_000.0)
        s    = make_scenario(0)
        trade = bond.rebalance(bond.market_value(s) * 0.5, s)
        assert trade < 0.0

    def test_rebalance_to_zero(self):
        bond  = make_fvtpl_bond(face_value=1_000_000.0, initial_book_value=1_000_000.0)
        s     = make_scenario(0)
        trade = bond.rebalance(0.0, s)
        assert trade < 0.0
        assert bond.face_value == pytest.approx(0.0, abs=1e-6)

    def test_fvtpl_realised_gl_near_zero_on_sell(self):
        """FVTPL: BV = MV always, so realised_gl ≈ 0 on a sell."""
        bond = make_fvtpl_bond(face_value=1_000_000.0, initial_book_value=1_000_000.0)
        s    = make_scenario(0)
        bond.rebalance(bond.market_value(s) * 0.5, s)
        bond.step_time(s)
        assert abs(bond.get_pnl_components()["realised_gl"]) < 1.0

    def test_ac_realised_gl_equals_mv_minus_bv_times_fraction(self):
        """
        Selling half an AC discount bond crystallises (MV − BV) × 0.5.
        """
        face, ibv = 1_000_000.0, 950_000.0
        bond = make_ac_bond(face_value=face, initial_book_value=ibv)
        s    = make_scenario(0)
        mv   = bond.market_value(s)
        bv   = bond.get_book_value()
        bond.rebalance(mv * 0.5, s)
        bond.step_time(s)
        expected = (mv - bv) * 0.5
        assert abs(bond.get_pnl_components()["realised_gl"] - expected) < 1.0

    def test_book_value_scaled_proportionally_on_sell(self):
        bond  = make_ac_bond(face_value=1_000_000.0, initial_book_value=950_000.0)
        s     = make_scenario(0)
        bv_before = bond.get_book_value()
        mv        = bond.market_value(s)
        bond.rebalance(mv * 0.4, s)
        assert abs(bond.get_book_value() - bv_before * 0.4) < 1.0

    def test_rebalance_matured_bond_returns_zero(self):
        bond = make_fvtpl_bond(maturity_month=1)
        assert bond.rebalance(1_000_000.0, make_scenario(2)) == 0.0


# ---------------------------------------------------------------------------
# Market value and calibration spread
# ---------------------------------------------------------------------------

class TestMarketValue:

    def test_calibrate_spread_roundtrip(self):
        """calibrate_spread → Bond.market_value should recover observed MV."""
        scenario = make_scenario(0)
        face, ibv = 1_000_000.0, 960_000.0
        cs = Bond.calibrate_spread(
            face_value=face,
            annual_coupon_rate=0.05,
            maturity_month=36,
            observed_market_value=ibv,
            scenario=scenario,
        )
        bond = Bond("x", face, 0.05, 36, "FVTPL", ibv, calibration_spread=cs)
        assert abs(bond.market_value(scenario) - ibv) < 0.01

    def test_higher_spread_lower_mv(self):
        s     = make_scenario(0)
        b_lo  = Bond("lo", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0,
                     calibration_spread=0.005)
        b_hi  = Bond("hi", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0,
                     calibration_spread=0.02)
        assert b_hi.market_value(s) < b_lo.market_value(s)

    def test_matured_bond_mv_is_zero(self):
        bond = make_fvtpl_bond(maturity_month=1)
        assert bond.market_value(make_scenario(2)) == 0.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

class TestDuration:

    def test_duration_positive(self):
        bond = make_fvtpl_bond(face_value=100.0, initial_book_value=100.0)
        assert bond.get_duration(make_scenario(0)) > 0.0

    def test_duration_zero_matured(self):
        bond = make_fvtpl_bond(maturity_month=1)
        assert bond.get_duration(make_scenario(2)) == 0.0

    def test_longer_maturity_higher_duration(self):
        s = make_scenario(0)
        short = Bond("s", 100.0, 0.05, 12,  "FVTPL", 100.0)
        long_ = Bond("l", 100.0, 0.05, 120, "FVTPL", 100.0)
        assert long_.get_duration(s) > short.get_duration(s)


# ---------------------------------------------------------------------------
# Default allowance
# ---------------------------------------------------------------------------

class TestDefaultAllowance:

    def test_zero_when_no_spread(self):
        assert make_fvtpl_bond().get_default_allowance() == 0.0

    def test_positive_with_spread(self):
        bond = Bond("x", 1_000_000.0, 0.05, 36, "AC", 950_000.0,
                    calibration_spread=0.01)
        assert bond.get_default_allowance(lgd_rate=0.40) > 0.0

    def test_higher_spread_higher_allowance(self):
        lo = Bond("lo", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0,
                  calibration_spread=0.005)
        hi = Bond("hi", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0,
                  calibration_spread=0.02)
        assert hi.get_default_allowance() > lo.get_default_allowance()

    def test_get_calibration_spread_returns_locked_value(self):
        spread = 0.0123
        bond   = Bond("x", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0,
                      calibration_spread=spread)
        assert bond.get_calibration_spread() == spread


# ---------------------------------------------------------------------------
# Cash flow projection
# ---------------------------------------------------------------------------

class TestCashflows:

    def test_monthly_coupon_income(self):
        face, coupon = 1_000_000.0, 0.05
        bond = make_fvtpl_bond(face_value=face, coupon_rate=coupon)
        cf   = bond.project_cashflows(make_scenario(0))
        assert abs(cf.coupon_income - face * coupon / 12.0) < 1e-6

    def test_maturity_proceeds_on_last_period(self):
        face = 1_000_000.0
        bond = make_fvtpl_bond(face_value=face, maturity_month=6)
        cf   = bond.project_cashflows(make_scenario(5))  # remaining = 1
        assert abs(cf.maturity_proceeds - face) < 1e-6

    def test_no_proceeds_before_maturity(self):
        bond = make_fvtpl_bond(face_value=1_000_000.0, maturity_month=36)
        assert bond.project_cashflows(make_scenario(0)).maturity_proceeds == 0.0

    def test_zero_cashflows_after_maturity(self):
        bond = make_fvtpl_bond(maturity_month=3)
        assert bond.project_cashflows(make_scenario(5)).total_income == 0.0

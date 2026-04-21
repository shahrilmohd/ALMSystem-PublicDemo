"""
Unit tests for engine/liability/bpa/dependant.py

Numerical anchor strategy
--------------------------
All BEL anchors use:
  - flat q_x = 0.02 (zero improvement) for both member and dependant
  - RiskFreeRateCurve.flat(0.03)
  - inflation_rate = 0.0
  - expense_pa = 0.0
  - weight = 1.0, pension_pa = 1200.0

BEL convolution anchor (2-period annual calendar):
---------------------------------------------------
Only trigger s=0 contributes (no periods after s=1):

  f_death(0) = sp_member[0] × q_member_period(0)
             = 1.0 × 0.02 = 0.02

  V_dep(0)   = Σ_{u=1}^{1}  pension × cond_sp × df(u) × dt(u)
  where:
    cond_sp = sp_dep[2] / sp_dep[1] = 0.98^2 / 0.98 = 0.98
    df(u=1) = DF at end of period 1 = 1 / 1.03^2
    dt(u=1) = 1.0

  V_dep(0) = 1200 × 0.98 × (1/1.03^2) × 1.0

  BEL = f_death(0) × V_dep(0) × weight
      = 0.02 × 1200 × 0.98 / 1.03^2

Each component is hand-verifiable.
"""
import numpy as np
import pandas as pd
import pytest

from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import LiabilityCashflows, Decrements
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.dependant import DependantLiability, REQUIRED_COLUMNS
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH, survival_probs_variable_dt
from engine.liability.bpa.mortality import q_x as qx_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def annual_2():
    return ProjectionCalendar(projection_years=2, monthly_years=0)


@pytest.fixture
def annual_10():
    return ProjectionCalendar(projection_years=10, monthly_years=0)


@pytest.fixture
def flat_basis():
    arr  = np.full(TABLE_LENGTH, 0.02)
    zero = np.zeros(TABLE_LENGTH)
    return MortalityBasis(arr.copy(), arr.copy(), zero.copy(), zero.copy(), ltr=0.0)


def make_assumptions(basis, discount_rate=0.03, inflation=0.0):
    ill = np.zeros(TABLE_LENGTH)
    return BPAAssumptions(
        mortality=basis,
        valuation_year=2023,
        discount_curve=RiskFreeRateCurve.flat(discount_rate),
        inflation_rate=inflation,
        rpi_rate=inflation,
        tv_rate=0.0,
        ill_health_rates=ill,
        expense_pa=0.0,
    )


def make_mp(
    member_age=65.0, member_sex="M",
    dependant_age=62.0, dependant_sex="F",
    weight=1.0, pension_pa=1200.0,
    lpi_cap=0.0, lpi_floor=0.0,
):
    return pd.DataFrame([{
        "mp_id":         "D001",
        "member_sex":    member_sex,
        "member_age":    member_age,
        "dependant_sex": dependant_sex,
        "dependant_age": dependant_age,
        "weight":        weight,
        "pension_pa":    pension_pa,
        "lpi_cap":       lpi_cap,
        "lpi_floor":     lpi_floor,
    }])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_column_raises(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        bad = pd.DataFrame([{"mp_id": "X", "member_sex": "M"}])
        with pytest.raises(ValueError, match="missing columns"):
            liability.get_bel(bad, make_assumptions(flat_basis), 0)

    def test_required_columns_present(self):
        assert "member_sex"    in REQUIRED_COLUMNS
        assert "dependant_age" in REQUIRED_COLUMNS
        assert "pension_pa"    in REQUIRED_COLUMNS
        assert "weight"        in REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# get_bel — convolution
# ---------------------------------------------------------------------------

class TestGetBel:

    def test_bel_positive_for_in_force(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        bel = liability.get_bel(make_mp(), make_assumptions(flat_basis), 0)
        assert bel > 0.0

    def test_bel_zero_weight_returns_zero(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        bel = liability.get_bel(make_mp(weight=0.0), make_assumptions(flat_basis), 0)
        assert bel == 0.0

    def test_bel_at_last_period_is_zero(self, annual_2, flat_basis):
        liability = DependantLiability(annual_2)
        n = len(annual_2.periods)
        assert liability.get_bel(make_mp(), make_assumptions(flat_basis), n) == 0.0

    def test_bel_two_period_numerical_anchor(self, annual_2, flat_basis):
        """
        2-period annual calendar, q=0.02 both sexes, r=0.03, no inflation.

        Only trigger s=0 contributes:
          f_death(0) = 1.0 × 0.02 = 0.02
          cond_sp    = 0.98^2 / 0.98 = 0.98
          df(period 1 end) = 1/1.03^2
          V_dep(0)   = 1200 × 0.98 × (1/1.03^2) × 1.0
          BEL        = 0.02 × 1200 × 0.98 / 1.03^2
        """
        liability = DependantLiability(annual_2)
        bel = liability.get_bel(make_mp(), make_assumptions(flat_basis), 0)
        expected = 0.02 * 1200.0 * 0.98 / (1.03 ** 2)
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_proportional_to_weight(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        assump = make_assumptions(flat_basis)
        bel1 = liability.get_bel(make_mp(weight=1.0), assump, 0)
        bel3 = liability.get_bel(make_mp(weight=3.0), assump, 0)
        assert bel3 == pytest.approx(3.0 * bel1, rel=1e-9)

    def test_bel_proportional_to_pension(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        assump = make_assumptions(flat_basis)
        bel_base   = liability.get_bel(make_mp(pension_pa=1200.0), assump, 0)
        bel_double = liability.get_bel(make_mp(pension_pa=2400.0), assump, 0)
        assert bel_double == pytest.approx(2.0 * bel_base, rel=1e-9)

    def test_bel_decreases_as_timestep_advances(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        assump = make_assumptions(flat_basis)
        bel0 = liability.get_bel(make_mp(), assump, 0)
        bel3 = liability.get_bel(make_mp(), assump, 3)
        assert bel0 > bel3

    def test_higher_member_mortality_raises_bel(self, annual_10):
        """More member deaths → more triggers → higher dependant BEL."""
        def make_basis(qx):
            arr  = np.full(TABLE_LENGTH, qx)
            zero = np.zeros(TABLE_LENGTH)
            return MortalityBasis(arr, arr, zero, zero, ltr=0.0)

        liability = DependantLiability(annual_10)
        bel_low  = liability.get_bel(make_mp(), make_assumptions(make_basis(0.01)), 0)
        bel_high = liability.get_bel(make_mp(), make_assumptions(make_basis(0.05)), 0)
        assert bel_high > bel_low

    def test_higher_dependant_mortality_lowers_bel(self, annual_10):
        """
        Higher dependant mortality → shorter dependant payment term → lower BEL.
        Uses sex-differentiated tables: male=member at 0.02, female=dependant varies.
        """
        def make_sex_basis(dep_qx):
            arr_m  = np.full(TABLE_LENGTH, 0.02)
            arr_f  = np.full(TABLE_LENGTH, dep_qx)
            zero   = np.zeros(TABLE_LENGTH)
            return MortalityBasis(arr_m, arr_f, zero, zero, ltr=0.0)

        def assump(dep_q):
            ill = np.zeros(TABLE_LENGTH)
            return BPAAssumptions(
                mortality=make_sex_basis(dep_q),
                valuation_year=2023,
                discount_curve=RiskFreeRateCurve.flat(0.03),
                inflation_rate=0.0, rpi_rate=0.0, tv_rate=0.0,
                ill_health_rates=ill, expense_pa=0.0,
            )

        liability = DependantLiability(annual_10)
        bel_low_mort  = liability.get_bel(make_mp(), assump(0.01), 0)
        bel_high_mort = liability.get_bel(make_mp(), assump(0.08), 0)
        assert bel_low_mort > bel_high_mort

    def test_higher_discount_rate_lowers_bel(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        bel_low  = liability.get_bel(make_mp(), make_assumptions(flat_basis, 0.02), 0)
        bel_high = liability.get_bel(make_mp(), make_assumptions(flat_basis, 0.05), 0)
        assert bel_low > bel_high

    def test_convolution_matches_loop_reference(self, annual_10, flat_basis):
        """
        The implementation's vectorised convolution must match a naive Python
        double loop computing the identical mathematics.

        Outer loop: trigger period s (when member dies)
        Inner loop: payment period u (when dependant receives pension)

        For each (s, u) pair:
          contribution = f_death(s)
                       × pension
                       × cond_sp(s→u)          — P(dep alive at u | alive at s+1)
                       × df(u)                  — spot discount factor to end of u
                       × dt(u)                  — period length
        """
        liability = DependantLiability(annual_10)
        assump = make_assumptions(flat_basis)
        mp = make_mp()
        bel_impl = liability.get_bel(mp, assump, 0)

        # --- Naive double-loop reference ---
        future = annual_10.periods
        n = len(future)
        dt_arr = np.array([p.year_fraction for p in future])  # array of period lengths in years
        t_arr  = np.array([p.time_in_years  for p in future])
        df_arr = np.array([                                   # array of discount factors to end of each period      
            assump.discount_curve.discount_factor(
                (p.time_in_years + p.year_fraction) * 12.0
            )
            for p in future
        ])

        row = mp.iloc[0]

        # Member survival and death density
        sp_mem = survival_probs_variable_dt(
            row["member_age"], row["member_sex"],
            dt_arr, assump.valuation_year, assump.mortality,
        )
        q_mem_period = np.array([
            1.0 - (1.0 - qx_fn(
                row["member_age"] + t_arr[k],
                row["member_sex"],
                assump.valuation_year + int(t_arr[k]),
                assump.mortality,
            )) ** dt_arr[k]
            for k in range(n)
        ])
        # f_death(s) = P(member alive at start of s) × P(member dies in s)
        f_death = sp_mem[:n] * q_mem_period

        # Dependant survival from start of BEL window
        sp_dep = survival_probs_variable_dt(
            row["dependant_age"], row["dependant_sex"],
            dt_arr, assump.valuation_year, assump.mortality,
        )

        bel_ref = 0.0
        for s in range(n - 1):                                             # for each trigger period s (member death in s)
            sp_trigger = sp_dep[s + 1]                                     # P(dep alive at trigger date)
            if sp_trigger <= 0.0:
                continue

            v_dep_s = 0.0                                                  # present value of dependant payments triggered by member death at s
            for u in range(s + 1, n):                                       # for each payment period u after trigger s
                # cond_sp: P(dep alive at end of u | alive at trigger)
                cond_sp = sp_dep[u + 1] / sp_trigger                        # conditional survival probability of dependant from trigger to payment date. Division is a conditional probability: P(A and B) / P(B) = P(A | B) where A = "dep alive at end of u" and B = "dep alive at trigger"
                # Pension cashflow at end of period u
                # = pension_pa × cond_sp × df(end of u) × dt(u)
                v_dep_s += (
                    row["pension_pa"]                                       # base pension amount                        
                    * cond_sp                                               # probability dependant is alive to receive payment at end of u, conditional on being alive at trigger
                    * df_arr[u]                                              # a spot discount factor to end of u                                          
                    * dt_arr[u]                                         # period length in years to end of u (for discounting and to convert annual pension to period pension if needed)     
                )
            bel_ref += f_death[s] * v_dep_s                             # contribution to BEL from member death in period s

        bel_ref *= float(row["weight"])                                 # 

        assert bel_impl == pytest.approx(bel_ref, rel=1e-9)


# ---------------------------------------------------------------------------
# project_cashflows
# ---------------------------------------------------------------------------

class TestProjectCashflows:

    def test_returns_liability_cashflows(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(make_mp(), make_assumptions(flat_basis), 0)
        assert isinstance(result, LiabilityCashflows)

    def test_premiums_and_claims_zero(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(make_mp(), make_assumptions(flat_basis), 0)
        assert result.premiums == 0.0
        assert result.death_claims == 0.0
        assert result.surrender_payments == 0.0

    def test_pension_positive_for_in_force(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(make_mp(), make_assumptions(flat_basis), 0)
        assert result.maturity_payments > 0.0

    def test_zero_weight_zero_cashflow(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(weight=0.0), make_assumptions(flat_basis), 0
        )
        assert result.maturity_payments == 0.0

    def test_cashflow_period_0_numerical_anchor(self, annual_10, flat_basis):
        """
        Period 0, dt=1.0, q_mem=0.02, no inflation, weight=1, pension=1200:
        CF = pension × inflation_idx × dt × q_mem_period × weight
           = 1200 × 1.0 × 1.0 × 0.02 × 1.0 = 24.0
        """
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(make_mp(), make_assumptions(flat_basis), 0)
        assert result.maturity_payments == pytest.approx(24.0, rel=1e-6)

    def test_timestep_recorded(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        result = liability.project_cashflows(make_mp(), make_assumptions(flat_basis), 3)
        assert result.timestep == 3


# ---------------------------------------------------------------------------
# get_decrements
# ---------------------------------------------------------------------------

class TestGetDecrements:

    def test_returns_decrements(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        dec = liability.get_decrements(make_mp(), make_assumptions(flat_basis), 0)
        assert isinstance(result := dec, Decrements)

    def test_in_force_identity(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        dec = liability.get_decrements(make_mp(), make_assumptions(flat_basis), 0)
        assert dec.in_force_end == pytest.approx(
            dec.in_force_start - dec.deaths, rel=1e-9
        )

    def test_lapses_maturities_zero(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        dec = liability.get_decrements(make_mp(), make_assumptions(flat_basis), 0)
        assert dec.lapses == 0.0
        assert dec.maturities == 0.0

    def test_deaths_numerical_anchor(self, annual_10, flat_basis):
        """deaths = weight × q_member_period = 1.0 × 0.02"""
        liability = DependantLiability(annual_10)
        dec = liability.get_decrements(make_mp(), make_assumptions(flat_basis), 0)
        assert dec.deaths == pytest.approx(0.02, rel=1e-6)

    def test_reserve_equals_bel(self, annual_10, flat_basis):
        liability = DependantLiability(annual_10)
        assump = make_assumptions(flat_basis)
        mp = make_mp()
        assert liability.get_reserve(mp, assump, 0) == pytest.approx(
            liability.get_bel(mp, assump, 0), rel=1e-9
        )

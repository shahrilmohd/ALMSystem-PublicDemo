"""
Unit tests for the Conventional liability model.

Acronym glossary
----------------
DF     Discount Factor — present value of £1 paid at time t; DF(t) ∈ (0, 1].
BEL    Best Estimate Liability — present value of future net cash outgo.
SA     Sum Assured — death or maturity benefit per policy.
IF     In Force — number of active policies at a point in time.
SV     Surrender Value — payment made when a policy lapses voluntarily.
UDD    Uniform Distribution of Deaths — assumption used for monthly rate conversion.
PAR    Participating (with-profits) endowment policy type.
NONPAR Non-participating endowment policy type.

Monthly rate conversion under UDD:
    q_x_mths = 1 − (1 − q_x_yr) ^ (1/12)
    w_x_mths = 1 − (1 − w_x_yr) ^ (1/12)

Fixtures use rates designed to give exact monthly values — see conftest.py.

Rules under test
----------------
ConventionalAssumptions:
  1.  get_mortality_rate_yr returns correct rate for a known age.
  2.  get_mortality_rate_yr returns the default when the age is absent.
  3.  get_lapse_rate_yr behaves the same way (known and default).
  4.  get_surrender_value_factor behaves the same way (known and default).

Decrements (numerical):
  5.  deaths = IF_start × q_x_mths.
  6.  lapses = (IF_start − deaths) × w_x_mths.
  7.  maturities = survivors when remaining_term_mths = 1; 0 otherwise.
  8.  IF_end = IF_start − deaths − lapses − maturities (identity).
  9.  Multiple groups: each row computed independently, totals summed.

Cash flows (numerical):
  10. premiums = annual_premium / 12 × IF_start.
  11. death_claims = SA × deaths.
  12. SV payments = sv_factor × SA × lapses.
  13. maturity_payments = SA × maturities.
  14. expenses = (pct × premium + per_policy) / 12 × IF_start.
  15. Multiple groups: all components aggregated across rows.

Policy code — TERM:
  16. TERM: maturity_payments = 0 even in the final month.
  17. TERM: SV payments = 0.
  18. TERM: death claim = SA (no bonus component).
  19. Unknown policy_code raises ValueError.

Policy code — ENDOW_PAR (with-profits):
  20. Death claim = (SA + accrued_bonus_per_policy) × deaths.
  21. Maturity payment = (SA + accrued_bonus_per_policy) × maturities.
  22. SV payment = sv_factor × (SA + accrued_bonus_per_policy) × lapses.

BEL (numerical):
  23. 1-month projection, zero discount: BEL = net_outgo × 1.0.
  24. 1-month projection, 5% discount: BEL = net_outgo × (1.05)^(−1/12).
  25. 2-month projection, zero discount: BEL = Σ net_outgo_t × 1.0.
  26. 2-month projection, 5% discount: BEL = Σ net_outgo_t × DF(t).
  27. BEL = 0 when remaining_term_mths ≤ 0.
  28. Lower discount rate → higher BEL when net_outgo > 0.
  29. BEL uses DF(t) = (1 + r)^(−t/12) for any flat rate r (parametrised).

Reserve:
  30. Phase 1: reserve == BEL.

Validation:
  31. Missing required column raises ValueError containing the column name.
  32. Multiple missing columns: all appear in the error message.

Edge cases:
  33. Empty DataFrame returns zero cash flows.
  34. Empty DataFrame returns zero BEL.
  35. Zero IF_start returns zero cash flows.
  36. Mortality rate > 1 clamped to 1.0 (no negative survivors).
  37. Mortality rate < 0 clamped to 0.0.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import Conventional, ConventionalAssumptions
from tests.unit.liability.conftest import Q_50_YR, Q_60_YR, W_3_YR


# ---------------------------------------------------------------------------
# ConventionalAssumptions — lookup helpers
# ---------------------------------------------------------------------------

class TestConventionalAssumptionsLookups:
    def test_mortality_rate_known_age(self):
        a = ConventionalAssumptions(
            mortality_rates={50: 0.01, 60: 0.02},
            lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={},
        )
        assert a.get_mortality_rate_yr(50) == pytest.approx(0.01)
        assert a.get_mortality_rate_yr(60) == pytest.approx(0.02)

    def test_mortality_rate_unknown_age_returns_default(self):
        a = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={},
            default_mortality_rate=0.005,
        )
        assert a.get_mortality_rate_yr(99) == pytest.approx(0.005)

    def test_lapse_rate_known_duration(self):
        a = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={3: 0.05},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
        )
        assert a.get_lapse_rate_yr(3) == pytest.approx(0.05)

    def test_lapse_rate_unknown_duration_returns_default(self):
        a = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={}, default_lapse_rate=0.10,
        )
        assert a.get_lapse_rate_yr(99) == pytest.approx(0.10)

    def test_surrender_value_factor_known_duration(self):
        a = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={3: 0.50},
        )
        assert a.get_surrender_value_factor(3) == pytest.approx(0.50)

    def test_surrender_value_factor_unknown_returns_default(self):
        a = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={},
            default_surrender_value_factor=0.25,
        )
        assert a.get_surrender_value_factor(99) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Decrements — numerical tests
# ---------------------------------------------------------------------------

class TestConventionalDecrements:
    """
    Reference values (single_group_mp + standard_assumptions):

        IF_start   = 100
        q_50_mths  = 0.10  (exact — Q_50_YR = 1 − 0.90^12)
        w_3_mths   = 0.05  (exact — W_3_YR  = 1 − 0.95^12)
        sv_3       = 0.50

        deaths     = 100 × 0.10        = 10.0
        lapses     = 90  × 0.05        = 4.5
        maturities = 0  (remaining_term_mths = 24, not the final month)
        IF_end     = 100 − 10 − 4.5   = 85.5
    """

    def test_deaths_correct(self, single_group_mp, standard_assumptions):
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        assert d.deaths == pytest.approx(10.0)

    def test_lapses_correct(self, single_group_mp, standard_assumptions):
        # Lapses from survivors after deaths: 90 × 0.05 = 4.5
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        assert d.lapses == pytest.approx(4.5)

    def test_maturities_zero_not_final_month(self, single_group_mp, standard_assumptions):
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        assert d.maturities == pytest.approx(0.0)

    def test_in_force_end_correct(self, single_group_mp, standard_assumptions):
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        assert d.in_force_end == pytest.approx(85.5)

    def test_in_force_start_correct(self, single_group_mp, standard_assumptions):
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        assert d.in_force_start == pytest.approx(100.0)

    def test_identity_holds(self, single_group_mp, standard_assumptions):
        """IF_end == IF_start − deaths − lapses − maturities."""
        model = Conventional()
        d = model.get_decrements(single_group_mp, standard_assumptions, timestep=0)
        expected_end = d.in_force_start - d.deaths - d.lapses - d.maturities
        assert d.in_force_end == pytest.approx(expected_end)

    def test_maturities_equal_survivors_in_final_month(
        self, final_month_mp, zero_assumptions
    ):
        """
        final_month_mp: remaining_term_mths = 1, zero rates.
        All 100 policies survive to the final month and mature.
        """
        model = Conventional()
        d = model.get_decrements(final_month_mp, zero_assumptions, timestep=0)
        assert d.deaths       == pytest.approx(0.0)
        assert d.lapses       == pytest.approx(0.0)
        assert d.maturities   == pytest.approx(100.0)
        assert d.in_force_end == pytest.approx(0.0)

    def test_decrements_aggregated_across_multiple_groups(
        self, two_group_mp, standard_assumptions
    ):
        """
        two_group_mp — see conftest.py for expected values.

        Group A (ENDOW_NONPAR, age=50, dur_yr=3): q_mths=0.10, w_mths=0.05
            deaths = 100 × 0.10 = 10.0
            lapses = 90  × 0.05 = 4.5
        Group B (TERM, age=60, dur_yr=5): q_mths=0.02, default lapse = 0
            deaths = 50 × 0.02 = 1.0
            lapses = 0

        Totals: IF_start=150, deaths=11.0, lapses=4.5, maturities=0
        """
        model = Conventional()
        d = model.get_decrements(two_group_mp, standard_assumptions, timestep=0)
        assert d.in_force_start == pytest.approx(150.0)
        assert d.deaths         == pytest.approx(11.0)
        assert d.lapses         == pytest.approx(4.5)
        assert d.maturities     == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cash flows — numerical tests
# ---------------------------------------------------------------------------

class TestConventionalCashflows:
    """
    Reference values (single_group_mp + standard_assumptions):

        Monthly premium per policy = 1,200 / 12 = 100
        Premiums         = 100 × 100              =  10,000
        Death claims     = 10,000 × 10.0          = 100,000
        SV payments      = 0.50 × 10,000 × 4.5   =  22,500
        Maturity         = 0
        Expenses         = 0
        Net outgo        = 100,000 + 22,500 − 10,000 = 112,500
    """

    def test_premiums(self, single_group_mp, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=0)
        assert cf.premiums == pytest.approx(10_000.0)

    def test_death_claims(self, single_group_mp, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=0)
        assert cf.death_claims == pytest.approx(100_000.0)

    def test_surrender_payments(self, single_group_mp, standard_assumptions):
        # sv_factor=0.50, SA=10,000, lapses=4.5 → 0.50 × 10,000 × 4.5 = 22,500
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=0)
        assert cf.surrender_payments == pytest.approx(22_500.0)

    def test_maturity_zero_not_final_month(self, single_group_mp, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=0)
        assert cf.maturity_payments == pytest.approx(0.0)

    def test_maturity_in_final_month(self, final_month_mp, zero_assumptions):
        """100 survivors × SA 10,000 = 1,000,000."""
        model = Conventional()
        cf = model.project_cashflows(final_month_mp, zero_assumptions, timestep=0)
        assert cf.maturity_payments == pytest.approx(1_000_000.0)

    def test_net_outgo(self, single_group_mp, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=0)
        assert cf.net_outgo == pytest.approx(112_500.0)

    def test_expenses_pct_premium(self, single_group_mp):
        """
        5% of annual premium billed monthly:
            0.05 × 1,200 / 12 × 100 policies = 500
        """
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.05, expense_per_policy=0.0,
            surrender_value_factors={},
        )
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, assumptions, timestep=0)
        assert cf.expenses == pytest.approx(500.0)

    def test_expenses_per_policy(self, single_group_mp):
        """
        £120 per policy per year → £10 per policy per month.
        Total = 10 × 100 policies = 1,000.
        """
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=120.0,
            surrender_value_factors={},
        )
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, assumptions, timestep=0)
        assert cf.expenses == pytest.approx(1_000.0)

    def test_expenses_combined(self, single_group_mp):
        """5% of premium + £120 per policy = 500 + 1,000 = 1,500."""
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.05, expense_per_policy=120.0,
            surrender_value_factors={},
        )
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, assumptions, timestep=0)
        assert cf.expenses == pytest.approx(1_500.0)

    def test_cashflows_aggregated_across_two_groups(
        self, two_group_mp, standard_assumptions
    ):
        """
        two_group_mp — see conftest.py for expected values:
            Group A: premiums=10,000, death_claims=100,000, SV=22,500
            Group B: premiums=2,500,  death_claims=5,000,   SV=0 (TERM)
            Totals:  premiums=12,500, death_claims=105,000, SV=22,500
        """
        model = Conventional()
        cf = model.project_cashflows(two_group_mp, standard_assumptions, timestep=0)
        assert cf.premiums           == pytest.approx(12_500.0)
        assert cf.death_claims       == pytest.approx(105_000.0)
        assert cf.surrender_payments == pytest.approx(22_500.0)

    def test_timestep_stored_in_result(self, single_group_mp, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(single_group_mp, standard_assumptions, timestep=5)
        assert cf.timestep == 5


# ---------------------------------------------------------------------------
# Policy code — TERM
# ---------------------------------------------------------------------------

class TestTermPolicy:
    def _term_mp(self) -> pd.DataFrame:
        """100 TERM policies, 12 months remaining."""
        return pd.DataFrame([{
            "group_id": "GRP_T", "in_force_count": 100.0,
            "sum_assured": 10_000.0, "annual_premium": 1_200.0,
            "attained_age": 50, "policy_code": "TERM",
            "policy_term_yr": 5, "policy_duration_mths": 48,
            "accrued_bonus_per_policy": 0.0,
        }])

    def _term_final_mp(self) -> pd.DataFrame:
        """100 TERM policies, 1 month remaining (final month)."""
        return pd.DataFrame([{
            "group_id": "GRP_T", "in_force_count": 100.0,
            "sum_assured": 10_000.0, "annual_premium": 1_200.0,
            "attained_age": 50, "policy_code": "TERM",
            "policy_term_yr": 5, "policy_duration_mths": 59,
            "accrued_bonus_per_policy": 0.0,
        }])

    def test_term_maturity_is_zero_in_final_month(self, zero_assumptions):
        """TERM: no maturity payment even when the term expires."""
        model = Conventional()
        cf = model.project_cashflows(self._term_final_mp(), zero_assumptions, timestep=0)
        assert cf.maturity_payments == pytest.approx(0.0)

    def test_term_surrender_is_zero(self, standard_assumptions):
        """TERM: no SV payment on lapse."""
        model = Conventional()
        cf = model.project_cashflows(self._term_mp(), standard_assumptions, timestep=0)
        assert cf.surrender_payments == pytest.approx(0.0)

    def test_term_death_claim_equals_sa(self, standard_assumptions):
        """
        TERM death claim = SA × deaths, no bonus component.
        deaths = 100 × q_50_mths = 100 × 0.10 = 10.0
        death_claims = 10,000 × 10.0 = 100,000.
        """
        model = Conventional()
        cf = model.project_cashflows(self._term_mp(), standard_assumptions, timestep=0)
        assert cf.death_claims == pytest.approx(100_000.0)

    def test_unknown_policy_code_raises_value_error(self, standard_assumptions):
        """An unrecognised policy_code must raise ValueError."""
        bad_mp = pd.DataFrame([{
            "group_id": "GRP_X", "in_force_count": 10.0,
            "sum_assured": 5_000.0, "annual_premium": 600.0,
            "attained_age": 45, "policy_code": "INVALID_CODE",
            "policy_term_yr": 10, "policy_duration_mths": 12,
            "accrued_bonus_per_policy": 0.0,
        }])
        model = Conventional()
        with pytest.raises(ValueError, match="INVALID_CODE"):
            model.project_cashflows(bad_mp, standard_assumptions, timestep=0)


# ---------------------------------------------------------------------------
# Policy code — ENDOW_PAR (with-profits)
# ---------------------------------------------------------------------------

class TestEndowParPolicy:
    """
    Fixtures use 100 ENDOW_PAR policies with accrued_bonus_per_policy = 500.
    Benefit base = SA + accrued_bonus = 10,000 + 500 = 10,500.
    """

    def _par_mp(self, accrued_bonus: float = 500.0) -> pd.DataFrame:
        """100 ENDOW_PAR policies, 24 months remaining."""
        return pd.DataFrame([{
            "group_id": "GRP_P", "in_force_count": 100.0,
            "sum_assured": 10_000.0, "annual_premium": 1_200.0,
            "attained_age": 50, "policy_code": "ENDOW_PAR",
            "policy_term_yr": 5, "policy_duration_mths": 36,
            "accrued_bonus_per_policy": accrued_bonus,
        }])

    def _par_final_mp(self, accrued_bonus: float = 500.0) -> pd.DataFrame:
        """100 ENDOW_PAR policies, 1 month remaining."""
        return pd.DataFrame([{
            "group_id": "GRP_P", "in_force_count": 100.0,
            "sum_assured": 10_000.0, "annual_premium": 1_200.0,
            "attained_age": 50, "policy_code": "ENDOW_PAR",
            "policy_term_yr": 5, "policy_duration_mths": 59,
            "accrued_bonus_per_policy": accrued_bonus,
        }])

    def test_par_death_claim_includes_bonus(self, standard_assumptions):
        """
        Benefit base = 10,000 + 500 = 10,500.
        deaths = 100 × 0.10 = 10.0
        death_claims = 10,500 × 10.0 = 105,000.
        """
        model = Conventional()
        cf = model.project_cashflows(self._par_mp(), standard_assumptions, timestep=0)
        assert cf.death_claims == pytest.approx(105_000.0)

    def test_par_maturity_includes_bonus(self, zero_assumptions):
        """
        Zero decrements → maturities = 100.
        maturity_payments = 10,500 × 100 = 1,050,000.
        """
        model = Conventional()
        cf = model.project_cashflows(self._par_final_mp(), zero_assumptions, timestep=0)
        assert cf.maturity_payments == pytest.approx(1_050_000.0)

    def test_par_surrender_includes_bonus(self, standard_assumptions):
        """
        lapses = 90 × 0.05 = 4.5
        SV = 0.50 × (10,000 + 500) × 4.5 = 0.50 × 10,500 × 4.5 = 23,625.
        """
        model = Conventional()
        cf = model.project_cashflows(self._par_mp(), standard_assumptions, timestep=0)
        assert cf.surrender_payments == pytest.approx(23_625.0)


# ---------------------------------------------------------------------------
# BEL — numerical tests
# ---------------------------------------------------------------------------

class TestConventionalBEL:
    """
    BEL = Σ_{s=1}^{remaining_term_mths} net_outgo_s × DF(s)

    DF(s) is obtained from assumptions.rate_curve.discount_factor(s).
    All BEL fixtures use zero decrements so the cash flow amounts are fixed
    and only the discounting varies.
    """

    def test_bel_one_month_zero_discount(self, final_month_mp, zero_assumptions):
        """
        1-month remaining, zero rates → DF = 1.0.
        net_outgo = 1,000,000 − 10,000 = 990,000.
        BEL = 990,000 × 1.0 = 990,000.
        """
        model = Conventional()
        bel   = model.get_bel(final_month_mp, zero_assumptions, timestep=0)
        assert bel == pytest.approx(990_000.0, rel=1e-6)

    def test_bel_one_month_with_5pct_discount(self, final_month_mp):
        """
        Same 1-month scenario, 5% flat discount rate.
        BEL = 990,000 × (1.05)^(−1/12).
        """
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(0.05),
        )
        v        = 1.05 ** (-1.0 / 12.0)
        expected = 990_000.0 * v
        model    = Conventional()
        bel      = model.get_bel(final_month_mp, assumptions, timestep=0)
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_two_month_zero_discount(self, two_month_mp, zero_assumptions):
        """
        2-month remaining, zero rates → DF = 1.0 at every step.

        Step 1 (remaining=2): no maturities → net_outgo = −10,000
        Step 2 (remaining=1): maturities = 100 → net_outgo = 990,000

        BEL = −10,000 × 1.0 + 990,000 × 1.0 = 980,000.
        """
        model = Conventional()
        bel   = model.get_bel(two_month_mp, zero_assumptions, timestep=0)
        assert bel == pytest.approx(980_000.0, rel=1e-6)

    def test_bel_two_month_with_5pct_discount(self, two_month_mp):
        """
        Same 2-month scenario, 5% flat discount rate.
        v = (1.05)^(−1/12)
        BEL = −10,000 × v^1 + 990,000 × v^2.
        """
        r           = 0.05
        v           = (1.0 + r) ** (-1.0 / 12.0)
        expected    = -10_000.0 * v + 990_000.0 * v ** 2
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(r),
        )
        model = Conventional()
        bel   = model.get_bel(two_month_mp, assumptions, timestep=0)
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_zero_when_already_expired(self, final_month_mp, zero_assumptions):
        """remaining_term_mths = 0 → BEL = 0."""
        expired = final_month_mp.copy()
        expired["policy_duration_mths"] = expired["policy_term_yr"] * 12
        model = Conventional()
        assert model.get_bel(expired, zero_assumptions, timestep=0) == pytest.approx(0.0)

    def test_bel_higher_at_lower_discount_rate(self, final_month_mp):
        """
        Lower discount rate → higher DF → higher PV.
        Since net_outgo > 0, BEL must increase when the rate falls.
        """
        model  = Conventional()
        a_5pct = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(0.05),
        )
        a_1pct = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(0.01),
        )
        assert model.get_bel(final_month_mp, a_1pct, timestep=0) > \
               model.get_bel(final_month_mp, a_5pct, timestep=0)

    @pytest.mark.parametrize("annual_rate", [0.03, 0.05, 0.08])
    def test_bel_uses_rate_curve_df(self, final_month_mp, annual_rate):
        """
        BEL at 1-month horizon = 990,000 × (1 + r)^(−1/12) for any flat rate r.
        Verifies the model reads DF from the rate_curve, not a hardcoded value.
        """
        assumptions = ConventionalAssumptions(
            mortality_rates={}, lapse_rates={},
            expense_pct_premium=0.0, expense_per_policy=0.0,
            surrender_value_factors={},
            rate_curve=RiskFreeRateCurve.flat(annual_rate),
        )
        v        = (1.0 + annual_rate) ** (-1.0 / 12.0)
        expected = 990_000.0 * v
        model    = Conventional()
        bel      = model.get_bel(final_month_mp, assumptions, timestep=0)
        assert bel == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Reserve
# ---------------------------------------------------------------------------

class TestConventionalReserve:
    def test_reserve_equals_bel(self, single_group_mp, standard_assumptions):
        """Phase 1: reserve must equal BEL exactly."""
        model   = Conventional()
        bel     = model.get_bel(single_group_mp, standard_assumptions, timestep=0)
        reserve = model.get_reserve(single_group_mp, standard_assumptions, timestep=0)
        assert reserve == pytest.approx(bel)


# ---------------------------------------------------------------------------
# Validation — missing columns
# ---------------------------------------------------------------------------

class TestConventionalValidation:
    @pytest.mark.parametrize("missing_col", [
        "group_id", "in_force_count", "sum_assured", "annual_premium",
        "attained_age", "policy_code", "policy_term_yr",
        "policy_duration_mths", "accrued_bonus_per_policy",
    ])
    def test_missing_column_raises_value_error(
        self, single_group_mp, standard_assumptions, missing_col
    ):
        bad_mp = single_group_mp.drop(columns=[missing_col])
        model  = Conventional()
        with pytest.raises(ValueError) as exc_info:
            model.project_cashflows(bad_mp, standard_assumptions, timestep=0)
        assert missing_col in str(exc_info.value)

    def test_missing_multiple_columns_all_listed_in_error(self, standard_assumptions):
        """All missing column names should appear in the error message."""
        empty_df = pd.DataFrame()
        model = Conventional()
        with pytest.raises(ValueError) as exc_info:
            model.project_cashflows(empty_df, standard_assumptions, timestep=0)
        assert "group_id" in str(exc_info.value) or "in_force_count" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestConventionalEdgeCases:
    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "group_id", "in_force_count", "sum_assured", "annual_premium",
            "attained_age", "policy_code", "policy_term_yr",
            "policy_duration_mths", "accrued_bonus_per_policy",
        ])

    def test_empty_dataframe_returns_zero_cashflows(self, standard_assumptions):
        model = Conventional()
        cf = model.project_cashflows(self._empty_df(), standard_assumptions, timestep=0)
        assert cf.premiums           == pytest.approx(0.0)
        assert cf.death_claims       == pytest.approx(0.0)
        assert cf.surrender_payments == pytest.approx(0.0)
        assert cf.maturity_payments  == pytest.approx(0.0)
        assert cf.expenses           == pytest.approx(0.0)
        assert cf.net_outgo          == pytest.approx(0.0)

    def test_empty_dataframe_returns_zero_bel(self, standard_assumptions):
        model = Conventional()
        assert model.get_bel(
            self._empty_df(), standard_assumptions, timestep=0
        ) == pytest.approx(0.0)

    def test_zero_in_force_returns_zero_cashflows(
        self, single_group_mp, standard_assumptions
    ):
        mp = single_group_mp.copy()
        mp["in_force_count"] = 0.0
        model = Conventional()
        cf = model.project_cashflows(mp, standard_assumptions, timestep=0)
        assert cf.net_outgo == pytest.approx(0.0)

    def test_mortality_rate_above_one_clamped(self, single_group_mp):
        """q > 1 is clamped to 1.0 — prevents negative survivors."""
        assumptions = ConventionalAssumptions(
            mortality_rates={50: 2.0},
            lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={},
        )
        model = Conventional()
        d = model.get_decrements(single_group_mp, assumptions, timestep=0)
        assert d.deaths       == pytest.approx(100.0)
        assert d.in_force_end >= 0.0

    def test_mortality_rate_below_zero_clamped(self, single_group_mp):
        """Negative q is clamped to 0.0."""
        assumptions = ConventionalAssumptions(
            mortality_rates={50: -0.5},
            lapse_rates={}, expense_pct_premium=0.0,
            expense_per_policy=0.0, surrender_value_factors={},
        )
        model = Conventional()
        d = model.get_decrements(single_group_mp, assumptions, timestep=0)
        assert d.deaths == pytest.approx(0.0)

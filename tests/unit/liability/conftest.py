"""
Shared fixtures for engine.liability unit tests.

Hand-calculated expected values used in numerical tests are documented
inline in each fixture so the maths can be verified independently.

Monthly rate conversion trick
-----------------------------
To make monthly decrements exact integers, we exploit:
    q_x_mths = 1 − (1 − q_x_yr) ^ (1/12)

So if we set q_x_yr = 1 − b^12 then q_x_mths = 1 − b exactly.

Constants used in standard_assumptions:
    Q_50_YR = 1 − 0.90^12  →  q_50_mths = 0.10  (exactly)
    Q_60_YR = 1 − 0.98^12  →  q_60_mths = 0.02  (exactly)
    W_3_YR  = 1 − 0.95^12  →  w_3_mths  = 0.05  (exactly)

These constants are defined at module level so test files can import them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions


# Annual rates that produce exact monthly rates (see module docstring)
Q_50_YR: float = 1.0 - 0.90 ** 12   # → q_50_mths = 0.10 exactly
Q_60_YR: float = 1.0 - 0.98 ** 12   # → q_60_mths = 0.02 exactly
W_3_YR:  float = 1.0 - 0.95 ** 12   # → w_3_mths  = 0.05 exactly


# ---------------------------------------------------------------------------
# Assumption fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_assumptions() -> ConventionalAssumptions:
    """
    All rates zero. Every policy survives to its final month unchanged.
    discount_factor(t) = 1.0 for all t (flat zero curve).
    """
    return ConventionalAssumptions(
        mortality_rates={},
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(0.0),
    )


@pytest.fixture
def standard_assumptions() -> ConventionalAssumptions:
    """
    Realistic assumptions with exact monthly rates (see module constants).

        q_50_yr = 1 − 0.90^12  →  q_50_mths = 0.10
        q_60_yr = 1 − 0.98^12  →  q_60_mths = 0.02
        w_3_yr  = 1 − 0.95^12  →  w_3_mths  = 0.05
        sv_3    = 0.50  (50% of benefit base at duration 3)
        discount = 5% flat curve
    """
    return ConventionalAssumptions(
        mortality_rates={50: Q_50_YR, 60: Q_60_YR},
        lapse_rates={3: W_3_YR},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={3: 0.50},
        rate_curve=RiskFreeRateCurve.flat(0.05),
    )


# ---------------------------------------------------------------------------
# Model point DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_group_mp() -> pd.DataFrame:
    """
    Single group: 100 ENDOW_NONPAR policies, age 50, 3 full years in force.

        policy_term_yr=5, policy_duration_mths=36
        remaining_term_mths = 5×12 − 36 = 24  (not final month)
        duration_yr = 36 // 12 = 3  (rate lookup key)

    With standard_assumptions:
        q_50_mths = 0.10,  w_3_mths = 0.05,  sv_3 = 0.50

        Deaths     = 100 × 0.10          = 10.0
        Lapses     = (100 − 10) × 0.05   = 4.5
        Maturities = 0  (remaining=24 > 1)
        IF end     = 100 − 10 − 4.5      = 85.5

        Monthly premium per policy = 1200 / 12 = 100
        Premiums         = 100 × 100             = 10,000
        Death claims     = 10,000 × 10.0         = 100,000
        Surrender        = 0.50 × 10,000 × 4.5   = 22,500
        Maturity         = 0
        Net outgo        = 100,000 + 22,500 − 10,000 = 112,500
    """
    return pd.DataFrame([{
        "group_id":                "GRP_A",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          5,
        "policy_duration_mths":    36,
        "accrued_bonus_per_policy": 0.0,
    }])


@pytest.fixture
def final_month_mp() -> pd.DataFrame:
    """
    Single group: 100 ENDOW_NONPAR policies, 1 month remaining.

        policy_term_yr=5, policy_duration_mths=59
        remaining_term_mths = 5×12 − 59 = 1  (final month)
        duration_yr = 59 // 12 = 4

    With zero_assumptions (zero mortality, zero lapses):
        Deaths     = 0
        Lapses     = 0
        Maturities = 100  (all survivors mature in final month)
        IF end     = 0

        Monthly premium = 1200 / 12 = 100 per policy
        Premiums         = 100 × 100    = 10,000
        Maturity         = 10,000 × 100 = 1,000,000
        Net outgo        = 1,000,000 − 10,000 = 990,000

        BEL = 990,000 × v^1  where v = (1.05)^(−1/12)
    """
    return pd.DataFrame([{
        "group_id":                "GRP_B",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            60,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          5,
        "policy_duration_mths":    59,
        "accrued_bonus_per_policy": 0.0,
    }])


@pytest.fixture
def two_month_mp() -> pd.DataFrame:
    """
    Single group: 100 ENDOW_NONPAR policies, 2 months remaining.

        policy_term_yr=5, policy_duration_mths=58
        remaining_term_mths = 5×12 − 58 = 2

    With zero_assumptions (zero mortality, zero lapses):
        Step 1 (remaining=2): no maturities
            Premiums = 10,000, outgo = 0
            Net outgo = −10,000

        Step 2 (after advance, policy_duration_mths=59, remaining=1):
            Maturities = 100 → maturity_payments = 1,000,000
            Net outgo = 1,000,000 − 10,000 = 990,000

        v = (1.05)^(−1/12)
        BEL = −10,000 × v^1 + 990,000 × v^2
    """
    return pd.DataFrame([{
        "group_id":                "GRP_C",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            40,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          5,
        "policy_duration_mths":    58,
        "accrued_bonus_per_policy": 0.0,
    }])


@pytest.fixture
def two_group_mp() -> pd.DataFrame:
    """
    Two groups with different policy types.

    Group A: ENDOW_NONPAR, 100 policies, age=50, dur_mths=36, remaining=24
    Group B: TERM,         50 policies, age=60, dur_mths=60, remaining=12
             (policy_term_yr=6 → remaining = 72−60 = 12 months)

    With standard_assumptions:
        Group A (dur_yr=3): q_50_mths=0.10, w_3_mths=0.05, sv_3=0.50
            Deaths     = 100 × 0.10        = 10.0
            Lapses     = 90 × 0.05         = 4.5
            Premiums   = 1200/12 × 100     = 10,000
            Death clms = 10,000 × 10       = 100,000
            Surrender  = 0.50 × 10,000 × 4.5 = 22,500

        Group B TERM (dur_yr=5, w_5=default 0):
            Deaths     = 50 × 0.02         = 1.0
            Lapses     = 0
            Premiums   = 600/12 × 50       = 2,500
            Death clms = 5,000 × 1.0       = 5,000
            Surrender  = 0  (TERM has no surrender)
            Maturity   = 0  (remaining=12 > 1)

    Totals:
        Premiums     = 10,000 + 2,500 = 12,500
        Death claims = 100,000 + 5,000 = 105,000
        Surrender    = 22,500
    """
    return pd.DataFrame([
        {
            "group_id": "GRP_A", "in_force_count": 100.0,
            "sum_assured": 10_000.0, "annual_premium": 1_200.0,
            "attained_age": 50, "policy_code": "ENDOW_NONPAR",
            "policy_term_yr": 5, "policy_duration_mths": 36,
            "accrued_bonus_per_policy": 0.0,
        },
        {
            "group_id": "GRP_B", "in_force_count": 50.0,
            "sum_assured": 5_000.0, "annual_premium": 600.0,
            "attained_age": 60, "policy_code": "TERM",
            "policy_term_yr": 6, "policy_duration_mths": 60,
            "accrued_bonus_per_policy": 0.0,
        },
    ])

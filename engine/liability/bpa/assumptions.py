"""
BPA assumption dataclasses — injected into all BPA liability classes.

Design
------
BPAAssumptions is a frozen dataclass that carries every actuarial assumption
needed by the four BPA liability classes.  It is constructed once by the run
mode orchestrator (BPARun, Step 20) and injected at each call to
project_cashflows(), get_bel(), and get_decrements().  No assumption is
read from a file inside the engine (CLAUDE.md rule 2).

Sub-structures
--------------
RetirementRates    — age-specific retirement probability rules (DECISIONS.md §25)
DependantAssumptions — proportion with dependant and age difference by sex
                       (DECISIONS.md §26)

Discount curve
--------------
BPAAssumptions carries a RiskFreeRateCurve (pre-MA) for BEL discounting.
This is the same curve type already used by DeterministicRun for conventional
products.  In Step 20 it is replaced by the post-MA adjusted curve — the
field name and type are unchanged; only the curve passed in differs.

Default values
--------------
BPAAssumptions.default() constructs a sensible basis for unit testing and
exploratory runs.  It uses MortalityBasis from a caller-supplied array
(tests pass a synthetic flat basis; production passes the S3/CMI tables
loaded by BPADataLoader).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.mortality import MortalityBasis


# ---------------------------------------------------------------------------
# RetirementRates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetirementRates:
    """
    Retirement probability schedule for deferred members (DECISIONS.md §25).

    Rates are annual probabilities applied at each projection period.

    Fields
    ------
    early_retirement_rate : float
        p.a. probability of early retirement for members with ERA ≤ age < NRA.
        Default 0.05 (5% p.a.).
    late_retirement_rate : float
        p.a. probability of late retirement for members with age > NRA.
        Default 0.10 (10% p.a.).
    """
    early_retirement_rate: float = 0.05
    late_retirement_rate:  float = 0.10

    def __post_init__(self) -> None:
        if not (0.0 <= self.early_retirement_rate <= 1.0):
            raise ValueError("early_retirement_rate must be in [0, 1]")
        if not (0.0 <= self.late_retirement_rate <= 1.0):
            raise ValueError("late_retirement_rate must be in [0, 1]")


# ---------------------------------------------------------------------------
# DependantAssumptions
# ---------------------------------------------------------------------------

# Age band labels used as keys in proportion_with_dependant
_AGE_BANDS = ("lt60", "60to70", "gt70")

# Default proportions from DECISIONS.md §26 Table
_DEFAULT_PROPORTION: dict[tuple[str, str], float] = {
    ("M", "lt60"):   0.80,
    ("M", "60to70"): 0.85,
    ("M", "gt70"):   0.80,
    ("F", "lt60"):   0.65,
    ("F", "60to70"): 0.70,
    ("F", "gt70"):   0.60,
}

# Default age differences from DECISIONS.md §26
# Positive = dependant is younger than member
_DEFAULT_AGE_DIFF: dict[tuple[str, str], float] = {
    ("M", "F"):  3.0,    # male member, female dependant: dep 3 years younger
    ("F", "M"): -3.0,    # female member, male dependant: dep 3 years older
    ("M", "M"):  0.0,
    ("F", "F"):  0.0,
}


def _age_band(age: float) -> str:
    if age < 60:
        return "lt60"
    elif age <= 70:
        return "60to70"
    return "gt70"


@dataclass(frozen=True)
class DependantAssumptions:
    """
    Dependant proportion and age difference assumptions (DECISIONS.md §26).

    proportion_with_dependant
        Dict keyed by (member_sex, age_band) → float in [0, 1].
        age_band is "lt60", "60to70", or "gt70".

    age_difference
        Dict keyed by (member_sex, dependant_sex) → float (years).
        Positive = dependant is younger than member.

    dependant_pension_fraction
        Dependant pension as a fraction of the member's pension.
        Typical values: 0.50, 0.67.  Default 0.50.
    """
    proportion_with_dependant: dict[tuple[str, str], float] = field(
        default_factory=lambda: dict(_DEFAULT_PROPORTION)
    )
    age_difference: dict[tuple[str, str], float] = field(
        default_factory=lambda: dict(_DEFAULT_AGE_DIFF)
    )
    dependant_pension_fraction: float = 0.50

    def __post_init__(self) -> None:
        for (sex, band), prop in self.proportion_with_dependant.items():
            if sex not in ("M", "F"):
                raise ValueError(f"proportion_with_dependant key sex must be M or F; got {sex!r}")
            if band not in _AGE_BANDS:
                raise ValueError(f"proportion_with_dependant key age_band must be one of {_AGE_BANDS}")
            if not (0.0 <= prop <= 1.0):
                raise ValueError(f"proportion_with_dependant value must be in [0, 1]; got {prop}")
        if not (0.0 < self.dependant_pension_fraction <= 1.0):
            raise ValueError("dependant_pension_fraction must be in (0, 1]")

    def proportion(self, member_sex: str, member_age: float) -> float:
        """Return the dependant proportion for this member."""
        return self.proportion_with_dependant[(member_sex, _age_band(member_age))]

    def dependant_age(self, member_sex: str, dependant_sex: str, member_age: float) -> float:
        """Return the dependant's age given the member's age and sex pairing."""
        diff = self.age_difference.get((member_sex, dependant_sex), 0.0)
        return member_age - diff


# ---------------------------------------------------------------------------
# BPAAssumptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPAAssumptions:
    """
    Complete assumption set for BPA liability projections.

    Injected by the run mode orchestrator at each call to the liability
    classes.  No field is read from a file at the model level.

    Parameters
    ----------
    mortality : MortalityBasis
        S3 base tables + CMI improvement parameters.  Loaded from CSV by
        BPADataLoader; tests use a synthetic flat basis.
    valuation_year : int
        Calendar year at the valuation date.  Used for CMI improvement
        projection in q_x().
    discount_curve : RiskFreeRateCurve
        Pre-MA risk-free rate curve for BEL discounting.  Replaced by the
        post-MA adjusted curve in Step 20 — field name and type unchanged.
        Use RiskFreeRateCurve.flat(rate) for tests and simple scenarios.
    inflation_rate : float
        Best estimate annual CPI rate (e.g. 0.025 = 2.5% p.a.).
        Used for LPI pension increases and deferred revaluation.
    rpi_rate : float
        Best estimate annual RPI rate.  Used where scheme rules specify
        RPI revaluation rather than CPI.
    tv_rate : float
        Annual transfer value election rate for deferred members.
        Set to 0.0 for full-buyout schemes.
    ill_health_rates : np.ndarray, shape (TABLE_LENGTH,)
        Annual ill-health retirement probability by integer age (index = age - 16).
        Default: 0.001 p.a. for ages 40–60 rising to 0.005 for 60–NRA.
    retirement : RetirementRates
        Retirement probability schedule for deferred members.
    dependant : DependantAssumptions
        Dependant proportion and age difference assumptions.
    expense_pa : float
        Annual expense per policy in £ (e.g. 150.0).
    """
    mortality:        MortalityBasis
    valuation_year:   int
    discount_curve:   RiskFreeRateCurve
    inflation_rate:   float
    rpi_rate:         float
    tv_rate:          float
    ill_health_rates: np.ndarray
    retirement:       RetirementRates   = field(default_factory=RetirementRates)
    dependant:        DependantAssumptions = field(default_factory=DependantAssumptions)
    expense_pa:       float             = 150.0

    def __post_init__(self) -> None:
        if self.valuation_year < 2000:
            raise ValueError(f"valuation_year looks wrong: {self.valuation_year}")
        if not isinstance(self.discount_curve, RiskFreeRateCurve):
            raise TypeError("discount_curve must be a RiskFreeRateCurve")
        if not (0.0 <= self.inflation_rate <= 0.20):
            raise ValueError(f"inflation_rate must be in [0, 0.20]; got {self.inflation_rate}")
        if not (0.0 <= self.tv_rate <= 1.0):
            raise ValueError(f"tv_rate must be in [0, 1]; got {self.tv_rate}")
        if self.expense_pa < 0.0:
            raise ValueError(f"expense_pa must be non-negative; got {self.expense_pa}")
        from engine.liability.bpa.mortality import TABLE_LENGTH
        if len(self.ill_health_rates) != TABLE_LENGTH:
            raise ValueError(
                f"ill_health_rates must have length {TABLE_LENGTH}; "
                f"got {len(self.ill_health_rates)}"
            )

    @classmethod
    def default(cls, mortality: MortalityBasis) -> BPAAssumptions:
        """
        Construct a sensible default assumption set for testing.

        Uses the supplied MortalityBasis (caller provides flat synthetic
        basis in tests; BPADataLoader provides S3/CMI basis in production).
        """
        from engine.liability.bpa.mortality import TABLE_LENGTH, MIN_TABLE_AGE

        ill_health = np.zeros(TABLE_LENGTH, dtype=float)
        for i in range(TABLE_LENGTH):
            age = i + MIN_TABLE_AGE
            if 40 <= age < 60:
                ill_health[i] = 0.001
            elif 60 <= age < 70:
                ill_health[i] = 0.003
            elif age >= 70:
                ill_health[i] = 0.005

        return cls(
            mortality=mortality,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.025,
            rpi_rate=0.03,
            tv_rate=0.02,
            ill_health_rates=ill_health,
        )

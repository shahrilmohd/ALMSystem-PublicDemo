"""
BPA mortality basis — S3 tables, CMI improvement model, survival probabilities.

Design (DECISIONS.md §19)
--------------------------
Mortality is a two-component structure:

  q(x, base_year + t) = q_S3(x, base_year) × AE_ratio(sex)
                        × Π[1 − f(x, s, LTR)]  for s = 1 to t

where f(x, s, LTR) is the CMI annual improvement factor.

The improvement model projects q_x forward in CALENDAR TIME; survival across
projection periods is a separate product and must not be conflated.

Enhanced / impaired lives use an age-rating shift on the same S3 ultimate
table (not a separate select table).  rating_years >= 0.

Table data is injected by BPADataLoader; the engine never reads files directly.

NOTE: Computation logic is not included in this public demo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TABLE_AGE: int = 16
MAX_TABLE_AGE: int = 120
TABLE_LENGTH:  int = MAX_TABLE_AGE - MIN_TABLE_AGE + 1   # 105 entries


# ---------------------------------------------------------------------------
# MortalityBasis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MortalityBasis:
    """
    Complete mortality basis for BPA projections.

    Holds the S3 base tables, CMI improvement parameters, and A/E ratios.
    All array parameters are indexed by age offset from MIN_TABLE_AGE (16).
    Table data is injected by BPADataLoader from CSV files.

    Parameters
    ----------
    base_table_male : np.ndarray, shape (TABLE_LENGTH,)
    base_table_female : np.ndarray, shape (TABLE_LENGTH,)
    initial_improvement_male : np.ndarray, shape (TABLE_LENGTH,)
    initial_improvement_female : np.ndarray, shape (TABLE_LENGTH,)
    base_year : int
    ltr : float  Long-term annual improvement rate.
    convergence_period : int  Years to converge from initial rate to LTR.
    ae_ratio_male : float
    ae_ratio_female : float
    """

    base_table_male:            np.ndarray
    base_table_female:          np.ndarray
    initial_improvement_male:   np.ndarray
    initial_improvement_female: np.ndarray
    base_year:                  int   = 2023
    ltr:                        float = 0.01
    convergence_period:         int   = 20
    ae_ratio_male:              float = 1.0
    ae_ratio_female:            float = 1.0

    def __post_init__(self) -> None:
        for name, arr in (
            ("base_table_male",            self.base_table_male),
            ("base_table_female",          self.base_table_female),
            ("initial_improvement_male",   self.initial_improvement_male),
            ("initial_improvement_female", self.initial_improvement_female),
        ):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"MortalityBasis.{name} must be a numpy array")
            if len(arr) != TABLE_LENGTH:
                raise ValueError(
                    f"MortalityBasis.{name} must have length {TABLE_LENGTH} "
                    f"(ages {MIN_TABLE_AGE}–{MAX_TABLE_AGE}); got {len(arr)}"
                )
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(
                    f"MortalityBasis.{name} values must be in [0, 1]"
                )
        if not (0.0 <= self.ltr <= 0.10):
            raise ValueError(f"ltr must be in [0, 0.10]; got {self.ltr}")
        if self.convergence_period < 1:
            raise ValueError(f"convergence_period must be >= 1; got {self.convergence_period}")
        if self.ae_ratio_male <= 0.0 or self.ae_ratio_female <= 0.0:
            raise ValueError("ae_ratio values must be positive")


# ---------------------------------------------------------------------------
# Public API — computation logic not included in public demo
# ---------------------------------------------------------------------------

def improvement_factor(
    effective_age: float,
    sex: str,
    calendar_year: int,
    basis: MortalityBasis,
) -> float:
    """
    CMI annual improvement factor f(x, s, LTR) at the given age and calendar year.

    Projects base mortality rates forward using CMI-style convergence from
    initial age-specific improvement rates toward the long-term rate (LTR).

    NOTE: Implementation not included in this public demo.
    """
    raise NotImplementedError(
        "BPA mortality computation is not included in the public demo."
    )


def q_x(
    effective_age: float,
    sex: str,
    calendar_year: int,
    basis: MortalityBasis,
) -> float:
    """
    Projected annual mortality rate at effective_age in calendar_year.

    Applies the CMI improvement model to project the S3 base table forward
    in calendar time.  For enhanced lives pass effective_age = actual_age +
    rating_years.

    NOTE: Implementation not included in this public demo.
    """
    raise NotImplementedError(
        "BPA mortality computation is not included in the public demo."
    )


def survival_probs_variable_dt(
    effective_age: float,
    sex: str,
    dt_array: np.ndarray,
    start_calendar_year: int,
    basis: MortalityBasis,
) -> np.ndarray:
    """
    Cumulative survival probabilities across variable-length projection periods.

    Supports hybrid calendars (e.g. monthly then annual).
    Returns ndarray of shape (n_periods + 1,); result[0] = 1.0.

    NOTE: Implementation not included in this public demo.
    """
    raise NotImplementedError(
        "BPA mortality computation is not included in the public demo."
    )

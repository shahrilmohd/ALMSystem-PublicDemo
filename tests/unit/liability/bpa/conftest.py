"""
Shared fixtures for BPA liability unit tests.

Synthetic MortalityBasis
------------------------
All tests use a flat synthetic basis rather than the real S3/CMI tables.
This makes tests deterministic, independent of CSV files, and easy to
reason about: q_x = 0.02 for all ages, improvement = 0.02 for all ages.

Flat basis properties that tests rely on:
- q_x at base_year  : 0.02 for all ages and both sexes
- improvement factor: starts at 0.02, converges to LTR=0.01 over 20 years
- AE ratios         : 1.0 (no adjustment)
"""
import numpy as np
import pytest

from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH


@pytest.fixture
def flat_basis() -> MortalityBasis:
    """
    Flat synthetic MortalityBasis: q_x = 0.02, initial improvement = 0.02,
    LTR = 0.01, convergence_period = 20, base_year = 2023.
    """
    flat_qx = np.full(TABLE_LENGTH, 0.02, dtype=float)
    flat_rf = np.full(TABLE_LENGTH, 0.02, dtype=float)
    return MortalityBasis(
        base_table_male=flat_qx.copy(),
        base_table_female=flat_qx.copy(),
        initial_improvement_male=flat_rf.copy(),
        initial_improvement_female=flat_rf.copy(),
        base_year=2023,
        ltr=0.01,
        convergence_period=20,
        ae_ratio_male=1.0,
        ae_ratio_female=1.0,
    )


@pytest.fixture
def zero_improvement_basis() -> MortalityBasis:
    """
    Basis with zero improvement (LTR=0, initial=0): q_x never changes.
    Useful for tests that want to isolate the survival product from
    the improvement projection.
    """
    flat_qx = np.full(TABLE_LENGTH, 0.02, dtype=float)
    zero_rf = np.zeros(TABLE_LENGTH, dtype=float)
    return MortalityBasis(
        base_table_male=flat_qx.copy(),
        base_table_female=flat_qx.copy(),
        initial_improvement_male=zero_rf.copy(),
        initial_improvement_female=zero_rf.copy(),
        base_year=2023,
        ltr=0.0,
        convergence_period=20,
    )

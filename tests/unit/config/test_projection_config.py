"""
Unit tests for ProjectionConfig.

Rules under test
----------------
1. Valid construction: all required fields accepted, defaults applied correctly.
2. projection_term_years must be > 0.
3. decision_timestep period must be >= projection_timestep period.
   Bad combinations (decision finer than projection):
     - annual projection  + monthly decisions  → reject
     - annual projection  + quarterly decisions → reject
     - quarterly projection + monthly decisions  → reject
   Valid combinations (decision same or coarser):
     - monthly projection + monthly decisions   → accept
     - monthly projection + quarterly decisions → accept
     - monthly projection + annual decisions    → accept
     - quarterly projection + quarterly decisions → accept
     - quarterly projection + annual decisions   → accept
     - annual projection  + annual decisions    → accept
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from engine.config.projection_config import (
    Currency,
    DecisionTimestep,
    ProjectionConfig,
    ProjectionTimestep,
)


# ---------------------------------------------------------------------------
# Happy path — construction and defaults
# ---------------------------------------------------------------------------

class TestProjectionConfigDefaults:
    def test_required_fields_only_applies_defaults(self):
        cfg = ProjectionConfig(
            valuation_date=date(2025, 12, 31),
            projection_term_years=30,
        )
        assert cfg.projection_timestep == ProjectionTimestep.MONTHLY
        assert cfg.decision_timestep == DecisionTimestep.ANNUAL
        assert cfg.currency == Currency.GBP

    def test_explicit_values_override_defaults(self):
        cfg = ProjectionConfig(
            valuation_date=date(2025, 12, 31),
            projection_term_years=10,
            projection_timestep="quarterly",
            decision_timestep="annual",
            currency="USD",
        )
        assert cfg.projection_timestep == ProjectionTimestep.QUARTERLY
        assert cfg.decision_timestep == DecisionTimestep.ANNUAL
        assert cfg.currency == Currency.USD

    def test_valuation_date_stored_as_date(self):
        cfg = ProjectionConfig(
            valuation_date="2025-12-31",
            projection_term_years=1,
        )
        assert cfg.valuation_date == date(2025, 12, 31)

    def test_projection_term_years_stored(self):
        cfg = ProjectionConfig(
            valuation_date=date(2025, 12, 31),
            projection_term_years=100,
        )
        assert cfg.projection_term_years == 100


# ---------------------------------------------------------------------------
# Rejection — projection_term_years
# ---------------------------------------------------------------------------

class TestProjectionTermValidation:
    def test_term_zero_rejected(self):
        with pytest.raises(ValidationError):
            ProjectionConfig(
                valuation_date=date(2025, 12, 31),
                projection_term_years=0,
            )

    def test_term_negative_rejected(self):
        with pytest.raises(ValidationError):
            ProjectionConfig(
                valuation_date=date(2025, 12, 31),
                projection_term_years=-5,
            )

    def test_term_one_accepted(self):
        cfg = ProjectionConfig(
            valuation_date=date(2025, 12, 31),
            projection_term_years=1,
        )
        assert cfg.projection_term_years == 1


# ---------------------------------------------------------------------------
# Rejection — decision_timestep finer than projection_timestep
# ---------------------------------------------------------------------------

class TestDecisionTimestepOrdering:
    """
    decision_timestep period (months) must be >= projection_timestep period.
    If decision is finer than projection, it is undefined behaviour → reject.
    """

    @pytest.mark.parametrize("projection,decision", [
        ("annual",    "monthly"),    # 12 months projection, 1 month decision → reject
        ("annual",    "quarterly"),  # 12 months projection, 3 month decision → reject
        ("quarterly", "monthly"),    # 3 months projection, 1 month decision → reject
    ])
    def test_decision_finer_than_projection_rejected(self, projection, decision):
        with pytest.raises(ValidationError) as exc_info:
            ProjectionConfig(
                valuation_date=date(2025, 12, 31),
                projection_term_years=30,
                projection_timestep=projection,
                decision_timestep=decision,
            )
        assert "decision_timestep" in str(exc_info.value)

    @pytest.mark.parametrize("projection,decision", [
        ("monthly",   "monthly"),    # equal → valid
        ("monthly",   "quarterly"),  # coarser → valid
        ("monthly",   "annual"),     # coarser → valid
        ("quarterly", "quarterly"),  # equal → valid
        ("quarterly", "annual"),     # coarser → valid
        ("annual",    "annual"),     # equal → valid
    ])
    def test_valid_timestep_combinations_accepted(self, projection, decision):
        cfg = ProjectionConfig(
            valuation_date=date(2025, 12, 31),
            projection_term_years=30,
            projection_timestep=projection,
            decision_timestep=decision,
        )
        assert cfg.projection_timestep.value == projection
        assert cfg.decision_timestep.value == decision

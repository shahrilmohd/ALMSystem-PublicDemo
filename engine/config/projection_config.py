"""
Projection time-dimension configuration for the ALM model.

Extracted to its own module so run_config.py, fund_config.py, and any
future config modules can import shared time-related enums and
ProjectionConfig without creating circular dependencies.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Time-related enumerations
# ---------------------------------------------------------------------------

class ProjectionTimestep(str, Enum):
    """
    Granularity of the liability cash flow projection.
    Asset valuation and strategic decisions always occur at decision_timestep
    frequency regardless of this setting.
    """
    MONTHLY   = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL    = "annual"


class DecisionTimestep(str, Enum):
    """
    Granularity of strategic decisions (asset rebalancing, bonus declarations).
    Independent of projection_timestep. Must be >= projection_timestep in duration
    (e.g. you can have monthly CFs with annual decisions, but not the reverse).
    """
    MONTHLY   = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL    = "annual"


class Currency(str, Enum):
    """
    Currency of all monetary values in the run.
    Explicit field prevents hardcoding and makes the value visible to the AI
    layer when it reads the config schema.
    """
    GBP = "GBP"
    USD = "USD"


# Rank used by the timestep ordering validator.
# Maps timestep value → number of months in one period.
_PERIOD_MONTHS: dict[str, int] = {
    "monthly":   1,
    "quarterly": 3,
    "annual":    12,
}


# ---------------------------------------------------------------------------
# ProjectionConfig
# ---------------------------------------------------------------------------

class ProjectionConfig(BaseModel):
    """
    Controls the time dimension of the projection.

    valuation_date:
        The start date of the projection (t=0).
        All cash flows are projected forward from this date.

    projection_term_years:
        Maximum number of years to project. Policies that lapse or mature
        before this term end naturally — the model does not force them to run
        to the full term.

    projection_timestep:
        Granularity of liability cash flow calculations.
        Standard: MONTHLY (monthly CFs).

    decision_timestep:
        Frequency of asset rebalancing and bonus rate decisions.
        Standard: ANNUAL (annual investment and crediting decisions).
        Must be >= projection_timestep in duration. For example, annual
        decisions with monthly CFs is valid; monthly decisions with annual
        CFs is not.

    currency:
        Reporting currency. Single value for the entire run.
    """
    valuation_date:        date               = Field(
        ...,
        description="Projection start date (t=0). Format: YYYY-MM-DD."
    )
    projection_term_years: int                = Field(
        ...,
        gt=0,
        description="Maximum projection length in years. Must be > 0."
    )
    projection_timestep:   ProjectionTimestep = Field(
        default=ProjectionTimestep.MONTHLY,
        description="Cash flow calculation frequency. Standard: monthly."
    )
    decision_timestep:     DecisionTimestep   = Field(
        default=DecisionTimestep.ANNUAL,
        description="Asset rebalancing and bonus crediting frequency."
    )
    currency:              Currency           = Field(
        default=Currency.GBP,
        description="Reporting currency for all monetary outputs."
    )

    @model_validator(mode="after")
    def decision_ge_projection(self) -> ProjectionConfig:
        """
        Enforce that decision_timestep period >= projection_timestep period.

        It is valid to project cash flows monthly but only make investment and
        crediting decisions annually. The reverse — making decisions more
        frequently than cash flows are projected — is undefined behaviour.
        """
        proj_months = _PERIOD_MONTHS[self.projection_timestep.value]
        dec_months  = _PERIOD_MONTHS[self.decision_timestep.value]
        if dec_months < proj_months:
            raise ValueError(
                f"decision_timestep ('{self.decision_timestep.value}', "
                f"{dec_months} month(s)) must be >= "
                f"projection_timestep ('{self.projection_timestep.value}', "
                f"{proj_months} month(s)) in duration. "
                "You cannot make decisions more frequently than cash flows are projected."
            )
        return self

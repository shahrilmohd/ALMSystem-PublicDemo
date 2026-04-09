"""
Pydantic schemas for the /results endpoints.

ResultRow         — one timestep of output; mirrors a result_records DB row.
ResultsResponse   — full result set returned by GET /results/{run_id}.
ResultsSummaryResponse — aggregated summary for GET /results/{run_id}/summary.

Asset fields (total_market_value, eir_income, etc.) are Optional[float].
They are None for LIABILITY_ONLY runs which have no asset model.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ResultRow(BaseModel):
    """
    One row of model output: a single (scenario_id, timestep, cohort_id) combination.

    Liability fields are always populated.
    Asset fields are None for LIABILITY_ONLY runs.
    cohort_id is None for all non-BPA runs.
    """
    # Index
    run_id:      str            = Field(..., description="Run identifier.")
    scenario_id: int            = Field(..., description="0 for deterministic; 1..N for stochastic.")
    timestep:    int            = Field(..., description="0-based projection month index.")
    cohort_id:   Optional[str] = Field(None, description="BPA cohort identifier; None for non-BPA runs.")

    # Liability cash flows
    premiums:            float = Field(..., description="Premium income this period.")
    death_claims:        float = Field(..., description="Death benefit payments this period.")
    surrender_payments:  float = Field(..., description="Surrender value payments this period.")
    maturity_payments:   float = Field(..., description="Maturity benefit payments this period.")
    expenses:            float = Field(..., description="Expense outgo this period.")
    net_outgo:           float = Field(..., description="Net cash outflow from the fund this period.")

    # Decrements
    in_force_start: float = Field(..., description="Number of policies in force at start of period.")
    deaths:         float = Field(..., description="Deaths decrement this period.")
    lapses:         float = Field(..., description="Lapses decrement this period.")
    maturities:     float = Field(..., description="Maturities decrement this period.")
    in_force_end:   float = Field(..., description="Number of policies in force at end of period.")

    # Valuation
    bel:     float = Field(..., description="Best Estimate Liability at end of period.")
    reserve: float = Field(..., description="Statutory reserve at end of period.")

    # Asset fields — None for LIABILITY_ONLY runs
    total_market_value: Optional[float] = Field(None, description="Total portfolio market value.")
    total_book_value:   Optional[float] = Field(None, description="Total portfolio book value.")
    cash_balance:       Optional[float] = Field(None, description="Fund cash account balance.")
    eir_income:         Optional[float] = Field(None, description="EIR amortisation income (AC/FVOCI bonds).")
    coupon_income:      Optional[float] = Field(None, description="Coupon cash received (FVTPL bonds).")
    dividend_income:    Optional[float] = Field(None, description="Equity dividend income.")
    unrealised_gl:      Optional[float] = Field(None, description="Unrealised gain/loss (FVTPL to P&L).")
    realised_gl:        Optional[float] = Field(None, description="Realised gain/loss on sales/maturities.")
    oci_reserve:        Optional[float] = Field(None, description="Cumulative OCI reserve (FVOCI bonds).")
    mv_ac:              Optional[float] = Field(None, description="Market value of AC-basis assets.")
    mv_fvtpl:           Optional[float] = Field(None, description="Market value of FVTPL-basis assets.")
    mv_fvoci:           Optional[float] = Field(None, description="Market value of FVOCI-basis assets.")

    model_config = {"from_attributes": True}


class ResultsResponse(BaseModel):
    """
    Returned by GET /results/{run_id}.

    rows:
        All result rows for the run, ordered by (scenario_id, cohort_id, timestep).
        For large stochastic runs use GET /results/{run_id}?format=csv to download
        as a file rather than loading everything into memory.
    """
    run_id:     str             = Field(..., description="Run identifier.")
    total_rows: int             = Field(..., description="Total number of result rows.")
    rows:       list[ResultRow] = Field(..., description="Result rows ordered by (scenario_id, cohort_id, timestep).")


class ResultsSummaryResponse(BaseModel):
    """
    Returned by GET /results/{run_id}/summary.

    A lightweight aggregated view — useful for the frontend dashboard without
    transferring the full result set.
    """
    run_id:                    str            = Field(..., description="Run identifier.")
    n_result_rows:             int            = Field(..., description="Total stored result rows.")
    n_scenarios:               int            = Field(..., description="Number of distinct scenarios in results.")
    n_timesteps:               int            = Field(..., description="Number of distinct timesteps in results.")
    final_bel:                 Optional[float] = Field(None, description="BEL at the last timestep (scenario 0, or mean across scenarios).")
    final_total_market_value:  Optional[float] = Field(None, description="Total market value at last timestep (scenario 0, or mean).")

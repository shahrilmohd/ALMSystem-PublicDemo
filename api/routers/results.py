"""
/results router — reading projection outputs.

Endpoints
---------
GET /results/{run_id}
    Return all result rows for a completed run as JSON, ordered by
    (scenario_id, cohort_id, timestep).
    Add ?format=csv to receive a CSV file download instead.
    Add ?scenario_id=N to filter to a single scenario.

GET /results/{run_id}/summary
    Return a lightweight aggregated summary: row counts, scenario count,
    timestep count, final BEL, final total market value.

Both endpoints return 404 if the run_id does not exist in the DB.
Both endpoints return an empty result set (not 404) if the run exists
but has no stored results yet (e.g. status is still PENDING or RUNNING).
"""
from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.result_schema import ResultRow, ResultsSummaryResponse, ResultsResponse
from storage.result_repository import ResultRepository
from storage.run_repository import RunRepository

router = APIRouter()


@router.get("/{run_id}/summary", response_model=ResultsSummaryResponse)
def get_results_summary(
    run_id: str,
    db: Session = Depends(get_db),
) -> ResultsSummaryResponse:
    """
    Return a lightweight summary for a run.

    Returns 404 if the run_id does not exist.
    Returns zeroed counts with None for numeric fields if no results are
    stored yet (run may still be PENDING or RUNNING).
    """
    _assert_run_exists(run_id, db)

    result_repo = ResultRepository(db)
    df = result_repo.get_dataframe(run_id)

    if df.empty:
        return ResultsSummaryResponse(
            run_id=run_id,
            n_result_rows=0,
            n_scenarios=0,
            n_timesteps=0,
            final_bel=None,
            final_total_market_value=None,
        )

    n_scenarios = int(df["scenario_id"].nunique())
    n_timesteps = int(df["timestep"].nunique())
    final_ts    = int(df["timestep"].max())

    # For deterministic runs (scenario_id=0) use that scenario.
    # For stochastic runs use the mean across all scenarios at the final timestep.
    final_rows = df[df["timestep"] == final_ts]
    final_bel  = float(final_rows["bel"].mean()) if not final_rows.empty else None

    tmv_col = final_rows["total_market_value"].dropna()
    final_tmv = float(tmv_col.mean()) if not tmv_col.empty else None

    return ResultsSummaryResponse(
        run_id=run_id,
        n_result_rows=len(df),
        n_scenarios=n_scenarios,
        n_timesteps=n_timesteps,
        final_bel=final_bel,
        final_total_market_value=final_tmv,
    )


@router.get("/{run_id}", response_model=ResultsResponse)
def get_results(
    run_id: str,
    scenario_id: Optional[int] = Query(default=None, description="Filter to a single scenario. Omit for all scenarios."),
    format: str = Query(default="json", description="Response format: 'json' (default) or 'csv'."),
    db: Session = Depends(get_db),
):
    """
    Return all result rows for a run.

    By default returns JSON.  Use ?format=csv to download a CSV file —
    recommended for large stochastic runs (120,000+ rows) to avoid
    loading the entire result set into browser memory.

    Returns 404 if the run_id does not exist.
    Returns an empty result set if the run exists but has no stored results.
    """
    _assert_run_exists(run_id, db)

    result_repo = ResultRepository(db)

    if scenario_id is not None:
        df = result_repo.get_scenario(run_id, scenario_id)
    else:
        df = result_repo.get_dataframe(run_id)

    if format.lower() == "csv":
        return _csv_response(run_id, df)

    rows = [ResultRow(**row) for row in df.to_dict(orient="records")] if not df.empty else []
    return ResultsResponse(run_id=run_id, total_rows=len(rows), rows=rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assert_run_exists(run_id: str, db: Session) -> None:
    """Raise 404 if no RunRecord exists for this run_id."""
    run_repo = RunRepository(db)
    if not run_repo.exists(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id!r}")


def _csv_response(run_id: str, df) -> StreamingResponse:
    """Return a StreamingResponse with the DataFrame serialised as CSV."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    filename = f"results_{run_id}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

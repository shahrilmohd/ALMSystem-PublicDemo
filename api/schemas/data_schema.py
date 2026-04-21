"""
Pydantic schemas for data query endpoints.

ModelPointsResponse         — returned by GET /runs/{run_id}/model_points
AssumptionTableResponse     — returned by GET /assumptions/{table_name}
AssumptionTablesListResponse — returned by GET /assumptions
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

_MAX_ROWS = 500  # hard cap applied by the router before returning


class ModelPointsResponse(BaseModel):
    """
    Model point rows for one run, optionally filtered by population type.

    data contains at most _MAX_ROWS rows; truncated is True when the
    underlying file had more rows and the response was clipped.
    """
    run_id:          str                   = Field(..., description="UUID of the run.")
    population_type: Optional[str]         = Field(
        None,
        description="Population filter applied ('in_payment', 'deferred', "
                    "'dependant', 'enhanced'), or None if all rows returned.",
    )
    row_count:       int                   = Field(..., description="Number of rows in this response.")
    truncated:       bool                  = Field(
        ...,
        description=f"True when the source had more than {_MAX_ROWS} rows and the "
                    "response was clipped to that limit.",
    )
    columns:         list[str]             = Field(..., description="Column names present in each data row.")
    data:            list[dict[str, Any]]  = Field(..., description="Model point rows as list of dicts.")


class AssumptionTableResponse(BaseModel):
    """
    Rows from a single named assumption table at a given valuation date.

    data contains at most _MAX_ROWS rows; truncated is True if clipped.
    """
    table_name:     str                   = Field(..., description="Logical table name (filename without extension).")
    valuation_date: str                   = Field(..., description="Valuation date used to select the snapshot.")
    row_count:      int                   = Field(..., description="Number of rows in this response.")
    truncated:      bool                  = Field(
        ...,
        description=f"True when the source had more than {_MAX_ROWS} rows.",
    )
    columns:        list[str]             = Field(..., description="Column names present in each data row.")
    data:           list[dict[str, Any]]  = Field(..., description="Table rows as list of dicts.")


class AssumptionTablesListResponse(BaseModel):
    """
    Names of all assumption tables available for a given valuation date.
    """
    valuation_date: str        = Field(..., description="Valuation date for which tables are listed.")
    tables:         list[str]  = Field(
        ...,
        description="Logical table names (CSV filenames without extension) found in the "
                    "snapshot directory for this valuation date.",
    )

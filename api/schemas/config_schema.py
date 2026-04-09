"""
Pydantic schemas for the /config endpoints.

Request schema:   ValidateConfigRequest  — body for POST /config/validate.
Response schemas: ValidateConfigResponse — validation result with any errors.
                  ConfigTemplateResponse — JSON schema for GET /config/template.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidateConfigRequest(BaseModel):
    """Body for POST /config/validate."""
    config_json: str = Field(
        ...,
        description="RunConfig JSON string to validate.",
    )


class ValidateConfigResponse(BaseModel):
    """
    Returned by POST /config/validate.

    is_valid:
        True if the config_json parses into a valid RunConfig with no errors.
        False if the JSON is malformed or the RunConfig structure is invalid.

    errors:
        Empty list when is_valid=True.
        One entry per validation error when is_valid=False.  Each entry is a
        human-readable string describing the problem and the field path.
    """
    is_valid: bool        = Field(..., description="True if the config is fully valid.")
    errors:   list[str]   = Field(default_factory=list, description="Validation error messages.")


class ConfigTemplateResponse(BaseModel):
    """
    Returned by GET /config/template.

    json_schema:
        The JSON Schema for RunConfig, generated from the Pydantic model.
        Clients (desktop app, AI layer) use this to:
          - Build a config form with the correct fields and types.
          - Understand which fields are required vs optional.
          - Discover all valid enum values without reading the source code.
    """
    json_schema: dict[str, Any] = Field(..., description="RunConfig JSON Schema (Pydantic model_json_schema()).")

"""
/config router — RunConfig validation and schema introspection.

Endpoints
---------
POST /config/validate
    Try to parse a config_json string as a RunConfig.
    Returns is_valid=True/False and a list of human-readable error messages.
    Use this before POST /runs to surface config problems early.

GET /config/template
    Return the RunConfig JSON Schema generated from the Pydantic model.
    Clients use this to build config forms or understand valid field values.

Architectural rule:
    This is the ONE place in the API layer that imports from engine/.
    It imports RunConfig for schema introspection and validation only —
    it never instantiates a run or touches the engine execution path.
"""
from __future__ import annotations

import json
import warnings

from fastapi import APIRouter
from pydantic import ValidationError

from api.schemas.config_schema import (
    ConfigTemplateResponse,
    ValidateConfigRequest,
    ValidateConfigResponse,
)
from engine.config.run_config import RunConfig

router = APIRouter()


@router.post("/validate", response_model=ValidateConfigResponse)
def validate_config(body: ValidateConfigRequest) -> ValidateConfigResponse:
    """
    Validate a RunConfig JSON string.

    Attempts to parse body.config_json using RunConfig.model_validate_json().
    Returns is_valid=True with an empty errors list on success.
    Returns is_valid=False with one error string per validation failure.

    Note: validation includes Pydantic field validators such as file-path
    existence checks.  A config that references non-existent paths will
    return is_valid=False with a descriptive error.  This is intentional —
    the same validation runs inside the worker at execution time.
    """
    try:
        RunConfig.model_validate_json(body.config_json)
        return ValidateConfigResponse(is_valid=True, errors=[])
    except json.JSONDecodeError as exc:
        return ValidateConfigResponse(
            is_valid=False,
            errors=[f"config_json is not valid JSON: {exc}"],
        )
    except ValidationError as exc:
        errors = [
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        return ValidateConfigResponse(is_valid=False, errors=errors)


@router.get("/template", response_model=ConfigTemplateResponse)
def get_config_template() -> ConfigTemplateResponse:
    """
    Return the RunConfig JSON Schema.

    The schema documents every field: its type, constraints, valid enum
    values, and description.  Use this to build a config form in the
    desktop app or to guide AI layer config modification.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schema = RunConfig.model_json_schema()
    return ConfigTemplateResponse(json_schema=schema)

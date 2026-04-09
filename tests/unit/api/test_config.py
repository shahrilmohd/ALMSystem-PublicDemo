"""
Unit tests for the /config router.

Coverage
--------
POST /config/validate
    - well-formed JSON but structurally invalid RunConfig → is_valid=False with errors
    - completely unparseable JSON → is_valid=False with parse error
    - missing config_json field in body → 422

GET /config/template
    - returns a dict with 'title' and 'properties' keys (JSON Schema structure)
    - schema includes RunConfig's top-level fields

Note on "valid config" tests:
    A fully valid RunConfig requires real file paths that exist on disk
    (liability data, asset data, assumption tables, fund config YAML).
    Testing a fully passing validate call is therefore an integration concern,
    not a unit concern.  These unit tests focus on the error-handling paths,
    which are the paths the validate endpoint exists to surface.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# POST /config/validate
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_invalid_json_returns_is_valid_false(self, client: TestClient):
        response = client.post(
            "/config/validate",
            json={"config_json": "not json at all {{{"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["is_valid"] is False
        assert len(body["errors"]) > 0

    def test_missing_required_fields_returns_errors(self, client: TestClient):
        # Valid JSON but missing all required RunConfig fields
        response = client.post(
            "/config/validate",
            json={"config_json": json.dumps({"some_unknown_key": 42})},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["is_valid"] is False
        assert len(body["errors"]) > 0

    def test_errors_are_human_readable_strings(self, client: TestClient):
        response = client.post(
            "/config/validate",
            json={"config_json": json.dumps({})},
        )

        body = response.json()
        assert all(isinstance(e, str) for e in body["errors"])

    def test_missing_body_field_returns_422(self, client: TestClient):
        # Sending an empty body — config_json is required
        response = client.post("/config/validate", json={})

        assert response.status_code == 422

    def test_wrong_enum_value_captured_in_errors(self, client: TestClient):
        # run_type is an enum — invalid value should surface as an error
        bad_config = json.dumps({"run_type": "NOT_A_VALID_RUN_TYPE"})
        response = client.post(
            "/config/validate",
            json={"config_json": bad_config},
        )

        body = response.json()
        assert body["is_valid"] is False
        # At least one error should mention run_type
        combined = " ".join(body["errors"]).lower()
        assert "run_type" in combined or "input should be" in combined


# ---------------------------------------------------------------------------
# GET /config/template
# ---------------------------------------------------------------------------

class TestGetConfigTemplate:
    def test_returns_200(self, client: TestClient):
        response = client.get("/config/template")

        assert response.status_code == 200

    def test_returns_json_schema_structure(self, client: TestClient):
        body = client.get("/config/template").json()

        # A Pydantic-generated JSON Schema always has 'title' and 'properties'
        assert "json_schema" in body
        schema = body["json_schema"]
        assert "title" in schema
        assert "properties" in schema

    def test_schema_contains_run_type_field(self, client: TestClient):
        schema = client.get("/config/template").json()["json_schema"]

        assert "run_type" in schema["properties"]

    def test_schema_is_deserializable_dict(self, client: TestClient):
        body = client.get("/config/template").json()

        # json_schema must be a plain dict (not a string that needs re-parsing)
        assert isinstance(body["json_schema"], dict)

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate

from backend.api import _PUBLIC_JOB_FIELDS

ROOT = Path(__file__).resolve().parents[2]


def _openapi() -> dict[str, object]:
    return yaml.safe_load((ROOT / "docs" / "openapi.yaml").read_text())


def test_job_schema_tracks_the_api_allowlist() -> None:
    schema = json.loads((ROOT / "docs" / "job.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    documented = set(schema["properties"])
    public_fields = set(_PUBLIC_JOB_FIELDS)

    assert documented == public_fields
    assert set(schema["required"]) == public_fields
    assert schema["additionalProperties"] is False
    assert schema["properties"]["completed_at"]["type"] == ["string", "null"]
    assert schema["properties"]["error"]["type"] == ["string", "null"]
    assert schema["properties"]["error_code"]["type"] == ["string", "null"]


def test_openapi_documents_actual_job_statuses_and_envelopes() -> None:
    document = _openapi()
    validate(document, base_uri=(ROOT / "docs" / "openapi.yaml").as_uri())

    create = document["paths"]["/jobs"]["post"]["responses"]
    assert "202" in create
    assert "201" not in create
    assert (
        create["202"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/JobsEnvelope"
    )

    operations = (
        ("/jobs/{job_id}", "get", "200"),
        ("/jobs/{job_id}/cancel", "post", "202"),
        ("/jobs/{job_id}/partial", "delete", "200"),
    )
    for path, method, status in operations:
        response = document["paths"][path][method]["responses"][status]
        assert (
            response["content"]["application/json"]["schema"]["$ref"]
            == "#/components/schemas/JobEnvelope"
        )


def test_openapi_documents_epoch_csrf_and_mixed_capabilities() -> None:
    schemas = _openapi()["components"]["schemas"]

    assert schemas["Session"]["properties"]["csrf_expires_at"]["type"] == "integer"
    assert schemas["Session"]["properties"]["capabilities"]["$ref"].endswith(
        "/Capabilities"
    )
    assert schemas["Capabilities"]["properties"]["providers"]["type"] == "array"
    assert schemas["Capabilities"]["properties"]["max_models_per_scan"]["const"] == 50
    assert schemas["InspectedModel"]["properties"]["download_token"]["type"] == [
        "string",
        "null",
    ]
    assert set(schemas["Health"]["required"]) == {
        "api_version",
        "extension_version",
        "status",
        "state",
    }

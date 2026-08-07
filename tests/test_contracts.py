"""L7 — Contracts (ARCHITECTURE-PLAN.md's L7 section): real JSON Schema
enforcement on top of healing.py's syntax-only repair. Covers the
`pipeline/contracts.py` module directly, plus the ConfigError path when the
optional `jsonschema` package isn't installed (this project's own
environment doesn't have it installed, which is exactly what makes that
specific test meaningful here rather than a hypothetical).
"""

import pytest

from modelrouter.core.errors import ConfigError
from modelrouter.pipeline.contracts import ContractViolation, corrective_prompt, validate_contract

try:
    import jsonschema  # noqa: F401
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

jsonschema_only = pytest.mark.skipif(not HAS_JSONSCHEMA, reason="needs the optional 'jsonschema' package")

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    "required": ["name", "age"],
}


@pytest.mark.skipif(HAS_JSONSCHEMA, reason="only meaningful when jsonschema is NOT installed")
def test_validate_contract_raises_config_error_without_jsonschema_installed():
    with pytest.raises(ConfigError, match="jsonschema"):
        validate_contract({"name": "a", "age": 1}, SCHEMA)


@jsonschema_only
def test_validate_contract_ok_for_a_valid_instance():
    result = validate_contract({"name": "Eggs", "age": 3}, SCHEMA)
    assert result.ok is True
    assert result.violations == ()


@jsonschema_only
def test_validate_contract_collects_every_violation_not_just_the_first():
    result = validate_contract({"age": "not a number"}, SCHEMA)   # missing name AND wrong type
    assert result.ok is False
    assert len(result.violations) == 2
    validators = {v.validator for v in result.violations}
    assert "required" in validators
    assert "type" in validators


@jsonschema_only
def test_validate_contract_reports_the_real_json_path():
    result = validate_contract({"name": "x", "age": "not a number"}, SCHEMA)
    assert result.ok is False
    assert result.violations[0].json_path == "$.age"


def test_corrective_prompt_names_each_violation():
    violations = (
        ContractViolation(message="'name' is a required property", json_path="$", validator="required"),
        ContractViolation(message="'not a number' is not of type 'number'", json_path="$.age", validator="type"),
    )
    prompt = corrective_prompt(violations)
    assert "$.age" in prompt
    assert "'name' is a required property" in prompt


def test_contract_violation_as_dict_is_json_serializable_plain_dict():
    violation = ContractViolation(message="msg", json_path="$.x", validator="type")
    d = violation.as_dict()
    assert d == {"message": "msg", "json_path": "$.x", "validator": "type"}
    import json
    json.dumps(d)   # must not raise

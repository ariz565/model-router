"""pipeline/tool_contracts.py — validation and repair of `tool_calls`, the
structured-output problem for tool calling that `contracts.py` doesn't cover.

The failures exercised here are the ones weak/small models actually produce:
double-encoded arguments, syntactically broken JSON, hallucinated tool names,
and missing required parameters. Schema-validation tests are gated on the
optional `jsonschema` package; the name and parse checks are not, because those
must work in a deployment that hasn't installed it.
"""

from __future__ import annotations

import json

import pytest

from modelrouter.pipeline.tool_contracts import (
    ISSUE_MALFORMED_ARGUMENTS,
    ISSUE_MISSING_NAME,
    ISSUE_SCHEMA_VIOLATION,
    ISSUE_UNKNOWN_TOOL,
    corrective_tool_prompt,
    extract_tool_schemas,
    repair_tool_calls,
    validate_tool_calls,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            "required": ["city"],
        },
    },
}
NO_ARG_TOOL = {"type": "function", "function": {"name": "ping"}}
TOOLS = [WEATHER_TOOL, NO_ARG_TOOL]


def _call(name: str, arguments, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


# ── extract_tool_schemas ──────────────────────────────────────────────────

def test_extract_maps_names_to_parameter_schemas():
    schemas = extract_tool_schemas(TOOLS)
    assert set(schemas) == {"get_weather", "ping"}
    assert schemas["get_weather"]["required"] == ["city"]


def test_a_tool_with_no_parameters_gets_an_empty_permissive_schema():
    """A zero-argument function is legal; treating it as invalid would reject
    correct calls."""
    assert extract_tool_schemas([NO_ARG_TOOL])["ping"] == {}


def test_extract_ignores_malformed_tool_entries_without_crashing():
    schemas = extract_tool_schemas([
        {"type": "function"},                      # no function block
        {"function": {"parameters": {}}},          # no name
        {"function": {"name": ""}},                # empty name
        "not-a-dict",
        WEATHER_TOOL,
    ])
    assert set(schemas) == {"get_weather"}


def test_extract_handles_none_and_empty():
    assert extract_tool_schemas(None) == {}
    assert extract_tool_schemas([]) == {}


# ── validate_tool_calls: names and parsing (no jsonschema needed) ──────────

def test_no_tool_calls_is_valid():
    assert validate_tool_calls(None, TOOLS).ok is True
    assert validate_tool_calls([], TOOLS).ok is True


def test_a_well_formed_call_validates():
    """Gated: `get_weather` has a non-empty schema, so the full happy path
    reaches `validate_contract`, which requires the optional `jsonschema`
    package. Deployments without it use `validate_schemas=False` (covered
    below) — the schema step is never silently skipped."""
    pytest.importorskip("jsonschema")
    result = validate_tool_calls([_call("get_weather", '{"city":"Paris"}')], TOOLS)
    assert result.ok is True
    assert result.issues == ()


def test_a_call_to_a_schemaless_tool_validates_without_jsonschema():
    """A zero-parameter tool has an empty schema, so no schema validation is
    attempted and no optional dependency is touched."""
    result = validate_tool_calls([_call("ping", "{}")], TOOLS)
    assert result.ok is True


def test_a_hallucinated_tool_name_is_reported():
    """Structured output that names a function nobody offered cannot be executed;
    passing it through unremarked hands the app a call it has no handler for."""
    result = validate_tool_calls([_call("delete_everything", "{}")], TOOLS)
    assert result.ok is False
    assert result.issues[0].issue == ISSUE_UNKNOWN_TOOL
    assert "get_weather" in result.issues[0].detail   # tells the model what WAS offered


def test_a_missing_function_name_is_reported():
    result = validate_tool_calls([{"id": "c1", "function": {"arguments": "{}"}}], TOOLS)
    assert result.issues[0].issue == ISSUE_MISSING_NAME


def test_unparseable_arguments_are_reported():
    result = validate_tool_calls([_call("get_weather", "{city: Paris")], TOOLS)
    assert result.issues[0].issue == ISSUE_MALFORMED_ARGUMENTS


def test_arguments_that_are_a_json_array_are_rejected():
    result = validate_tool_calls([_call("get_weather", '["Paris"]')], TOOLS)
    assert result.issues[0].issue == ISSUE_MALFORMED_ARGUMENTS


def test_already_parsed_dict_arguments_are_accepted():
    """Some SDKs pre-parse `arguments`; rejecting that shape would break them."""
    pytest.importorskip("jsonschema")
    result = validate_tool_calls([_call("get_weather", {"city": "Paris"})], TOOLS)
    assert result.ok is True


def test_absent_arguments_are_treated_as_no_arguments():
    assert validate_tool_calls([_call("ping", None)], TOOLS).ok is True
    assert validate_tool_calls([_call("ping", "")], TOOLS).ok is True


def test_with_no_tools_declared_any_name_passes_through():
    """Nothing to check against, and rejecting everything would break
    passthrough use."""
    assert validate_tool_calls([_call("anything", "{}")], None).ok is True


def test_the_issue_index_points_at_the_offending_call():
    result = validate_tool_calls(
        [_call("ping", "{}"), _call("nope", "{}")], TOOLS, validate_schemas=False,
    )
    assert len(result.issues) == 1
    assert result.issues[0].index == 1


def test_validation_is_json_serializable_for_a_trace():
    result = validate_tool_calls([_call("nope", "{}")], TOOLS)
    json.dumps(result.as_dict())   # must not raise


# ── validate_tool_calls: schema conformance (needs jsonschema) ─────────────

def test_a_missing_required_parameter_is_a_schema_violation():
    pytest.importorskip("jsonschema")
    result = validate_tool_calls([_call("get_weather", '{"units":"c"}')], TOOLS)
    assert result.ok is False
    assert result.issues[0].issue == ISSUE_SCHEMA_VIOLATION
    assert "city" in result.issues[0].detail


def test_a_wrong_parameter_type_is_a_schema_violation():
    pytest.importorskip("jsonschema")
    result = validate_tool_calls([_call("get_weather", '{"city":123}')], TOOLS)
    assert result.issues[0].issue == ISSUE_SCHEMA_VIOLATION


def test_schema_validation_can_be_skipped_so_name_checks_still_work_without_jsonschema():
    """The most common failures must be catchable in a deployment that hasn't
    installed the optional package."""
    result = validate_tool_calls(
        [_call("get_weather", '{"units":"c"}')], TOOLS, validate_schemas=False,
    )
    assert result.ok is True    # schema not checked...
    unknown = validate_tool_calls([_call("nope", "{}")], TOOLS, validate_schemas=False)
    assert unknown.issues[0].issue == ISSUE_UNKNOWN_TOOL   # ...but names still are


# ── repair_tool_calls ─────────────────────────────────────────────────────

def _arguments_of(repair, index: int = 0):
    return json.loads(repair.repaired[index]["function"]["arguments"])


def test_double_encoded_arguments_are_unwrapped():
    """The single most common weak-model tool-calling failure: `arguments` is a
    JSON string whose CONTENT is JSON."""
    double = json.dumps('{"city":"Paris"}')      # a JSON string containing JSON
    repair = repair_tool_calls([_call("get_weather", double)], TOOLS)

    assert repair.changed is True
    assert _arguments_of(repair) == {"city": "Paris"}
    assert repair.unresolved == ()


def test_syntactically_broken_arguments_are_healed():
    """A trailing comma — one of the repairs `heal_json` genuinely performs.
    (Single-quoted keys are NOT healed by it, so this doesn't claim they are.)"""
    repair = repair_tool_calls([_call("get_weather", '{"city": "Paris",}')], TOOLS)
    assert _arguments_of(repair) == {"city": "Paris"}


def test_arguments_buried_in_prose_are_extracted():
    """Weak models frequently narrate around the JSON."""
    repair = repair_tool_calls(
        [_call("get_weather", 'Sure! {"city":"Paris"} — hope that helps.')], TOOLS,
    )
    assert _arguments_of(repair) == {"city": "Paris"}


def test_arguments_wrapped_in_a_markdown_fence_are_healed():
    fenced = '```json\n{"city":"Paris"}\n```'
    repair = repair_tool_calls([_call("get_weather", fenced)], TOOLS)
    assert _arguments_of(repair) == {"city": "Paris"}


def test_already_correct_arguments_are_normalized_but_reported_unchanged_in_meaning():
    repair = repair_tool_calls([_call("get_weather", '{"city":"Paris"}')], TOOLS)
    assert _arguments_of(repair) == {"city": "Paris"}
    assert repair.unresolved == ()


def test_a_dict_arguments_value_is_serialized_to_the_wire_shape():
    """Providers and SDKs expect a string; a repaired call must be
    indistinguishable in form from a well-formed one."""
    repair = repair_tool_calls([_call("get_weather", {"city": "Paris"})], TOOLS)
    assert isinstance(repair.repaired[0]["function"]["arguments"], str)
    assert _arguments_of(repair) == {"city": "Paris"}


def test_repair_never_invents_a_missing_required_parameter():
    """The core restraint: fabricating an argument for a function about to be
    EXECUTED is worse than reporting the failure."""
    pytest.importorskip("jsonschema")
    repair = repair_tool_calls([_call("get_weather", '{"units":"c"}')], TOOLS)

    assert _arguments_of(repair) == {"units": "c"}     # untouched
    assert "city" not in _arguments_of(repair)          # nothing invented
    # The schema violation is then surfaced by validation, not silently fixed.
    assert validate_tool_calls(repair.repaired, TOOLS).ok is False


def test_repair_never_guesses_a_tool_name_and_reports_it_unresolved():
    repair = repair_tool_calls([_call("delete_everythng", "{}")], TOOLS)
    assert repair.repaired[0]["function"]["name"] == "delete_everythng"   # not "corrected"
    assert repair.unresolved[0].issue == ISSUE_UNKNOWN_TOOL


def test_irreparable_arguments_are_passed_through_not_dropped():
    """Dropping a call would silently change what the model asked for."""
    repair = repair_tool_calls([_call("get_weather", "this is not json at all")], TOOLS)
    assert len(repair.repaired) == 1
    assert repair.unresolved[0].issue == ISSUE_MALFORMED_ARGUMENTS


def test_repair_preserves_the_call_id_and_type():
    """The id is what a caller correlates its tool RESULT back to; losing it
    breaks the round trip."""
    repair = repair_tool_calls([_call("get_weather", '{"city":"Paris"}', "call_xyz")], TOOLS)
    assert repair.repaired[0]["id"] == "call_xyz"
    assert repair.repaired[0]["type"] == "function"


def test_repair_handles_multiple_calls_independently():
    double = json.dumps('{"city":"Rome"}')
    repair = repair_tool_calls([
        _call("get_weather", '{"city":"Paris"}', "c1"),
        _call("get_weather", double, "c2"),
        _call("ping", "", "c3"),
    ], TOOLS)

    assert len(repair.repaired) == 3
    assert _arguments_of(repair, 1) == {"city": "Rome"}
    assert repair.changed is True


def test_repair_of_nothing_is_a_no_op():
    repair = repair_tool_calls(None, TOOLS)
    assert repair.repaired == []
    assert repair.changed is False


def test_a_non_dict_call_is_reported_rather_than_crashing():
    repair = repair_tool_calls(["not-a-call"], TOOLS)
    assert repair.unresolved[0].issue == ISSUE_MISSING_NAME


# ── corrective_tool_prompt ────────────────────────────────────────────────

def test_the_corrective_prompt_names_the_exact_problems():
    """A model told only to "fix your tool call" is no more likely to succeed
    than it was the first time."""
    issues = validate_tool_calls([_call("nope", "{}")], TOOLS).issues
    prompt = corrective_tool_prompt(issues)

    assert "nope" in prompt
    assert "get_weather" in prompt          # what it WAS allowed to call
    assert "call #0" in prompt


def test_the_corrective_prompt_warns_against_the_double_encoding_mistake():
    issues = validate_tool_calls([_call("get_weather", "broken")], TOOLS).issues
    prompt = corrective_tool_prompt(issues)
    assert "string containing JSON" in prompt

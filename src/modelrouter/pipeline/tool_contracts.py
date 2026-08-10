"""Tool-call validation and repair — the structured-output problem for
`tool_calls`, which `contracts.py` deliberately does not cover.

**Why this is a separate concern from `response_format="json_schema"`.**
`contracts.py` validates ONE response body against ONE caller-supplied schema.
A tool-calling turn is a different shape entirely: the model emits N calls, each
naming a tool, each carrying an `arguments` string that must be valid JSON AND
must satisfy *that specific tool's* parameter schema — which came from the
caller's `tools` array, not from a `json_schema` field. Bolting this onto
`validate_contract()` would mean one function with two unrelated modes.

**Why it matters more than the JSON case.** Small and open-weight models are
markedly worse at tool calling than at prose: the failures seen in practice are
`arguments` that isn't valid JSON at all, a hallucinated tool name that was never
offered, arguments missing required parameters, and — the most common — arguments
double-encoded as a JSON *string* containing JSON. Every one of those is
detectable and most are mechanically repairable, which is exactly what makes a
weaker, cheaper model usable for tool calling at all. That is the point of this
module: it is what turns "this model is 10x cheaper but unreliable at tools" into
a measurable, bounded risk.

**Repair is conservative and never invents values.** `repair_tool_calls()` will
unwrap double-encoded JSON and fix syntax via `heal_json`, because both recover
the model's actual intent. It will NOT fill in a missing required parameter, coerce
a type, or guess a tool name — inventing an argument for a function that is about
to be *executed* is far worse than reporting the failure. Fabricating a plausible
argument to a `delete_account(user_id=…)` call is the specific outcome this
restraint exists to prevent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from modelrouter.pipeline.contracts import ContractViolation, validate_contract
from modelrouter.pipeline.healing import heal_json

__all__ = [
    "ToolCallIssue", "ToolCallValidation", "ToolCallRepair",
    "extract_tool_schemas", "validate_tool_calls", "repair_tool_calls",
    "corrective_tool_prompt",
    "ISSUE_UNKNOWN_TOOL", "ISSUE_MALFORMED_ARGUMENTS", "ISSUE_SCHEMA_VIOLATION",
    "ISSUE_MISSING_NAME",
]

ISSUE_UNKNOWN_TOOL = "unknown_tool"
ISSUE_MALFORMED_ARGUMENTS = "malformed_arguments"
ISSUE_SCHEMA_VIOLATION = "schema_violation"
ISSUE_MISSING_NAME = "missing_name"


@dataclass(frozen=True)
class ToolCallIssue:
    """`index` is the position in the `tool_calls` array, so a caller can point at
    the exact call rather than saying "one of them was wrong"."""

    index: int
    tool_name: str | None
    issue: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "index": self.index, "tool_name": self.tool_name,
            "issue": self.issue, "detail": self.detail,
        }


@dataclass(frozen=True)
class ToolCallValidation:
    ok: bool
    issues: tuple[ToolCallIssue, ...] = ()

    def as_dict(self) -> dict:
        return {"ok": self.ok, "issues": [i.as_dict() for i in self.issues]}


@dataclass(frozen=True)
class ToolCallRepair:
    """`repaired` is the corrected `tool_calls` list. `changed` reports whether
    anything was actually altered — a caller uses it to decide whether to
    re-record the response, and a trace uses it to show that a repair happened
    rather than silently presenting repaired output as what the model said."""

    repaired: list[dict]
    changed: bool
    unresolved: tuple[ToolCallIssue, ...] = ()


def extract_tool_schemas(tools: list[dict] | None) -> dict[str, dict]:
    """Maps tool name → its JSON Schema, from the caller's OpenAI-shaped `tools`
    array (`{"type": "function", "function": {"name", "parameters"}}`).

    Tolerant of a tool with no `parameters` (a zero-argument function is legal),
    which becomes an empty schema that accepts anything — NOT a validation
    failure. Being strict there would reject correct calls to valid tools."""
    schemas: dict[str, dict] = {}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        schemas[name] = parameters if isinstance(parameters, dict) else {}
    return schemas


def _parse_arguments(raw: object) -> tuple[object | None, str | None]:
    """Returns `(parsed, error)`.

    Handles the three real shapes seen from models:
      1. a JSON object string — the correct form;
      2. an already-parsed dict — some SDKs pre-parse it;
      3. a double-encoded string, i.e. a JSON string whose *content* is JSON.
         This is the single most common weak-model tool-calling failure, and
         unwrapping it recovers the model's genuine intent rather than guessing.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        # An absent arguments blob means "no arguments", which is valid for a
        # zero-parameter tool and is caught by schema validation otherwise.
        return {}, None
    if not isinstance(raw, str):
        return None, f"arguments must be a JSON object or string, got {type(raw).__name__}"

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"arguments is not valid JSON: {e}"

    if isinstance(parsed, str):
        # Double-encoded: the outer parse yielded another string. Try once more.
        try:
            inner = json.loads(parsed)
        except (json.JSONDecodeError, ValueError):
            return None, "arguments was a JSON string that did not contain a JSON object"
        return (inner, None) if isinstance(inner, dict) else (
            None, "double-encoded arguments did not contain a JSON object"
        )
    if not isinstance(parsed, dict):
        return None, f"arguments must decode to an object, got {type(parsed).__name__}"
    return parsed, None


def validate_tool_calls(
    tool_calls: list[dict] | None, tools: list[dict] | None, *, validate_schemas: bool = True,
) -> ToolCallValidation:
    """Checks each call for: a present tool name, a name that was actually
    OFFERED, parseable arguments, and (when `validate_schemas`) conformance to
    that tool's parameter schema.

    `validate_schemas=False` skips only the JSON-Schema step, so the name and
    parse checks still run without needing the optional `jsonschema` package —
    which is what lets the most common failures be caught in a deployment that
    hasn't installed it.

    An unknown tool name is reported even though the model "successfully"
    produced structured output: a call to a function that was never offered
    cannot be executed, and passing it through unremarked hands the application a
    call it has no handler for."""
    if not tool_calls:
        return ToolCallValidation(ok=True)

    schemas = extract_tool_schemas(tools)
    issues: list[ToolCallIssue] = []

    for index, call in enumerate(tool_calls):
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        name = function.get("name")

        if not isinstance(name, str) or not name:
            issues.append(ToolCallIssue(
                index=index, tool_name=None, issue=ISSUE_MISSING_NAME,
                detail="tool call has no function name",
            ))
            continue

        # Only enforce "was this offered" when the caller actually declared
        # tools. With no `tools` array there is nothing to check against, and
        # rejecting every call would break passthrough use.
        if schemas and name not in schemas:
            issues.append(ToolCallIssue(
                index=index, tool_name=name, issue=ISSUE_UNKNOWN_TOOL,
                detail=f"{name!r} was not among the offered tools: {sorted(schemas)}",
            ))
            continue

        parsed, error = _parse_arguments(function.get("arguments"))
        if error is not None:
            issues.append(ToolCallIssue(
                index=index, tool_name=name, issue=ISSUE_MALFORMED_ARGUMENTS, detail=error,
            ))
            continue

        if validate_schemas and name in schemas and schemas[name]:
            result = validate_contract(parsed, schemas[name])
            for violation in result.violations:
                issues.append(ToolCallIssue(
                    index=index, tool_name=name, issue=ISSUE_SCHEMA_VIOLATION,
                    detail=f"{violation.json_path}: {violation.message}",
                ))

    return ToolCallValidation(ok=not issues, issues=tuple(issues))


def repair_tool_calls(
    tool_calls: list[dict] | None, tools: list[dict] | None,
) -> ToolCallRepair:
    """Mechanically fixes what can be fixed without inventing anything:
    double-encoded arguments are unwrapped, and syntactically-broken arguments
    are run through `heal_json` (the same repair the JSON response path uses).

    Deliberately does NOT repair: a missing required parameter, a type mismatch,
    or an unknown tool name. Those need a value this module does not have, and
    guessing an argument for a function the application is about to EXECUTE is
    strictly worse than reporting the failure — see the module docstring.

    Arguments are re-serialized as a compact JSON string, matching the wire shape
    every provider and SDK expects, so a repaired call is indistinguishable in
    form from a well-formed one."""
    if not tool_calls:
        return ToolCallRepair(repaired=[], changed=False)

    schemas = extract_tool_schemas(tools)
    repaired: list[dict] = []
    unresolved: list[ToolCallIssue] = []
    changed = False

    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            unresolved.append(ToolCallIssue(
                index=index, tool_name=None, issue=ISSUE_MISSING_NAME,
                detail=f"tool call is not an object, got {type(call).__name__}",
            ))
            continue

        function = call.get("function")
        function = dict(function) if isinstance(function, dict) else {}
        name = function.get("name")
        raw_arguments = function.get("arguments")

        parsed, error = _parse_arguments(raw_arguments)
        if error is not None:
            # Last resort: the same syntax repair the JSON response path uses.
            healed = heal_json(raw_arguments if isinstance(raw_arguments, str) else "")
            if healed.ok and isinstance(healed.value, dict):
                parsed = healed.value
            else:
                unresolved.append(ToolCallIssue(
                    index=index, tool_name=name if isinstance(name, str) else None,
                    issue=ISSUE_MALFORMED_ARGUMENTS, detail=error,
                ))
                repaired.append(call)   # pass through untouched rather than drop it
                continue

        serialized = json.dumps(parsed, separators=(",", ":"))
        if serialized != raw_arguments:
            changed = True
        function["arguments"] = serialized

        if isinstance(name, str) and schemas and name not in schemas:
            unresolved.append(ToolCallIssue(
                index=index, tool_name=name, issue=ISSUE_UNKNOWN_TOOL,
                detail=f"{name!r} was not among the offered tools: {sorted(schemas)}",
            ))

        repaired.append({**call, "function": function})

    return ToolCallRepair(
        repaired=repaired, changed=changed, unresolved=tuple(unresolved),
    )


def corrective_tool_prompt(issues: tuple[ToolCallIssue, ...]) -> str:
    """The message appended for one corrective round-trip, naming the exact
    problems per call. Same reasoning as `contracts.corrective_prompt()`: a model
    asked to "fix your tool call" with no specifics is no more likely to succeed
    than it was the first time.

    Includes the offered tool names for an unknown-tool issue, because that is the
    one failure a model genuinely cannot correct without being told what it was
    allowed to call."""
    lines = [
        "Your previous tool call(s) were rejected. Fix exactly these problems and "
        "reply with corrected tool calls only:",
    ]
    for issue in issues:
        target = f"call #{issue.index}"
        if issue.tool_name:
            target += f" ({issue.tool_name})"
        lines.append(f"- {target}: {issue.detail}")
    lines.append(
        "Emit each tool call's arguments as a single JSON object — not as a "
        "string containing JSON, and not with trailing commentary."
    )
    return "\n".join(lines)

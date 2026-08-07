"""L7 — Contracts (ARCHITECTURE-PLAN.md's L7 section): the upgrade from
healing.py's syntax-only JSON repair to real semantic ENFORCEMENT — validate
a response against a caller-supplied JSON Schema, not just check that it
parses. `heal_json()` already answers "is this valid JSON at all"; this
module answers "does this valid JSON actually match the shape the caller
asked for."

Needs the optional `jsonschema` package (see pyproject.toml's `contracts`
extra) — lazily imported, same zero-required-deps discipline every other
real integration in this codebase already follows (openai/anthropic/ollama
adapters all import their SDK lazily inside __init__, not at module load
time). Without it installed, validate_contract() raises a clear ConfigError
naming the exact install command, rather than an opaque ImportError three
frames deep in someone else's stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelrouter.core.errors import ConfigError


@dataclass(frozen=True)
class ContractViolation:
    message: str        # human-readable, e.g. "'age' is a required property"
    json_path: str       # e.g. "$.age" -- confirmed against jsonschema's real ValidationError.json_path
    validator: str        # the failed schema keyword, e.g. "required", "type", "enum"

    def as_dict(self) -> dict:
        """JSON-serializable shape -- what actually lands in the pipeline
        trace and in ContractViolationError.violations. Never the dataclass
        instance itself, which server.py's JSON response body can't encode."""
        return {"message": self.message, "json_path": self.json_path, "validator": self.validator}


@dataclass(frozen=True)
class ContractResult:
    ok: bool
    violations: tuple[ContractViolation, ...] = ()


def validate_contract(instance: object, schema: dict) -> ContractResult:
    """Validates `instance` (already-parsed JSON, e.g. HealingResult.value)
    against `schema` (a real JSON Schema dict — the caller's own, never
    fabricated here). Collects EVERY violation, not just the first, sorted
    by the path in `instance` they occurred at — a caller correcting several
    fields at once needs the full list, not one-at-a-time discovery."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as e:
        raise ConfigError(
            "JSON Schema contract enforcement needs the optional 'jsonschema' "
            "package. Install it with: pip install -e '.[contracts]'"
        ) from e

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    violations = tuple(
        ContractViolation(message=e.message, json_path=e.json_path, validator=str(e.validator))
        for e in errors
    )
    return ContractResult(ok=not violations, violations=violations)


def corrective_prompt(violations: tuple[ContractViolation, ...]) -> str:
    """The message appended for `ChatRequest.contract_policy="retry"`'s one
    corrective round-trip — names the EXACT violations rather than a vague
    "fix your JSON," since a model correcting blind is no better than not
    correcting at all."""
    lines = "\n".join(f"- {v.json_path}: {v.message}" for v in violations)
    return (
        "Your previous response did not match the required JSON Schema. "
        "Reply again with ONLY corrected JSON that fixes these specific problems:\n"
        f"{lines}"
    )

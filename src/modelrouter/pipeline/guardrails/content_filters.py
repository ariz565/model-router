"""Content-level filters: prompt-injection patterns, PII presets, and
user-defined custom filters.

PII detection here is regex/preset-based only (the doc describes OpenRouter's
own filters as "NLP + preset-based" — this is the preset half; real NLP entity
detection is a genuine, separate upgrade, not faked as equivalent here).

Prompt-injection patterns are a real, documented OWASP-pattern starter set,
not an exhaustive/adversarially-robust detector — injection detection is an
open problem industry-wide; this catches the common, well-known phrasings the
architecture doc itself references, nothing more is claimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ModelGroup(str, Enum):
    """The four independently-toggleable ZDR scopes, per the architecture doc."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    NON_FRONTIER = "non_frontier"


class FilterAction(str, Enum):
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class ContentFilter:
    """A user-defined regex content filter. No lookaheads/lookbehinds or
    backreferences allowed — the doc states this explicitly, to keep
    evaluation bounded/fast (an unbounded regex is a real DoS vector on
    attacker-controlled input, which every request's message content is)."""

    name: str
    pattern: str
    action: FilterAction = FilterAction.BLOCK

    def __post_init__(self) -> None:
        if re.search(r"\(\?[=!<]", self.pattern):
            raise ValueError(f"content filter {self.name!r}: lookaheads/lookbehinds not allowed")
        if re.search(r"\\[1-9]", self.pattern):
            raise ValueError(f"content filter {self.name!r}: backreferences not allowed")
        self._compiled = re.compile(self.pattern)

    def scan(self, text: str) -> bool:
        return self._compiled.search(text) is not None

    def redact(self, text: str) -> str:
        return self._compiled.sub("[REDACTED]", text)


# Built-in PII presets (regex-only — see module docstring for the NLP caveat).
PII_PRESETS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone_us": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

# OWASP-pattern-based prompt-injection heuristics — a real starter set of the
# common, documented phrasings, not an exhaustive/robust detector.
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all )?(previous|prior|above)", re.I),
    re.compile(r"you are now (in )?(developer|debug|admin|jailbreak) mode", re.I),
    re.compile(r"reveal (your |the )?(system prompt|instructions)", re.I),
    re.compile(r"act as (if you (were|are)|an?) (unrestricted|unfiltered|dan)\b", re.I),
    re.compile(r"pretend (you have |there are )?no (rules|restrictions|guidelines)", re.I),
]


def scan_for_injection(text: str) -> str | None:
    """Returns the matched pattern's source, or None if clean."""
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None

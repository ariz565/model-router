"""Response healing — repair a model's malformed JSON output before it hits
the caller's parser. Non-streaming only: a stream can't be retroactively
patched once tokens are already delivered to the caller (the same reasoning
retry_policy.py inherits from app/services/llm.py's stream_chat() docstring
elsewhere in this repo — a stream cannot be restarted or edited mid-flight).

Implements the four specific repairs the architecture doc names:
  1. strip markdown code fences (```json ... ```)
  2. extract a JSON object/array out of surrounding prose
  3. fix unquoted object keys
  4. fix missing/trailing commas and unbalanced closing brackets

Cannot fix truncation (the doc's own stated limit) — a genuinely cut-off JSON
value has no safe repair; inventing the missing content would silently
fabricate data, which is the exact failure mode this whole project treats as
unacceptable everywhere else, so heal_json() reports ok=False rather than
guessing when every repair attempt still fails to parse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def strip_markdown_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    return match.group(1) if match else text


def extract_json_span(text: str) -> str | None:
    """Finds the first balanced {...} or [...] span in text, ignoring braces
    inside string literals (a naive brace-counter would miscount a `}` that
    appears inside a quoted string value)."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def fix_unquoted_keys(text: str) -> str:
    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)


def fix_trailing_commas(text: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', text)


def fix_missing_commas_between_pairs(text: str) -> str:
    """A common LLM mistake: `"a": 1 "b": 2` missing the comma between pairs."""
    return re.sub(r'("|\d|true|false|null)(\s*\n\s*)("[\w]+"\s*:)', r'\1,\2\3', text)


def close_unbalanced_brackets(text: str) -> str:
    opens = {"{": "}", "[": "]"}
    stack: list[str] = []
    for ch in text:
        if ch in opens:
            stack.append(opens[ch])
        elif ch in opens.values() and stack and stack[-1] == ch:
            stack.pop()
    return text + "".join(reversed(stack))


@dataclass
class HealingResult:
    ok: bool
    value: object
    healed: bool                # True if any repair changed the text, even if parsing still failed
    original: str
    repaired_text: str
    error: str | None = None


def heal_json(raw_text: str) -> HealingResult:
    """Attempts, in order: parse as-is -> strip fences -> extract a balanced
    span from prose -> structural repairs (keys/commas/brackets). Returns the
    first successful parse. Never guesses a value when every attempt still
    fails — ok=False with the last parser error instead."""
    original = raw_text

    try:
        return HealingResult(ok=True, value=json.loads(raw_text), healed=False, original=original, repaired_text=raw_text)
    except json.JSONDecodeError:
        pass

    stripped = strip_markdown_fences(raw_text).strip()
    try:
        return HealingResult(ok=True, value=json.loads(stripped), healed=True, original=original, repaired_text=stripped)
    except json.JSONDecodeError:
        pass

    span = extract_json_span(stripped)
    candidate = span if span is not None else stripped
    try:
        return HealingResult(ok=True, value=json.loads(candidate), healed=True, original=original, repaired_text=candidate)
    except json.JSONDecodeError:
        pass

    repaired = candidate
    repaired = fix_unquoted_keys(repaired)
    repaired = fix_missing_commas_between_pairs(repaired)
    repaired = fix_trailing_commas(repaired)
    repaired = close_unbalanced_brackets(repaired)
    try:
        return HealingResult(ok=True, value=json.loads(repaired), healed=True, original=original, repaired_text=repaired)
    except json.JSONDecodeError as e:
        return HealingResult(ok=False, value=None, healed=True, original=original, repaired_text=repaired, error=str(e))

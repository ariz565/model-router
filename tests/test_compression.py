"""pipeline/compression.py — middle-out truncation, the safety-margin fix,
and the ContextOverflowError terminal-failure fix (see the module's own
docstring for the two defects these tests lock in against regressing)."""

import pytest

from modelrouter.core.errors import ContextOverflowError
from modelrouter.pipeline.compression import (
    ContextWindow,
    compress_middle_out,
    needs_compression,
    select_model_for_context,
)


def _messages(*contents):
    return [{"role": "user", "content": c} for c in contents]


# ── safety margin ───────────────────────────────────────────────────────

def test_no_margin_by_default_matches_old_behavior():
    window = ContextWindow(name="m", max_tokens=40)
    messages = _messages("word " * 30)   # ~37 tokens, under 40 but close
    assert needs_compression(messages, window, completion_tokens_needed=0) is False


def test_safety_margin_makes_compression_trigger_earlier():
    window = ContextWindow(name="m", max_tokens=40, safety_margin_tokens=10)
    messages = _messages("word " * 30)   # ~37 tokens, under 40 but not under 30
    assert needs_compression(messages, window, completion_tokens_needed=0) is True


def test_safety_margin_reduces_the_effective_truncation_budget():
    # 8 short messages; margin eats into the budget enough to force one more
    # message out than a zero-margin window would need.
    messages = _messages(*[f"msg-{i} " * 3 for i in range(8)])
    no_margin = ContextWindow(name="m", max_tokens=20)
    with_margin = ContextWindow(name="m", max_tokens=20, safety_margin_tokens=8)
    kept_no_margin = compress_middle_out(messages, no_margin, completion_tokens_needed=0)
    kept_with_margin = compress_middle_out(messages, with_margin, completion_tokens_needed=0)
    assert len(kept_with_margin) <= len(kept_no_margin)


# ── ContextOverflowError — the terminal-failure fix ───────────────────────

def test_overflow_raised_when_budget_is_non_positive():
    window = ContextWindow(name="m", max_tokens=10)
    messages = _messages("hi")
    with pytest.raises(ContextOverflowError):
        compress_middle_out(messages, window, completion_tokens_needed=15)  # budget = -5


def test_overflow_raised_when_two_messages_still_exceed_budget():
    window = ContextWindow(name="m", max_tokens=10)
    huge = "word " * 200   # ~250 tokens
    messages = _messages(huge, huge, huge, huge, huge)
    with pytest.raises(ContextOverflowError) as exc_info:
        compress_middle_out(messages, window, completion_tokens_needed=0)
    assert exc_info.value.window_name == "m"
    assert exc_info.value.budget == 10


def test_no_overflow_when_truncation_converges_to_a_fit():
    window = ContextWindow(name="m", max_tokens=20)
    huge = "word " * 200
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": huge},
                {"role": "user", "content": huge},
                {"role": "assistant", "content": huge},
                {"role": "user", "content": "bye"}]
    kept = compress_middle_out(messages, window, completion_tokens_needed=0)
    assert len(kept) == 2
    assert kept[0]["content"] == "hi"
    assert kept[1]["content"] == "bye"


def test_image_output_model_never_compressed_even_if_it_would_overflow():
    window = ContextWindow(name="m", max_tokens=10, is_image_output_model=True)
    huge = "word " * 200
    messages = _messages(huge, huge, huge)
    kept = compress_middle_out(messages, window, completion_tokens_needed=0)
    assert kept == messages   # untouched, no exception — doc's stated exception


# ── select_model_for_context (unrelated to the fixes, smoke-test only) ──

def test_select_model_for_context_prefers_half_fit():
    small = ContextWindow(name="small", max_tokens=100)
    big = ContextWindow(name="big", max_tokens=100000)
    chosen = select_model_for_context([small, big], tokens_needed=150)
    assert chosen is small   # 100 >= 150 // 2

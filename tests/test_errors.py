"""core/errors.py — the centralized classification/messaging/exception
module. Covers: status-code classification (matches retry_policy's old
inline logic exactly, now delegated here), Retry-After parsing, message
formatting, and ModelRouter's own typed exceptions."""

import pytest

from modelrouter.core.errors import (
    AllCandidatesExhaustedError,
    ConfigError,
    InternalError,
    ModelRouterError,
    NoAdapterError,
    classify_error,
    format_error_message,
    raise_if_failed,
)
from modelrouter.core.types import RouterMetadata


class _HttpError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"http {status_code}")
        self.status_code = status_code
        self.response = type("Resp", (), {"headers": headers or {}})()


# ── classify_error() ────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504, 529])
def test_retryable_statuses(status):
    info = classify_error(_HttpError(status))
    assert info.retryable is True
    assert info.status_code == status


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_deterministic_4xx_not_retryable(status):
    info = classify_error(_HttpError(status))
    assert info.retryable is False


def test_unknown_5xx_still_retryable():
    assert classify_error(_HttpError(599)).retryable is True


def test_timeout_and_connection_errors_retryable():
    assert classify_error(TimeoutError()).retryable is True
    assert classify_error(ConnectionError()).retryable is True


def test_unknown_exception_fails_fast():
    info = classify_error(ValueError("bad input"))
    assert info.retryable is False
    assert info.status_code is None


def test_retry_after_seconds_form_extracted():
    exc = _HttpError(429, headers={"retry-after": "5"})
    info = classify_error(exc)
    assert info.retry_after_s == 5.0


def test_retry_after_omitted_when_respect_retry_after_false():
    exc = _HttpError(429, headers={"retry-after": "5"})
    info = classify_error(exc, respect_retry_after=False)
    assert info.retry_after_s is None


def test_error_message_uses_known_status_wording():
    info = classify_error(_HttpError(401))
    assert "authentication failed" in info.message.lower()


def test_error_message_falls_back_for_unknown_status():
    info = classify_error(_HttpError(599))
    assert "599" in info.message


def test_error_message_for_exception_with_no_status_code():
    message = format_error_message(ValueError("bad input"))
    assert "ValueError" in message
    assert "bad input" in message


# ── ModelRouter's own exception hierarchy ───────────────────────────────

def test_no_adapter_error_lists_known_providers():
    exc = NoAdapterError("ghostprovider", frozenset({"openai", "anthropic"}))
    assert "ghostprovider" in str(exc)
    assert "openai" in str(exc)
    assert isinstance(exc, ModelRouterError)


def test_config_error_is_a_model_router_error():
    assert isinstance(ConfigError("missing key"), ModelRouterError)


def test_all_candidates_exhausted_error_message():
    exc = AllCandidatesExhaustedError("openai:gpt-5.4-nano", attempts=3)
    assert "gpt-5.4-nano" in str(exc)
    assert "3" in str(exc)


def test_raise_if_failed_raises_on_none_response():
    metadata = RouterMetadata(requested_model="openai:gpt-5.4-nano", attempt=2)
    with pytest.raises(AllCandidatesExhaustedError):
        raise_if_failed(None, metadata)


def test_raise_if_failed_no_op_on_success():
    metadata = RouterMetadata(requested_model="openai:gpt-5.4-nano", served_by="openai:gpt-5.4-nano", attempt=1)
    raise_if_failed(object(), metadata)   # must not raise


def test_internal_error_is_a_model_router_error():
    assert isinstance(InternalError("boom"), ModelRouterError)


def test_internal_error_preserves_cause_chain():
    original = RuntimeError("unexpected bug")
    try:
        try:
            raise original
        except RuntimeError as e:
            raise InternalError("chat() failed unexpectedly: unexpected bug") from e
    except InternalError as wrapped:
        assert wrapped.__cause__ is original

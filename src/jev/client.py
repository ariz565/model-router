"""Production-oriented Jev HTTP clients with injectable transport and retries."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from email.message import Message
from typing import Callable, Mapping, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from jev.errors import (
    JevAPIError, JevAuthenticationError, JevConfigurationError,
    JevConnectionError, JevRateLimitError, JevValidationError,
)
from jev.models import JsonValue, Question, SystemOneResponse, parse_response, request_payload

DEFAULT_BASE_URL = "https://api.typesafe.ai"


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    headers: Mapping[str, str]
    data: object


@runtime_checkable
class Transport(Protocol):
    def request(self, method: str, url: str, *, headers: Mapping[str, str], json_body: dict | None,
                timeout_s: float) -> TransportResponse: ...


@runtime_checkable
class AsyncTransport(Protocol):
    async def request(self, method: str, url: str, *, headers: Mapping[str, str],
                      json_body: dict | None, timeout_s: float) -> TransportResponse: ...

    async def aclose(self) -> None: ...


class UrllibTransport:
    def __init__(self, *, max_response_bytes: int = 4 * 1024 * 1024):
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self._max_response_bytes = max_response_bytes

    def request(self, method: str, url: str, *, headers: Mapping[str, str], json_body: dict | None,
                timeout_s: float) -> TransportResponse:
        body = json.dumps(json_body, separators=(",", ":")).encode() if json_body is not None else None
        request = Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - fixed operator URL by default
                return TransportResponse(response.status, dict(response.headers.items()),
                                         _decode_limited(response.read(self._max_response_bytes + 1),
                                                         self._max_response_bytes))
        except HTTPError as exc:
            return TransportResponse(
                exc.code, dict(exc.headers.items()) if isinstance(exc.headers, Message) else {},
                _decode_limited(exc.read(self._max_response_bytes + 1), self._max_response_bytes),
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise JevConnectionError(f"Jev API connection failed: {type(exc).__name__}") from exc


def _decode_limited(body: bytes, maximum: int) -> object:
    if len(body) > maximum:
        raise JevValidationError(f"Jev API response exceeds {maximum} bytes")
    try:
        return json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JevValidationError("Jev API returned a non-JSON response") from exc


class HttpxAsyncTransport:
    """Connection-pooled async transport. ``httpx`` is an optional dependency."""

    def __init__(self, *, max_connections: int = 100, max_keepalive_connections: int = 20,
                 verify: bool | str = True, proxy: str | None = None,
                 max_response_bytes: int = 4 * 1024 * 1024):
        if min(max_connections, max_keepalive_connections, max_response_bytes) < 1:
            raise ValueError("connection and response limits must be positive")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on installation extras
            raise JevConfigurationError(
                "AsyncJevClient requires the 'jev' extra: pip install modelrouter[jev]"
            ) from exc
        self._httpx = httpx
        self._maximum = max_response_bytes
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_keepalive_connections),
            verify=verify, proxy=proxy,
        )

    async def request(self, method: str, url: str, *, headers: Mapping[str, str],
                      json_body: dict | None, timeout_s: float) -> TransportResponse:
        try:
            async with self._client.stream(method, url, headers=dict(headers), json=json_body,
                                           timeout=timeout_s) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self._maximum:
                        raise JevValidationError(f"Jev API response exceeds {self._maximum} bytes")
                    chunks.append(chunk)
                body = b"".join(chunks)
                return TransportResponse(
                    response.status_code, dict(response.headers),
                    _decode_limited(body, self._maximum),
                )
        except self._httpx.RequestError as exc:
            raise JevConnectionError(f"Jev API connection failed: {type(exc).__name__}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


class _ThreadedTransport:
    """Compatibility adapter for existing synchronous custom transports."""

    def __init__(self, transport: Transport):
        self._transport = transport

    async def request(self, method: str, url: str, *, headers: Mapping[str, str],
                      json_body: dict | None, timeout_s: float) -> TransportResponse:
        return await asyncio.to_thread(
            self._transport.request, method, url, headers=headers,
            json_body=json_body, timeout_s=timeout_s,
        )

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            raise ValueError("invalid retry policy")


@dataclass(frozen=True)
class ClientEvent:
    name: str
    attempt: int
    duration_s: float
    status_code: int | None = None
    request_id: str | None = None
    error_type: str | None = None


class JevClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = DEFAULT_BASE_URL,
                 model: str = "jev-latest", timeout_s: float = 30.0,
                 total_timeout_s: float | None = None, max_request_bytes: int = 4 * 1024 * 1024,
                 retry_policy: RetryPolicy | None = None, transport: Transport | None = None,
                 on_event: Callable[[ClientEvent], None] | None = None):
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise JevConfigurationError("TYPESAFE_API_KEY or api_key is required")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme != "https" and not (
            parsed_url.scheme == "http" and parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise JevConfigurationError("base_url must use HTTPS (HTTP is allowed only for exact localhost URLs)")
        self._base_url = base_url.rstrip("/")
        self._model = model
        if timeout_s <= 0 or (total_timeout_s is not None and total_timeout_s <= 0):
            raise JevConfigurationError("timeouts must be positive")
        if max_request_bytes < 1:
            raise JevConfigurationError("max_request_bytes must be positive")
        self._timeout_s = timeout_s
        self._total_timeout_s = total_timeout_s
        self._max_request_bytes = max_request_bytes
        self._retry = retry_policy or RetryPolicy()
        self._transport = transport or UrllibTransport()
        self._on_event = on_event

    def system_one(self, *, state: JsonValue, questions: Mapping[str, Question],
                   model: str | None = None) -> SystemOneResponse:
        payload = request_payload(state, questions, model or self._model)
        response = self._request("POST", "/v1/systemone", payload)
        if not isinstance(response.data, dict):
            raise JevValidationError("Jev API response must be an object")
        return parse_response(response.data, questions)

    def list_models(self) -> list[dict]:
        response = self._request("GET", "/v1/models", None)
        if not isinstance(response.data, dict) or not isinstance(response.data.get("models"), list):
            raise JevValidationError("malformed models response")
        return list(response.data["models"])

    def _request(self, method: str, path: str, body: dict | None) -> TransportResponse:
        _validate_request_size(body, self._max_request_bytes)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                   "User-Agent": "jev-python/1.0"}
        deadline = time.monotonic() + self._total_timeout_s if self._total_timeout_s else None
        for attempt in range(self._retry.max_attempts):
            started = time.monotonic()
            timeout = _remaining_timeout(deadline, self._timeout_s)
            try:
                response = self._transport.request(method, self._base_url + path, headers=headers,
                                                   json_body=body, timeout_s=timeout)
            except JevConnectionError as exc:
                self._emit(ClientEvent("request.failed", attempt, time.monotonic() - started,
                                       error_type=type(exc).__name__))
                if attempt + 1 == self._retry.max_attempts:
                    raise
                _sleep_with_deadline(self._delay(attempt, None), deadline)
                continue
            request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
            self._emit(ClientEvent("request.completed", attempt, time.monotonic() - started,
                                   status_code=response.status_code, request_id=request_id))
            if 200 <= response.status_code < 300:
                return response
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                if attempt + 1 < self._retry.max_attempts:
                    _sleep_with_deadline(self._delay(attempt, _header(response.headers, "retry-after")), deadline)
                    continue
            self._raise_api(response, request_id)
        raise AssertionError("retry loop exhausted without returning or raising")

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        parsed = _retry_after_seconds(retry_after)
        if parsed is not None:
            return min(parsed, self._retry.max_delay_s)
        delay = min(self._retry.base_delay_s * (2 ** attempt), self._retry.max_delay_s)
        return random.uniform(0, delay) if self._retry.jitter else delay

    def _raise_api(self, response: TransportResponse, request_id: str | None) -> None:
        message = "Jev API request failed"
        if isinstance(response.data, dict):
            raw = response.data.get("message") or response.data.get("error")
            if isinstance(raw, str):
                message = raw
        cls = JevAuthenticationError if response.status_code in {401, 403} else (
            JevRateLimitError if response.status_code == 429 else JevAPIError)
        raise cls(response.status_code, message, request_id=request_id)

    def _emit(self, event: ClientEvent) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass  # telemetry must never change the request outcome


class AsyncJevClient(JevClient):
    """Native async client with pooling, cancellation, retry and explicit shutdown.

    A synchronous custom transport remains supported through a worker-thread
    adapter for backwards compatibility. The default transport is native async.
    """

    def __init__(self, api_key: str | None = None, *, transport: AsyncTransport | Transport | None = None,
                 **kwargs):
        # Initialise shared validation/configuration without retaining its sync transport.
        super().__init__(api_key, transport=UrllibTransport(), **kwargs)
        if transport is None:
            self._async_transport: AsyncTransport = HttpxAsyncTransport()
        elif inspect.iscoroutinefunction(transport.request):
            self._async_transport = transport  # type: ignore[assignment]
        else:
            self._async_transport = _ThreadedTransport(transport)  # type: ignore[arg-type]

    async def system_one(self, *, state: JsonValue, questions: Mapping[str, Question],
                         model: str | None = None) -> SystemOneResponse:
        payload = request_payload(state, questions, model or self._model)
        response = await self._async_request("POST", "/v1/systemone", payload)
        if not isinstance(response.data, dict):
            raise JevValidationError("Jev API response must be an object")
        return parse_response(response.data, questions)

    async def list_models(self) -> list[dict]:
        response = await self._async_request("GET", "/v1/models", None)
        if not isinstance(response.data, dict) or not isinstance(response.data.get("models"), list):
            raise JevValidationError("malformed models response")
        return list(response.data["models"])

    async def _async_request(self, method: str, path: str, body: dict | None) -> TransportResponse:
        _validate_request_size(body, self._max_request_bytes)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                   "User-Agent": "jev-python/1.0"}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._total_timeout_s if self._total_timeout_s else None
        for attempt in range(self._retry.max_attempts):
            started = loop.time()
            timeout = _remaining_timeout(deadline, self._timeout_s, now=loop.time())
            try:
                response = await asyncio.wait_for(
                    self._async_transport.request(
                        method, self._base_url + path, headers=headers,
                        json_body=body, timeout_s=timeout,
                    ), timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                error = JevConnectionError("Jev API operation deadline exceeded")
                self._emit(ClientEvent("request.failed", attempt, loop.time() - started,
                                       error_type=type(error).__name__))
                if attempt + 1 == self._retry.max_attempts or _deadline_expired(deadline, loop.time()):
                    raise error from exc
                await _async_sleep_with_deadline(self._delay(attempt, None), deadline)
                continue
            except JevConnectionError as exc:
                self._emit(ClientEvent("request.failed", attempt, loop.time() - started,
                                       error_type=type(exc).__name__))
                if attempt + 1 == self._retry.max_attempts:
                    raise
                await _async_sleep_with_deadline(self._delay(attempt, None), deadline)
                continue
            request_id = _header(response.headers, "x-request-id")
            self._emit(ClientEvent("request.completed", attempt, loop.time() - started,
                                   status_code=response.status_code, request_id=request_id))
            if 200 <= response.status_code < 300:
                return response
            if response.status_code in {408, 409, 429} or response.status_code >= 500:
                if attempt + 1 < self._retry.max_attempts:
                    await _async_sleep_with_deadline(
                        self._delay(attempt, _header(response.headers, "retry-after")), deadline,
                    )
                    continue
            self._raise_api(response, request_id)
        raise AssertionError("retry loop exhausted without returning or raising")

    async def aclose(self) -> None:
        await self._async_transport.aclose()

    async def __aenter__(self) -> "AsyncJevClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    return next((value for key, value in headers.items() if key.lower() == target), None)


def _retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            return max(0.0, (parsed - current).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _validate_request_size(body: dict | None, maximum: int) -> None:
    if body is not None and len(json.dumps(body, separators=(",", ":")).encode()) > maximum:
        raise JevValidationError(f"Jev API request exceeds {maximum} bytes")


def _deadline_expired(deadline: float | None, now: float | None = None) -> bool:
    return deadline is not None and (now if now is not None else time.monotonic()) >= deadline


def _remaining_timeout(deadline: float | None, per_attempt: float, *, now: float | None = None) -> float:
    if deadline is None:
        return per_attempt
    remaining = deadline - (now if now is not None else time.monotonic())
    if remaining <= 0:
        raise JevConnectionError("Jev API operation deadline exceeded")
    return min(per_attempt, remaining)


def _sleep_with_deadline(delay: float, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() + delay >= deadline:
        raise JevConnectionError("Jev API operation deadline exceeded")
    time.sleep(delay)


async def _async_sleep_with_deadline(delay: float, deadline: float | None) -> None:
    loop = asyncio.get_running_loop()
    if deadline is not None and loop.time() + delay >= deadline:
        raise JevConnectionError("Jev API operation deadline exceeded")
    await asyncio.sleep(delay)

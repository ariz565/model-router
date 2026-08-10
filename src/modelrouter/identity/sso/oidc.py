"""OIDC Authorization Code Flow with PKCE — the protocol layer.

Deliberately split into two halves, because they have different testability and
different risk profiles:

- **Pure functions** (everything above `OidcClient`): PKCE generation,
  authorization-URL construction, ID-token *claim* validation, the
  `email_verified` coercion, the issuer mix-up check, and the open-redirect
  check. No network, no crypto library, fully deterministic — and this is where
  the large majority of real SSO vulnerabilities live, so it is where the tests
  concentrate.
- **`OidcClient`** (IO): discovery, code exchange, and JWS signature
  verification against the IdP's JWKS. Lazily imports `httpx`/`joserfc` so this
  module is importable — and the pure half fully testable — with neither
  installed.

**Standards this follows, by name.** RFC 9700 (OAuth 2.0 Security Best Current
Practice) and OIDC Core §3.1.3.7 for ID-token validation. Notably:
PKCE `S256` is used unconditionally even though this is a *confidential*
client (OAuth 2.1 makes it mandatory for all clients and it costs nothing);
`redirect_uri` is exact-string matched with no wildcards; and the `iss`
response parameter is checked because a multi-IdP relying party is precisely
the shape that IdP mix-up attacks target.

**What is deliberately NOT requested:** `offline_access`/refresh tokens. This
flow authenticates a human; it never calls the IdP's APIs on their behalf, so
holding a long-lived refresh token would be storing a credential with no use
and real blast radius.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import urlencode

__all__ = [
    "SsoError", "OidcProtocolError", "IdTokenInvalidError", "IssuerMismatchError",
    "UnsafeRedirectError",
    "DEFAULT_SCOPES", "CLOCK_SKEW_LEEWAY_SECONDS",
    "generate_state", "generate_nonce", "generate_pkce_pair",
    "build_authorization_url", "coerce_email_verified", "validate_id_token_claims",
    "assert_issuer_matches", "validate_return_to", "discovery_url_for",
    "OidcClient",
]

DEFAULT_SCOPES = ("openid", "email", "profile")

# Tolerance for `exp`/`iat` comparisons. Some clock skew between us and an IdP
# is normal and unavoidable; a leeway this small still rejects a genuinely
# expired token while not failing logins over a two-second drift.
CLOCK_SKEW_LEEWAY_SECONDS = 60

_STATE_BYTES = 32
_NONCE_BYTES = 32
_VERIFIER_BYTES = 32


class SsoError(Exception):
    """Base for every SSO failure. An HTTP layer should render ALL of these as
    one generic "sign-in failed" to the user: the specific reason (unknown
    issuer, bad nonce, expired transaction) is diagnostic information for our
    logs, and echoing it back to an unauthenticated caller turns the login
    endpoint into an oracle for probing our validation rules."""


class OidcProtocolError(SsoError):
    """The IdP's response was malformed or the flow was used incorrectly."""


class IdTokenInvalidError(SsoError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"ID token rejected: {reason}")


class IssuerMismatchError(SsoError):
    """The callback claims to come from a different IdP than the one this
    transaction was started against — the IdP mix-up attack. Separate from
    `IdTokenInvalidError` because it must be loud in logs: a legitimate client
    never produces it."""

    def __init__(self, expected: str, received: str):
        self.expected = expected
        self.received = received
        super().__init__(f"issuer mismatch: expected {expected!r}, got {received!r}")


class UnsafeRedirectError(SsoError):
    def __init__(self, value: str):
        self.value = value
        super().__init__(f"refusing to redirect to {value!r}")


# ── Random values ─────────────────────────────────────────────────────────

def generate_state() -> str:
    """CSRF protection AND our transaction lookup key. Unguessable, so a
    third party cannot fabricate a callback that matches a real in-flight
    login."""
    return secrets.token_urlsafe(_STATE_BYTES)


def generate_nonce() -> str:
    """Replay protection: bound into the authorization request and compared
    against the ID token's `nonce` claim, which proves the token was minted for
    THIS request rather than replayed from another."""
    return secrets.token_urlsafe(_NONCE_BYTES)


def generate_pkce_pair() -> tuple[str, str]:
    """Returns `(code_verifier, code_challenge)` using the `S256` method.

    `plain` is prohibited by current guidance and not offered here — a
    `plain` challenge is the verifier, so anyone who intercepts the
    authorization request can redeem the code themselves, which defeats the
    entire mechanism."""
    verifier = secrets.token_urlsafe(_VERIFIER_BYTES)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


# ── Authorization request ─────────────────────────────────────────────────

def discovery_url_for(issuer: str) -> str:
    return f"{issuer.rstrip('/')}/.well-known/openid-configuration"


def build_authorization_url(
    authorization_endpoint: str, *, client_id: str, redirect_uri: str,
    state: str, nonce: str, code_challenge: str,
    scopes: Iterable[str] = DEFAULT_SCOPES, extra: dict[str, str] | None = None,
) -> str:
    """`extra` exists for IdP-specific parameters (`hd` for Google Workspace,
    `login_hint`, `prompt`) without this function growing a keyword per vendor.

    `code_challenge_method` is hardcoded to `S256` rather than parameterized:
    making it configurable would mean offering `plain` as a supported option,
    and there is no legitimate reason to choose it."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if extra:
        params.update(extra)
    separator = "&" if "?" in authorization_endpoint else "?"
    return f"{authorization_endpoint}{separator}{urlencode(params)}"


# ── Claim validation ──────────────────────────────────────────────────────

def coerce_email_verified(raw: Any) -> bool:
    """**Fails closed for anything that is not unambiguously true.**

    This exists because of a real, published vulnerability class: an
    implementation that type-asserted this claim as `bool` treated a
    string `"false"` — or an absent claim — as verified, yielding full account
    takeover. IdPs genuinely do send `true`/`false` as strings, as `1`/`0`, and
    sometimes omit the claim entirely.

    So: only literal `True`, the strings `"true"`/`"1"` (case-insensitively),
    and the integer `1` mean verified. Everything else — including `None`, a
    missing claim, an unexpected type, or any other string — is NOT verified."""
    if raw is True:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw == 1
    return False


def validate_id_token_claims(
    claims: dict, *, expected_issuer: str, expected_audience: str,
    expected_nonce: str, now: datetime | None = None,
    leeway_seconds: int = CLOCK_SKEW_LEEWAY_SECONDS,
) -> None:
    """Every check OIDC Core §3.1.3.7 requires, in one place, raising on the
    first failure. Signature verification is NOT here — that happens in
    `OidcClient.verify_id_token()` before this is called, because it needs the
    JWKS. Both are mandatory; neither is sufficient alone (a validly-signed
    token from the wrong issuer, or for a different audience, is still an
    attack).

    `sub` is required and must be non-empty because it is the ONLY key this
    system uses to resolve an identity to a user. A token with no stable
    subject cannot be safely turned into a login at all."""
    current = now or datetime.now(timezone.utc)
    leeway = timedelta(seconds=leeway_seconds)

    issuer = claims.get("iss")
    if issuer != expected_issuer:
        raise IssuerMismatchError(expected_issuer, str(issuer))

    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if expected_audience not in audiences:
        raise IdTokenInvalidError(f"aud {audience!r} does not contain our client_id")

    # `azp` is only required when there are multiple audiences, but when the IdP
    # sends it, it must be us -- otherwise the token was minted for a different
    # client that happens to list us as an audience.
    authorized_party = claims.get("azp")
    if authorized_party is not None and authorized_party != expected_audience:
        raise IdTokenInvalidError(f"azp {authorized_party!r} is not our client_id")

    if claims.get("nonce") != expected_nonce:
        # Never log or echo the expected value -- it's a live secret until the
        # transaction is consumed.
        raise IdTokenInvalidError("nonce does not match this login attempt")

    expiry = claims.get("exp")
    if not isinstance(expiry, (int, float)):
        raise IdTokenInvalidError("missing or non-numeric exp")
    if datetime.fromtimestamp(expiry, tz=timezone.utc) + leeway <= current:
        raise IdTokenInvalidError("token has expired")

    issued_at = claims.get("iat")
    if not isinstance(issued_at, (int, float)):
        raise IdTokenInvalidError("missing or non-numeric iat")
    if datetime.fromtimestamp(issued_at, tz=timezone.utc) - leeway > current:
        raise IdTokenInvalidError("token was issued in the future")

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise IdTokenInvalidError("missing sub")


def assert_issuer_matches(received_issuer: str | None, expected_issuer: str) -> None:
    """Mix-up defense (RFC 9207 / RFC 9700 §2.1) applied to the `iss`
    *response parameter* on the callback, before any token exchange happens.

    Why it matters here specifically: this relying party talks to many IdPs
    through ONE callback URL. Without this check, an attacker who controls (or
    registers) one tenant's IdP can start a flow, receive a code, and present it
    on the callback as though it came from a different, higher-value tenant's
    IdP. Checking `iss` against the issuer captured when the transaction started
    closes that.

    An IdP that omits `iss` is tolerated (it predates RFC 9207) — the ID token's
    own `iss` claim is still validated in `validate_id_token_claims`, so the
    defense holds either way; this is the earlier, cheaper of the two checks."""
    if received_issuer is None:
        return
    if received_issuer != expected_issuer:
        raise IssuerMismatchError(expected_issuer, received_issuer)


def validate_return_to(return_to: str | None) -> str | None:
    """Open-redirect defense (RFC 9700 §2.1).

    Only a same-site absolute PATH is allowed — `/dashboard`, never
    `https://evil.example/x`, and never `//evil.example` (which browsers treat
    as protocol-relative and is the classic bypass of a naive "starts with /"
    check). Backslashes are rejected too, since some parsers normalize `/\\` to
    `//`."""
    if return_to is None or return_to == "":
        return None
    if not return_to.startswith("/"):
        raise UnsafeRedirectError(return_to)
    if return_to.startswith("//") or return_to.startswith("/\\"):
        raise UnsafeRedirectError(return_to)
    if "\\" in return_to or "\n" in return_to or "\r" in return_to:
        raise UnsafeRedirectError(return_to)
    return return_to


# ── IO: discovery, token exchange, signature verification ─────────────────

class OidcClient:
    """The network/crypto half. Everything it does is delegated to established
    libraries (`httpx` for transport, `joserfc` for JWS/JWKS) rather than
    hand-rolled — this codebase does not implement crypto (`agents.md` #5), and
    JWS verification is exactly the kind of thing that looks fine and is subtly
    wrong.

    `joserfc` specifically, not `authlib.jose`: the latter is deprecated by
    Authlib itself and slated for removal, and `joserfc` is the maintainer's
    own successor.

    `http_client`/`jwks_resolver` are injectable so the orchestration in
    `service.py` can be tested end to end without a live IdP."""

    def __init__(self, *, timeout_seconds: float = 10.0, http_client=None, jwks_resolver=None):
        self._timeout = timeout_seconds
        self._http = http_client
        self._jwks_resolver = jwks_resolver
        self._metadata_cache: dict[str, dict] = {}

    # -- discovery --

    def discover(self, connection) -> dict:
        """Fetches and caches the IdP's OpenID configuration.

        Cached per issuer for the process lifetime: endpoints change rarely, and
        re-fetching on every login would put an IdP's availability directly on
        our login latency path. A rotated signing key is a different concern and
        is handled by the JWKS lookup, which is not cached here."""
        url = connection.discovery_url or discovery_url_for(connection.issuer)
        if url in self._metadata_cache:
            return self._metadata_cache[url]
        metadata = self._get_json(url)
        # The discovery document's own `issuer` must match what we have on file,
        # or a hijacked discovery URL could silently repoint a tenant's login.
        if metadata.get("issuer") != connection.issuer:
            raise IssuerMismatchError(connection.issuer, str(metadata.get("issuer")))
        for required in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not metadata.get(required):
                raise OidcProtocolError(f"discovery document is missing {required!r}")
        self._metadata_cache[url] = metadata
        return metadata

    # -- token exchange --

    def exchange_code(self, connection, *, code: str, redirect_uri: str, code_verifier: str) -> dict:
        """Confidential-client exchange: `client_secret` in the POST body along
        with the PKCE `code_verifier`. `redirect_uri` is re-sent because the IdP
        must confirm it matches the one in the authorization request — this is
        the server side of exact-match redirect validation."""
        metadata = self.discover(connection)
        response = self._post_form(metadata["token_endpoint"], {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": connection.client_id,
            "client_secret": connection.client_secret,
            "code_verifier": code_verifier,
        })
        if "id_token" not in response:
            # An OIDC token response without an ID token means we authenticated
            # nobody. Treated as a hard failure rather than falling back to the
            # userinfo endpoint, which would accept an access token that was
            # never proven to be about the person in front of us.
            raise OidcProtocolError("token response contained no id_token")
        return response

    # -- signature verification --

    def verify_id_token(self, connection, id_token: str) -> dict:
        """Verifies the JWS signature against the IdP's JWKS and returns the
        claims. Claim validation is the CALLER's next step
        (`validate_id_token_claims`) — kept separate so neither can be
        accidentally skipped by doing "the other one"."""
        metadata = self.discover(connection)
        jwks = self._resolve_jwks(metadata["jwks_uri"])

        from joserfc import jwt
        from joserfc.jwk import KeySet

        key_set = KeySet.import_key_set(jwks)
        # Algorithms are allowlisted explicitly. Accepting whatever the token's
        # own header names is the classic JWT flaw -- it lets an attacker pick
        # `none`, or downgrade an RSA key into an HMAC secret.
        token = jwt.decode(id_token, key_set, algorithms=["RS256", "RS384", "RS512", "ES256", "ES384"])
        return dict(token.claims)

    # -- transport helpers --

    def _resolve_jwks(self, jwks_uri: str) -> dict:
        if self._jwks_resolver is not None:
            return self._jwks_resolver(jwks_uri)
        return self._get_json(jwks_uri)

    def _get_json(self, url: str) -> dict:
        if self._http is not None:
            return self._http.get_json(url)
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def _post_form(self, url: str, data: dict) -> dict:
        if self._http is not None:
            return self._http.post_form(url, data)
        import httpx

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, data=data)
            response.raise_for_status()
            return response.json()

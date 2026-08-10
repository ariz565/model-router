"""The SSO HTTP surface: login, callback, logout, org switching, back-channel
logout, and per-tenant IdP configuration.

**Mounted only when SSO is configured.** `server.py` includes this router
exclusively when `app.state.sso is not None`, so with SSO off these routes don't
exist at all — a 404, not a 501 or a disabled-feature branch. That is what makes
the module genuinely removable: delete the directory and the two wiring lines in
`server.py` and nothing else changes.

**Every SSO failure renders as the same generic error.** `SsoError` covers
unknown state, expired transaction, nonce mismatch, issuer mismatch, refused
linking — and the user-facing message is identical for all of them, because the
caller is unauthenticated and telling them which validation rule they tripped
turns the callback into an oracle for probing it. The specific reason goes to
the server log. The two exceptions are `SsoEmailNotVerifiedError` and
`SsoNotConfiguredError`, which describe an OPERATOR misconfiguration the person
signing in cannot exploit and genuinely needs relayed to their admin.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from modelrouter.identity.authz import AuthzContext, Principal
from modelrouter.identity.roles import Permission
from modelrouter.identity.sso.oidc import SsoError, UnsafeRedirectError
from modelrouter.identity.sso.service import (
    SsoEmailNotVerifiedError,
    SsoNotConfiguredError,
    SsoService,
)
from modelrouter.identity.sso.sessions import COOKIE_KWARGS, SESSION_COOKIE_NAME

router = APIRouter(tags=["sso"])

_GENERIC_SSO_FAILURE = "sign-in failed; please try again or contact your administrator"


class ConfigureConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issuer: str = Field(min_length=8, max_length=512)
    client_id: str = Field(min_length=1, max_length=512)
    client_secret: str = Field(min_length=1, max_length=2048)
    discovery_url: str | None = None


class SwitchOrgRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)


class BackchannelLogoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logout_token: str = Field(min_length=1)


def _sso(request: Request) -> SsoService:
    service = getattr(request.app.state, "sso", None)
    if service is None:      # unreachable while mounting is conditional; a guard, not a branch
        raise HTTPException(status_code=404, detail="SSO is not configured")
    return service


def _require(permission: str):
    from modelrouter.server import require_authz

    return require_authz(permission)


def _fail(exc: SsoError) -> HTTPException:
    """One place that decides what an unauthenticated caller learns.

    Logged with the real reason, answered with a generic one — except for the two
    operator-misconfiguration cases, whose detail is actionable and not
    exploitable."""
    print(f"[modelrouter.sso] sign-in failed: {type(exc).__name__}: {exc}")
    if isinstance(exc, (SsoEmailNotVerifiedError, SsoNotConfiguredError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=_GENERIC_SSO_FAILURE)


# ── Login ─────────────────────────────────────────────────────────────────

@router.get("/auth/sso/login")
async def sso_login(
    request: Request, org: str | None = None, email: str | None = None,
    return_to: str | None = None,
):
    """Starts the flow and 302s to the IdP.

    Either `org` (an explicit tenant id, e.g. from an org picker) or `email`
    (whose domain must be VERIFIED by exactly one tenant) selects the
    connection. `redirect_uri` is taken from server configuration, never from
    this request — a caller-supplied redirect target is the open-redirect hole
    that exact-match registration exists to close.

    `status_code=302` rather than the default 307: this must become a plain GET
    at the IdP, and 307 preserves the method."""
    try:
        start = _sso(request).begin_login(
            redirect_uri=request.app.state.sso_redirect_uri,
            tenant_id=org, email=email, return_to=return_to,
        )
    except UnsafeRedirectError as e:
        raise HTTPException(status_code=400, detail="invalid return_to") from e
    except SsoError as e:
        raise _fail(e) from e
    return RedirectResponse(start.authorization_url, status_code=302)


@router.get("/auth/sso/callback")
async def sso_callback(
    request: Request, code: str | None = None, state: str | None = None,
    iss: str | None = None, error: str | None = None,
    error_description: str | None = None,
):
    """The IdP redirects here. Sets the session cookie and 302s onward.

    An `error` parameter (the user declined consent, the IdP refused) is handled
    before anything else — it is a normal outcome, not an exception, and there
    is no code to exchange.

    `iss` is forwarded to the service for the mix-up check. `code`/`state` are
    both required; a callback missing either is malformed rather than
    interesting."""
    if error:
        print(f"[modelrouter.sso] identity provider returned error={error!r} ({error_description!r})")
        raise HTTPException(status_code=400, detail=_GENERIC_SSO_FAILURE)
    if not code or not state:
        raise HTTPException(status_code=400, detail=_GENERIC_SSO_FAILURE)

    try:
        result = _sso(request).complete_login(state=state, code=code, received_issuer=iss)
    except SsoError as e:
        raise _fail(e) from e

    response = RedirectResponse(result.return_to or "/", status_code=302)
    # `max_age` deliberately matches the ABSOLUTE session lifetime, not the idle
    # one: the browser should keep presenting the cookie while the session could
    # still be valid, and the server decides idle expiry (which slides) on every
    # request. A cookie that expired on the idle schedule would sign people out
    # even while they were active.
    max_age = int((result.session.absolute_expires_at - result.session.created_at).total_seconds())
    response.set_cookie(
        SESSION_COOKIE_NAME, result.session_token, max_age=max_age, **COOKIE_KWARGS,
    )
    return response


# ── Session inspection, logout, org switching ─────────────────────────────

@router.get("/auth/sso/me")
async def sso_whoami(request: Request):
    """Who am I, in which org, with what permissions — what a UI needs on load.

    Deliberately session-only: an API key has no "me" (it's a machine), and
    answering for one would blur the two credential types that
    `require_principal` exists to keep distinct."""
    from modelrouter.server import require_principal

    principal: Principal = await require_principal(request.headers.get("Authorization"), request)
    if not principal.is_human:
        raise HTTPException(status_code=403, detail="this endpoint is for signed-in users")

    identity_repo = request.app.state.identity
    user = identity_repo.get_user(principal.subject_id)
    memberships = identity_repo.list_orgs_for_user(principal.subject_id)
    return {
        "user_id": principal.subject_id,
        "email": user.email if user else None,
        "name": user.name if user else None,
        "active_tenant_id": principal.tenant_id,
        "role": principal.role,
        # Every org this human belongs to -- the data an org switcher needs. Safe
        # to return because it is keyed on the authenticated user, never on a
        # tenant id the caller supplied.
        "organizations": [
            {"tenant_id": m.tenant_id, "role": m.role, "status": m.status} for m in memberships
        ],
    }


@router.post("/auth/sso/logout", status_code=204)
async def sso_logout(request: Request, response: Response):
    """Revokes the session server-side AND clears the cookie. Idempotent, and
    deliberately never an error: a logout that 401s because the session already
    expired is a worse experience than one that always succeeds, and there is
    nothing to protect — the desired end state is "not signed in" either way."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        _sso(request).logout(token)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return None


@router.post("/auth/sso/switch-org")
async def sso_switch_org(body: SwitchOrgRequest, request: Request):
    """Mints a NEW session for another org this user belongs to, and revokes the
    old one. Membership in the target is re-verified — being signed in to one org
    implies nothing about another."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="not signed in")
    try:
        result = _sso(request).switch_org(token, body.tenant_id)
    except SsoError as e:
        raise _fail(e) from e

    max_age = int((result.session.absolute_expires_at - result.session.created_at).total_seconds())
    response = Response(status_code=200)
    response.set_cookie(
        SESSION_COOKIE_NAME, result.session_token, max_age=max_age, **COOKIE_KWARGS,
    )
    response.headers["content-type"] = "application/json"
    response.body = b'{"switched": true}'
    return response


@router.post("/auth/sso/backchannel-logout")
async def sso_backchannel_logout(body: BackchannelLogoutRequest, request: Request):
    """OIDC back-channel logout, called by the IdP (not a browser) when a
    session ends on their side — an admin terminating it, or a user being
    deactivated. The only mechanism that covers those cases; front-channel
    logout only covers the user clicking "sign out" here.

    **Honest scope boundary:** the `logout_token` is a JWT that MUST be
    signature-verified against the IdP's JWKS before being acted on, and this
    endpoint does NOT do that yet — so it is registered but refuses to act,
    rather than trusting an unverified token to revoke sessions (which would
    itself be a denial-of-service primitive: anyone could log anyone out).
    Completing it means resolving which connection the token came from by its
    `iss`, verifying it, and checking the `events` claim — real work, deliberately
    not faked here."""
    raise HTTPException(
        status_code=501,
        detail=(
            "back-channel logout is registered but not yet verifying logout tokens; "
            "session revocation via DELETE /v1/orgs/me/members/{user_id} is available"
        ),
    )


# ── Per-tenant IdP configuration (owner only) ─────────────────────────────

@router.get("/v1/orgs/me/sso-connection")
async def get_sso_connection(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.SSO_MANAGE))],
):
    """The `client_secret` is never returned — not even to an owner. There is no
    legitimate read-back use (you re-enter it to rotate it), and an endpoint that
    echoes it turns any owner-level access into secret exfiltration."""
    connection = _sso(request)._sso.get_connection_for_tenant(ctx.scope.tenant_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="no SSO connection configured")
    return {
        "connection_id": connection.connection_id, "protocol": connection.protocol,
        "issuer": connection.issuer, "client_id": connection.client_id,
        "discovery_url": connection.discovery_url, "status": connection.status,
        "created_at": connection.created_at.isoformat() if connection.created_at else None,
    }


@router.put("/v1/orgs/me/sso-connection", status_code=201)
async def configure_sso_connection(
    body: ConfigureConnectionRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.SSO_MANAGE))],
):
    """`PUT`, not `POST` — configuring SSO is idempotent replacement: one org has
    exactly one active connection, and re-submitting replaces it (the previous
    row is disabled, not deleted, so the change is auditable and reversible).

    `sso:manage` is an OWNER-only permission: whoever controls the IdP connection
    controls who can sign in as anyone in the org, which is a strictly larger
    power than any admin capability."""
    connection = _sso(request)._sso.create_connection(
        ctx.scope.tenant_id, issuer=body.issuer, client_id=body.client_id,
        client_secret=body.client_secret, discovery_url=body.discovery_url,
    )
    return {
        "connection_id": connection.connection_id, "issuer": connection.issuer,
        "status": connection.status,
    }


@router.delete("/v1/orgs/me/sso-connection", status_code=204)
async def disable_sso_connection(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.SSO_MANAGE))],
):
    """Disables rather than deletes, so turning SSO off is reversible and leaves
    a trail. Existing sessions are NOT revoked here — disabling the connection
    stops new logins; ending current ones is a separate, deliberate action (see
    member removal), because those are genuinely different intentions."""
    _sso(request)._sso.disable_connection(ctx.scope.tenant_id)
    return None

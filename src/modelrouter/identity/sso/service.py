"""`SsoService` — the OIDC login flow, session lifecycle, and (most
importantly) the account-linking rules.

**The linking rules are the security core of this file.** Everything else here
is protocol plumbing; this is where a mistake becomes cross-tenant account
takeover. The rules, in the order they're applied on a callback:

1. **An existing identity wins outright.** If `(connection_id, sub)` is on
   file, that IS the user. Email is not consulted at all. This is the normal
   path for every returning user, and it is immune to any email claim the IdP
   sends.
2. **A first-time login requires a provably-verified email.** If the IdP does
   not assert `email_verified` (coerced fail-closed — see
   `oidc.coerce_email_verified`), the login is refused. Not downgraded, not
   logged-in-with-caveats: refused. Accepting an unverified email would let a
   tenant's IdP admin mint a login carrying any address they like.
3. **Linking to an EXISTING user additionally requires a verified domain.**
   Even with a verified email, attaching this IdP subject to an account that
   already exists is only safe if the tenant has PROVEN it controls that email's
   domain. Without that, tenant A's IdP could assert
   `email=ceo@tenant-b.com` and be handed tenant B's user record — which is
   exactly the published nOAuth attack class.
4. **A brand-new user is created only from a verified email.** This is what
   prevents email squatting: without it, a hostile IdP could pre-create a user
   row holding `ceo@bigcorp.com`, and a later legitimate invitation to that
   address would resolve to the squatter's record.

**Membership is separate from authentication.** Successfully proving who you
are does not decide which org you may act in. `jit_provisioning` (default on)
grants a least-privilege `viewer` membership on first login, which is the
correct semantics for org-owned SSO — if you can authenticate against Acme's
own IdP, you are an Acme person. With it off, only pre-invited users can sign
in, which is what an org wanting strict allow-listing configures.

**The session is bound to (user, tenant) at mint time** and the active tenant
is never read from a request. Switching orgs re-checks membership and mints a
NEW session.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modelrouter.identity.authz import Principal, session_principal
from modelrouter.identity.models import PROVIDER_OIDC, MEMBERSHIP_ACTIVE, User
from modelrouter.identity.ports import IdentityRepo
from modelrouter.identity.roles import ROLE_VIEWER
from modelrouter.identity.sso import sessions as sessions_module
from modelrouter.identity.sso.models import AuthTransaction, Session, SsoConnection
from modelrouter.identity.sso.oidc import (
    OidcClient,
    SsoError,
    assert_issuer_matches,
    build_authorization_url,
    coerce_email_verified,
    generate_nonce,
    generate_pkce_pair,
    generate_state,
    validate_id_token_claims,
    validate_return_to,
)
from modelrouter.identity.sso.ports import SsoRepo

__all__ = [
    "SsoService", "LoginStart", "LoginResult",
    "SsoNotConfiguredError", "SsoTransactionInvalidError",
    "SsoEmailNotVerifiedError", "SsoLinkRefusedError", "SsoAccessDeniedError",
]

DEFAULT_TRANSACTION_TTL_MINUTES = 10


class SsoNotConfiguredError(SsoError):
    """No active connection for the resolved tenant, or the tenant couldn't be
    resolved at all."""


class SsoTransactionInvalidError(SsoError):
    """Unknown, already-consumed, or expired `state`. One error for all three:
    telling an unauthenticated caller which it was is a probing oracle."""


class SsoEmailNotVerifiedError(SsoError):
    """The IdP did not assert a verified email on a first-time login. The fix is
    the IdP admin configuring the claim, so the message says so — this one is
    safe to surface because it's an operator misconfiguration, not a secret."""

    def __init__(self):
        super().__init__(
            "the identity provider did not assert a verified email address for this user; "
            "enable the email/email_verified claims on the SSO application"
        )


class SsoLinkRefusedError(SsoError):
    """An account with this email already exists, and this tenant has not
    verified it controls the domain — so linking is refused rather than risking
    a cross-tenant account takeover."""

    def __init__(self, email: str):
        self.email = email
        super().__init__(
            "refusing to link this identity provider account to an existing user; "
            "verify ownership of the email domain for this organization first"
        )


class SsoAccessDeniedError(SsoError):
    """Authentication succeeded but the person is not a member of this org and
    just-in-time provisioning is disabled."""


class LoginStart:
    __slots__ = ("authorization_url", "state", "tenant_id")

    def __init__(self, authorization_url: str, state: str, tenant_id: str):
        self.authorization_url = authorization_url
        self.state = state
        self.tenant_id = tenant_id


class LoginResult:
    __slots__ = ("session", "session_token", "user", "return_to")

    def __init__(self, session: Session, session_token: str, user: User, return_to: str | None):
        self.session = session
        self.session_token = session_token
        self.user = user
        self.return_to = return_to


class SsoService:
    def __init__(
        self, sso_repo: SsoRepo, identity_repo: IdentityRepo, *,
        oidc_client: OidcClient | None = None,
        jit_provisioning: bool = True,
        jit_default_role: str = ROLE_VIEWER,
        transaction_ttl_minutes: int = DEFAULT_TRANSACTION_TTL_MINUTES,
        idle_timeout_minutes: int = sessions_module.DEFAULT_IDLE_TIMEOUT_MINUTES,
        absolute_timeout_hours: int = sessions_module.DEFAULT_ABSOLUTE_TIMEOUT_HOURS,
        now_fn=None,
    ):
        self._sso = sso_repo
        self._identity = identity_repo
        self._oidc = oidc_client or OidcClient()
        self._jit = jit_provisioning
        self._jit_role = jit_default_role
        self._transaction_ttl = transaction_ttl_minutes
        self._idle_minutes = idle_timeout_minutes
        self._absolute_hours = absolute_timeout_hours
        self._now = now_fn or (lambda: datetime.now(timezone.utc))

    # ── Step 1: start the flow ────────────────────────────────────────────

    def begin_login(
        self, *, redirect_uri: str, tenant_id: str | None = None, email: str | None = None,
        return_to: str | None = None,
    ) -> LoginStart:
        """Either an explicit `tenant_id` (an org selector in the UI) or an
        `email` whose domain has been VERIFIED by exactly one tenant.

        `redirect_uri` is supplied by the CALLER (server.py builds it from its
        own configuration), never taken from the incoming request — a
        request-supplied redirect_uri is the open-redirect/token-theft hole that
        exact-match registration exists to close."""
        connection = self._resolve_connection(tenant_id=tenant_id, email=email)
        metadata = self._oidc.discover(connection)
        safe_return_to = validate_return_to(return_to)

        state = generate_state()
        nonce = generate_nonce()
        verifier, challenge = generate_pkce_pair()
        issued_at = self._now()

        self._sso.create_transaction(AuthTransaction(
            state=state, tenant_id=connection.tenant_id, connection_id=connection.connection_id,
            issuer=connection.issuer, nonce=nonce, code_verifier=verifier,
            redirect_uri=redirect_uri, return_to=safe_return_to,
            expires_at=issued_at + timedelta(minutes=self._transaction_ttl),
            created_at=issued_at,
        ))
        url = build_authorization_url(
            metadata["authorization_endpoint"], client_id=connection.client_id,
            redirect_uri=redirect_uri, state=state, nonce=nonce, code_challenge=challenge,
        )
        return LoginStart(url, state, connection.tenant_id)

    # ── Step 2: handle the callback ───────────────────────────────────────

    def complete_login(self, *, state: str, code: str, received_issuer: str | None = None) -> LoginResult:
        """Ordering here is security-relevant and not arbitrary:

        1. consume the transaction FIRST (atomic, single-use) so a replayed
           callback cannot be processed twice even concurrently;
        2. check its expiry;
        3. check the issuer BEFORE spending the code, so a mix-up attempt never
           reaches the token endpoint;
        4. exchange, verify the signature, then validate the claims;
        5. only then resolve the identity and mint a session."""
        transaction = self._sso.consume_transaction(state)
        if transaction is None:
            raise SsoTransactionInvalidError("unknown or already-used login state")
        if transaction.expires_at <= self._now():
            raise SsoTransactionInvalidError("this login attempt expired")

        connection = self._sso.get_connection(transaction.connection_id)
        if connection is None or not connection.is_active:
            raise SsoNotConfiguredError("the SSO connection for this login is no longer active")

        assert_issuer_matches(received_issuer, transaction.issuer)

        tokens = self._oidc.exchange_code(
            connection, code=code, redirect_uri=transaction.redirect_uri,
            code_verifier=transaction.code_verifier,
        )
        claims = self._oidc.verify_id_token(connection, tokens["id_token"])
        validate_id_token_claims(
            claims, expected_issuer=connection.issuer, expected_audience=connection.client_id,
            expected_nonce=transaction.nonce, now=self._now(),
        )

        user = self._resolve_user(connection, claims)
        self._ensure_membership(connection.tenant_id, user.user_id)

        record, token = sessions_module.build_session(
            user_id=user.user_id, tenant_id=connection.tenant_id, now=self._now(),
            idle_timeout_minutes=self._idle_minutes,
            absolute_timeout_hours=self._absolute_hours,
            idp_session_id=claims.get("sid"),
        )
        self._sso.create_session(record)
        return LoginResult(record, token, user, transaction.return_to)

    # ── Per-request authentication ────────────────────────────────────────

    def authenticate(self, session_token: str) -> tuple[Session, Principal] | None:
        """Resolves a cookie into a `(Session, Principal)`, or `None` for any
        reason at all — unknown, expired, revoked, or the membership is gone.

        **The membership is re-checked on EVERY request**, and the role is read
        fresh rather than from the session row. That is what makes "remove this
        person" and "demote this person" take effect immediately instead of
        whenever their session happens to expire. It costs one indexed lookup,
        which is the right price for revocation actually working."""
        token_hash = sessions_module.hash_session_token(session_token)
        session = self._sso.get_session_by_token_hash(token_hash)
        if session is None:
            return None

        now = self._now()
        if not session.is_valid_at(now):
            return None

        membership = self._identity.get_org_membership(session.tenant_id, session.user_id)
        if membership is None or not membership.is_active:
            # Their access was removed mid-session: revoke rather than merely
            # deny, so the dead session stops being retried on every request.
            self._sso.revoke_session(session.session_id)
            return None

        user = self._identity.get_user(session.user_id)
        if user is None or not user.is_active:
            self._sso.revoke_session(session.session_id)
            return None

        self._sso.touch_session(
            session.session_id,
            idle_expires_at=sessions_module.refreshed_idle_expiry(
                now, idle_timeout_minutes=self._idle_minutes,
            ),
            last_seen_at=now,
        )
        return session, session_principal(session, membership.role)

    # ── Logout and org switching ──────────────────────────────────────────

    def logout(self, session_token: str) -> None:
        """Idempotent — logging out twice, or with a token that was never valid,
        is not an error worth surfacing."""
        session = self._sso.get_session_by_token_hash(
            sessions_module.hash_session_token(session_token),
        )
        if session is not None:
            self._sso.revoke_session(session.session_id)

    def switch_org(self, session_token: str, target_tenant_id: str) -> LoginResult:
        """Mints a NEW session for another org the user belongs to, and revokes
        the old one.

        A new session rather than mutating the existing row's `tenant_id`: the
        session is the authority on which org the caller is acting in, so
        changing it in place would mean a token's meaning silently changes
        underneath anything holding it. Membership in the target org is
        re-verified here — being signed in to org A implies nothing about org B."""
        current = self._sso.get_session_by_token_hash(
            sessions_module.hash_session_token(session_token),
        )
        if current is None or not current.is_valid_at(self._now()):
            raise SsoTransactionInvalidError("not signed in")

        membership = self._identity.get_org_membership(target_tenant_id, current.user_id)
        if membership is None or not membership.is_active:
            raise SsoAccessDeniedError("not a member of that organization")

        user = self._identity.get_user(current.user_id)
        if user is None or not user.is_active:
            raise SsoAccessDeniedError("this account is not active")

        record, token = sessions_module.build_session(
            user_id=current.user_id, tenant_id=target_tenant_id, now=self._now(),
            idle_timeout_minutes=self._idle_minutes,
            absolute_timeout_hours=self._absolute_hours,
            idp_session_id=current.idp_session_id,
        )
        self._sso.create_session(record)
        self._sso.revoke_session(current.session_id)
        return LoginResult(record, token, user, None)

    def handle_backchannel_logout(self, idp_session_id: str) -> int:
        """OIDC back-channel logout. Returns how many sessions were revoked.

        This is the only path that works when an admin terminates a session at
        the IdP or deactivates a user there — front-channel logout only covers
        the user clicking "sign out" in our own UI."""
        return self._sso.revoke_sessions_for_idp_session(idp_session_id)

    def revoke_user_sessions(self, user_id: str, *, tenant_id: str | None = None) -> int:
        """Called when someone is removed from an org (scoped) or deactivated
        entirely (`tenant_id=None`)."""
        return self._sso.revoke_sessions_for_user(user_id, tenant_id=tenant_id)

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_connection(self, *, tenant_id: str | None, email: str | None) -> SsoConnection:
        if tenant_id is not None:
            connection = self._sso.get_connection_for_tenant(tenant_id)
            if connection is None or not connection.is_active:
                raise SsoNotConfiguredError("this organization has no active SSO connection")
            return connection

        if email is None:
            raise SsoNotConfiguredError("either an organization or an email address is required")

        _, _, domain = email.strip().lower().partition("@")
        if not domain:
            raise SsoNotConfiguredError("that does not look like an email address")
        record = self._identity.get_domain(domain)
        # An UNVERIFIED domain claim must never route a login: otherwise anyone
        # could claim a domain, attach their own IdP, and receive logins intended
        # for the real owner of that domain.
        if record is None or not record.is_verified:
            raise SsoNotConfiguredError("no organization has verified that email domain")
        connection = self._sso.get_connection_for_tenant(record.tenant_id)
        if connection is None or not connection.is_active:
            raise SsoNotConfiguredError("this organization has no active SSO connection")
        return connection

    def _resolve_user(self, connection: SsoConnection, claims: dict) -> User:
        """See this module's docstring for the four linking rules; this is their
        implementation, in the same order."""
        subject = claims["sub"]

        # Rule 1: an existing identity wins outright. Email is not consulted.
        identity = self._identity.get_identity(connection.connection_id, subject)
        if identity is not None:
            user = self._identity.get_user(identity.user_id)
            if user is None or not user.is_active:
                raise SsoAccessDeniedError("this account is not active")
            self._identity.touch_identity(identity.identity_id)
            return user

        # Rule 2: a first-time login requires a verified email.
        email = claims.get("email")
        email_trusted = coerce_email_verified(claims.get("email_verified"))
        if not isinstance(email, str) or not email.strip() or not email_trusted:
            raise SsoEmailNotVerifiedError()
        normalized = email.strip().lower()

        existing = self._identity.get_user_by_email(normalized)
        if existing is not None:
            # Rule 3: linking to an existing account also requires that this
            # tenant has proven it controls the email's domain.
            if not self._tenant_owns_domain(connection.tenant_id, normalized):
                raise SsoLinkRefusedError(normalized)
            user = existing
        else:
            # Rule 4: a brand-new user, from a verified email only.
            user = self._identity.create_user(
                normalized, name=claims.get("name"), email_verified_at=self._now(),
            )

        self._identity.create_identity(
            user.user_id, connection_id=connection.connection_id, provider=PROVIDER_OIDC,
            provider_subject=subject, email=normalized, email_trusted=True,
        )
        return user

    def _tenant_owns_domain(self, tenant_id: str, email: str) -> bool:
        _, _, domain = email.partition("@")
        if not domain:
            return False
        record = self._identity.get_domain(domain)
        return record is not None and record.is_verified and record.tenant_id == tenant_id

    def _ensure_membership(self, tenant_id: str, user_id: str) -> None:
        membership = self._identity.get_org_membership(tenant_id, user_id)
        if membership is not None and membership.is_active:
            return
        if membership is not None:
            # A row exists but isn't active (suspended). JIT must NOT quietly
            # reactivate it -- someone suspended this person on purpose.
            raise SsoAccessDeniedError("this membership is not active")
        if not self._jit:
            raise SsoAccessDeniedError("no membership in this organization")
        self._identity.create_org_membership(
            tenant_id, user_id, self._jit_role, status=MEMBERSHIP_ACTIVE,
        )

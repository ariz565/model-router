"""identity/sso/ — the OIDC flow end to end against a fake IdP, with the
attack classes as first-class test subjects.

No network, no `httpx`, no `joserfc`: `OidcClient` is subclassed to stub the two
IO methods (`discover` is real logic over injected JSON; `verify_id_token`
stands in for JWS verification). That split is deliberate in `oidc.py` — the
signature check needs a crypto library, but every rule an attacker actually
targets (nonce, issuer, audience, expiry, email trust, linking, single-use
state) is pure logic and is exercised for real here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modelrouter.identity.memory import InMemoryIdentityRepo
from modelrouter.identity.roles import ROLE_ADMIN, ROLE_OWNER, ROLE_VIEWER
from modelrouter.identity.sso.memory import InMemorySsoRepo
from modelrouter.identity.sso.oidc import (
    CLOCK_SKEW_LEEWAY_SECONDS,
    IdTokenInvalidError,
    IssuerMismatchError,
    OidcClient,
    UnsafeRedirectError,
    assert_issuer_matches,
    build_authorization_url,
    coerce_email_verified,
    generate_pkce_pair,
    validate_id_token_claims,
    validate_return_to,
)
from modelrouter.identity.sso.service import (
    SsoAccessDeniedError,
    SsoEmailNotVerifiedError,
    SsoLinkRefusedError,
    SsoNotConfiguredError,
    SsoService,
    SsoTransactionInvalidError,
)
from modelrouter.identity.sso.sessions import (
    COOKIE_KWARGS,
    SESSION_COOKIE_NAME,
    hash_session_token,
)

TENANT_A = "tn_a"
TENANT_B = "tn_b"
ISSUER = "https://idp.acme.example"
CLIENT_ID = "mr-client"
REDIRECT_URI = "https://gateway.example.com/auth/sso/callback"


# ── PKCE ──────────────────────────────────────────────────────────────────

def test_pkce_challenge_is_the_s256_of_the_verifier():
    import base64
    import hashlib

    verifier, challenge = generate_pkce_pair()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    assert challenge == expected
    assert "=" not in challenge          # base64url, unpadded, per the RFC


def test_pkce_pairs_are_unique_per_call():
    pairs = {generate_pkce_pair()[0] for _ in range(100)}
    assert len(pairs) == 100


# ── Authorization URL ─────────────────────────────────────────────────────

def test_authorization_url_carries_every_required_parameter():
    url = build_authorization_url(
        f"{ISSUER}/authorize", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI,
        state="st", nonce="no", code_challenge="ch",
    )
    for fragment in ("response_type=code", "client_id=mr-client", "state=st",
                     "nonce=no", "code_challenge=ch", "code_challenge_method=S256"):
        assert fragment in url


def test_authorization_url_always_uses_s256_never_plain():
    url = build_authorization_url(
        f"{ISSUER}/authorize", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI,
        state="s", nonce="n", code_challenge="c",
    )
    assert "code_challenge_method=S256" in url
    assert "plain" not in url


def test_authorization_url_appends_correctly_to_an_endpoint_with_a_query():
    url = build_authorization_url(
        f"{ISSUER}/authorize?tenant=x", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI,
        state="s", nonce="n", code_challenge="c",
    )
    assert "?tenant=x&response_type=code" in url


# ── email_verified coercion: the fail-closed rule ────────────────────────

@pytest.mark.parametrize("raw", [True, "true", "True", "TRUE", "1", 1])
def test_email_verified_accepts_only_unambiguous_truth(raw):
    assert coerce_email_verified(raw) is True


@pytest.mark.parametrize("raw", [
    False, "false", "False", "0", 0, None, "", "yes", "maybe", [], {}, object(), 2, -1,
])
def test_email_verified_fails_closed_for_everything_else(raw):
    """The published vulnerability this guards: a bool type-assertion that
    treated the STRING "false" (or an absent claim) as verified, yielding full
    account takeover. `"yes"` is included deliberately — plausible-looking but
    not something any spec defines, so it must not be trusted."""
    assert coerce_email_verified(raw) is False


# ── ID token claim validation ─────────────────────────────────────────────

def _claims(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    base = {
        "iss": ISSUER, "aud": CLIENT_ID, "sub": "idp-subject-1", "nonce": "the-nonce",
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "iat": int(now.timestamp()),
        "email": "alice@acme.example", "email_verified": True,
    }
    base.update(overrides)
    return base


def _validate(**overrides):
    validate_id_token_claims(
        _claims(**overrides), expected_issuer=ISSUER, expected_audience=CLIENT_ID,
        expected_nonce="the-nonce",
    )


def test_a_well_formed_token_validates():
    _validate()


def test_a_wrong_issuer_is_rejected():
    with pytest.raises(IssuerMismatchError):
        _validate(iss="https://evil.example")


def test_a_wrong_audience_is_rejected():
    with pytest.raises(IdTokenInvalidError):
        _validate(aud="some-other-client")


def test_a_list_audience_containing_us_is_accepted():
    _validate(aud=["another-client", CLIENT_ID])


def test_an_azp_that_is_not_us_is_rejected():
    """A token minted for a DIFFERENT client that merely lists us as an
    audience must not be accepted as our own."""
    with pytest.raises(IdTokenInvalidError):
        _validate(aud=[CLIENT_ID, "other"], azp="other")


def test_a_mismatched_nonce_is_rejected():
    with pytest.raises(IdTokenInvalidError):
        _validate(nonce="a-replayed-nonce")


def test_a_missing_nonce_is_rejected():
    claims = _claims()
    del claims["nonce"]
    with pytest.raises(IdTokenInvalidError):
        validate_id_token_claims(
            claims, expected_issuer=ISSUER, expected_audience=CLIENT_ID,
            expected_nonce="the-nonce",
        )


def test_an_expired_token_is_rejected():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    with pytest.raises(IdTokenInvalidError):
        _validate(exp=int(past.timestamp()))


def test_a_token_within_the_clock_skew_leeway_is_still_accepted():
    """Small clock drift between us and an IdP is normal; failing logins over a
    two-second skew would be a self-inflicted outage."""
    just_expired = datetime.now(timezone.utc) - timedelta(seconds=CLOCK_SKEW_LEEWAY_SECONDS // 2)
    _validate(exp=int(just_expired.timestamp()))


def test_a_token_issued_far_in_the_future_is_rejected():
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(IdTokenInvalidError):
        _validate(iat=int(future.timestamp()))


def test_a_missing_or_empty_sub_is_rejected():
    """`sub` is the only key used to resolve an identity, so a token without one
    cannot be turned into a login at all."""
    with pytest.raises(IdTokenInvalidError):
        _validate(sub="")
    claims = _claims()
    del claims["sub"]
    with pytest.raises(IdTokenInvalidError):
        validate_id_token_claims(
            claims, expected_issuer=ISSUER, expected_audience=CLIENT_ID,
            expected_nonce="the-nonce",
        )


def test_non_numeric_exp_or_iat_is_rejected():
    with pytest.raises(IdTokenInvalidError):
        _validate(exp="soon")
    with pytest.raises(IdTokenInvalidError):
        _validate(iat=None)


# ── Mix-up defense ────────────────────────────────────────────────────────

def test_issuer_mismatch_on_the_callback_is_rejected():
    with pytest.raises(IssuerMismatchError):
        assert_issuer_matches("https://attacker-idp.example", ISSUER)


def test_a_matching_callback_issuer_passes():
    assert_issuer_matches(ISSUER, ISSUER)


def test_an_absent_callback_issuer_is_tolerated():
    """IdPs predating RFC 9207 don't send `iss`; the ID token's own claim is
    still validated, so the defense holds either way."""
    assert_issuer_matches(None, ISSUER)


# ── Open-redirect defense ─────────────────────────────────────────────────

def test_a_relative_path_return_to_is_allowed():
    assert validate_return_to("/dashboard") == "/dashboard"


def test_none_and_empty_return_to_are_fine():
    assert validate_return_to(None) is None
    assert validate_return_to("") is None


@pytest.mark.parametrize("value", [
    "https://evil.example/x",
    "//evil.example",          # protocol-relative: the classic "starts with /" bypass
    "/\\evil.example",         # some parsers normalize this to //
    "/path\\with\\backslash",
    "/path\nSet-Cookie: x=y",  # header injection via CRLF
    "/path\rlocation: y",
    "javascript:alert(1)",
])
def test_unsafe_return_to_values_are_refused(value):
    with pytest.raises(UnsafeRedirectError):
        validate_return_to(value)


# ── End-to-end flow against a fake IdP ────────────────────────────────────

class _FakeOidcClient(OidcClient):
    """Stubs only the two IO methods. Everything else — URL construction, claim
    validation, the linking rules — is the real implementation."""

    def __init__(self, *, claims: dict | None = None, issuer: str = ISSUER):
        super().__init__()
        self._issuer = issuer
        self._claims = claims
        self.exchanges: list[dict] = []

    def discover(self, connection) -> dict:
        return {
            "issuer": self._issuer,
            "authorization_endpoint": f"{self._issuer}/authorize",
            "token_endpoint": f"{self._issuer}/token",
            "jwks_uri": f"{self._issuer}/jwks",
        }

    def exchange_code(self, connection, *, code, redirect_uri, code_verifier) -> dict:
        self.exchanges.append({
            "code": code, "redirect_uri": redirect_uri, "code_verifier": code_verifier,
        })
        return {"id_token": "fake.jwt.token", "token_type": "Bearer"}

    def verify_id_token(self, connection, id_token) -> dict:
        return dict(self._claims or {})


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now = self.now + timedelta(**kw)


def _env(*, claims: dict | None = None, jit: bool = True, issuer: str = ISSUER):
    identity = InMemoryIdentityRepo()
    sso = InMemorySsoRepo()
    clock = _Clock()
    client = _FakeOidcClient(claims=claims, issuer=issuer)
    service = SsoService(
        sso, identity, oidc_client=client, jit_provisioning=jit, now_fn=clock,
    )
    connection = sso.create_connection(
        TENANT_A, issuer=issuer, client_id=CLIENT_ID, client_secret="secret",
    )
    return identity, sso, service, client, clock, connection


def _flow_claims(clock: _Clock, nonce: str, **overrides) -> dict:
    now = clock()
    base = {
        "iss": ISSUER, "aud": CLIENT_ID, "sub": "idp-subject-1", "nonce": nonce,
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "iat": int(now.timestamp()),
        "email": "alice@acme.example", "email_verified": True, "name": "Alice",
    }
    base.update(overrides)
    return base


def _complete(service, sso, client, clock, state, **claim_overrides):
    transaction = sso._transactions[state]
    client._claims = _flow_claims(clock, transaction.nonce, **claim_overrides)
    return service.complete_login(state=state, code="the-code")


def test_a_full_login_creates_a_user_a_membership_and_a_session():
    identity, sso, service, client, clock, _connection = _env()

    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    assert start.authorization_url.startswith(f"{ISSUER}/authorize?")
    assert start.tenant_id == TENANT_A

    result = _complete(service, sso, client, clock, start.state)

    assert result.user.email == "alice@acme.example"
    assert result.session.tenant_id == TENANT_A
    # JIT provisioning grants least privilege, not the role of whoever set it up.
    assert identity.get_org_membership(TENANT_A, result.user.user_id).role == ROLE_VIEWER
    # The identity is keyed on the IdP subject, and the email is recorded as
    # trusted because the IdP asserted email_verified.
    stored = identity.get_identity(_connection_id(sso), "idp-subject-1")
    assert stored is not None and stored.email_trusted is True


def _connection_id(sso) -> str:
    return next(iter(sso._connections))


def test_the_code_exchange_sends_the_pkce_verifier_and_the_registered_redirect_uri():
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    _complete(service, sso, client, clock, start.state)

    exchange = client.exchanges[0]
    assert exchange["redirect_uri"] == REDIRECT_URI
    assert exchange["code_verifier"]                       # PKCE actually used
    assert exchange["code"] == "the-code"


def test_a_returning_user_reuses_the_same_user_and_does_not_duplicate_the_identity():
    identity, sso, service, client, clock, _connection = _env()
    first_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    first = _complete(service, sso, client, clock, first_start.state)

    second_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    second = _complete(service, sso, client, clock, second_start.state)

    assert first.user.user_id == second.user.user_id
    assert first.session.session_id != second.session.session_id   # fresh session each login


def test_a_returning_user_logs_in_even_if_the_idp_stops_sending_email():
    """Rule 1: once an identity exists, the email claim is not consulted at all
    — so an IdP that later omits or changes it cannot affect who this is."""
    _identity, sso, service, client, clock, _connection = _env()
    first_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    first = _complete(service, sso, client, clock, first_start.state)

    second_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    transaction = sso._transactions[second_start.state]
    claims = _flow_claims(clock, transaction.nonce)
    del claims["email"]
    del claims["email_verified"]
    client._claims = claims
    second = service.complete_login(state=second_start.state, code="c")

    assert second.user.user_id == first.user.user_id


# ── State/transaction handling ────────────────────────────────────────────

def test_a_state_can_only_be_used_once():
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    _complete(service, sso, client, clock, start.state)

    with pytest.raises(SsoTransactionInvalidError):
        service.complete_login(state=start.state, code="the-code")


def test_an_unknown_state_is_refused():
    _identity, _sso, service, _client, _clock, _connection = _env()
    with pytest.raises(SsoTransactionInvalidError):
        service.complete_login(state="never-issued", code="c")


def test_an_expired_transaction_is_refused():
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    clock.advance(minutes=11)          # default transaction TTL is 10 minutes

    with pytest.raises(SsoTransactionInvalidError):
        _complete(service, sso, client, clock, start.state)


def test_a_mix_up_callback_is_refused_before_the_code_is_spent():
    """The code must never reach the token endpoint if the issuer is wrong."""
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(IssuerMismatchError):
        service.complete_login(
            state=start.state, code="c", received_issuer="https://attacker-idp.example",
        )
    assert client.exchanges == []      # nothing was exchanged


def test_a_replayed_nonce_is_refused():
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    client._claims = _flow_claims(clock, "a-different-nonce")

    with pytest.raises(IdTokenInvalidError):
        service.complete_login(state=start.state, code="c")


def test_purging_expired_transactions_removes_only_the_stale_ones():
    _identity, sso, service, _client, clock, _connection = _env()
    service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    clock.advance(minutes=11)
    fresh = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    removed = sso.purge_expired_transactions(before=clock())

    assert removed == 1
    assert fresh.state in sso._transactions


# ── The linking rules ─────────────────────────────────────────────────────

def test_a_first_login_without_a_verified_email_is_refused():
    """Rule 2. Refused, not downgraded — accepting an unverified email would let
    a tenant's IdP admin mint a login carrying any address they like."""
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoEmailNotVerifiedError):
        _complete(service, sso, client, clock, start.state, email_verified=False)


def test_a_first_login_with_email_verified_as_the_string_false_is_refused():
    """The exact shape of the published CVE: a string, not a bool."""
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoEmailNotVerifiedError):
        _complete(service, sso, client, clock, start.state, email_verified="false")


def test_linking_to_an_existing_user_is_refused_without_a_verified_domain():
    """Rule 3 — the nOAuth attack class. Tenant A's IdP asserts an email that
    already belongs to an existing account; without proof that tenant A controls
    that domain, linking is refused rather than handing over the account."""
    identity, sso, service, client, clock, _connection = _env()
    victim = identity.create_user("ceo@bigcorp.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoLinkRefusedError):
        _complete(service, sso, client, clock, start.state, email="ceo@bigcorp.example")

    # The victim's account is untouched: no identity was attached to it.
    assert identity.get_identity(_connection_id(sso), "idp-subject-1") is None
    assert identity.get_org_membership(TENANT_A, victim.user_id) is None


def test_linking_to_an_existing_user_succeeds_with_a_verified_domain():
    identity, sso, service, client, clock, _connection = _env()
    existing = identity.create_user("alice@acme.example")
    identity.add_domain(TENANT_A, "acme.example", "tok")
    identity.mark_domain_verified(TENANT_A, "acme.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    result = _complete(service, sso, client, clock, start.state)

    assert result.user.user_id == existing.user_id


def test_another_tenants_verified_domain_does_not_permit_linking():
    identity, sso, service, client, clock, _connection = _env()
    identity.create_user("alice@acme.example")
    identity.add_domain(TENANT_B, "acme.example", "tok")     # verified by the WRONG tenant
    identity.mark_domain_verified(TENANT_B, "acme.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoLinkRefusedError):
        _complete(service, sso, client, clock, start.state)


def test_the_same_human_at_two_tenants_is_one_user_with_two_identities():
    """The consultant case, and the reason identities are scoped to a
    connection rather than users being scoped to a tenant."""
    identity, sso, service, client, clock, _connection = _env()
    identity.add_domain(TENANT_A, "acme.example", "t")
    identity.mark_domain_verified(TENANT_A, "acme.example")
    first_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    first = _complete(service, sso, client, clock, first_start.state)

    # A second tenant, its own IdP connection, the same human's email -- and
    # that tenant has also verified the domain.
    second_connection = sso.create_connection(
        TENANT_B, issuer=ISSUER, client_id=CLIENT_ID, client_secret="s2",
    )
    identity.add_domain(TENANT_B, "acme.example", "t")
    identity.mark_domain_verified(TENANT_B, "acme.example")
    second_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_B)
    transaction = sso._transactions[second_start.state]
    client._claims = _flow_claims(clock, transaction.nonce, sub="other-idp-subject")
    second = service.complete_login(state=second_start.state, code="c")

    assert first.user.user_id == second.user.user_id          # ONE human
    assert second.session.tenant_id == TENANT_B
    assert identity.get_identity(second_connection.connection_id, "other-idp-subject") is not None


# ── Membership gating ─────────────────────────────────────────────────────

def test_with_jit_disabled_a_non_member_cannot_sign_in():
    _identity, sso, service, client, clock, _connection = _env(jit=False)
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoAccessDeniedError):
        _complete(service, sso, client, clock, start.state)


def test_with_jit_disabled_a_pre_invited_member_can_sign_in():
    identity, sso, service, client, clock, _connection = _env(jit=False)
    invited = identity.create_user("alice@acme.example")
    identity.create_org_membership(TENANT_A, invited.user_id, ROLE_ADMIN)
    identity.add_domain(TENANT_A, "acme.example", "t")
    identity.mark_domain_verified(TENANT_A, "acme.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    result = _complete(service, sso, client, clock, start.state)
    assert result.user.user_id == invited.user_id


def test_a_suspended_membership_is_not_silently_reactivated_by_jit():
    """Someone suspended this person on purpose; JIT must not undo that."""
    identity, sso, service, client, clock, _connection = _env(jit=True)
    user = identity.create_user("alice@acme.example")
    identity.create_org_membership(TENANT_A, user.user_id, ROLE_ADMIN)
    identity.set_org_membership_status(TENANT_A, user.user_id, "suspended")
    identity.add_domain(TENANT_A, "acme.example", "t")
    identity.mark_domain_verified(TENANT_A, "acme.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)

    with pytest.raises(SsoAccessDeniedError):
        _complete(service, sso, client, clock, start.state)


def test_a_deactivated_user_cannot_sign_in():
    identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    first = _complete(service, sso, client, clock, start.state)
    identity.deactivate_user(first.user.user_id)

    second_start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    with pytest.raises(SsoAccessDeniedError):
        _complete(service, sso, client, clock, second_start.state)


# ── Home-realm discovery ──────────────────────────────────────────────────

def test_login_by_email_domain_requires_a_verified_domain():
    identity, _sso, service, _client, _clock, _connection = _env()
    identity.add_domain(TENANT_A, "acme.example", "tok")     # claimed, NOT verified

    with pytest.raises(SsoNotConfiguredError):
        service.begin_login(redirect_uri=REDIRECT_URI, email="alice@acme.example")

    identity.mark_domain_verified(TENANT_A, "acme.example")
    start = service.begin_login(redirect_uri=REDIRECT_URI, email="alice@acme.example")
    assert start.tenant_id == TENANT_A


def test_login_with_neither_tenant_nor_email_is_refused():
    _identity, _sso, service, _client, _clock, _connection = _env()
    with pytest.raises(SsoNotConfiguredError):
        service.begin_login(redirect_uri=REDIRECT_URI)


def test_login_for_a_tenant_without_a_connection_is_refused():
    _identity, _sso, service, _client, _clock, _connection = _env()
    with pytest.raises(SsoNotConfiguredError):
        service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_B)


def test_a_disabled_connection_stops_new_logins():
    _identity, sso, service, _client, _clock, _connection = _env()
    sso.disable_connection(TENANT_A)
    with pytest.raises(SsoNotConfiguredError):
        service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)


def test_an_unsafe_return_to_is_rejected_at_login_start():
    _identity, _sso, service, _client, _clock, _connection = _env()
    with pytest.raises(UnsafeRedirectError):
        service.begin_login(
            redirect_uri=REDIRECT_URI, tenant_id=TENANT_A, return_to="https://evil.example",
        )


def test_a_safe_return_to_survives_the_round_trip():
    _identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(
        redirect_uri=REDIRECT_URI, tenant_id=TENANT_A, return_to="/projects/api",
    )
    result = _complete(service, sso, client, clock, start.state)
    assert result.return_to == "/projects/api"


# ── Session lifecycle ─────────────────────────────────────────────────────

def _logged_in(**kw):
    identity, sso, service, client, clock, connection = _env(**kw)
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    result = _complete(service, sso, client, clock, start.state)
    return identity, sso, service, client, clock, result


def test_authenticate_resolves_a_session_into_a_principal():
    _identity, _sso, service, _client, _clock, result = _logged_in()
    authenticated = service.authenticate(result.session_token)

    assert authenticated is not None
    session, principal = authenticated
    assert principal.kind == "user_session"
    assert principal.tenant_id == TENANT_A
    assert principal.subject_id == result.user.user_id
    assert principal.role == ROLE_VIEWER
    assert principal.session_id == session.session_id


def test_only_the_token_hash_is_stored_never_the_token():
    _identity, sso, _service, _client, _clock, result = _logged_in()
    stored = sso.get_session_by_token_hash(hash_session_token(result.session_token))
    assert stored is not None
    assert stored.token_hash != result.session_token


def test_an_unknown_token_authenticates_as_none():
    _identity, _sso, service, _client, _clock, _result = _logged_in()
    assert service.authenticate("not-a-real-token") is None


def test_the_role_is_read_fresh_so_a_promotion_takes_effect_immediately():
    """The reason the role is not cached on the session row."""
    identity, _sso, service, _client, _clock, result = _logged_in()
    identity.set_org_role(TENANT_A, result.user.user_id, ROLE_OWNER)

    _session, principal = service.authenticate(result.session_token)
    assert principal.role == ROLE_OWNER


def test_removing_the_member_mid_session_immediately_kills_the_session():
    """Revocation actually working is the whole reason sessions are
    server-side rather than JWTs."""
    identity, _sso, service, _client, _clock, result = _logged_in()
    identity.remove_org_membership(TENANT_A, result.user.user_id)

    assert service.authenticate(result.session_token) is None


def test_a_session_killed_by_removal_is_revoked_not_merely_denied():
    identity, sso, service, _client, _clock, result = _logged_in()
    identity.remove_org_membership(TENANT_A, result.user.user_id)
    service.authenticate(result.session_token)

    assert sso.get_session_by_token_hash(result.session.token_hash).revoked_at is not None


def test_deactivating_the_user_mid_session_kills_the_session():
    identity, _sso, service, _client, _clock, result = _logged_in()
    identity.deactivate_user(result.user.user_id)
    assert service.authenticate(result.session_token) is None


def test_the_idle_timeout_expires_an_unused_session():
    _identity, _sso, service, _client, clock, result = _logged_in()
    clock.advance(minutes=31)          # default idle timeout is 30 minutes
    assert service.authenticate(result.session_token) is None


def test_using_a_session_slides_the_idle_timeout_forward():
    _identity, _sso, service, _client, clock, result = _logged_in()
    for _ in range(5):
        clock.advance(minutes=20)
        assert service.authenticate(result.session_token) is not None


def test_the_absolute_timeout_expires_even_a_continuously_used_session():
    """Idle-only expiry would let a stolen token be kept alive forever."""
    _identity, _sso, service, _client, clock, result = _logged_in()
    for _ in range(40):
        clock.advance(minutes=20)
        service.authenticate(result.session_token)
    assert service.authenticate(result.session_token) is None


def test_logout_revokes_the_session_and_is_idempotent():
    _identity, _sso, service, _client, _clock, result = _logged_in()
    service.logout(result.session_token)
    assert service.authenticate(result.session_token) is None
    service.logout(result.session_token)                 # second call: no error
    service.logout("never-existed")


def test_revoking_a_users_sessions_is_scoped_to_one_tenant_when_asked():
    identity, _sso, service, _client, _clock, result = _logged_in()
    identity.create_org_membership(TENANT_B, result.user.user_id, ROLE_VIEWER)
    other = service.switch_org(result.session_token, TENANT_B)

    service.revoke_user_sessions(result.user.user_id, tenant_id=TENANT_B)

    assert service.authenticate(other.session_token) is None


def test_back_channel_logout_revokes_every_session_for_that_idp_session():
    identity, sso, service, client, clock, _connection = _env()
    start = service.begin_login(redirect_uri=REDIRECT_URI, tenant_id=TENANT_A)
    result = _complete(service, sso, client, clock, start.state, sid="idp-sid-1")
    assert result.session.idp_session_id == "idp-sid-1"

    revoked = service.handle_backchannel_logout("idp-sid-1")

    assert revoked == 1
    assert service.authenticate(result.session_token) is None


def test_back_channel_logout_for_an_unknown_sid_revokes_nothing():
    _identity, _sso, service, _client, _clock, _result = _logged_in()
    assert service.handle_backchannel_logout("no-such-sid") == 0


# ── Org switching ─────────────────────────────────────────────────────────

def test_switching_org_mints_a_new_session_and_revokes_the_old_one():
    identity, _sso, service, _client, _clock, result = _logged_in()
    identity.create_org_membership(TENANT_B, result.user.user_id, ROLE_ADMIN)

    switched = service.switch_org(result.session_token, TENANT_B)

    assert switched.session.tenant_id == TENANT_B
    assert switched.session_token != result.session_token
    assert service.authenticate(result.session_token) is None      # old one is dead
    _session, principal = service.authenticate(switched.session_token)
    assert principal.tenant_id == TENANT_B
    assert principal.role == ROLE_ADMIN


def test_switching_to_an_org_you_do_not_belong_to_is_refused():
    """Being signed in to org A must imply nothing about org B."""
    _identity, _sso, service, _client, _clock, result = _logged_in()
    with pytest.raises(SsoAccessDeniedError):
        service.switch_org(result.session_token, TENANT_B)


def test_switching_with_an_invalid_session_is_refused():
    _identity, _sso, service, _client, _clock, _result = _logged_in()
    with pytest.raises(SsoTransactionInvalidError):
        service.switch_org("not-a-session", TENANT_A)


# ── Cookie policy ─────────────────────────────────────────────────────────

def test_the_cookie_name_uses_the_host_prefix_and_the_attributes_it_requires():
    """`__Host-` makes the browser enforce Secure + no Domain + Path=/ rather
    than trusting us to remember them."""
    assert SESSION_COOKIE_NAME.startswith("__Host-")
    assert COOKIE_KWARGS["secure"] is True
    assert COOKIE_KWARGS["httponly"] is True
    assert COOKIE_KWARGS["path"] == "/"
    assert "domain" not in COOKIE_KWARGS


def test_samesite_is_lax_not_strict():
    """Strict would drop the cookie on the cross-site top-level navigation back
    from the IdP, breaking login outright."""
    assert COOKIE_KWARGS["samesite"] == "lax"

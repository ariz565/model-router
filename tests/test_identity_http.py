"""End-to-end HTTP tests for the identity + SSO surface: real registration, real
invitations, real RBAC enforcement, real cross-tenant isolation — through the
actual FastAPI app, not the service layer.

The whole file is gated on `fastapi`, which is an optional dependency of this
project (`pip install -e ".[server]"`). Where it's absent these skip; where it's
present they exercise the same code a deployment runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient   # noqa: E402

from modelrouter.identity.roles import (   # noqa: E402
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
)


@pytest.fixture
def client(monkeypatch):
    """A fresh app per test. `MODELROUTER_STORAGE` is pinned to `memory` so no
    test can leave state in a SQLite file that another test then reads."""
    monkeypatch.setenv("MODELROUTER_STORAGE", "memory")
    monkeypatch.delenv("MODELROUTER_SSO_ENABLED", raising=False)
    from modelrouter import server as server_module

    with TestClient(server_module.app) as test_client:
        yield test_client


def _register(client, name="Acme", email="founder@acme.example") -> dict:
    response = client.post("/v1/orgs", json={"name": name, "owner_email": email})
    assert response.status_code == 201, response.text
    return response.json()


def _api_key_for(client, tenant_id: str) -> str:
    """Mints a machine key for a tenant directly through the repo — the HTTP
    surface has no key-creation endpoint yet, and inventing one here just to
    write a test would be testing something that doesn't ship."""
    from modelrouter import server as server_module

    _record, plaintext = server_module.app.state.tenancy_repo.create_api_key(tenant_id, "test key")
    return plaintext


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _session_for(client, tenant_id: str, user_id: str) -> str:
    """Builds a real session row and returns its cookie token, standing in for a
    completed SSO login (which needs a live IdP). Everything downstream — the
    Principal cutover, `require_authz`, membership re-checks — is the real code
    path."""
    from modelrouter import server as server_module
    from modelrouter.identity.sso.memory import InMemorySsoRepo
    from modelrouter.identity.sso.sessions import build_session
    from modelrouter.identity.sso.service import SsoService

    state = server_module.app.state
    if getattr(state, "sso", None) is None:
        state.sso = SsoService(InMemorySsoRepo(), state.identity)
    record, token = build_session(user_id=user_id, tenant_id=tenant_id)
    state.sso._sso.create_session(record)
    return token


# ── Registration ──────────────────────────────────────────────────────────

def test_registration_returns_tenant_owner_and_default_workspace(client):
    body = _register(client)
    assert body["org_name"] == "Acme"
    assert body["role"] == ROLE_OWNER
    assert body["default_workspace"]["slug"] == "default"
    assert body["tenant_id"].startswith("tn_")
    assert body["user_id"].startswith("usr_")


def test_registration_rejects_an_invalid_email(client):
    response = client.post("/v1/orgs", json={"name": "Acme", "owner_email": "not-an-email"})
    assert response.status_code == 422


def test_registration_rejects_unknown_fields(client):
    response = client.post(
        "/v1/orgs",
        json={"name": "Acme", "owner_email": "a@b.example", "role": "owner"},
    )
    assert response.status_code == 422   # extra="forbid" -- can't smuggle a role in


# ── Authentication ────────────────────────────────────────────────────────

def test_management_routes_require_credentials(client):
    assert client.get("/v1/orgs/me").status_code == 401
    assert client.get("/v1/workspaces").status_code == 401


def test_an_invalid_bearer_token_is_a_generic_401(client):
    response = client.get("/v1/orgs/me", headers=_headers("mr_not-a-real-key"))
    assert response.status_code == 401
    # The message must not distinguish unknown from revoked from suspended.
    assert response.json()["detail"] == "missing or invalid credentials"


def test_an_api_key_authenticates_and_reports_its_permissions(client):
    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])

    response = client.get("/v1/orgs/me", headers=_headers(key))

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == org["tenant_id"]
    assert "route:invoke" in body["your_permissions"]
    assert "member:invite" not in body["your_permissions"]   # machines don't administer


def test_a_human_session_authenticates_via_cookie(client):
    org = _register(client)
    token = _session_for(client, org["tenant_id"], org["user_id"])

    response = client.get("/v1/orgs/me", cookies={"__Host-mr_session": token})

    assert response.status_code == 200
    assert response.json()["your_role"] == ROLE_OWNER


def test_a_bearer_key_wins_over_a_stale_cookie(client):
    """An explicit credential must beat an ambient one, or a leftover browser
    cookie would silently act as the wrong subject."""
    first = _register(client, name="Acme", email="a@acme.example")
    second = _register(client, name="Beta", email="b@beta.example")
    key = _api_key_for(client, second["tenant_id"])
    cookie = _session_for(client, first["tenant_id"], first["user_id"])

    response = client.get(
        "/v1/orgs/me", headers=_headers(key), cookies={"__Host-mr_session": cookie},
    )

    assert response.json()["tenant_id"] == second["tenant_id"]


# ── RBAC enforcement ──────────────────────────────────────────────────────

def _member_session(client, org: dict, email: str, role: str) -> str:
    from modelrouter import server as server_module

    repo = server_module.app.state.identity
    user = repo.create_user(email)
    repo.create_org_membership(org["tenant_id"], user.user_id, role)
    return _session_for(client, org["tenant_id"], user.user_id)


def test_a_viewer_cannot_create_a_workspace(client):
    org = _register(client)
    cookie = _member_session(client, org, "viewer@acme.example", ROLE_VIEWER)

    response = client.post(
        "/v1/workspaces", json={"slug": "eng", "name": "Engineering"},
        cookies={"__Host-mr_session": cookie},
    )

    assert response.status_code == 403
    assert "workspace:create" in response.json()["detail"]


def test_an_admin_can_create_a_workspace_and_a_project(client):
    org = _register(client)
    cookie = _member_session(client, org, "admin@acme.example", ROLE_ADMIN)
    jar = {"__Host-mr_session": cookie}

    workspace = client.post(
        "/v1/workspaces", json={"slug": "eng", "name": "Engineering"}, cookies=jar,
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["workspace_id"]

    project = client.post(
        f"/v1/workspaces/{workspace_id}/projects",
        json={"slug": "api", "name": "API"}, cookies=jar,
    )
    assert project.status_code == 201
    assert project.json()["workspace_id"] == workspace_id


def test_an_admin_cannot_change_roles_but_an_owner_can(client):
    org = _register(client)
    from modelrouter import server as server_module

    repo = server_module.app.state.identity
    target = repo.create_user("target@acme.example")
    repo.create_org_membership(org["tenant_id"], target.user_id, ROLE_MEMBER)

    admin_jar = {"__Host-mr_session": _member_session(client, org, "adm@acme.example", ROLE_ADMIN)}
    refused = client.patch(
        f"/v1/orgs/me/members/{target.user_id}", json={"role": ROLE_ADMIN}, cookies=admin_jar,
    )
    assert refused.status_code == 403

    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    allowed = client.patch(
        f"/v1/orgs/me/members/{target.user_id}", json={"role": ROLE_ADMIN}, cookies=owner_jar,
    )
    assert allowed.status_code == 200
    assert allowed.json()["role"] == ROLE_ADMIN


def test_an_invalid_role_is_rejected_by_validation(client):
    org = _register(client)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    response = client.patch(
        f"/v1/orgs/me/members/{org['user_id']}", json={"role": "superuser"}, cookies=jar,
    )
    assert response.status_code == 422


def test_the_last_owner_cannot_be_removed_over_http(client):
    org = _register(client)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}

    response = client.delete(f"/v1/orgs/me/members/{org['user_id']}", cookies=jar)

    assert response.status_code == 409
    assert "owner" in response.json()["detail"].lower()


def test_a_slug_conflict_is_a_409(client):
    org = _register(client)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    client.post("/v1/workspaces", json={"slug": "eng", "name": "Engineering"}, cookies=jar)

    again = client.post("/v1/workspaces", json={"slug": "eng", "name": "Again"}, cookies=jar)

    assert again.status_code == 409


def test_a_malformed_slug_is_rejected(client):
    org = _register(client)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    for bad in ("Engineering", "has space", "-leading", "a" * 70):
        response = client.post("/v1/workspaces", json={"slug": bad, "name": "x"}, cookies=jar)
        assert response.status_code == 422, bad


# ── Cross-tenant isolation ────────────────────────────────────────────────

def test_another_tenants_workspace_is_a_404_not_a_403(client):
    """Confirming "it exists, just not for you" is itself a cross-tenant leak."""
    mine = _register(client, name="Acme", email="a@acme.example")
    theirs = _register(client, name="Beta", email="b@beta.example")
    their_jar = {"__Host-mr_session": _session_for(client, theirs["tenant_id"], theirs["user_id"])}
    their_ws = client.post(
        "/v1/workspaces", json={"slug": "secret", "name": "Secret"}, cookies=their_jar,
    ).json()["workspace_id"]

    my_jar = {"__Host-mr_session": _session_for(client, mine["tenant_id"], mine["user_id"])}
    response = client.get(f"/v1/workspaces/{their_ws}", cookies=my_jar)

    assert response.status_code == 404


def test_another_tenants_project_is_a_404(client):
    mine = _register(client, name="Acme", email="a@acme.example")
    theirs = _register(client, name="Beta", email="b@beta.example")
    their_jar = {"__Host-mr_session": _session_for(client, theirs["tenant_id"], theirs["user_id"])}
    their_project = client.post(
        f"/v1/workspaces/{theirs['default_workspace']['workspace_id']}/projects",
        json={"slug": "api", "name": "API"}, cookies=their_jar,
    ).json()["project_id"]

    my_jar = {"__Host-mr_session": _session_for(client, mine["tenant_id"], mine["user_id"])}
    assert client.get(f"/v1/projects/{their_project}", cookies=my_jar).status_code == 404


def test_listing_workspaces_never_includes_another_tenants(client):
    mine = _register(client, name="Acme", email="a@acme.example")
    theirs = _register(client, name="Beta", email="b@beta.example")
    their_jar = {"__Host-mr_session": _session_for(client, theirs["tenant_id"], theirs["user_id"])}
    client.post("/v1/workspaces", json={"slug": "theirs", "name": "Theirs"}, cookies=their_jar)

    my_jar = {"__Host-mr_session": _session_for(client, mine["tenant_id"], mine["user_id"])}
    listed = client.get("/v1/workspaces", cookies=my_jar).json()["workspaces"]

    assert [w["slug"] for w in listed] == ["default"]


def test_creating_a_project_in_another_tenants_workspace_is_refused(client):
    mine = _register(client, name="Acme", email="a@acme.example")
    theirs = _register(client, name="Beta", email="b@beta.example")
    my_jar = {"__Host-mr_session": _session_for(client, mine["tenant_id"], mine["user_id"])}

    response = client.post(
        f"/v1/workspaces/{theirs['default_workspace']['workspace_id']}/projects",
        json={"slug": "api", "name": "API"}, cookies=my_jar,
    )

    assert response.status_code == 404


# ── Invitations over HTTP ─────────────────────────────────────────────────

def test_the_full_invitation_round_trip(client):
    org = _register(client)
    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}

    created = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "newbie@acme.example", "role": ROLE_MEMBER}, cookies=owner_jar,
    )
    assert created.status_code == 201
    token = created.json()["token"]
    assert token

    # A listing must never echo the token back.
    listed = client.get("/v1/orgs/me/invitations", cookies=owner_jar).json()["invitations"]
    assert len(listed) == 1
    assert "token" not in listed[0]

    # The invitee signs in (a real session) and accepts.
    from modelrouter import server as server_module

    invitee = server_module.app.state.identity.create_user("newbie@acme.example")
    invitee_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], invitee.user_id)}
    accepted = client.post(
        "/v1/invitations/accept", json={"token": token}, cookies=invitee_jar,
    )
    assert accepted.status_code == 200
    assert accepted.json()["tenant_id"] == org["tenant_id"]


def test_an_admin_cannot_invite_an_owner_over_http(client):
    org = _register(client)
    admin_jar = {"__Host-mr_session": _member_session(client, org, "adm@acme.example", ROLE_ADMIN)}

    response = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "x@acme.example", "role": ROLE_OWNER}, cookies=admin_jar,
    )

    assert response.status_code == 403


def test_a_member_cannot_invite_anyone(client):
    org = _register(client)
    jar = {"__Host-mr_session": _member_session(client, org, "m@acme.example", ROLE_MEMBER)}
    response = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "x@acme.example", "role": ROLE_VIEWER}, cookies=jar,
    )
    assert response.status_code == 403


def test_an_api_key_cannot_accept_an_invitation(client):
    """An invitation is addressed to a PERSON; a machine credential proves a
    tenant, not a person."""
    org = _register(client)
    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    token = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "n@acme.example", "role": ROLE_MEMBER}, cookies=owner_jar,
    ).json()["token"]

    key = _api_key_for(client, org["tenant_id"])
    response = client.post("/v1/invitations/accept", json={"token": token}, headers=_headers(key))

    assert response.status_code == 403


def test_a_wrong_email_cannot_accept_an_invitation(client):
    org = _register(client)
    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    token = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "intended@acme.example", "role": ROLE_MEMBER}, cookies=owner_jar,
    ).json()["token"]

    from modelrouter import server as server_module

    interloper = server_module.app.state.identity.create_user("someone@else.example")
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], interloper.user_id)}

    response = client.post("/v1/invitations/accept", json={"token": token}, cookies=jar)

    assert response.status_code == 400
    assert "not valid" in response.json()["detail"]


def test_a_revoked_invitation_cannot_be_accepted(client):
    org = _register(client)
    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    created = client.post(
        "/v1/orgs/me/invitations",
        json={"email": "n@acme.example", "role": ROLE_MEMBER}, cookies=owner_jar,
    ).json()
    client.delete(
        f"/v1/orgs/me/invitations/{created['invitation']['invitation_id']}", cookies=owner_jar,
    )

    from modelrouter import server as server_module

    invitee = server_module.app.state.identity.create_user("n@acme.example")
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], invitee.user_id)}
    response = client.post("/v1/invitations/accept", json={"token": created["token"]}, cookies=jar)

    assert response.status_code == 400


# ── Immediate revocation ──────────────────────────────────────────────────

def test_removing_a_member_immediately_kills_their_session(client):
    """The reason sessions are server-side and the role is re-read per request."""
    org = _register(client)
    from modelrouter import server as server_module

    repo = server_module.app.state.identity
    member = repo.create_user("m@acme.example")
    repo.create_org_membership(org["tenant_id"], member.user_id, ROLE_ADMIN)
    member_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], member.user_id)}
    assert client.get("/v1/orgs/me", cookies=member_jar).status_code == 200

    owner_jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}
    removed = client.delete(f"/v1/orgs/me/members/{member.user_id}", cookies=owner_jar)
    assert removed.status_code == 204

    assert client.get("/v1/orgs/me", cookies=member_jar).status_code == 401


def test_a_promotion_takes_effect_on_the_very_next_request(client):
    org = _register(client)
    from modelrouter import server as server_module

    repo = server_module.app.state.identity
    member = repo.create_user("m@acme.example")
    repo.create_org_membership(org["tenant_id"], member.user_id, ROLE_VIEWER)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], member.user_id)}
    assert client.post(
        "/v1/workspaces", json={"slug": "eng", "name": "E"}, cookies=jar,
    ).status_code == 403

    repo.set_org_role(org["tenant_id"], member.user_id, ROLE_ADMIN)

    assert client.post(
        "/v1/workspaces", json={"slug": "eng", "name": "E"}, cookies=jar,
    ).status_code == 201


# ── Domains ───────────────────────────────────────────────────────────────

def test_adding_a_domain_returns_a_verification_token_and_is_unverified(client):
    org = _register(client)
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}

    response = client.post(
        "/v1/orgs/me/domains", json={"domain": "acme.example"}, cookies=jar,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["verified"] is False
    assert body["verification_token"]
    assert "TXT" in body["instructions"]


def test_an_admin_cannot_add_a_domain(client):
    """`org:manage` is owner-only — a domain claim is the input to SSO routing."""
    org = _register(client)
    jar = {"__Host-mr_session": _member_session(client, org, "adm@acme.example", ROLE_ADMIN)}
    response = client.post("/v1/orgs/me/domains", json={"domain": "x.example"}, cookies=jar)
    assert response.status_code == 403


# ── SSO routes when SSO is not configured ─────────────────────────────────

def test_sso_routes_answer_404_when_sso_is_not_configured(client):
    """SSO absent is a normal state, not a degraded one — the routes exist in the
    schema but refuse to act."""
    assert client.get("/auth/sso/login?org=tn_x", follow_redirects=False).status_code == 404


def test_backchannel_logout_refuses_to_act_on_unverified_tokens(client):
    """Registered but honest: acting on an unverified logout token would be a
    denial-of-service primitive (anyone could log anyone out)."""
    org = _register(client)
    _session_for(client, org["tenant_id"], org["user_id"])   # ensures app.state.sso exists
    response = client.post("/auth/sso/backchannel-logout", json={"logout_token": "x.y.z"})
    assert response.status_code == 501


# ── The existing LLM surface still works after the Principal cutover ──────

def test_an_api_key_can_still_call_the_chat_surface(client):
    """The cutover replaced `require_tenant_key` with `require_principal`
    everywhere; machine credentials must be completely unaffected."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    org = _register(client)
    key = _api_key_for(client, org["tenant_id"])
    server_module.app.state.accounting.purchase_credits(org["tenant_id"], 5.0)
    server_module.app.state.router = ModelRouter({"fake": FakeProviderAdapter("fake")})

    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        headers=_headers(key),
    )

    assert response.status_code == 200, response.text


def test_a_human_session_can_also_call_the_chat_surface(client):
    """The Principal abstraction's payoff: a signed-in human can use the product
    without needing a machine key minted for them."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    org = _register(client)
    server_module.app.state.accounting.purchase_credits(org["tenant_id"], 5.0)
    server_module.app.state.router = ModelRouter({"fake": FakeProviderAdapter("fake")})
    jar = {"__Host-mr_session": _session_for(client, org["tenant_id"], org["user_id"])}

    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        cookies=jar,
    )

    assert response.status_code == 200, response.text


def test_a_viewer_session_is_still_authenticated_for_chat(client):
    """Documenting a real, deliberate scope boundary rather than implying
    otherwise: the chat routes authenticate a Principal but do NOT yet check
    `route:invoke`, so a viewer reaches them. Wiring `require_authz` into the
    chat surface is a small, separate change; this test pins the CURRENT
    behavior so that change is a visible decision instead of a silent drift."""
    from modelrouter import server as server_module
    from modelrouter.providers.adapters import FakeProviderAdapter
    from modelrouter.router import ModelRouter

    org = _register(client)
    server_module.app.state.accounting.purchase_credits(org["tenant_id"], 5.0)
    server_module.app.state.router = ModelRouter({"fake": FakeProviderAdapter("fake")})
    jar = {"__Host-mr_session": _member_session(client, org, "v@acme.example", ROLE_VIEWER)}

    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "models": ["fake:model-a"]},
        cookies=jar,
    )

    assert response.status_code == 200

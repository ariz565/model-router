"""The management HTTP surface: organizations, workspaces, projects, members,
invitations, domains, and the audit trail.

**Its own module, and the only one in `identity/` that imports FastAPI.**
`identity/__init__.py` deliberately does not import this, so the whole package
stays usable as a library and testable with `fastapi` absent. `server.py`
mounts it with one `include_router` line.

**No tenant id appears in any path.** Every route is implicitly scoped to the
caller's own org, taken from their credential — `/v1/orgs/me/...`, never
`/v1/orgs/{tenant_id}/...`. A tenant id in the path is an invitation to write
the handler that forgets to compare it against the principal's, which is the
single most common multi-tenant vulnerability. Workspace and project ids DO
appear in paths (they have to), and `require_authz` verifies each one belongs to
the caller's tenant before granting anything.

**404, not 403, for another tenant's resource.** Confirming "that exists, just
not for you" is itself a cross-tenant leak, so a resource outside the caller's
org is indistinguishable from one that never existed.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from modelrouter.core.errors import (
    AlreadyMemberError,
    EmailAlreadyRegisteredError,
    IdentityError,
    InvitationInvalidError,
    LastOwnerError,
    NotAMemberError,
    ProjectNotFoundError,
    RoleEscalationError,
    SlugConflictError,
    WorkspaceNotFoundError,
)
from modelrouter.identity.authz import AuthzContext, Principal
from modelrouter.identity.roles import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
    SCOPE_ORG,
    SCOPE_PROJECT,
    SCOPE_WORKSPACE,
    Permission,
)

router = APIRouter(tags=["identity"])

_SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}$"


# ── Request/response models ───────────────────────────────────────────────

class RegisterOrgRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    owner_email: EmailStr
    owner_name: str | None = Field(default=None, max_length=200)


class CreateWorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A slug reaches URLs and unique indexes, so it is constrained here rather
    # than sanitized later: lowercase, alphanumeric plus hyphens.
    slug: str = Field(pattern=_SLUG_PATTERN)
    name: str = Field(min_length=1, max_length=200)


class CreateProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=_SLUG_PATTERN)
    name: str = Field(min_length=1, max_length=200)


class InviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    # An unknown role is rejected by validation before any handler runs, and
    # the service re-checks it anyway -- the role a caller may grant is bounded
    # by their own rank, which only the service can know.
    role: str = Field(pattern=f"^({ROLE_OWNER}|{ROLE_ADMIN}|{ROLE_MEMBER}|{ROLE_VIEWER})$")
    scope_level: str = Field(default=SCOPE_ORG, pattern=f"^({SCOPE_ORG}|{SCOPE_WORKSPACE}|{SCOPE_PROJECT})$")
    workspace_id: str | None = None
    project_id: str | None = None
    ttl_days: int = Field(default=7, ge=1, le=30)


class AcceptInviteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1)


class ChangeRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=f"^({ROLE_OWNER}|{ROLE_ADMIN}|{ROLE_MEMBER}|{ROLE_VIEWER})$")


class AddDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=3, max_length=253)


# ── Shared plumbing ───────────────────────────────────────────────────────

def _service(request: Request):
    return request.app.state.identity_service


def _repo(request: Request):
    return request.app.state.identity


def _require(permission: str):
    """Local alias for `server.py`'s dependency factory, imported lazily so this
    module doesn't import `server` at module scope (which would be circular —
    `server` includes this router)."""
    from modelrouter.server import require_authz

    return require_authz(permission)


def _handle(exc: IdentityError) -> HTTPException:
    """Maps this package's typed errors onto HTTP status codes in ONE place, so
    every endpoint answers consistently.

    - 404 for "doesn't exist in your tenant" — never 403, see the module
      docstring on why.
    - 409 for a genuine conflict the caller can resolve (slug taken, already a
      member, last owner).
    - 403 for a refused escalation.
    - 400 for an invitation that can't be used, whose message is deliberately
      identical for every underlying cause."""
    if isinstance(exc, (WorkspaceNotFoundError, ProjectNotFoundError, NotAMemberError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (SlugConflictError, AlreadyMemberError, LastOwnerError,
                        EmailAlreadyRegisteredError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, RoleEscalationError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, InvitationInvalidError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def _workspace_dict(workspace) -> dict:
    return {
        "workspace_id": workspace.workspace_id, "slug": workspace.slug,
        "name": workspace.name,
        "created_at": workspace.created_at.isoformat() if workspace.created_at else None,
    }


def _project_dict(project) -> dict:
    return {
        "project_id": project.project_id, "workspace_id": project.workspace_id,
        "slug": project.slug, "name": project.name,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    }


def _member_dict(membership, user) -> dict:
    return {
        "user_id": membership.user_id, "role": membership.role, "status": membership.status,
        "email": user.email if user else None, "name": user.name if user else None,
        "joined_at": membership.created_at.isoformat() if membership.created_at else None,
    }


def _invitation_dict(invitation) -> dict:
    """Never includes the token or its hash. The token exists once, in the
    creation response; echoing it in a listing would make every admin read a
    way to harvest live credentials."""
    return {
        "invitation_id": invitation.invitation_id, "email": invitation.email,
        "role": invitation.role, "scope_level": invitation.scope_level,
        "workspace_id": invitation.workspace_id, "project_id": invitation.project_id,
        "expires_at": invitation.expires_at.isoformat(),
        "accepted_at": invitation.accepted_at.isoformat() if invitation.accepted_at else None,
        "revoked_at": invitation.revoked_at.isoformat() if invitation.revoked_at else None,
    }


# ── Registration (unauthenticated by nature) ──────────────────────────────

@router.post("/v1/orgs", status_code=201)
async def register_organization(body: RegisterOrgRequest, request: Request):
    """Creates an organization, its first owner, and a default workspace.

    Necessarily unauthenticated — there is no credential to present before your
    org exists. That makes it the one spam-exposed endpoint here; rate limiting
    and bot mitigation belong at the edge (API Gateway/WAF), not in application
    code, and this is NOT rate limited by ModelRouter itself. Stated rather than
    implied, because an unauthenticated create endpoint deserves to be noticed.
    """
    try:
        result = _service(request).register_organization(
            body.name, str(body.owner_email), owner_name=body.owner_name,
            actor_ip=request.client.host if request.client else None,
        )
    except IdentityError as e:
        raise _handle(e) from e
    return {
        "tenant_id": result.tenant.tenant_id,
        "org_name": result.tenant.name,
        "user_id": result.user.user_id,
        "role": result.membership.role,
        "default_workspace": _workspace_dict(result.workspace),
    }


# ── The caller's own org ──────────────────────────────────────────────────

@router.get("/v1/orgs/me")
async def get_my_org(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
):
    tenant = request.app.state.tenancy_repo.get_tenant(ctx.scope.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="organization not found")
    return {
        "tenant_id": tenant.tenant_id, "name": tenant.name, "status": tenant.status,
        # The caller's own effective role and permissions -- what a UI needs to
        # decide which controls to render, without guessing our rules.
        "your_role": ctx.role,
        "your_permissions": sorted(ctx.permissions),
    }


@router.get("/v1/orgs/me/members")
async def list_members(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_READ))],
):
    repo = _repo(request)
    memberships = repo.list_org_memberships(ctx.scope.tenant_id)
    return {"members": [
        _member_dict(m, repo.get_user(m.user_id)) for m in memberships
    ]}


@router.patch("/v1/orgs/me/members/{user_id}")
async def change_member_role(
    user_id: str, body: ChangeRoleRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_ROLE_CHANGE))],
):
    try:
        _service(request).change_org_role(ctx, user_id, body.role)
    except IdentityError as e:
        raise _handle(e) from e
    return {"user_id": user_id, "role": body.role}


@router.delete("/v1/orgs/me/members/{user_id}", status_code=204)
async def remove_member(
    user_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_REMOVE))],
):
    try:
        _service(request).remove_member(ctx, user_id)
    except IdentityError as e:
        raise _handle(e) from e
    # Removing someone must also end their live sessions immediately, or their
    # access survives until a cookie happens to expire. Only applies when SSO is
    # configured; with API keys only there are no sessions to revoke.
    sso_service = getattr(request.app.state, "sso", None)
    if sso_service is not None:
        sso_service.revoke_user_sessions(user_id, tenant_id=ctx.scope.tenant_id)
    return None


# ── Invitations ───────────────────────────────────────────────────────────

@router.post("/v1/orgs/me/invitations", status_code=201)
async def create_invitation(
    body: InviteRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_INVITE))],
):
    """The response contains the plaintext token EXACTLY once — it is never
    stored and never appears in any listing. The caller is responsible for
    emailing it; ModelRouter deliberately has no mail transport, so it cannot
    quietly become a spam relay."""
    try:
        invitation, token = _service(request).invite_member(
            ctx, str(body.email), body.role, scope_level=body.scope_level,
            workspace_id=body.workspace_id, project_id=body.project_id,
            ttl_days=body.ttl_days,
        )
    except IdentityError as e:
        raise _handle(e) from e
    except ValueError as e:      # an invalid scope/target combination
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"invitation": _invitation_dict(invitation), "token": token}


@router.get("/v1/orgs/me/invitations")
async def list_invitations(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_INVITE))],
):
    return {"invitations": [
        _invitation_dict(i) for i in _repo(request).list_invitations(ctx.scope.tenant_id)
    ]}


@router.delete("/v1/orgs/me/invitations/{invitation_id}", status_code=204)
async def revoke_invitation(
    invitation_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_INVITE))],
):
    _service(request).revoke_invitation(ctx, invitation_id)
    return None


@router.post("/v1/invitations/accept")
async def accept_invitation(body: AcceptInviteRequest, request: Request):
    """Requires an authenticated HUMAN session, and takes the accepting email
    from that session's user record — never from the request body.

    That is the whole security property: if the email could be supplied here,
    every invitation in the system would be redeemable by anyone who obtained a
    token. An API key is explicitly rejected because a machine credential proves
    a tenant, not a person, and an invitation is addressed to a person.

    Consequence worth stating plainly: without SSO configured there is no way to
    prove an email address, so invitation acceptance is unavailable. That's a
    real, documented limitation rather than a hole — the alternative would be
    trusting a self-asserted address."""
    from modelrouter.server import require_principal

    principal: Principal = await require_principal(
        request.headers.get("Authorization"), request,
    )
    if not principal.is_human:
        raise HTTPException(
            status_code=403,
            detail="accepting an invitation requires a signed-in user session, not an API key",
        )
    user = _repo(request).get_user(principal.subject_id)
    if user is None:
        raise HTTPException(status_code=403, detail="no user record for this session")

    try:
        accepted_user, tenant_id = _service(request).accept_invitation(
            body.token, authenticated_email=user.email,
            actor_ip=request.client.host if request.client else None,
        )
    except IdentityError as e:
        raise _handle(e) from e
    return {"tenant_id": tenant_id, "user_id": accepted_user.user_id}


# ── Workspaces ────────────────────────────────────────────────────────────

@router.get("/v1/workspaces")
async def list_workspaces(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.WORKSPACE_READ))],
):
    return {"workspaces": [
        _workspace_dict(w) for w in _repo(request).list_workspaces(ctx.scope.tenant_id)
    ]}


@router.post("/v1/workspaces", status_code=201)
async def create_workspace(
    body: CreateWorkspaceRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.WORKSPACE_CREATE))],
):
    try:
        workspace = _service(request).create_workspace(ctx, body.slug, body.name)
    except IdentityError as e:
        raise _handle(e) from e
    return _workspace_dict(workspace)


@router.get("/v1/workspaces/{workspace_id}")
async def get_workspace(
    workspace_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.WORKSPACE_READ))],
):
    workspace = _repo(request).get_workspace(ctx.scope.tenant_id, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return _workspace_dict(workspace)


@router.delete("/v1/workspaces/{workspace_id}", status_code=204)
async def archive_workspace(
    workspace_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.WORKSPACE_DELETE))],
):
    try:
        _service(request).archive_workspace(ctx, workspace_id)
    except IdentityError as e:
        raise _handle(e) from e
    return None


# ── Projects ──────────────────────────────────────────────────────────────

@router.get("/v1/workspaces/{workspace_id}/projects")
async def list_projects(
    workspace_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.PROJECT_READ))],
):
    if _repo(request).get_workspace(ctx.scope.tenant_id, workspace_id) is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return {"projects": [
        _project_dict(p) for p in _repo(request).list_projects(ctx.scope.tenant_id, workspace_id)
    ]}


@router.post("/v1/workspaces/{workspace_id}/projects", status_code=201)
async def create_project(
    workspace_id: str, body: CreateProjectRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.PROJECT_CREATE))],
):
    try:
        project = _service(request).create_project(ctx, workspace_id, body.slug, body.name)
    except IdentityError as e:
        raise _handle(e) from e
    return _project_dict(project)


@router.get("/v1/projects/{project_id}")
async def get_project(
    project_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.PROJECT_READ))],
):
    project = _repo(request).get_project(ctx.scope.tenant_id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return _project_dict(project)


@router.delete("/v1/projects/{project_id}", status_code=204)
async def archive_project(
    project_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.PROJECT_DELETE))],
):
    try:
        _service(request).archive_project(ctx, project_id)
    except IdentityError as e:
        raise _handle(e) from e
    return None


@router.get("/v1/projects/{project_id}/members")
async def list_project_members(
    project_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.MEMBER_READ))],
):
    repo = _repo(request)
    if repo.get_project(ctx.scope.tenant_id, project_id) is None:
        raise HTTPException(status_code=404, detail="project not found")
    return {"members": [
        _member_dict(m, repo.get_user(m.user_id))
        for m in repo.list_project_memberships(ctx.scope.tenant_id, project_id)
    ]}


# ── Domains (SSO home-realm discovery; useful even without SSO enabled) ───

@router.get("/v1/orgs/me/domains")
async def list_domains(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_READ))],
):
    return {"domains": [
        {"domain": d.domain, "verified": d.is_verified,
         "verification_token": d.verification_token}
        for d in _repo(request).list_domains(ctx.scope.tenant_id)
    ]}


@router.post("/v1/orgs/me/domains", status_code=201)
async def add_domain(
    body: AddDomainRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.ORG_MANAGE))],
):
    """Claiming a domain grants NOTHING until it is verified. The returned
    token is what the org publishes as a DNS TXT record to prove control."""
    from modelrouter.core.ids import new_id

    record = _repo(request).add_domain(
        ctx.scope.tenant_id, body.domain, new_id("dvt"),
    )
    return {
        "domain": record.domain, "verified": False,
        "verification_token": record.verification_token,
        "instructions": (
            f"Publish a DNS TXT record at _modelrouter-verify.{record.domain} "
            f"with the value {record.verification_token}, then POST to this "
            f"domain's /verify endpoint."
        ),
    }

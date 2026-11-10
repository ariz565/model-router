"""Tenant-scoped API-key and billing management routes.

There is intentionally no tenant identifier in these URLs: management is
always scoped to the authenticated caller's own organization.  Secrets are
returned only by key creation and never written to audit records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modelrouter.identity.audit import AuditRecord
from modelrouter.identity.authz import AuthzContext
from modelrouter.identity.roles import Permission

router = APIRouter(tags=["tenancy"])


class CreateApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    budget_usd: float | None = Field(default=None, ge=0)
    token_ceiling: int | None = Field(default=None, ge=1)


class PurchaseCreditsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_usd: float = Field(gt=0, le=1_000_000)
    currency_display: str | None = Field(default=None, min_length=3, max_length=8)
    fx_rate: float | None = Field(default=None, gt=0)


def _require(permission: str):
    from modelrouter.server import require_authz
    return require_authz(permission)


def _key_view(key) -> dict:
    return {
        "key_id": key.key_id, "name": key.name, "prefix": key.prefix, "status": key.status,
        "created_at": key.created_at, "last_used_at": key.last_used_at,
        "revoked_at": key.revoked_at, "budget_usd": key.budget_usd,
        "token_ceiling": key.token_ceiling,
    }


def _audit(request: Request, ctx: AuthzContext, action: str, *, after: dict | None = None) -> None:
    request.app.state.audit.append(AuditRecord(
        tenant_id=ctx.scope.tenant_id, action=action, occurred_at=datetime.now(timezone.utc),
        actor_kind=ctx.principal.kind, actor_id=ctx.principal.subject_id,
        actor_ip=request.client.host if request.client else None, scope_level="org",
        scope_id=ctx.scope.tenant_id, after=after,
    ))


@router.get("/v1/keys")
async def list_api_keys(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.APIKEY_READ))],
):
    return {"keys": [_key_view(key) for key in request.app.state.tenancy_repo.list_api_keys(ctx.scope.tenant_id)]}


@router.post("/v1/keys", status_code=201)
async def create_api_key(
    body: CreateApiKeyRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.APIKEY_CREATE))],
):
    key, plaintext = request.app.state.tenancy_repo.create_api_key(
        ctx.scope.tenant_id, body.name, budget_usd=body.budget_usd, token_ceiling=body.token_ceiling,
    )
    _audit(request, ctx, "apikey.created", after={"key_id": key.key_id, "name": key.name})
    return {"key": _key_view(key), "secret": plaintext}


@router.delete("/v1/keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.APIKEY_REVOKE))],
):
    key = next((item for item in request.app.state.tenancy_repo.list_api_keys(ctx.scope.tenant_id)
                if item.key_id == key_id), None)
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    request.app.state.tenancy_repo.revoke_api_key(key_id)
    _audit(request, ctx, "apikey.revoked", after={"key_id": key_id})
    return None


@router.get("/v1/billing/balance")
async def get_balance(
    request: Request, ctx: Annotated[AuthzContext, Depends(_require(Permission.BILLING_READ))],
):
    account = request.app.state.accounting.balance(ctx.scope.tenant_id)
    return {"purchased_usd": account.purchased_usd, "spent_usd": account.spent_usd,
            "reserved_usd": account.reserved_usd, "available_usd": account.available_usd}


@router.post("/v1/billing/credits", status_code=201)
async def purchase_credits(
    body: PurchaseCreditsRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.BILLING_MANAGE))],
):
    request.app.state.accounting.purchase_credits(
        ctx.scope.tenant_id, body.amount_usd,
        currency_display=body.currency_display, fx_rate=body.fx_rate,
    )
    _audit(request, ctx, "billing.credits_purchased", after={"amount_usd": body.amount_usd})
    return await get_balance(request, ctx)

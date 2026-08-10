# Identity, Multi-Tenancy, RBAC, and SSO

How a human becomes an authenticated, authorized actor inside exactly one
organization — and why each piece is shaped the way it is.

> **Verification status, stated up front.** The logic in this package is
> executed-verified: **265 tests** covering RBAC resolution, the service-layer
> security rules, both storage tiers against one shared contract suite, and the
> full OIDC flow against a fake IdP. The **HTTP layer** (`http.py`,
> `sso/http.py`) is compile-checked and has **34 end-to-end tests written**, but
> those are skip-gated on `fastapi`, which is not installed in the current
> development environment — so they have not been executed. Run
> `pip install -e ".[server]" && python -m pytest tests/test_identity_http.py`
> to execute them. Anything below marked *not built* is genuinely not built.

---

## 1. The shape of the model

```
Tenant  ( == the Organization; L1's tenancy/models.py )
  │
  ├── TenantDomain      "we own @acme.com"  → verified by DNS TXT
  ├── SsoConnection     "we log in via Okta"  (0 or 1 active)
  ├── ApiKey            machine credentials    (L1)
  │
  ├── OrgMembership     User ←→ Tenant + role      ← REQUIRED to reach anything
  │
  └── Workspace  (many)
        └── Project  (many)
              └── ProjectMembership   optional role ELEVATION on this project

User  ( GLOBAL — one row per human, across every org )
  └── Identity  (many)   one per (SsoConnection, IdP subject)
```

Four decisions define this, each with a reason it had to be that way:

**`Tenant` IS the organization.** No parallel `organizations` table.
`tenant_id` is already the tenancy key threaded through accounting, traces,
evidence, and evaluation; introducing a second root entity would mean either
renaming all of that or having two identifiers that mean the same thing. The
second is how cross-tenant bugs happen.

**Users are global, one row per human.** `email` is unique system-wide, and
belonging to an org is a membership row. The rule underneath it:
*authentication is global, authorization is tenant-scoped.* The alternative —
one user row per human per org — is what Slack shipped and cannot undo: the same
person ends up with N unrelated accounts and no way to switch context. It also
makes "one consultant, two customer orgs, two different IdPs" trivially
expressible: the **connection** varies, the human doesn't.

**Identity is keyed on `(connection_id, provider_subject)` — never email.** An
IdP's `email` claim is attacker-controllable in real deployments (a tenant admin
on several major IdPs can set an arbitrary email on a user they control), so
resolving an SSO login to a user by email is a published cross-tenant
account-takeover path. The immutable subject the IdP issues is the only safe key.

**Memberships are soft-deleted, and uniqueness is enforced by PARTIAL indexes.**
The audit trail must show that someone was removed, and re-inviting someone you
removed last week must not collide with the tombstone. A plain
`UNIQUE(tenant_id, user_id)` would reject that re-invite forever, so every
uniqueness rule is `... WHERE deleted_at IS NULL`.

### Isolation strategy

Shared tables keyed by `tenant_id` (the "pool" model), with three layers of
defense against the one failure that matters — a missing tenant predicate:

1. **The repository makes it structurally hard.** Every tenant-scoped method
   takes `tenant_id` as its first positional parameter. There is deliberately no
   `get_project(project_id)` that looks up globally, because if one existed
   someone would eventually call it in a request path and the resulting
   cross-tenant read would look reasonable at review time.
2. **`tenant_id` leads every composite index**, so the safe query is also the
   fast one and nobody is tempted to "optimize" the predicate away.
3. **Postgres Row-Level Security** as a database-level backstop — designed for,
   *not built*. When this runs on Postgres in production, the pattern is
   `FORCE ROW LEVEL SECURITY`, a dedicated non-owner role, and `SET LOCAL
   app.current_tenant` inside the transaction-opening middleware. The critical
   footgun to avoid: plain `SET` instead of `SET LOCAL` leaks one tenant's
   context into the next request under a connection pooler.

---

## 2. RBAC

Four system roles with integer ranks. Gaps of 100 are intentional so a future
role can slot between two existing ones without renumbering stored comparisons.

| Role | Rank | Gains over the role below |
|---|---|---|
| `viewer` | 100 | read the org, members, workspaces, projects, billing, traces |
| `member` | 200 | `route:invoke`, create/update projects, read API keys |
| `admin` | 300 | invite/remove members, manage workspaces, delete projects, manage keys, read audit |
| `owner` | 400 | `org:manage`/`org:delete`, `member:role_change`, `billing:manage`, `sso:manage` |

Three deliberate boundaries:

- **`viewer` cannot `route:invoke`.** A "read-only" role that can silently spend
  the org's credits is not read-only in the sense anyone means it.
- **`admin` cannot touch billing or SSO.** Both are "can bankrupt or compromise
  the entire org" powers. Whoever controls the IdP connection controls who can
  sign in as anyone.
- **`admin` cannot change roles.** An admin who can promote can make a second
  account an owner. Role changes are owner-only.

**Code checks permissions, never role names.** Every enforcement point asks
"does this principal hold `project:create`", so roles can be resplit or made
customer-defined later without touching a single call site.

**Roles are code, not rows — compatibly.** The four are frozen definitions;
membership rows store a role *slug* in a `TEXT` column, not a foreign key. That's
the simplest thing that fully works today and the shape that stays correct when
custom roles arrive: add a `roles` table, the slug becomes a lookup key, and no
membership row, index, or call site changes.

### Inheritance: down, never sideways

Effective role on a resource is `max(org_role, workspace_role, project_role)` by
rank. A workspace or project membership row is an **elevation for that subtree
only** — absence means inherit, not deny.

```
org: viewer
 ├── workspace "eng"   ← workspace_membership: admin
 │     └── project "api"      → effective: admin   (inherited down)
 └── workspace "ops"          → effective: viewer  (NO sideways leak)
```

A lower subtree role never demotes an inherited one: an org admin who is a
project `viewer` is still admin there.

### No escalation

`can_grant(granter, granted)` is `rank(granter) >= rank(granted)`. Note `>=`,
not `>`: an owner **must** be able to appoint a second owner, or a sole owner
leaving orphans the organization. Enforced at invitation creation **and again at
acceptance**, because the inviter may have been demoted in between and a stale
invitation must not outrank its issuer's current role.

### Enforcement: `Depends`, not decorators

```python
@router.post("/v1/workspaces", status_code=201)
async def create_workspace(
    body: CreateWorkspaceRequest, request: Request,
    ctx: Annotated[AuthzContext, Depends(_require(Permission.WORKSPACE_CREATE))],
):
    ...
```

This is a FastAPI constraint, not a style preference. A dependency participates
in the DI graph, so it can declare its own `Depends(require_principal)` and read
path params; a decorator would have to scrape `kwargs` for both. Dependencies are
cached per request, overridable via `app.dependency_overrides` in tests, and
appear in the OpenAPI schema. Most importantly, **FastAPI introspects the handler
signature to build request validation** — a decorator that isn't scrupulous with
`functools.wraps` silently corrupts that, with no error message.

Authorization is not middleware either: Starlette middleware runs *before* route
resolution, so it has no path params and would degrade into regex-matching URLs.

---

## 3. One `Principal` for two kinds of caller

```
Authorization: Bearer mr_…  ─┐
                             ├─→  require_principal()  ─→  Principal  ─→ everything downstream
Cookie: __Host-mr_session   ─┘
```

A machine key and a human session authenticate by entirely different mechanisms
and resolve to one frozen `Principal`. Everything below — accounting, tracing,
routing, authorization — receives a `Principal` and cannot tell which it was,
which keeps "is this a key or a person" branching out of every handler.

- **Bearer is tried first.** An explicit credential must beat an ambient one, or
  a leftover browser cookie would silently act as the wrong subject.
- **API keys get `member`-equivalent permissions.** They can invoke models and
  read their tenant's usage/traces — exactly what a valid key could already do,
  so nothing regressed — and nothing administrative. Per-key scopes (a key
  restricted to one project, or read-only) are *not built*; when they arrive they
  become a field on `ApiKey` that **narrows** this set, never widens it.
- **Every failure is the same opaque 401.** Missing, malformed, unknown, revoked,
  suspended tenant, membership removed — one message, because the caller's next
  action is identical and distinguishing them tells an attacker which guess was
  closer.
- **Tenant identity comes from the credential, never the request.** A tenant id
  in a path or an `X-Org-Id` header is untrusted input to be *compared* against
  the principal's. No management route has a tenant id in its path at all —
  `/v1/orgs/me/...` — which removes the opportunity to write the handler that
  forgets to compare.

---

## 4. SSO: OIDC Authorization Code Flow + PKCE

### Why OIDC and not SAML

Every enterprise IdP that matters — Okta, Entra ID, Google Workspace, Auth0,
Keycloak — speaks OIDC as a first-class protocol. SAML's residual demand is
procurement-driven, not technical. `SsoConnection.protocol` is a discriminator so
SAML becomes a new value plus a new client module rather than a schema change,
and when it is needed it will use a library: SAML's attack surface (XML signature
wrapping, XXE, canonicalization) is where hand-rolled implementations die.

### The flow

```
  ┌────────┐                    ┌──────────────┐                ┌─────────┐
  │Browser │                    │ ModelRouter  │                │  IdP    │
  └───┬────┘                    └──────┬───────┘                └────┬────┘
      │ GET /auth/sso/login?org=…      │                             │
      │ ──────────────────────────────>│                             │
      │                    resolve connection (tenant, or verified   │
      │                      email domain — verified ONLY)           │
      │                    mint state + nonce + PKCE verifier        │
      │                    persist AuthTransaction  (SERVER-side)    │
      │ <── 302 to IdP ────────────────│                             │
      │ ────────────────── authenticate ───────────────────────────> │
      │ <── 302 /auth/sso/callback?code&state&iss ─────────────────── │
      │ ──────────────────────────────>│                             │
      │                    1. consume transaction  (ATOMIC, once)    │
      │                    2. check expiry                           │
      │                    3. check iss  ← mix-up defense, BEFORE    │
      │                       spending the code                      │
      │                    4. exchange code + PKCE verifier ───────> │
      │                    5. verify JWS via JWKS   <─── id_token ── │
      │                    6. validate claims (iss/aud/azp/exp/      │
      │                       iat/nonce/sub)                         │
      │                    7. resolve identity → linking rules       │
      │                    8. membership check / JIT                 │
      │                    9. mint session, set __Host- cookie       │
      │ <── 302 return_to + Set-Cookie ─│                            │
```

Order in step 1–3 is security-relevant: the transaction is consumed *first* so a
replayed callback cannot be processed twice even concurrently, and the issuer is
checked *before* the code is spent so a mix-up attempt never reaches the token
endpoint.

### What is validated, and why each one

| Check | Prevents |
|---|---|
| PKCE `S256` (never `plain`) | Intercepted authorization code being redeemed by someone else |
| `state`, single-use, server-side | CSRF; and it's the transaction lookup key |
| `nonce` vs. ID token claim | ID token replay from another request |
| `iss` response param vs. stored issuer | **IdP mix-up** — critical here, since this is a multi-IdP RP behind one callback |
| Exact-string `redirect_uri`, from config | Open redirect / token theft |
| `aud` contains our `client_id`; `azp == client_id` when present | A token minted for a different client that merely lists us |
| `exp`/`iat` with 60s leeway | Expired tokens, while tolerating normal clock drift |
| `sub` present and non-empty | A login with no stable subject to key an identity on |
| `return_to` must be a same-site path | Open redirect — including `//evil.example` and `/\evil.example`, the classic "starts with `/`" bypasses |

### `email_verified` fails closed

```python
True, "true", "1", 1          → verified
everything else               → NOT verified
(False, "false", "0", 0, None, missing, "yes", wrong types)
```

This exists because of a real published CVE: an implementation type-asserted the
claim as `bool`, so a string `"false"` — or an absent claim — was treated as
verified, yielding full account takeover. IdPs genuinely send this as a bool, a
string, and an integer.

### The account-linking rules — the security core

Applied in order on every callback:

1. **An existing identity wins outright.** If `(connection_id, sub)` is on file,
   that IS the user. Email is not consulted at all. This is the normal path for
   every returning user and is immune to any email claim the IdP sends — even if
   the IdP later stops sending one.
2. **A first-time login requires a verified email.** No `email_verified` →
   **refused**, not downgraded. Otherwise a tenant's IdP admin could mint a login
   carrying any address they like.
3. **Linking to an EXISTING user additionally requires a verified domain.** Even
   with a verified email, attaching this IdP subject to an account that already
   exists is only safe if the tenant has *proven* it controls that email's
   domain. Without this, tenant A's IdP could assert `email=ceo@tenant-b.com` and
   be handed tenant B's account — the **nOAuth** attack class.
4. **A new user is created only from a verified email.** This prevents squatting:
   otherwise a hostile IdP could pre-create a user row holding
   `ceo@bigcorp.com`, and a later legitimate invitation to that address would
   resolve to the squatter's record.

### Membership is separate from authentication

Proving who you are doesn't decide which org you may act in.
`jit_provisioning` (default on) grants a least-privilege `viewer` membership on
first login — correct semantics for org-owned SSO, since authenticating against
Acme's own IdP means you are an Acme person. With it off, only pre-invited users
can sign in. A **suspended** membership is never silently reactivated by JIT:
someone suspended that person on purpose.

### Sessions: opaque, server-side, not JWTs

The requirement that settles it: when someone is removed from an org, access must
stop *now*. A JWT can't be revoked without a server-side blocklist — at which
point you have server state anyway, plus signature verification on every request,
for a strictly weaker guarantee.

- Only a **hash** of the token is stored (32 random bytes, domain-separated from
  API-key and invitation hashes so one subsystem's hash can never be presented as
  another's credential).
- **Two independent expiries.** `idle_expires_at` slides forward on use;
  `absolute_expires_at` never moves. Idle-only would let a stolen token be kept
  alive forever by using it.
- **Membership and role are re-read on every request**, never cached on the
  session row — that is what makes removal and demotion immediate. A session
  whose membership vanished is *revoked*, not merely denied, so it stops being
  retried.
- **Rotated on login and on org switch**, never refreshed in place — session
  fixation defense.
- **Cookie:** `__Host-mr_session`, `HttpOnly`, `Secure`, `SameSite=Lax`,
  `Path=/`, no `Domain`. The `__Host-` prefix makes the browser *enforce* Secure
  + no-Domain + Path=/ rather than trusting us to remember. `Lax` and not
  `Strict` because `Strict` would drop the cookie on the cross-site top-level
  navigation back from the IdP, breaking login outright.
- **Org switching mints a new session** after re-verifying membership in the
  target. Being signed in to org A implies nothing about org B, and the active
  org is never read from a header or query param.

### Home-realm discovery

`?org=<tenant_id>` (an explicit picker) or `?email=…`, where the domain must be
**verified** by exactly one tenant. An unverified claim routes nothing —
otherwise anyone could claim `@bigcorp.com`, attach their own IdP, and receive
logins meant for the real owner. Verification is a DNS TXT record; a verified
domain only *proposes* a connection, and membership is what authorizes access.

---

## 5. Invitations

Bearer credentials, treated as such:

- 32 random bytes; **only a hash is stored**, so a leaked database is not a mass
  org takeover.
- **Expiry is a column, not encoded in the token** — so it is revocable and the
  verifier never trusts attacker-supplied data about its own validity.
- **Role and target scope are bound at creation and never read from the
  acceptance request.** That single rule prevents the classic escalation bug
  (accept while passing `role=owner`), which has produced real CVEs.
- **Single-use via an atomic claim in the repository** (`UPDATE … WHERE
  accepted_at IS NULL`, checking `rowcount`), not a read-then-write in the
  service — two concurrent acceptances must not both succeed.
- **Acceptance requires an authenticated human session**, and the email comes
  from that session's user record, never the request body. An API key is
  rejected: it proves a tenant, not a person. *Consequence, stated plainly:*
  without SSO configured there is no way to prove an email, so invitation
  acceptance is unavailable. That's a real documented limitation, not a hole —
  the alternative is trusting a self-asserted address.
- **Email binding with a verified-domain allowance.** Exact match, or the address
  is on a domain this tenant verified. That handles the single most common SSO
  support ticket (invited `first.last@acme.com`, IdP asserts `flast@acme.com`)
  without weakening anything, because the company proved it owns the domain.
- **A subtree invitation grants `viewer` at the org**, not the invited role:
  "you are a project admin" must not silently mean "you are an org admin."

---

## 6. Audit trail

A separate, append-only, hash-chained table — deliberately **not** L0's
`EventStore`. Three concrete reasons:

1. L0's sequence is **global and billing replays it**. Injecting membership churn
   would make every billing projection scan and discard authz noise, and couple
   audit write volume to accounting recovery time.
2. `store/events.py`'s own documented scope call: only money and traces are
   event-sourced; memberships are mutable reference data. Being *both* mutable
   rows and an event stream invites drift, which is worse than either alone.
3. Retention and query shape diverge — audit wants multi-year retention,
   tamper-evidence, and `(tenant, actor, time)` filtering for customer export;
   billing events want compaction.

Written in the same transaction as the mutation it records, which removes the
dual-write drift hazard. **Denials are recorded too** (`authz.permission_denied`,
`authz.escalation_refused`) — the most valuable lines in a security review are
the ones most systems throw away.

Hash chaining is **tamper-evident, not tamper-proof**, and the honest limit is
stated in the code: an attacker with write access to the whole table could
recompute the chain. Defending that requires shipping the chain head somewhere
they don't control — an operator decision this module doesn't make silently.

---

## 7. Module map and removability

```
identity/
  models.py      records; read its docstring first
  roles.py       permissions, roles, ranks, inheritance, no-escalation
  authz.py       Principal, ResourceScope, resolve_authz  ← no web framework
  ports.py       IdentityRepo seam (tenant_id always first)
  memory.py      zero-infra tier, same invariants as SQL
  sqlite_repo.py durable tier   +  schema.sql
  invitations.py token generation/hashing, bound construction
  audit.py       hash-chained authz audit log
  service.py     the ONLY place policy is enforced
  factory.py     env-driven construction
  http.py        FastAPI router  ← the only file here that imports FastAPI
  sso/
    models.py, oidc.py, sessions.py, ports.py, memory.py,
    sqlite_repo.py, schema.sql, service.py, factory.py, http.py
```

**`identity/sso/` is independently removable.** Nothing in the codebase imports
it except two clearly-marked lines in `server.py`. Delete the directory, those
lines, and the `[sso]` extra — nothing else changes behavior, because
`create_sso_service()` already returns `None` when SSO is unconfigured and that
is the *same code path* the absent state uses. There is no feature flag consulted
at request time: `app.state.sso is None` **is** the off state, so the off path has
no code of its own to get wrong.

`identity/__init__.py` does not import `http.py`, so the package stays usable as
a library and testable with `fastapi` absent.

---

## 8. Configuration

```bash
MODELROUTER_STORAGE=memory|sqlite      # one var switches every storage tier
MODELROUTER_KEY_HASH_SECRET=…          # pepper for key/session/invite hashes
MODELROUTER_BYOK_MASTER_KEY=…          # required by the SQLite SSO tier (seals client secrets)

MODELROUTER_SSO_ENABLED=true           # absent ⇒ SSO not constructed at all
MODELROUTER_SSO_REDIRECT_URI=https://gateway.example.com/auth/sso/callback
MODELROUTER_SSO_JIT_PROVISIONING=true  # false ⇒ only pre-invited users may sign in
```

Neither `identity/` nor `identity/sso/` has a Redis or Postgres tier yet, and
their factories **fail loudly** rather than falling through to SQLite when
`MODELROUTER_STORAGE` names one — silently storing memberships and IdP secrets
somewhere other than where the operator configured is a security-relevant
surprise.

---

## 9. Endpoints

| Method | Path | Permission |
|---|---|---|
| `POST` | `/v1/orgs` | — (registration; **not rate limited here** — that's the edge's job) |
| `GET` | `/v1/orgs/me` | `org:read` |
| `GET` | `/v1/orgs/me/members` | `member:read` |
| `PATCH` | `/v1/orgs/me/members/{user_id}` | `member:role_change` (owner) |
| `DELETE` | `/v1/orgs/me/members/{user_id}` | `member:remove` (+ revokes their sessions) |
| `POST` `GET` | `/v1/orgs/me/invitations` | `member:invite` |
| `DELETE` | `/v1/orgs/me/invitations/{id}` | `member:invite` |
| `POST` | `/v1/invitations/accept` | authenticated **human session** |
| `GET` `POST` | `/v1/workspaces` | `workspace:read` / `workspace:create` |
| `GET` `DELETE` | `/v1/workspaces/{workspace_id}` | `workspace:read` / `workspace:delete` |
| `GET` `POST` | `/v1/workspaces/{workspace_id}/projects` | `project:read` / `project:create` |
| `GET` `DELETE` | `/v1/projects/{project_id}` | `project:read` / `project:delete` |
| `GET` | `/v1/projects/{project_id}/members` | `member:read` |
| `GET` `POST` | `/v1/orgs/me/domains` | `org:read` / `org:manage` (owner) |
| `GET` | `/auth/sso/login` | — |
| `GET` | `/auth/sso/callback` | — |
| `GET` | `/auth/sso/me` | human session |
| `POST` | `/auth/sso/logout` | — (idempotent) |
| `POST` | `/auth/sso/switch-org` | human session |
| `GET` `PUT` `DELETE` | `/v1/orgs/me/sso-connection` | `sso:manage` (owner) |

Another tenant's resource is always **404**, never 403 — confirming "it exists,
just not for you" is itself a cross-tenant leak.

---

## 10. Not built — named, not implied

- **Postgres/Redis tiers** for `IdentityRepo`, `AuditLog`, and `SsoRepo`. The
  factories refuse those backends rather than silently using SQLite.
- **Postgres RLS.** Designed for above; needs the Postgres tier first.
- **Back-channel logout token verification.** The endpoint is registered and
  returns **501**: acting on an unverified logout token would itself be a
  denial-of-service primitive (anyone could log anyone out). Completing it means
  resolving the connection by the token's `iss`, verifying the JWS, and checking
  the `events` claim.
- **DNS TXT verification execution.** Domains can be claimed and a token issued;
  the resolver check that flips `verified_at` is not wired. Until then, verified
  domains only exist if set directly through the repo — and **unverified domains
  grant nothing**, so the security property holds.
- **SCIM 2.0** provisioning/deprovisioning. JIT covers onboarding but is silent
  about offboarding, which is where orphaned-account rates come from.
- **SAML.** Discriminator present, no implementation.
- **`route:invoke` on the chat surface.** The LLM endpoints authenticate a
  `Principal` but don't yet check this permission, so a `viewer` session reaches
  them. A test pins the current behavior so closing it is a visible decision
  rather than silent drift.
- **Per-key scopes**, custom roles, and workspace/project membership management
  endpoints (the repo and resolution support all three; only HTTP surfaces are
  missing).
- **Email delivery.** Invitation tokens are returned to the API caller; there is
  no mail transport, deliberately, so this cannot become a spam relay.

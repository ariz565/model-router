"""Static (AST-level) checks on `server.py` that catch import-time breakage
WITHOUT needing `fastapi` installed.

**Why this file exists.** During the `require_tenant_key` → `require_principal`
cutover, three routes used the `dependencies=[Depends(require_tenant_key)]` form
rather than the annotated-parameter form, and a mechanical replace missed them.
`py_compile` passed (a stale name is a runtime lookup, not a syntax error) and the
test suite passed (every server test skips without `fastapi`), so the module
would have raised `NameError` on first import in any environment that HAD
fastapi — i.e. production, and nowhere else.

These checks close that specific blind spot: they parse the source and verify
name resolution and route hygiene using only the standard library, so they run in
every environment regardless of which optional dependencies are present.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SERVER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src" / "modelrouter" / "server.py"
)


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))


@pytest.fixture(scope="module")
def defined_names(tree: ast.Module) -> set[str]:
    """Every name `server.py` can resolve at module scope: its imports, its
    module-level assignments, and its own function/class definitions."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _depends_targets(tree: ast.Module) -> list[tuple[str, int]]:
    """Every `Depends(X)` where X is a bare name, with its line number.

    Only bare names are collected: `Depends(_require("x"))` and other call
    expressions resolve through their own callee, which this same check covers
    separately when that callee is itself a bare name."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id == "Depends" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name):
                found.append((first.id, node.lineno))
            elif isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
                found.append((first.func.id, node.lineno))
    return found


def test_every_depends_target_is_a_resolvable_name(tree, defined_names):
    """The exact bug this file was written for: a renamed dependency still
    referenced by an old name."""
    unresolved = [
        (name, line) for name, line in _depends_targets(tree)
        if name not in defined_names
    ]
    assert not unresolved, (
        f"server.py references undefined names inside Depends(...): {unresolved}. "
        f"This would raise NameError at import time wherever fastapi is installed."
    )


def test_at_least_one_depends_target_was_actually_found(tree):
    """Guards the guard: if the AST walk stopped matching (say `Depends` gets
    imported under an alias), the check above would pass vacuously and silently
    stop protecting anything."""
    assert len(_depends_targets(tree)) > 0


def test_the_renamed_auth_dependency_is_gone_everywhere(tree):
    """`require_tenant_key` was replaced by `require_principal` in a hard cutover
    (agents.md #1: no compatibility shims). Any surviving reference is a
    leftover, not a second supported path."""
    source = SERVER_PATH.read_text(encoding="utf-8")
    offending = [
        (i, line.strip())
        for i, line in enumerate(source.splitlines(), start=1)
        if "require_tenant_key" in line and not line.lstrip().startswith("#")
    ]
    assert not offending, f"stale require_tenant_key references: {offending}"


def test_no_route_is_left_completely_unauthenticated_by_accident(tree):
    """Every route must either be deliberately public or carry an auth
    dependency. The allowlist is explicit so adding a new public route is a
    conscious edit to this test rather than an omission nobody notices."""
    public_paths = {
        "/health",
        "/v1/models",
        "/v1/providers",
        "/v1/orgs",                     # registration: no credential can exist yet
        "/v1/invitations/accept",       # resolves its own principal internally
        "/auth/sso/login",
        "/auth/sso/callback",
        "/auth/sso/me",                 # resolves its own principal internally
        "/auth/sso/logout",
        "/auth/sso/switch-org",
        "/auth/sso/backchannel-logout",
    }
    # Authenticated, but NOT by `require_principal` — kept separate from
    # `public_paths` because these are not public and mislabelling them would
    # make this guard read as if they were. `/metrics` spans every tenant, so no
    # tenant credential may reach it; it authenticates as infrastructure via
    # MODELROUTER_METRICS_TOKEN and 404s when that is unset.
    token_authenticated_paths = {"/metrics"}

    unauthenticated: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            route = _route_path(decorator)
            if route is None or route in public_paths or route in token_authenticated_paths:
                continue
            decorator_source = ast.dump(decorator)
            signature_source = ast.dump(ast.arguments(
                posonlyargs=node.args.posonlyargs, args=node.args.args,
                vararg=node.args.vararg, kwonlyargs=node.args.kwonlyargs,
                kw_defaults=[d for d in node.args.kw_defaults if d is not None],
                kwarg=node.args.kwarg, defaults=node.args.defaults,
            ))
            if "require_principal" not in decorator_source + signature_source:
                unauthenticated.append((route, node.name))

    assert not unauthenticated, (
        f"routes with neither auth nor an entry in the public allowlist: {unauthenticated}"
    )


def test_the_prometheus_endpoint_is_token_gated_and_fails_closed(tree):
    """`/metrics` exposes EVERY tenant's volume and spend, so it must (a) never
    accept a tenant credential and (b) 404 rather than serve openly when no
    operator token is configured. Verified statically because the route can't be
    exercised without fastapi installed.

    Inspected via AST with the DOCSTRING EXCLUDED — the docstring legitimately
    mentions `require_principal` while explaining why it is not used, and a raw
    text search reads that prose as if it were code."""
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "prometheus_metrics"
    )
    statements = function.body
    if (statements and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)):
        statements = statements[1:]        # drop the docstring
    code = "\n".join(ast.dump(s) for s in statements)
    signature = ast.dump(function.args)

    assert "METRICS_TOKEN_ENV_VAR" in code
    assert "compare_digest" in code, "token comparison must be constant-time"
    assert "404" in code, "an unset token must fail closed, not serve openly"
    assert "require_principal" not in code + signature, (
        "a tenant credential must never reach cross-tenant data"
    )


def _route_path(decorator: ast.Call) -> str | None:
    """The literal path from `@app.get("/x")` / `@router.post("/y")`."""
    callee = decorator.func
    if not isinstance(callee, ast.Attribute):
        return None
    if callee.attr not in ("get", "post", "put", "patch", "delete"):
        return None
    if not decorator.args:
        return None
    first = decorator.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None

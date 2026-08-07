"""CLI — `python -m modelrouter <command> ...`, mirroring this repo's other
labs' `python -m <lab> <command>` convention.

Keys come from modelrouter.config.Settings (environment variables / a .env
file — see .env.example), never as CLI arguments: a secret typed on a
command line ends up in shell history and `ps` output, which is a real,
avoidable leak (agents.md #7 — decide for the long term, not a convenient
shortcut). `--backend all` builds every adapter Settings can find a key for;
a specific `--backend <name>` builds just that one, still through Settings so
`providers` and `chat` never disagree about what's configured.

Every subcommand's failure path goes through core.errors.format_error_message
so a user sees the SAME classification/wording chat()/generate_image()/etc.
already computed internally — not a second, differently-worded explanation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from modelrouter.accounting import create_accounting_service
from modelrouter.config import KNOWN_PROVIDER_ENV_VARS, Settings
from modelrouter.core.errors import ConfigError, format_error_message
from modelrouter.core.types import ChatRequest, ImageGenerationRequest, SpeechRequest, TranscriptionRequest
from modelrouter.router import ModelRouter
from modelrouter.store.factory import resolve_backend
from modelrouter.tenancy import create_tenancy_repo

# Every provider name the CLI's --backend flag understands directly, beyond
# whatever Settings.build_adapters() finds configured — "fake" and "all" are
# CLI-only conveniences, not real provider names.
_KNOWN_BACKENDS = ["fake", "ollama", "all", *sorted(KNOWN_PROVIDER_ENV_VARS)]


def _ensure_utf8_streams() -> None:
    """Windows consoles often default to a legacy codepage (cp1252) rather
    than UTF-8, which mangles the em-dashes in core.errors' status messages
    into `?`/`�` — this reconfigures stdout/stderr to UTF-8 explicitly rather
    than degrading every message to plain ASCII to dodge the display bug.

    Only touches a genuine io.TextIOWrapper backed by a real OS file
    descriptor — NOT whatever a test harness (pytest's capsys, a redirected
    StringIO, etc.) has substituted for sys.stdout/stderr. Calling
    .reconfigure() on pytest's capture stream detaches it from capsys's
    redirection machinery entirely (writes silently escape to the real
    terminal instead of being captured) — checking isinstance(..., TextIOWrapper)
    with a working .fileno() is what actually distinguishes "a real console
    stream worth fixing the codepage on" from "something else is managing
    this stream and reconfiguring it would break that."""
    import io

    for stream in (sys.stdout, sys.stderr):
        if not isinstance(stream, io.TextIOWrapper):
            continue
        try:
            stream.fileno()   # raises for non-fd-backed streams (e.g. some test/CI harnesses)
        except (OSError, ValueError):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass   # not fatal to the command either way — worst case, the old mangled display


def _build_adapters(backend: str, settings: Settings) -> dict:
    def _report_skip(name: str, exc: Exception) -> None:
        print(f"  (skipping {name}: {exc})", file=sys.stderr)

    if backend == "fake":
        from modelrouter.providers.adapters import FakeProviderAdapter

        return {"fake": FakeProviderAdapter("fake")}
    if backend == "all":
        return settings.build_adapters(on_skip=_report_skip)
    if backend not in _KNOWN_BACKENDS:
        raise ValueError(f"unknown backend: {backend!r}. Known: {_KNOWN_BACKENDS}")
    return settings.build_adapters(include=frozenset({backend}), on_skip=_report_skip)


def _print_metadata(metadata, *, file=None) -> None:
    # file=None (resolved to the CURRENT sys.stderr inside the call), not a
    # `file=sys.stderr` default — a default argument is evaluated ONCE at
    # module-import time, so it would permanently bind to whatever sys.stderr
    # object existed then (e.g. under pytest, the real stream, before capsys
    # ever patches it) rather than whichever stream is live at call time.
    if file is None:
        file = sys.stderr
    print("\n--- metadata ---", file=file)
    print(json.dumps({
        "served_by": metadata.served_by,
        "attempt": metadata.attempt,
        "model_fallback_index": metadata.model_fallback_index,
        "skipped": [{"spec": s.spec, "reason": s.reason} for s in metadata.skipped],
        "pipeline": metadata.pipeline,
    }, indent=2), file=file)


def _print_failure(metadata) -> None:
    print(f"No provider succeeded. attempt={metadata.attempt}", file=sys.stderr)
    for skipped in metadata.skipped:
        print(f"  [skipped] {skipped.spec}: {skipped.reason}", file=sys.stderr)
    for a in metadata.attempts:
        detail = a.error_message or a.error_type or "?"
        print(f"  [attempt {a.attempt}] {a.provider}:{a.model} -> {a.outcome} ({detail})", file=sys.stderr)


async def _run_chat(args: argparse.Namespace) -> int:
    settings = Settings()
    adapters = _build_adapters(args.backend, settings)
    if not adapters:
        print("No adapters available for this backend. Run `python -m modelrouter providers` "
              "to see what's configured.", file=sys.stderr)
        return 1

    router = ModelRouter(adapters)
    request = ChatRequest(messages=[{"role": "user", "content": args.prompt}], model=args.models[0])

    if args.stream:
        return await _run_chat_streaming(router, request, args)

    response, metadata = await router.chat(request, models=args.models)

    if response is None:
        _print_failure(metadata)
        return 1

    print(response.choices[0].message["content"])
    if args.metadata:
        _print_metadata(metadata)
    return 0


async def _run_chat_streaming(router: ModelRouter, request: ChatRequest, args: argparse.Namespace) -> int:
    """`router.stream_chat()` end to end — the one streaming surface
    testable in an environment with no `fastapi` installed (see
    tests/test_cli.py), which is exactly why it exists alongside the SSE
    endpoint rather than as a substitute for it."""
    stream = router.stream_chat(request, models=args.models)
    try:
        async for delta in stream:
            if delta.content:
                print(delta.content, end="", flush=True)
        print()   # the final newline stdout would otherwise be missing
    except Exception as e:
        print(f"\nStream failed: {format_error_message(e)}", file=sys.stderr)
        return 1

    metadata = await stream.metadata()
    if metadata.served_by is None:
        _print_failure(metadata)
        return 1
    if args.metadata:
        _print_metadata(metadata)
    return 0


async def _run_image(args: argparse.Namespace) -> int:
    settings = Settings()
    adapters = _build_adapters(args.backend, settings)
    if not adapters:
        print("No adapters available for this backend.", file=sys.stderr)
        return 1

    router = ModelRouter(adapters)
    request = ImageGenerationRequest(prompt=args.prompt, model=args.models[0].partition(":")[2], n=args.n)
    response, metadata = await router.generate_image(request, models=args.models)

    if response is None:
        _print_failure(metadata)
        return 1

    for i, image in enumerate(response.images):
        print(image.url or f"(image {i}: base64 data, {len(image.b64_json or '')} chars)")
    if args.metadata:
        _print_metadata(metadata)
    return 0


async def _run_speech(args: argparse.Namespace) -> int:
    settings = Settings()
    adapters = _build_adapters(args.backend, settings)
    if not adapters:
        print("No adapters available for this backend.", file=sys.stderr)
        return 1

    router = ModelRouter(adapters)
    request = SpeechRequest(text=args.text, model=args.models[0].partition(":")[2])
    response, metadata = await router.speech(request, models=args.models)

    if response is None:
        _print_failure(metadata)
        return 1

    with open(args.out, "wb") as f:
        f.write(response.audio_bytes)
    print(f"Wrote {len(response.audio_bytes)} bytes to {args.out}")
    if args.metadata:
        _print_metadata(metadata)
    return 0


async def _run_transcribe(args: argparse.Namespace) -> int:
    settings = Settings()
    adapters = _build_adapters(args.backend, settings)
    if not adapters:
        print("No adapters available for this backend.", file=sys.stderr)
        return 1

    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    router = ModelRouter(adapters)
    request = TranscriptionRequest(
        audio_bytes=audio_bytes, model=args.models[0].partition(":")[2], filename=args.audio_file,
    )
    response, metadata = await router.transcribe(request, models=args.models)

    if response is None:
        _print_failure(metadata)
        return 1

    print(response.text)
    if args.metadata:
        _print_metadata(metadata)
    return 0


def _warn_if_ephemeral() -> None:
    """L1/L3's admin surface, for now: this CLI, not a separate HTTP admin
    API that doesn't exist yet (see ARCHITECTURE-PLAN.md's L2 registry
    section for the same reasoning applied there). `create_tenancy_repo()`/
    `create_accounting_service()` read MODELROUTER_STORAGE fresh on every
    invocation, same as server.py's lifespan does — with the zero-infra
    default (`memory`, or unset), whatever this command creates vanishes
    the moment the process exits, so a SEPARATE `keys create` afterward
    would find no tenant at all. Only meaningful across process boundaries
    (e.g. provisioning a tenant the running `serve` process can actually
    see) when MODELROUTER_STORAGE=sqlite points both at the same file."""
    if resolve_backend() == "memory":
        print(
            "NOTE: MODELROUTER_STORAGE is 'memory' (the default) — this tenant/key/credit "
            "will NOT persist beyond this command. Set MODELROUTER_STORAGE=sqlite (and "
            "MODELROUTER_SQLITE_PATH if you want a specific file) to provision something a "
            "separately-running `serve` process — or a later CLI invocation — can actually see.",
            file=sys.stderr,
        )


def _run_tenants_create(args: argparse.Namespace) -> int:
    _warn_if_ephemeral()
    repo = create_tenancy_repo()
    tenant = repo.create_tenant(args.name, token_ceiling=args.token_ceiling)
    print(f"Created tenant {tenant.tenant_id!r} ({tenant.name!r}), status={tenant.status}, "
          f"token_ceiling={tenant.token_ceiling}")
    return 0


def _run_tenants_list(_args: argparse.Namespace) -> int:
    repo = create_tenancy_repo()
    tenants = repo.list_tenants()
    if not tenants:
        print("No tenants. (If MODELROUTER_STORAGE=memory, that's expected — nothing persists across processes.)")
        return 0
    for t in tenants:
        print(f"  {t.tenant_id:<24} {t.name:<20} {t.status}")
    return 0


def _run_keys_create(args: argparse.Namespace) -> int:
    from modelrouter.core.errors import TenantNotFoundError

    _warn_if_ephemeral()
    repo = create_tenancy_repo()
    try:
        record, plaintext_key = repo.create_api_key(
            args.tenant_id, args.name, budget_usd=args.budget_usd, token_ceiling=args.token_ceiling,
        )
    except TenantNotFoundError as e:
        print(f"Error: {e}. Run `tenants list` to see what exists.", file=sys.stderr)
        return 1
    print(f"Created key {record.key_id!r} for tenant {args.tenant_id!r}.")
    print(f"API key (shown once, never again — copy it now): {plaintext_key}")
    return 0


def _run_keys_list(args: argparse.Namespace) -> int:
    repo = create_tenancy_repo()
    keys = repo.list_api_keys(args.tenant_id)
    if not keys:
        print("No keys for this tenant.")
        return 0
    for k in keys:
        print(f"  {k.key_id:<24} {k.prefix:<12} {k.name:<20} {k.status}")
    return 0


def _run_keys_revoke(args: argparse.Namespace) -> int:
    repo = create_tenancy_repo()
    repo.revoke_api_key(args.key_id)
    print(f"Revoked {args.key_id!r} (a no-op if it was already gone or never existed).")
    return 0


def _run_credits_add(args: argparse.Namespace) -> int:
    from modelrouter.core.errors import TenantNotFoundError

    _warn_if_ephemeral()
    repo = create_tenancy_repo()
    if repo.get_tenant(args.tenant_id) is None:
        print(f"Error: {TenantNotFoundError(args.tenant_id)}. Run `tenants list` to see what exists.", file=sys.stderr)
        return 1
    accounting = create_accounting_service()
    accounting.purchase_credits(args.tenant_id, args.amount_usd)
    account = accounting.balance(args.tenant_id)
    print(f"Purchased ${args.amount_usd:.2f} for tenant {args.tenant_id!r}. Now available: ${account.available_usd:.2f}.")
    return 0


def _run_usage(args: argparse.Namespace) -> int:
    accounting = create_accounting_service()
    account = accounting.balance(args.tenant_id)
    print(f"Tenant {args.tenant_id!r}:")
    print(f"  purchased:  ${account.purchased_usd:.6f}")
    print(f"  spent:      ${account.spent_usd:.6f}")
    print(f"  reserved:   ${account.reserved_usd:.6f}")
    print(f"  available:  ${account.available_usd:.6f}")
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    """Launches the HTTP server (modelrouter.server:app) via uvicorn — the
    "run this once, call it from another application" deployment mode. Lazy
    imports uvicorn (same graceful-degradation contract as every real
    provider SDK) so `python -m modelrouter chat ...` keeps working without
    fastapi/uvicorn installed; only `serve` needs the `server` extra."""
    try:
        import uvicorn
    except ImportError:
        print(
            "The HTTP server needs the optional `server` extra: "
            "pip install -e '.[server]'  (or: pip install fastapi uvicorn[standard] python-multipart)",
            file=sys.stderr,
        )
        return 2
    uvicorn.run("modelrouter.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _run_providers(_args: argparse.Namespace) -> int:
    """Answers "where do my keys go, and what's actually configured right
    now" directly from the CLI — no need to read config.py's source to find
    out. Ollama is listed separately since it needs no key at all."""
    settings = Settings()
    print("Provider API keys (via environment variables or a .env file):")
    for provider_name, env_var in sorted(KNOWN_PROVIDER_ENV_VARS.items()):
        status = "configured" if settings.get(provider_name) else "not set"
        print(f"  {provider_name:<12} {env_var:<22} {status}")
    print("\nOllama needs no API key — it's available if a local server is running.")
    print("\nSee .env.example for the exact keys to set, and README.md's Configuration section.")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_streams()

    parser = argparse.ArgumentParser(prog="python -m modelrouter")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=_KNOWN_BACKENDS, default="fake")
        p.add_argument(
            "--models", nargs="+", required=True,
            help="Ordered provider:model fallback array, e.g. anthropic:claude-opus-4-5 openai:gpt-5.4-mini",
        )
        p.add_argument("--metadata", action="store_true", help="Print RouterMetadata to stderr")

    chat = sub.add_parser("chat", help="Send one chat request through the router")
    chat.add_argument("prompt", help="The user message")
    chat.add_argument("--stream", action="store_true",
                       help="Stream the response token-by-token (Phase 3) instead of waiting for completion")
    _add_common_args(chat)

    image = sub.add_parser("image", help="Generate an image")
    image.add_argument("prompt", help="The image prompt")
    image.add_argument("--n", type=int, default=1, help="Number of images to generate")
    _add_common_args(image)

    speech = sub.add_parser("speech", help="Text-to-speech")
    speech.add_argument("text", help="The text to speak")
    speech.add_argument("--out", default="speech.mp3", help="Output audio file path")
    _add_common_args(speech)

    transcribe = sub.add_parser("transcribe", help="Speech-to-text")
    transcribe.add_argument("audio_file", help="Path to an audio file")
    _add_common_args(transcribe)

    sub.add_parser("providers", help="List provider API-key configuration status")

    serve = sub.add_parser("serve", help="Run the HTTP server (needs the `server` extra)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development only)")

    # ── L1/L3 admin surface — the CLI, since there's no admin HTTP API yet ──
    tenants = sub.add_parser("tenants", help="Manage tenants (L1)")
    tenants_sub = tenants.add_subparsers(dest="tenants_command", required=True)
    tenants_create = tenants_sub.add_parser("create", help="Create a tenant")
    tenants_create.add_argument("name")
    tenants_create.add_argument("--token-ceiling", type=int, default=None,
                                 help="Tenant-wide max_tokens ceiling (Part 3.3) — min()'d with any per-key ceiling")
    tenants_sub.add_parser("list", help="List tenants")

    keys = sub.add_parser("keys", help="Manage API keys (L1)")
    keys_sub = keys.add_subparsers(dest="keys_command", required=True)
    keys_create = keys_sub.add_parser("create", help="Create an API key for a tenant")
    keys_create.add_argument("tenant_id")
    keys_create.add_argument("--name", default="cli-issued key")
    keys_create.add_argument("--budget-usd", type=float, default=None)
    keys_create.add_argument("--token-ceiling", type=int, default=None)
    keys_list = keys_sub.add_parser("list", help="List a tenant's API keys (never shows plaintext)")
    keys_list.add_argument("tenant_id")
    keys_revoke = keys_sub.add_parser("revoke", help="Revoke an API key")
    keys_revoke.add_argument("key_id")

    credits_ = sub.add_parser("credits", help="Manage tenant credit (L3)")
    credits_sub = credits_.add_subparsers(dest="credits_command", required=True)
    credits_add = credits_sub.add_parser("add", help="Purchase credit for a tenant")
    credits_add.add_argument("tenant_id")
    credits_add.add_argument("amount_usd", type=float)

    usage = sub.add_parser("usage", help="Show a tenant's credit balance (L3)")
    usage.add_argument("tenant_id")

    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return asyncio.run(_run_chat(args))
        if args.command == "image":
            return asyncio.run(_run_image(args))
        if args.command == "speech":
            return asyncio.run(_run_speech(args))
        if args.command == "transcribe":
            return asyncio.run(_run_transcribe(args))
        if args.command == "providers":
            return _run_providers(args)
        if args.command == "serve":
            return _run_serve(args)
        if args.command == "tenants":
            if args.tenants_command == "create":
                return _run_tenants_create(args)
            if args.tenants_command == "list":
                return _run_tenants_list(args)
        if args.command == "keys":
            if args.keys_command == "create":
                return _run_keys_create(args)
            if args.keys_command == "list":
                return _run_keys_list(args)
            if args.keys_command == "revoke":
                return _run_keys_revoke(args)
        if args.command == "credits":
            if args.credits_command == "add":
                return _run_credits_add(args)
        if args.command == "usage":
            return _run_usage(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2
    except Exception as e:   # last-resort: still a real message, not a raw traceback
        print(f"Unexpected error: {format_error_message(e)}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

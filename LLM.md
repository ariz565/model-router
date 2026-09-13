# ModelRouter engineering guide

ModelRouter is a self-hosted, multi-provider LLM gateway. It routes requests, enforces tenant isolation and budgets, and records operational evidence. Changes must preserve those guarantees across the library, CLI, HTTP server, and supported storage backends.

## Working principles

- Prefer small, readable changes that solve the stated problem completely.
- Treat API keys, prompts, tenant data, and billing records as sensitive. Never log, commit, or expose them.
- Do not silently fall back to a weaker storage, authorization, encryption, or billing mode. Fail clearly during startup or configuration validation.
- Keep every storage-backed domain on a coherent backend. A multi-replica deployment requires shared, atomic state for authentication, authorization, accounting, and traces.
- Use dependency injection for external services and make optional integrations explicit.
- Preserve wire compatibility deliberately. Validate boundary inputs and reject unsupported shapes clearly.

## Testing and verification

- Add regression tests for every bug fix and meaningful tests for new behavior.
- Test behavior rather than implementation details.
- Run focused tests while developing, then run the full suite before handoff.
- Keep test dependencies compatible and bounded. Update dependency constraints when a supported resolution breaks the suite.
- Do not use real customer data or live provider credentials in tests.

## Code quality

- Prefer standard libraries and established protocol implementations over custom substitutes.
- Keep functions focused and use early returns to avoid deep nesting.
- Use precise types and validate untrusted input at the boundary.
- Avoid speculative abstractions and unnecessary configuration.
- Add comments only when they explain non-obvious security, concurrency, or protocol decisions. Keep them short and current.
- Keep Python lines at or below 120 characters.

## Production changes

- A deployment configuration must be executable as written. Add a smoke test for startup-critical settings.
- Multi-replica accounting must make reservations atomically in shared storage before provider work begins.
- Health endpoints must distinguish process liveness from readiness of required dependencies.
- Apply request-size limits and rate limiting at the service boundary or enforce and document a required upstream control.
- Pin compatible dependency ranges and keep runtime configuration in environment variables or a secrets manager.

## Repository hygiene

- Do not commit runtime databases, credentials, generated artifacts, or local environment files.
- Preserve unrelated user changes in a dirty worktree.
- Do not use destructive Git commands unless explicitly requested.

"""Time-ordered, non-enumerable identifiers — UUIDv7 (RFC 9562 §5.7) behind a
prefixed-string convention matching the `tn_`/`key_` shape L1 already uses.

**Why not `uuid4()`, which the rest of this codebase currently uses.** Two
independent reasons, both structural rather than stylistic:

1. **Index locality.** A random UUID as a primary key inserts at a random
   point in the B-tree every time, splitting pages and thrashing cache. A
   time-ordered ID appends, which is the difference between an index that
   stays healthy at millions of rows and one that steadily degrades. This is
   the documented reason UUIDv7/ULID exist at all.
2. **Enumerability.** Sequential integers (`/projects/1`, `/projects/2`) let
   anyone measure your volume and sweep for IDOR. UUIDv7 leaks a creation
   timestamp — which for an org/project/user record is not sensitive — and
   nothing else.

**Why implemented here rather than taking a dependency** (`agents.md` #5 says
prefer established libraries, so this needs justifying, not assuming): the
layout is 10 lines of fully-specified bit packing from a published RFC, with
no algorithm to get subtly wrong and nothing to keep up with. Python's stdlib
`uuid.uuid7()` exists only in 3.14+, and this project supports 3.10+ — so the
alternative is a dependency for ~10 lines, which is the wrong trade. When
3.14 is the floor, this becomes a one-line delegation to the stdlib.

**IDs are opaque to callers and must never be parsed for authorization.**
`identity/` always scopes a lookup by `(tenant_id, resource_id)`, never by
`resource_id` alone — an unguessable ID raises the cost of an IDOR attempt but
authorizes nothing on its own.
"""

from __future__ import annotations

import secrets
import time
import uuid

__all__ = ["uuid7", "new_id"]


def uuid7() -> uuid.UUID:
    """RFC 9562 §5.7 layout: 48-bit big-endian Unix timestamp in
    milliseconds, 4-bit version (7), 12 bits random, 2-bit variant (0b10),
    62 bits random. The 74 random bits are drawn from `secrets`, not
    `random` — these end up in URLs, so they must not be predictable from a
    seeded PRNG.

    **Ordering is millisecond-granular, not strictly monotonic.** IDs minted
    within the same millisecond sort by their random bits, i.e. arbitrarily.
    RFC 9562 permits replacing the 12 `rand_a` bits with a sub-millisecond
    counter to get strict monotonicity; that is deliberately NOT done here,
    because it requires process-wide mutable state (a lock, a last-timestamp,
    a counter) and buys nothing for the reason this function exists — every ID
    minted in a given millisecond lands in the same region of the index
    whatever their internal order. Callers must therefore never treat these as
    a sequence number: `EventStore.seq` is what this codebase uses when true
    ordering is required."""
    timestamp_ms = int(time.time() * 1000)
    raw = bytearray(timestamp_ms.to_bytes(6, "big") + secrets.token_bytes(10))
    raw[6] = (raw[6] & 0x0F) | 0x70   # version 7 in the high nibble of byte 6
    raw[8] = (raw[8] & 0x3F) | 0x80   # RFC 4122/9562 variant in the top 2 bits of byte 8
    return uuid.UUID(bytes=bytes(raw))


def new_id(prefix: str) -> str:
    """`f"{prefix}_{uuid7().hex}"` — e.g. `usr_0192f3c4d5e678901234567890abcdef`.

    The prefix is a real operational feature, not decoration: an ID that says
    what it is turns "why is this 404ing" into a one-glance answer when a
    caller passes a workspace ID to a project endpoint. The full 32 hex chars
    are kept (never truncated the way L1's older `uuid4().hex[:16]` records
    are) because truncation would cut into the random half — the leading 12
    characters are all timestamp, so a short prefix of a UUIDv7 is mostly
    predictable, which is exactly the property you don't want in a URL."""
    if not prefix or not prefix.isalnum():
        raise ValueError(f"prefix must be a non-empty alphanumeric string, got {prefix!r}")
    return f"{prefix}_{uuid7().hex}"

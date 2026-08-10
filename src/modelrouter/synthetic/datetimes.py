"""Datetime parsing, epoch conversion, and formatting — the ONE place that
understands timestamp representations, shared by profiling, generation, and
chronology enforcement.

**Why this exists as its own module rather than being inlined three times.**
Parsing a database's datetime representation and re-emitting it in the same shape
are the same knowledge used twice, in opposite directions. Keeping that knowledge
in one place is what guarantees round-trip fidelity: whatever `parse_datetime()`
can read, `format_datetime()` can write back in the same shape, and neither
`profiling.py` nor `engines/statistical.py` has to agree with the other about
format conventions by coincidence.

**Epoch seconds as the universal internal representation.** A timestamp is a
monotonic numeric quantity, so every statistical operation this platform already
has for numbers — quantile ladders, interpolation, correlation — applies to a
timestamp unchanged once it's epoch seconds. That's why `DatetimeProfile` stores
epoch quantiles rather than inventing a second, timestamp-specific statistics
module.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

__all__ = [
    "parse_datetime", "to_epoch_seconds", "from_epoch_seconds", "format_datetime",
    "is_date_only",
]

# Formats tried in order after `fromisoformat()` (which already covers the
# overwhelming majority of real data: SQLite/Postgres ISO-8601 with 'T' or space
# separators). These cover the handful of common non-ISO conventions actually seen
# in enterprise exports — trailing 'Z' (which stdlib `fromisoformat` only accepts
# from 3.11) and a bare SQL DATE with no time component at all.
_FALLBACK_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y")


def parse_datetime(value) -> datetime | None:
    """Accepts whatever a `DataSource` might hand back for a datetime column:
    a native `datetime`/`date` (the in-memory / driver-converted case), an epoch
    number, or a string in one of several common conventions.

    Returns `None` for anything unparseable rather than raising — a single
    malformed row must not abort profiling an entire column; the null-rate
    accounting in `profiling.py` already treats "couldn't get a value" and "NULL"
    as the same practical outcome."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    # `fromisoformat` handles the 'Z' UTC suffix only from Python 3.11; this
    # project's floor is 3.10, so it is normalized by hand rather than raising a
    # confusing error on a perfectly valid ISO-8601 timestamp.
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for pattern in _FALLBACK_FORMATS:
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def to_epoch_seconds(value: datetime) -> float:
    return value.timestamp()


def from_epoch_seconds(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def is_date_only(values: list[datetime]) -> bool:
    """True when every observed instant fell exactly at midnight — i.e. the
    source column is genuinely a DATE, not a DATETIME that happens to be sparse.

    Read from the DATA rather than trusted from the declared SQL type: a column
    declared `TIMESTAMP` that has only ever been written date-only values should
    still round-trip as a date, and getting this from real observations rather
    than a type string is one fewer thing that can disagree with reality."""
    if not values:
        return False
    return all(
        v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0
        for v in values
    )


def format_datetime(value: datetime, *, date_only: bool) -> str:
    """The inverse of `parse_datetime()` for the shapes this module produces:
    `date_only` emits a bare `YYYY-MM-DD`; otherwise a full ISO-8601 instant.

    Always UTC and always explicit about it (`+00:00`, never a bare offset-naive
    string) — an offset-naive timestamp re-inserted into a database that assumes a
    different timezone convention is a correctness bug, not a formatting nicety."""
    if date_only:
        return value.date().isoformat()
    return value.astimezone(timezone.utc).isoformat()

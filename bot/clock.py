"""
Single source of truth for "what time is it" and "what trading day is it".

WHY THIS MODULE EXISTS (this is a real correctness issue, not style):

The bot's daily-loss-limit reset, the risk overlay's intraday kill-switch
reset, and the compliance day boundary ALL depend on agreeing on when "a new
day" starts. Before this module, three different call sites used three
different mechanisms:

  - `orchestrator.run_cycle()`      -> `datetime.utcnow()`  (naive, deprecated)
  - `data_feed.ReplayDataFeed`      -> `date.today()`       (SERVER LOCAL TIME)
  - `compliance.update_equity()`    -> `datetime.utcnow()`  (naive, deprecated)

`date.today()` reads the host machine's local timezone. A bot deployed on a
VPS in, say, Sydney (UTC+10/+11) would roll its trading day over 10-11 hours
before the exchange does. That would silently reset the daily loss limit at
the wrong time -- allowing a second full -$10,000 daily loss inside a single
real exchange day, which is exactly the failure mode that busts a prop eval.
`utcnow()` returns a NAIVE datetime, which compares incorrectly against any
aware datetime and is deprecated in Python 3.12+ (this sandbox runs 3.14).

Everything now goes through here, is timezone-AWARE, and is UTC-anchored.

RESET HOUR: Bybit derivatives settle and roll their daily boundary at
00:00 UTC. HyroTrader's daily drawdown is evaluated against a daily starting
balance on the same UTC-midnight boundary. `DAILY_RESET_HOUR_UTC = 0` encodes
that. It is exposed as a module constant rather than hardcoded inline so that
if HyroTrader is ever confirmed to use a different cutoff (e.g. a broker-local
5pm New York roll, which some firms use), it is a one-line change in one place
rather than a hunt through three modules.

NOTE ON THE RESET HOUR: 00:00 UTC is the operationally standard reading and
matches Bybit's own settlement boundary, but it has NOT been independently
confirmed with HyroTrader support. It is flagged in BUILD_PLAN.md alongside
the (also unconfirmed) Phase 1->2 drawdown-floor carry-over question.
"""
import datetime as dt
from typing import Optional

UTC = dt.timezone.utc

# Hour of the UTC day at which the trading day rolls over. See module docstring.
DAILY_RESET_HOUR_UTC = 0


def now_utc() -> dt.datetime:
    """Current time as a timezone-AWARE UTC datetime. The only clock read."""
    return dt.datetime.now(UTC)


def ensure_utc(ts: dt.datetime) -> dt.datetime:
    """
    Coerce a datetime to aware-UTC.

    A NAIVE datetime is INTERPRETED AS UTC rather than as server-local time.
    This is the safe choice here: every naive timestamp this bot produces
    internally (state files written by older versions, test fixtures, CLI
    `--as-of` parsing) was produced on a UTC-intent code path. Interpreting
    naive as local would silently reintroduce the exact bug this module fixes.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def trading_day(ts: Optional[dt.datetime] = None) -> dt.date:
    """
    The trading-day date that timestamp `ts` falls within, honouring
    DAILY_RESET_HOUR_UTC. With the default reset hour of 0 this is simply the
    UTC calendar date; the offset arithmetic is kept so that changing the
    reset hour actually works instead of silently doing nothing.
    """
    ts = ensure_utc(ts or now_utc())
    return (ts - dt.timedelta(hours=DAILY_RESET_HOUR_UTC)).date()


def next_reset(ts: Optional[dt.datetime] = None) -> dt.datetime:
    """Aware-UTC timestamp of the next daily rollover strictly after `ts`."""
    ts = ensure_utc(ts or now_utc())
    today = trading_day(ts)
    candidate = dt.datetime.combine(
        today, dt.time(hour=DAILY_RESET_HOUR_UTC), tzinfo=UTC
    )
    if candidate <= ts:
        candidate += dt.timedelta(days=1)
    return candidate


def parse_iso(s: Optional[str]) -> Optional[dt.datetime]:
    """Parse an ISO timestamp from the state file into aware-UTC, or None."""
    if not s:
        return None
    return ensure_utc(dt.datetime.fromisoformat(s))


def to_iso(ts: dt.datetime) -> str:
    """Serialize an aware-UTC datetime for the state file."""
    return ensure_utc(ts).isoformat()

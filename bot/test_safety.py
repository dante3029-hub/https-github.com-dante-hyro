#!/usr/bin/env python3
"""
Safety-critical regression tests. These cover the failure modes that lose
money or bust the eval account, as opposed to the wiring tests in
smoke_test_bot.py. Every test here corresponds to a bug that was actually
found in this codebase, not a hypothetical.

Run: python3 bot/test_safety.py
"""
import sys
import os
import datetime as dt

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import numpy as np

from bot import clock
from bot.orchestrator import HyroTraderBot, _due
from bot.data_feed import DataFeed, MarketSnapshot
from portfolio_layer.portfolio import SLEEVE_NAMES

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------- fake feed
class _FakeSnapshot:
    """Minimal stand-in for signal_engine's snapshot -- every sleeve holds
    a single +1.0 BTC weight so any failure to flatten is immediately
    visible as non-zero gross notional."""
    def __init__(self):
        w = {"BTC": 1.0}
        self.main_weights = dict(w)
        self.flow_weights = dict(w)
        self.delta_weights = dict(w)
        self.relvol_weights = dict(w)
        self.universe_a_coins = ["BTC"]
        self.universe_b_coins = ["BTC"]


class FakeFeed(DataFeed):
    """Deterministic feed -- no CSV / network reads, so these tests are fast
    and cannot be perturbed by data staleness."""
    def get_snapshot(self, as_of_date, short_tracker_state, bos_tracker_state):
        rng = np.random.default_rng(0)
        hist = {k: rng.normal(0, 0.01, 300) for k in SLEEVE_NAMES}
        return MarketSnapshot(
            as_of_date=as_of_date or dt.date(2026, 8, 9),
            signal_snapshot=_FakeSnapshot(),
            sleeve_histories=hist,
            short_weights={"BTC": 1.0},
            bos_weights={"BTC": 1.0},
            data_source="fake",
        )


def _fresh_bot(tmp):
    if os.path.exists(tmp):
        os.remove(tmp)
    return HyroTraderBot(data_feed=FakeFeed(), dry_run=True, state_path=tmp)


# ------------------------------------------------------------------ tests
def test_kill_switch_dominates_cadence():
    """
    BUG #4 (severe). size_portfolio() correctly zeroes every sleeve when the
    kill switch trips, but the orchestrator's cadence carry-forward used to
    restore the PREVIOUS cycle's non-zero targets for every sleeve that
    wasn't due to rebalance -- re-arming the book at the exact moment it must
    be flat. Reproduced pre-fix at $45,460.50 gross with the switch tripped.
    """
    print("\ntest_kill_switch_dominates_cadence")
    tmp = "/tmp/_safety_ks.json"
    bot = _fresh_bot(tmp)
    t0 = dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)

    r1 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0)
    check("cycle 1 establishes a non-zero book", r1["gross_notional"] > 0,
          f"gross=${r1['gross_notional']:,.2f}")

    # trip the -$3,000 intraday kill switch, then run a cycle where only the
    # 4h-cadence sleeves are even eligible to rebalance.
    rs = bot.state.get_risk_state()
    tripped = rs.on_intraday_pnl_update(-3100.0)
    bot.state.set_risk_state(rs)
    check("kill switch trips at -$3,100", tripped)

    r2 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0 + dt.timedelta(hours=5))
    check("account_multiplier is 0.0 once tripped", r2["account_multiplier"] == 0.0,
          f"got {r2['account_multiplier']}")
    check("gross notional forced to $0.00 (kill switch beats cadence)",
          r2["gross_notional"] == 0.0, f"got ${r2['gross_notional']:,.2f}")
    check("forced_flat flag is set", r2.get("forced_flat") is True)
    check("no stale target survives in persisted state",
          all(not v for v in bot.state.last_dollar_targets.values()),
          f"{ {k: len(v) for k, v in bot.state.last_dollar_targets.items()} }")


def test_kill_switch_clears_next_session():
    """The kill switch is a PER-SESSION block, not permanent. After a new
    session starts the bot must be able to re-arm -- otherwise a single bad
    morning ends the whole eval attempt."""
    print("\ntest_kill_switch_clears_next_session")
    tmp = "/tmp/_safety_ks2.json"
    bot = _fresh_bot(tmp)
    t0 = dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)
    bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0)
    rs = bot.state.get_risk_state()
    rs.on_intraday_pnl_update(-3100.0)
    bot.state.set_risk_state(rs)
    r_killed = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0 + dt.timedelta(hours=5))
    check("flat while tripped", r_killed["gross_notional"] == 0.0)

    r_next = bot.run_cycle(as_of_date=dt.date(2026, 8, 10), now=t0 + dt.timedelta(hours=30))
    check("re-arms on the next session", r_next["gross_notional"] > 0,
          f"gross=${r_next['gross_notional']:,.2f}")
    check("account_multiplier restored to L=1.70", abs(r_next["account_multiplier"] - 1.70) < 1e-9,
          f"got {r_next['account_multiplier']}")


def test_compliance_bust_forces_flat():
    """A busted account must go flat, not keep trading."""
    print("\ntest_compliance_bust_forces_flat")
    tmp = "/tmp/_safety_comp.json"
    bot = _fresh_bot(tmp)
    t0 = dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)
    bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0)
    # drive equity below the $180,000 max-drawdown floor
    bot.state.equity = 179_000.0
    r = bot.run_cycle(as_of_date=dt.date(2026, 8, 10), now=t0 + dt.timedelta(hours=30))
    check("compliance reports busted", r["compliance"]["busted"] is True)
    check("gross notional forced to $0.00 on bust", r["gross_notional"] == 0.0,
          f"got ${r['gross_notional']:,.2f}")
    check("forced_flat flag set on bust", r.get("forced_flat") is True)


def test_naive_timestamp_upgrade_does_not_crash():
    """
    BUG #3. A state file written by the previous build stored NAIVE
    timestamps. Once run_cycle() switched to aware-UTC, subtracting the two
    raised TypeError -- i.e. the bot crashed on first cycle after upgrade.
    """
    print("\ntest_naive_timestamp_upgrade_does_not_crash")
    legacy_naive = dt.datetime(2026, 8, 9, 0, 0, 0).isoformat()
    aware_now = dt.datetime(2026, 8, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
    try:
        due = _due(legacy_naive, 4, aware_now)
        check("_due tolerates a legacy naive timestamp", due is True, f"due={due}")
    except TypeError as e:
        check("_due tolerates a legacy naive timestamp", False, str(e))


def test_trading_day_is_utc_not_local():
    """
    BUG #2. date.today() reads the HOST timezone. On a Sydney VPS (UTC+10)
    the trading day would roll over ~10h early, resetting the daily loss
    limit mid-exchange-day and permitting a second full -$10,000 daily loss.
    """
    print("\ntest_trading_day_is_utc_not_local")
    late = dt.datetime(2026, 8, 9, 23, 30, tzinfo=dt.timezone.utc)
    early = dt.datetime(2026, 8, 10, 0, 30, tzinfo=dt.timezone.utc)
    check("23:30Z is still 2026-08-09", clock.trading_day(late) == dt.date(2026, 8, 9),
          str(clock.trading_day(late)))
    check("00:30Z is already 2026-08-10", clock.trading_day(early) == dt.date(2026, 8, 10),
          str(clock.trading_day(early)))
    check("now_utc() is timezone-aware", clock.now_utc().tzinfo is not None)


def test_dry_run_never_places_orders():
    """Belt-and-braces: the dry-run path must not touch an exchange client."""
    print("\ntest_dry_run_never_places_orders")

    class ExplodingClient:
        def __getattr__(self, item):
            raise AssertionError(f"dry-run must not call exchange_client.{item}")

    tmp = "/tmp/_safety_dry.json"
    if os.path.exists(tmp):
        os.remove(tmp)
    bot = HyroTraderBot(data_feed=FakeFeed(), exchange_client=ExplodingClient(),
                        dry_run=True, state_path=tmp)
    r = bot.run_cycle(as_of_date=dt.date(2026, 8, 9),
                      now=dt.datetime(2026, 8, 9, tzinfo=dt.timezone.utc))
    check("dry-run completed without touching the client", r["execution"].startswith("SKIPPED"))


def test_live_requires_client():
    print("\ntest_live_requires_client")
    try:
        HyroTraderBot(data_feed=FakeFeed(), exchange_client=None, dry_run=False,
                      state_path="/tmp/_safety_live.json")
        check("refuses live with no exchange client", False, "constructor did not raise")
    except ValueError as e:
        check("refuses live with no exchange client", True, str(e)[:60])


if __name__ == "__main__":
    test_kill_switch_dominates_cadence()
    test_kill_switch_clears_next_session()
    test_compliance_bust_forces_flat()
    test_naive_timestamp_upgrade_does_not_crash()
    test_trading_day_is_utc_not_local()
    test_dry_run_never_places_orders()
    test_live_requires_client()

    print(f"\n{'='*60}")
    print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL SAFETY TESTS PASSED")

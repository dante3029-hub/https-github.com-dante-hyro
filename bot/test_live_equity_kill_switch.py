#!/usr/bin/env python3
"""
End-to-end verification that the -$3,000 kill switch and drawdown throttle
actually fire off the LIVE equity mark path (run_cycle -> state.equity ->
risk_state.update_equity()/on_intraday_pnl_update()), not just off a manually
poked risk_state as in test_safety.py. That manual-poke pattern proves
size_portfolio() respects an already-tripped risk_state; it does NOT prove
anything ever trips risk_state in the live cycle itself. This test does not
touch risk_state directly at all -- only bot.state.equity, exactly like a
real exchange wallet-balance read would.

Run: python3 bot/test_live_equity_kill_switch.py
"""
import sys
import os
import datetime as dt

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import numpy as np

from bot.orchestrator import HyroTraderBot
from bot.data_feed import DataFeed, MarketSnapshot
from portfolio_layer.portfolio import SLEEVE_NAMES

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class _FakeSnapshot:
    def __init__(self):
        w = {"BTC": 1.0}
        self.main_weights = dict(w)
        self.flow_weights = dict(w)
        self.delta_weights = dict(w)
        self.relvol_weights = dict(w)
        self.universe_a_coins = ["BTC"]
        self.universe_b_coins = ["BTC"]


class FakeFeed(DataFeed):
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


def test_live_equity_dip_trips_kill_switch_same_cycle():
    print("\ntest_live_equity_dip_trips_kill_switch_same_cycle")
    tmp = "/tmp/_live_ks.json"
    bot = _fresh_bot(tmp)
    t0 = dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)

    r1 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0)
    day_start_equity = bot.state.get_compliance_state(None if False else __import__("bot.config", fromlist=["HYRO_COMPLIANCE_CONFIG"]).HYRO_COMPLIANCE_CONFIG).day_start_equity
    check("cycle 1 establishes a non-zero book", r1["gross_notional"] > 0,
          f"gross=${r1['gross_notional']:,.2f}")
    check("cycle 1 account_multiplier at full L (no throttle, no trip)",
          abs(r1["account_multiplier"] - 1.70) < 1e-9, f"got {r1['account_multiplier']}")

    # Simulate a genuine intraday mark-to-market loss of $3,500 -- crosses the
    # -$3,000 kill switch threshold but is BELOW the $10k daily loss floor,
    # i.e. exactly the gap the kill switch exists to cover on its own.
    bot.state.equity = day_start_equity - 3500.0
    r2 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0 + dt.timedelta(hours=4))
    check("kill switch trips off the LIVE equity mark (no manual risk_state poke)",
          r2["account_multiplier"] == 0.0, f"got {r2['account_multiplier']}")
    check("book forced flat THIS cycle (same cycle as the equity mark that tripped it)",
          r2["gross_notional"] == 0.0, f"got ${r2['gross_notional']:,.2f}")
    check("forced_flat flag set", r2.get("forced_flat") is True)
    check("compliance NOT busted (this is well inside the $10k daily floor -- "
          "the kill switch, not compliance, is what caught it)",
          r2["compliance"]["busted"] is False, f"{r2['compliance']}")

    # Equity recovers intraday (unrealized mark bounces back) -- kill switch
    # must NOT self-clear mid-session.
    bot.state.equity = day_start_equity - 500.0
    r3 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0 + dt.timedelta(hours=8))
    check("stays flat after partial recovery, same session (kill switch is sticky for the day)",
          r3["gross_notional"] == 0.0, f"got ${r3['gross_notional']:,.2f}")

    # Next calendar day -- must re-arm.
    r4 = bot.run_cycle(as_of_date=dt.date(2026, 8, 10), now=t0 + dt.timedelta(hours=30))
    check("re-arms next session", r4["gross_notional"] > 0, f"gross=${r4['gross_notional']:,.2f}")
    check("account_multiplier restored to L=1.70 next session",
          abs(r4["account_multiplier"] - 1.70) < 1e-9, f"got {r4['account_multiplier']}")


def test_live_equity_dip_below_kill_switch_but_within_normal_dd_does_not_trip():
    """Sanity check the threshold itself: a $2,000 intraday dip (below the
    $3,000 kill switch) must NOT trip anything -- guards against an
    off-by-threshold error in the wiring."""
    print("\ntest_live_equity_dip_below_kill_switch_but_within_normal_dd_does_not_trip")
    tmp = "/tmp/_live_ks2.json"
    bot = _fresh_bot(tmp)
    t0 = dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)
    r1 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0)
    from bot import config as _cfg
    day_start_equity = bot.state.get_compliance_state(_cfg.HYRO_COMPLIANCE_CONFIG).day_start_equity

    bot.state.equity = day_start_equity - 2000.0
    r2 = bot.run_cycle(as_of_date=dt.date(2026, 8, 9), now=t0 + dt.timedelta(hours=4))
    check("$2,000 intraday dip (below $3,000 threshold) does NOT trip the kill switch",
          r2["account_multiplier"] == 1.70, f"got {r2['account_multiplier']}")
    check("book stays live at $2,000 dip", r2["gross_notional"] > 0,
          f"gross=${r2['gross_notional']:,.2f}")


if __name__ == "__main__":
    test_live_equity_dip_trips_kill_switch_same_cycle()
    test_live_equity_dip_below_kill_switch_but_within_normal_dd_does_not_trip()
    print(f"\nPASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)

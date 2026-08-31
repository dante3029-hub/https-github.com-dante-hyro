#!/usr/bin/env python3
"""
End-to-end integration check: proves the STATIC drawdown default actually
takes effect through the real orchestrator.run_cycle() path (bot/config.py
-> bot/state.py -> portfolio_layer/compliance.py), not just in an isolated
unit test of compliance.py.

Scenario: run the real orchestrator through a Phase-1 peak to $232,000, roll
to Phase 2, then mark equity down to $206,000 (a routine 3% dip). Under the
OLD trailing-default behavior this would have force-flattened the book on a
false compliance bust ($212k inherited floor breached). Under the fixed
STATIC default it must NOT bust.

Run: python3 bot/test_live_wiring_static_dd.py
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
from portfolio_layer.compliance import DrawdownType
from bot import config as bot_config

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class _FlatSnapshot:
    def __init__(self):
        w = {"BTC": 0.0}
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
            as_of_date=as_of_date, signal_snapshot=_FlatSnapshot(), sleeve_histories=hist,
            short_weights={"BTC": 0.0}, bos_weights={"BTC": 0.0}, data_source="fake",
        )


def main():
    print("test_live_wiring_static_dd")

    # 0. Config-level check: the actual object the orchestrator will use.
    check("bot.config.HYRO_COMPLIANCE_CONFIG.drawdown_type is STATIC",
          bot_config.HYRO_COMPLIANCE_CONFIG.drawdown_type == DrawdownType.STATIC,
          detail=str(bot_config.HYRO_COMPLIANCE_CONFIG.drawdown_type))

    tmp = "/tmp/_live_wiring_static_dd.json"
    if os.path.exists(tmp):
        os.remove(tmp)
    bot = HyroTraderBot(data_feed=FakeFeed(), dry_run=True, state_path=tmp)

    # Phase 1: run a cycle, then mark equity up to a $232,000 peak via the
    # real state object exactly as the orchestrator would after a wallet
    # balance read, and run another cycle so update_equity() sees it.
    bot.run_cycle(as_of_date=dt.date(2026, 1, 1), now=dt.datetime(2026, 1, 1, 12))
    bot.state.equity = 232_000.0
    report1 = bot.run_cycle(as_of_date=dt.date(2026, 1, 2), now=dt.datetime(2026, 1, 2, 12))
    check("Phase 1 peak $232,000 processed without bust",
          not report1["compliance"]["busted"], detail=str(report1["compliance"]))

    # Roll to Phase 2 the same way the real compliance state would (fresh
    # floor reopen at 90% of baseline under STATIC, regardless of the peak).
    cs = bot.state.get_compliance_state(bot_config.HYRO_COMPLIANCE_CONFIG)
    cs.advance_to_phase2()
    bot.state.set_compliance_state(cs)
    check("Phase 2 floor reopened at $180,000 (not carried over from the $232,000 peak)",
          cs.dd_floor_dollars == 180_000.0, detail=f"${cs.dd_floor_dollars:,.0f}")

    # Phase 2: mark a routine 3% dip to $206,000.
    bot.state.equity = 206_000.0
    report2 = bot.run_cycle(as_of_date=dt.date(2026, 2, 1), now=dt.datetime(2026, 2, 1, 12))
    check("Phase 2 dip to $206,000 does NOT bust and does NOT force flat",
          not report2["compliance"]["busted"] and not report2["forced_flat"],
          detail=str(report2["compliance"]))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)
    print("ALL LIVE-WIRING CHECKS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end dry-run smoke test using ReplayDataFeed against real historical
data. Proves: signal_engine -> event_sleeves -> sleeve_history ->
portfolio_layer.size_portfolio -> risk_overlay -> compliance -> (execution
skipped) -> state persistence wiring is correct, with exact numbers logged
at every stage (per standing rule: state exact numeric results).

Also runs a restart-recovery test: save state mid-run, load a FRESH
HyroTraderBot from that file, and confirm continuity (cadence timestamps,
dollar targets, cycle_count all carry over correctly rather than resetting).

This is NOT a backtest of the strategy's P&L -- it does not simulate
multi-day equity evolution. It proves the WIRING is correct: that each
component's output is valid input to the next, using the current market
snapshot at each of several forced `now` timestamps to also exercise
cadence gating (per-sleeve rebalance-due logic).
"""
import sys
import os
import shutil
import datetime as dt
import logging

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot.data_feed import ReplayDataFeed
from bot.orchestrator import HyroTraderBot
from bot import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("SmokeTest")

TEST_STATE_PATH = "/home/user/workspace/bot_runtime/smoke_test_state.json"


def fmt(report, label):
    print(f"\n=== {label} ===")
    print(f"  cycle={report['cycle']}  as_of={report['as_of']}  data_source={report['data_source']}")
    print(f"  rebalanced_sleeves={report.get('rebalanced_sleeves', [])}")
    print(f"  account_multiplier={report['account_multiplier']:.6f}")
    print(f"  sleeve_multipliers={ {k: round(v,6) for k,v in report['sleeve_multipliers'].items()} }")
    print(f"  gross_notional=${report['gross_notional']:,.2f}")
    nonzero = {c: round(v, 2) for c, v in report['combined_dollar_targets'].items() if abs(v) > 1}
    print(f"  nonzero_positions ({len(nonzero)}): {nonzero}")
    print(f"  compliance: equity=${report['compliance']['equity']:,.0f}  "
          f"max_dd_floor=${report['compliance']['max_dd_floor']:,.0f}  "
          f"distance_to_max_dd_floor=${report['compliance']['distance_to_max_dd_floor']:,.0f}  "
          f"busted={report['compliance']['busted']}")
    print(f"  execution: {report['execution']}")
    if report["flags"]:
        print(f"  FLAGS ({len(report['flags'])}):")
        for f in report["flags"]:
            print(f"    - {f}")


def main():
    if os.path.exists(TEST_STATE_PATH):
        os.remove(TEST_STATE_PATH)

    base_now = dt.datetime(2026, 8, 9, 6, 0, 0)

    # --- Cycle 1: fresh bot, first-ever cycle -- every sleeve rebalances ---
    bot = HyroTraderBot(data_feed=ReplayDataFeed(), dry_run=True, state_path=TEST_STATE_PATH)
    r1 = bot.run_cycle(as_of_date=None, now=base_now)
    fmt(r1, "Cycle 1 (fresh state, t=0h)")
    assert r1["execution"].startswith("SKIPPED"), "dry-run must never place orders"
    assert set(r1["rebalanced_sleeves"]) == set(config.SLEEVE_NAMES_), \
        f"first-ever cycle must rebalance every sleeve, got {r1['rebalanced_sleeves']}"

    # --- Cycle 2: +2h -- only Short/BOS (4h cadence) should be gated OFF too (2h < 4h) ---
    r2 = bot.run_cycle(as_of_date=None, now=base_now + dt.timedelta(hours=2))
    fmt(r2, "Cycle 2 (+2h, same bot instance)")
    assert r2["rebalanced_sleeves"] == [], \
        f"+2h should be too soon for ANY sleeve (shortest cadence is 4h), got {r2['rebalanced_sleeves']}"
    assert r2["combined_dollar_targets"] == r1["combined_dollar_targets"], \
        "with nothing due for rebalance, dollar targets must carry forward UNCHANGED, not drift or reset"

    # --- Cycle 3: +5h -- Short/BOS (4h cadence) now due, Main/Flow/DELTA/RELVOL still not ---
    r3 = bot.run_cycle(as_of_date=None, now=base_now + dt.timedelta(hours=5))
    fmt(r3, "Cycle 3 (+5h, same bot instance)")
    assert set(r3["rebalanced_sleeves"]) == {"short", "bos"}, \
        f"+5h should trigger only short/bos (4h cadence), got {r3['rebalanced_sleeves']}"

    # --- Restart-recovery test: fresh HyroTraderBot instance loading the SAME state file ---
    bot2 = HyroTraderBot(data_feed=ReplayDataFeed(), dry_run=True, state_path=TEST_STATE_PATH)
    assert bot2.state.cycle_count == bot.state.cycle_count, \
        f"restart must preserve cycle_count: expected {bot.state.cycle_count}, got {bot2.state.cycle_count}"
    assert bot2.state.last_rebalance == bot.state.last_rebalance, \
        "restart must preserve per-sleeve last_rebalance timestamps exactly"
    assert bot2.state.last_dollar_targets == bot.state.last_dollar_targets, \
        "restart must preserve last computed dollar targets exactly"
    print("\n=== Restart-recovery check ===")
    print(f"  cycle_count preserved: {bot2.state.cycle_count}")
    print(f"  last_rebalance preserved: {bot2.state.last_rebalance}")
    print("  PASS -- fresh bot instance loading the same state file recovers full continuity")

    # --- Cycle 4 on the RESTARTED bot, +75h from base (past Main's 72h cadence) ---
    r4 = bot2.run_cycle(as_of_date=None, now=base_now + dt.timedelta(hours=75))
    fmt(r4, "Cycle 4 (restarted bot instance, +75h)")
    assert "main" in r4["rebalanced_sleeves"], \
        f"+75h should trigger main (72h cadence), got {r4['rebalanced_sleeves']}"

    print("\n=== ALL SMOKE TEST ASSERTIONS PASSED ===")
    print(f"State file: {TEST_STATE_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Testnet burn-in harness.

Runs the bot through a compressed sequence of cycles and asserts the invariants
that must hold before any real capital is committed. Designed so the SAME
harness runs offline (replay feed + mock exchange) and against the real Bybit
demo/testnet account, with only the --mode flag changing.

    # runs anywhere, no credentials, no network:
    python3 bot/burn_in.py --mode offline --cycles 12

    # requires Bybit demo credentials AND a non-geo-blocked host:
    BYBIT_API_KEY=... BYBIT_API_SECRET=... python3 bot/burn_in.py --mode demo --cycles 12

WHAT THE OFFLINE MODE DOES AND DOES NOT PROVE
  proves:      cycle sequencing, cadence, state persistence and restart
               recovery, kill-switch and compliance flattening, dollar->contract
               conversion, quantization against instrument filters, position
               diffing, order convergence, reduce-before-increase ordering,
               stop attachment, per-symbol failure isolation.
  DOES NOT prove: connectivity, authentication, signing, rate limits, real
               fills, slippage, partial fills, exchange-side rejections,
               funding, or liquidation behaviour.

Only a run against the real venue proves the second list. That run cannot
happen from this sandbox: api.bybit.com returns HTTP 403 "The Amazon CloudFront
distribution is configured to block access from your country" on every
endpoint, authenticated or not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import traceback

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot import clock, config
from bot.orchestrator import HyroTraderBot
from bot.data_feed import DataFeed, MarketSnapshot, ReplayDataFeed
from bot.mock_exchange import MockBybitClient, instruments_from_panel
from bot.sleeve_history import compute_sleeve_histories
from signal_engine.engine import compute_snapshot
from bot.event_sleeves import ShortSleeveTracker, BOSSleeveTracker

logger = logging.getLogger("BurnIn")


class CachedReplayFeed(DataFeed):
    """
    ReplayDataFeed recomputes the entire panel on every call (~40s/cycle), which
    makes a 12-cycle burn-in take over ten minutes of pure recomputation of
    identical numbers.

    Only the two expensive, purely date-dependent computations are cached
    (compute_snapshot / compute_sleeve_histories). The SHORT and BOS trackers
    are still stepped on every single cycle from the live tracker state, because
    they are path-dependent -- caching those would silently freeze the tracker
    evolution and make the burn-in test nothing at all.
    """

    def __init__(self):
        self._cache: dict = {}
        self.recomputes = 0
        self.calls = 0

    def get_snapshot(self, as_of_date, short_tracker_state, bos_tracker_state):
        self.calls += 1
        key = as_of_date or clock.trading_day()
        if key not in self._cache:
            self.recomputes += 1
            self._cache[key] = (compute_snapshot(key), compute_sleeve_histories(key))
        snap, hist = self._cache[key]

        # path-dependent -- always stepped, never cached
        short_weights = ShortSleeveTracker(short_tracker_state).update(snap.universe_a_coins)
        bos_weights = BOSSleeveTracker(bos_tracker_state).update(snap.universe_b_coins)

        return MarketSnapshot(
            as_of_date=key, signal_snapshot=snap, sleeve_histories=hist,
            short_weights=short_weights, bos_weights=bos_weights,
            data_source="replay(cached)",
        )


CHECKS: list = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def run_burn_in(mode: str, cycles: int, state_path: str, hours_step: int = 4,
                start_dt=None) -> bool:
    print("=" * 74)
    print(f"BURN-IN  mode={mode}  cycles={cycles}  step={hours_step}h")
    print("=" * 74)

    if os.path.exists(state_path):
        os.remove(state_path)

    if mode == "offline":
        # Instruments are built from the real panel so the mock covers the full
        # traded universe (~34 concurrent positions), not a 15-symbol subset.
        client = MockBybitClient(instruments=instruments_from_panel())
        feed = CachedReplayFeed()
        # paper_trade=False ON PURPOSE. The mock IS the sandbox; orders must
        # actually land in its book, otherwise fills never happen, the book
        # never converges, and the flatten checks are vacuous. paper_trade=True
        # is for the real venue, where sending nothing is the point.
        bot = HyroTraderBot(data_feed=feed, exchange_client=client,
                            dry_run=False, paper_trade=False, state_path=state_path)
    elif mode in ("demo", "testnet"):
        from bot.exchange_client import ExchangeMode, HardenedBybitClient
        key, sec = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
        if not key or not sec:
            print("\nABORT: BYBIT_API_KEY / BYBIT_API_SECRET not set in the environment.")
            print("This harness will not invent credentials or fall back to a mock while")
            print("claiming to have run against the venue.")
            return False
        client = HardenedBybitClient(key, sec, mode=ExchangeMode.DEMO)
        feed = ReplayDataFeed()   # no caching against the real venue
        bot = HyroTraderBot(data_feed=feed, exchange_client=client,
                            dry_run=False, paper_trade=True, state_path=state_path)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    t0 = start_dt or dt.datetime(2026, 8, 9, 0, 0, tzinfo=dt.timezone.utc)
    reports = []

    print(f"\n--- phase 1: {cycles} sequential cycles -------------------------------")
    for i in range(cycles):
        now = t0 + dt.timedelta(hours=i * hours_step)
        last_now = now
        try:
            r = bot.run_cycle(as_of_date=clock.trading_day(now), now=now)
            reports.append(r)
        except Exception:
            record(f"cycle {i} completed", False, traceback.format_exc(limit=3))
            return False
    record(f"all {cycles} cycles completed without exception", True)

    gross = [r["gross_notional"] for r in reports]
    record("at least one cycle produced a non-empty book", any(g > 0 for g in gross),
           f"max gross=${max(gross):,.0f}")
    record("no cycle produced a NaN/negative gross notional",
           all(g == g and g >= 0 for g in gross))

    # order plans
    n_orders = [r.get("execution_plan", {}).get("n_orders", 0) for r in reports]
    n_err = [r.get("execution_plan", {}).get("n_errors", 0) for r in reports]
    record("execution plans were built every cycle",
           all("execution_plan" in r for r in reports))
    record("no execution-planning errors", sum(n_err) == 0, f"total errors={sum(n_err)}")
    record("first cycle placed orders", n_orders[0] > 0, f"n_orders[0]={n_orders[0]}")

    # With fills actually applied, a cycle that follows one with the same
    # targets must place ZERO orders. If it does not, the bot is churning the
    # book every cycle and paying spread for nothing.
    record("book converges -- a repeat cycle places no orders",
           min(n_orders[1:]) == 0 if len(n_orders) > 1 else False,
           f"seq={n_orders}")
    record("at least one REBALANCE fired (turnover actually tested)",
           sum(1 for n in n_orders[1:] if n > 0) > 0,
           f"cycles with orders after the first: {[i+1 for i, n in enumerate(n_orders[1:]) if n > 0]}")
    record("order count never grows without a rebalance",
           n_orders[-1] <= n_orders[0],
           f"first={n_orders[0]} last={n_orders[-1]}")

    print(f"\n--- phase 2: restart recovery ------------------------------------")
    cyc_before = bot.state.cycle_count
    tgt_before = {k: dict(v) for k, v in bot.state.last_dollar_targets.items()}
    bot2 = HyroTraderBot(data_feed=feed, exchange_client=client,
                         dry_run=False, paper_trade=(mode != "offline"),
                         state_path=state_path)
    record("cycle_count survives restart", bot2.state.cycle_count == cyc_before,
           f"{bot2.state.cycle_count} vs {cyc_before}")
    record("dollar targets survive restart byte-identically",
           bot2.state.last_dollar_targets == tgt_before)

    print(f"\n--- phase 3: kill switch under live execution --------------------")
    # The trip MUST be inside the same trading day as the next cycle. run_cycle
    # calls risk_state.start_new_session() whenever the session date changes,
    # which correctly re-arms the kill switch -- tripping it on day N and then
    # running a cycle on day N+1 tests nothing.
    now = last_now + dt.timedelta(minutes=30)
    assert clock.trading_day(now) == clock.trading_day(last_now), "trip must be same-day"
    rs = bot2.state.get_risk_state()
    rs.on_intraday_pnl_update(-3100.0)
    bot2.state.set_risk_state(rs)
    rk = bot2.run_cycle(as_of_date=clock.trading_day(now), now=now)
    record("kill switch zeroes the target book", rk["gross_notional"] == 0.0,
           f"gross=${rk['gross_notional']:,.2f}")
    record("forced_flat is reported", rk.get("forced_flat") is True)
    flat_orders = rk.get("execution_orders", [])
    record("every order issued while flat is reduceOnly",
           all(o["reduceOnly"] for o in flat_orders) if flat_orders else True,
           f"{len(flat_orders)} order(s)")
    if mode == "offline":
        record("exchange book is actually empty after the flatten",
               len(client.get_positions()) == 0,
               f"remaining={[p['symbol'] for p in client.get_positions()]}")

    print(f"\n--- phase 4: compliance bust -------------------------------------")
    # Set the WALLET, not state.equity: run_cycle overwrites state.equity from
    # exchange_client.get_wallet_balance() before the compliance check, so
    # assigning state.equity directly is silently discarded.
    client.equity = 179_000.0
    now2 = now + dt.timedelta(hours=26)
    rb = bot2.run_cycle(as_of_date=clock.trading_day(now2), now=now2)
    record("compliance reports busted", rb["compliance"]["busted"] is True)
    record("busted account is forced flat", rb["gross_notional"] == 0.0,
           f"gross=${rb['gross_notional']:,.2f}")

    print("\n" + "=" * 74)
    npass = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"BURN-IN RESULT: {npass}/{len(CHECKS)} checks passed")
    failed = [n for n, ok, _ in CHECKS if not ok]
    if failed:
        print("FAILED CHECKS:")
        for f in failed:
            print("  -", f)
    print("=" * 74)
    if mode == "offline":
        print("\nSCOPE: offline mode. Connectivity, auth, rate limits, real fills,")
        print("partial fills, slippage and liquidation behaviour are NOT covered.")
        print("This is not a substitute for a real testnet run.")
    return not failed


def main() -> int:
    ap = argparse.ArgumentParser(description="HyroTrader bot burn-in harness")
    ap.add_argument("--mode", default="offline", choices=["offline", "demo", "testnet"])
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--hours-step", type=int, default=4)
    ap.add_argument("--state", default=os.path.join(config.STATE_DIR, "burn_in_state.json"))
    ap.add_argument("--start-date", default=None,
                    help="YYYY-MM-DD. Use a date INSIDE the panel's history so that "
                         "successive cycles see genuinely different data and rebalances "
                         "actually fire. Defaults to today, which is past the end of the "
                         "panel and therefore produces zero turnover.")
    a = ap.parse_args()
    sd = None
    if a.start_date:
        y, m, d = (int(x) for x in a.start_date.split("-"))
        sd = dt.datetime(y, m, d, tzinfo=dt.timezone.utc)
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ok = run_burn_in(a.mode, a.cycles, a.state, a.hours_step, start_dt=sd)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

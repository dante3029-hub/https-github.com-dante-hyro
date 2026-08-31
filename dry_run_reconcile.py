#!/usr/bin/env python3
"""
dry_run_reconcile.py — ITEM 1 of the pre-flight.

WHAT THIS IS FOR
----------------
The one live failure mode nothing in the test suite can catch: the bot's idea of
its positions drifting from the exchange's. Partial fills, rejected orders, tick
/ lot rounding, and manual intervention all cause it, and every subsequent
rebalance compounds the error because targets are diffed against a WRONG base.

This places NO orders. It pulls the real account state, computes what the bot
WOULD do this cycle, and prints the diff for you to eyeball before any capital
moves.

RUN IT
------
    export BYBIT_API_KEY=...      # READ-ONLY key is sufficient and preferred
    export BYBIT_API_SECRET=...
    python3 dry_run_reconcile.py

Run it once per cycle for a full session before going live. What you are looking
for is: qty mismatches, symbols the exchange holds that the bot does not know
about, and orders below the exchange minimum.

EXIT CODES
    0  clean          — bot state and exchange agree
    1  DRIFT          — reconciliation mismatch, do NOT go live
    2  setup error    — credentials/config problem
"""
from __future__ import annotations

import os
import sys
import datetime as dt

ROOT = os.path.dirname(os.path.abspath(__file__))
# ROOT first so `from bot.exchange_client import ...` resolves as a package;
# bot/ also added so `import config` (used by the bot's own modules) works.
_LEGACY = os.path.join(ROOT, "existing_botcode", "botcode")
for _p in (ROOT, os.path.join(ROOT, "bot"), _LEGACY):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    # ---------------------------------------------------------------- config
    try:
        # `import config` is ambiguous here -- existing_botcode/botcode/config.py
        # is also on sys.path and shadows the one we want. Load bot/config.py
        # explicitly by file location so the mode flag can't come from the
        # legacy module.
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            'hyro_bot_config', os.path.join(ROOT, 'bot', 'config.py'))
        cfg = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(cfg)
        from portfolio_layer import portfolio as pf
        from portfolio_layer.risk_overlay import (
            RiskState, KILL_SWITCH_DOLLARS, DAILY_LOSS_LIMIT,
        )
    except Exception as e:                                   # noqa: BLE001
        print(f"SETUP ERROR: cannot import bot modules: {e}")
        return 2

    key = os.environ.get("BYBIT_API_KEY")
    sec = os.environ.get("BYBIT_API_SECRET")
    if not key or not sec:
        print("SETUP ERROR: set BYBIT_API_KEY and BYBIT_API_SECRET "
              "(a READ-ONLY key is enough for this script)")
        return 2

    print("=" * 74)
    print("DRY-RUN RECONCILIATION — NO ORDERS WILL BE PLACED")
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()}")
    print("=" * 74)

    # ------------------------------------------------- 1. live account state
    #
    # 2026-08-12 FIX. This previously constructed the RAW BybitClient(key, sec),
    # which BYPASSES the three-mode routing in bot/exchange_client.py and would
    # inherit whatever endpoint the legacy client hardcoded -- the same client
    # whose "testnet" flag actually pointed at api-demo.bybit.com. Same class of
    # defect as the kill switch being defined but never called.
    #
    # Now routes through HardenedBybitClient, which raises on an unknown mode
    # and never silently defaults. Mode comes from BYBIT_USE_DEMO (default true).
    try:
        from bot.exchange_client import HardenedBybitClient, ExchangeMode
        mode = ExchangeMode.DEMO if cfg.BYBIT_USE_DEMO else ExchangeMode.MAINNET
        client = HardenedBybitClient(key, sec, mode=mode)
        print(f"\nexchange mode            {mode}")
        print(f"base url                 {client.base_url}")
        if mode is ExchangeMode.MAINNET:
            print("\n*** MAINNET — REAL MONEY. Set BYBIT_USE_DEMO=true for demo. ***")
        balance = client.get_wallet_balance()
        positions = client.get_positions()
    except Exception as e:                                   # noqa: BLE001
        print(f"SETUP ERROR: exchange call failed: {e}")
        return 2

    equity = float(balance)   # BybitClient already returns a float
    print(f"\naccount equity           ${equity:,.2f}")
    print(f"open positions on venue  {len(positions)}")

    live = {}
    for p in positions:
        sym = p.get("symbol")
        size = float(p.get("size", 0) or 0)
        if size == 0:
            continue
        side = 1 if str(p.get("side", "")).lower().startswith("b") else -1
        mark = float(p.get("mark_price", 0) or 0)
        live[sym] = dict(qty=size * side, notional=size * mark * side,
                         entry=float(p.get("entry_price", 0) or 0),
                         stop=float(p.get("stop_loss", 0) or 0))

    # ------------------------------------------------- 2. bot's stored state
    try:
        from orchestrator import BotState             # persisted bot view
        state = BotState.load()
        stored = getattr(state, "positions", {}) or {}
    except Exception as e:                                   # noqa: BLE001
        print(f"\nWARNING: could not load persisted BotState ({e}).")
        print("Treating the bot as holding NOTHING — every live position will")
        print("show as UNKNOWN below. That is expected on a first run.")
        stored = {}

    # ------------------------------------------------- 3. reconcile
    print("\n" + "-" * 74)
    print("RECONCILIATION  (exchange truth vs bot state)")
    print("-" * 74)
    print(f"{'symbol':<14}{'exchange qty':>15}{'bot qty':>13}{'status':>16}")

    drift = []
    for sym in sorted(set(live) | set(stored)):
        ex = live.get(sym, {}).get("qty", 0.0)
        bo = float(stored.get(sym, 0.0) or 0.0)
        if abs(ex - bo) < 1e-9:
            status = "ok"
        elif sym not in stored:
            status = "UNKNOWN TO BOT"; drift.append(sym)
        elif sym not in live:
            status = "BOT PHANTOM"; drift.append(sym)
        else:
            status = "QTY MISMATCH"; drift.append(sym)
        print(f"{sym:<14}{ex:>15.6f}{bo:>13.6f}{status:>16}")
    if not (live or stored):
        print("  (no positions on either side — clean slate)")

    # ------------------------------------------------- 4. risk state preview
    print("\n" + "-" * 74)
    print("RISK STATE THIS CYCLE")
    print("-" * 74)
    rs = RiskState.new(equity)
    rs.start_new_session(dt.date.today())
    gross = sum(abs(v["notional"]) for v in live.values())
    print(f"  gross exposure          ${gross:,.2f}  ({gross/equity if equity else 0:.2f}x)")
    print(f"  exposure multiplier     {rs.target_exposure_multiplier():.3f}")
    print(f"  kill switch threshold  -${KILL_SWITCH_DOLLARS:,.0f}")
    print(f"  firm daily limit       -${DAILY_LOSS_LIMIT:,.0f}")
    print(f"  live stop              {cfg.LIVE_STOP_LOSS_FRAC:.0%}")
    print(f"  max leg notional        ${pf.MAX_NOTIONAL_PER_LEG:,.0f}")
    print(f"  aggregate loss cap      ${pf.AGGREGATE_MAX_LOSS:,.0f}")

    # ------------------------------------------------- 5. per-leg rule check
    print("\n" + "-" * 74)
    print("PER-LEG COMPLIANCE ON CURRENT POSITIONS  (3% rule = $6,000)")
    print("-" * 74)
    breaches = []
    for sym, v in sorted(live.items()):
        n = abs(v["notional"])
        stop_loss = n * cfg.LIVE_STOP_LOSS_FRAC
        gap_loss = n * pf.ASSUMED_ADVERSE_GAP
        flag = ""
        if stop_loss > pf.MAX_LOSS_PER_TRADE:
            flag = "STOP-OUT BREACH"; breaches.append(sym)
        elif gap_loss > pf.MAX_LOSS_PER_TRADE:
            flag = "gap-risk over limit"
        has_stop = v.get("stop", 0) > 0
        if not has_stop:
            flag = (flag + " | NO STOP ON VENUE").strip(" |")
            breaches.append(sym)
        print(f"  {sym:<12} notional ${n:>10,.0f}   stop-out ${stop_loss:>8,.0f}"
              f"   gap ${gap_loss:>8,.0f}   venue_stop={'yes' if has_stop else 'NO'}  {flag}")
    if not live:
        print("  (no open positions)")

    # ------------------------------------------------- 6. verdict
    print("\n" + "=" * 74)
    if drift:
        print(f"RESULT: DRIFT DETECTED on {len(drift)} symbol(s): {', '.join(drift)}")
        print("Do NOT go live until reconciliation is clean. Every rebalance")
        print("diffs targets against this state; a wrong base compounds.")
        print("=" * 74)
        return 1
    if breaches:
        print(f"RESULT: PER-TRADE RULE BREACH on {', '.join(breaches)}")
        print("=" * 74)
        return 1
    print("RESULT: CLEAN — bot state and exchange agree, no per-leg breaches.")
    print("Run this once per cycle for a full session before funding.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

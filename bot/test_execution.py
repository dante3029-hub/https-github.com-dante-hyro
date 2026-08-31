#!/usr/bin/env python3
"""
Tests for the live execution path (bot/execution.py).

This is the code that turns numbers into orders. Every test below maps to a
specific way this layer can lose money.

Run: python3 bot/test_execution.py
"""
import os
import sys

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot.execution import build_execution_plan, execute_plan, MIN_TRADE_USD
from bot.mock_exchange import MockBybitClient

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_dollars_are_converted_to_contracts():
    """The old stub passed DOLLARS through as `qty`. At BTC $61,000 that would
    have sent 9000 CONTRACTS instead of ~0.147 -- a 61,000x over-order."""
    print("\ntest_dollars_are_converted_to_contracts")
    c = MockBybitClient()
    plan = build_execution_plan(c, {"BTC": 9000.0})
    check("exactly one order planned", len(plan.orders) == 1, str(plan.orders))
    o = plan.orders[0]
    expected = 9000.0 / 61000.0
    check("qty is in CONTRACTS not dollars", abs(o.qty_contracts - expected) < 0.001,
          f"qty={o.qty_contracts:.6f} expected~{expected:.6f}")
    check("qty is nowhere near the dollar figure", o.qty_contracts < 1.0,
          f"qty={o.qty_contracts}")
    check("side is Buy for a positive target", o.side == "Buy")


def test_quantization_respects_instrument_filters():
    print("\ntest_quantization_respects_instrument_filters")
    c = MockBybitClient()
    # XRP has qtyStep 1.0 -> must be a whole number of contracts
    plan = build_execution_plan(c, {"XRP": 5000.0})
    o = [x for x in plan.orders if x.symbol == "XRPUSDT"][0]
    check("XRP qty is a whole number (step=1.0)", float(o.qty_str) == int(float(o.qty_str)),
          f"qty_str={o.qty_str}")
    # a target too small to meet minOrderQty must be SKIPPED, not sent as zero
    c2 = MockBybitClient(instruments={"BTCUSDT": (61_000.0, 0.001, 0.1)})
    plan2 = build_execution_plan(c2, {"BTC": 500.0})   # 500/61000 = 0.0082 < min 0.1
    check("sub-minimum order is skipped, not sent", len(plan2.orders) == 0 and len(plan2.skipped) == 1,
          f"orders={len(plan2.orders)} skipped={plan2.skipped}")


def test_diffs_against_existing_position():
    """Must trade only the DELTA. Trading the full target on top of an existing
    position doubles the book."""
    print("\ntest_diffs_against_existing_position")
    c = MockBybitClient()
    c.seed_position("SOLUSDT", "Buy", 60.0)          # 60 * 145 = $8,700 long
    plan = build_execution_plan(c, {"SOL": 10_000.0})
    o = [x for x in plan.orders if x.symbol == "SOLUSDT"][0]
    check("trades only the $1,300 delta", abs(o.usd_delta - 1300.0) < 1.0,
          f"usd_delta={o.usd_delta:+,.2f}")
    check("delta order is a Buy", o.side == "Buy")
    check("not flagged reduceOnly when increasing", o.reduce_only is False)


def test_no_trade_band_suppresses_churn():
    print("\ntest_no_trade_band_suppresses_churn")
    c = MockBybitClient()
    c.seed_position("SOLUSDT", "Buy", 69.0)           # 69*145 = $10,005
    plan = build_execution_plan(c, {"SOL": 10_000.0})  # $5 delta
    check("$5 delta on a $10k position is suppressed", len(plan.orders) == 0,
          f"orders={plan.orders}")
    check("suppression is recorded, not silent", len(plan.skipped) == 1, str(plan.skipped))


def test_closing_ignores_the_band():
    """A close must never be suppressed by the no-trade band -- otherwise tiny
    residual positions accumulate forever."""
    print("\ntest_closing_ignores_the_band")
    c = MockBybitClient()
    c.seed_position("XRPUSDT", "Buy", 20.0)           # 20 * 0.52 = $10.40, under MIN_TRADE_USD
    plan = build_execution_plan(c, {})
    check(f"${MIN_TRADE_USD} band does not block a close",
          len(plan.orders) == 1 and plan.orders[0].reason == "close",
          f"orders={plan.orders} skipped={plan.skipped}")
    check("close is reduceOnly", plan.orders[0].reduce_only is True)
    check("close is the opposite side", plan.orders[0].side == "Sell")


def test_side_flip_is_not_reduce_only():
    """Flipping long->short crosses zero. Marking it reduceOnly would have the
    exchange reject or truncate it, silently leaving a long on the book."""
    print("\ntest_side_flip_is_not_reduce_only")
    c = MockBybitClient()
    c.seed_position("SOLUSDT", "Buy", 60.0)           # +$8,700
    plan = build_execution_plan(c, {"SOL": -8_700.0})  # target -$8,700
    o = [x for x in plan.orders if x.symbol == "SOLUSDT"][0]
    check("flip is NOT reduceOnly", o.reduce_only is False)
    check("flip trades the full 2x distance", abs(o.usd_delta + 17_400.0) < 1.0,
          f"usd_delta={o.usd_delta:+,.2f}")
    check("flip is a Sell", o.side == "Sell")


def test_reducers_are_ordered_before_increasers():
    """Peak margin during the rebalance must not exceed either endpoint."""
    print("\ntest_reducers_are_ordered_before_increasers")
    c = MockBybitClient()
    c.seed_position("SOLUSDT", "Buy", 200.0)          # $29,000 -> will reduce
    plan = build_execution_plan(c, {"SOL": 5_000.0, "ETH": 20_000.0, "LINK": 15_000.0})
    reduce_idx = [i for i, o in enumerate(plan.orders) if o.reduce_only]
    incr_idx = [i for i, o in enumerate(plan.orders) if not o.reduce_only]
    check("at least one reducer and one increaser", bool(reduce_idx) and bool(incr_idx),
          f"reduce={reduce_idx} incr={incr_idx}")
    check("every reducer precedes every increaser", max(reduce_idx) < min(incr_idx),
          f"order={[(o.symbol, o.reason) for o in plan.orders]}")


def test_force_flat_closes_everything():
    print("\ntest_force_flat_closes_everything")
    c = MockBybitClient()
    for sym, sz in [("SOLUSDT", 60.0), ("ETHUSDT", 3.0), ("XRPUSDT", 5000.0)]:
        c.seed_position(sym, "Buy", sz)
    plan = build_execution_plan(c, {"SOL": 10_000.0, "ETH": 9_000.0}, force_flat=True)
    check("all 3 live positions get a closing order", len(plan.orders) == 3,
          f"{[(o.symbol, o.reason) for o in plan.orders]}")
    check("every order is reduceOnly", all(o.reduce_only for o in plan.orders))
    check("no order opens new exposure", all(o.reason == "flatten" for o in plan.orders))
    res = execute_plan(c, plan, dry_run=False)
    check("after execution the book is empty", len(c.get_positions()) == 0,
          f"remaining={c.get_positions()}")
    check("no failures", len(res["failed"]) == 0, str(res["failed"]))


def test_one_bad_symbol_does_not_abort_the_cycle():
    print("\ntest_one_bad_symbol_does_not_abort_the_cycle")
    c = MockBybitClient(failing_symbols={"LINKUSDT"})
    plan = build_execution_plan(c, {"SOL": 10_000.0, "LINK": 8_000.0, "ETH": 12_000.0})
    res = execute_plan(c, plan, dry_run=False)
    check("the good orders still went through", len(res["sent"]) == 2,
          f"sent={[s['symbol'] for s in res['sent']]}")
    check("the bad one is reported, not swallowed", len(res["failed"]) == 1
          and res["failed"][0]["symbol"] == "LINKUSDT", str(res["failed"]))


def test_unknown_symbol_is_an_error_not_a_silent_skip():
    print("\ntest_unknown_symbol_is_an_error_not_a_silent_skip")
    c = MockBybitClient()
    plan = build_execution_plan(c, {"NOTACOIN": 5_000.0})
    check("unknown symbol recorded as an error", len(plan.errors) == 1,
          f"errors={plan.errors}")
    check("no order planned for it", len(plan.orders) == 0)


def test_position_read_failure_plans_nothing():
    """If we cannot see the current book we must not guess. Planning orders
    against an assumed-empty book would double the position."""
    print("\ntest_position_read_failure_plans_nothing")

    class Blind(MockBybitClient):
        def get_positions(self, symbol=None):
            raise ConnectionError("exchange unreachable")

    plan = build_execution_plan(Blind(), {"SOL": 10_000.0, "ETH": 9_000.0})
    check("zero orders planned when positions unreadable", len(plan.orders) == 0)
    check("failure is recorded", len(plan.errors) == 1, str(plan.errors))


def test_paper_mode_sends_nothing():
    print("\ntest_paper_mode_sends_nothing")
    c = MockBybitClient()
    plan = build_execution_plan(c, {"SOL": 10_000.0, "ETH": 9_000.0})
    res = execute_plan(c, plan, dry_run=True)
    check("plan is non-empty", len(plan.orders) == 2)
    check("nothing was actually sent", len(c.order_log) == 0, f"order_log={c.order_log}")
    check("every entry marked SENT=False", all(s["SENT"] is False for s in res["sent"]))
    check("exchange book unchanged", len(c.get_positions()) == 0)


def test_stops_attached_to_new_positions_only():
    print("\ntest_stops_attached_to_new_positions_only")
    c = MockBybitClient()
    c.seed_position("SOLUSDT", "Buy", 200.0)          # will be reduced
    plan = build_execution_plan(c, {"SOL": 5_000.0, "ETH": 20_000.0})
    res = execute_plan(c, plan, dry_run=False, stop_loss_frac=0.60)
    stopped = {s["symbol"] for s in res["stops_set"]}
    check("stop attached to the newly opened ETH", "ETHUSDT" in stopped, str(stopped))
    check("no stop re-sent for the pure reduction", "SOLUSDT" not in stopped, str(stopped))
    eth = [s for s in res["stops_set"] if s["symbol"] == "ETHUSDT"][0]
    check("long stop is BELOW the mark", eth["sl"] < 3400.0, f"sl={eth['sl']} mark=3400")
    check("stop is 60% adverse", abs(eth["sl"] - 3400.0 * 0.40) < 1e-6, f"sl={eth['sl']}")


def test_short_stop_is_above_mark():
    print("\ntest_short_stop_is_above_mark")
    c = MockBybitClient()
    plan = build_execution_plan(c, {"ETH": -20_000.0})
    res = execute_plan(c, plan, dry_run=False, stop_loss_frac=0.60)
    eth = [s for s in res["stops_set"] if s["symbol"] == "ETHUSDT"][0]
    check("short stop is ABOVE the mark", eth["sl"] > 3400.0, f"sl={eth['sl']} mark=3400")
    check("short stop is 60% adverse", abs(eth["sl"] - 3400.0 * 1.60) < 1e-6, f"sl={eth['sl']}")



def test_close_leaves_no_dust():
    """REGRESSION: burn_in.py phase 3, 2026-08-09.

    A close order was sized by round-tripping contracts -> USD -> contracts and
    then FLOORING to the instrument step, which lost up to one full step on
    every leg. A kill-switch flatten reported gross $0.00 while leaving
    residual exposure on all 34 positions.

    The sizes below are chosen to be exact multiples of the 0.1 step that do
    NOT survive the float round-trip -- the original test suite passed only
    because its sizes happened to round-trip cleanly.
    """
    print("\ntest_close_leaves_no_dust")
    cases = [("ICPUSDT", 2.131, 1422.1), ("AVAXUSDT", 6.247, 14.3),
             ("SOLUSDT", 73.48, 0.6), ("UNIUSDT", 3.84, 11.6)]
    c = MockBybitClient(instruments={s: (m, 0.1, 0.1) for s, m, _ in cases})
    for sym, _, size in cases:
        c._positions[sym] = {"side": "Sell" if size > 100 else "Buy", "size": size}

    plan = build_execution_plan(c, {}, force_flat=True)
    check("a close order is planned for every position",
          len(plan.orders) == len(cases), f"{len(plan.orders)} order(s)")
    for o in plan.orders:
        want = dict((s, sz) for s, _, sz in cases)[o.symbol]
        check(f"{o.symbol} closes the FULL size, no dust",
              abs(o.qty_contracts - want) < 1e-9,
              f"order={o.qty_str} position={want}")

    execute_plan(c, plan, dry_run=False)
    check("book is completely empty after the flatten",
          len(c.get_positions()) == 0,
          f"remaining={[(p['symbol'], p['size']) for p in c.get_positions()]}")


def test_round_trip_converges():
    """Running the same targets twice must produce no orders the second time.
    A plan that never converges churns fees every single cycle."""
    print("\ntest_round_trip_converges")
    c = MockBybitClient()
    targets = {"SOL": 10_000.0, "ETH": -9_000.0, "LINK": 6_000.0, "XRP": 4_000.0}
    p1 = build_execution_plan(c, targets)
    execute_plan(c, p1, dry_run=False)
    p2 = build_execution_plan(c, targets)
    check("first pass places orders", len(p1.orders) == 4, f"{len(p1.orders)}")
    check("second pass places ZERO orders (converged)", len(p2.orders) == 0,
          f"residual={[(o.symbol, round(o.usd_delta,2)) for o in p2.orders]}")


if __name__ == "__main__":
    for fn in [test_dollars_are_converted_to_contracts,
               test_quantization_respects_instrument_filters,
               test_diffs_against_existing_position,
               test_no_trade_band_suppresses_churn,
               test_closing_ignores_the_band,
               test_side_flip_is_not_reduce_only,
               test_reducers_are_ordered_before_increasers,
               test_force_flat_closes_everything,
               test_one_bad_symbol_does_not_abort_the_cycle,
               test_unknown_symbol_is_an_error_not_a_silent_skip,
               test_position_read_failure_plans_nothing,
               test_paper_mode_sends_nothing,
               test_stops_attached_to_new_positions_only,
               test_short_stop_is_above_mark,
               test_close_leaves_no_dust, test_round_trip_converges]:
        fn()

    print(f"\n{'='*64}")
    print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("ALL EXECUTION TESTS PASSED")

#!/usr/bin/env python3
"""
Regression tests for the protective-stop sync bug (found 2026-08-09).

THE BUG: ShortSleeveTracker computed a chandelier trailing stop on every 4h
bar and discarded it. BOSSleeveTracker held a 2-ATR stop and 3-ATR TP that
existed only in memory. _attach_stops() only fires for symbols present in an
execution plan and only sets the static 60% stop. So between rebalances a live
event-sleeve position ran on its loose entry stop while the bot's internal
state believed it was protected at a much tighter level. The backtest exits on
the trail; the live bot would not have.

These tests assert the levels actually reach the venue.
"""
import os, sys
WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from bot import config
from bot.execution import sync_protective_stops
from bot.event_sleeves import short_protective_stops, bos_protective_levels
from bot.mock_exchange import MockBybitClient

PASS = FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def mk_client(symbols):
    """Client holding an open SHORT in each symbol (event sleeves are short-only)."""
    c = MockBybitClient(equity=200_000.0)
    for sym in symbols:
        c.seed_position(sym, "Sell", 10.0)
    return c


print("=" * 64)
print("PROTECTIVE STOP SYNC -- regression tests")
print("=" * 64)

# ---------------------------------------------------------------- accessors
print("\n[1] state accessors")
short_state = {"slots": {
    "ICP": {"entry_price": 10.0, "lowest_close_high": 9.0, "chand_stop": 9.8},
    "ADA": {"entry_price": 1.0, "lowest_close_high": 0.9},   # no bar yet
}}
sl = short_protective_stops(short_state)
check("short: slot with a trail is returned", sl.get("ICP") == 9.8, sl)
check("short: slot with no bar yet is omitted", "ADA" not in sl, sl)
check("short: empty state is safe", short_protective_stops({}) == {}, "")
check("short: None state is safe", short_protective_stops(None) == {}, "")

bos_state = {"slots": {"SOL": {"stop": 155.0, "tp": 130.0},
                       "AVAX": {"stop": 22.0, "tp": 18.0}}}
bs, bt = bos_protective_levels(bos_state)
check("bos: stops returned", bs == {"SOL": 155.0, "AVAX": 22.0}, bs)
check("bos: tps returned", bt == {"SOL": 130.0, "AVAX": 18.0}, bt)
check("bos: None state is safe", bos_protective_levels(None) == ({}, {}), "")

# ---------------------------------------------------- the stop actually sent
print("\n[2] stops reach the exchange (the actual bug)")
c = mk_client({"ICPUSDT": 10.0, "SOLUSDT": 150.0})
res = sync_protective_stops(c, {"ICPUSDT": 9.8, "SOLUSDT": 155.0}, dry_run=False)
sent = {d["symbol"]: d["sl"] for d in c.sl_log}
check("modify_sl called for every symbol", set(sent) == {"ICPUSDT", "SOLUSDT"}, sent)
check("ICP stop level exact", sent.get("ICPUSDT") == 9.8, sent)
check("SOL stop level exact", sent.get("SOLUSDT") == 155.0, sent)
check("all reported synced", len(res["stops_synced"]) == 2 and not res["stops_failed"], res)
check("SENT flag true when live", all(d["SENT"] for d in res["stops_synced"]), res)

# ------------------------------------------------------------ trail tightens
print("\n[3] a tightening trail is re-sent each cycle")
c = mk_client({"ICPUSDT": 10.0})
for level in (9.9, 9.7, 9.55):
    sync_protective_stops(c, {"ICPUSDT": level}, dry_run=False)
levels = [d["sl"] for d in c.sl_log if d["symbol"] == "ICPUSDT"]
check("three amends recorded", len(levels) == 3, levels)
check("levels ratchet down", levels == [9.9, 9.7, 9.55], levels)

# --------------------------------------------------------------------- TP
print("\n[4] BOS take-profit is pushed")
c = mk_client({"SOLUSDT": 150.0})
res = sync_protective_stops(c, {"SOLUSDT": 155.0}, dry_run=False,
                            tps_by_symbol={"SOLUSDT": 130.0})
check("tp recorded at venue", [d["tp"] for d in c.tp_log] == [130.0], c.tp_log)
check("tp reported synced", len(res["tps_synced"]) == 1, res)

# --------------------------------------------------------------- dry run
print("\n[5] dry-run / paper sends nothing")
c = mk_client({"ICPUSDT": 10.0})
res = sync_protective_stops(c, {"ICPUSDT": 9.8}, dry_run=True,
                            tps_by_symbol={"ICPUSDT": 8.0})
check("no SL hit the venue", c.sl_log == [], c.sl_log)
check("no TP hit the venue", c.tp_log == [], c.tp_log)
check("still reported for the audit trail", len(res["stops_synced"]) == 1, res)
check("SENT flag false in dry run", res["stops_synced"][0]["SENT"] is False, res)

# ------------------------------------------------------------- failures
print("\n[6] partial failure does not block the rest")
c = mk_client({"ICPUSDT": 10.0})          # SOLUSDT has NO position -> modify_sl False
res = sync_protective_stops(c, {"ICPUSDT": 9.8, "SOLUSDT": 155.0}, dry_run=False)
check("good symbol still tightened", [d["sl"] for d in c.sl_log] == [9.8], c.sl_log)
check("bad symbol reported failed", [d["symbol"] for d in res["stops_failed"]] == ["SOLUSDT"], res)
check("no exception raised", True)


class Raiser(MockBybitClient):
    def modify_sl(self, symbol, sl_price):
        raise RuntimeError("venue 10001: position idx not match")


c = Raiser(equity=200_000.0)
c.seed_position("ICPUSDT", "Sell", 10.0)
res = sync_protective_stops(c, {"ICPUSDT": 9.8}, dry_run=False)
check("exception captured, not propagated", len(res["stops_failed"]) == 1, res)
check("error text preserved", "10001" in res["stops_failed"][0]["error"], res)

# ------------------------------------------------------------ sanity guards
print("\n[7] guards")
c = mk_client({"ICPUSDT": 10.0})
sync_protective_stops(c, {"ICPUSDT": 0.0, "SOLUSDT": -5.0}, dry_run=False)
check("zero/negative stops never sent", c.sl_log == [], c.sl_log)
check("empty dict is a no-op", sync_protective_stops(c, {}, dry_run=False)["stops_synced"] == [], "")
check("symbol mapping is coin+USDT", config.to_bybit_symbol("ICP") == "ICPUSDT", "")

# --------------------------------------------- end-to-end through the state
print("\n[8] end-to-end: tracker state -> venue")
c = mk_client({"ICPUSDT": 10.0, "SOLUSDT": 150.0})
s_sl = short_protective_stops({"slots": {"ICP": {"chand_stop": 9.8}}})
b_sl, b_tp = bos_protective_levels({"slots": {"SOL": {"stop": 155.0, "tp": 130.0}}})
stops = {config.to_bybit_symbol(k): v for k, v in list(s_sl.items()) + list(b_sl.items())}
tps = {config.to_bybit_symbol(k): v for k, v in b_tp.items()}
sync_protective_stops(c, stops, dry_run=False, tps_by_symbol=tps)
check("short trail reached venue", {"symbol": "ICPUSDT", "sl": 9.8} in c.sl_log, c.sl_log)
check("bos stop reached venue", {"symbol": "SOLUSDT", "sl": 155.0} in c.sl_log, c.sl_log)
check("bos tp reached venue", {"symbol": "SOLUSDT", "tp": 130.0} in c.tp_log, c.tp_log)




# ==========================================================================
# [9] NEGATIVE-CONTROL-BACKED TESTS
#
# The first eight groups above were VACUOUS with respect to the original bug:
# deleting `slot["chand_stop"] = chand_stop` from ShortSleeveTracker left all
# 31 of them green, because they fed hand-built state dicts into the accessors
# and never ran the real tracker. Caught by reverting the fix and re-running.
#
# These drive the ACTUAL tracker update loop over synthetic 4h bars and assert
# the trail is persisted, and assert the orchestrator actually CALLS the sync
# (a fix that is never invoked is inert code, which is the other half of how
# this bug survived).
# ==========================================================================
import types
import pandas as pd
import bot.event_sleeves as ev

print("\n[9] real tracker persists the trail (negative-control backed)")


def fake_bars(highs, atr=1.0, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(highs), freq="4h", tz="UTC")
    return pd.DataFrame({"high": highs, "atr": [atr] * len(highs)}, index=idx)


_orig_builder = ev._short_build_signals
try:
    # entry high 100, then highs fall -> chandelier trail must ratchet DOWN
    ev._short_build_signals = lambda coin: fake_bars([98.0, 96.0, 94.0])
    st = {"slots": {"ICP": {"entry_price": 100.0,
                            "entry_ts": "2025-12-31 00:00:00+00:00",
                            "lowest_close_high": 100.0}}}
    tracker = ev.ShortSleeveTracker(st)
    tracker.update(["ICP"])

    slot = st["slots"].get("ICP")
    check("tracker still holds the slot", slot is not None, st)
    check("tracker PERSISTED chand_stop (the bug)",
          slot is not None and "chand_stop" in slot, slot)
    expected = 94.0 + ev.CHAND_MULT * 1.0
    check(f"trail equals lowest_high + {ev.CHAND_MULT}*ATR = {expected}",
          slot is not None and abs(slot.get("chand_stop", -1) - expected) < 1e-9, slot)
    got = ev.short_protective_stops(st)
    check("accessor surfaces the REAL tracker's trail",
          abs(got.get("ICP", -1) - expected) < 1e-9, got)

    # BUG #8: an idle cycle with ZERO new bars must not exit the position.
    # Pre-fix this force-exited the slot, because the walk restarted at
    # entry_ts and re-tested old bars against an already-ratcheted trail.
    tracker.update(["ICP"])
    check("BUG #8: idle cycle does NOT exit the slot",
          "ICP" in st["slots"], st["slots"])

    # trail must tighten as highs fall further
    ev._short_build_signals = lambda coin: fake_bars([98.0, 96.0, 94.0, 90.0])
    tracker.update(["ICP"])
    tighter = st["slots"]["ICP"]["chand_stop"]
    check("trail TIGHTENS on new lower highs", tighter < expected,
          f"{tighter} vs {expected}")
    check("tightened trail is exactly 90 + 2*ATR", abs(tighter - 92.0) < 1e-9, tighter)

    # and a genuine breach must still exit
    ev._short_build_signals = lambda coin: fake_bars([98.0, 96.0, 94.0, 90.0, 93.0])
    tracker.update(["ICP"])
    check("genuine breach of the trail still exits",
          "ICP" not in st["slots"], st["slots"])
finally:
    ev._short_build_signals = _orig_builder

print("\n[10] orchestrator actually CALLS the sync each cycle")
src = open(os.path.join(WORKSPACE, "bot", "orchestrator.py")).read()
check("orchestrator references sync_protective_stops",
      "sync_protective_stops" in src, "")
check("sync sits OUTSIDE the dry-run/else execution branch",
      src.index("sync_protective_stops") > src.index("# 8b."), "")
check("sync runs before state is persisted",
      src.index("sync_protective_stops") < src.index("# 9. persist"), "")
check("kill-switch path is guarded", "if killed:" in src.split("# 8b.")[1][:2000], "")

print("\n" + "=" * 64)
print(f"PASSED {PASS} / {PASS + FAIL}")
if FAILURES:
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL PROTECTIVE STOP TESTS PASSED")

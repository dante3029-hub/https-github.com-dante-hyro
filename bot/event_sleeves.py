"""
Event-driven slot tracker for the Short and BOS sleeves.

signal_engine.sleeve_short.latest_signal() / sleeve_bos.latest_signal() only
answer "is there a fresh trigger right now" -- both modules' own docstrings
explicitly say max-concurrent-slot management (6 slots, 1/6 sizing per slot),
trailing-stop updates, and "already short this coin" suppression are Phase
3/4 responsibilities, not yet built anywhere. This module is that missing
piece.

Design (both sleeves: MAX_CONCURRENT=6 slots, 1/6 gross weight per slot,
short-only, so a fully-deployed sleeve's abs-weight sums to 1.0, matching the
"~1.0 gross when fully deployed" contract of
portfolio_layer.portfolio.size_portfolio()'s sleeve_raw_weights input):

  - State is a plain dict (JSON-serializable), owned by bot/state.py, passed
    in here and mutated in place -- restart recovery is just "reload the
    JSON, keep calling update()".
  - update(coins) does two things every cycle:
      1. Walk every OPEN slot forward on the underlying causal bar data
         (not just the latest_signal() snapshot, which only describes the
         newest bar) and close it if its exit condition has fired at any
         bar since entry.
      2. For coins with no open slot and count(open) < MAX_CONCURRENT, open
         a new slot if signal_engine's latest_signal() says triggered.
  - Returns {coin: weight}, weight = -1/MAX_CONCURRENT per open slot
    (negative -- both sleeves are short-only), directly usable as
    sleeve_raw_weights["short"] / ["bos"] for size_portfolio().

CAVEATS, carried forward not smoothed over:
  - Short sleeve's exit replay reuses sleeve_S_reconstructed.build_signals(),
    which is itself flagged "RECONSTRUCTION, NOT VERIFICATION" (prose-spec
    rebuild, several parameters ASSUMED, not validated against the spec's
    claimed 0.78 solo Sharpe). This tracker inherits that caveat unchanged.
  - BOS's exit replay assumes the underlying HIST CSV rows are append-only
    (a fixed START date, new rows only ever added at the end) so a stored
    array index for the entry bar stays valid across cycles. True for the
    static local CSVs this bot currently reads (see AUDIT_FINDINGS.md /
    BUILD_PLAN.md Phase 4 live-data-gap finding) -- would need to switch to
    timestamp-based lookup if/when a live, possibly-reordering feed replaces
    them.
  - BOS's frozen last-confirmed-swing-high (used for the "close above
    swing high" exit) is captured ONCE at entry and never updated while the
    slot is open, matching reference_impl.run_bos()'s own behavior of
    skipping pivot updates during an open simulated trade (`i = j+1`) -- an
    intentional match, not an oversight.
"""
import os
import sys
import numpy as np

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from sleeve_S_reconstructed import build_signals as _short_build_signals, CHAND_MULT, MAX_CONCURRENT as _SHORT_MAX
from reference_impl import _atr, _pivots, _load_4h, BOS_PIVOT_K, BOS_ATR_STOP, BOS_TP_ATR, BOS_EXIT_BAR, BOS_MAXPOS

import signal_engine.sleeve_short as sig_short_mod
import signal_engine.sleeve_bos as sig_bos_mod


def new_tracker_state() -> dict:
    return {"slots": {}}


class ShortSleeveTracker:
    """Bear-regime market-structure short, 4h bars, chandelier trail exit.
    See module docstring for the RECONSTRUCTION-NOT-VERIFICATION caveat."""

    MAX_CONCURRENT = _SHORT_MAX  # 6, per sleeve_S_reconstructed.py

    def __init__(self, state: dict):
        self.state = state
        self.state.setdefault("slots", {})

    def update(self, coins: list) -> dict:
        slots = self.state["slots"]

        # 1) walk open slots forward, check chandelier exit on every bar since entry
        for coin in list(slots.keys()):
            df = _short_build_signals(coin)
            if df is None:
                continue
            slot = slots[coin]
            # BUG FIX 2026-08-09 (#8): resume from the last bar already
            # evaluated, NOT from entry.
            #
            # The walk used to restart at entry_ts every cycle while
            # lowest_close_high persisted in state. So on cycle 2 the bars from
            # cycle 1 were re-tested against a trail that had already ratcheted
            # DOWN, and a high the position had legitimately survived now
            # breached it. Reproducible with ZERO new market data: a slot open
            # after cycle 1 was force-exited on cycle 2. Live, this would have
            # flushed the entire short sleeve every 4h and made the chandelier
            # trail meaningless. last_checked_ts was already being written here
            # and never read -- the same defect class as chand_stop.
            resume_from = slot.get("last_checked_ts") or slot["entry_ts"]
            bars = df[df.index > resume_from]
            exited = False
            for ts, row in bars.iterrows():
                slot["lowest_close_high"] = min(slot["lowest_close_high"], float(row["high"]))
                chand_stop = slot["lowest_close_high"] + CHAND_MULT * float(row["atr"])
                # BUG FIX 2026-08-09: the trail was computed here and then
                # thrown away. Nothing ever pushed it to the exchange, so a
                # live short ran on its loose entry stop while the bot's
                # internal model believed the stop was tightening every 4h
                # bar. Persist it so sync_protective_stops() can send it.
                slot["chand_stop"] = chand_stop
                if float(row["high"]) >= chand_stop:
                    del slots[coin]
                    exited = True
                    break
            if not exited and len(bars):
                slot["last_checked_ts"] = str(bars.index[-1])

        # 2) open new slots if room + triggered, in coin-list order (no
        #    priority ranking specified anywhere upstream -- first-come)
        for coin in coins:
            if len(slots) >= self.MAX_CONCURRENT:
                break
            if coin in slots:
                continue
            sig = sig_short_mod.latest_signal(coin)
            if sig and sig["triggered"]:
                high_at_entry = sig["initial_stop"] - CHAND_MULT * sig["atr"]
                slots[coin] = dict(
                    entry_price=sig["entry_ref"],
                    entry_ts=str(sig["timestamp"]),
                    lowest_close_high=high_at_entry,
                )

        return self._weights(coins)

    def protective_stops(self) -> dict:
        """{coin: stop_price} for every open short slot, at the CURRENT trail.

        Short slots are short-only, so the stop always sits ABOVE the entry and
        ratchets DOWN as the trail tightens. Consumed by
        execution.sync_protective_stops().
        """
        out = {}
        for coin, slot in self.state.get("slots", {}).items():
            sl = slot.get("chand_stop")
            if sl is not None and sl > 0:
                out[coin] = float(sl)
        return out

    def _weights(self, coins):
        w = {c: 0.0 for c in coins}
        for coin in self.state["slots"]:
            if coin in w:
                w[coin] = -1.0 / self.MAX_CONCURRENT
        return w


class BOSSleeveTracker:
    """Market-structure short continuation, 4h bars, 3-ATR TP / 2-ATR stop /
    close-above-frozen-swing-high / 30-bar max hold. This IS the validated
    sleeve from reference_impl.py (not a reconstruction) -- only the
    slot-management wrapper here is new."""

    MAX_CONCURRENT = BOS_MAXPOS  # 6, per reference_impl.py

    def __init__(self, state: dict):
        self.state = state
        self.state.setdefault("slots", {})

    def update(self, coins: list) -> dict:
        slots = self.state["slots"]

        for coin in list(slots.keys()):
            self._check_exit(coin, slots[coin])
            if slots[coin].get("_closed"):
                del slots[coin]

        for coin in coins:
            if len(slots) >= self.MAX_CONCURRENT:
                break
            if coin in slots:
                continue
            sig = sig_bos_mod.latest_signal(coin)
            if sig and sig["triggered"]:
                d4 = _load_4h(coin)
                if d4 is None:
                    continue
                ts, o, hh, ll, cl, vv, tb = d4
                entry_i = sig["i"] + 1  # tradeable from open of i+1, per sleeve_bos.py
                if entry_i >= len(ts):
                    continue
                # frozen last-confirmed swing high as of the trigger bar --
                # recomputed here since sleeve_bos.latest_signal() doesn't
                # expose it directly. Must match its own internal walk.
                ph, pl = _pivots(hh, ll, BOS_PIVOT_K)
                last_h = np.nan
                for k in range(BOS_PIVOT_K + 20, sig["i"] + 1):
                    if not np.isnan(ph[k]):
                        last_h = ph[k]
                slots[coin] = dict(
                    entry_i=int(entry_i),
                    entry_ts=int(ts[entry_i]),
                    entry_price=sig["entry_ref"],
                    stop=sig["stop"],
                    tp=sig["tp"],
                    frozen_last_high=float(last_h) if not np.isnan(last_h) else None,
                    bars_checked=0,
                )

        return self._weights(coins)

    def protective_stops(self) -> dict:
        """{coin: stop_price} for every open BOS slot.

        BOS stops are FIXED at entry (2.0 x ATR above, short-only) and do not
        trail, but they still have to be pushed to the exchange -- otherwise
        the 2-ATR stop and 3-ATR TP exist only inside this tracker and the live
        position is unprotected between 4h checks.
        """
        return {c: float(s["stop"]) for c, s in self.state.get("slots", {}).items()
                if s.get("stop")}

    def protective_tps(self) -> dict:
        """{coin: take_profit_price} for every open BOS slot (3.0 x ATR below)."""
        return {c: float(s["tp"]) for c, s in self.state.get("slots", {}).items()
                if s.get("tp")}

    def _check_exit(self, coin, slot):
        d4 = _load_4h(coin)
        if d4 is None:
            return
        ts, o, hh, ll, cl, vv, tb = d4
        n = len(cl)
        entry_i = slot["entry_i"]
        # entry index may no longer align if the underlying file was
        # rebuilt/reordered -- guard rather than silently mis-index.
        if entry_i >= n or int(ts[entry_i]) != slot["entry_ts"]:
            slot["_closed"] = True
            return

        j_start = max(entry_i, slot["entry_i"] + slot["bars_checked"])
        j_end = min(entry_i + BOS_EXIT_BAR, n)
        last_h = slot.get("frozen_last_high")
        for j in range(j_start, j_end):
            if hh[j] >= slot["stop"]:
                slot["_closed"] = True
                return
            if ll[j] <= slot["tp"]:
                slot["_closed"] = True
                return
            if last_h is not None and cl[j] > last_h:
                slot["_closed"] = True
                return
            slot["bars_checked"] = j - slot["entry_i"] + 1
        if entry_i + slot["bars_checked"] >= entry_i + BOS_EXIT_BAR:
            slot["_closed"] = True  # 30-bar max hold reached

    def _weights(self, coins):
        w = {c: 0.0 for c in coins}
        for coin in self.state["slots"]:
            if coin in w:
                w[coin] = -1.0 / self.MAX_CONCURRENT
        return w


# --------------------------------------------------------------------------
# State-derived protective levels.
#
# The orchestrator holds the raw tracker STATE DICTS (mutated in place by
# get_snapshot), not the tracker objects. These read the protective levels
# straight off that state so the stop sync does not depend on holding a live
# tracker instance.
# --------------------------------------------------------------------------

def short_protective_stops(short_state: dict) -> dict:
    """{coin: chandelier stop} for open short slots. Empty until a slot has
    survived at least one 4h bar past entry (chand_stop is written by the
    forward walk). Slots opened this cycle already carry an exchange stop
    from _attach_stops, so nothing is left unprotected."""
    out = {}
    for coin, slot in (short_state or {}).get("slots", {}).items():
        sl = slot.get("chand_stop")
        if sl is not None and float(sl) > 0:
            out[coin] = float(sl)
    return out


def bos_protective_levels(bos_state: dict) -> tuple:
    """({coin: 2-ATR stop}, {coin: 3-ATR take-profit}) for open BOS slots.
    Fixed at entry, but must still be pushed -- they previously existed only
    inside the tracker."""
    stops, tps = {}, {}
    for coin, slot in (bos_state or {}).get("slots", {}).items():
        if slot.get("stop"):
            stops[coin] = float(slot["stop"])
        if slot.get("tp"):
            tps[coin] = float(slot["tp"])
    return stops, tps


def stop_fracs_from_state(short_state: dict, bos_state: dict) -> dict:
    """{"short": {coin: stop_frac}, "bos": {coin: stop_frac}} for open event-
    sleeve slots, derived from each slot's OWN entry price and current
    protective stop -- consumed by portfolio_layer.portfolio.size_portfolio()'s
    max-loss-per-trade cap.

    short/bos do NOT use the fixed leg-level stop (LIVE_STOP_LOSS_FRAC=0.40)
    that main/flow/delta/relvol use (chandelier trail for short, fixed
    2x-ATR for bos), so defaulting them to 0.60 in the sizer would be wrong
    in either direction. This computes the REAL per-slot fraction where
    available.

    Short slots opened THIS cycle have no chand_stop yet (it is only written
    after the tracker's forward walk survives at least one 4h bar past entry
    -- see ShortSleeveTracker.update()/protective_stops() docstrings), so
    there is no real stop distance to read for them yet. Those coins are
    simply omitted here; size_portfolio() then falls back to its own
    DEFAULT_STOP_FRAC (0.60), which is the CONSERVATIVE direction (short's
    real chandelier stop is typically tighter than 60% of entry, so assuming
    0.60 implies a smaller/tighter notional cap than the true one -- it can
    only under-size, never let a leg through that the real stop would have
    flagged)."""
    out = {"short": {}, "bos": {}}
    for coin, slot in (short_state or {}).get("slots", {}).items():
        entry = slot.get("entry_price")
        stop = slot.get("chand_stop")
        if entry and stop and entry > 0:
            out["short"][coin] = abs(float(stop) - float(entry)) / float(entry)
    for coin, slot in (bos_state or {}).get("slots", {}).items():
        entry = slot.get("entry_price")
        stop = slot.get("stop")
        if entry and stop and entry > 0:
            out["bos"][coin] = abs(float(stop) - float(entry)) / float(entry)
    return out

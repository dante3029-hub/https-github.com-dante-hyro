"""
Slot tracker for the four price-event sleeves: sr, srflip, pattern, fvg.

ONE class, parameterised by sleeve name, rather than four near-identical ones.
signal_engine.sleeve_price_events.latest_signal() only answers "is there a
fresh trigger on the latest closed bar"; slot management, exits and restart
recovery live here -- the same split as ShortSleeveTracker and BOSSleeveTracker.

CONTRACT (matches the existing trackers so size_portfolio() sees a uniform shape)
  - MAX_CONCURRENT slots, 1/MAX_CONCURRENT gross per slot, so a fully deployed
    sleeve's absolute weights sum to 1.0
  - state is a plain JSON-serialisable dict owned by bot/state.py, mutated in
    place; restart recovery is "reload the JSON and keep calling update()"
  - update(coins) returns {coin: weight}, directly usable as
    sleeve_raw_weights[<sleeve>]

THREE THINGS THIS GETS RIGHT THAT COST REAL MONEY TO LEARN

1. **The OI gate is applied at ORDER time, not at detection.**
   On SOL 6h there are 60 raw sr breaks. Gating on OI at the SIGNAL bar keeps
   31; gating at the ENTRY bar keeps 23; the two sets overlap by only 12 --
   29%. The backtest gates at entry. Both are causal (OI is a live snapshot,
   observable when the order goes in), but they select almost different
   strategies. latest_signal() is called with oi_expanding=None so it DETECTS
   only, and this tracker queries OI when it opens the slot.

2. **The exit walk resumes from `last_checked_ts`, never from entry.**
   ShortSleeveTracker carried a documented bug where the walk restarted at
   entry every cycle while trailing state persisted, so bars already survived
   were re-tested against a ratcheted stop and force-exited a slot with ZERO
   new market data. Same defect class avoided here.

3. **One position per coin per sleeve.**
   A coin already holding a slot cannot open another. Without this, sr stacked
   4 of its 6 slots into one coin across 5,304 coin-hours, and fvg 3 slots
   across 16,525. That concentration earned nothing -- capping it was free and
   improved three of four sleeves.

EXITS
  stop   checked from the ENTRY BAR onward, not entry+1. Skipping the entry bar
         made tight stops look monotonically better (fvg at 0.25 ATR "scored"
         2.84; it is really 0.06).
  time   after `hold_bars` bars on the sleeve's own timeframe.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from signal_engine.sleeve_price_events import PARAMS, latest_signal, MAX_CONCURRENT


def new_long_tracker_state() -> dict:
    """Fresh state for one sleeve. JSON-serialisable throughout."""
    return {"slots": {}}


class PriceEventTracker:
    """Slot tracker for one price-event sleeve.

    bars_fn(coin)  -> that sleeve's native-timeframe OHLCV, newest last,
                      last bar CLOSED. None if unavailable.
    oi_fn(coin, ts)-> True if open interest expanded on the bar at `ts`.
                      Only called for sleeves whose config needs it, and only
                      at ORDER time.
    """

    def __init__(self, sleeve: str, state: dict,
                 bars_fn: Callable[[str], Optional[pd.DataFrame]],
                 oi_fn: Optional[Callable[[str, pd.Timestamp], bool]] = None):
        if sleeve not in PARAMS:
            raise ValueError(f"unknown sleeve {sleeve!r}")
        self.sleeve = sleeve
        self.params = PARAMS[sleeve]
        self.state = state
        self.state.setdefault("slots", {})
        self.bars_fn = bars_fn
        self.oi_fn = oi_fn
        self.max_concurrent = MAX_CONCURRENT

    # ── exits ────────────────────────────────────────────────────────────
    def _walk_open_slots(self) -> List[str]:
        """Advance every open slot and close any whose exit has fired.
        Returns the coins closed this cycle (for logging)."""
        slots = self.state["slots"]
        closed = []
        for coin in list(slots.keys()):
            slot = slots[coin]
            bars = self.bars_fn(coin)
            if bars is None or bars.empty:
                continue                      # no data: hold, do not guess
            # resume from the last bar already evaluated -- NEVER from entry
            resume = slot.get("last_checked_ts") or slot["entry_ts"]
            try:
                fresh = bars[bars.index > pd.Timestamp(resume)]
            except Exception:
                continue
            if fresh.empty:
                continue
            side = int(slot["side"])
            stop = float(slot["stop"])
            exited = False
            for ts, row in fresh.iterrows():
                slot["bars_held"] = int(slot.get("bars_held", 0)) + 1
                hit = ((side > 0 and float(row["low"]) <= stop) or
                       (side < 0 and float(row["high"]) >= stop))
                if hit:
                    slot["exit_reason"] = "stop"
                    del slots[coin]
                    closed.append(coin)
                    exited = True
                    break
                if slot["bars_held"] >= int(slot["hold_bars"]):
                    slot["exit_reason"] = "time"
                    del slots[coin]
                    closed.append(coin)
                    exited = True
                    break
            if not exited:
                slot["last_checked_ts"] = str(fresh.index[-1])
        return closed

    # ── entries ──────────────────────────────────────────────────────────
    def _try_open(self, coins: List[str]) -> List[str]:
        """Open new slots where there is room and a fresh trigger."""
        slots = self.state["slots"]
        opened = []
        for coin in coins:
            if len(slots) >= self.max_concurrent:
                break
            if coin in slots:
                continue                      # one position per coin per sleeve
            bars = self.bars_fn(coin)
            if bars is None or len(bars) < 700:
                continue
            sig = latest_signal(self.sleeve, bars, oi_expanding=None)
            if sig is None:
                continue
            # THE OI GATE, AT ORDER TIME -- see the module docstring
            if sig.needs_oi:
                if self.oi_fn is None:
                    continue                  # cannot verify: do not trade
                try:
                    if not self.oi_fn(coin, bars.index[-1]):
                        continue
                except Exception:
                    continue
            # entry is the NEXT bar's open, which has not printed yet. The
            # last close is the best reference available now; execution
            # reports the real fill and the stop is recomputed from it.
            entry_ref = float(sig.entry_ref)
            slots[coin] = {
                "side": int(sig.side),
                "entry_ts": str(bars.index[-1]),
                "last_checked_ts": str(bars.index[-1]),
                "entry_ref": entry_ref,
                "stop": float(sig.stop_from(entry_ref)),
                "atr": float(sig.atr),
                "stop_atr": float(sig.stop_atr),
                "hold_bars": int(sig.hold_bars),
                "bars_held": 0,
            }
            opened.append(coin)
        return opened

    # ── the cycle ────────────────────────────────────────────────────────
    def update(self, coins: List[str]) -> Dict[str, float]:
        """Run one cycle: walk exits, then open new slots.

        Returns {coin: weight}. An EMPTY dict is normal for an event sleeve in
        a quiet period -- sleeve_health knows the expected firing rate and only
        escalates when a sleeve is flat far longer than it should be.
        """
        self._walk_open_slots()
        self._try_open(coins)
        w = 1.0 / self.max_concurrent
        return {c: s["side"] * w for c, s in self.state["slots"].items()}

    # ── for the execution layer ──────────────────────────────────────────
    def protective_stops(self) -> Dict[str, float]:
        """{coin: stop price} for every open slot, so the execution layer can
        keep a real stop order on the exchange rather than relying on the bot
        being alive to close the position."""
        return {c: float(s["stop"]) for c, s in self.state["slots"].items()}

    def set_actual_entry(self, coin: str, fill_price: float) -> None:
        """Recompute the stop from the ACTUAL fill.

        The stop must sit a fixed ATR distance from where you really got in,
        not from the previous close. Placing it off the signal bar's close
        while entering at the next open is lookahead -- it was worth -0.21 on
        fvg in the backtest before it was fixed.
        """
        slot = self.state["slots"].get(coin)
        if not slot:
            return
        side = int(slot["side"])
        slot["entry_px"] = float(fill_price)
        slot["stop"] = float(fill_price - side * slot["stop_atr"] * slot["atr"])

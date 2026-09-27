"""
orchestrator_v2.py — one cycle of the nine-sleeve book, with the health check
wired in as a HARD GATE.

This is the glue: it does not compute signals (signal_engine does) and does not
size positions (portfolio_layer does). It decides which sleeves are due, asks
them for targets, runs the health check, and refuses to trade if anything is
CRITICAL.

THE GATE IS THE POINT
Three failures on this project ran silently:
  * BOS produced zero weights on 300+ consecutive cycles while the log said
    "rebalanced=['short','bos']"
  * run/ went 408 hours stale while cycles kept "succeeding"
  * the taker column had taker_buy > total volume on 60% of bars
Each was findable in ONE cycle. `HALT_ON_CRITICAL` makes that automatic: a
sleeve whose data cannot be verified does not trade.

SEPARATION
  signal_engine.sleeve_*        -> what each sleeve wants
  bot.price_event_tracker       -> slot management for the event sleeves
  portfolio_layer.size_portfolio-> netting, caps, dollar sizing
  sleeve_health                 -> is any of this trustworthy
  THIS FILE                     -> cadence, orchestration, and the gate
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CROSS_SECTIONAL = ("delta", "relvol", "skew", "oirank")
EVENT_SLEEVES = ("sr", "srflip", "pattern", "fvg")
CASCADE = "cascade"


@dataclass
class CycleResult:
    """Everything one cycle produced, including WHY it did or did not trade."""
    ts: pd.Timestamp
    due: List[str] = field(default_factory=list)
    raw_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    findings: list = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""
    orders_placed: int = 0

    def summary(self) -> str:
        if self.halted:
            return f"{self.ts}  HALTED -- {self.halt_reason}"
        live = sum(1 for w in self.raw_weights.values() if w)
        return (f"{self.ts}  due={len(self.due)}  sleeves_with_positions={live}"
                f"  orders={self.orders_placed}")


def hours_since(ts_str: Optional[str], now: pd.Timestamp) -> Optional[float]:
    if not ts_str:
        return None
    try:
        return (now - pd.Timestamp(ts_str)).total_seconds() / 3600.0
    except Exception:
        return None


class Orchestrator:
    """One cycle of the book.

    cadence:     {sleeve: hours}
    xs_fn:       (sleeve, t) -> (weights_array, ok) for cross-sectional sleeves
    trackers:    {sleeve: PriceEventTracker} for the event sleeves
    cascade_fn:  () -> (weights_array, ok)
    coins_fn:    (sleeve) -> the coin list for that sleeve's universe
    """

    def __init__(self, state: dict, cadence: Dict[str, int],
                 xs_fn: Callable, trackers: Dict[str, object],
                 cascade_fn: Callable, coins_fn: Callable,
                 expectations: list, sanity_files: Dict[str, str],
                 halt_on_critical: bool = True):
        self.state = state
        self.state.setdefault("last_rebalance", {})
        self.state.setdefault("flat_cycles", {})
        self.cadence = cadence
        self.xs_fn = xs_fn
        self.trackers = trackers
        self.cascade_fn = cascade_fn
        self.coins_fn = coins_fn
        self.expectations = expectations
        self.sanity_files = sanity_files
        self.halt_on_critical = halt_on_critical

    # ── cadence ──────────────────────────────────────────────────────────
    def _due(self, sleeve: str, now: pd.Timestamp) -> bool:
        """Event sleeves are checked EVERY cycle regardless of cadence -- their
        cadence describes their bar size, not a rebalance schedule. A slot can
        need closing at any time."""
        if sleeve in EVENT_SLEEVES:
            return True
        last = self.state["last_rebalance"].get(sleeve)
        h = hours_since(last, now)
        return h is None or h >= self.cadence.get(sleeve, 24)

    # ── one cycle ────────────────────────────────────────────────────────
    def run_cycle(self, now: Optional[pd.Timestamp] = None) -> CycleResult:
        now = now or pd.Timestamp.now(tz="UTC")
        res = CycleResult(ts=now)
        weights: Dict[str, Dict[str, float]] = {}

        for sleeve in list(CROSS_SECTIONAL) + [CASCADE] + list(EVENT_SLEEVES):
            if not self._due(sleeve, now):
                # not due: carry the previous target forward untouched
                weights[sleeve] = self.state.get("targets", {}).get(sleeve, {})
                continue
            res.due.append(sleeve)
            try:
                if sleeve in CROSS_SECTIONAL:
                    w = self.xs_fn(sleeve)
                elif sleeve == CASCADE:
                    w = self.cascade_fn()
                else:
                    tr = self.trackers.get(sleeve)
                    w = tr.update(self.coins_fn(sleeve)) if tr else {}
            except Exception as e:
                # a sleeve that throws is NOT silently treated as flat -- that
                # is how BOS hid. Record it and let the health check see it.
                log.exception("sleeve %s raised", sleeve)
                res.findings.append(
                    _finding("CRITICAL", sleeve, "sleeve raised an exception",
                             f"{type(e).__name__}: {e}"))
                w = None
            weights[sleeve] = w if w is not None else {}
            if w:
                self.state["last_rebalance"][sleeve] = str(now)

        # track how long each sleeve has been flat, for the health check
        for sleeve, w in weights.items():
            if w and any(abs(v) > 1e-9 for v in w.values()):
                self.state["flat_cycles"][sleeve] = 0
            else:
                self.state["flat_cycles"][sleeve] = \
                    int(self.state["flat_cycles"].get(sleeve, 0)) + 1

        res.raw_weights = weights
        self.state["targets"] = weights

        # ── THE GATE ──
        from sleeve_health import run_health_check, CRITICAL
        res.findings += run_health_check(
            self.expectations,
            weights_by_sleeve=weights,
            flat_cycles=self.state["flat_cycles"],
            hours_since_rebalance={
                s: hours_since(self.state["last_rebalance"].get(s), now)
                for s in self.cadence},
            sanity_files=self.sanity_files,
            now=now)
        crit = [f for f in res.findings if f.severity == CRITICAL]
        if crit and self.halt_on_critical:
            res.halted = True
            res.halt_reason = f"{len(crit)} CRITICAL: {crit[0].message}"
        return res


def _finding(sev, sleeve, msg, detail=""):
    from sleeve_health import Finding
    return Finding(sev, sleeve, msg, detail)

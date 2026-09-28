"""
fast_detect.py — incremental detectors: precompute once, then read bar i in O(1).

WHY
The detectors recompute from scratch on every call. `sr2.signals()` builds
levels across the whole series; srflip loops every bar hunting breaks; the
pattern detector rescans all 16 patterns. That is O(n) per call.

Live that is fine -- one call per coin per 6h close. But it makes a bar-by-bar
replay O(n^2): a six-month replay across ten coins did not finish. Which means
the ONE test that exercises the live code path end to end could not be run
routinely -- exactly the test you most want before and during paper trading.

APPROACH
Build the per-bar arrays ONCE per coin, then `fired(i)` is an array read. The
arrays are constructed causally, so reading index i never sees anything after
i -- the same guarantee the bar-by-bar path gives, at a fraction of the cost.

These must produce IDENTICAL signals to the slow path. That is asserted in
verify_against_slow() rather than assumed.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd


class FastSR:
    """S/R break and flip. Levels built once, then indexed."""

    def __init__(self, bars: pd.DataFrame, flip_window: int = 20):
        import sr2
        self.n = len(bars)
        sup, sup1, res, res1 = sr2.levels(bars)
        self.low = bars['low'].to_numpy(float)
        self.close = bars['close'].to_numpy(float)
        # break: LOW crosses above the outer edge, matching the Pine
        # (ta.crossover(low, resistanceLevel_1)). Using close instead broke
        # parity and turned the second half negative.
        L, S = sr2.signals(bars, 'break_res')
        self.break_at = np.asarray(L, dtype=bool)
        # flip: after a break, price returns to the INNER level and holds
        self.flip_at = np.zeros(self.n, dtype=bool)
        broke, lvl = None, np.nan
        for i in range(1, self.n):
            if np.isfinite(res1[i]) and self.low[i] > res1[i] and self.low[i-1] <= res1[i-1]:
                broke, lvl = i, res[i]
            elif (broke is not None and i - broke <= flip_window
                  and np.isfinite(lvl) and self.low[i] <= lvl and self.close[i] > lvl):
                self.flip_at[i] = True
                broke = None

    def fired(self, i: int, mode: str = 'break') -> bool:
        if i < 1 or i >= self.n:
            return False
        return bool(self.break_at[i] if mode == 'break' else self.flip_at[i])


class FastFVG:
    """Fair value gap -- a pure 3-bar pattern. Verified an exact match to the
    LuxAlgo Pine (575 of 575 gaps identical)."""

    def __init__(self, bars: pd.DataFrame, threshold: float = 0.0):
        h = bars['high'].to_numpy(float)
        l = bars['low'].to_numpy(float)
        c = bars['close'].to_numpy(float)
        n = len(c)
        self.n = n
        self.side = np.zeros(n, dtype=int)
        thr = threshold / 100.0
        for i in range(2, n):
            if h[i-2] > 0 and l[i] > h[i-2] and c[i-1] > h[i-2]:
                if (l[i] - h[i-2]) / h[i-2] > thr:
                    self.side[i] = 1
            elif h[i] > 0 and h[i] < l[i-2] and c[i-1] < l[i-2]:
                if (l[i-2] - h[i]) / h[i] > thr:
                    self.side[i] = -1

    def fired(self, i: int) -> int:
        return int(self.side[i]) if 0 <= i < self.n else 0


class FastPattern:
    """MarkitTick, indexed by DETECTION bar -- never the historical break bar
    the detector also reports. Entering at the break is lookahead and once
    inflated a backtest from 1.04 to 6.06."""

    def __init__(self, bars: pd.DataFrame, bull_only: bool = True):
        from markittick_detector import detect
        n = len(bars)
        self.n = n
        self.side = np.zeros(n, dtype=int)
        for rec in detect(bars):
            is_bull = rec[2]
            if bull_only and not is_bull:
                continue
            det = rec[7] if len(rec) > 7 else rec[0]
            if 0 <= det < n:
                self.side[det] = 1 if is_bull else -1

    def fired(self, i: int) -> int:
        return int(self.side[i]) if 0 <= i < self.n else 0


def verify_against_slow(bars: pd.DataFrame, sleeve: str,
                        lo: int = 3000, hi: int = 3100) -> dict:
    """Assert the fast path fires on exactly the same bars as the slow one.

    Run this whenever a detector changes. A fast path that silently diverges
    is worse than no fast path.
    """
    from signal_engine.sleeve_price_events import latest_signal
    sample = range(lo, min(hi, len(bars)))
    slow = {i for i in sample
            if latest_signal(sleeve, bars.iloc[:i+1], oi_expanding=None) is not None}
    if sleeve in ('sr', 'srflip'):
        f = FastSR(bars)
        mode = 'break' if sleeve == 'sr' else 'flip'
        fast = {i for i in sample if f.fired(i, mode)}
    elif sleeve == 'fvg':
        f = FastFVG(bars)
        fast = {i for i in sample if f.fired(i) != 0}
    else:
        f = FastPattern(bars)
        fast = {i for i in sample if f.fired(i) != 0}
    return dict(sleeve=sleeve, identical=(slow == fast),
                slow=len(slow), fast=len(fast),
                only_slow=sorted(slow - fast)[:5],
                only_fast=sorted(fast - slow)[:5])

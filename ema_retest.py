"""EMA retest sleeve — pull back to a rising EMA, HOLD it, then enter.

MECHANISM
  1. EMA is rising and price is above it      -> uptrend confirmed
  2. price pulls back and TOUCHES the EMA     -> the retest
  3. price closes above the EMA for `hold_bars` consecutive bars -> it HELD
  4. enter long at the next open

Step 3 is the part being tested. A touch that slices straight through is not a
retest; requiring N closes above filters those out.

Structurally different from sr/srflip (which use volume-derived levels) and
from breakout (which buys strength). This buys a pullback in a confirmed trend.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd

TAKER = '/tmp/hyro/taker_data'
FEE = 0.00085


def atr(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def trades(d, ema_len=100, hold_bars=2, stop_atr=3.0, max_hold=20,
           touch_window=5):
    """Returns (entry_ts, exit_ts, side)."""
    c = d['close'].to_numpy(float)
    h = d['high'].to_numpy(float)
    l = d['low'].to_numpy(float)
    o = d['open'].to_numpy(float)
    ema = pd.Series(c).ewm(span=ema_len, adjust=False).mean().to_numpy()
    A = atr(d).to_numpy()
    n = len(c)
    out = []
    i = ema_len + 5
    while i < n - max_hold - 2:
        # uptrend: EMA rising over the last 10 bars, price above it
        if not (np.isfinite(ema[i]) and ema[i] > ema[i-10] and c[i] > ema[i]):
            i += 1
            continue
        # a touch in the last `touch_window` bars: low dipped to/below the EMA
        touched = any(l[j] <= ema[j] for j in range(max(0, i-touch_window), i+1))
        if not touched:
            i += 1
            continue
        # HELD: the last `hold_bars` closes are all above the EMA
        held = all(c[j] > ema[j] for j in range(i-hold_bars+1, i+1))
        if not held:
            i += 1
            continue
        eb = i + 1
        if eb >= n - 1 or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1
            continue
        stp = o[eb] - stop_atr*A[i]
        ex = min(eb + max_hold, n - 1)
        for j in range(eb, min(eb + 1 + max_hold, n)):   # entry bar INCLUDED
            if l[j] <= stp:
                ex = j
                break
        out.append((d.index[eb], d.index[ex], 1))
        i = eb + max_hold          # no overlapping entries in one coin
    return out

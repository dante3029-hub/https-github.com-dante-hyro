#!/usr/bin/env python3
"""
sleeve_fvg.py — Fair Value Gap, ONE implementation for live and backtest.

Same pattern as sleeve_sr.py: a single `step()` advancing a state machine by
one bar using only data up to that bar. Backtesting sweeps it; the live bot
calls it once with the latest closed bar. Parity is structural.

WHY THIS SLEEVE GETS THIS TREATMENT FIRST
It is the largest in the book (3.28 Sharpe, 204% annual) and the least
verified. Rebuilding S/R this way found two bugs inside an hour.

DETECTION — verified line-by-line against LuxAlgo's Pine
    bull_fvg = low > high[2] and close[1] > high[2]
               and (low - high[2]) / high[2] > threshold
    bear_fvg = high < low[2] and close[1] < low[2]
               and (low[2] - high) / high > threshold
    threshold = thresholdPer/100, default 0 (auto mode uses a running mean of
                (high-low)/low, tested and worse: 0.69 vs 1.16)

A gap is a price range that NEVER TRADED -- price moved fast enough to skip it.
That is the imbalance being traded.

THE TRADING RULE IS MINE, NOT THE SOURCE'S
The LuxAlgo indicator only DRAWS boxes. It has no entry, stop or exit. What
follows was arrived at by sweep and is not validated by the Pine:

    direction   WITH the gap (bull gap -> long)
    entry       open of the next bar after detection
    stop        1 x ATR(14)  -- tight: a short-horizon imbalance that does not
                resolve quickly has failed
    exit        10 bars, or the stop
    filter      open interest expanding on the entry bar
    bars        12h
    concurrent  max 6, hard-capped

Sweeps behind those choices:
    fading the gap            -0.92   (gaps continue, they do not fill)
    entering on the retest     0.14 at the edge, WORSE the deeper you wait
                               (touch 0.14 / 50% -0.18 / full fill -0.34)
    take-profit at 3 ATR       0.63 vs 1.33 without -- capping winners hurts
    stop 1 ATR                 1.44 at hold 10; 2 ATR is the WORST of four
    hold 30                    every variant has a weak or negative 2nd half

KNOWN FIXES ALREADY APPLIED
  * stop was placed off c[eb] / A[eb] -- the entry bar's CLOSE -- while entering
    at its OPEN. Lookahead worth -0.21. Now uses o[eb] and ATR at the signal bar.
  * a fixed 1/6 position size does NOT cap concurrency. FVG ran at a mean of
    12.5 simultaneous positions, 2.1x levered. Capping RAISED Sharpe 3.29 ->
    3.49 and cut max drawdown 91% -> 28%.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ── parameters — single source of truth for live and backtest ────────────────
THRESHOLD_PCT = 0.0        # source default; auto mode tested worse
STOP_ATR = 1.0             # tight, per the sweep
HOLD_BARS = 10
ATR_PERIOD = 14
TIMEFRAME_H = 12
MAX_CONCURRENT = 6


@dataclass
class FVGState:
    """FVG needs almost no memory -- detection is a pure 3-bar pattern. The
    state exists for interface symmetry with sleeve_sr and to carry the last
    processed bar so the live bot can detect gaps in its data feed."""
    last_bar: Optional[str] = None
    last_signal_bar: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "FVGState":
        return FVGState(**json.loads(s))


def _atr(high, low, close, n=ATR_PERIOD):
    m = len(close)
    tr = np.zeros(m)
    tr[0] = high[0] - low[0]
    for i in range(1, m):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    out = np.full(m, np.nan)
    a = np.nan
    for i in range(m):
        a = tr[i] if not np.isfinite(a) else (a*(n-1) + tr[i]) / n
        if i >= n:
            out[i] = a
    return out


def prepare(df: pd.DataFrame) -> dict:
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    return {'o': o, 'h': h, 'l': l, 'c': c,
            'atr': _atr(h, l, c),
            'index': df.index, 'n': len(c)}


def step(P: dict, i: int, st: FVGState, threshold: float = THRESHOLD_PCT) -> dict:
    """Advance one bar. Uses ONLY data up to and including bar i.

    Returns {'bull': bool, 'bear': bool, 'gap_hi': float, 'gap_lo': float}.
    """
    out = {'bull': False, 'bear': False, 'gap_hi': None, 'gap_lo': None}
    if i < 2 or i >= P['n']:
        return out
    h, l, c = P['h'], P['l'], P['c']
    thr = threshold / 100.0
    # bullish: a gap between high[i-2] and low[i]
    if h[i-2] > 0 and l[i] > h[i-2] and c[i-1] > h[i-2]:
        if (l[i] - h[i-2]) / h[i-2] > thr:
            out['bull'] = True
            out['gap_hi'], out['gap_lo'] = float(l[i]), float(h[i-2])
    # bearish: a gap between low[i-2] and high[i]
    elif h[i] > 0 and h[i] < l[i-2] and c[i-1] < l[i-2]:
        if (l[i-2] - h[i]) / h[i] > thr:
            out['bear'] = True
            out['gap_hi'], out['gap_lo'] = float(l[i-2]), float(h[i])
    st.last_bar = str(P['index'][i])
    if out['bull'] or out['bear']:
        st.last_signal_bar = st.last_bar
    return out


def live_signal(df: pd.DataFrame, state_json: Optional[str] = None,
                oi_expanding: bool = True) -> dict:
    """LIVE entry point. `df` is the coin's 12h bars, newest last and CLOSED.

    Returns the trigger on that bar plus entry reference, stop and hold.
    """
    P = prepare(df)
    st = FVGState.from_json(state_json) if state_json else FVGState()
    i = P['n'] - 1
    res = step(P, i, st)
    a = P['atr'][i]
    side = 1 if res['bull'] else (-1 if res['bear'] else 0)
    fires = bool(side != 0 and oi_expanding)
    # entry is the NEXT bar's open; c[i] is the best reference available now
    ref = float(P['c'][i])
    return {
        'fires': fires,
        'side': side,
        'gap_hi': res['gap_hi'],
        'gap_lo': res['gap_lo'],
        'entry_ref': ref,
        'stop': float(ref - side*STOP_ATR*a) if (side and np.isfinite(a)) else None,
        'exit_after_bars': HOLD_BARS,
        'state': st.to_json(),
    }


def backtest(df: pd.DataFrame, oi_up: Optional[pd.Series] = None,
             stop_atr: float = STOP_ATR, hold: int = HOLD_BARS,
             threshold: float = THRESHOLD_PCT, fade: bool = False) -> list:
    """BACKTEST — sweeps the SAME step() function.

    Returns [(entry_ts, exit_ts, entry_px, exit_px, ret, side)].
    """
    P = prepare(df)
    st = FVGState()
    o, h, l, c = P['o'], P['h'], P['l'], P['c']
    trades = []
    for i in range(2, P['n'] - hold - 2):
        res = step(P, i, st, threshold)
        side = 1 if res['bull'] else (-1 if res['bear'] else 0)
        if side == 0:
            continue
        if fade:
            side = -side
        eb = i + 1
        a = P['atr'][i]                      # ATR at the SIGNAL bar
        if eb >= P['n'] or not np.isfinite(a) or a <= 0:
            continue
        if oi_up is not None:
            v = oi_up.reindex([P['index'][eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
        ent = o[eb]                          # entry at the OPEN
        stp = ent - side*stop_atr*a          # stop off the ENTRY, not the close
        ex, px = min(eb + hold, P['n'] - 1), None
        # check the ENTRY BAR too. Starting at eb+1 skipped any stop hit during
        # the bar you entered on -- which made tighter stops look monotonically
        # better (0.25ATR "2.84" vs 1.0ATR 1.05). Pure artifact.
        for j in range(eb, min(eb + 1 + hold, P['n'])):
            if (side > 0 and l[j] <= stp) or (side < 0 and h[j] >= stp):
                ex, px = j, stp
                break
        if px is None:
            px = c[ex]
        trades.append((P['index'][eb], P['index'][ex], float(ent), float(px),
                       float(side*(px - ent)/ent), int(side)))
    return trades

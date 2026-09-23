#!/usr/bin/env python3
"""
sleeve_sr.py — S/R break and S/R flip, ONE implementation for live and backtest.

WHY THIS EXISTS
Running sr2.py (backtest) and hourly_sim.py (portfolio) side by side produced
403 vs 572 trades on the same universe and window, and flipped the exit
conclusion: trade-level said 3xATR beat 2xATR, hourly said the reverse. Two
implementations, two answers, no way to know which the live bot would do.

This module has ONE signal function. Backtesting sweeps it bar by bar; the live
bot calls it once with the latest closed bar. Parity is structural, not
something to test for.

THE MECHANISM

  LEVEL FORMATION
    pivot on CLOSE, `pivot_k` bars either side -> only confirmed k bars later
    delta volume checked at the CONFIRMATION bar, not the pivot bar
      (a pivot high is a bearish candle; its own delta is always negative, so
       testing it there can never fire -- this cost several hours to find)
    resistance forms at a pivot HIGH when delta vol is at its 2-bar LOW
    support    forms at a pivot LOW  when delta vol is at its 2-bar HIGH
    box width = ATR(200) x box_mult, so res1 = res + width

  TRIGGERS
    sr      close crosses ABOVE the outer edge res1 -- price cleared the box
    srflip  after that break, price returns to the INNER level res (low touches
            it, close stays above) within `flip_window` bars. The level has
            flipped from resistance to support and held.

  EXECUTION
    entry at the OPEN of the next bar, never the signal bar
    stop  = entry - stop_atr x ATR(14)
    exit  on stop or after `hold` bars
    sr additionally requires OPEN INTEREST EXPANDING on the entry bar

STATE
srflip needs memory: "resistance at X was broken at bar N". That is carried in
the SRState object so the live bot can persist it across restarts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ── parameters — single source of truth for live and backtest ────────────────
PIVOT_K = 20          # bars each side of a pivot (level confirmed K bars later)
VOL_LEN = 2           # lookback for the delta-volume extreme
VOL_SCALE = 2.5       # source scales the THRESHOLD, not the value compared
BOX_MULT = 1.0        # box width = ATR(200) x this
ATR_BOX = 200
ATR_STOP_PERIOD = 14
FLIP_WINDOW = 20      # bars after a break in which a retest still counts

# exits — 2ATR vs 3ATR is UNRESOLVED between the two old implementations.
# Defaulting to 3.0; the backtest in this file is what should settle it.
STOP_ATR = 3.0
HOLD_BARS = 15
MAX_CONCURRENT = 6


@dataclass
class SRState:
    """Everything the sleeve must remember between bars. JSON-serialisable so
    the live bot can persist it across restarts."""
    level: Optional[float] = None        # current resistance (inner edge)
    level_outer: Optional[float] = None  # outer edge = level + box width
    support: Optional[float] = None
    support_outer: Optional[float] = None
    broke_at: Optional[int] = None       # bar index of the last break
    broke_level: Optional[float] = None  # the level that was broken
    last_bar: Optional[str] = None       # timestamp of the last bar processed

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "SRState":
        return SRState(**json.loads(s))


def _atr(high, low, close, n):
    """Wilder ATR as a numpy array."""
    n_bars = len(close)
    tr = np.zeros(n_bars)
    tr[0] = high[0] - low[0]
    for i in range(1, n_bars):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    out = np.full(n_bars, np.nan)
    a = np.nan
    for i in range(n_bars):
        a = tr[i] if not np.isfinite(a) else (a*(n-1) + tr[i]) / n
        if i >= n:
            out[i] = a
    return out


def prepare(df: pd.DataFrame) -> dict:
    """Precompute everything that does not depend on the bar being evaluated.
    Cheap to call once per scan in live; called once per coin in backtest."""
    o = df['open'].to_numpy(float)
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    v = df['volume'].to_numpy(float)
    # upAndDownVolume() from the source signs volume by CANDLE DIRECTION, not
    # by taker flow. Using real taker delta here is different data and produced
    # 631 trades vs the source-faithful 403.
    sign = np.where(c > o, 1.0, np.where(c < o, -1.0, np.nan))
    sign = pd.Series(sign).ffill().fillna(1.0).to_numpy()
    dv = sign * v
    s = pd.Series(dv)
    return {
        'o': o, 'h': h, 'l': l, 'c': c, 'dv': dv,
        'vol_hi': (s / VOL_SCALE).rolling(VOL_LEN).max().to_numpy(),
        'vol_lo': (s / VOL_SCALE).rolling(VOL_LEN).min().to_numpy(),
        'atr_box': _atr(h, l, c, ATR_BOX),
        'atr_stop': _atr(h, l, c, ATR_STOP_PERIOD),
        'index': df.index,
        'n': len(c),
    }


def step(P: dict, i: int, st: SRState) -> dict:
    """Advance the state machine by ONE bar and report any trigger.

    Uses ONLY data up to and including bar i. This is the single function that
    both the backtest and the live bot call, which is what guarantees they
    agree.

    Returns {'break': bool, 'flip': bool, 'level': float|None}.
    """
    out = {'break': False, 'flip': False, 'level': None}
    if i < PIVOT_K * 2 + 2 or i >= P['n']:
        return out

    # ── level formation: a pivot K bars back, confirmed by delta volume HERE ──
    p = i - PIVOT_K                       # the candidate pivot bar
    c, h, l, dv = P['c'], P['h'], P['l'], P['dv']
    lo_w, hi_w = p - PIVOT_K, p + PIVOT_K + 1
    if lo_w >= 0 and hi_w <= i + 1:
        seg = c[lo_w:hi_w]
        w = P['atr_box'][i] * BOX_MULT if np.isfinite(P['atr_box'][i]) else 0.0
        # resistance: pivot HIGH on close + delta volume at its 2-bar LOW
        if c[p] == seg.max() and np.isfinite(P['vol_lo'][i]) and dv[i] < P['vol_lo'][i]:
            st.level, st.level_outer = float(c[p]), float(c[p] + w)
        # support: pivot LOW on close + delta volume at its 2-bar HIGH
        if c[p] == seg.min() and np.isfinite(P['vol_hi'][i]) and dv[i] > P['vol_hi'][i]:
            st.support, st.support_outer = float(c[p]), float(c[p] - w)

    # ── trigger 1: BREAK — LOW crosses above the outer edge ──
    # source: brekout_res := ta.crossover(low, resistanceLevel_1)
    # It is the LOW, not the close -- the whole candle must clear the box.
    if st.level_outer is not None and i >= 1:
        if l[i] > st.level_outer and l[i-1] <= st.level_outer:
            out['break'] = True
            out['level'] = st.level
            st.broke_at, st.broke_level = i, st.level

    # ── trigger 2: FLIP — price returns to the broken level and holds ──
    if (st.broke_at is not None and st.broke_level is not None
            and 0 < i - st.broke_at <= FLIP_WINDOW):
        if l[i] <= st.broke_level and c[i] > st.broke_level:
            out['flip'] = True
            out['level'] = st.broke_level
            st.broke_at = None            # consume it — one retest per break

    st.last_bar = str(P['index'][i])
    return out


def live_signal(df: pd.DataFrame, state_json: Optional[str] = None,
                oi_expanding: bool = True) -> dict:
    """LIVE entry point. Pass the coin's full bar history (newest last, the
    last bar CLOSED) and the persisted state. Returns any trigger on that bar
    plus the new state to save.

    oi_expanding: whether open interest rose on this bar. Required for the
    BREAK trigger; the FLIP does not use it.
    """
    P = prepare(df)
    st = SRState.from_json(state_json) if state_json else SRState()
    # replay from the start if there is no state, else just the final bar
    start = PIVOT_K*2 + 2 if state_json is None else P['n'] - 1
    res = {'break': False, 'flip': False, 'level': None}
    for i in range(start, P['n']):
        res = step(P, i, st)
    i = P['n'] - 1
    a = P['atr_stop'][i]
    entry_ref = float(P['c'][i])          # next bar's open is the real entry
    return {
        'break': bool(res['break'] and oi_expanding),
        'flip': bool(res['flip']),
        'level': res['level'],
        'entry_ref': entry_ref,
        'stop': float(entry_ref - STOP_ATR*a) if np.isfinite(a) else None,
        'exit_after_bars': HOLD_BARS,
        'state': st.to_json(),
    }


def backtest(df: pd.DataFrame, oi_up: Optional[pd.Series] = None,
             which: str = 'break', stop_atr: float = STOP_ATR,
             hold: int = HOLD_BARS) -> list:
    """BACKTEST entry point — sweeps the SAME step() function bar by bar.

    Returns [(entry_ts, exit_ts, entry_px, exit_px, ret)].
    """
    P = prepare(df)
    st = SRState()
    o, h, l, c = P['o'], P['h'], P['l'], P['c']
    trades = []
    for i in range(PIVOT_K*2 + 2, P['n'] - hold - 2):
        res = step(P, i, st)
        fired = res['break'] if which == 'break' else res['flip']
        if not fired:
            continue
        eb = i + 1
        a = P['atr_stop'][i]
        if eb >= P['n'] or not np.isfinite(a) or a <= 0:
            continue
        if which == 'break' and oi_up is not None:
            v = oi_up.reindex([P['index'][eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
        ent = o[eb]
        stp = ent - stop_atr*a
        ex = min(eb + hold, P['n'] - 1)
        px = c[ex]
        for j in range(eb + 1, min(eb + 1 + hold, P['n'])):
            if l[j] <= stp:
                ex, px = j, stp
                break
        trades.append((P['index'][eb], P['index'][ex], float(ent), float(px),
                       float((px - ent)/ent)))
    return trades

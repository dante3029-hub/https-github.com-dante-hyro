"""Calendar-anchored levels — yearly open, weekly high/low, Monday open.

STRUCTURALLY DIFFERENT from sr/srflip, which derive levels from volume-
confirmed pivots. These are anchored to the CLOCK: everyone sees the same
yearly open on the same bar, which is why they get watched.

CAUSALITY
Every level uses only COMPLETED periods. Last week's high is known the moment
this week starts and never changes. The yearly open is known from 1 Jan. No
level is ever computed from the period it is being applied to.

TESTED
  break     price closes above/below the level having been on the other side
  retest    price returns to a broken level and holds
  reject    price touches the level and closes back away from it
"""
import os
import numpy as np, pandas as pd

TAKER = '/tmp/hyro/taker_data'
FEE = 0.00085


def levels(bars: pd.DataFrame) -> pd.DataFrame:
    """All calendar levels, aligned to `bars`. Each column is the level that
    was KNOWN at that bar."""
    idx = bars.index
    o, h, l, c = bars['open'], bars['high'], bars['low'], bars['close']
    out = pd.DataFrame(index=idx)

    # yearly open — first open of the calendar year, known from 1 Jan
    yr = idx.year
    out['year_open'] = o.groupby(yr).transform('first')

    # previous WEEK high/low/open — completed weeks only
    wk = idx.to_period('W')
    wh = h.groupby(wk).max()
    wl = l.groupby(wk).min()
    wo = o.groupby(wk).first()
    out['prev_wk_high'] = wk.map(wh.shift(1)).astype(float)
    out['prev_wk_low'] = wk.map(wl.shift(1)).astype(float)
    out['this_wk_open'] = wk.map(wo).astype(float)      # Monday open

    # Monday's LOW — known once Monday closes, applies to the rest of the week
    mon = bars[idx.dayofweek == 0]
    if len(mon):
        ml = mon['low'].groupby(mon.index.to_period('W')).min()
        out['monday_low'] = wk.map(ml).astype(float)
    else:
        out['monday_low'] = np.nan

    # previous MONTH high/low
    mo = idx.to_period('M')
    mh = h.groupby(mo).max()
    mlw = l.groupby(mo).min()
    out['prev_mo_high'] = mo.map(mh.shift(1)).astype(float)
    out['prev_mo_low'] = mo.map(mlw.shift(1)).astype(float)

    # previous DAY high/low
    dy = idx.to_period('D')
    dh = h.groupby(dy).max()
    dl = l.groupby(dy).min()
    out['prev_day_high'] = dy.map(dh.shift(1)).astype(float)
    out['prev_day_low'] = dy.map(dl.shift(1)).astype(float)
    return out


def atr(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def signals(bars, lv, level_name, mode):
    """mode: break_up | retest | reject. Returns bar indices that fire."""
    c = bars['close'].to_numpy(float)
    l = bars['low'].to_numpy(float)
    h = bars['high'].to_numpy(float)
    L = lv[level_name].to_numpy(float)
    n = len(c)
    fires = []
    broke = None
    for i in range(1, n):
        if not (np.isfinite(L[i]) and np.isfinite(L[i-1])):
            continue
        if mode == 'break_up':
            if c[i] > L[i] and c[i-1] <= L[i-1]:
                fires.append(i)
        elif mode == 'retest':
            if c[i] > L[i] and c[i-1] <= L[i-1]:
                broke = i
            elif (broke is not None and 0 < i - broke <= 20
                  and l[i] <= L[i] and c[i] > L[i]):
                fires.append(i)
                broke = None
        elif mode == 'reject':
            if h[i] >= L[i] and c[i] < L[i] and c[i-1] < L[i-1]:
                fires.append(i)
    return fires

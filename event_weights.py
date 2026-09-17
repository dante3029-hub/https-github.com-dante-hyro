"""Convert the four EVENT sleeves to daily weight matrices.

The portfolio simulator nets positions across sleeves before costing, but so
far it only covered the 4 cross-sectional sleeves. S/R, FVG, BOS and breakout
are event-driven -- discrete trades with entries and exits -- so they need
expressing as a coin x day weight matrix before they can be netted.

Two things this should reveal that the 4-sleeve version could not:
  * event sleeves can all want the SAME coin at once, where cross-sectional
    sleeves spread across a ranking. Netting should matter more.
  * the per-coin cap never bound with 4 sleeves. It may here.

Sizing: each sleeve runs max 6 concurrent at 1/6 weight, matching the live
compliance layer.
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd

TAKER = '/tmp/hyro/taker_data'
OIDIR = '/tmp/hyro/oi_data'
FEE = 0.00085
MAX_CONCURRENT = 6


def load_tf(sym, tf):
    df = pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('ts').sort_index()
    r = '1D' if tf == 24 else f'{tf}h'
    return pd.DataFrame({
        'open': df['open'].resample(r).first(),
        'high': df['high'].resample(r).max(),
        'low': df['low'].resample(r).min(),
        'close': df['close'].resample(r).last(),
        'volume': df['volume'].resample(r).sum(),
        'delta': df['delta'].resample(r).sum(),
    }).dropna()


def trades_to_weights(trades, daily_index, coins):
    """trades: (coin, entry_ts, exit_ts, side). Positions held between the two,
    capped at MAX_CONCURRENT per day and sized 1/MAX_CONCURRENT."""
    W = pd.DataFrame(0.0, index=daily_index, columns=coins)
    per = 1.0 / MAX_CONCURRENT
    for coin, t0, t1, side in trades:
        if coin not in W.columns:
            continue
        lo = daily_index.searchsorted(t0)
        hi = daily_index.searchsorted(t1)
        if lo >= len(daily_index):
            continue
        hi = min(max(hi, lo + 1), len(daily_index))
        W.iloc[lo:hi, W.columns.get_loc(coin)] += side * per
    # enforce max concurrent: scale any day where the sleeve is over-committed
    gross = W.abs().sum(axis=1)
    over = gross > 1.0
    if over.any():
        W.loc[over] = W.loc[over].div(gross[over], axis=0)
    return W


def atr(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def load_oi(sym, tf):
    p = f'{OIDIR}/{sym}_oi_1h.csv'
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d['ts'] = pd.to_datetime(d['timestamp'], unit='ms')
    s = d.set_index('ts').sort_index()['open_interest']
    r = '1D' if tf == 24 else f'{tf}h'
    od = s.resample(r).last()
    return (od - od.shift(1)) > 0


# ── the four event sleeves, each returning (coin, entry_ts, exit_ts, side) ──

def sr_trades(syms, tf=6, stop_atr=3.0, hold=15):
    import sr2
    out = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = sr2.load(s, tf)
        except Exception:
            continue
        if len(d) < 400:
            continue
        oi = load_oi(s, tf)
        if oi is None:
            continue
        L, S = sr2.signals(d, 'break_res')
        A = sr2.atr(d, 14).values
        lo, hi, c = d['low'].values, d['high'].values, d['close'].values
        n = len(d)
        for i in range(n - hold - 2):
            if not L[i]:
                continue
            eb = i + 1
            if eb >= n or not np.isfinite(A[i]) or A[i] <= 0:
                continue
            v = oi.reindex([d.index[eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
            stp = c[eb] - stop_atr * A[i]
            ex = min(eb + hold, n - 1)
            for j in range(eb + 1, min(eb + 1 + hold, n)):
                if lo[j] <= stp:
                    ex = j
                    break
            out.append((s, d.index[eb], d.index[ex], 1))
    return out


def fvg_trades(syms, tf=12, stop_atr=1.0, hold=10):
    import fvg as FV
    out = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = FV.load(s, tf)
        except Exception:
            continue
        if len(d) < 400:
            continue
        oi = load_oi(s, tf)
        if oi is None:
            continue
        A = FV.atr(d).values
        lo, hi, c = d['low'].values, d['high'].values, d['close'].values
        n = len(d)
        for (i, is_bull, gmax, gmin) in FV.fvgs(d, 0.0, False):
            eb = i + 1
            if eb >= n - hold or not np.isfinite(A[eb]) or A[eb] <= 0:
                continue
            v = oi.reindex([d.index[eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
            side = 1 if is_bull else -1
            stp = c[eb] - side * stop_atr * A[eb]
            ex = min(eb + hold, n - 1)
            for j in range(eb + 1, min(eb + 1 + hold, n)):
                if (side > 0 and lo[j] <= stp) or (side < 0 and hi[j] >= stp):
                    ex = j
                    break
            out.append((s, d.index[eb], d.index[ex], side))
    return out


def bos_trades(syms, tf=8, hold=30):
    import bos_dir as BD
    out = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        d = BD.load4h(s, tf)
        if d is None:
            continue
        o, h, l, c = (d['open'].values, d['high'].values,
                      d['low'].values, d['close'].values)
        dn = d['delta'].values
        A = BD.atr(h, l, c)
        ph, pl = BD.pivots(h, l, BD.PIVOT_K)
        lastH = lastL = prevH = prevL = np.nan
        state = 0
        i = BD.PIVOT_K + 1
        while i < len(c) - 1:
            if np.isfinite(ph[i]):
                prevH, lastH = lastH, ph[i]
            if np.isfinite(pl[i]):
                prevL, lastL = lastL, pl[i]
            if all(np.isfinite(x) for x in (lastH, prevH, lastL, prevL)):
                if lastH > prevH and lastL > prevL:
                    state = 1
                elif lastH < prevH and lastL < prevL:
                    state = -1
            trig = (state == -1 and np.isfinite(lastL)
                    and c[i] < lastL and c[i-1] >= lastL and dn[i] < 0)
            if trig and np.isfinite(A[i]) and A[i] > 0 and i + 1 < len(c):
                eb = i + 1
                ent = o[eb]
                stp = ent + BD.ATR_STOP * A[i]
                tp = ent - BD.TP_ATR * A[i]
                ex = min(eb + hold, len(c) - 1)
                for j in range(eb, min(eb + hold, len(c))):
                    if h[j] >= stp or l[j] <= tp:
                        ex = j
                        break
                out.append((s, d.index[eb], d.index[ex], -1))
                i = eb + hold
            i += 1
    return out


def breakout_trades(syms, lookback=50, stop_atr=3.0, hold=20):
    import breakout as B
    out = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = B.load(s, 24)
        except Exception:
            continue
        if len(d) < 200:
            continue
        o, h, l, c = (d['open'].values, d['high'].values,
                      d['low'].values, d['close'].values)
        A = B.atr(d).values
        n = len(d)
        i = lookback + 2
        while i < n - 2:
            if not (c[i] > np.nanmax(h[i-lookback:i])) or not np.isfinite(A[i]) or A[i] <= 0:
                i += 1
                continue
            eb = i + 1
            if eb >= n - 1:
                break
            stp = o[eb] - stop_atr * A[i]
            ex = min(eb + hold, n - 1)
            for j in range(eb + 1, min(eb + 1 + hold, n)):
                if l[j] <= stp:
                    ex = j
                    break
            out.append((s, d.index[eb], d.index[ex], 1))
            i = eb + hold
    return out

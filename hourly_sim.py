"""Hourly portfolio simulator — every sleeve at its NATIVE timeframe.

WHY THIS REPLACES THE DAILY VERSION: the daily simulator collapsed 6h, 8h and
12h sleeves onto a daily grid, so P&L was measured close-to-close on daily bars
rather than at the actual entry and exit prices. Reconciliation showed S/R
466.9% trade-level vs 286.6% daily-marked -- a 180-point gap that was my
conversion, not a real finding. Same for breakout: 2101% vs 890%.

DESIGN
  * hourly grid is the common resolution; every sleeve maps onto it
  * every sleeve emits (coin, entry_ts, exit_ts, side, weight)
  * cross-sectional sleeves hold a weight vector between rebalances
  * event sleeves hold a position from its actual entry bar to its actual exit
  * positions netted per coin per hour, ONE cost on the net change
  * marked to market hourly, so the equity path is real

Entry and exit timestamps come from each sleeve's own bar grid, so a 6h trade
entering at 06:00 and exiting at 18:00 is held for exactly 12 hours.
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd

TAKER = '/tmp/hyro/taker_data'
OIDIR = '/tmp/hyro/oi_data'
FEE = 0.00085
CORE24 = ['1000PEPE','1000RATS','1000SHIB','AAVE','ADA','AVAX','BCH','BNB','DOGE','DOT',
          'ETH','FIL','LDO','LINK','LTC','NEAR','SOL','SUI','TRX','UNI','WLD','XLM','XRP','ZEC']


def hourly_panel(syms):
    """Hourly close matrix — the grid everything maps onto."""
    P = {}
    for s in syms:
        p = f'{TAKER}/{s}_1h.csv'
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('ts').sort_index()
        if len(df) >= 5000:
            P[s] = df['close']
    idx = None
    for v in P.values():
        idx = v.index if idx is None else idx.union(v.index)
    idx = idx.sort_values()
    PX = pd.DataFrame({c: P[c].reindex(idx).ffill() for c in sorted(P)})
    return idx, PX, PX.pct_change().fillna(0.0)


def bars(sym, tf):
    """OHLCV at a sleeve's native timeframe, from the same hourly source."""
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


def oi_up(sym, tf):
    p = f'{OIDIR}/{sym}_oi_1h.csv'
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d['ts'] = pd.to_datetime(d['timestamp'], unit='ms')
    s = d.set_index('ts').sort_index()['open_interest']
    r = '1D' if tf == 24 else f'{tf}h'
    od = s.resample(r).last()
    return (od - od.shift(1)) > 0


def positions_to_hourly(trades, hidx, coins, per_position=1.0, max_gross=None):
    """trades: (coin, entry_ts, exit_ts, side). Held on the hourly grid between
    the two timestamps, at each sleeve's own position size.

    max_gross: scale the sleeve down on any hour where it exceeds this gross.
    WITHOUT it, per_position=1/6 does NOT cap concurrency -- FVG ran at a mean
    of 12.5 simultaneous positions, i.e. 2.1x levered. Capping it raised its
    Sharpe 3.29 -> 3.49 and cut max drawdown 91% -> 28%.
    """
    W = pd.DataFrame(0.0, index=hidx, columns=coins)
    arr = {c: np.zeros(len(hidx)) for c in coins}
    for coin, t0, t1, side in trades:
        if coin not in arr:
            continue
        lo = hidx.searchsorted(t0, side='left')
        hi = hidx.searchsorted(t1, side='left')
        if lo >= len(hidx):
            continue
        hi = min(max(hi, lo + 1), len(hidx))
        arr[coin][lo:hi] += side * per_position
    for c in coins:
        W[c] = arr[c]
    if max_gross is not None:
        gr = W.abs().sum(axis=1)
        sc = (max_gross / gr.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
        W = W.mul(sc, axis=0)
    return W


def atr(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def simulate(sleeve_weights, RH, coin_cap=0.15, gross_cap=1.0, fee=FEE,
             net=True, scale=None):
    """sleeve_weights: name -> hourly weight DataFrame. Returns hourly P&L."""
    names = list(sleeve_weights)
    if not net:
        pnl = None
        for k, w in sleeve_weights.items():
            ww = w * (scale.get(k, 1.0) if scale else 1.0) / len(names)
            g = (ww.shift(1) * RH).sum(axis=1)
            c = ww.diff().abs().sum(axis=1) * fee
            r = (g - c).fillna(0.0)
            pnl = r if pnl is None else pnl + r
        return pnl
    tot = None
    for k, w in sleeve_weights.items():
        ww = w * (scale.get(k, 1.0) if scale else 1.0)
        tot = ww.copy() if tot is None else tot.add(ww, fill_value=0.0)
    tot = tot / len(names)
    tot = tot.clip(-coin_cap, coin_cap)
    gr = tot.abs().sum(axis=1)
    sc = (gross_cap / gr.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    tot = tot.mul(sc, axis=0)
    g = (tot.shift(1) * RH).sum(axis=1)
    c = tot.diff().abs().sum(axis=1) * fee
    return (g - c).fillna(0.0), tot


def stats_hourly(x):
    """Annualise from hourly: 24*365 periods."""
    x = x.dropna()
    if len(x) < 500 or x.std() == 0:
        return 0.0, 0.0, 0.0
    eq = x.cumsum()
    return (float(x.mean()/x.std()*np.sqrt(24*365)),
            float(x.mean()*24*365*100),
            float((eq.cummax()-eq).max()*100))


# ─────────────── the eight sleeves, each emitting hourly weights ───────────────

def xs_hourly(syms, hidx, coins, sig_fn, n, hold_days, tf_hours=24,
              lookback_needed=60):
    """Cross-sectional sleeve. Rebalances every `hold_days` on its own daily
    grid, then the weight vector is held on the hourly grid until the next
    rebalance. Phase-averaged over all `hold_days` start offsets."""
    d = {}
    for s in syms:
        p = f'{TAKER}/{s}_1h.csv'
        if os.path.exists(p):
            d[s] = bars(s, 24)
    didx = None
    for v in d.values():
        didx = v.index if didx is None else didx.union(v.index)
    cs = [c for c in sorted(d) if c in coins]
    PXd = pd.DataFrame({c: d[c]['close'].reindex(didx) for c in cs})
    VOLd = pd.DataFrame({c: d[c]['volume'].reindex(didx) for c in cs})
    DNd = pd.DataFrame({c: d[c]['delta'].reindex(didx) for c in cs})
    Rd = PXd.pct_change()
    acc = pd.DataFrame(0.0, index=didx, columns=cs)
    for ph in range(hold_days):
        cur = pd.Series(0.0, index=cs)
        for t in range(lookback_needed, len(didx)):
            if (t - ph) % hold_days == 0:
                s = sig_fn(t, Rd, VOLd, DNd)
                v = s.dropna()
                if len(v) >= 2*n + 2:
                    o = v.sort_values()
                    cur = pd.Series(0.0, index=cs)
                    cur[o.index[-n:]] = 0.5/n
                    cur[o.index[:n]] = -0.5/n
            acc.iloc[t] += cur.values
    acc /= hold_days
    # daily weights -> hourly, held forward
    W = pd.DataFrame(0.0, index=hidx, columns=coins)
    a = acc.reindex(hidx, method='ffill').fillna(0.0)
    for c in cs:
        W[c] = a[c].values
    return W


def sr_hourly(syms, hidx, coins, tf=6, stop_atr=3.0, hold=15):
    import sr2
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, tf)
        except Exception:
            continue
        if len(d) < 400:
            continue
        ou = oi_up(s, tf)
        if ou is None:
            continue
        sr2.SYMS = [s]
        L, S = sr2.signals(d, 'break_res')
        A = atr(d).values
        o, lo, c = d['open'].values, d['low'].values, d['close'].values
        n = len(d)
        for i in range(n - hold - 2):
            if not L[i]:
                continue
            eb = i + 1
            if eb >= n or not np.isfinite(A[i]) or A[i] <= 0:
                continue
            v = ou.reindex([d.index[eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
            # entry is at o[eb]; c[eb] is not known until that bar closes.
            stp = o[eb] - stop_atr*A[i]
            ex = min(eb + hold, n - 1)
            for j in range(eb, min(eb + 1 + hold, n)):
                if lo[j] <= stp:
                    ex = j
                    break
            tr.append((s, d.index[eb], d.index[ex], 1))
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6, max_gross=1.0), len(tr)


def fvg_hourly(syms, hidx, coins, tf=12, stop_atr=1.0, hold=10):
    import fvg as FV
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, tf)
        except Exception:
            continue
        if len(d) < 400:
            continue
        ou = oi_up(s, tf)
        if ou is None:
            continue
        A = atr(d).values
        o, lo, hi, c = (d['open'].values, d['low'].values,
                        d['high'].values, d['close'].values)
        n = len(d)
        for (i, is_bull, gmax, gmin) in FV.fvgs(d, 0.0, False):
            eb = i + 1
            if eb >= n - hold or not np.isfinite(A[eb]) or A[eb] <= 0:
                continue
            v = ou.reindex([d.index[eb]], method='ffill')
            if not (len(v) and bool(v.iloc[0])):
                continue
            side = 1 if is_bull else -1
            # was c[eb] and A[eb] -- both include the entry bar, which is
            # lookahead when entering at its OPEN.
            stp = o[eb] - side*stop_atr*A[i]
            ex = min(eb + hold, n - 1)
            for j in range(eb, min(eb + 1 + hold, n)):
                if (side > 0 and lo[j] <= stp) or (side < 0 and hi[j] >= stp):
                    ex = j
                    break
            tr.append((s, d.index[eb], d.index[ex], side))
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6, max_gross=1.0), len(tr)


def bos_hourly(syms, hidx, coins, tf=8, hold=30):
    import bos_dir as BD
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, tf)
        except Exception:
            continue
        if len(d) < 400:
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
            if (state == -1 and np.isfinite(lastL) and c[i] < lastL
                    and c[i-1] >= lastL and dn[i] < 0
                    and np.isfinite(A[i]) and A[i] > 0 and i + 1 < len(c)):
                eb = i + 1
                ent = o[eb]
                stp = ent + BD.ATR_STOP*A[i]
                tp = ent - BD.TP_ATR*A[i]
                ex = min(eb + hold, len(c) - 1)
                for j in range(eb, min(eb + hold, len(c))):
                    if h[j] >= stp or l[j] <= tp:
                        ex = j
                        break
                tr.append((s, d.index[eb], d.index[ex], -1))
                i = eb + hold
            i += 1
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6, max_gross=1.0), len(tr)


def breakout_hourly(syms, hidx, coins, lookback=50, stop_atr=3.0, hold=20):
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, 24)
        except Exception:
            continue
        if len(d) < 200:
            continue
        o, h, l, c = (d['open'].values, d['high'].values,
                      d['low'].values, d['close'].values)
        A = atr(d).values
        n = len(d)
        i = lookback + 2
        while i < n - 2:
            if (not c[i] > np.nanmax(h[i-lookback:i])) or not np.isfinite(A[i]) or A[i] <= 0:
                i += 1
                continue
            eb = i + 1
            if eb >= n - 1:
                break
            stp = o[eb] - stop_atr*A[i]
            ex = min(eb + hold, n - 1)
            for j in range(eb, min(eb + 1 + hold, n)):
                if l[j] <= stp:
                    ex = j
                    break
            tr.append((s, d.index[eb], d.index[ex], 1))
            i = eb + hold
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6, max_gross=1.0), len(tr)


def cascade_hourly(syms, hidx, coins, z_thr=-2.5, hold=3, n_long=5):
    """Cascade as an EVENT sleeve with explicit entry/exit timestamps.

    Previously built inline by writing a weight per daily bar and forward-filling
    onto the hourly grid. That is fragile -- if the reset logic misfires the
    ffill holds a position indefinitely. Emitting (coin, entry, exit) trades and
    reusing positions_to_hourly makes the hold length explicit and testable.
    """
    d = {}
    for s in syms:
        if os.path.exists(f'{TAKER}/{s}_1h.csv'):
            b = bars(s, 24)
            if len(b) >= 200:
                d[s] = b
    didx = None
    for v in d.values():
        didx = v.index if didx is None else didx.union(v.index)
    cs = [c for c in sorted(d) if c in coins]
    PXd = pd.DataFrame({c: d[c]['close'].reindex(didx) for c in cs})
    Rd = PXd.pct_change()
    mkt = Rd.mean(axis=1)
    tr = []
    t = 60
    while t < len(didx) - hold - 1:
        v = mkt.iloc[t-20:t].std()
        if v > 0 and mkt.iloc[t]/v <= z_thr:
            worst = Rd.iloc[t].dropna().sort_values().index[:n_long]
            ex = min(t + hold, len(didx) - 1)
            for c in worst:
                tr.append((c, didx[t+1], didx[ex], 1))
            t += hold          # no overlapping cascades
        t += 1
    return positions_to_hourly(tr, hidx, coins,
                               per_position=1.0/max(n_long, 1),
                               max_gross=1.0), len(tr)


def pattern_hourly(syms, hidx, coins, tf=6, stop_atr=2.0, hold=15,
                   bull_only=True, use_oi=True):
    """MarkitTick 16-pattern sleeve on its native 6h grid.

    Tested standalone at 1.03 (6h, bullish only, +OI filter) but never made it
    into the book. Entry at the open of the bar AFTER detection -- NOT the
    historical break bar, which is up to 20 bars in the past and would be
    lookahead (that error inflated an early backtest from 1.04 to 6.06).
    """
    from markittick_detector import detect
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, tf)
        except Exception:
            continue
        if len(d) < 700:
            continue
        ou = oi_up(s, tf) if use_oi else None
        if use_oi and ou is None:
            continue
        A = atr(d).values
        o, h, l, c = (d['open'].values, d['high'].values,
                      d['low'].values, d['close'].values)
        n = len(d)
        for rec in detect(d):
            b, ts, is_bull, name, entry, stop, target = rec[:7]
            det_bar = rec[7] if len(rec) > 7 else b
            if bull_only and not is_bull:
                continue
            eb = det_bar + 1
            if eb >= n - 1 or not np.isfinite(A[eb]) or A[eb] <= 0:
                continue
            if use_oi:
                v = ou.reindex([d.index[eb]], method='ffill')
                if not (len(v) and bool(v.iloc[0])):
                    continue
            side = 1 if is_bull else -1
            stp = o[eb] - side*stop_atr*A[eb - 1]   # A[eb] includes the entry bar
            ex = min(eb + hold, n - 1)
            for j in range(eb, min(eb + 1 + hold, n)):
                if (side > 0 and l[j] <= stp) or (side < 0 and h[j] >= stp):
                    ex = j
                    break
            tr.append((s, d.index[eb], d.index[ex], side))
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6, max_gross=1.0), len(tr)


def srflip_hourly(syms, hidx, coins, tf=6, stop_atr=3.0, hold=15, window=20):
    """S/R FLIP: after price breaks resistance, buy when it returns to that
    level and holds. The level flips from resistance to support -- `res_is_sup`
    in the ChartPrime source, which I had never tested.

    Different trigger from sr_hourly (which buys the BREAK). This buys the
    RETEST, so it enters on weakness rather than strength.
    """
    import sr2
    tr = []
    for s in syms:
        if not os.path.exists(f'{TAKER}/{s}_1h.csv'):
            continue
        try:
            d = bars(s, tf)
        except Exception:
            continue
        if len(d) < 400:
            continue
        sup, sup1, res, res1 = sr2.levels(d)
        A = atr(d).values
        o, h, l, c = (d['open'].values, d['high'].values,
                      d['low'].values, d['close'].values)
        n = len(d)
        broke = None
        lvl = np.nan
        for i in range(1, n - hold - 2):
            # source: brekout_res := ta.crossover(LOW, resistanceLevel_1).
            # Was c[i] (close) -- the deviation fixed in sleeve_sr.py but
            # left here. The whole candle must clear the box.
            if np.isfinite(res1[i]) and l[i] > res1[i] and l[i-1] <= res1[i-1]:
                broke, lvl = i, res[i]
            if (broke is not None and i > broke and i - broke <= window
                    and np.isfinite(lvl) and l[i] <= lvl and c[i] > lvl
                    and np.isfinite(A[i]) and A[i] > 0):
                eb = i + 1
                if eb >= n - 1:
                    break
                stp = o[eb] - stop_atr*A[i]
                ex = min(eb + hold, n - 1)
                for j in range(eb, min(eb + 1 + hold, n)):
                    if l[j] <= stp:
                        ex = j
                        break
                tr.append((s, d.index[eb], d.index[ex], 1))
                broke = None
    return positions_to_hourly(tr, hidx, coins, per_position=1.0/6,
                               max_gross=1.0), len(tr)

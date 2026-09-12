"""Portfolio simulator -- nets positions ACROSS sleeves before costing.

WHY THIS DIFFERS FROM THE BLEND:
The blend sums independent daily return streams. That overstates costs (each
sleeve pays its own fees even when two sleeves want opposite sides of the same
coin) and ignores caps (two sleeves at 15% on one coin is 30% real exposure,
which the live compliance layer would reject).

This version:
  1. every sleeve emits a WEIGHT per coin per day
  2. weights are summed per coin -> net desired position
  3. per-coin cap applied to the NET
  4. gross scaled to a target if it exceeds the leverage limit
  5. ONE set of costs on the NET change, not per sleeve

Cost model unchanged: 0.055% fee + 0.03% slippage = 0.00085 per unit traded.
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'; OIDIR='/tmp/hyro/oi_data'
FEE = 0.00085
COIN_CAP = 0.15          # max net weight per coin
GROSS_CAP = 1.00         # max total gross exposure
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})


def panel():
    F = {}
    for s in SYMS:
        df = pd.read_csv(f'{TAKER}/{s}_1h.csv')
        df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('ts').sort_index()
        d = pd.DataFrame({'close': df['close'].resample('1D').last(),
                          'volume': df['volume'].resample('1D').sum(),
                          'delta': df['delta'].resample('1D').sum()}).dropna()
        if len(d) >= 400:
            F[s] = d
    idx = None
    for v in F.values():
        idx = v.index if idx is None else idx.union(v.index)
    cs = sorted(F)
    PX = pd.DataFrame({c: F[c]['close'].reindex(idx) for c in cs})
    VOL = pd.DataFrame({c: F[c]['volume'].reindex(idx) for c in cs})
    DN = pd.DataFrame({c: F[c]['delta'].reindex(idx) for c in cs})
    return idx, PX, PX.pct_change(), VOL, DN


def xs_weights(R, sig, n=5, hold=14, phase=0):
    """Cross-sectional sleeve -> daily weight matrix."""
    W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    cur = pd.Series(0.0, index=R.columns)
    for t in range(60, len(R.index)):
        if (t - phase) % hold == 0:
            s = sig(t); v = s.dropna()
            if len(v) >= 2*n + 2:
                o = v.sort_values()
                cur = pd.Series(0.0, index=R.columns)
                cur[o.index[-n:]] = 0.5/n
                cur[o.index[:n]] = -0.5/n
        W.iloc[t] = cur.values
    return W


def event_weights(R, trades, hold):
    """Event sleeve -> weight matrix. trades: list of (coin, entry_date, side)."""
    W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    per = 1.0/6                       # max 6 concurrent, equal size
    for coin, d0, side in trades:
        if coin not in W.columns:
            continue
        try:
            i = W.index.get_loc(d0)
        except KeyError:
            continue
        j = min(i + hold, len(W.index))
        W.iloc[i:j, W.columns.get_loc(coin)] += side*per
    return W


def simulate(sleeves, R, coin_cap=COIN_CAP, gross_cap=GROSS_CAP, fee=FEE,
             net=True, risk_parity=False, vol_lb=60):
    """risk_parity: scale each sleeve by its TRAILING realised vol before
    summing. The earlier blend did this implicitly (dividing by each sleeve's
    full-sample std), which is worth ~1.1 Sharpe -- but that used the WHOLE
    sample's vol, which is not knowable live. Here it is estimated from a
    trailing window only."""
    """sleeves: dict name -> weight DataFrame. Returns daily P&L series."""
    idx = R.index
    scales = {}
    if risk_parity:
        for k, W in sleeves.items():
            w = W.reindex(index=idx, columns=R.columns).fillna(0.0)
            r = (w.shift(1)*R).sum(axis=1)
            v = r.rolling(vol_lb).std().shift(1)          # trailing only
            scales[k] = (1.0/v.replace(0, np.nan)).clip(upper=v.median()*0 + 1e9)
    tot = None
    for k, W in sleeves.items():
        w = W.reindex(index=idx, columns=R.columns).fillna(0.0)
        if risk_parity:
            sc = scales[k]
            sc = (sc/sc.mean()).fillna(1.0).clip(0.2, 5.0)
            w = w.mul(sc, axis=0)
        tot = w.copy() if tot is None else tot.add(w, fill_value=0.0)
    tot = tot / len(sleeves)
    if not net:
        # cost each sleeve separately -- what the blend implicitly does
        pnl = np.zeros(len(idx)); prev = {k: None for k in sleeves}
        for k, W in sleeves.items():
            w = W.reindex(index=idx, columns=R.columns).fillna(0.0)/len(sleeves)
            g = (w.shift(1)*R).sum(axis=1)
            c = w.diff().abs().sum(axis=1)*fee
            pnl += (g - c).fillna(0.0).values
        return pd.Series(pnl, index=idx)
    tot = tot.clip(-coin_cap, coin_cap)         # per-coin cap on the NET
    gross = tot.abs().sum(axis=1)
    scale = (gross_cap/gross.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    tot = tot.mul(scale, axis=0)
    gross_pnl = (tot.shift(1)*R).sum(axis=1)
    cost = tot.diff().abs().sum(axis=1)*fee     # ONE cost on the net change
    return (gross_pnl - cost).fillna(0.0)


def stats(x):
    x = x.dropna()
    if len(x) < 50 or x.std() == 0:
        return 0.0, 0.0, 0.0
    eq = x.cumsum()
    return (float(x.mean()/x.std()*np.sqrt(365)),
            float(x.mean()*365*100),
            float((eq.cummax()-eq).max()*100))

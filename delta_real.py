"""Delta sleeve on MATCHED-VENUE taker data.

The previous version ranked on clean_panel's taker column, where taker volume
exceeded total volume on 60% of bars -- two venues spliced together. Whatever
that measured, it was not order-flow imbalance. This uses Binance futures
klines where volume and taker_buy come from the same response.
"""
import os, glob
import numpy as np, pandas as pd
TAKER='/tmp/hyro/taker_data'
FEE, CAP = 0.00085, 0.15
SYMS = sorted({os.path.basename(f).replace('_1h.csv','') for f in glob.glob(f'{TAKER}/*_1h.csv')})

def load(sym):
    df = pd.read_csv(f'{TAKER}/{sym}_1h.csv')
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('ts').sort_index()
    return pd.DataFrame({
        'close': df['close'].resample('1D').last(),
        'volume': df['volume'].resample('1D').sum(),
        'delta': df['delta'].resample('1D').sum(),
    }).dropna()

def panel():
    fr = {}
    for s in SYMS:
        try: d = load(s)
        except Exception: continue
        if len(d) >= 400: fr[s] = d
    idx = None
    for d in fr.values(): idx = d.index if idx is None else idx.union(d.index)
    cs = sorted(fr)
    PX = pd.DataFrame({c: fr[c]['close'].reindex(idx) for c in cs})
    VOL = pd.DataFrame({c: fr[c]['volume'].reindex(idx) for c in cs})
    DN = pd.DataFrame({c: fr[c]['delta'].reindex(idx) for c in cs})
    return idx, PX, PX.pct_change(), VOL, DN

def xs(R, sig, n=5, hold=7, fee=FEE, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    idx = R.index; acc = np.zeros(len(idx))
    for ph in range(hold):
        wp = pd.Series(0.0, index=R.columns); out = np.zeros(len(idx))
        for t in range(60, len(idx)):
            r = R.iloc[t].fillna(0.0); carry = float((wp*r).sum())
            if (t-ph) % hold != 0: out[t] = carry; continue
            s = sig(t); v = s.dropna()
            if len(v) < 2*n+2: out[t] = carry; continue
            o = v.sort_values(); w = pd.Series(0.0, index=R.columns)
            if shuffle:
                pick = list(o.index); rng.shuffle(pick)
                w[pick[-n:]] = 0.5/n; w[pick[:n]] = -0.5/n
            else:
                w[o.index[-n:]] = 0.5/n; w[o.index[:n]] = -0.5/n
            out[t] = float((w*r).sum()) - float((w-wp).abs().sum())*fee
            wp = w
        acc += out
    return pd.Series(acc[60:]/hold, index=idx[60:])

def stats(x):
    x = x.dropna()
    if len(x) < 50 or x.std() == 0: return 0.0, 0.0
    return float(x.mean()/x.std()*np.sqrt(365)), float(x.mean()*365*100)

if __name__ == '__main__':
    idx, PX, R, VOL, DN = panel()
    print(f'{len(R.columns)} coins, {len(idx)} days\n')
    print(f"{'  construction':<34}{'Sharpe':>9}{'ctrl':>8}{'GAP':>8}{'1st':>8}{'2nd':>8}{'both':>7}")
    def rep(lbl, sig, **kw):
        r = xs(R, sig, **kw); c = xs(R, sig, shuffle=True, seed=5, **kw)
        sh = stats(r)[0]; cs = stats(c)[0]
        h = len(r)//2; s1, s2 = stats(r.iloc[:h])[0], stats(r.iloc[h:])[0]
        print(f'  {lbl:<32}{sh:>9.2f}{cs:>8.2f}{sh-cs:>8.2f}{s1:>8.2f}{s2:>8.2f}'
              f'{("yes" if s1>0 and s2>0 else "NO"):>7}')
    def z(x): return (x-x.mean())/x.std() if x.std()>0 else x*0
    for lb in (1,3,5):
        rep(f'raw delta, {lb}d lookback', lambda t,k=lb: z(DN.iloc[t-k:t].sum()))
    for lb in (1,3,5):
        rep(f'delta/volume, {lb}d lookback',
            lambda t,k=lb: z(DN.iloc[t-k:t].sum()/VOL.iloc[t-k:t].sum().replace(0,np.nan)))
    rep('delta/volume, 3d, hold 5', lambda t: z(DN.iloc[t-3:t].sum()/VOL.iloc[t-3:t].sum().replace(0,np.nan)), hold=5)
    rep('delta/volume, 3d, N=8', lambda t: z(DN.iloc[t-3:t].sum()/VOL.iloc[t-3:t].sum().replace(0,np.nan)), n=8)
    print('\n  previous version (corrupt column): 0.51, halves -0.15/1.24, NO')

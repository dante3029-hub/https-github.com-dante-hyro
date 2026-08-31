"""The path to a faster pass is NOT a better signal - it is a tighter risk
envelope, which lets you run more size at the same bust probability.

Adds to the stable baseline (always long top-K by atr_pct):
  1. dynamic vol targeting  - scale exposure by target / trailing realised vol
  2. drawdown throttle      - cut exposure as equity falls from HWM
Both are mechanical and require NO regime forecast.
"""
import pandas as pd, numpy as np
import regime_fast as B   # reuse precomputed SEL / weights

TAKER, SLIP = B.TAKER, B.SLIP

def run(K=5, rebal_stride=1, base_vol=0.20, vt_win=8, vt_cap=2.0, use_vt=True,
        thr=None, start=None, end=None, label=""):
    """thr: list of (dd_threshold, exposure_mult) sorted ascending, e.g.
       [(0.04,1.0),(0.07,0.5),(1.0,0.0)]"""
    ts_all = [t for t in B.TS
              if (start is None or t >= start) and (end is None or t < end)]
    ts_all = ts_all[::rebal_stride]
    recs, prev, hist, eq, hwm = [], {}, [], 1.0, 1.0
    for i, ts in enumerate(ts_all):
        s = B.SEL[ts]
        # ---- exposure multiplier ----
        m = 1.0
        if use_vt and len(hist) >= vt_win:
            rv = np.std(hist[-vt_win:]) * np.sqrt(8760/(72*rebal_stride))
            if rv > 1e-6: m = min(base_vol/rv, vt_cap)
        if thr is not None:
            dd = (hwm - eq)/hwm
            for lim, mult in thr:
                if dd <= lim: m *= mult; break
        sides = {c: 1 for c in s['hi'][-K:]} if K <= len(s['hi']) else {c: 1 for c in s['hi']}
        w = B.weights(sides, s['v'])
        if w is None: continue
        w = {c: x*m for c, x in w.items()}
        g = sum(w[c]*s['r'].get(c, np.nan) for c in w)
        if not np.isfinite(g): continue
        fn = sum(-np.sign(w[c])*abs(w[c])*(s['f'].get(c, 0.0) or 0.0) for c in w)
        turn = sum(abs(w.get(c,0)-prev.get(c,0)) for c in set(w)|set(prev))
        cost = turn*(TAKER+SLIP)
        net = g+fn-cost
        recs.append(dict(ts=s['t1'], gross=g, fund=fn, cost=cost, net=net,
                         turn=turn, mult=m))
        hist.append(net); prev = w
        eq *= (1+net); hwm = max(hwm, eq)
    if len(recs) < 25: return None
    R = pd.DataFrame(recs).set_index('ts')
    py = 8760/(72*rebal_stride)
    e = (1+R.net).cumprod(); yrs = (R.index[-1]-R.index[0]).days/365.25
    dd = (e.cummax()-e)/e.cummax(); cagr = e.iloc[-1]**(1/yrs)-1
    return dict(label=label, n=len(R), yrs=round(yrs,2), CAGR=cagr,
                Sharpe=R.net.mean()/R.net.std()*np.sqrt(py), MaxDD=dd.max(),
                Calmar=cagr/dd.max() if dd.max()>0 else np.nan,
                hit=(R.net>0).mean(), skew=R.net.skew(), mult=R['mult'].mean(),
                eq=e, R=R)

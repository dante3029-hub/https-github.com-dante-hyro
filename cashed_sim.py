#!/usr/bin/env python3
"""cashed_sim.py -- S/R break + OI rank on Cashed Instant rules.
Sized on the FIRST half only, judged on the SECOND half (out-of-sample sizing).
Rules: 3% daily loss (vs start-of-day equity), 6% static floor, payout at +1%,
then reset. Rolling start every day of the second half, max 365 days."""
import numpy as np, pandas as pd
sr = pd.read_csv("ps_oos_fixed_daily.csv", index_col=0, parse_dates=True)["sr_6h"]
oi = pd.read_csv("oos_book_daily_plateau.csv", index_col=0, parse_dates=True)["oirank_h3"]
df = pd.concat([sr, oi], axis=1).fillna(0.0)
df = df[df.index >= df.index[60]]                         # warm-up
mid = df.index[len(df) // 2]
A, B = df[df.index < mid], df[df.index >= mid]
w = 0.5 / A.std()                                          # equal risk, first-half vol only
unitA, unitB = (A * w).sum(1), (B * w).sum(1)
print("corr first half", round(A.corr().iloc[0,1],2), " split", mid.date())

def worst(u, s):
    v = u.values * s
    le = np.cumsum(np.log1p(v)); dd = np.expm1(le - np.maximum.accumulate(np.maximum(le,0))).min()
    return v.min(), dd
lo, hi = 0.0, 10.0
for _ in range(60):
    m = (lo+hi)/2; wd, dd = worst(unitA, m)
    lo, hi = (m, hi) if (wd >= -0.03 and dd >= -0.06) else (lo, m)
size = 0.8 * lo
wdA, ddA = worst(unitA, size); wdB, ddB = worst(unitB, size)
print(f"size {size:.4f}  first half: worst day {wdA*100:.2f}% maxDD {ddA*100:.2f}%  |  second half: worst day {wdB*100:.2f}% maxDD {ddB*100:.2f}%")

def cycles(u, s, target=0.01, floor=0.94, daily=0.03, maxd=365):
    v = u.values * s; out = []
    for st in range(len(v)):
        e = 1.0; res = ("open", None)
        for k in range(st, min(st+maxd, len(v))):
            sod = e; e *= 1 + v[k]
            if e < sod*(1-daily) or e < floor: res = ("bust", k-st+1); break
            if e >= 1 + target: res = ("pay", k-st+1); break
        if res[0] == "open" and st + maxd > len(v): continue   # ran out of data, uncounted
        out.append(res)
    return out
for lab, u in (("first half (in-sample)", unitA), ("SECOND half (out-of-sample)", unitB)):
    r = cycles(u, size); n = len(r)
    pay = [d for k,d in r if k=="pay"]; bust = sum(k=="bust" for k,_ in r); op = sum(k=="open" for k,_ in r)
    print(f"{lab}: starts {n}  payout {len(pay)/n*100:.1f}%  bust {bust/n*100:.1f}%  no result in a year {op/n*100:.1f}%"
          f"  median days to payout {np.median(pay) if pay else float('nan'):.0f}  (p75 {np.percentile(pay,75) if pay else float('nan'):.0f})")
print("mean daily return at this size: first", f"{unitA.mean()*size*100:.3f}%", " second", f"{unitB.mean()*size*100:.3f}%")

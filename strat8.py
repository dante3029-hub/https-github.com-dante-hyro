#!/usr/bin/env python3
"""
strat8.py — post-liquidation-cascade reversion, rebuilt from 5m bars.

RUN THIS ON THE SERVER (the 5m data is 489MB and stays there). It writes a
small daily return series to strat8_returns.csv, which is what the blend needs.

THE STRATEGY, as validated earlier: an ABSOLUTE trigger, not a multiple of a
trailing average. A 4x volume spike combined with a 5% wick marks a genuine
cascade; a "4x the trailing mean" test fires ~50 times a coin per year and is
mostly noise. The exhaustion wait IS the edge -- with it 1.51, without it 1.17.

  trigger : volume > 4x the 288-bar (24h) mean AND wick >= 5% of price
  wait    : until the cascade stops extending (up to 12 bars)
  entry   : next bar open, LONG only
  target  : 50% retrace of the flush
  stop    : below the wick low
  cap     : 144 bars (12h)
"""
from __future__ import annotations
import os, csv, glob, sys
import datetime as dt
import numpy as np

M5 = os.path.expanduser("~/bot_hyrotrader_v1/m5")
FEE = 0.00085
MAX_HOLD = 144
VOL_MULT = 4.0
WICK_PCT = 0.05
RETRACE = 0.50


def load(path):
    ts, o, h, l, c, v = [], [], [], [], [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            try:
                ts.append(int(row[1])); o.append(float(row[2]))
                h.append(float(row[3])); l.append(float(row[4]))
                c.append(float(row[5])); v.append(float(row[6]))
            except (ValueError, IndexError):
                continue
    if len(c) < 20000:
        return None
    return (np.array(ts), np.array(o), np.array(h),
            np.array(l), np.array(c), np.array(v))


def run_coin(path):
    d = load(path)
    if d is None:
        return []
    ts, o, h, l, c, v = d
    n = len(c)
    vma = np.full(n, np.nan)
    for i in range(288, n):
        vma[i] = v[i-288:i].mean()
    out = []
    i = 300
    while i < n - MAX_HOLD - 3:
        if not np.isfinite(vma[i]) or vma[i] <= 0:
            i += 1; continue
        wick = (c[i] - l[i]) / max(c[i], 1e-9)
        if not (v[i] > VOL_MULT * vma[i] and wick >= WICK_PCT):
            i += 1; continue
        # wait for the cascade to stop extending -- this is the edge
        j, guard = i, 0
        while j < n - 1 and guard < 12 and l[j+1] < l[j]:
            j += 1; guard += 1
        ei = j + 1
        if ei >= n - MAX_HOLD:
            i = j + 1; continue
        low = l[i:j+1].min()
        hi = h[max(0, i-12):i+1].max()
        ent = o[ei]
        tgt = low + RETRACE * (hi - low)
        stop = low * 0.999
        if tgt <= ent or stop >= ent:
            i = j + 1; continue
        px = None
        for k in range(ei, min(ei + MAX_HOLD, n)):
            if l[k] <= stop:
                px = stop; break
            if h[k] >= tgt:
                px = tgt; break
        if px is None:
            px = c[min(ei + MAX_HOLD, n - 1)]
        exit_ts = ts[min(ei + MAX_HOLD, n - 1)]
        out.append((exit_ts, (px - ent) / ent - 2 * FEE))
        i = ei + MAX_HOLD
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(f"{M5}/*_5m.csv"))
    if not files:
        print(f"no 5m files under {M5}"); sys.exit(1)
    trades = []
    for p in files:
        coin = os.path.basename(p).replace("_5m.csv", "")
        t = run_coin(p)
        trades += t
        print(f"  {coin:<10}{len(t):>5} trades", flush=True)
    if not trades:
        print("no trades"); sys.exit(1)
    by = {}
    for exts, r in trades:
        day = dt.datetime.fromtimestamp(exts/1000, tz=dt.timezone.utc).date()
        by.setdefault(day, []).append(r)
    lo, hi = min(by), max(by)
    rows, day = [], lo
    while day <= hi:
        vv = by.get(day, [])
        rows.append((day, sum(x * min(1/len(vv), 0.15) for x in vv) if vv else 0.0))
        day += dt.timedelta(days=1)
    with open(os.path.expanduser("~/strat8_returns.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "ret"])
        w.writerows(rows)
    a = np.array([r for _, r in rows])
    sh = a.mean()/a.std()*np.sqrt(365) if a.std() > 0 else 0
    half = len(a)//2
    s1 = a[:half].mean()/a[:half].std()*np.sqrt(365) if a[:half].std() > 0 else 0
    s2 = a[half:].mean()/a[half:].std()*np.sqrt(365) if a[half:].std() > 0 else 0
    print(f"\n  {len(trades)} trades, {len(rows)} days")
    print(f"  Sharpe {sh:.2f}   halves {s1:.2f} / {s2:.2f}")
    print(f"  written -> ~/strat8_returns.csv  (push this, not the 5m data)")

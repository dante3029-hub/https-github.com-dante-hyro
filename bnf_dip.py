#!/usr/bin/env python3
"""
bnf_dip.py -- Takashi Kotegawa (BNF) style deviation buy, on crypto perps.
Pre-registered; written before the first run, one run.

  BNF bought stocks that had fallen far below their 25-day moving average
  and sold the rebound toward it.

RULES
  bars       daily (UTC), built from 1h bars
  universe   top-30 coins by median dollar volume
  signal     daily close at least X% below its 25-day simple moving average
             X = 20% and X = 30% (two cells, both reported)
  entry      next day's open, long only, one position per coin at a time
  exit       first daily close at or above the 25-day SMA, or the close of
             day 10, whichever comes first. No stop (BNF's rule is the
             reversion itself); worst trade reported so the risk is visible
  costs      0.17% round trip (funding not modelled)

CONTROL    for every real trade, a random entry day on the same coin held for
           the SAME number of days: isolates whether buying the deep dip beats
           buying that coin at a random time
VERDICT    excess over random > 0, monthly-clustered t > 2, both halves > 0
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

ROUND_TRIP = 0.0017
SMA = 25
MAX_HOLD = 10
SIMS = 40


def daily(d1h):
    d = d1h.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                "close": "last", "volume": "sum"})
    cnt = d1h["close"].resample("1D").count()
    return d[cnt == 24].dropna()


def trades(d, thresh):
    o, c = d.open.values, d.close.values
    sma = pd.Series(c).rolling(SMA).mean().values
    out, t, n = [], SMA, len(c)
    while t < n - 1:
        if np.isfinite(sma[t]) and c[t] <= sma[t] * (1 - thresh):
            eb = t + 1
            ex = min(eb + MAX_HOLD - 1, n - 1)
            for j in range(eb, ex + 1):
                if c[j] >= sma[j]:
                    ex = j
                    break
            out.append((eb, ex, c[ex] / o[eb] - 1 - ROUND_TRIP))
            t = ex + 1
            continue
        t += 1
    return out


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    import fvg_revisit as FV
    rg = np.random.default_rng(zlib.crc32(b"bnf_dip"))
    coins = FV.top_by_liquidity(SR.universe(), EA.load_bars, 30)
    data = {}
    for coin in coins:
        d1 = EA.load_bars(coin, 1)
        if d1 is not None:
            dd = daily(d1)
            if len(dd) > SMA + 60:
                data[coin] = dd
    print(f"\n  BNF deviation buy -- daily, {len(data)} coins, pre-registered\n")
    for thresh in (0.20, 0.30):
        rows = []
        for coin, d in data.items():
            for eb, ex, r in trades(d, thresh):
                rows.append((d.index[eb], coin, eb, ex, r))
        df = pd.DataFrame(rows, columns=["ts", "coin", "eb", "ex", "ret"]).sort_values("ts")
        if len(df) < 30:
            print(f"  {int(thresh*100)}% below SMA25: only {len(df)} trades, no verdict\n")
            continue
        ctrl = []
        for _ in range(SIMS):
            acc = []
            for _, row in df.iterrows():
                d = data[row.coin]
                hold = row.ex - row.eb
                eb = int(rg.integers(SMA, len(d) - hold - 1))
                acc.append(d.close.values[eb + hold] / d.open.values[eb] - 1 - ROUND_TRIP)
            ctrl.append(np.mean(acc))
        cm = float(np.mean(ctrl))
        mon = df.groupby(df.ts.dt.to_period("M")).ret.mean()
        tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon))) if len(mon) > 2 else np.nan
        mid = df.ts.iloc[len(df) // 2]
        h1, h2 = df.ret[df.ts < mid].mean(), df.ret[df.ts >= mid].mean()
        exc = df.ret.mean() - cm
        ok = exc > 0 and tm > 2 and h1 > 0 and h2 > 0
        days = df.ts.dt.floor("D").nunique()
        print(f"  {int(thresh*100)}% below SMA25")
        print(f"    trades {len(df)} on {days} distinct days   win {(df.ret > 0).mean()*100:.0f}%"
              f"   avg hold {(df.ex - df.eb + 1).mean():.1f} days   back to SMA {((df.ex - df.eb + 1) < MAX_HOLD).mean()*100:.0f}%")
        print(f"    avg return {df.ret.mean()*100:+.2f}%   median {df.ret.median()*100:+.2f}%"
              f"   random same coin/hold {cm*100:+.2f}%   excess {exc*100:+.2f}%")
        print(f"    best {df.ret.max()*100:+.1f}%   worst {df.ret.min()*100:+.1f}%")
        print(f"    monthly t {tm:+.2f} ({int((mon > 0).sum())}/{len(mon)} months up)"
              f"   halves {h1*100:+.2f}% / {h2*100:+.2f}%")
        print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ema_stocks.py -- EMA cross longs on US large caps. Pre-registered, one run.
Runs on the server (needs Yahoo Finance).

  python3 ema_stocks.py

RULES (fixed before the first run)
  universe   same 99 S&P 100 tickers as bnf_stocks.py, daily, 2005 -> today
  entry      daily close where the 20 EMA crosses ABOVE the 50 EMA;
             buy the next day's open, long only, one position per stock
  exit       the first close where the 20 EMA is back BELOW the 50 EMA,
             sold at the next day's open
  versions   A: no stop
             B: 5% stop below entry; a day whose low touches it exits at the
                stop, or at the open if the stock gaps below it
  costs      0.10% round trip
  control    random entry, same stock, same holding length (and for B the
             same 5% stop): does the cross pick better moments than chance?
  verdict    excess over random > 0, monthly t > 2, both halves positive
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import bnf_stocks as BS

ROUND_TRIP = 0.0010
FAST, SLOW = 20, 50
STOP = 0.05
SIMS = 40


def load(tickers):
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start="2005-01-01", auto_adjust=True, progress=False)
        except Exception as exc:
            print(f"  skip {t}: {exc}")
            continue
        if d is None or d.empty:
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        d = d[(d.open > 0) & (d.close > 0)]
        if len(d) > SLOW + 250:
            out[t] = d
    return out


def hold_trade(d, eb, xb, stop):
    """Long from open[eb] to open[xb]; optional stop % below entry."""
    o, l = d.open.values, d.low.values
    ent = o[eb]
    if stop:
        sp = ent * (1 - stop)
        for j in range(eb, xb):
            if l[j] <= sp:
                fill = sp if j == eb else min(sp, o[j])
                return fill / ent - 1 - ROUND_TRIP, j - eb + 1
    return o[xb] / ent - 1 - ROUND_TRIP, xb - eb


def signals(d):
    c = d.close
    f, s = c.ewm(span=FAST, adjust=False).mean().values, c.ewm(span=SLOW, adjust=False).mean().values
    n, out, t = len(c), [], SLOW
    while t < n - 2:
        if f[t] > s[t] and f[t - 1] <= s[t - 1]:
            x = t + 1
            while x < n - 1 and f[x] > s[x]:
                x += 1
            out.append((t + 1, min(x + 1, n - 1)))      # buy next open, sell open after the cross-down
            t = x + 1
            continue
        t += 1
    return out


def main() -> int:
    rg = np.random.default_rng(zlib.crc32(b"ema_stocks"))
    data = load(BS.TICKERS)
    print(f"\n  EMA 20/50 cross longs -- US large caps, {len(data)} stocks, daily, pre-registered\n")
    for lab, stop in (("A  no stop", 0.0), ("B  5% stop", STOP)):
        rows = []
        for tk, d in data.items():
            for eb, xb in signals(d):
                if xb <= eb:
                    continue
                r, held = hold_trade(d, eb, xb, stop)
                rows.append((d.index[eb], tk, eb, xb - eb, held, r))
        df = pd.DataFrame(rows, columns=["ts", "tk", "eb", "plan", "held", "ret"]).sort_values("ts")
        ctrl = []
        for _ in range(SIMS):
            acc = []
            for tk, plan in zip(df.tk, df.plan):
                d = data[tk]
                eb = int(rg.integers(SLOW, len(d) - plan - 1))
                acc.append(hold_trade(d, eb, eb + plan, stop)[0])
            ctrl.append(np.mean(acc))
        cm = float(np.mean(ctrl))
        mon = df.groupby(df.ts.dt.to_period("M")).ret.mean()
        tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
        mid = df.ts.iloc[len(df) // 2]
        h1, h2 = df.ret[df.ts < mid].mean(), df.ret[df.ts >= mid].mean()
        exc = df.ret.mean() - cm
        ok = exc > 0 and tm > 2 and h1 > 0 and h2 > 0
        rec = df[df.ts >= df.ts.max() - pd.Timedelta(days=4 * 365)]
        print(f"  {lab}")
        print(f"    trades {len(df)} (~{len(df) / 21.9:.0f}/yr)   win {(df.ret > 0).mean()*100:.0f}%"
              f"   avg hold {df.held.mean():.0f} days" + (f"   stopped {(df.held < df.plan).mean()*100:.0f}%" if stop else ""))
        print(f"    avg {df.ret.mean()*100:+.2f}%   random same stock/hold {cm*100:+.2f}%   excess {exc*100:+.2f}%"
              f"   best {df.ret.max()*100:+.0f}%   worst {df.ret.min()*100:+.0f}%")
        print(f"    monthly t {tm:+.2f} ({int((mon > 0).sum())}/{len(mon)} months up)   halves {h1*100:+.2f}% / {h2*100:+.2f}%"
              f"   last 4 years {rec.ret.mean()*100:+.2f}% ({len(rec)} trades)")
        print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

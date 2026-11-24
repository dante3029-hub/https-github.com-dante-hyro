#!/usr/bin/env python3
"""
bnf_stocks.py -- BNF deviation buy on US large-cap stocks. Pre-registered,
one run. Runs on your server (this needs internet access to Yahoo Finance).

  pip install yfinance pandas numpy --break-system-packages
  python3 bnf_stocks.py

RULES (fixed before the first run)
  universe   today's S&P 100 members, daily bars, 2005 -> today
             (today's members = survivorship bias: companies that collapsed
              and left the index are missing, which flatters dip-buying)
  signal     daily close at least X% below its 25-day simple moving average
             X = 10% and X = 15%   (large caps move ~1/3 as much as crypto
             alts, so 20% would almost never fire)
  entry      next day's open, long only, one position per stock at a time
  exit       first close back at/above the 25-day SMA, or the close of day 10
  costs      0.10% round trip (Hyperliquid stock perps are dearer than a broker)
  control    random entry, same stock, same holding length
  verdict    excess over random > 0, monthly t > 2, both halves positive
  reported   by year, and with the 10 best entry days removed (2008 / 2020
             crash rebounds could carry the whole result)
"""
from __future__ import annotations

import sys
import zlib

import numpy as np
import pandas as pd

TICKERS = """AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK
BMY BRK-B C CAT CHTR CL CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR
F FDX GD GE GILD GM GOOGL GS HD HON IBM INTC INTU ISRG JNJ JPM KO LIN LLY LMT
LOW MA MCD MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NOW NVDA ORCL PEP
PFE PG PM QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS TSLA TXN UNH UNP UPS USB V
VZ WFC WMT XOM""".split()

ROUND_TRIP = 0.0010
SMA = 25
MAX_HOLD = 10
SIMS = 40


def load(tickers):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance missing: pip install yfinance --break-system-packages")
    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start="2005-01-01", auto_adjust=True, progress=False)
        except Exception as exc:               # network / ticker errors: skip, never crash
            print(f"  skip {t}: {exc}")
            continue
        if d is None or d.empty:
            print(f"  skip {t}: no data")
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "close"]].dropna()
        d = d[(d.open > 0) & (d.close > 0)]
        if len(d) > SMA + 250:
            out[t] = d
    return out


def trades(d, thresh):
    o, c = d.open.values, d.close.values
    sma = pd.Series(c).rolling(SMA).mean().values
    out, t, n = [], SMA, len(c)
    while t < n - 1:
        if c[t] <= sma[t] * (1 - thresh):
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
    rg = np.random.default_rng(zlib.crc32(b"bnf_stocks"))
    data = load(TICKERS)
    print(f"\n  BNF deviation buy -- US large caps, {len(data)} stocks, daily, pre-registered\n")
    for thresh in (0.10, 0.15):
        rows = [(d.index[eb], tk, eb, ex, r) for tk, d in data.items() for eb, ex, r in trades(d, thresh)]
        df = pd.DataFrame(rows, columns=["ts", "tk", "eb", "ex", "ret"]).sort_values("ts")
        if len(df) < 30:
            print(f"  {int(thresh*100)}%: only {len(df)} trades, no verdict\n")
            continue
        ctrl = []
        for _ in range(SIMS):
            acc = []
            for tk, eb0, ex0 in zip(df.tk, df.eb, df.ex):
                d, hold = data[tk], ex0 - eb0
                eb = int(rg.integers(SMA, len(d) - hold - 1))
                acc.append(d.close.values[eb + hold] / d.open.values[eb] - 1 - ROUND_TRIP)
            ctrl.append(np.mean(acc))
        cm = float(np.mean(ctrl))
        mon = df.groupby(df.ts.dt.to_period("M")).ret.mean()
        tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
        mid = df.ts.iloc[len(df) // 2]
        h1, h2 = df.ret[df.ts < mid].mean(), df.ret[df.ts >= mid].mean()
        exc = df.ret.mean() - cm
        ok = exc > 0 and tm > 2 and h1 > 0 and h2 > 0
        day = df.groupby(df.ts.dt.floor("D")).ret.mean().sort_values(ascending=False)
        x = df[~df.ts.dt.floor("D").isin(day.index[:10])]
        print(f"  {int(thresh*100)}% below SMA25")
        print(f"    trades {len(df)} on {len(day)} days   win {(df.ret > 0).mean()*100:.0f}%"
              f"   avg hold {(df.ex - df.eb + 1).mean():.1f}d")
        print(f"    avg {df.ret.mean()*100:+.2f}%   random {cm*100:+.2f}%   excess {exc*100:+.2f}%"
              f"   worst {df.ret.min()*100:+.1f}%")
        print(f"    monthly t {tm:+.2f} ({int((mon > 0).sum())}/{len(mon)} months up)"
              f"   halves {h1*100:+.2f}% / {h2*100:+.2f}%   without 10 best days {x.ret.mean()*100:+.2f}%")
        yr = df.groupby(df.ts.dt.year).ret.agg(["count", "mean"])
        print("    by year: " + "  ".join(f"{y}:{int(r['count'])}/{r['mean']*100:+.1f}%" for y, r in yr.iterrows()))
        rec = df[df.ts >= df.ts.max() - pd.Timedelta(days=4 * 365)]
        if len(rec):
            rc = np.mean([np.mean([data[tk].close.values[(e := int(rg.integers(SMA, len(data[tk]) - (x1 - x0) - 1))) + (x1 - x0)]
                                   / data[tk].open.values[e] - 1 - ROUND_TRIP
                                   for tk, x0, x1 in zip(rec.tk, rec.eb, rec.ex)]) for _ in range(SIMS)])
            print(f"    LAST 4 YEARS: trades {len(rec)}  avg {rec.ret.mean()*100:+.2f}%  random {rc*100:+.2f}%"
                  f"  win {(rec.ret > 0).mean()*100:.0f}%")
        print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
portfolio_sim.py -- the three surviving edges as one portfolio. Runs on the server
(needs Yahoo Finance for stocks; pulls the crypto sleeves from GitHub).

  python3 portfolio_sim.py

RULES (fixed before the first run)
  BNF stocks   bnf_stocks.py rules at 10% below SMA25, run as a real account:
               10% of equity per position, at most 10 open (never leveraged);
               when more stocks signal than free slots, the deepest drops first.
               Marked to market daily; 0.10% round trip charged at entry.
  S/R break    crypto, daily P&L from the audited backtest (stop-bar fix applied)
  OI rank      crypto, daily P&L from the audited backtest
  combining    each sleeve scaled to the same volatility (vol measured on the
               FIRST half of the overlap only), equal risk, rebalanced daily
  planning     every result is also shown with returns HALVED -- live trading
               usually earns about half of a clean backtest
  output       per sleeve and combined: yearly return, volatility, Sharpe,
               worst drawdown, worst day, correlations; then the account
               numbers at 10% / 15% / 20% yearly volatility
"""
from __future__ import annotations

import io
import math
import urllib.request

import numpy as np
import pandas as pd

import bnf_stocks as BS

THRESH = 0.10
POS_FRAC = 0.10
MAX_POS = 10
PEAK_OPEN = 0          # diagnostics: most positions ever open at once
CRYPTO_URL = ("https://raw.githubusercontent.com/dante3029-hub/https-github.com-dante-hyro/"
              "research/nov-2026/results/crypto_sleeves_daily.csv")


def bnf_book(data):
    """Daily return of the BNF stock account. Signals at close t, entry open t+1."""
    sig = []
    for tk, d in data.items():
        c = d.close.values
        sma = pd.Series(c).rolling(BS.SMA).mean().values
        for eb, ex, _ in BS.trades(d, THRESH):
            depth = c[eb - 1] / sma[eb - 1] - 1          # how far below the SMA at the signal
            sig.append((d.index[eb], tk, eb, ex, depth))
    sig.sort(key=lambda x: (x[0], x[4]))                 # same day: deepest first
    days = sorted(set().union(*[set(d.index) for d in data.values()]))
    days = pd.DatetimeIndex(days)
    by_day = {}
    for s in sig:
        by_day.setdefault(s[0], []).append(s)
    open_pos, rets, taken, skipped = [], [], 0, 0
    for day in days:
        # enter today's signals at the open (slots permitting)
        for ts, tk, eb, ex, _ in by_day.get(day, []):
            if len(open_pos) < MAX_POS:
                open_pos.append({"tk": tk, "eb": eb, "ex": ex, "i": eb})
                taken += 1
            else:
                skipped += 1
        global PEAK_OPEN
        PEAK_OPEN = max(PEAK_OPEN, len(open_pos))
        r = 0.0
        keep = []
        for p in open_pos:
            d = data[p["tk"]]
            i = p["i"]
            if i >= len(d) or d.index[i] != day:          # stock not trading today
                keep.append(p)
                continue
            o, c = d.open.values, d.close.values
            if i == p["eb"]:
                pr = c[i] / o[i] - 1 - BS.ROUND_TRIP
            else:
                pr = c[i] / c[i - 1] - 1
            r += POS_FRAC * pr
            p["i"] = i + 1
            if i < p["ex"]:
                keep.append(p)
        open_pos = keep
        rets.append(r)
    out = pd.Series(rets, index=days, name="bnf_stocks")
    print(f"  BNF account: {taken} trades taken, {skipped} skipped for lack of a free slot")
    return out


def stats(r, periods=365):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * math.sqrt(periods)
    sh = r.mean() / r.std() * math.sqrt(periods) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sh, dd, r.min()


def line(name, r, periods=365):
    cagr, vol, sh, dd, wd = stats(r, periods)
    print(f"    {name:<22}{cagr*100:>8.1f}%{vol*100:>8.1f}%{sh:>8.2f}{dd*100:>9.1f}%{wd*100:>9.2f}%")


def head():
    print(f"    {'':<22}{'per yr':>9}{'vol':>9}{'Sharpe':>8}{'worst DD':>10}{'worst day':>10}")


def main() -> int:
    data = BS.load(BS.TICKERS)
    print(f"\n  PORTFOLIO -- BNF stocks + S/R break + OI rank\n")
    bnf = bnf_book(data)

    print("\n  1. BNF stocks as a real account, 2005 -> today (10% per position, max 10)")
    head()
    line("BNF stocks", bnf, 252)
    line("  last 4 years", bnf[bnf.index >= bnf.index[-1] - pd.Timedelta(days=4 * 365)], 252)
    yr = (1 + bnf).groupby(bnf.index.year).prod() - 1
    print("    by year: " + "  ".join(f"{y}:{v*100:+.0f}%" for y, v in yr.items()))
    print(f"    average share of money in use: {(bnf != 0).mean()*100:.0f}% of days with a position")

    try:
        raw = urllib.request.urlopen(CRYPTO_URL, timeout=30).read()
        cr = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    except Exception as exc:
        print(f"\n  could not download crypto sleeves ({exc}); stopping after the stock part")
        return 0

    # calendar-day panel; stocks earn 0 on weekends and holidays
    idx = pd.date_range(cr.index.min(), cr.index.max(), freq="D")
    P = pd.DataFrame({"bnf_stocks": bnf.reindex(idx).fillna(0.0),
                      "sr_break": cr["sr_6h"].reindex(idx).fillna(0.0),
                      "oi_rank": cr["oirank_h3"].reindex(idx).fillna(0.0)})
    P = P[P.index >= P.index[60]]
    mid = P.index[len(P) // 2]
    vol1 = P[P.index < mid].std()
    target = 0.10 / math.sqrt(365)                      # each sleeve at 10%/yr vol
    S = P * (target / vol1)

    print(f"\n  2. The three together, {P.index[0].date()} -> {P.index[-1].date()} (each sleeve at equal risk)")
    print("    correlations: " + "  ".join(f"{a}/{b} {S[a].corr(S[b]):+.2f}"
                                          for a, b in (("bnf_stocks", "sr_break"), ("bnf_stocks", "oi_rank"),
                                                       ("sr_break", "oi_rank"))))
    head()
    for col in S:
        line(col, S[col])
    combo = S.mean(axis=1)
    line("COMBINED", combo)
    line("COMBINED, half returns", combo - combo.mean() / 2)

    print("\n  3. What an account would see, scaled to a chosen risk level")
    print(f"    {'yearly volatility':<22}{'backtest /yr':>13}{'planning /yr':>13}{'per week':>10}{'worst DD':>10}{'planning DD':>12}")
    base_vol = combo.std() * math.sqrt(365)
    for tv in (0.10, 0.15, 0.20):
        k = tv / base_vol
        full, half = combo * k, (combo - combo.mean() / 2) * k
        c1, _, _, d1, _ = stats(full)
        c2, _, _, d2, _ = stats(half)
        print(f"    {tv*100:>5.0f}%{'':<16}{c1*100:>12.1f}%{c2*100:>12.1f}%{c2/52*100:>9.2f}%{d1*100:>9.1f}%{d2*100:>11.1f}%")
    print("\n    'planning' = backtest returns halved; plan on that column, hope for the other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

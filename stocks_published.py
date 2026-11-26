#!/usr/bin/env python3
"""
stocks_published.py -- the classic published stock/macro strategies, one run
each, plus how every stock strategy (BNF included) does in bull markets,
chop and declines. Runs on the server (Yahoo Finance).

  python3 stocks_published.py        (needs bnf_stocks.py in the same folder)

RULES (from the literature, fixed before the first run)
 1 STOCK MOMENTUM (Jegadeesh & Titman 1993; "12-1" momentum)
     score   return over the last 12 months EXCLUDING the most recent month
     book    each month-end, long the top 10 of the S&P 100 names, equal
             weight; compared with holding all of them equally (and a short
             leg version: long top 10, short bottom 10)
     note    today's S&P 100 = survivorship bias, which FLATTERS momentum
 2 MACRO TREND (Moskowitz, Ooi & Pedersen 2012, time-series momentum)
     markets SPY QQQ IWM EFA EEM TLT IEF GLD SLV USO DBC UUP (ETFs)
     signal  sign of the 12-month return, checked each month-end
     size    each market scaled to equal volatility (60-day), gross 1.0
     version long/short, and long-or-flat
 3 SPY 200-DAY (the simplest trend filter): long SPY when it closes above its
     200-day average, otherwise cash
 COSTS   0.10% per side on everything traded
 CONTROL momentum: random 10 names each month (40 seeds)
         trend: each market's signal series circularly shifted (40 seeds)
 VERDICT excess over control > 0 in both halves, monthly t of the excess > 2
 REGIMES SPY's 200-day trend at t-1: bull (above, 200d average rising), bear
         (below, falling), chop (mixed). Descriptive only, nothing tuned on it.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

import bnf_stocks as BS

COST = 0.0010
SEEDS = 40
MACRO = "SPY QQQ IWM EFA EEM TLT IEF GLD SLV USO DBC UUP".split()


def closes(data):
    return pd.DataFrame({k: v["close"] for k, v in data.items()}).sort_index()


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.month]).max())


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return (eq.iloc[-1] ** (1 / yrs) - 1, r.std() * math.sqrt(252),
            r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0, (eq / eq.cummax() - 1).min())


def verdict(name, real, ctrl):
    exc = real - ctrl
    m = exc.resample("ME").sum()
    m = m[m != 0]
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))
    mid = real.index[len(real) // 2]
    h1, h2 = exc[exc.index < mid].mean() * 252, exc[exc.index >= mid].mean() * 252
    cg, vol, sh, dd = stats(real)
    cg2, _, sh2, dd2 = stats(ctrl)
    ok = h1 > 0 and h2 > 0 and t > 2
    print(f"  {name}")
    print(f"    strategy {cg*100:+6.1f}%/yr  Sharpe {sh:4.2f}  worst DD {dd*100:5.1f}%   |  control {cg2*100:+6.1f}%/yr"
          f"  Sharpe {sh2:4.2f}  DD {dd2*100:5.1f}%")
    print(f"    excess {exc.mean()*252*100:+.1f}%/yr  halves {h1*100:+.1f}% / {h2*100:+.1f}%  monthly t {t:+.2f}"
          f"  ({int((m > 0).sum())}/{len(m)} months)   verdict: {'PASS' if ok else 'REJECT'}\n")


def momentum_book(px, pick):
    """Hold the names `pick(t)` chooses at each month-end, equal weight, daily returns."""
    rets = px.pct_change()
    me = month_ends(px.index)
    w = pd.Series(0.0, index=px.columns)
    out = []
    for i, day in enumerate(px.index):
        r = (w * rets.loc[day].fillna(0)).sum() if i else 0.0
        if day in me:
            nw = pick(i)
            r -= (nw - w).abs().sum() * COST
            w = nw
        out.append(r)
    return pd.Series(out, index=px.index)


def trend_book(px, shift_seed=None, long_only=False):
    rets = px.pct_change()
    sig = np.sign(px / px.shift(252) - 1)
    if long_only:
        sig = sig.clip(lower=0)
    if shift_seed is not None:
        rg = np.random.default_rng(shift_seed)
        for c in sig:
            ok = sig[c].dropna()
            if len(ok) > 300:
                sig.loc[ok.index, c] = np.roll(ok.values, int(rg.integers(60, len(ok) - 60)))
    vol = rets.rolling(60).std()
    me = month_ends(px.index)
    w = pd.Series(0.0, index=px.columns)
    out = []
    for i, day in enumerate(px.index):
        r = (w * rets.loc[day].fillna(0)).sum() if i else 0.0
        if day in me and i > 260:
            raw = (sig.iloc[i] / vol.iloc[i]).replace([np.inf, -np.inf], np.nan).fillna(0)
            g = raw.abs().sum()
            nw = raw / g if g > 0 else raw * 0
            r -= (nw - w).abs().sum() * COST
            w = nw
        out.append(r)
    return pd.Series(out, index=px.index)


def main() -> int:
    rg = np.random.default_rng(zlib.crc32(b"stocks_published"))
    stocks = BS.load(BS.TICKERS)
    macro = BS.load(MACRO)
    px = closes(stocks)
    mx = closes(macro).dropna(how="all")
    print(f"\n  Published stock/macro strategies -- {px.shape[1]} stocks, {mx.shape[1]} markets\n")

    # 1 momentum
    score = px.shift(21) / px.shift(252) - 1
    def top10(i):
        s = score.iloc[i].dropna()
        w = pd.Series(0.0, index=px.columns)
        if len(s) >= 30:
            w[s.nlargest(10).index] = 0.1
        return w
    def ls10(i):
        s = score.iloc[i].dropna()
        w = pd.Series(0.0, index=px.columns)
        if len(s) >= 30:
            w[s.nlargest(10).index] = 0.05
            w[s.nsmallest(10).index] = -0.05
        return w
    def rand10(seed):
        g = np.random.default_rng(seed)
        def f(i):
            s = score.iloc[i].dropna()
            w = pd.Series(0.0, index=px.columns)
            if len(s) >= 30:
                w[g.choice(s.index, 10, replace=False)] = 0.1
            return w
        return f
    real = momentum_book(px, top10)
    ctrl = pd.concat([momentum_book(px, rand10(zlib.crc32(f"m{s}".encode()))) for s in range(SEEDS)], axis=1).mean(axis=1)
    real, ctrl = real[real.index >= px.index[300]], ctrl[ctrl.index >= px.index[300]]
    verdict("1  STOCK MOMENTUM 12-1, top 10 of S&P 100, monthly (vs random 10)", real, ctrl)
    ls = momentum_book(px, ls10)
    ls = ls[ls.index >= px.index[300]]
    verdict("1b STOCK MOMENTUM long top 10 / short bottom 10 (vs flat)", ls, ls * 0)
    mom_daily = real

    # 2 macro trend
    for lo, lab in ((False, "long/short"), (True, "long or flat")):
        real = trend_book(mx, long_only=lo)
        ctrl = pd.concat([trend_book(mx, shift_seed=zlib.crc32(f"t{lo}{s}".encode()), long_only=lo)
                          for s in range(SEEDS)], axis=1).mean(axis=1)
        st = mx.index[300]
        verdict(f"2  MACRO TREND 12-month, {mx.shape[1]} markets, {lab}", real[real.index >= st], ctrl[ctrl.index >= st])
        if not lo:
            trend_daily = real[real.index >= st]

    # 3 SPY 200-day
    spy = mx["SPY"].dropna()
    above = (spy > spy.rolling(200).mean()).shift(1, fill_value=False)
    r = spy.pct_change().fillna(0)
    switch = above.astype(int).diff().abs().fillna(0)
    strat = r * above - switch * COST
    strat = strat[strat.index >= spy.index[220]]
    verdict("3  SPY above 200-day average, else cash (vs always holding SPY)", strat, r[strat.index])

    # regimes
    bnf = None
    try:
        import portfolio_sim as PS
        bnf = PS.bnf_book(stocks)
    except Exception as exc:
        print(f"  (BNF account skipped: {exc})")
    sma = spy.rolling(200).mean()
    up = (spy > sma) & (sma.diff(20) > 0)
    dn = (spy < sma) & (sma.diff(20) < 0)
    reg = pd.Series(np.where(up, "bull", np.where(dn, "bear", "chop")), index=spy.index).shift(1)
    S = pd.DataFrame({"momentum": mom_daily, "macro trend": trend_daily, "SPY 200d": strat})
    if bnf is not None:
        S["BNF dip"] = bnf
    S["SPY itself"] = r
    S = S.dropna(how="all").fillna(0)
    reg = reg.reindex(S.index)
    print("  REGIMES (SPY 200-day trend, known at the time)  days: "
          + "  ".join(f"{k} {int((reg == k).sum())}" for k in ("bull", "chop", "bear")))
    print(f"  {'strategy':<14}" + "".join(f"{k + ' %/yr':>13}" for k in ("bull", "chop", "bear")))
    for c in S:
        print(f"  {c:<14}" + "".join(f"{S[c][reg == k].mean()*252*100:>12.1f}%" for k in ("bull", "chop", "bear")))
    print("\n  daily correlations:")
    print(S.drop(columns="SPY itself").corr().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
forensic_check.py -- try to BREAK the BNF and portfolio code. Every test states
what a bug would look like before it runs.

  1 hand case        a series with a known answer: entry bar, exit bar, return
  2 future tamper    scramble every bar after a cutoff; no trade that exits
                     before the cutoff may change (else: lookahead)
  3 random walk      prices with no pattern: the strategy must show ~0 edge over
                     random entries (else: the code manufactures profit)
  4 independent      a second BNF implementation written a different way
                     (vectorised scan, no shared code): identical trade lists
  5 account          portfolio: never > MAX_POS open, fee charged once per trade,
                     daily P&L of each trade compounds back to its trade return
  6 crypto BNF       tests 2-4 on bnf_dip.py with the real crypto daily data
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import bnf_stocks as BS
import bnf_dip as BD
import portfolio_sim as PS

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def frame(c, o=None, start="2010-01-01"):
    idx = pd.bdate_range(start, periods=len(c))
    o = c if o is None else o
    return pd.DataFrame({"open": o, "close": c}, index=idx)


def independent_trades(d, thresh, sma_n, max_hold, cost):
    """Second implementation: different structure, no shared code."""
    c = d["close"].to_numpy(float)
    o = d["open"].to_numpy(float)
    sma = np.convolve(c, np.ones(sma_n) / sma_n, mode="full")[:len(c)]
    sma[:sma_n - 1] = np.nan
    below = c <= sma * (1 - thresh)
    out, busy_until = [], -1
    for t in np.flatnonzero(below):
        if t < sma_n or t <= busy_until or t >= len(c) - 1:
            continue
        e = t + 1
        window = np.arange(e, min(e + max_hold, len(c)))
        back = window[c[window] >= sma[window]]
        x = int(back[0]) if len(back) else int(window[-1])
        out.append((e, x, c[x] / o[e] - 1 - cost))
        busy_until = x
    return out


def random_walk(n, rg, vol=0.02):
    r = rg.normal(0, vol, n)
    c = 100 * np.exp(np.cumsum(r))
    o = c * np.exp(rg.normal(0, vol / 4, n))
    return frame(c, o)


def edge_vs_random(frames, trades_fn, cost, rg, sims=20):
    real, ctrl = [], []
    for d in frames:
        tr = trades_fn(d)
        real += [r for _, _, r in tr]
        c, o = d.close.values, d.open.values
        for _ in range(sims):
            for eb, ex, _ in tr:
                hold = ex - eb
                s = int(rg.integers(30, len(c) - hold - 1))
                ctrl.append(c[s + hold] / o[s] - 1 - cost)
    real, ctrl = np.array(real), np.array(ctrl)
    se = real.std(ddof=1) / np.sqrt(len(real))
    return real.mean(), ctrl.mean(), (real.mean() - ctrl.mean()) / se, len(real)


def main() -> int:
    rg = np.random.default_rng(7)
    print("\n  FORENSIC CHECK -- BNF stocks, BNF crypto, portfolio sim\n")

    # ---------- 1 hand case ----------
    c = np.r_[np.full(40, 100.0), [88.0], np.linspace(90, 99, 5), np.full(20, 100.0)]
    o = c.copy(); o[41] = 89.0
    d = frame(c, o)
    tr = BS.trades(d, 0.10)
    # SMA25 at bar 40 = (24*100 + 88)/25 = 99.52; 88 <= 0.9*99.52 = 89.57 -> signal at 40
    # entry open[41] = 89; closes 90..99 stay below the SMA until c >= sma
    sma = pd.Series(c).rolling(25).mean().values
    exp_x = next(j for j in range(41, 51) if c[j] >= sma[j])
    exp = (41, exp_x, c[exp_x] / 89.0 - 1 - BS.ROUND_TRIP)
    check("1 hand case: one trade, right entry/exit/return",
          len(tr) == 1 and tr[0][:2] == exp[:2] and abs(tr[0][2] - exp[2]) < 1e-12,
          f"got {tr}, expected {exp}")
    check("1 no trade without a 10% drop", BS.trades(frame(np.full(80, 100.0)), 0.10) == [])

    # ---------- 2 future tamper ----------
    bad = 0; tested = 0
    for k in range(30):
        d = random_walk(1500, rg)
        cut = int(rg.integers(300, 1300))
        d2 = d.copy()
        d2.iloc[cut:, :] = d2.iloc[cut:, :].values * np.exp(rg.normal(0, 0.3, (len(d2) - cut, 2)))
        a = [t for t in BS.trades(d, 0.10) if t[1] < cut]
        b = [t for t in BS.trades(d2, 0.10) if t[1] < cut]
        tested += len(a); bad += a != b
    check("2 future tamper (stocks code): trades before cutoff unchanged", bad == 0, f"{tested} trades, {bad} series changed")

    # ---------- 3 random walk ----------
    frames = [random_walk(5000, rg) for _ in range(40)]
    rm, cm, t, n = edge_vs_random(frames, lambda d: BS.trades(d, 0.10), BS.ROUND_TRIP, rg)
    check("3 random walk: no edge on patternless prices", abs(t) < 3,
          f"{n} trades, strategy {rm*100:+.3f}% vs random {cm*100:+.3f}%, t {t:+.2f}")
    # positive control: mean-reverting prices MUST show an edge (proves the test can detect one)
    mr = []
    for _ in range(40):
        x = np.zeros(5000); e = rg.normal(0, 0.03, 5000)
        for i in range(1, 5000):
            x[i] = 0.9 * x[i - 1] + e[i]
        cc = 100 * np.exp(x + np.cumsum(rg.normal(0, 0.005, 5000)))
        mr.append(frame(cc, cc * np.exp(rg.normal(0, 0.003, 5000))))
    rm2, cm2, t2, n2 = edge_vs_random(mr, lambda d: BS.trades(d, 0.10), BS.ROUND_TRIP, rg)
    check("3 positive control: mean-reverting prices DO show an edge", t2 > 3,
          f"{n2} trades, strategy {rm2*100:+.3f}% vs random {cm2*100:+.3f}%, t {t2:+.2f}")

    # ---------- 4 independent implementation ----------
    mism = 0; total = 0
    for _ in range(40):
        d = random_walk(3000, rg, vol=0.03)
        a = BS.trades(d, 0.10)
        b = independent_trades(d, 0.10, BS.SMA, BS.MAX_HOLD, BS.ROUND_TRIP)
        total += len(a)
        same = len(a) == len(b) and all(x[:2] == y[:2] and abs(x[2] - y[2]) < 1e-12 for x, y in zip(a, b))
        mism += not same
    check("4 independent rebuild (stocks): identical trade lists", mism == 0, f"{total} trades, {mism} series differ")

    # ---------- 5 account ----------
    data = {f"S{k}": random_walk(3000, rg, vol=0.035) for k in range(40)}
    peak = {"n": 0}
    orig_trades = BS.trades
    book = PS.bnf_book(data)
    # rerun the book loop instrumented for open-position count
    sig = sorted([(dd.index[eb], tk, eb, ex) for tk, dd in data.items() for eb, ex, _ in BS.trades(dd, PS.THRESH)])
    check("5 account: daily returns finite, no day beyond 10 positions' worth",
          np.isfinite(book).all() and book.abs().max() <= PS.MAX_POS * PS.POS_FRAC * 1.0 + 1e-9,
          f"largest daily move {book.abs().max()*100:.2f}%")
    # one-trade account: the daily path must compound back to the trade's own return
    one = {"A": data["S0"]}
    trs = BS.trades(one["A"], PS.THRESH)
    if trs:
        eb, ex, r = trs[0]
        saved = PS.POS_FRAC
        PS.POS_FRAC = 1.0
        b1 = PS.bnf_book({"A": one["A"].iloc[: ex + 1]})
        PS.POS_FRAC = saved
        dd = one["A"]
        gross_path = np.prod(1 + b1.values)
        expect = (dd.close.values[eb] / dd.open.values[eb] - BS.ROUND_TRIP) * dd.close.values[ex] / dd.close.values[eb]
        check("5 account: a trade's daily P&L compounds to its trade return (fee once)",
              abs(gross_path - expect) < 1e-9, f"path {gross_path:.6f} vs expected {expect:.6f}; trade list says {1+r:.6f}")
    PS.PEAK_OPEN = 0
    PS.bnf_book(data)
    check("5 account: never more than MAX_POS positions open", PS.PEAK_OPEN <= PS.MAX_POS,
          f"peak open {PS.PEAK_OPEN} of {PS.MAX_POS}, {len(sig)} signals offered")
    saved = PS.MAX_POS; PS.MAX_POS = 2; PS.PEAK_OPEN = 0
    PS.bnf_book(data)
    check("5 account: cap of 2 also respected (cap actually binds)", PS.PEAK_OPEN <= 2, f"peak open {PS.PEAK_OPEN}")
    PS.MAX_POS = saved

    # ---------- 6 crypto BNF on real data ----------
    try:
        import event_audit as EA
        import short_risk as SR
        coins = SR.universe()[:40]
        bad = tested = mism = total = 0
        for coin in coins:
            d1 = EA.load_bars(coin, 1)
            if d1 is None:
                continue
            d = BD.daily(d1)
            if len(d) < 200:
                continue
            a = BD.trades(d, 0.20)
            b = independent_trades(d, 0.20, BD.SMA, BD.MAX_HOLD, BD.ROUND_TRIP)
            total += len(a)
            mism += not (len(a) == len(b) and all(x[:2] == y[:2] and abs(x[2] - y[2]) < 1e-12 for x, y in zip(a, b)))
            cut = len(d) // 2
            d2 = d.copy()
            d2.iloc[cut:, :] = d2.iloc[cut:, :].values * np.exp(rg.normal(0, 0.3, (len(d2) - cut, d2.shape[1])))
            x1 = [t for t in BD.trades(d, 0.20) if t[1] < cut]
            x2 = [t for t in BD.trades(d2, 0.20) if t[1] < cut]
            tested += len(x1); bad += x1 != x2
        check("6 crypto BNF: future tamper on real data", bad == 0, f"{tested} trades, {bad} coins changed")
        check("6 crypto BNF: independent rebuild on real data", mism == 0, f"{total} trades, {mism} coins differ")
        cf = []
        for coin in coins[:20]:
            d1 = EA.load_bars(coin, 1)
            if d1 is not None:
                d = BD.daily(d1)
                # same coin's daily returns shuffled = same volatility, no pattern
                r = np.log(d.close).diff().dropna().to_numpy(copy=True)
                rg.shuffle(r)
                cc = d.close.values[0] * np.exp(np.r_[0, np.cumsum(r)])
                cf.append(frame(cc, np.r_[cc[0], cc[:-1]]))
        rm3, cm3, t3, n3 = edge_vs_random(cf, lambda d: BD.trades(d, 0.20), BD.ROUND_TRIP, rg)
        check("6 crypto BNF: shuffled real returns show no edge", abs(t3) < 3,
              f"{n3} trades, strategy {rm3*100:+.2f}% vs random {cm3*100:+.2f}%, t {t3:+.2f}")
    except Exception as exc:
        check("6 crypto BNF", False, f"could not run: {exc}")

    print(f"\n  {'ALL TESTS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
published.py -- three strategies with published evidence, tested once each on
our data. Rules taken from the literature and fixed before the first run.

DATA   63 coins, daily, Feb 2023 - Sep 2026 (oos_book.build_panels): prices,
       daily returns R, daily funding FN (charged to longs, paid to shorts).
COSTS  0.085% per side on turnover (taker + slippage), funding every day.
PERIODS  PRE = 2023 -> 2024-02-12 (never used by anything in this project)
         FIT = 2024-02-13 -> 2026-08-22, POST = after.

1  XS MOMENTUM (Liu, Tsyvinski & Wu, J. Finance 2022: momentum at 1-4 weeks)
   score   past 21-day return, measured to the close of t-1
   book    long the top 6, short the bottom 6 (about a quintile), equal weight,
           dollar neutral, gross 1.0; hold 7 days, 7 staggered sub-books
   control same engine with random scores (40 seeds)

2  TREND (crypto trend-following literature; 10/40-day SMA favoured)
   signal  per coin: +1 if SMA10 > SMA40 at t-1, else -1;  version L = long or flat
   size    each coin scaled to equal volatility (20-day vol at t-1), gross 1.0
   universe top 30 by median dollar volume
   control each coin's signal series circularly shifted by a random offset
           (same persistence and turnover, wrong timing), 40 seeds

3  FUNDING CARRY, perps only (cross-sectional version of the carry trade)
   score   minus the trailing 7-day mean funding to t-1: long the cheapest,
           short the most expensive; 5 per side, hold 7 days, 7 sub-books
   universe coins with funding data
   control random scores, 40 seeds

VERDICT  excess over control > 0 in PRE and in FIT, monthly t of the excess
         > 2 over the whole sample. Anything less is a rejection.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

import oos_book as OB

SEEDS = 40
FEE = OB.FEE


def periods(dates):
    first = dates[OB.WARMUP]
    return {"PRE": (dates >= first) & (dates < OB.FIT_START),
            "FIT": (dates >= OB.FIT_START) & (dates <= OB.FIT_END),
            "ALL": dates >= first}


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else 0.0


def mt(dates, x):
    s = pd.Series(x, index=dates).resample("ME").sum()
    s = s[s != 0]
    return float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), int((s > 0).sum()), len(s)


def report(name, dates, real, ctrl):
    cm = ctrl.mean(axis=0)
    exc = real - cm
    per = periods(dates)
    print(f"  {name}")
    ok = True
    for pn, m in per.items():
        t, up, nm = mt(dates[m], exc[m])
        ann = real[m].mean() * 365 * 100
        print(f"    {pn:<4} real Sharpe {sharpe(real[m]):5.2f}  ({ann:+6.1f}%/yr)  random {sharpe(cm[m]):5.2f}"
              f"  excess {sharpe(exc[m]):5.2f}  t {t:5.2f}  {up}/{nm} months")
        if pn in ("PRE", "FIT") and sharpe(exc[m]) <= 0:
            ok = False
        if pn == "ALL" and t <= 2:
            ok = False
    print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")
    return exc


def trend_book(PX, R, FN, cols, shift_seed=None, long_only=False):
    T, C = PX.shape
    px = pd.DataFrame(PX)
    s10, s40 = px.rolling(10).mean().values, px.rolling(40).mean().values
    sig = np.where(np.isfinite(s10) & np.isfinite(s40), np.where(s10 > s40, 1.0, -1.0), np.nan)
    if long_only:
        sig = np.where(sig > 0, 1.0, np.where(np.isfinite(sig), 0.0, np.nan))
    if shift_seed is not None:
        rg = np.random.default_rng(shift_seed)
        for j in range(C):
            ok = np.flatnonzero(np.isfinite(sig[:, j]))
            if len(ok) > 50:
                seg = sig[ok, j]
                sig[ok, j] = np.roll(seg, int(rg.integers(20, len(seg) - 20)))
    vol = pd.DataFrame(R).rolling(20).std().values
    out = np.zeros(T)
    w_prev = np.zeros(C)
    for t in range(OB.WARMUP, T):
        s, v = sig[t - 1].copy(), vol[t - 1]
        s[~cols] = np.nan
        valid = np.isfinite(s) & np.isfinite(v) & (v > 0) & np.isfinite(PX[t])
        w = np.zeros(C)
        if valid.sum() >= 5:
            raw = np.where(valid, s / np.where(valid, v, 1.0), 0.0)
            g = np.abs(raw).sum()
            if g > 0:
                w = raw / g
        r = np.nan_to_num(R[t])
        out[t] = np.sum(w * r) - np.abs(w - w_prev).sum() * FEE - np.sum(w * np.nan_to_num(FN[t]))
        w_prev = w
    return out


def main() -> int:
    P = OB.build_panels()
    PX, R, FN, dates = P["PX"], P["R"], P["FN"], P["dates"]
    T, C = PX.shape
    print(f"\n  Published strategies, one run each -- {C} coins, {dates[0].date()} -> {dates[-1].date()}\n")

    # 1 cross-sectional momentum
    lp = np.log(PX)
    def mom(t):
        if t < 23:
            return np.full(C, np.nan)
        return lp[t - 1] - lp[t - 22]
    real = OB.run_sleeve(mom, PX, R, FN, 6, 7, 7, None)
    ctrl = np.array([OB.run_sleeve(OB.sig_random(C, zlib.crc32(f"mom{s}".encode())), PX, R, FN, 6, 7, 7, None)
                     for s in range(SEEDS)])
    e1 = report("1  XS MOMENTUM: 21-day return, top 6 / bottom 6, weekly", dates, real, ctrl)

    # 2 trend
    liq = np.nanmedian(P["QV"], axis=0)
    top30 = np.zeros(C, bool)
    top30[np.argsort(np.nan_to_num(liq, nan=-1))[-30:]] = True
    for lo, lab in ((False, "long/short"), (True, "long or flat")):
        real = trend_book(PX, R, FN, top30, long_only=lo)
        ctrl = np.array([trend_book(PX, R, FN, top30, shift_seed=zlib.crc32(f"tr{lo}{s}".encode()), long_only=lo)
                         for s in range(SEEDS)])
        report(f"2  TREND: SMA10 vs SMA40, vol-scaled, top 30, {lab}", dates, real, ctrl)

    # 3 funding carry
    fcols = np.asarray(P["has_fn"], bool)
    fn = pd.DataFrame(FN).rolling(7, min_periods=5).mean().values
    def carry(t):
        return -fn[t - 1] if t >= 8 else np.full(C, np.nan)
    real = OB.run_sleeve(carry, PX, R, FN, 5, 7, 7, None, fcols)
    ctrl = np.array([OB.run_sleeve(OB.sig_random(C, zlib.crc32(f"fc{s}".encode())), PX, R, FN, 5, 7, 7, None, fcols)
                     for s in range(SEEDS)])
    print(f"  (funding universe: {int(fcols.sum())} coins)")
    report("3  FUNDING CARRY: short richest funding, long cheapest, weekly", dates, real, ctrl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

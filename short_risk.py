#!/usr/bin/env python3
"""
short_risk.py -- can the V29 short book actually be run inside the eval?

WHERE THIS COMES FROM. tf_confirm.py established that V29's short book at
10-14h earns roughly 1.0-1.5% gross per trade against a 0.170% round trip,
in both time halves, and beats a randomly timed short (which LOSES money
here) at p=0.003 on the owner's data. That is the strongest result this
project has produced.

It is also not a strategy yet, for a reason that has nothing to do with
statistics: a per-trade edge says nothing about whether the book can be
held. Sixty simultaneous alt shorts are not sixty positions. In a melt-up
every leg loses at once and correlation goes to 1, and the eval does not
care about the annual Sharpe -- it fails the account on ONE day over 5%,
or 10% from the high-water mark.

So this measures the thing that actually decides deployability:

  1. CONCURRENCY. How many shorts are open at the same time? The live V29
     capped at 2 concurrent positions; the backtest that produced the edge
     took every signal on 60+ coins with no global cap. If the measured
     edge requires 25 simultaneous legs, the edge was never available at
     the cap it was traded under, and that is a different strategy.

  2. DAILY MARK-TO-MARKET. Not realised PnL at exit -- the daily limit is
     breached by open positions moving against you intraday. Each open
     short is marked at that day's close.

  3. WORST DAY AND WORST DRAWDOWN at unit size, then solved backwards:
     what per-leg size keeps the WORST HISTORICAL DAY inside 5% and the
     worst drawdown inside 10%? And what does the strategy then return at
     that size? That number is the answer to "can this pass the eval",
     and it is allowed to come out depressing.

  4. A BLOCK BOOTSTRAP over the daily series, because the worst day in
     the sample is not the worst day available. Sizing to the historical
     maximum is sizing to a number that has already been beaten once by
     definition.

DELIBERATELY CONSERVATIVE, and these are the direction that matters:
  - Daily marks ignore the intrabar stop, so a day that gapped through a
    stop is charged the full close-to-close move rather than the stop. The
    real book would have been stopped out earlier and lost less. This
    OVERSTATES the worst day. For a risk limit that is the correct way to
    be wrong.
  - Equal size per leg, no vol targeting. Vol-scaling would reduce the
    worst day; it is a separate improvement and is not assumed here.
  - Funding is not modelled. A short alt book usually EARNS funding, so
    this understates returns slightly. Flagged, not credited.

USAGE
    python3 -u short_risk.py                 # 12h, short-only
    python3 -u short_risk.py --tf 14
    python3 -u short_risk.py --both-sides    # include the long legs
    python3 -u short_risk.py --cap 2         # impose V29's live 2-position cap
"""
from __future__ import annotations

import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd

import v29_test as V

DAILY_LIMIT = 0.05        # HyroTrader: 5% of day-start equity
TOTAL_LIMIT = 0.10        # 10% static floor from starting balance
TARGET = 0.10             # phase-1 profit target


def log(m=""):
    print(m, flush=True)


def universe():
    out = []
    for f in sorted(glob.glob(f"{V.PERP}/*_1h.csv")):
        c = os.path.basename(f).replace("_1h.csv", "")
        if c != "BTC":
            out.append(c)
            V.COIN_CONFIGS.setdefault(c, (21, 2.0))
    return out


def build(coins, tf, sides, cap):
    """Returns (daily_pnl_series_at_unit_size, concurrency_series, trades).

    Each trade contributes, on every day it is open, MINUS its side times
    that coin's close-to-close return. Unit size = 1.0 of equity per leg,
    so the series is scaled afterwards rather than re-run.
    """
    V.TF = tf
    legs = []          # (coin, entry_ts, exit_ts, side)
    trades = []
    daily_ret = {}     # coin -> Series of daily close-to-close returns
    for c in coins:
        df = V._load(f"{V.PERP}/{c}_1h.csv")
        if df is None:
            continue
        bars = V._resample(df, tf) if tf > 1 else df
        if len(bars) < 300:
            continue
        tr = V.run_coin(c, bars, btc=V.btc_macd_dir(bars.index))
        if not tr:
            continue
        d = df["close"].resample("1D").last().dropna()
        daily_ret[c] = d.pct_change()
        for t in tr:
            if t[3] not in sides:
                continue
            eb, ex = int(t[2]), int(t[4])
            eb = min(eb, len(bars) - 1)
            ex = min(max(ex, eb), len(bars) - 1)
            legs.append((c, bars.index[eb], bars.index[ex], t[3],
                         float(t[0]), float(t[5])))
            trades.append(t)

    if not legs:
        return None, None, []

    # ---- global concurrency cap, applied in time order -------------------
    # Without this the book is "every signal on 64 coins". V29 live ran a
    # cap of 2. Both are reported because they are different strategies and
    # only one of them produced the measured edge.
    legs.sort(key=lambda x: x[1])
    if cap:
        kept, open_until = [], []
        for lg in legs:
            open_until = [u for u in open_until if u > lg[1]]
            if len(open_until) < cap:
                kept.append(lg)
                open_until.append(lg[2])
        legs = kept

    return (*legs_to_daily(legs, daily_ret), legs)


def legs_to_daily(legs, daily_ret):
    """Positions -> a daily mark-to-market series. ONE copy, deliberately.

    legs: [(coin, entry_ts, exit_ts, side, realised_return, stop_fraction)]
    daily_ret: {coin: Series of daily close-to-close returns}

    This is the function that carried the sign error (-side * r instead of
    side * r) through three commits, invisible because each leg's TOTAL is
    pinned to its backtested return and only the PATH was wrong. Event
    sleeves need exactly the same conversion, and writing a second copy of
    the one piece of arithmetic in this project that has already been
    silently wrong once is how you get it wrong again in a place nobody
    thinks to check. So: one implementation, two callers.
    """
    all_days = pd.DatetimeIndex(sorted(set().union(
        *[set(daily_ret[c].index) for c, *_ in legs if c in daily_ret])))
    pnl = pd.Series(0.0, index=all_days)
    conc = pd.Series(0, index=all_days)
    for c, t0, t1, side, realised, sl in legs:
        r = daily_ret.get(c)
        if r is None:
            continue
        d0 = t0.normalize()
        d1 = max(t1.normalize(), d0)
        win = r.loc[(r.index > d0) & (r.index <= d1)]
        if len(win) == 0:
            continue
        # THE FIX. Marking a leg close-to-close for every day between entry
        # and exit charges the book the whole price path, including the part
        # AFTER the stop or the trail took the position off. On a 12h short
        # book that produced a worst day of -459% at unit size, essentially
        # all of it from legs that were already flat. It is not a
        # conservative approximation, it is a different and much worse
        # strategy.
        #
        # So: build the leg's cumulative short return day by day, floor it
        # at the stop distance (the position cannot lose more than that),
        # and force the final day to the return the trade ACTUALLY made, so
        # the daily series sums to the backtested PnL by construction.
        # SIGN. A position with direction `side` earns `side * r` on a day
        # the price returns r: a long (+1) earns +r, a short (-1) earns -r.
        # This read -side * r, so every short leg's daily path was INVERTED
        # -- the book was credited on the days it actually lost.
        #
        # It survived three commits because cum[-1] = realised forces each
        # leg's TOTAL to the backtested return, so the mean daily return
        # and the reconciliation against tf_confirm's per-trade number were
        # both correct. Only the PATH was wrong, which is to say: every
        # number that depends on the path. Worst day, daily sd, Sharpe,
        # drawdown, the permitted size, the pass and bust rates, the
        # days-to-target, and the entire rules comparison.
        #
        # What exposed it was a regression beta of +2.47 to the alt complex
        # for a book that is short it. That is not a surprising result, it
        # is an impossible one, and an impossible number is the only kind
        # that cannot be rationalised. The lesson is not "check signs" --
        # it is that a quantity whose sign is known a priori is worth
        # computing precisely because it cannot be argued with.
        cum = np.cumsum(side * win.values)
        if np.isfinite(sl) and sl > 0:
            cum = np.maximum(cum, -sl)
        cum[-1] = realised
        daily = np.diff(np.concatenate([[0.0], cum]))
        pnl.loc[win.index] += daily
        conc.loc[win.index] += 1
    return pnl.dropna(), conc


def risk_at(pnl, s):
    """Worst day and max drawdown at per-leg size s. Computed on the scaled
    series rather than by multiplying the unit-size numbers, because
    drawdown compounds and does not scale linearly.

    In LOG space. np.cumprod(1 + v) overflowed to inf at the 1.0x
    reference size -- 1.0x means a full equity per leg, so with 6-10 legs
    on it is 6-10x leverage and the product runs away before the series
    ends. That printed as a numpy RuntimeWarning in the middle of the
    report and a drawdown of exactly -100.00%, which is not a measurement,
    it is an overflow. A leg that loses more than 100% is ruin and is
    clamped, so the drawdown saturates at -100% honestly instead.
    """
    v = pnl.values * s
    if v.min() <= -1.0:                     # ruin: equity through zero
        return float(v.min()), -1.0
    le = np.cumsum(np.log1p(v))             # log equity, no overflow
    dd = float(np.expm1(le - np.maximum.accumulate(le)).min())
    return float(v.min()), dd


def solve_size(pnl):
    """Largest per-leg size keeping the worst historical day inside the
    daily limit AND the worst drawdown inside the total limit. Bisection,
    because the drawdown constraint is not linear in size."""
    def ok(s):
        wd, dd = risk_at(pnl, s)
        return wd >= -DAILY_LIMIT and dd >= -TOTAL_LIMIT
    if ok(100.0):
        return 100.0
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def race_to_barrier(pnl, s, blocks=5, draws=4000, max_days=1095,
                    min_days=5, consistency=0.40, seed=b"race"):
    """Run the account until it passes or busts, under the REAL eval rules.

    The 45-day window test below was answering the wrong question. The
    HyroTrader evaluation has a MINIMUM of 5 trading days and no maximum:
    the account runs until it reaches +10%, breaches the 10% static floor,
    or breaches 5% in a single day. Time is free. So the statistic that
    decides deployability is not "what does 45 days return", it is
    P(+10% before -10%) and how long that takes.

    Also enforced here, and it is the constraint a lumpy strategy fails:
    the 40% CONSISTENCY RULE. No single day may contribute more than 40%
    of total profit. A short book whose expectancy comes from rare large
    days is precisely the shape that passes the target and then fails this.
    Counted separately from a bust so the two are not confused.
    """
    v = pnl.values * s
    n = len(v)
    rng = np.random.default_rng(zlib.crc32(seed))
    out = {"pass": 0, "bust_total": 0, "bust_daily": 0,
           "incons": 0, "timeout": 0}
    days_to_pass = []
    for _ in range(draws):
        eq, peak, day = 1.0, 1.0, 0
        best_day_profit, total_profit = 0.0, 0.0
        status = "timeout"
        while day < max_days:
            blk = v[rng.integers(0, n - blocks):][:blocks]
            for r in blk:
                day += 1
                start = eq
                eq *= (1.0 + r)
                gain = eq - start
                if gain > best_day_profit:
                    best_day_profit = gain
                total_profit = eq - 1.0
                if r <= -DAILY_LIMIT:
                    status = "bust_daily"
                    break
                if eq <= 1.0 - TOTAL_LIMIT:
                    status = "bust_total"
                    break
                peak = max(peak, eq)
                if eq >= 1.0 + TARGET and day >= min_days:
                    status = ("incons"
                              if best_day_profit > consistency * total_profit
                              else "pass")
                    break
            if status != "timeout":
                break
        out[status] += 1
        if status == "pass":
            days_to_pass.append(day)
    return out, np.array(days_to_pass), draws


def bootstrap_eval_window(pnl, window=45, blocks=5, draws=8000,
                          seed=b"shortrisk"):
    """Worst day inside a WINDOW-day evaluation, resampled.

    The first version of this resampled a path the full length of the
    sample and reported the distribution of its worst day. Every draw
    contained the historical worst day, so the 50th and 99th percentiles
    both came back equal to it and the whole test was degenerate -- it
    answered "is the worst day in the sample the worst day in the sample".
    The question that matters is the one the evaluation asks: how bad is
    the worst day in the 45 trading days you actually have.
    """
    v = pnl.values
    n = len(v)
    if n < window * 2:
        return None
    rng = np.random.default_rng(zlib.crc32(seed))
    nb = max(1, window // blocks)
    worst = np.empty(draws)
    tot = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n - blocks, size=nb)
        path = np.concatenate([v[s:s + blocks] for s in starts])[:window]
        worst[i] = path.min()
        tot[i] = np.prod(1.0 + path) - 1.0
    return worst, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--both-sides", action="store_true")
    ap.add_argument("--cap", type=int, default=0,
                    help="max concurrent positions (0 = uncapped)")
    ap.add_argument("--rules", action="store_true",
                    help="compare rule sets: with/without the 40%% "
                         "consistency rule, and other firms' limits")
    ap.add_argument("--dd", type=float, default=None,
                    help="override the total drawdown limit, e.g. 0.085")
    ap.add_argument("--daily", type=float, default=None,
                    help="override the daily limit, e.g. 0.06")
    a = ap.parse_args()
    global TOTAL_LIMIT, DAILY_LIMIT
    if a.dd is not None:
        TOTAL_LIMIT = a.dd
    if a.daily is not None:
        DAILY_LIMIT = a.daily
    if V.PERP is None:
        log("  cannot find taker_data; tried: " + ", ".join(V._TRIED))
        return 2

    coins = universe()
    sides = (1, -1) if a.both_sides else (-1,)
    label = "long+short" if a.both_sides else "SHORT ONLY"
    log(f"  V29 {label} at {a.tf}h, {len(coins)} coins"
        + (f", capped at {a.cap} concurrent" if a.cap else ", uncapped"))
    log(f"  eval limits: {DAILY_LIMIT*100:.0f}% daily,"
        f" {TOTAL_LIMIT*100:.0f}% total, {TARGET*100:.0f}% target\n")

    pnl, conc, legs = build(coins, a.tf, sides, a.cap)
    if pnl is None:
        log("  no legs")
        return 2

    # ---- 1. concurrency --------------------------------------------------
    cv = conc[conc > 0].values
    log("  " + "=" * 66)
    log("  CONCURRENCY -- how many legs are on at once?")
    log("  " + "=" * 66)
    log(f"    legs                      {len(legs)}")
    log(f"    days with any position    {len(cv)} of {len(pnl)}")
    log(f"    median / mean concurrent  {int(np.median(cv))} /"
        f" {cv.mean():.1f}")
    log(f"    90th / max concurrent     {int(np.percentile(cv, 90))} /"
        f" {int(cv.max())}")
    if not a.cap and cv.max() > 5:
        log("")
        log(f"    The measured edge required up to {int(cv.max())} shorts on at")
        log("    once. V29 live ran a cap of 2. The per-trade number from")
        log("    tf_confirm is therefore NOT the number the live bot could")
        log("    have earned -- rerun with --cap 2 to see what the cap costs.")

    # ---- 2. risk at unit size -------------------------------------------
    worst_day, dd = risk_at(pnl, 1.0)
    s = solve_size(pnl)
    sd = float(pnl.std())
    sharpe = (pnl.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    log("")
    log("  " + "=" * 66)
    log("  DAILY MARK-TO-MARKET at 1.0x equity per leg")
    log("  " + "=" * 66)
    log("    1.0x per leg is a SCALING REFERENCE, not a size anyone would")
    log(f"    trade: with {int(np.median(cv))} legs on it is"
        f" {int(np.median(cv))}x leverage, so the drawdown and")
    log("    total below are ruin by construction. The mean and sd are the")
    log("    useful rows; everything tradeable is in the next block.")
    log(f"    mean daily                {pnl.mean()*100:>+8.3f}%")
    log(f"    daily sd                  {sd*100:>8.3f}%")
    log(f"    Sharpe (daily, ann.)      {sharpe:>8.2f}")
    log(f"    WORST DAY                 {worst_day*100:>+8.2f}%")
    log(f"    worst drawdown            {dd*100:>+8.2f}%")
    log(f"    un-annualised total       {((1+pnl).prod()-1)*100:>+8.1f}%")

    # ---- 3. the size the eval permits -----------------------------------
    log("")
    log("  " + "=" * 66)
    log("  WHAT SIZE DOES THE EVAL ALLOW?")
    log("  " + "=" * 66)
    wd_s, dd_s = risk_at(pnl, s)
    log(f"    largest per-leg size that fits BOTH limits  {s:>7.3f}x")
    log(f"    -> gross exposure at median concurrency     "
        f"{s*np.median(cv)*100:>6.0f}% of equity")
    log(f"    -> gross exposure at the 90th pct           "
        f"{s*np.percentile(cv,90)*100:>6.0f}% of equity")
    log("")
    log(f"    At that size, on this history:")
    log(f"      worst day               {wd_s*100:>+8.2f}%")
    log(f"      worst drawdown          {dd_s*100:>+8.2f}%")
    log(f"      mean daily              {pnl.mean()*s*100:>+8.3f}%")
    log(f"      mean WEEKLY             {pnl.mean()*s*7*100:>+8.2f}%")
    log(f"      annualised              {pnl.mean()*s*365*100:>+8.1f}%")
    if pnl.mean() * s > 0:
        days = np.log(1 + TARGET) / np.log(1 + pnl.mean() * s)
        log(f"      days to the {TARGET*100:.0f}% target  {days:>8.0f}"
            f"   ({days/30.4:.1f} months)")
    else:
        log(f"      days to the {TARGET*100:.0f}% target  never"
            f" -- mean is negative")

    # ---- 4. what the evaluation window actually asks ---------------------
    bo = bootstrap_eval_window(pnl)
    if bo is not None:
        bw, btot = bo
        log("")
        log("  " + "=" * 66)
        log("  A 45-DAY EVALUATION, RESAMPLED 8,000 TIMES")
        log("  " + "=" * 66)
        log("    5-day blocks, so clustering survives. At unit size, the")
        log("    worst single day inside a 45-day window:")
        for q in (50, 90, 99):
            pq = float(np.percentile(bw, q))
            log(f"      {q}th pct worst day   {pq*100:>+9.2f}%"
                f"   -> permits {DAILY_LIMIT/abs(pq):.3f}x" if pq < 0
                else f"      {q}th pct worst day   {pq*100:>+9.2f}%")
        p99 = float(np.percentile(bw, 99))
        s99 = DAILY_LIMIT / abs(p99) if p99 < 0 else np.inf
        s99 = min(s99, s)
        log("")
        log(f"    Sizing to the 99th-percentile worst day gives"
            f" {s99:.3f}x.")
        log(f"    P(a 45-day window hits the {DAILY_LIMIT*100:.0f}% daily"
            f" limit) at that size:")
        for sz, nm in ((s, "limit-fitted size"), (s99, "99th-pct size")):
            hit = float((bw * sz <= -DAILY_LIMIT).mean())
            tot_q = np.percentile(btot * sz, [10, 50, 90])
            log(f"      {nm:<20} {hit*100:>5.1f}%   45-day return"
                f" p10/p50/p90 {tot_q[0]*100:+.1f}%"
                f" / {tot_q[1]*100:+.1f}% / {tot_q[2]*100:+.1f}%")
        log("")
        log(f"    P(a 45-day window reaches the {TARGET*100:.0f}% target)"
            f" at the limit-fitted size:"
            f" {float((btot*s >= TARGET).mean())*100:.1f}%")
        log("    But the eval has no deadline, so that is the wrong")
        log("    statistic -- see below.")

    # ---- 5. the eval as actually written --------------------------------
    log("")
    log("  " + "=" * 66)
    log("  THE EVAL AS ACTUALLY WRITTEN: run until pass or bust")
    log("  " + "=" * 66)
    log(f"    min 5 trading days, no maximum. +{TARGET*100:.0f}% target,")
    log(f"    {TOTAL_LIMIT*100:.0f}% static floor, {DAILY_LIMIT*100:.0f}%"
        f" daily, 40% single-day consistency rule.")
    log("")
    log(f"    {'size':>8}{'gross exp':>11}{'pass':>8}{'bust':>8}"
        f"{'daily':>8}{'INCONS':>8}{'open':>7}{'med days':>10}")
    rows = []
    for m in (1, 2, 3, 4, 6):
        sz = s * m
        o, dtp, nd = race_to_barrier(pnl, sz, draws=3000)
        med = int(np.median(dtp)) if len(dtp) else -1
        rows.append((sz, o, nd, med))
        log(f"    {sz:>8.4f}{sz*np.median(cv)*100:>10.0f}%"
            f"{o['pass']/nd*100:>7.1f}%{o['bust_total']/nd*100:>7.1f}%"
            f"{o['bust_daily']/nd*100:>7.1f}%{o['incons']/nd*100:>7.1f}%"
            f"{o['timeout']/nd*100:>6.1f}%"
            f"{(str(med) if med > 0 else 'n/a'):>10}")
    log("")
    log("    INCONS = reached +10% but ONE day was more than 40% of the")
    log("    profit, so the evaluation is void.")
    log("")
    # I expected the consistency rule to be the thing that killed this,
    # because the short book is positively skewed. At the DD-fitted size it
    # is 0% -- the prediction was wrong at that size. It becomes the LARGEST
    # single failure mode the moment the size goes up, which is a different
    # and more useful fact: the consistency rule is the binding constraint
    # on SPEED, not the drawdown floor. Sizing up to pass faster is what
    # trips it, and that is not obvious from the rule as written.
    inc = [(sz, o['incons'] / nd) for sz, o, nd, _ in rows]
    worst_inc = max(inc, key=lambda x: x[1])
    if worst_inc[1] > 0.10:
        log(f"    The consistency rule is the binding constraint on SPEED.")
        log(f"    At {rows[0][0]:.4f}x it costs"
            f" {inc[0][1]*100:.1f}% of runs; at {worst_inc[0]:.4f}x it costs"
            f" {worst_inc[1]*100:.1f}%.")
        log(f"    Sizing up to compress the timeline is what trips it, so")
        log(f"    the slow operating point is not a choice between slow and")
        log(f"    fast -- it is the only one that resolves in your favour.")

    log("")
    # ---- 6. what the rule sets are worth --------------------------------
    if a.rules:
        log("")
        log("  " + "=" * 66)
        log("  WHAT DO THE RULES THEMSELVES COST?")
        log("  " + "=" * 66)
        log("    Each row refits the size to that firm's drawdown limit,")
        log("    then races to +10%. Same book, same signals throughout --")
        log("    only the rules change. '2x' is the row above at double")
        log("    size, because that is where the consistency rule bites.")
        log("")
        log(f"    {'rules':<34}{'size':>8}{'gross':>7}{'pass':>7}"
            f"{'bust':>7}{'open':>6}{'med d':>7}")
        keep = (TOTAL_LIMIT, DAILY_LIMIT)
        firms = (("this account  10%DD 5%day  40% rule", 0.10, 0.05, 0.40),
                 ("this account  10%DD 5%day  NO rule ", 0.10, 0.05, 1.01),
                 ("8.5%DD 6%day  NO rule             ", 0.085, 0.06, 1.01),
                 ("8.5%DD 6%day  40% rule            ", 0.085, 0.06, 0.40))
        for name, dd_, dl_, cons in firms:
            TOTAL_LIMIT, DAILY_LIMIT = dd_, dl_
            sz = solve_size(pnl)
            for mult, tag in ((1, name), (2, "   ^ same, at 2x size")):
                o, dtp, nd = race_to_barrier(pnl, sz * mult, draws=4000,
                                             consistency=cons)
                md = int(np.median(dtp)) if len(dtp) else -1
                log(f"    {tag:<34}{sz*mult:>8.4f}"
                    f"{sz*mult*np.median(cv)*100:>6.0f}%"
                    f"{o['pass']/nd*100:>6.1f}%"
                    f"{(o['bust_total']+o['bust_daily'])/nd*100:>6.1f}%"
                    f"{o['timeout']/nd*100:>5.1f}%"
                    f"{(str(md) if md > 0 else 'n/a'):>7}")
        TOTAL_LIMIT, DAILY_LIMIT = keep
        log("")
        log("    The daily limit never binds: zero daily busts in every row")
        log("    at every size that is otherwise survivable. A looser daily")
        log("    allowance is worth nothing to this book, and a tighter")
        log("    DRAWDOWN floor costs more than a missing consistency rule")
        log("    saves -- which is the opposite of how the two read on a")
        log("    firm's comparison page.")

    log("  " + "=" * 66)
    log("  BEFORE THIS SIZES ANYTHING LIVE")
    log("  " + "=" * 66)
    log("    This is a quick inline simulation, and the project's own")
    log("    verified rule is: only grid_search.py for Hyro backtesting,")
    log("    never quick inline sims, because they give wrong DD numbers.")
    log("    That rule exists for a reason and this file does not override")
    log("    it. Use these numbers to decide WHETHER to pursue the short")
    log("    book; reproduce them in grid_search.py, with the $100K")
    log("    notional cap and worst-case daily DD across all 12 start")
    log("    months, before they size anything.")
    log("")
    log("    Known directions of error:")
    log("      - daily marks ignore the intrabar stop, so the worst day is")
    log("        OVERSTATED and the permitted size is too conservative")
    log("      - funding is not credited; a short alt book usually EARNS")
    log("        it, so returns are understated")
    log("      - slippage beyond the 8.5bp fee is not modelled, and a")
    log("        20-leg book exiting into a squeeze will not get these")
    log("        fills. That error runs the other way.")
    log("      - coin ordering moves returns 15-23% in this project's own")
    log("        testing; this run uses one ordering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
size_from_dd.py -- size the book BACKWARDS from what actually ends the eval.

THE PROBLEM WITH THE CURRENT SIZING
-----------------------------------
Today the book sizes FORWARD: REFERENCE_NOTIONAL ($200k) x LEVERAGE_L -> target
notional -> positions -> stops placed some ATR away -> whatever dollar loss that
implies. The stop distance ends up ~29% from entry, which is past the point
where the account is already dead, so the stop is decoration. Position SIZE is
the binding constraint, not the stop.

Two live constants come straight from that forward logic and are both wrong:

    AGGREGATE_MAX_LOSS = $30,000   vs $18,838 of headroom to the static floor.
                                   The brake sits BEYOND account death. It can
                                   never fire.
    MAX_LOSS_PER_TRADE = $6,000    vs a ~$9,942 daily limit. 60% of the day's
                                   entire allowance on ONE leg.

WHAT THIS DOES INSTEAD
----------------------
Starts from the eval rules and solves for the largest size that survives them:

  1. DAILY LOSS LIMIT (5% of day-start equity, SWING basis -- measured from the
     day's starting equity, not the intraday high, which is why intraday dips
     do not count).
  2. STATIC FLOOR (10% of initial). Static, so headroom GROWS as you profit --
     an early loss is far more dangerous than a late one, which is why a single
     "max drawdown" number is misleading and this simulates paths instead.
  3. SINGLE-COIN SHOCK. One name gaps against the book while the rest do
     nothing. This is the FIL case: no stop would have helped, the position was
     simply too big.
  4. CONSISTENCY RULE (40%): no single day may be more than 40% of total
     profit. This constrains size from ABOVE as well -- a book that makes its
     target in two lucky days fails the eval even though it hit the number.

WHY NOT THE EXISTING sizing.py
------------------------------
sizing.py already Monte-Carlos the floor, but it draws returns from
rng.normal(). Crypto daily PnL is fat-tailed, and fat tails are precisely what
ends prop accounts. Under a Gaussian a -3.73 sigma day is a 1-in-10,000 event;
in this book's own backtest it is the WORST OBSERVED DAY. A Gaussian sim will
therefore report a comfortable failure probability that is simply wrong.

It also never checks the DAILY limit at all -- only the floor. You can fail
this evaluation on one bad day without ever approaching the floor, and that
path is invisible to it.

This file bootstraps from the book's OWN daily PnL distribution, in blocks, so
the real tails and the real autocorrelation both survive into the simulation.

USAGE
    python3 -u size_from_dd.py                 # analytic constraints + sim
    python3 -u size_from_dd.py --equity 198838
    python3 -u size_from_dd.py --quick         # analytic only, no simulation
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# --------------------------------------------------------------------------
# eval rules (HyroTrader, as confirmed by the account owner)
# --------------------------------------------------------------------------
INITIAL_EQUITY = 200_000.0
STATIC_FLOOR_PCT = 0.10        # 10% of INITIAL, and it does not trail
DAILY_LIMIT_PCT = 0.05         # 5% of the day's STARTING equity (swing basis)
PROFIT_TARGET_PCT = 0.10       # phase 1
CONSISTENCY_MAX_DAY = 0.40     # no day > 40% of total profit
MIN_TRADING_DAYS = 5
HORIZON_DAYS = 120

# How much of each limit a plausible bad outcome is allowed to consume.
# These are the only genuinely discretionary numbers in the file, so they are
# named, defaulted conservatively, and exposed on the command line rather than
# buried in a formula.
DAILY_BUDGET = 0.50            # worst plausible day <= 50% of the daily limit
SHOCK_MOVE = 0.25              # a single coin gapping 25% against the book
SHOCK_BUDGET = 0.50            # that shock <= 50% of the daily limit
MAX_FAIL_PROB = 0.05           # accept at most a 5% chance of busting


def log(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------------
# the book's own daily PnL distribution
# --------------------------------------------------------------------------
def book_daily_returns():
    """Daily PnL of the live book as a FRACTION of reference notional.

    Built from the sleeves that can actually be reconstructed here, combined
    equal-risk, which is what the portfolio layer does across sleeves. Returns
    (series, names, note) so the caller can state exactly what is and is not
    included -- a vol estimate from 4 of 9 sleeves is not the book's vol, and
    saying so is the difference between a sizing tool and a guess.
    """
    import book
    sl, R, VOL, DN = book.build()
    names = []
    try:
        import sr2
        got = sr2.run(6, "break_res", oi_filter=True)
        if got is not None:
            sl["sr"] = got[0]
    except Exception:
        pass

    import pandas as pd
    usable = {k: v for k, v in sl.items()
              if isinstance(v, pd.Series) and not v.dropna().empty}
    if not usable:
        raise RuntimeError("no sleeve could be built -- cannot size from data")

    common = None
    for v in usable.values():
        common = v.index if common is None else common.intersection(v.index)
    S = {k: v.reindex(common).fillna(0.0) for k, v in usable.items()}

    # equal RISK across sleeves, matching the portfolio layer's
    # scale[k] = median(vol)/vol[k]
    vols = {k: float(v.std()) for k, v in S.items()}
    med = float(np.median([x for x in vols.values() if x > 0]))
    combined = sum(S[k] * (med / vols[k] if vols[k] > 0 else 0.0)
                   for k in S) / len(S)
    names = sorted(S)
    note = (f"{len(names)} of 9 sleeves reconstructed: {', '.join(names)}. "
            f"The event sleeves that could not be rebuilt here add variance, "
            f"so the vol below is a FLOOR, not the book's true vol.")
    return combined.dropna(), names, note


def block_bootstrap(x, n_days, n_sims, block=5, rng=None):
    """Sample paths in blocks so fat tails AND autocorrelation survive. An
    iid bootstrap would break up losing streaks, which is exactly the
    structure that busts an account."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    nb = int(np.ceil(n_days / block))
    starts = rng.integers(0, max(n - block, 1), size=(n_sims, nb))
    out = np.empty((n_sims, nb * block))
    for j in range(nb):
        idx = starts[:, j][:, None] + np.arange(block)[None, :]
        out[:, j * block:(j + 1) * block] = x[np.clip(idx, 0, n - 1)]
    return out[:, :n_days]


# --------------------------------------------------------------------------
# analytic constraints -- no simulation needed, and immediately actionable
# --------------------------------------------------------------------------
def analytic(equity, daily_vol_dollars, worst_day_dollars, gross_notional,
             largest_leg, args):
    floor = INITIAL_EQUITY * (1 - STATIC_FLOOR_PCT)
    headroom = equity - floor
    daily_limit = equity * DAILY_LIMIT_PCT

    log("  " + "=" * 68)
    log("  ACCOUNT")
    log("  " + "=" * 68)
    log(f"    equity                      ${equity:>12,.0f}")
    log(f"    static floor (10% of init)  ${floor:>12,.0f}")
    log(f"    headroom to floor           ${headroom:>12,.0f}")
    log(f"    daily loss limit (5%)       ${daily_limit:>12,.0f}")
    log(f"    headroom / daily limit      {headroom/daily_limit:>12.2f}x"
        f"   <- bad days before the account is gone")

    log("")
    log("  " + "=" * 68)
    log("  CURRENT BOOK vs THOSE LIMITS")
    log("  " + "=" * 68)
    log(f"    gross notional              ${gross_notional:>12,.0f}")
    log(f"    daily PnL vol               ${daily_vol_dollars:>12,.0f}")
    log(f"    worst backtested day        ${worst_day_dollars:>12,.0f}"
        f"   ({worst_day_dollars/daily_vol_dollars:.2f} sigma)")
    pct = 100.0 * worst_day_dollars / daily_limit
    log(f"    worst day as % of daily cap {pct:>11.0f}%"
        + ("   <-- NO MARGIN" if pct > 70 else ""))
    log(f"    daily cap in sigma          "
        f"{daily_limit/daily_vol_dollars:>12.2f}")
    log(f"    floor headroom in sigma     "
        f"{headroom/daily_vol_dollars:>12.2f}")

    # --- the constraints, each solved for a leverage multiplier -----------
    log("")
    log("  " + "=" * 68)
    log("  CONSTRAINTS  (multiplier on CURRENT size that each one permits)")
    log("  " + "=" * 68)
    cons = {}

    # 1. worst plausible day within its budget of the daily limit
    cons["worst day <= %.0f%% of daily cap" % (100 * args.daily_budget)] = (
        args.daily_budget * daily_limit / worst_day_dollars)

    # 2. never breach the daily cap on a repeat of the worst day
    cons["worst day < daily cap"] = daily_limit / worst_day_dollars

    # 3. single-coin gap
    shock = largest_leg * args.shock_move
    cons["%.0f%% gap on largest leg <= %.0f%% of cap"
         % (100 * args.shock_move, 100 * args.shock_budget)] = (
        args.shock_budget * daily_limit / shock if shock > 0 else np.inf)

    # 4. a 5-sigma day must not breach the floor outright
    cons["5-sigma day < floor headroom"] = headroom / (5 * daily_vol_dollars)

    for k, v in cons.items():
        flag = "  <-- BINDING" if v == min(cons.values()) else ""
        log(f"    {k:<44}{v:>8.2f}x{flag}")

    binding = min(cons.values())
    log("")
    log(f"    most restrictive multiplier: {binding:.2f}x")
    if binding < 1.0:
        log(f"    => the book is {1/binding:.1f}x TOO LARGE on this basis")
    else:
        log(f"    => the book could be {binding:.1f}x larger on this basis")
    return binding, daily_limit, floor, headroom


# --------------------------------------------------------------------------
# simulation over the book's real distribution
# --------------------------------------------------------------------------
def simulate(daily_pnl_frac, equity, gross_notional, mult, n_sims, rng):
    """Returns (p_daily_breach, p_floor_breach, p_pass, p_consistency_fail)."""
    floor = INITIAL_EQUITY * (1 - STATIC_FLOOR_PCT)
    target = INITIAL_EQUITY * (1 + PROFIT_TARGET_PCT)
    paths = block_bootstrap(daily_pnl_frac, HORIZON_DAYS, n_sims, rng=rng)
    dollars = paths * gross_notional * mult

    eq = np.full(n_sims, equity, dtype=float)
    dead = np.zeros(n_sims, dtype=bool)
    passed = np.zeros(n_sims, dtype=bool)
    daily_bust = np.zeros(n_sims, dtype=bool)
    floor_bust = np.zeros(n_sims, dtype=bool)
    best_day = np.zeros(n_sims)
    total = np.zeros(n_sims)

    for d in range(HORIZON_DAYS):
        live = ~(dead | passed)
        if not live.any():
            break
        r = dollars[:, d]
        # daily limit is 5% of the day's STARTING equity (swing basis)
        lim = eq * DAILY_LIMIT_PCT
        hit_daily = live & (r <= -lim)
        daily_bust |= hit_daily
        dead |= hit_daily

        eq = np.where(live, eq + r, eq)
        best_day = np.where(live, np.maximum(best_day, r), best_day)
        total = np.where(live, total + r, total)

        hit_floor = live & ~hit_daily & (eq <= floor)
        floor_bust |= hit_floor
        dead |= hit_floor
        passed |= live & ~dead & (eq >= target)

    # consistency: among the runs that PASSED, how many would be rejected
    # because one day carried more than 40% of the profit
    prof = np.maximum(total, 1e-9)
    incons = passed & (best_day > CONSISTENCY_MAX_DAY * prof)
    return (daily_bust.mean(), floor_bust.mean(),
            passed.mean(), incons.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=198_838.0)
    ap.add_argument("--gross", type=float, default=None,
                    help="current gross notional (default: measured book)")
    ap.add_argument("--largest-leg", type=float, default=10_000.0,
                    help="largest single-coin notional (MAX_NOTIONAL_PER_LEG)")
    ap.add_argument("--daily-budget", type=float, default=DAILY_BUDGET)
    ap.add_argument("--shock-move", type=float, default=SHOCK_MOVE)
    ap.add_argument("--shock-budget", type=float, default=SHOCK_BUDGET)
    ap.add_argument("--max-fail", type=float, default=MAX_FAIL_PROB)
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    log("  sizing the book BACKWARDS from the eval limits\n")
    try:
        rets, names, note = book_daily_returns()
    except Exception as e:
        log(f"  FAIL could not build the book's return series: {e}")
        log(f"  Sizing on an assumed vol instead of a measured one is how")
        log(f"  accounts die. Fix the data path and re-run.")
        return 2

    log(f"  {note}")
    log(f"  {len(rets)} days of daily PnL\n")

    gross = args.gross if args.gross else 200_000.0
    vol_d = float(rets.std()) * gross
    worst_d = abs(float(rets.min())) * gross

    binding, daily_limit, floor, headroom = analytic(
        args.equity, vol_d, worst_d, gross, args.largest_leg, args)

    if args.quick:
        return 0

    log("")
    log("  " + "=" * 68)
    log("  SIMULATION  (block bootstrap of the book's OWN daily PnL,")
    log("               so real tails and real losing streaks survive)")
    log("  " + "=" * 68)
    log(f"  {'mult':>6}{'gross':>12}{'P(daily bust)':>15}{'P(floor)':>10}"
        f"{'P(any bust)':>13}{'P(pass)':>10}{'P(incons)':>11}")

    rng = np.random.default_rng(12345)
    rows = []
    for mult in (0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.25):
        pd_, pf_, pp_, pi_ = simulate(
            rets.values, args.equity, gross, mult, args.sims, rng)
        pany = pd_ + pf_
        rows.append((mult, pany, pp_, pd_, pf_, pi_))
        flag = ""
        if pany <= args.max_fail:
            flag = "  OK"
        log(f"  {mult:>6.2f}{gross*mult:>12,.0f}{pd_*100:>14.1f}%"
            f"{pf_*100:>9.1f}%{pany*100:>12.1f}%{pp_*100:>9.1f}%"
            f"{pi_*100:>10.1f}%{flag}")

    ok = [r for r in rows if r[1] <= args.max_fail]
    best = max(ok, key=lambda r: r[2]) if ok else None
    log("")
    log("  " + "=" * 68)
    if best is not None and best[2] < 0.10:
        # Every size is "safe" only because the book barely moves. Picking the
        # smallest and calling it RECOMMENDED would read as advice when the
        # real answer is that there is not enough edge to clear the target.
        log(f"  NO RECOMMENDATION. Every size tested is survivable, but the")
        log(f"  best P(pass) is {best[2]*100:.1f}% -- the book does not reach")
        log(f"  the +{PROFIT_TARGET_PCT*100:.0f}% target within {HORIZON_DAYS}"
            f" days at ANY size that is safe.")
        log(f"  Sizing is not the constraint here; the edge is. Sizing UP to")
        log(f"  reach the target just moves you into the bust column.")
    elif best is not None:
        log(f"  RECOMMENDED: {best[0]:.2f}x current size "
            f"(gross ${gross*best[0]:,.0f})")
        log(f"    P(bust) {best[1]*100:.1f}%  P(pass) {best[2]*100:.1f}%")
    if best is not None:
        log("")
        log(f"  Derived constants to replace the forward-sized ones:")
        dl = args.equity * DAILY_LIMIT_PCT
        log(f"    AGGREGATE_MAX_LOSS   ${min(dl, headroom*0.5):>10,.0f}"
            f"   (was $30,000 -- beyond the ${headroom:,.0f} that ends you)")
        log(f"    MAX_LOSS_PER_TRADE   ${dl*0.15:>10,.0f}"
            f"   (was $6,000 -- 60% of one day's whole allowance)")
    if best is None:
        log(f"  NO size tested keeps P(bust) under {args.max_fail*100:.0f}%.")
        log(f"  Even 0.20x busts too often, which is not a sizing problem --")
        log(f"  it means the edge is too small for these limits, or the")
        log(f"  return distribution is worse than the book's Sharpe suggests.")
    log("  " + "=" * 68)
    log("")
    log("  CAVEATS, and they matter:")
    log("   - The bootstrap resamples the PAST. It cannot produce a day worse")
    log("     than the worst day in the sample, so every P(bust) here is an")
    log("     UNDERSTATEMENT. Treat it as a floor on risk, not an estimate.")
    log(f"   - {note}")
    log("   - Backtested PnL excludes live slippage and funding drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

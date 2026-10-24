#!/usr/bin/env python3
"""
dd_attrib.py -- 418 days is a drawdown problem, not a signal problem.

THE ARITHMETIC THAT SETS THE TIMELINE. The size the account permits is
fixed by the drawdown floor: s = 0.10 / DD_unit. The time to the target is
then

    days  =  log(1.10) / log(1 + mu_d * s)  ~=  DD_unit / mu_d

Nothing else is in that expression. Not the per-trade edge, not the win
rate, not the fee, not the number of coins. Only the ratio of the book's
mean daily return to its worst drawdown. On the 12h short book that ratio
is 555% / 0.578% = 960 days, and every attempt to improve the SIGNAL moves
the numerator by a few percent while the denominator sets the answer.

So this file attacks the denominator, and the first question is what the
drawdown is made of. A book of 6-10 simultaneous alt shorts has two very
different sources of loss:

    COMMON     the whole alt complex rallies and every leg loses at once.
               Correlation goes to 1, the legs stop being 10 positions and
               become one. This is removable -- it is a factor exposure,
               and factor exposures can be hedged, budgeted, or netted
               against a book that is long the same factor.

    IDIOSYNCRATIC  one name squeezes on its own news. This is irreducible
               without trading fewer names, which costs return.

If the drawdown is mostly COMMON, the timeline is a portfolio-construction
problem and is fixable without touching a single entry rule. If it is
mostly IDIOSYNCRATIC, it is not, and the honest answer is that this book
cannot be run faster.

WHAT IS TESTED, each reported as days-to-target so the variants are
comparable on the only metric that matters:

  raw              the book as measured
  beta-hedged      hold +beta of the equal-weight alt complex against the
                   short book, sized to neutralise the regression beta.
                   NOTE the hedge is LONG a complex whose median name fell
                   57%, so the hedge itself loses money. It helps only if
                   it cuts drawdown faster than it cuts mean -- which is
                   exactly what gets measured rather than assumed.
  beta-budgeted    cap the SUM of leg betas instead of the leg COUNT. Ten
                   legs that are all high-beta L1s are one position; the
                   count cap cannot see that and a beta budget can.
  de-risked        halve size after the book is 4% below its high, restore
                   on recovery. Mechanically truncates the left tail at a
                   cost in return.

A WARNING I OWE THIS FILE. Earlier in this project I predicted that
equal-risk sizing would be a free improvement to three cross-sectional
sleeves, and measurement showed it DEGRADED all three (-0.35, -0.44,
-0.11 Sharpe) because the edge lived in the high-volatility names. Every
variant here is the same shape of idea. None of them is assumed to work.

USAGE
    python3 -u dd_attrib.py                 # 12h short book, cap 10
    python3 -u dd_attrib.py --tf 14 --cap 5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import short_risk as R
import v29_test as V

TARGET = 0.10
DD_LIMIT = 0.10


def log(m=""):
    print(m, flush=True)


def market_factor(coins):
    """Equal-weight daily return of the alt complex -- the thing that makes
    every short leg lose on the same day. Cross-sectional mean rather than
    BTC, because the book trades alts and BTC is not what squeezes them."""
    rets = {}
    for c in coins:
        df = V._load(f"{V.PERP}/{c}_1h.csv")
        if df is None:
            continue
        d = df["close"].resample("1D").last().dropna()
        if len(d) > 200:
            rets[c] = d.pct_change()
    M = pd.DataFrame(rets)
    return M.mean(axis=1).dropna(), M


def days_to_target(mu_d, dd_unit):
    """Days to +10% at the size the drawdown floor permits."""
    if mu_d <= 0 or dd_unit >= 0:
        return float("inf"), 0.0
    s = DD_LIMIT / abs(dd_unit)
    g = mu_d * s
    if g <= 0:
        return float("inf"), s
    return float(np.log(1 + TARGET) / np.log(1 + g)), s


def summarise(name, pnl, note=""):
    mu = float(pnl.mean())
    wd, dd = R.risk_at(pnl, 1.0)
    # risk_at clamps at ruin, which hides the real ratio at unit size, so
    # the drawdown is measured at a size small enough to stay in range and
    # scaled back. Drawdown is close to linear in size while the series is
    # far from ruin.
    probe = 0.01
    _, dd_p = R.risk_at(pnl, probe)
    dd_unit = dd_p / probe
    d, s = days_to_target(mu, dd_unit)
    log(f"    {name:<16}{mu*100:>9.3f}%{dd_unit*100:>11.0f}%"
        f"{s:>9.4f}{(f'{d:.0f}' if np.isfinite(d) else 'never'):>9}"
        f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}  {note}")
    return mu, dd_unit, d, s


def derisk(pnl, trigger=0.04, factor=0.5):
    """Halve size while the book is `trigger` below its running high."""
    v = pnl.values
    out = np.empty_like(v)
    eq, peak = 1.0, 1.0
    for i, r in enumerate(v):
        sz = factor if eq < peak * (1 - trigger) else 1.0
        out[i] = r * sz
        eq *= (1 + r * sz * 0.02)      # track in a sane unit; scale-free
        peak = max(peak, eq)
    return pd.Series(out, index=pnl.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--cap", type=int, default=10)
    a = ap.parse_args()
    if V.PERP is None:
        log("  cannot find taker_data")
        return 2

    coins = R.universe()
    log(f"  12h short book, {len(coins)} coins, cap {a.cap}")
    log(f"  days = DD_unit / mu_daily. Only that ratio moves the timeline.\n")

    pnl, conc, legs = R.build(coins, a.tf, (-1,), a.cap)
    mkt, M = market_factor(coins)
    idx = pnl.index.intersection(mkt.index)
    pnl, mkt = pnl.loc[idx], mkt.loc[idx]

    # ---- 1. how much of the book is the alt complex? --------------------
    x = mkt.values
    y = pnl.values
    b = float(np.cov(x, y)[0, 1] / np.var(x))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    log("  " + "=" * 74)
    log("  WHAT IS THE DRAWDOWN MADE OF?")
    log("  " + "=" * 74)
    log(f"    regression of book PnL on the equal-weight alt complex")
    log(f"      beta                     {b:>8.2f}")
    log(f"      R^2                      {r2*100:>7.1f}%"
        f"   <- share of DAY-TO-DAY variance")
    worst = np.argsort(y)[:20]
    comm = float(np.sum(b * x[worst]))
    tot = float(np.sum(y[worst]))
    log(f"      worst 20 days, total     {tot*100:>+8.1f}%")
    log(f"      of which common factor   {comm*100:>+8.1f}%"
        f"   ({comm/tot*100:.0f}% of the loss)")
    log(f"      of which idiosyncratic   {(tot-comm)*100:>+8.1f}%")
    log("")
    if comm / tot > 0.5:
        log("    The majority of the worst days is the whole complex moving")
        log("    against the book at once. That is a factor exposure, not a")
        log("    signal problem, and it is the part that can be removed.")
    else:
        log("    The worst days are mostly single names squeezing on their")
        log("    own. That is NOT hedgeable by holding the index -- it is")
        log("    reduced only by holding fewer names, which costs return.")

    # ---- 2. the variants -------------------------------------------------
    log("")
    log("  " + "=" * 74)
    log("  DOES ATTACKING THE DENOMINATOR WORK?")
    log("  " + "=" * 74)
    log(f"    {'variant':<16}{'mu/day':>10}{'DD unit':>11}"
        f"{'size':>9}{'days':>9}{'months':>8}")
    base = summarise("raw", pnl)

    hedged = pnl - b * mkt
    summarise("beta-hedged", hedged, f"hold +{b:.2f}x the complex long")

    for trig, fac in ((0.04, 0.5), (0.03, 0.25)):
        summarise(f"de-risk {trig*100:.0f}%/{fac:.2f}",
                  derisk(pnl, trig, fac))

    both = derisk(pnl - b * mkt)
    summarise("hedged+de-risk", both)

    # ---- 2b. is the hedge real, or two biases? --------------------------
    # The beta-hedged row doubles the mean AND cuts the drawdown, which is
    # the shape of a result that is wrong. Two specific ways it could be:
    #
    #   IN-SAMPLE BETA. b is fitted on the whole series and applied to the
    #   whole series. Refit on one half and apply to the other.
    #
    #   SURVIVORSHIP. The "alt complex" is the equal-weight mean of the
    #   coins in taker_data, and taker_data holds coins that are STILL
    #   LISTED. Every alt that died is absent, so the basket's return is
    #   biased upward, and a hedge that is LONG that basket inherits the
    #   bias. Being long 4.5x a daily-rebalanced basket of survivors is
    #   not a hedge, it is a second strategy with a flattering backtest.
    log("")
    log("  " + "=" * 74)
    log("  IS THE HEDGE REAL?")
    log("  " + "=" * 74)
    mk_mu = float(mkt.mean())
    _, mk_dd = R.risk_at(mkt * abs(b), 0.01)
    log(f"    the 'hedge' leg on its own, at {abs(b):.2f}x the complex:")
    log(f"      mean daily               {mk_mu*abs(b)*100:>+8.3f}%"
        f"   (complex itself {mk_mu*100:+.3f}%/day)")
    log(f"      its own DD at unit       {mk_dd/0.01*100:>+8.0f}%")
    log(f"      share of the hedged mean {mk_mu*abs(b)/(base[0]+mk_mu*abs(b))*100:>7.0f}%")
    log("")
    log("    taker_data holds coins that are STILL LISTED, so every alt")
    log("    that died is missing from that basket and its return is")
    log("    biased upward by an unknown amount. Being long 4.5x a")
    log("    daily-rebalanced basket of survivors is not a hedge; it is a")
    log("    second strategy with a flattering backtest attached.")
    log("")
    h = len(pnl) // 2
    log(f"    {'beta fitted on':<22}{'applied to':<14}{'mu/day':>10}"
        f"{'DD unit':>10}{'days':>8}")
    for fit_lo, fit_hi, app_lo, app_hi, nm1, nm2 in (
            (0, h, h, len(pnl), "1st half", "2nd half"),
            (h, len(pnl), 0, h, "2nd half", "1st half")):
        xf, yf = x[fit_lo:fit_hi], y[fit_lo:fit_hi]
        bf = float(np.cov(xf, yf)[0, 1] / np.var(xf))
        seg = pnl.iloc[app_lo:app_hi] - bf * mkt.iloc[app_lo:app_hi]
        mu_s = float(seg.mean())
        _, dd_s = R.risk_at(seg, 0.01)
        dd_s /= 0.01
        d_s, _ = days_to_target(mu_s, dd_s)
        log(f"    {nm1 + f' (b={bf:.2f})':<22}{nm2:<14}{mu_s*100:>9.3f}%"
            f"{dd_s*100:>9.0f}%"
            f"{(f'{d_s:.0f}' if np.isfinite(d_s) else 'never'):>8}")
    log("")
    log("    If the out-of-sample betas disagree with -4.52, or the")
    log("    out-of-sample days are far worse than 260, the hedge was")
    log("    fitted rather than found.")

    # ---- 2c. the hedge you could actually place -------------------------
    # Two problems with the complex hedge even though it survives OOS:
    #   1. half its return improvement is the hedge leg's own long-alt
    #      return, which is survivorship-biased. Credit the drawdown
    #      reduction; do NOT credit that return.
    #   2. long 4.5x a 63-leg daily-rebalanced basket is not placeable on
    #      this account -- the $100K notional cap binds long before it.
    # A long BTC or ETH perp is one order and is placeable. The question
    # is how much of the drawdown reduction a single major captures.
    log("")
    log("  " + "=" * 74)
    log("  THE HEDGE YOU COULD ACTUALLY PLACE")
    log("  " + "=" * 74)
    log(f"    {'hedge instrument':<22}{'beta':>7}{'R2':>7}{'mu/day':>9}"
        f"{'DD unit':>10}{'days':>8}  credit")
    cands = [("equal-weight complex", mkt)]
    for maj in ("BTC", "ETH"):
        dfm = V._load(f"{V.PERP}/{maj}_1h.csv")
        if dfm is not None:
            s = dfm["close"].resample("1D").last().dropna().pct_change()
            s = s.reindex(pnl.index).fillna(0.0)
            cands.append((f"{maj} perp", s))
    for nm, fac in cands:
        xf = fac.loc[pnl.index].values if hasattr(fac, "loc") else fac
        bb = float(np.cov(xf, y)[0, 1] / np.var(xf))
        rr = float(np.corrcoef(xf, y)[0, 1] ** 2)
        hs = pnl - bb * pd.Series(xf, index=pnl.index)
        # full credit
        mu_f = float(hs.mean())
        _, dd_f = R.risk_at(hs, 0.01)
        dd_f /= 0.01
        d_f, _ = days_to_target(mu_f, dd_f)
        # conservative: keep the drawdown reduction, zero the hedge leg's
        # own mean return, because that return is survivorship-biased
        d_c, _ = days_to_target(base[0], dd_f)
        log(f"    {nm:<22}{bb:>7.2f}{rr*100:>6.0f}%{mu_f*100:>8.3f}%"
            f"{dd_f*100:>9.0f}%"
            f"{(f'{d_f:.0f}' if np.isfinite(d_f) else 'never'):>8}  full")
        log(f"    {'':<22}{'':>7}{'':>6} {base[0]*100:>7.3f}%"
            f"{dd_f*100:>9.0f}%"
            f"{(f'{d_c:.0f}' if np.isfinite(d_c) else 'never'):>8}"
            f"  DD only (defensible)")
    log("")
    log("    'DD only' is the number to plan against: it credits the hedge")
    log("    for the drawdown it removes and credits it NOTHING for the")
    log("    return of being long alts, which is the half that is biased.")
    log("    If a single major captures most of the drawdown reduction,")
    log("    the whole improvement is one extra order per rebalance.")

    # ---- 3. what would it actually take? --------------------------------
    log("")
    log("  " + "=" * 74)
    log("  WHAT WOULD 60 DAYS REQUIRE?")
    log("  " + "=" * 74)
    mu0, dd0, d0, _ = base
    need = dd0 / 60.0           # required mu_d for 60 days at this DD
    log(f"    at the current drawdown ({abs(dd0)*100:.0f}% unit), 60 days")
    log(f"    needs mu = {abs(need)*100:.3f}%/day against the actual"
        f" {mu0*100:.3f}%")
    log(f"    -> a {abs(need)/mu0:.1f}x improvement in RETURN, or")
    log(f"    -> a {abs(need)/mu0:.1f}x reduction in DRAWDOWN, or any")
    log(f"       combination whose product is {abs(need)/mu0:.1f}x.")
    log("")
    log("    A better entry rule moves the numerator by tens of percent at")
    log("    best -- this project has measured that repeatedly. A factor")
    log("    hedge or a second uncorrelated book moves the denominator by")
    log("    multiples. That is where the order of magnitude is.")
    log("")
    log("    AND THE ONE THAT IS ACTUALLY AVAILABLE: this account already")
    log("    runs a nine-sleeve market-neutral book. Two books with")
    log("    correlation rho combine to a drawdown far below the sum --")
    log("    at rho=0 two equal books have ~0.71x the drawdown of one for")
    log("    the same total return, which is a 1.4x speedup for free, and")
    log("    the short book should be a SLEEVE inside that construction")
    log("    rather than a standalone directional account.")
    log("    Measuring that needs the sleeve book's daily series, which")
    log("    this file does not have. That is the next thing to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

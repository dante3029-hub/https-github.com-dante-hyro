#!/usr/bin/env python3
"""
daily_stop.py -- buy size by truncating the worst day.

THE PROBLEM, measured. At the current size the book's worst day in 1,273
days is -4.36% against a 5% daily limit. Nothing busts. Double the size
and NINE days are fatal; triple it and thirty-two are. The bust
breakdown at 3x is 36.4% daily against 9.2% drawdown-floor, four to one.

So the binding constraint is not the drawdown any more. It is a single
day. And that reframes what to build: you do not need a better average
day, you need the left tail clipped. Improve only the worst twenty days
and the size cap moves.

THE ASYMMETRY NOBODY NOTICED. The book already has a daily PROFIT cap --
flatten at +3.64% to satisfy the consistency rule. It has no daily LOSS
cap. The account can lose an unbounded amount in a session and the only
thing stopping it is that no day in this sample happened to be bad
enough. That is not a risk control, it is luck that has held so far.

THE RULE. Flatten everything when the book is down L% from the day's
STARTING equity, and stand down until tomorrow. The firm measures the
daily limit on a swing basis -- from day-start equity, not the intraday
high -- so a day-start-referenced stop maps exactly onto the rule being
enforced.

Truncating the worst day at L means the worst day IS L by construction,
and the permitted size becomes 5%/L instead of 5%/4.36%. The cost is
the days you stop out of that would have recovered.

THE HONEST PART, AND IT MATTERS. The daily series here is close to
close, so it cannot see intraday paths. Truncating it at -L assumes you
only ever stop on days that CLOSED below -L -- which flatters the rule
badly, because in reality you also stop on days that dipped through -L
and recovered by the close. Those are pure cost.

So the intraday low is modelled as

    est_low = close - c * sd_daily

and the stop fires when est_low <= -L, paying -(L + slip). c = 0 is the
dishonest close-only version and is shown ONLY as the upper bound.
c = 0.5 and c = 1.0 are the ones to read. slip is charged because you
do not get out at exactly -L in a fast market.

RESULT: THE DAILY STOP IS REJECTED, AND FINDING OUT WHY EXPOSED A
BIGGER ERROR OF MINE.

Under trailing-drawdown sizing the stop buys nothing: the worst day
falls 4.36% -> 1.70% and the permitted size does not move, because the
DRAWDOWN floor is what binds at that size, not the daily limit.

But the floor I was enforcing is the wrong one. HyroTrader's rule is a
STATIC 10% from the STARTING balance, not 10% peak-to-trough. Once the
account is up 5% a static floor leaves 15% of room; a trailing one
still leaves 10%. solve_size has been enforcing the harsher rule all
along, so every size in this project was too conservative.

Correcting it moves the binding constraint to the DAILY limit (worst
day lands exactly on -5.00%) and allows 0.0186 instead of 0.0162.

With the static floor the stop DOES buy size -- up to 3.1x at a 4%
stop -- but the pass rate collapses, because a bigger size means a bad
early sequence breaches the floor before any profit cushion exists:

  stop   size   vs base   pass   floor bust   med days
  none  0.0186    1.00x  86.6%         8.6%        61
  2.0%  0.0309    1.66x  70.5%        29.5%        34
  3.0%  0.0438    2.35x  60.3%        39.7%        23
  4.0%  0.0582    3.13x  49.8%        50.2%        17

So the stop is a way to buy speed with pass probability, not a free
risk control. Rejected on those terms.

WHAT SURVIVED instead: volatility targeting, which was never tested
because an earlier REACTIVE de-risk (cut after a 4% drawdown) failed
so badly. Predictive is not reactive -- volatility clusters, and
scaling by trailing vol cuts exposure before the bad stretch rather
than after it.

USAGE
    python3 -u daily_stop.py                  # the (stop, size) surface
    python3 -u daily_stop.py --c 1.0 --slip 0.004
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import short_risk as SR


def log(m=""):
    print(m, flush=True)


def apply_daily_stop(x, L, c=0.5, slip=0.002):
    """Flatten at -L from day-start. `x` MUST already be in return units.

    FIRST VERSION WAS UNIT-WRONG and the output was obvious nonsense:
    99.8% pass, 2 days to target, size 0.24. I passed the NORMALISED
    book (sd ~ 1.0 in normalised units) and compared it against a 4%
    stop, so est_low = v - c*sd was hugely negative almost everywhere
    and the rule fired on 41-88% of all days. A percentage threshold
    against a unitless series.

    The fix is not only the units. The stop and the size are mutually
    dependent -- stopping at -L changes the worst day, which changes the
    permitted size, which changes what -L means as a fraction of the
    daily series. So the stop has to be applied INSIDE the size search,
    not to a normalised series beforehand. See solve_size_with_stop.

    c scales the assumed intraday excursion below the close, so days
    that dipped through -L and recovered by the close still pay the
    stop. Without it the rule flatters itself badly.
    """
    if L is None or L <= 0:
        return x, 0
    v = np.asarray(x, dtype=float)
    sd = float(np.std(v))
    hit = (v - c * sd) <= -L
    out = v.copy()
    out[hit] = -(L + slip)
    idx = x.index if hasattr(x, "index") else None
    return (pd.Series(out, index=idx) if idx is not None else out), int(hit.sum())


def solve_size_with_stop(X, L, c, slip):
    """Largest per-leg size whose STOPPED series fits both eval limits.

    Bisection, with the stop applied to the scaled series at every
    candidate size, because the two are circular.
    """
    def ok(s):
        xs, _ = apply_daily_stop(X * s, L, c, slip)
        wd, dd = SR.risk_at(xs, 1.0)
        return wd >= -SR.DAILY_LIMIT and dd >= -SR.TOTAL_LIMIT
    lo, hi = 0.0, 5.0
    if ok(hi):
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def best_book():
    """Four verified sleeves (filtered) at 60% + V29 shorts at 40%."""
    import book
    import dominance as DM
    import regime as RG
    idx, PX, R, VOL, DN = book.panel()
    A, B, j = RG.books()
    cands = RG.candidates(idx, list(R.columns))
    h2 = cands["H2 |taker imbalance|"].reindex(j)
    dom, _, _ = DM.dominance_series(R, idx, force_proxy=True)
    dm = dom.rolling(5).sum().shift(1).reindex(j)
    h = len(j) // 2
    mask = ((dm > dm.iloc[:h].quantile(0.80)).fillna(False)
            | (h2 > h2.iloc[:h].quantile(0.80)).fillna(False))
    Af = A.copy()
    Af[mask] = 0.0
    return 0.6 * Af + 0.4 * B


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=None,
                    help="intraday excursion in sd (default: sweep)")
    ap.add_argument("--slip", type=float, default=0.002)
    ap.add_argument("--draws", type=int, default=3000)
    a = ap.parse_args()
    import survivors as SV

    X = best_book()
    base = SR.solve_size(X)
    wd0, dd0 = SR.risk_at(X, base)
    log(f"\n  book: filtered sleeves 60% + V29 shorts 40%")
    log(f"  no daily stop: size {base:.4f}, worst day {wd0*100:+.2f}%, "
        f"DD {dd0*100:+.1f}%, {SV.days_of(X)[0]:.0f} days\n")

    cs = [a.c] if a.c is not None else [0.0, 0.5, 1.0]
    cap = SR.derived_day_cap()
    for c in cs:
        log(f"  {'='*72}")
        log(f"  intraday excursion c = {c}"
            + ("   <- close-only, UPPER BOUND, not a real number"
               if c == 0 else "")
            + f"   slip {a.slip*100:.1f}%")
        log(f"  {'='*72}")
        log(f"  {'stop':>6}{'stops':>7}{'size':>8}{'worst d':>9}{'DD':>7}"
            f"{'days':>7}{'pass':>7}{'daily':>7}{'floor':>7}{'med d':>7}")
        for L in (None, 0.040, 0.030, 0.025, 0.020, 0.015):
            s = (SR.solve_size(X) if L is None
                 else solve_size_with_stop(X, L, c, a.slip))
            if not np.isfinite(s) or s <= 0:
                continue
            xs, n = apply_daily_stop(X * s, L, c, a.slip)
            wd, dd = SR.risk_at(xs, 1.0)
            o, dtp, nd = SR.race_to_barrier(xs, 1.0, draws=a.draws,
                                            day_cap=cap)
            med = int(np.median(dtp)) if len(dtp) else -1
            d = SV.days_of(xs)[0]
            lbl = "none" if L is None else f"{L*100:.1f}%"
            log(f"  {lbl:>6}{n:>7}{s:>8.4f}{wd*100:>8.2f}%{dd*100:>6.1f}%"
                f"{(f'{d:.0f}' if np.isfinite(d) else '--'):>7}"
                f"{o['pass']/nd*100:>6.1f}%{o['bust_daily']/nd*100:>6.1f}%"
                f"{o['bust_total']/nd*100:>6.1f}%{med:>7}")
        log("")
    log("  'stops' is how many of the 1,273 days the rule fires on.")
    log("  Read the c=0.5 and c=1.0 blocks. c=0 cannot happen: it assumes")
    log("  you never stop out of a day that recovered by the close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

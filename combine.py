#!/usr/bin/env python3
"""
combine.py -- the only lever left is correlation.

THE ARITHMETIC, restated because it is the whole argument. The size the
account permits is set by the drawdown floor, s = 0.10 / DD_unit, so

    days to +10%  ~=  DD_unit / mu_daily

and that ratio is SCALE-INVARIANT: double the size and both terms double.
So the timeline cannot be improved by sizing, only by changing the shape
of the return stream. Measured so far on the 12h short book:

    raw                                 775 days
    alt-complex hedge, DD credit only   492 days   (1.6x, and the
                                                    hedge is not placeable
                                                    under a $100K cap)
    drawdown de-risking                4935 days   (much worse)
    a better entry rule                 tens of percent, per this
                                        project's own repeated measurement

Getting to 60 days needs 13x. Hedging found 1.6x. The remaining factor
has to come from somewhere structural, and there is exactly one structural
thing left: this account already runs a market-neutral cross-sectional
book, and two return streams with correlation rho combine to

    sd_blend  =  sqrt(w^2 sd_a^2 + (1-w)^2 sd_b^2 + 2 w(1-w) rho sd_a sd_b)

while the mean blends linearly. At rho = 0 and equal volatilities the
blend has 0.71x the volatility for the average of the two means. If the
two books have similar mu/sd, that is a 1.4x improvement in the ratio for
nothing. At rho < 0 it is more. That is the number this file measures,
and if rho turns out to be strongly POSITIVE the idea is dead and the
short book has nothing to add to what is already running.

WHAT IS AND IS NOT BEING COMBINED. The live book has nine sleeves. This
rebuilds the five that book.build() and sr2 produce -- delta, relvol,
skew, cascade, sr -- and NOT oirank, pattern, srflip or fvg, which live in
other modules with different interfaces. So:

  - this is a PARTIAL replica of the live book, and its absolute Sharpe
    is not the live book's Sharpe. Do not quote it as such.
  - the quantity being measured is the CORRELATION between a
    cross-sectional market-neutral stream and a directional short stream,
    and a correlation estimated from five of nine sleeves is a reasonable
    estimate of the correlation of all nine, because the missing sleeves
    are the same kind of object as the present ones.
  - audit note: of the nine, only relvol, oirank, pattern and sr survived
    their own tests. delta, skew, fvg and srflip were static-tilt or
    indistinguishable from noise, and cascade was idle. relvol and sr are
    in this replica; oirank and pattern are not. So the replica contains
    two of the four real sleeves and three of the five dead ones, which
    means its mean is UNDERSTATED relative to a book of only the
    survivors.

Reported either way, because the correlation is the point and the level
is not.

USAGE
    python3 -u combine.py
    python3 -u combine.py --tf 14 --cap 5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

TARGET = 0.10
DD_LIMIT = 0.10


def log(m=""):
    print(m, flush=True)


def ratio_days(x):
    """days-to-target for a daily return series, scale-invariant.

    days = log(1.10) / log(1 + mu * s) with s = 0.10/DD. Computed at a
    probe size small enough that the drawdown is measured in the linear
    regime rather than near ruin.
    """
    import short_risk as R
    x = x.dropna()
    mu = float(x.mean())
    if mu <= 0:
        return float("inf"), 0.0, mu
    probe = 0.01 / max(x.std(), 1e-12)        # scale to ~1% daily sd
    xs_ = x * probe
    _, dd = R.risk_at(xs_, 1.0)
    if dd >= 0:
        return float("inf"), 0.0, mu
    mu_s = mu * probe
    s = DD_LIMIT / abs(dd)
    g = mu_s * s
    if g <= 0:
        return float("inf"), dd, mu
    return float(np.log(1 + TARGET) / np.log(1 + g)), dd, mu_s


def norm(x, target_sd=0.01):
    """Rescale to a common daily volatility so the blend weights mean what
    they look like. days-to-target is scale-invariant, so this changes
    nothing about the conclusion."""
    x = x.dropna()
    sd = float(x.std())
    return x * (target_sd / sd) if sd > 0 else x


def sleeve_book():
    """Daily series of the cross-sectional book, as far as it rebuilds."""
    import book
    sl, R_, VOL, DN = book.build()
    # sr2.run can return (None, ...) rather than raising -- it did here,
    # and assigning that into the dict produced an AttributeError three
    # frames later on 'NoneType' has no attribute 'index'. Validate what
    # comes back instead of trusting the call not to throw.
    try:
        import sr2
        got = sr2.run(6, "break_res", oi_filter=True)
        cand = got[0] if isinstance(got, tuple) else got
        if isinstance(cand, pd.Series) and len(cand) > 100:
            sl["sr"] = cand
        else:
            log(f"    (sr2.run returned {type(cand).__name__}, not a usable"
                f" series -- continuing without the sr sleeve)")
    except Exception as e:
        log(f"    (sr2 unavailable: {type(e).__name__}: {e};"
            f" continuing without the sr sleeve)")
    sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
    if not sl:
        raise RuntimeError("no sleeve series could be built")
    idx = None
    for v in sl.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    S = {k: v.reindex(idx).fillna(0.0) for k, v in sl.items()}
    return S, pd.Series(np.mean([v.values for v in S.values()], axis=0),
                        index=idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--cap", type=int, default=10)
    a = ap.parse_args()

    import short_risk as R
    import v29_test as V

    log(f"  combining the cross-sectional book with the {a.tf}h short book")
    log(f"  metric throughout: days to +10% = DD_unit / mu_daily,")
    log(f"  which is scale-invariant, so sizing cannot change it.\n")

    log("  building the cross-sectional sleeves...")
    S, sleeves = sleeve_book()
    log(f"    {len(S)} sleeves: {', '.join(S)}")
    log(f"    {len(sleeves)} days "
        f"({sleeves.index[0].date()} -> {sleeves.index[-1].date()})")

    log("  building the short book...")
    coins = R.universe()
    shorts, conc, legs = R.build(coins, a.tf, (-1,), a.cap)
    log(f"    {len(legs)} legs, {len(shorts)} days")

    idx = sleeves.index.intersection(shorts.index)
    if len(idx) < 300:
        log(f"  only {len(idx)} overlapping days -- aborting")
        return 2
    A = norm(sleeves.loc[idx])          # cross-sectional book
    B = norm(shorts.loc[idx])           # short book
    log(f"    {len(idx)} overlapping days "
        f"({idx[0].date()} -> {idx[-1].date()})")

    # ---- 1. the number the whole idea rests on --------------------------
    rho = float(np.corrcoef(A.values, B.values)[0, 1])
    log("")
    log("  " + "=" * 70)
    log("  THE CORRELATION")
    log("  " + "=" * 70)
    log(f"    rho(cross-sectional book, short book)   {rho:>+7.3f}")
    log("")
    log(f"    {'sleeve':<12}{'rho to short book':>20}")
    for k, v in S.items():
        vv = v.reindex(idx).fillna(0.0)
        if vv.std() > 0:
            log(f"    {k:<12}"
                f"{float(np.corrcoef(vv.values, B.values)[0,1]):>20.3f}")
    log("")
    if rho > 0.3:
        log("    Strongly positive. The short book is already largely")
        log("    inside what the sleeve book earns, and adding it buys")
        log("    little. The idea is dead and this is where it stops.")
    elif abs(rho) <= 0.3:
        log("    Near zero. These are genuinely different return streams,")
        log("    which is the condition the blend below needs.")
    else:
        log("    Negative. Better than independent -- the blend reduces")
        log("    drawdown by more than diversification alone would give.")

    # ---- 2. the blend ----------------------------------------------------
    log("")
    log("  " + "=" * 70)
    log("  THE BLEND, both normalised to the same daily volatility")
    log("  " + "=" * 70)
    log(f"    {'w(short)':>10}{'mu/day':>10}{'sd/day':>9}{'DD unit':>10}"
        f"{'days':>8}{'months':>8}")
    best = (None, float("inf"))
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        x = (1 - w) * A + w * B
        d, dd, mu = ratio_days(x)
        log(f"    {w:>10.1f}{mu*100:>9.3f}%{float(x.std())*100:>8.3f}%"
            f"{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")
        if d < best[1]:
            best = (w, d)
    d0, _, _ = ratio_days(A)
    d1, _, _ = ratio_days(B)
    log("")
    log(f"    cross-sectional book alone   "
        f"{(f'{d0:.0f} days' if np.isfinite(d0) else 'never')}")
    log(f"    short book alone             "
        f"{(f'{d1:.0f} days' if np.isfinite(d1) else 'never')}")
    if best[0] is not None and np.isfinite(best[1]):
        log(f"    best blend  w={best[0]:.1f}            {best[1]:.0f} days")
        ref = min(d0, d1)
        if np.isfinite(ref) and ref > 0:
            log(f"    improvement over the better one alone  "
                f"{ref/best[1]:.2f}x")
    log("")
    log("    HALF-SAMPLE CHECK on the best blend, because a blend weight")
    log("    chosen on the whole sample is a fitted parameter:")
    if best[0] is not None:
        x = (1 - best[0]) * A + best[0] * B
        h = len(x) // 2
        for nm, seg in (("1st half", x.iloc[:h]), ("2nd half", x.iloc[h:])):
            d_, dd_, mu_ = ratio_days(seg)
            log(f"      {nm:<10}{mu_*100:>9.3f}%{dd_*100:>9.0f}%"
                f"{(f'{d_:.0f}' if np.isfinite(d_) else 'never'):>8} days")
        # and the weight refit per half
        for nm, lo, hi in (("fit 1st, use 2nd", 0, h),
                           ("fit 2nd, use 1st", h, len(x))):
            bw, bd = None, float("inf")
            for w in np.arange(0, 1.01, 0.1):
                dd_, _, _ = ratio_days((1 - w) * A.iloc[lo:hi]
                                       + w * B.iloc[lo:hi])
                if dd_ < bd:
                    bw, bd = w, dd_
            oth = slice(h, len(x)) if lo == 0 else slice(0, h)
            d_o, _, _ = ratio_days((1 - bw) * A.iloc[oth] + bw * B.iloc[oth])
            log(f"      {nm:<18}w={bw:.1f}  -> "
                f"{(f'{d_o:.0f}' if np.isfinite(d_o) else 'never')} days"
                f" out of sample")

    log("")
    log("  " + "=" * 70)
    log("  WHAT THIS IS AND IS NOT")
    log("  " + "=" * 70)
    log("    This rebuilds 5 of the 9 live sleeves. Of the 4 that survived")
    log("    their own audits (relvol, oirank, pattern, sr) it contains 2,")
    log("    and it carries 3 of the 5 that did not. So the LEVEL here")
    log("    understates a book of only the survivors, and the absolute")
    log("    Sharpe is not the live book's.")
    log("    The CORRELATION is the measurement that matters and it does")
    log("    not depend on the level. If it is near zero, blending is the")
    log("    one remaining multiple-sized lever; if it is strongly")
    log("    positive, the short book adds nothing to what already runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

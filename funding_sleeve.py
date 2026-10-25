#!/usr/bin/env python3
"""
funding_sleeve.py -- cross-sectional funding carry as a fifth sleeve.

THE TRADE. Perpetual funding is paid by longs to shorts when the rate is
positive and the other way when negative. Measured on this universe it is
positive 79.2% of 8-hour periods, averaging +0.0066%, which annualises to
about +7.3% to a flat short. So: short the coins paying the most funding,
long the ones paying least, equal risk, market neutral. The book collects
the DISPERSION in funding rather than its level, which is what makes it a
cross-sectional sleeve rather than a directional carry bet.

Built on book.xs so it is directly comparable to relvol and oirank --
same rebalance machinery, same fee model, same universe, same window.

TWO QUESTIONS, AND THEY ARE DIFFERENT:

  PRICE-ONLY   does funding PREDICT price returns? A cross-sectional
               signal like any other. If high-funding coins subsequently
               fall, funding is an information signal and the carry is a
               bonus on top.

  CARRY-TOTAL  what does the trade actually earn? Price return PLUS the
               funding received or paid. This is the real PnL and it is
               the number that belongs next to the other sleeves.

Reporting only the second would hide whether the sleeve works or whether
it is just harvesting a structural payment. Both are printed.

SIGN CONVENTION, which is the easiest thing here to get backwards and
has already cost this project three commits once. A position w (positive
= long) pays funding at rate r when r > 0, so

    funding PnL = -w * r

Short a positive-funding coin (w < 0, r > 0) and -w*r is positive: you
are paid. The selftest asserts exactly that case rather than trusting
the algebra.

And because book.xs sorts ASCENDING and longs the TOP of the sort, the
signal fed to it is NEGATIVE funding -- so the top of the sort is the
most negative funding (long) and the bottom is the highest (short).
Getting that backwards produces a sleeve that pays funding instead of
collecting it, and it would still look like a plausible return series.

PARITY. xs_carry() duplicates book.xs so it can accumulate carry, which
means it can drift from it. With carry disabled it must reproduce
book.xs to floating-point equality, and --selftest checks that before
any number is reported.

VERDICT, MEASURED: NOT A FIFTH SLEEVE.

    price only      mu 0.011%/day   Sharpe 0.21   2,668 days
    price + carry   mu 0.024%/day   Sharpe 0.46   1,132 days
    carry alone     mu 0.017%/day   Sharpe 14.68

Two findings, and they point opposite ways.

The carry is nearly riskless. Sharpe 14.68, because the funding payment
is known in advance and arrives on schedule -- there is almost no
variance in it. But it is 0.017%/day, about 6% a year.

The price signal is worthless. Sharpe 0.21. High-funding coins do not
subsequently underperform in this sample, so funding carries no
cross-sectional information about returns; it is a payment, not a
prediction.

The problem is that you cannot collect the payment without holding the
positions, and the price variance of those positions is roughly seventy
times the carry's. A Sharpe-14.68 component drowns inside a Sharpe-0.21
one and the combination lands at 0.46.

Blended into the four verified sleeves plus the short book it makes
things monotonically WORSE -- 122 days at zero weight, 123 at 0.05, 140
at 0.20, 162 at 0.30 -- because its standalone ratio is nine times worse
than the base book's and rho to the short book is +0.268 rather than
negative. Half-samples are 486 and 2,514 days, so it is unstable too.

WHERE THE FUNDING VALUE ACTUALLY IS: already inside the short book,
uncounted. Those legs are short 79% positive-funding coins and earn
+0.0992% per trade in funding, a +7.9% uplift on the 1.251% gross edge
that the backtest currently discards. That is the real finding here --
not a new sleeve, a correction to an existing number.

USAGE
    python3 -u funding_sleeve.py --selftest
    python3 -u funding_sleeve.py --blend
    python3 -u funding_sleeve.py
    python3 -u funding_sleeve.py --lookback 3 --n 5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import book

FEE = book.FEE


def log(m=""):
    print(m, flush=True)


def load_funding(index, coins):
    """Daily funding rate per coin, summed over the 8-hour periods and
    aligned to the price index. Missing coins are dropped loudly."""
    out, missing = {}, []
    for c in coins:
        p = f"funding_data/{c}_funding.csv"
        if not os.path.exists(p):
            missing.append(c)
            continue
        d = pd.read_csv(p)
        d["ts"] = pd.to_datetime(d["ts"], unit="ms")
        s = d.set_index("ts")["rate"].sort_index()
        out[c] = s.resample("1D").sum().reindex(index).fillna(0.0)
    if missing:
        log(f"    no funding data for {len(missing)}: "
            f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    return pd.DataFrame(out)


def xs_carry(R, FUND, sig, n=5, hold=7, carry=True):
    """book.xs, plus the funding leg.

    Mirrors book.xs exactly: `hold` overlapping tranches, each rebalanced
    every `hold` days, weights +0.5/n on the top n of the sorted signal
    and -0.5/n on the bottom n, turnover charged at FEE. The only
    addition is the funding accrual on the held weights.
    """
    idx = R.index
    acc = np.zeros(len(idx))
    F = FUND.reindex(columns=R.columns).fillna(0.0) if carry else None
    for ph in range(hold):
        wp = pd.Series(0.0, index=R.columns)
        out = np.zeros(len(idx))
        for t in range(60, len(idx)):
            r = R.iloc[t].fillna(0.0)
            # funding accrues on the weights held INTO day t
            fund = (-float((wp * F.iloc[t]).sum()) if carry else 0.0)
            caryy = float((wp * r).sum()) + fund
            if (t - ph) % hold != 0:
                out[t] = caryy
                continue
            s = sig(t)
            v = s.dropna()
            if len(v) < 2 * n + 2:
                out[t] = caryy
                continue
            o = v.sort_values()
            w = pd.Series(0.0, index=R.columns)
            w[o.index[-n:]] = 0.5 / n
            w[o.index[:n]] = -0.5 / n
            out[t] = (float((w * r).sum()) + fund
                      - float((w - wp).abs().sum()) * FEE)
            wp = w
        acc += out
    return pd.Series(acc[60:] / hold, index=idx[60:])


def build(lookback=3, n=5, hold=7):
    idx, PX, R, VOL, DN = book.panel()
    FUND = load_funding(idx, list(R.columns))
    cols = [c for c in R.columns if c in FUND.columns]
    R = R[cols]
    FUND = FUND[cols]

    def z(x):
        return (x - x.mean()) / x.std() if x.std() > 0 else x * 0

    # NEGATIVE funding, because book.xs longs the TOP of the sort:
    # top = most negative funding = the leg we want to be long.
    def sig(t):
        return z(-FUND.iloc[t - lookback:t].mean())

    price_only = xs_carry(R, FUND, sig, n, hold, carry=False)
    total = xs_carry(R, FUND, sig, n, hold, carry=True)
    return price_only, total, R, FUND, sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--blend", action="store_true",
                    help="test it as an addition to the existing book")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import survivors as SV
    log(f"  cross-sectional funding carry: short the highest-funding"
        f" coins, long the lowest")
    log(f"  lookback {a.lookback}d, {a.n} per side, {a.hold}d hold\n")
    price_only, total, R, FUND, _ = build(a.lookback, a.n, a.hold)

    allf = FUND.values.flatten()
    allf = allf[np.isfinite(allf)]
    log(f"    universe {len(R.columns)} coins, {len(R)} days")
    log(f"    daily funding: mean {allf.mean()*100:+.4f}%, "
        f"positive {(allf>0).mean()*100:.1f}% of coin-days\n")

    log(f"    {'variant':<16}{'mu/day':>10}{'DD unit':>10}{'days':>8}"
        f"{'months':>8}{'Sharpe':>9}")
    for nm, s in (("price only", price_only), ("price + carry", total)):
        d, dd, mu, sd = SV.days_of(s)
        sh = (s.mean() / s.std() * np.sqrt(365)) if s.std() > 0 else 0.0
        log(f"    {nm:<16}{mu*100:>9.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}{sh:>9.2f}")
    carry_only = total - price_only
    log(f"    {'carry alone':<16}{carry_only.mean()*100:>9.3f}%"
        f"{'':>10}{'':>8}{'':>8}"
        f"{(carry_only.mean()/carry_only.std()*np.sqrt(365) if carry_only.std()>0 else 0):>9.2f}")

    log("")
    h = len(total) // 2
    log("    half-samples (price + carry):")
    for nm, seg in (("1st", total.iloc[:h]), ("2nd", total.iloc[h:])):
        d, dd, mu, _ = SV.days_of(seg)
        log(f"      {nm:<6}{mu*100:>9.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8} days")

    if a.blend:
        return blend_test(total)

    # correlation against the four verified sleeves
    log("")
    log("    correlation to the verified sleeves:")
    try:
        import event_daily as ED
        import survivors as SV2
        sl, R2, VOL2, DN2 = book.build()
        sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
        oi = SV2.build_oirank(R2, R2.index)
        if oi is not None:
            sl["oirank"] = oi
        for s_ in ("pattern", "sr"):
            v, _ = ED.sleeve_series(s_, quiet=True)
            if v is not None:
                sl[s_] = v
        for k in ("relvol", "oirank", "pattern", "sr"):
            if k not in sl:
                continue
            j = total.index.intersection(sl[k].index)
            if len(j) < 300:
                continue
            log(f"      {k:<10}"
                f"{float(np.corrcoef(total.loc[j], sl[k].loc[j])[0,1]):>+7.3f}")
    except Exception as e:
        log(f"      (skipped: {type(e).__name__}: {e})")
    return 0


def blend_test(fund):
    """Does it ADD to the four verified sleeves + the short book?

    The only question that matters for a candidate sleeve. A standalone
    number is interesting; whether it improves the book you already have
    is the decision.
    """
    import event_daily as ED
    import short_risk as SR
    import survivors as SV
    sl, R, VOL, DN = book.build()
    sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
    oi = SV.build_oirank(R, R.index)
    if oi is not None:
        sl["oirank"] = oi
    for s_ in ("pattern", "sr"):
        v, _ = ED.sleeve_series(s_, quiet=True)
        if v is not None:
            sl[s_] = v
    ver = [k for k in ("relvol", "oirank", "pattern", "sr") if k in sl]
    idx = None
    for k in ver:
        idx = sl[k].index if idx is None else idx.intersection(sl[k].index)
    idx = idx.intersection(fund.index)
    sh, _, _ = SR.build(SR.universe(), 12, (-1,), 10)
    j = idx.intersection(sh.index)

    def nz(s):
        s = s.reindex(j).fillna(0.0)
        return s / s.std() if s.std() > 0 else s
    A = pd.Series(np.mean([nz(sl[k]).values for k in ver], axis=0), index=j)
    A, B, F = nz(A), nz(sh), nz(fund)
    base = 0.6 * A + 0.4 * B
    log("")
    log(f"  rho(funding, 4-sleeve book) {np.corrcoef(F, A)[0,1]:+.3f}"
        f"   rho(funding, short book) {np.corrcoef(F, B)[0,1]:+.3f}")
    log("")
    log(f"  {'funding wt':>11}{'mu/day':>10}{'DD unit':>10}{'days':>8}")
    best = (None, 1e9)
    for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        d, dd, mu, _ = SV.days_of((1 - w) * base + w * F)
        log(f"  {w:>11.2f}{mu*100:>9.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}")
        if d < best[1]:
            best = (w, d)
    log("")
    if best[0] == 0.0:
        log("  REJECTED: every positive weight makes the book worse. Its")
        log("  standalone ratio is far below the base book's and it is not")
        log("  negatively correlated enough to pay for that.")
    else:
        log(f"  best weight {best[0]:.2f} -> {best[1]:.0f} days")
    return 0


def selftest() -> int:
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok = ok and cond
        log(f"    [{'ok' if cond else 'FAIL'}] {name}")

    log("  SIGN: a short in a positive-funding coin must be PAID.")
    w = pd.Series({"A": -0.5})          # short
    r = pd.Series({"A": 0.01})          # positive funding
    pnl = -float((w * r).sum())
    chk(f"short 0.5 at +1% funding -> {pnl:+.4f} (positive)", pnl > 0)
    w = pd.Series({"A": +0.5})
    pnl = -float((w * r).sum())
    chk(f"long 0.5 at +1% funding -> {pnl:+.4f} (negative)", pnl < 0)

    log("")
    log("  PARITY: xs_carry with carry OFF must equal book.xs exactly.")
    idx, PX, R, VOL, DN = book.panel()
    FUND = load_funding(idx, list(R.columns))
    cols = [c for c in R.columns if c in FUND.columns]
    R, FUND = R[cols], FUND[cols]

    def z(x):
        return (x - x.mean()) / x.std() if x.std() > 0 else x * 0

    def sig(t):
        return z(VOL.iloc[t - 1][cols] / VOL.iloc[t - 21:t - 1][cols].mean())

    a = book.xs(R, sig, n=8, hold=7)
    b = xs_carry(R, FUND, sig, n=8, hold=7, carry=False)
    j = a.index.intersection(b.index)
    md = float(np.abs(a.loc[j].values - b.loc[j].values).max())
    chk(f"max |difference| = {md:.2e} over {len(j)} days", md < 1e-12)

    log("")
    log("  DIRECTION: the sleeve must RECEIVE funding on net, not pay it.")
    price_only, total, R2, FUND2, _ = build()
    carry = (total - price_only)
    chk(f"mean carry {carry.mean()*100:+.4f}%/day is positive",
        carry.mean() > 0)

    log("")
    log(f"  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

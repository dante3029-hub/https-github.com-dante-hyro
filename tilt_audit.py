#!/usr/bin/env python3
"""
tilt_audit.py -- how much of each LIVE sleeve is a static coin tilt?

WHY THIS EXISTS
---------------
While screening a spot-flow candidate, --diag's static-tilt test was pointed
at `perp_only k=3` -- a near-copy of the live `delta` sleeve -- and reported
that 88% of its Sharpe is reproduced by holding a FIXED portfolio of its
average weights and never retrading.

If that holds for the real `delta` construction, that sleeve is not a flow
timing signal. It is a three-year bet on being long some coins and short
others, sized and risk-budgeted as though it were a signal.

This matters beyond one sleeve. The book's sleeve correlation was previously
decomposed as ~33% market beta plus "timing overlap". If several
cross-sectional sleeves are largely static coin tilts, the real shared
exposure is THE SAME FIXED COIN BETS, which is a worse explanation: the
correlation matrix would understate how much the sleeves fail together,
because a tilt that stops working stops working for all of them at once.

WHAT IT TESTS
-------------
For each cross-sectional sleeve, using its EXACT live construction copied
from book.build() rather than a paraphrase:

  1. STATIC TILT   hold the average weight vector, never retrade. The share
                   of Sharpe this reproduces is the share that is coin
                   selection rather than timing.
  2. PERSISTENCE   % of rebalances each coin is held, per side.
  3. LEAVE-ONE-OUT (--loo) Sharpe with each coin dropped, so a result
                   resting on one name cannot hide. ZEC has already been
                   found carrying 0.68 Sharpe of a spot variant.

Only the four DENSE CROSS-SECTIONAL sleeves are audited here (delta, relvol,
skew, and oirank when its data is present). The event sleeves (sr, srflip,
pattern, fvg, cascade) fire on their own schedules and hold a handful of
names at a time, so "the average weight vector" is not a portfolio they ever
resemble, and the test would not mean the same thing. They need their own
treatment and do not get a misleading number here.

HONEST LIMITS
-------------
- The static book is a fixed-weight portfolio rebalanced daily to constant
  weights, costed at zero. That FLATTERS the static leg, so a high static
  share is a conservative finding and a low one is not proof of anything.
- One sample, one universe, 3.7 years. A static tilt that paid here is not
  guaranteed to fail forward; it is simply not evidence of a signal.

USAGE
    python3 -u tilt_audit.py            # static-tilt + persistence
    python3 -u tilt_audit.py --loo       # adds leave-one-coin-out (slower)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import book
from book import FEE, stats, xs
from spot_flow import _collect, xs_rw   # fork + equal-risk weights,
                                        # both proven against book.xs


def log(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------------
# EXACT live constructions, copied verbatim from book.build().
# If book.build() changes, these must change with it -- the parity test
# below is what catches the drift.
# --------------------------------------------------------------------------
Z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def live_sleeves(R, VOL, DN):
    """name -> (signal_fn, n_per_side, hold), matching book.build() exactly."""
    return {
        "delta": (
            lambda t: Z(DN.iloc[t - 3:t].sum()
                        / VOL.iloc[t - 3:t].sum().replace(0, np.nan)),
            5, 5),
        "relvol": (
            lambda t: Z(VOL.iloc[t - 1] / VOL.iloc[t - 21:t - 1].mean()),
            8, 7),
        "skew": (
            lambda t: R.iloc[t - 45:t].skew(),
            5, 45),
    }


def parity_check(R, VOL, DN) -> bool:
    """The sleeve definitions above are a COPY. Prove the copy still equals
    what book.build() produces, or every number in this report is about a
    strategy the bot does not run."""
    live, _, _, _ = book.build()
    ok = True
    for name, (sig, n, hold) in live_sleeves(R, VOL, DN).items():
        if name not in live:
            log(f"  {name:<8} MISSING from book.build() -- copy is stale")
            ok = False
            continue
        mine = xs(R, sig, n=n, hold=hold)
        theirs = live[name].reindex(mine.index)
        gap = float((mine - theirs).abs().max())
        tag = "OK" if gap < 1e-12 else f"MISMATCH {gap:.2e}"
        log(f"  {name:<8} matches book.build(): {tag}")
        if gap >= 1e-12:
            ok = False
    return ok


def audit_one(name, R, VOL, DN, sig, n, hold, loo=False):
    series, hist = _collect(R, sig, n, hold)
    ref = xs(R, sig, n=n, hold=hold)
    gap = float((series - ref).abs().max())
    if gap > 1e-12:
        log(f"  {name}: instrumented loop does not reproduce xs "
            f"({gap:.2e}) -- skipping")
        return None

    sh, ann = stats(ref)
    W = pd.DataFrame([h[2] for h in hist])
    wbar = W.mean()
    static = (R.fillna(0.0) * wbar).sum(axis=1).reindex(ref.index).fillna(0.0)
    s_sh, s_ann = stats(static)
    share = (100.0 * max(s_sh, 0.0) / sh) if sh > 0 else float("nan")

    pl = (W > 0).mean() * 100
    ps = (W < 0).mean() * 100
    tilted = int(((pl > 70) | (ps > 70)).sum())

    log("")
    log(f"  {'='*66}")
    log(f"  {name}   Sharpe {sh:.2f}   ann {ann:.1f}%   "
        f"{len(hist)} rebalances   n={n} hold={hold}")
    log(f"  {'='*66}")
    share_txt = (f"-> reproduces {share:.0f}% of Sharpe"
                 if np.isfinite(share)
                 else "-> share undefined (sleeve Sharpe <= 0)")
    log(f"    static book  Sharpe {s_sh:>6.2f}  ann {s_ann:>6.1f}%  "
        f"{share_txt}")
    # with Sharpe <= 0 the "share of Sharpe" is undefined, not small --
    # printing nan% next to "mostly timing" would read as a pass
    verdict = ("n/a (Sharpe<=0)" if not np.isfinite(share) else
               "STATIC TILT" if share > 70 else
               "MIXED" if share > 40 else "mostly timing")
    log(f"    verdict: {verdict}")
    log(f"    coins held >70% on one side: {tilted}")

    top = (pl + ps).sort_values(ascending=False).index[:6]
    log(f"    most-held: " + "  ".join(
        f"{c}({pl[c]:.0f}L/{ps[c]:.0f}S)" for c in top))

    # EQUAL-RISK vs EQUAL-DOLLAR inside the sleeve. book.xs gives every
    # position 0.5/n, so the highest-vol holding dominates sleeve PnL. That
    # is a weighting artifact, not a signal property, and it is the most
    # likely reason one coin (ZEC, repeatedly) can decide a sleeve's result.
    rw = xs_rw(R, sig, n=n, hold=hold)
    rw_sh, rw_ann = stats(rw)
    log(f"    equal-RISK   Sharpe {rw_sh:>6.2f}  ann {rw_ann:>6.1f}%  "
        f"(equal-dollar {sh:.2f}, {rw_sh-sh:+.2f})")

    worst = None
    if loo:
        rows = []
        for c in list(R.columns):
            keep = [x for x in R.columns if x != c]
            s2 = live_sleeves(R[keep], VOL[keep], DN[keep]).get(name)
            if s2 is None:
                break
            sg, nn, hh = s2
            rows.append((c, stats(xs(R[keep], sg, n=nn, hold=hh))[0]))
        if rows:
            rows.sort(key=lambda x: x[1])
            worst = (rows[0][0], sh - rows[0][1])
            log(f"    leave-one-out: dropping {rows[0][0]} costs "
                f"{worst[1]:+.2f} Sharpe "
                f"({', '.join(f'{c} {v:.2f}' for c, v in rows[:3])})")
    return dict(name=name, sharpe=sh, static=s_sh, share=share,
                tilted=tilted, worst=worst, rw=rw_sh)


def main() -> int:
    loo = "--loo" in sys.argv
    idx, PX, R, VOL, DN = book.panel()
    log(f"  {len(R.columns)} coins, {len(idx)} days "
        f"({idx[0].date()} -> {idx[-1].date()})\n")

    log("  PARITY -- the sleeve definitions here are a COPY of book.build();")
    log("  proving the copy is still exact before trusting any number.")
    if not parity_check(R, VOL, DN):
        log("")
        log("  ABORT: a sleeve definition has drifted from book.build().")
        log("  Fix the copy in live_sleeves() first -- otherwise this report")
        log("  describes a strategy the bot does not run.")
        return 2

    out = []
    for name, (sig, n, hold) in live_sleeves(R, VOL, DN).items():
        r = audit_one(name, R, VOL, DN, sig, n, hold, loo=loo)
        if r:
            out.append(r)

    log("")
    log("  " + "=" * 66)
    log("  SUMMARY")
    log("  " + "=" * 66)
    log(f"  {'sleeve':<10}{'$-wt':>8}{'risk-wt':>9}{'static':>8}"
        f"{'share':>7}{'verdict':>15}")
    for r in out:
        fin = np.isfinite(r["share"])
        v = ("n/a (Sh<=0)" if not fin else
             "STATIC TILT" if r["share"] > 70 else
             "MIXED" if r["share"] > 40 else "mostly timing")
        sc = f"{r['share']:>5.0f}%" if fin else f"{'--':>6}"
        log(f"  {r['name']:<10}{r['sharpe']:>8.2f}{r['rw']:>9.2f}"
            f"{r['static']:>8.2f}{sc:>7}{v:>15}")
    gains = [r for r in out if r["rw"] > r["sharpe"] + 0.15]
    if gains:
        log("")
        log(f"  Equal-RISK weighting IMPROVES "
            f"{', '.join(r['name'] for r in gains)}. These sleeves were")
        log(f"  being dominated by their most volatile holding. This is a")
        log(f"  free fix -- no new signal, no extra parameter, just weights")
        log(f"  consistent with what the portfolio layer already does across")
        log(f"  sleeves. Worth applying to the live book.")

    bad = [r for r in out if np.isfinite(r["share"]) and r["share"] > 70]
    mixed = [r for r in out
             if np.isfinite(r["share"]) and 40 < r["share"] <= 70]
    log("")
    if bad:
        log(f"  {len(bad)} sleeve(s) are mostly a STATIC COIN TILT: "
            f"{', '.join(r['name'] for r in bad)}")
        log(f"  These are risk-budgeted as timing signals but are really a")
        log(f"  fixed long/short bet on specific coins over one 3.7-year")
        log(f"  sample. Two consequences worth taking seriously:")
        log(f"    1. The forward Sharpe should not be expected to hold -- a")
        log(f"       characteristic that paid is not a signal that repeats.")
        log(f"    2. Sleeve correlation UNDERSTATES joint risk. Sleeves")
        log(f"       sharing the same fixed tilts fail together, which the")
        log(f"       correlation matrix does not show while the tilt works.")
    elif mixed:
        log(f"  {len(mixed)} sleeve(s) are partly static: "
            f"{', '.join(r['name'] for r in mixed)}")
    else:
        log("  No sleeve is mostly a static tilt.")
    log("")
    log("  NOTE the static book pays ZERO costs here, which flatters it. A")
    log("  high static share is therefore conservative; a low one is not")
    log("  proof that the sleeve is genuinely timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

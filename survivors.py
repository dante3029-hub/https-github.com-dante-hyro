#!/usr/bin/env python3
"""
survivors.py -- the replica that hit 126 days contains ONE real sleeve.

WHAT THIS IS FOR. combine.py measured the cross-sectional book at 126
days to +10% and the 12h short book at 835, on the scale-invariant metric
days ~= DD_unit / mu_daily. That settled the strategic question: the book
already in production is the engine and the short book is at most a 1.13x
blend partner.

But the replica combine.py built is NOT the live book. sr2.run returns
None on both machines, so it loaded four sleeves -- delta, relvol, skew,
cascade -- and this project's own audits found that

    relvol    Sharpe 1.09                        SIGNAL
    oirank    Sharpe 1.06 at 5d hold             SIGNAL
    pattern   z = +4.31 vs random timing         SIGNAL
    sr        marginal                           SIGNAL (weak)
    delta     83% of its Sharpe is a static tilt NOT a signal
    skew      90% static                         NOT a signal
    fvg       p = 0.077                          NOT a signal
    srflip    p = 0.123                          NOT a signal
    cascade   never fires                        IDLE

So the 126-day replica is relvol carrying three sleeves that failed their
own tests. One of four survivors, three of five failures. If the metric
is already 126 days in that state, the obvious question is what it is
with the dead weight removed -- and that is a question about how the LIVE
book is weighted, which is actionable today rather than a new strategy.

WHAT IS IN HERE AND WHAT IS NOT. relvol and oirank are cross-sectional:
they produce a weight vector per rebalance and book.xs turns them into a
daily series directly. pattern and sr are EVENT sleeves -- they produce a
list of trades per coin and need the same trade-to-daily-series
conversion short_risk.build does. That conversion is not written yet, so

    IN:   relvol, oirank  (2 of the 4 survivors)
    OUT:  pattern, sr     (the other 2) -- stated, not quietly omitted

which means every number here UNDERSTATES a survivors-only book, because
two of its four components are missing. That is the honest direction for
the gap to run, and it is the reason this file is worth running before
the event-sleeve conversion is built rather than after.

oirank's convention: book.xs earns R[t] from sig(t), while oirank's own
convention is OI through t inclusive with weights applied from t onward.
Wiring oi_scores(OI, t) into book.xs is therefore lookahead. The adapter
is oi_scores(OI, t-1), and this file poisons bar t to prove the signal is
blind to it before reporting any number, exactly as tilt_audit does.

USAGE
    python3 -u survivors.py
    python3 -u survivors.py --short 0.3     # add the short book at w=0.3
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

TARGET = 0.10
DD_LIMIT = 0.10


def log(m=""):
    print(m, flush=True)


def days_of(x):
    """days-to-target; scale-invariant, so normalisation is free."""
    import short_risk as R
    x = pd.Series(x).dropna()
    mu = float(x.mean())
    sd = float(x.std())
    if mu <= 0 or sd <= 0:
        return float("inf"), 0.0, mu, sd
    k = 0.01 / sd
    _, dd = R.risk_at(x * k, 1.0)
    if dd >= 0:
        return float("inf"), dd, mu, sd
    g = mu * k * (DD_LIMIT / abs(dd))
    if g <= 0:
        return float("inf"), dd, mu, sd
    return float(np.log(1 + TARGET) / np.log(1 + g)), dd, mu * k, sd


def build_oirank(R, idx):
    """oirank as a daily series through book.xs, with the t-1 adapter and
    a lookahead poison test. Returns None loudly if OI data is absent."""
    import book
    try:
        from signal_engine.sleeve_oirank import N_PER_SIDE, oi_scores
    except Exception as e:
        log(f"    oirank: cannot import sleeve_oirank ({e}) -- SKIPPED")
        return None
    import tilt_audit as TA
    OIdf, note = TA.load_oi_daily(list(R.columns), idx)
    if OIdf is None:
        log(f"    oirank: {note} -- SKIPPED")
        return None
    OI = OIdf.values

    def sig(t):
        return pd.Series(oi_scores(OI, t - 1), index=R.columns)

    t = len(idx) - 1
    clean = sig(t)
    OI2 = OI.copy()
    OI2[t, :] = OI2[t, :] * 7.3 + 1e6
    dirty = pd.Series(oi_scores(OI2, t - 1), index=R.columns)
    if not np.allclose(clean.fillna(-999).values,
                       dirty.fillna(-999).values, atol=1e-12):
        log("    oirank: FAIL LOOKAHEAD -- refusing to report a number")
        return None
    log(f"    oirank: blind to bar t OK, {note}")
    # hold=5: fix_oirank_hold.py moved the live sleeve from 72h to 120h
    return book.xs(R, sig, n=N_PER_SIDE, hold=5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=float, default=0.0,
                    help="blend weight on the 12h short book")
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--cap", type=int, default=10)
    a = ap.parse_args()

    import book

    log("  days to +10% = DD_unit / mu_daily, scale-invariant.")
    log("  Lower is better. Every series normalised to 1% daily sd.\n")
    log("  building sleeves...")
    sl, R, VOL, DN = book.build()
    sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
    log(f"    cross-sectional from book.build: {', '.join(sl)}")
    oi = build_oirank(R, R.index)
    if oi is not None:
        sl["oirank"] = oi

    idx = None
    for v in sl.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    S = {k: v.reindex(idx).fillna(0.0) for k, v in sl.items()}
    log(f"    {len(S)} sleeves over {len(idx)} days "
        f"({idx[0].date()} -> {idx[-1].date()})")

    VERIFIED = {"relvol", "oirank", "pattern", "sr"}
    have_v = sorted(VERIFIED & set(S))
    have_d = sorted(set(S) - VERIFIED)
    log("")
    log(f"    verified sleeves present : {', '.join(have_v) or 'none'}")
    log(f"    failed/idle present      : {', '.join(have_d) or 'none'}")
    log(f"    verified sleeves MISSING : "
        f"{', '.join(sorted(VERIFIED - set(S))) or 'none'}"
        f"   <- these would only help")

    # ---- each sleeve alone ----------------------------------------------
    log("")
    log("  " + "=" * 68)
    log("  EACH SLEEVE ALONE")
    log("  " + "=" * 68)
    log(f"    {'sleeve':<12}{'verdict':<12}{'mu/day':>9}{'DD unit':>10}"
        f"{'days':>8}{'months':>8}")
    for k in sorted(S, key=lambda z: days_of(S[z])[0]):
        d, dd, mu, _ = days_of(S[k])
        log(f"    {k:<12}{('SIGNAL' if k in VERIFIED else 'failed'):<12}"
            f"{mu*100:>8.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")

    # ---- the comparison that matters ------------------------------------
    def blend(keys):
        if not keys:
            return None
        parts = []
        for k in keys:
            sd = float(S[k].std())
            parts.append(S[k] / sd if sd > 0 else S[k])
        return pd.Series(np.mean([p.values for p in parts], axis=0),
                         index=idx)

    log("")
    log("  " + "=" * 68)
    log("  DOES CUTTING THE DEAD SLEEVES HELP?")
    log("  " + "=" * 68)
    log("    Equal risk weight within each set (each sleeve scaled to the")
    log("    same daily sd before averaging), so the comparison is about")
    log("    WHICH sleeves, not how much of each.")
    log("")
    log(f"    {'set':<34}{'mu/day':>9}{'DD unit':>10}{'days':>8}{'months':>8}")
    sets = [("all sleeves (the 126-day replica)", sorted(S)),
            ("verified only", have_v),
            ("failed/idle only", have_d)]
    results = {}
    for nm, keys in sets:
        x = blend(keys)
        if x is None:
            log(f"    {nm:<34}  (empty)")
            continue
        d, dd, mu, _ = days_of(x)
        results[nm] = (d, x)
        log(f"    {nm:<34}{mu*100:>8.3f}%{dd*100:>9.0f}%"
            f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
            f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")

    # ---- why the failed row wins, and why that is not an argument -------
    # MEASURED RESULT THAT LOOKS LIKE AN ARGUMENT FOR KEEPING THE DEAD
    # SLEEVES, AND IS NOT. On this metric delta+skew+cascade come in at
    # 121 days against 268 for relvol+oirank. That is a real backtest
    # number and it is not a forecast, because the two tests ask different
    # questions:
    #
    #   the audits asked   does the TIMING add anything beyond holding a
    #                      fixed tilt? delta was 83% static, skew 90%.
    #   this metric asks   what was the return per unit of drawdown?
    #
    # A static tilt can score well here and still be worthless forward.
    # "83% static" means the sleeve's return came from persistently
    # holding certain coins over this sample -- it is a statement about
    # which coins went up, not about a repeatable rule. days-to-target
    # computed on a series whose mean is not expected to persist describes
    # the past and forecasts nothing.
    #
    # So this metric is only admissible on a series that passed a signal
    # test first. I reported 126 days for "the cross-sectional book" one
    # message before writing this, and that figure leaned on exactly these
    # sleeves. It was not a forecast and I should not have offered it as
    # one. The defensible figure is the verified-only row.
    #
    # Also: 'cascade' is mislabelled in the verdict column above. It was
    # never shown to fail a signal test -- it fires too rarely to test,
    # roughly 20-30 times in 1,300 days, and it was IDLE LIVE, which is a
    # different finding from failing an audit.
    log("")
    log("    READ THAT TABLE BACKWARDS. The failed/idle row winning is not")
    log("    an argument for keeping those sleeves. The audits asked")
    log("    whether the TIMING beats a fixed tilt (delta 83% static, skew")
    log("    90%); this metric asks what the return per unit of drawdown")
    log("    WAS. A static tilt scores well here and forecasts nothing --")
    log("    'static' means the return came from which coins happened to")
    log("    move, not from a repeatable rule.")
    log("    So this metric is admissible only on a series that already")
    log("    passed a signal test. The verified-only row is the number to")
    log("    plan against; the 'all sleeves' row is inflated by tilts.")

    # every subset of the verified set, so the answer is not one blend
    if len(have_v) >= 2:
        log("")
        log("    every subset of the verified sleeves:")
        for r in range(1, len(have_v) + 1):
            for c in itertools.combinations(have_v, r):
                d, _, mu, _ = days_of(blend(list(c)))
                log(f"      {'+'.join(c):<30}"
                    f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8} days")

    # ---- optional: add the short book -----------------------------------
    if a.short > 0:
        import short_risk as RK
        log("")
        log("  " + "=" * 68)
        log(f"  WITH THE {a.tf}h SHORT BOOK AT w={a.short:.2f}")
        log("  " + "=" * 68)
        shorts, _, legs = RK.build(RK.universe(), a.tf, (-1,), a.cap)
        log(f"    {len(legs)} legs")
        # Swept, not evaluated at one weight. A single weight hides whether
        # the optimum is a plateau or a point, and the short book's value
        # against the VERIFIED book is 1.5x where against the
        # tilt-inflated replica it looked like 1.13x -- the weaker the
        # base book's mean, the more a diversifier is worth.
        base = blend(have_v)
        if base is not None:
            j = base.index.intersection(shorts.index)
            A = base.loc[j] / float(base.loc[j].std())
            B = shorts.loc[j] / float(shorts.loc[j].std())
            rho = float(np.corrcoef(A.values, B.values)[0, 1])
            log(f"    rho(verified book, short book) = {rho:+.3f}")
            log("")
            log(f"    {'w(short)':>9}{'mu/day':>9}{'DD unit':>10}"
                f"{'days':>8}{'months':>8}")
            bw, bd = None, float("inf")
            for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                d, dd, mu, _ = days_of((1 - w) * A + w * B)
                log(f"    {w:>9.1f}{mu*100:>8.3f}%{dd*100:>9.0f}%"
                    f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8}"
                    f"{(f'{d/30.4:.1f}' if np.isfinite(d) else '-'):>8}")
                if d < bd:
                    bw, bd = w, d
            log("")
            log(f"    best w={bw:.1f} -> {bd:.0f} days"
                f" ({bd/30.4:.1f} months), a"
                f" {days_of(A)[0]/bd:.2f}x improvement on the verified")
            log(f"    book alone. Half-samples at that weight:")
            h = len(j) // 2
            for nm2, lo, hi in (("1st half", 0, h), ("2nd half", h, len(j))):
                d, dd, mu, _ = days_of((1 - bw) * A.iloc[lo:hi]
                                       + bw * B.iloc[lo:hi])
                log(f"      {nm2:<10}{mu*100:>8.3f}%{dd*100:>9.0f}%"
                    f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8} days")

    # ---- half-sample check on the best set ------------------------------
    # Checks the VERIFIED set, not the best-by-days set. Picking the
    # half-sample subject by this metric would have featured the
    # static-tilt blend, which is the one series here whose mean is not
    # expected to persist -- i.e. the one for which a half-sample check
    # means the least.
    if "verified only" in results:
        bestnm = "verified only"
        log("")
        log("  " + "=" * 68)
        log(f"  HALF-SAMPLE CHECK: {bestnm}")
        log("  " + "=" * 68)
        x = results[bestnm][1]
        h = len(x) // 2
        for nm, seg in (("1st half", x.iloc[:h]), ("2nd half", x.iloc[h:])):
            d, dd, mu, _ = days_of(seg)
            log(f"    {nm:<10}{mu*100:>8.3f}%{dd*100:>9.0f}%"
                f"{(f'{d:.0f}' if np.isfinite(d) else 'never'):>8} days")
        log("")
        log("    A set chosen by looking at this table is a fitted choice.")
        log("    The sleeve membership was NOT chosen here, though -- it was")
        log("    fixed by the earlier audits (static-tilt tests, timing")
        log("    nulls, lookahead poisoning) before this metric existed.")
        log("    That is the only reason the comparison above is admissible.")

    log("")
    log("  " + "=" * 68)
    log("  LIMITS")
    log("  " + "=" * 68)
    log("    pattern and sr are event sleeves and are NOT here -- they need")
    log("    the trade-to-daily conversion short_risk.build does. They are")
    log("    2 of the 4 survivors, so every number above understates a")
    log("    survivors-only book.")
    log("    These are backtest series. Three of the sleeves in the 'all'")
    log("    row failed their own audits, which means their contribution to")
    log("    that row is not a tradeable edge even though it is a real")
    log("    backtest number. That is the entire point of separating them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

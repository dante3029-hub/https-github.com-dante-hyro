#!/usr/bin/env python3
"""
rp_book.py -- "who is opening, who is closing": a new, pre-registered strategy.

DESIGN PRINCIPLES (each one taken from a measured result in this project)
  * Positioning, not price.   OI rank was the only sleeve whose edge survived
                              2023, data the book never saw (oos_diag.py).
  * Cross-sectional, not      Every time-series signal tested (MACD, FVG,
    timing.                   delta patterns at five resolutions) failed.
  * Market-neutral.           Drawdown against a static floor is what limits
                              size (strategy-positioning-book.md).
  * Daily bars, market fills. No limit-touch fills (the FVG revisit died at
                              0.10% penetration) and no pattern detectors.

THE IDEA
Open interest says whether positions are being OPENED or CLOSED. Taker delta
says which side is AGGRESSIVE. Together:

    OI up,   buying   new longs opened with urgency      follow  -> long
    OI up,   selling  new shorts opened with urgency     follow  -> short
    OI down, buying   shorts forced to cover             fade    -> short
    OI down, selling  longs liquidated                   fade    -> long

score = z(OI change) x z(net taker flow), cross-sectionally. Large positive =
long, large negative = short. Follow fresh conviction, fade forced closing.

Neither input alone is this strategy. The existing OI sleeve ignores which side
is opening; delta alone died at every resolution. The product is the claim.

PRE-REGISTERED -- fixed before the first run, never changed afterwards
  universe   the 25 coins with OI data
  lookbacks  1, 3, 7 days, rank-averaged (no single horizon is chosen)
  book       long top 5 / short bottom 5, 0.5/N each, dollar-neutral
  hold       3 days, phase-averaged
  stop       none
  costs      0.055% fee + 0.03% slippage per side, funding charged
  verdict    excess Sharpe over 40 random books, by period and calendar year
  rejection  correlation > 0.6 with OI rank means it is a rehash, not new

Engine and panels are reused from oos_book.py (infrastructure, not signal).

USAGE
    python3 rp_book.py --selftest
    python3 rp_book.py
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

import oos_book as OB

LOOKBACKS = (1, 3, 7)
N = 5
HOLD = 3
SEEDS = 40


def _ranks(x):
    out = np.full(x.shape[0], np.nan)
    ok = np.isfinite(x)
    k = int(ok.sum())
    if k < 2:
        return out
    order = np.argsort(x[ok])
    r = np.empty(k)
    r[order] = np.arange(k) / (k - 1)
    out[ok] = r
    return out


def _z(x):
    ok = np.isfinite(x)
    if ok.sum() < 4:
        return np.full_like(x, np.nan)
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / s if s > 0 else np.full_like(x, np.nan)


def sig_rp(OI, DN, lookbacks=LOOKBACKS):
    """Score at engine bar t uses data through t-1 only (weights from t are
    applied to R[t] = close(t-1) -> close(t))."""
    def f(t):
        stack = []
        for k in lookbacks:
            if t - 1 - k < 0:
                continue
            now, then = OI[t - 1], OI[t - 1 - k]
            with np.errstate(divide="ignore", invalid="ignore"):
                doi = np.where((then > 0) & (now > 0), np.log(now / then), np.nan)
            blk = DN[t - k:t]
            flow = np.nansum(blk, axis=0)
            flow[np.isfinite(blk).sum(axis=0) < k] = np.nan
            prod = _z(doi) * _z(flow)
            if np.isfinite(prod).sum() >= 2 * N + 2:
                stack.append(_ranks(prod))
        if not stack:
            return np.full(OI.shape[1], np.nan)
        M = np.vstack(stack)
        out = np.nanmean(M, axis=0) if np.isfinite(M).any() else M[0]
        out[np.isfinite(M).sum(axis=0) < len(stack)] = np.nan
        return out
    return f


def quadrant_of(OI, DN, t, k=3):
    now, then = OI[t - 1], OI[t - 1 - k]
    with np.errstate(divide="ignore", invalid="ignore"):
        doi = np.log(now / then)
    flow = np.nansum(DN[t - k:t], axis=0)
    return doi, flow


def selftest(P) -> int:
    T = P["PX"].shape[0]
    f_full = sig_rp(P["OI"], P["DN"])
    t = T // 2
    cut = {k: (v[:t + 1] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
           for k, v in P.items()}
    f_cut = sig_rp(cut["OI"], cut["DN"])
    a, b = f_full(t), f_cut(t)
    assert np.allclose(np.nan_to_num(a, nan=9), np.nan_to_num(b, nan=9)), \
        "signal changes when the future is removed -- lookahead"
    print("  signal is causal (truncation-invariant): OK")
    # direction check on a constructed case: OI up + buying must outrank
    # OI up + selling
    C = 12
    OI = np.ones((10, C)); DN = np.zeros((10, C))
    OI[8, :] = 1.0
    OI[8, 0] = 1.5; DN[5:8, 0] = 0.3     # opened, buying     -> top
    OI[8, 1] = 1.5; DN[5:8, 1] = -0.3    # opened, selling    -> bottom
    OI[8, 2] = 0.6; DN[5:8, 2] = 0.3     # closed, buying     -> bottom (fade)
    OI[8, 3] = 0.6; DN[5:8, 3] = -0.3    # closed, selling    -> top (fade)
    for j in range(4, C):
        OI[8, j] = 1.0 + 0.01 * j; DN[5:8, j] = 0.01 * (j % 3)
    s = sig_rp(OI, DN, lookbacks=(3,))(9)
    assert s[0] > s[1] and s[3] > s[2], f"quadrant logic wrong: {s[:4]}"
    print("  quadrant logic (follow opening, fade closing): OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    P = OB.build_panels()
    if a.selftest:
        return selftest(P)

    dates = P["dates"]
    cols = P["has_oi"]
    C = len(P["coins"])
    sig = sig_rp(P["OI"], P["DN"])
    real = OB.run_sleeve(sig, P["PX"], P["R"], P["FN"], N, HOLD, HOLD, None, cols)
    ctrl = np.array([OB.run_sleeve(OB.sig_random(C, zlib.crc32(f"rp{s}".encode())),
                                   P["PX"], P["R"], P["FN"], N, HOLD, HOLD,
                                   None, cols) for s in range(SEEDS)])
    cm = ctrl.mean(axis=0)
    exc = real - cm
    oi = OB.run_sleeve(OB.sig_oi(P["OI"]), P["PX"], P["R"], P["FN"], 5, 3, 3,
                       None, cols)

    start = OB.WARMUP + 8
    idx = pd.DatetimeIndex(dates)
    m_all = np.arange(len(dates)) >= start
    print("\n  'Who is opening, who is closing' -- pre-registered, single run")
    print(f"  {idx[start].date()} -> {idx[-1].date()}, 25 OI coins, "
          f"N={N}, hold {HOLD}d, lookbacks {LOOKBACKS}\n")

    def row(lab, m):
        cs = np.array([OB.sharpe(c[m]) for c in ctrl])
        z = (OB.sharpe(real[m]) - cs.mean()) / cs.std() if cs.std() > 0 else 0
        tm, nm, mp = OB.monthly_t(idx[m], exc[m])
        print(f"  {lab:<10}{int(m.sum()):>6}{OB.sharpe(real[m]):>8.2f}"
              f"{cs.mean():>8.2f}{z:>7.2f}{OB.sharpe(exc[m]):>10.2f}"
              f"{tm:>7.2f}{mp:>5}/{nm:<3}{exc[m].mean()*365*100:>7.1f}%")

    print(f"  {'period':<10}{'days':>6}{'Sharpe':>8}{'random':>8}{'z':>7}"
          f"{'excess Sh':>10}{'t_mon':>7}{'mon+':>8}{'exc ann':>8}")
    row("ALL", m_all)
    per = OB.periods(idx)
    for k, m in per.items():
        row(k, m & m_all)
    for y in sorted(set(idx.year)):
        m = (idx.year == y) & m_all
        if m.sum() > 60:
            row(str(y), m)

    rho = np.corrcoef(real[m_all], oi[m_all])[0, 1]
    print(f"\n  correlation with OI rank: {rho:.2f}  "
          f"({'REHASH -- reject' if rho > 0.6 else 'new information'})")

    # diagnostic only (not used to change anything): which quadrant pays?
    R = P["R"]
    qs = {"open+buy": [], "open+sell": [], "close+buy": [], "close+sell": []}
    for t in range(start, len(dates) - HOLD):
        doi, flow = quadrant_of(P["OI"], P["DN"], t)
        fwd = np.nansum(R[t:t + HOLD], axis=0)
        xs = fwd - np.nanmean(fwd[cols])
        for j in np.flatnonzero(cols):
            if not (np.isfinite(doi[j]) and np.isfinite(flow[j]) and np.isfinite(xs[j])):
                continue
            key = ("open" if doi[j] > 0 else "close") + ("+buy" if flow[j] > 0 else "+sell")
            qs[key].append(xs[j])
    print("\n  diagnostic: next-3-day return vs the cross-section, by quadrant")
    for k, v in qs.items():
        v = np.array(v)
        print(f"    {k:<11} n={len(v):>6}  mean {v.mean()*100:+.3f}%  "
              f"t {v.mean()/(v.std()/np.sqrt(len(v))):+.2f}")

    pd.DataFrame({"rp": real, "rp_excess": exc, "oirank": oi},
                 index=idx).to_csv("rp_book_daily.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

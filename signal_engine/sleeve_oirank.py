"""
OIRANK sleeve — cross-sectional open-interest change, 5 per side, 3-day
rebalance, rank averaged across 3/5/7/10/14-day lookbacks.

Long the biggest OI EXPANSION, short the biggest CONTRACTION. Rising open
interest means new positions being opened, not existing ones closing.

WHY THIS SLEEVE EXISTS
Every other sleeve reads PRICE or VOLUME. sr, srflip, pattern, fvg, breakout,
EMA retest, BTC-pair signals -- all answer "is price continuing up?" and all
correlate 0.3-0.6 with each other. Adding another changes nothing.

Open interest is POSITIONING. It is the only non-price input in the book, and
that shows in the correlations:

    delta -0.05   relvol 0.31   skew 0.01   cascade -0.04
    sr     0.15   pattern 0.17  fvg  0.13   srflip  0.05

Book 3.81 -> 4.13, bull 3.43 -> 3.93, maxDD 4.4% -> 3.7%.

LOOKBACK IS A BLEND, DELIBERATELY
A single 7-day lookback scored 1.41 against ~0.95 at both 5d and 10d -- a 45%
spike, which is the signature of a fitted parameter:

    lookback   best across holds
    3d         0.92
    5d         0.97
    7d         1.41   <- spike
    10d        0.97
    14d        0.49

Averaging the RANK across five lookbacks costs 0.07 in the book (4.20 -> 4.13)
and removes the dependence. The WIDER blend beats the narrow 5/7/10 one
(4.13 vs 4.08), which is what a real effect does and a fitted one does not.

VALIDATION
Random-coin-selection control on the same bars with the same turnover scored
-1.83 / -1.88 / -1.45 against the real 1.34. Gap +2.78 to +3.22.

The N sweep is a clean plateau: 1.09 / 1.19 / 1.34 / 1.33 / 1.05 for n=3..8.

A NOTE ON WHAT THIS IS NOT
Raw predictive correlation is near zero -- OI change vs NEXT-day return
averages +0.010 per coin. Same-day is +0.287, but that is just the mechanical
fact that OI rises as price rises. The edge is CROSS-SECTIONAL (which coins
relative to each other), not time-series (when to be in the market).

That distinction caused a real bug during development: an early test set
weights from sig(t) and applied them to R(t), the same bar. It produced +4.09.
Lagged properly it is 1.34. The `t` convention here is strict -- the score at
bar t uses only OI up to and including t, and the caller applies the weights
from bar t onward.
"""
import numpy as np

from .rank_weights import rank_to_weights

LOOKBACKS = (3, 5, 7, 10, 14)
N_PER_SIDE = 5
CADENCE_HOURS = 3 * 24           # 3-day hold


def _pct_change_at(OI: np.ndarray, t: int, k: int) -> np.ndarray:
    """OI change over k bars ending at t. NaN where either end is missing or
    the base is non-positive."""
    C = OI.shape[1]
    if t < k:
        return np.full(C, np.nan)
    now, then = OI[t, :], OI[t - k, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where((then > 0) & np.isfinite(then) & np.isfinite(now),
                       now / then - 1.0, np.nan)
    return out


def _to_ranks(x: np.ndarray) -> np.ndarray:
    """Percentile rank of the finite entries, NaN preserved. Ranking before
    averaging means one lookback with an extreme outlier cannot dominate."""
    out = np.full(x.shape[0], np.nan)
    ok = np.isfinite(x)
    k = int(ok.sum())
    if k < 2:
        return out
    order = np.argsort(x[ok])
    r = np.empty(k, dtype=float)
    r[order] = np.arange(k, dtype=float) / max(k - 1, 1)
    out[ok] = r
    return out


def oi_scores(OI: np.ndarray, t: int, lookbacks=LOOKBACKS) -> np.ndarray:
    """Mean percentile rank of OI change across several lookbacks.

    OI: (T, C) daily open-interest matrix.
    t:  bar index to score AT, using only OI up to and including t.
    """
    C = OI.shape[1]
    stack = []
    for k in lookbacks:
        chg = _pct_change_at(OI, t, k)
        if np.isfinite(chg).sum() >= 4:
            stack.append(_to_ranks(chg))
    if not stack:
        return np.full(C, np.nan)
    M = np.vstack(stack)
    counts_pre = np.isfinite(M).sum(axis=0)
    out = np.full(M.shape[1], np.nan)
    any_ok = counts_pre > 0
    if any_ok.any():
        with np.errstate(invalid='ignore'):
            out[any_ok] = np.nanmean(M[:, any_ok], axis=0)
    # a coin scored by only one lookback is not comparable to one scored by
    # five; require at least half of them
    counts = np.isfinite(M).sum(axis=0)
    out[counts < max(1, len(stack) // 2)] = np.nan
    return out


def latest_target_weights(OI: np.ndarray, t: int = None, n: int = N_PER_SIDE,
                          lookbacks=LOOKBACKS):
    """
    OI: (T, C) daily open-interest matrix, same column order as the price
        matrices from data_loader.
    t:  bar index to score at (defaults to the last available bar).

    Returns (weights, ok).
    """
    if t is None:
        t = OI.shape[0] - 1
    s = oi_scores(OI, t, lookbacks)
    return rank_to_weights(s, n)

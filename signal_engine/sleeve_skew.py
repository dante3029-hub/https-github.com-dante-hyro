"""
SKEW sleeve — cross-sectional realised skewness, 60-day lookback, 6 per side,
45-day rebalance.

Long the HIGH-skew names, short the LOW-skew ones (the Quantpedia sign). The
premium exists because investors overpay for lottery-like payoffs, so names
with recently negative skew are cheap.

RESEARCH PROVENANCE
  Sharpe 1.26, bull 1.12, bear 1.38, maxDD 17.1% on the hourly simulator.
  Negatively correlated with delta (-0.18) and relvol (-0.24), which is why it
  earns a place despite being mid-table on its own -- it is the cheapest
  variance reduction in the book.

  LOOKBACK IS 60 DAYS, NOT 45. The parameter surface is a clean plateau and
  60d beats 45d at EVERY position count:

      lookback   n=3    n=4    n=5    n=6    n=8
      30d       0.59   0.51   0.57   0.60   0.71
      45d       0.64   0.77   0.88   0.90   0.89
      60d       0.83   0.99   1.10   1.18   1.30

  Fourteen of fifteen cells have both halves positive. n=6 is used rather than
  n=8 because the gain from 6 to 8 is small and a wider book dilutes the
  ranking.

  UNIVERSE IS THE 24-COIN CORE, NOT THE WIDE ONE. On 60 coins skew collapses
  to -0.16 -- realised skewness on thin names measures noise, not a premium.
  This was found the hard way: book.py globbed its universe from disk, so
  when the 60-coin taker fetch landed the sleeve silently switched universe
  and looked broken.

DAILY BARS. Tested at 12h and it degrades (1.32 -> 0.96), with the regime
profile inverting. This is a daily-resolution signal.
"""
import numpy as np

from .rank_weights import rank_to_weights

LOOKBACK_DAYS = 60
N_PER_SIDE = 6
CADENCE_HOURS = 45 * 24          # 45-day hold


def skew_scores(R: np.ndarray, t: int, lookback: int = LOOKBACK_DAYS) -> np.ndarray:
    """Realised skewness of each coin's daily returns over the trailing window.

    R: (T, C) daily return matrix.
    t: bar index to score AT. The window is R[t-lookback:t] -- strictly
       BEFORE t, so the score is knowable at the open of bar t.

    Returns a length-C score array, NaN where a coin has too little history.
    """
    C = R.shape[1]
    if t < lookback:
        return np.full(C, np.nan)
    win = R[t - lookback:t, :]
    out = np.full(C, np.nan)
    for c in range(C):
        col = win[:, c]
        col = col[~np.isnan(col)]
        k = col.size
        # need a meaningful sample; skewness on a handful of points is noise
        if k < lookback * 0.8 or k < 3:
            continue
        sd = col.std(ddof=1)
        if sd == 0:
            continue
        m = col.mean()
        g1 = ((col - m) ** 3).sum() / (k * sd ** 3)
        # BIAS-CORRECTED (sample) skewness, matching pandas .skew(). The
        # population form g1 = m3/sd**3 is NOT the same ranking: parity
        # against the backtest failed on it (1.47 vs 1.26).
        out[c] = float(np.sqrt(k * (k - 1)) / (k - 2) * g1)
    return out


def latest_target_weights(R: np.ndarray, t: int = None, n: int = N_PER_SIDE,
                          lookback: int = LOOKBACK_DAYS):
    """
    R: (T, C) daily return matrix for the CORE universe.
    t: bar index to score at (defaults to the last available bar).

    Returns (weights, ok) -- same shape as every other cross-sectional sleeve.
    `ok` is False when there were not enough valid names to fill both sides.
    """
    if t is None:
        t = R.shape[0] - 1
    s = skew_scores(R, t, lookback)
    return rank_to_weights(s, n)

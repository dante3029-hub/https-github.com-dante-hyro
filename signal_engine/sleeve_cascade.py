"""
CASCADE sleeve — market-wide flush fade. Long the worst-hit names for 3 days
after the equal-weight market drops 2.5 sigma in one session.

NOT A RANKING SLEEVE
Unlike delta/relvol/skew/oirank this is EVENT-DRIVEN: it is flat most of the
time and only takes a position after a specific market event. It holds a
position on roughly 3.3% of bars. It still returns the same (weights, ok)
shape so the orchestrator treats it uniformly, but `ok=True` with an all-zero
weight vector is the NORMAL state, not a failure.

That distinction matters for the health check: a flat cascade is expected; a
flat delta is not.

RESEARCH PROVENANCE
  Sharpe 0.49 hourly-marked, bull -0.32, bear 1.15, maxDD 36.8%.

  It is the weakest sleeve in the book and the only one with a negative bull
  half. It survives because it is nearly uncorrelated with everything
  (-0.04 to 0.12) and fires exactly when the rest of the book is hurting --
  a market-wide flush is the one event that moves every price sleeve against
  you at once.

  Its Sharpe is MUCH higher on a daily-marked basis (1.17) than hourly (0.49).
  That is not a bug: a sleeve holding a position 3.3% of the time has its mean
  diluted across every flat hour while continuous marking still picks up the
  variance. Dense sleeves show the opposite (fvg: 2.96 hourly, 1.09 daily).
  The hourly figure is the one used in the book because that is what the
  equity curve actually does.

MECHANISM, in order
  1. market return = equal-weight mean across the core universe
  2. z = today's market return / trailing 20-day stdev of that mean
  3. if z <= -2.5, go long the 5 worst-performing coins that day
  4. hold 3 days, then flat unless another cascade fires
  5. no overlapping cascades -- a new one cannot open while one is running

The 2.5-sigma threshold is ABSOLUTE, not a multiple of a trailing average.
That distinction was learned expensively elsewhere in this project: a
multiple-of-average threshold on a small base fires on noise (~50 times per
coin per year) and inverts the sign of the result.
"""
import numpy as np

Z_THRESHOLD = -2.5
VOL_WINDOW = 20
N_LONG = 5
HOLD_DAYS = 3
CADENCE_HOURS = 24


def market_z(R: np.ndarray, t: int, window: int = VOL_WINDOW) -> float:
    """Today's equal-weight market return in trailing standard deviations.

    R: (T, C) daily return matrix.
    t: bar index. Uses R[t] for the move and R[t-window:t] for the scale --
       the scale is strictly BEFORE t, so no same-bar information leaks in.

    Returns NaN when there is not enough history or the market has no variance.
    """
    if t < window:
        return float('nan')
    ok_rows = np.isfinite(R).sum(axis=1) > 0
    mkt = np.full(R.shape[0], np.nan)
    if ok_rows.any():
        with np.errstate(invalid='ignore'):
            mkt[ok_rows] = np.nanmean(R[ok_rows, :], axis=1)
    hist = mkt[t - window:t]
    hist = hist[~np.isnan(hist)]
    if hist.size < window * 0.8:
        return float('nan')
    sd = hist.std()
    if sd <= 0 or not np.isfinite(mkt[t]):
        return float('nan')
    return float(mkt[t] / sd)


def latest_target_weights(R: np.ndarray, t: int = None, n_long: int = N_LONG,
                          z_threshold: float = Z_THRESHOLD,
                          window: int = VOL_WINDOW):
    """
    R: (T, C) daily return matrix for the core universe.
    t: bar index to evaluate at (defaults to the last available bar).

    Returns (weights, ok).

    weights is all-zero when no cascade has fired -- that is the NORMAL state
    for this sleeve, not an error. `ok` is False only when the market z-score
    could not be computed at all (insufficient history), which IS worth
    flagging.

    HOLDING is the caller's responsibility. This returns the target for a
    freshly-fired cascade; the orchestrator's cadence gating holds it for
    HOLD_DAYS. That matches how every other sleeve here works -- the sleeve
    says what it wants NOW, the orchestrator decides when to ask again.
    """
    C = R.shape[1]
    if t is None:
        t = R.shape[0] - 1
    z = market_z(R, t, window)
    if not np.isfinite(z):
        return np.zeros(C), False
    if z > z_threshold:
        return np.zeros(C), True          # no cascade: flat, and that is fine
    today = R[t, :]
    valid = np.isfinite(today)
    if valid.sum() < n_long:
        return np.zeros(C), True
    # long the worst performers of the flush
    order = np.argsort(np.where(valid, today, np.inf))
    w = np.zeros(C)
    w[order[:n_long]] = 1.0 / n_long
    return w, True

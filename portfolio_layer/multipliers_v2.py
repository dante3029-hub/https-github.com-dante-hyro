"""
portfolio_layer/multipliers_v2.py — equal-risk scaling for N sleeves.

REPLACES `position_sizer.compute_sleeve_multipliers`, which is hardcoded to
the old six (main/short/flow/delta/relvol/bos) and built around an A/B combo
structure those sleeves had. It cannot take nine, and the nine-sleeve book does
not use that structure.

WHAT THE RESEARCH ACTUALLY DOES

    scale[k] = median(vol) / vol[k]

Each sleeve is scaled so it contributes the same RISK, then the portfolio layer
nets across sleeves, caps per coin, caps gross, and charges ONE cost on the net
change. That is `hourly_sim.simulate`, which produced every number in the book
(4.13 Sharpe, 3.7% maxDD), so the live sizing must match it or the live book is
not the book that was validated.

WHY EQUAL RISK AND NOT EQUAL WEIGHT
The sleeves differ in volatility by ~4x. fvg alone would dominate an
equal-weight book: it holds a position ~95% of the time against cascade's 3.3%.
Equal-risk is what makes a 0.49-Sharpe sleeve worth carrying next to a 2.96 one
-- cascade earns its place by being uncorrelated and firing exactly when the
price sleeves hurt, not by its own return.

TWO THINGS THIS DELIBERATELY DOES NOT DO

**No optimisation.** Not mean-variance, not risk parity with a covariance
matrix, not Kelly. Every one of those fits weights to the sample and this
project has been burned repeatedly by parameters that looked great in-sample
(fvg at 0.25 ATR "scored" 2.84; it is really 0.06). The median is a robust
statistic with nothing to fit.

**No sleeve gets a multiplier above MAX_MULT.** A sleeve whose trailing vol
collapses toward zero -- which happens when it is FLAT, not when it is safe --
would otherwise get an enormous multiplier and blow up the moment it trades
again. cascade sits flat 97% of the time, so this is not hypothetical.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── the vols the research measured, full sample ──────────────────────────
#
# A live bot starts with NO return history, so equal-risk cannot scale
# anything and every sleeve defaults to 1.0. That is not a small problem:
# fvg's daily vol is 0.0293 and oirank's is 0.0045, so equal WEIGHT would let
# fvg take roughly 6.5x the risk oirank does, and the book being run would not
# be the book that was validated.
#
# These are measured over the full 1,315-day sample by hourly_sim, which is
# what produced every number in the book (4.13 Sharpe, 3.7% maxDD). Using them
# as the prior is MORE faithful than a trailing 60-day window, not less.
#
# They are a PRIOR, not a fixture: once a sleeve has `MIN_OBS_FOR_LIVE` days of
# live history, its own trailing vol takes over. Blending the two would hide
# the moment live diverges from research, which is exactly what the paper
# month exists to detect.
SLEEVE_VOL_DAILY = {
    'cascade': 0.012690,
    'delta':   0.010008,
    'fvg':     0.029302,
    'oirank':  0.004495,
    'pattern': 0.015951,
    'relvol':  0.007449,
    'skew':    0.009128,
    'sr':      0.015094,
    'srflip':  0.010289,
}

MIN_OBS_FOR_LIVE = 60      # days of live history before trusting it over the prior

MIN_OBS = 20            # below this, a vol estimate is noise
MAX_MULT = 4.0          # cap -- a flat sleeve has near-zero vol, not low risk
MIN_MULT = 0.1
DEFAULT_MULT = 1.0


def _trailing_vol(hist: Optional[np.ndarray], window: int = 60) -> float:
    """Standard deviation of a sleeve's recent returns. NaN if unusable."""
    if hist is None:
        return float("nan")
    a = np.asarray(hist, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < MIN_OBS:
        return float("nan")
    v = float(a[-window:].std())
    return v if v > 0 else float("nan")


def compute_multipliers_equal_risk(
        sleeve_return_history: Dict[str, np.ndarray],
        window: int = 60,
        target_daily_vol: Optional[float] = None,
        account_multiplier: float = 1.0) -> Dict[str, float]:
    """Equal-risk multiplier per sleeve.

    sleeve_return_history: {sleeve: np.ndarray of per-period returns}
    target_daily_vol: dollar vol target for the BLENDED book. When given, the
        whole set is scaled so the blend lands near it.
    account_multiplier: risk-overlay output. 0.0 means the kill switch has
        tripped -- every sleeve goes to zero.

    Returns {sleeve: multiplier}. A sleeve with no usable history gets
    DEFAULT_MULT and is logged, NOT silently dropped -- a sleeve that vanishes
    from the book without saying so is the failure mode this project keeps
    hitting.
    """
    if account_multiplier <= 0.0:
        logger.warning("account_multiplier is %.3f -- every sleeve forced to 0",
                       account_multiplier)
        return {k: 0.0 for k in sleeve_return_history}

    vols, from_prior = {}, []
    for k, h in sleeve_return_history.items():
        a = np.asarray(h, dtype=float) if h is not None else np.array([])
        a = a[np.isfinite(a)]
        v = _trailing_vol(h, window) if a.size >= MIN_OBS_FOR_LIVE else float("nan")
        if np.isfinite(v):
            vols[k] = v
        elif k in SLEEVE_VOL_DAILY:
            # fall back to the researched vol rather than defaulting to 1.0.
            # Defaulting would size a 0.029-vol sleeve the same as a 0.0045 one.
            vols[k] = SLEEVE_VOL_DAILY[k]
            from_prior.append(k)
        else:
            from_prior.append(k)
    missing = [k for k in from_prior if k not in vols]
    if not vols:
        logger.error("no sleeve has usable return history -- every multiplier "
                     "defaults to %.1f. The book is NOT risk-balanced.",
                     DEFAULT_MULT)
        return {k: DEFAULT_MULT * account_multiplier for k in sleeve_return_history}
    if from_prior:
        known = [k for k in from_prior if k in SLEEVE_VOL_DAILY]
        if known:
            logger.info("using researched vols for %s (< %d days of live "
                        "history yet)", ", ".join(sorted(known)), MIN_OBS_FOR_LIVE)
    if missing:
        logger.warning("no history AND no researched vol for %s -- defaulting "
                       "to %.1f, the book is NOT risk-balanced for these",
                       ", ".join(sorted(missing)), DEFAULT_MULT)

    target = float(np.median(list(vols.values())))
    out: Dict[str, float] = {}
    for k in sleeve_return_history:
        if k in vols:
            m = target / vols[k]
            if m > MAX_MULT:
                logger.warning("sleeve %s multiplier %.1f capped at %.1f -- its "
                               "trailing vol is near zero, which usually means "
                               "FLAT, not safe", k, m, MAX_MULT)
            out[k] = float(np.clip(m, MIN_MULT, MAX_MULT))
        else:
            out[k] = DEFAULT_MULT

    if target_daily_vol is not None and target > 0:
        # scale the whole set so the blend lands near the dollar target.
        # sqrt(n) assumes rough independence across sleeves, which is close
        # enough here: the highest pairwise correlation in the book is 0.45.
        n = max(len(out), 1)
        blend_vol = target * np.sqrt(n) / n
        if blend_vol > 0:
            k_scale = target_daily_vol / blend_vol
            out = {k: v * k_scale for k, v in out.items()}

    return {k: v * account_multiplier for k, v in out.items()}


def compute_sleeve_multipliers(sleeve_return_history: Dict[str, np.ndarray],
                               window: int = 60,
                               target_daily_vol: Optional[float] = None,
                               account_multiplier: float = 1.0,
                               **_ignored) -> Dict[str, float]:
    """Drop-in name for portfolio.py, taking a DICT rather than six positional
    history arrays. `**_ignored` absorbs the old keyword arguments so a stale
    call site fails loudly on the missing dict rather than silently sizing on
    defaults."""
    return compute_multipliers_equal_risk(
        sleeve_return_history, window=window,
        target_daily_vol=target_daily_vol,
        account_multiplier=account_multiplier)

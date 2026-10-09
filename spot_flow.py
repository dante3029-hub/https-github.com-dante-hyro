#!/usr/bin/env python3
"""
spot_flow.py -- does spot-vs-perp taker-flow divergence carry an edge?

THE HYPOTHESIS
--------------
All nine live sleeves read the perp tape. Spot flow is a different tape.
The informative state is DISAGREEMENT between them:

    spot flow > perp flow   ->  cash buyers absorbing leveraged sellers
    spot flow < perp flow   ->  holders distributing into leveraged chasers

Signal (scale-free, which is mandatory -- see SPOT_FLOW.md on the 1000x
contract trap):

    SFN = spot_delta / spot_volume        per coin per day, in [-1, 1]
    PFN = perp_delta / perp_volume        per coin per day, in [-1, 1]
    div(t) = z( sum SFN[t-k:t] ) - z( sum PFN[t-k:t] )

Both legs are z-scored cross-sectionally BEFORE subtracting, so a coin whose
spot pair is denominated 1000x differently contributes identically. This is
the whole reason the signal is built on ratios and not on raw delta.

HARNESS
-------
`xs()`, `stats()`, `panel()` and FEE are IMPORTED FROM book.py, not
reimplemented. That is deliberate. The last time a sleeve got a fresh test
harness (oirank) it reported Sharpe +4.09 because the harness applied
weights from sig(t) to R(t) on the same bar; the real number was 1.34. The
one in book.py has the timing invariant that matters:

    sig(t) may only read .iloc[:t]   (strictly before t)
    R.iloc[t] = PX[t]/PX[t-1] - 1    (the return it then earns)

So a signal formed on the close of t-1 earns the t-1 -> t return. Any new
signal function added here MUST respect that slice. `--audit` checks it.

MULTIPLE TESTING -- AND A HARNESS BUG THAT WAS CAUGHT HERE
----------------------------------------------------------
This script sweeps lookbacks x holds x variants, so the best config wins
partly by luck and must be measured against a null.

An earlier version of this file REPORTED "clears the multiple-testing bar"
ON A PANEL BUILT FROM RANDOM NUMBERS. Two things were wrong, and only one of
them is the one you would guess:

  1. THE DECISIVE BUG: the bar was a parametric
     `null.mean() + 2.3*null.std()`, and the null was computed once for an
     arbitrary config (`div k=3 h=7`) and then applied to whichever config
     won -- which was `disagree k=3`, a config that scores only ~11 names
     instead of 24 and therefore has a much wider null. Judging a thin,
     high-variance config against a wide config's null is precisely how a
     high-variance config wins a sweep on noise.

     Fixed by (a) computing the null FOR THE WINNING CONFIG, and (b) using
     an empirical p-value Bonferroni-corrected by the number of configs
     tried. On the noise panel the winner scored +0.76 against a null
     centred at -1.04 -- a 2.6-sigma single-test draw, p=0.025 -- which
     Bonferroni over 21 configs correctly turns into p=0.52, rejected.

  2. The null is a CIRCULAR TIME SHIFT (roll the signal panel by a random
     offset, wrapping) rather than book.xs(shuffle=True). The shift
     preserves the signal's own autocorrelation and cross-sectional shape by
     construction and destroys only its alignment with forward returns.

     HONESTY NOTE: the original justification written here -- that
     shuffle=True inflates turnover and so over-penalises the null -- was
     ASSERTED AND THEN MEASURED FALSE. Turnover is 1.574 (shift) vs 1.576
     (shuffle) vs 1.570 (real), a 0.3% difference worth 0.03% annualised.
     The shuffle null did come out systematically lower (-1.61 vs -1.04 on
     the same panel and config), but the mechanism for that gap was NOT
     established, so it is not claimed. The circular shift is preferred on
     the a-priori ground above, not on the strength of that gap.

WHY THE NULL IS CENTRED NEAR -1.0, NOT 0
----------------------------------------
A no-edge version of this strategy does not score 0; it scores its own cost
drag. Fees are ~7% annualised against ~12% book vol, so "no edge" is roughly
Sharpe -1. That is why the real result must be compared against the measured
null and never against zero -- and why a positive raw Sharpe is not, on its
own, evidence of anything.

USAGE
-----
    python3 -u spot_flow.py --audit         # timing + data integrity only
    python3 -u spot_flow.py --sweep         # the full grid + null
    python3 -u spot_flow.py --sweep --nulls 500   # finer p resolution
    python3 -u spot_flow.py --selftest      # prove the verdict rejects noise
    python3 -u spot_flow.py --corr          # correlation vs live sleeves
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import book
from book import CORE24, FEE, stats, xs

# One resolver, shared with fetch_spot.py. Two modules disagreeing about
# where the data lives is the bug that broke this on the server.
from fetch_spot import resolve_dir

SPOT, _SPOT_TRIED = resolve_dir("HYRO_SPOT_DATA_DIR", "spot_data", True)
# Funding is OPTIONAL: without it the confluence variants are skipped and
# the plain divergence sweep still runs. Missing data must narrow the test,
# never silently change what is being tested.
FUND_DIR, _FUND_TRIED = resolve_dir("HYRO_FUNDING_DATA_DIR", "funding_data", True)

MIN_DAYS = 400          # same floor book.panel() uses
MIN_BASE_OBS = 60       # minimum observations for a coin's trailing
                        # baseline mean to count as a 'long-run average'
Z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0


def log(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------------
# spot panel, aligned onto the perp index
# --------------------------------------------------------------------------
def load_spot_daily(sym: str) -> pd.DataFrame | None:
    """1h spot bars -> daily volume and delta. Mirrors book.load_daily so the
    resample boundaries are identical; a one-bar offset between the two
    panels would manufacture divergence out of nothing."""
    path = f"{SPOT}/{sym}_spot_1h.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"    ! {sym}: unreadable ({e})")
        return None
    if not {"open_time", "volume", "delta"} <= set(df.columns):
        log(f"    ! {sym}: unexpected columns {list(df.columns)}")
        return None
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("ts").sort_index()
    out = pd.DataFrame({
        "volume": df["volume"].resample("1D").sum(),
        "delta": df["delta"].resample("1D").sum(),
    }).dropna()
    return out if len(out) >= MIN_DAYS else None


def load_funding_daily(sym: str, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Funding settles every 8h -> 3 observations a day, so the daily
    aggregate is their SUM (the actual cost of holding that day), not their
    mean. Written by fetch_funding.py as <COIN>_funding.csv: ts,rate."""
    if FUND_DIR is None:
        return None
    path = f"{FUND_DIR}/{sym}_funding.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not {"ts", "rate"} <= set(df.columns):
        return None
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    s = df.set_index("t").sort_index()["rate"].resample("1D").sum()
    return s.reindex(idx)


def funding_panel(idx: pd.DatetimeIndex, coins: list[str]):
    out, covered = {}, []
    for c in coins:
        s = load_funding_daily(c, idx)
        if s is None or s.notna().sum() < MIN_DAYS:
            out[c] = pd.Series(np.nan, index=idx)
            continue
        out[c] = s
        covered.append(c)
    return pd.DataFrame(out, columns=coins), covered


def spot_panel(idx: pd.DatetimeIndex, coins: list[str]):
    """Returns (SVOL, SDN, covered) reindexed onto the PERP index so every
    matrix shares one calendar. Coins with no spot data become all-NaN
    columns, which rank_to_weights-style guards then drop -- they do not
    silently become zeros, because a zero divergence is a tradeable score
    and a missing one must not be."""
    sv, sd, covered = {}, {}, []
    for c in coins:
        d = load_spot_daily(c)
        if d is None:
            sv[c] = pd.Series(np.nan, index=idx)
            sd[c] = pd.Series(np.nan, index=idx)
            continue
        sv[c] = d["volume"].reindex(idx)
        sd[c] = d["delta"].reindex(idx)
        covered.append(c)
    return (pd.DataFrame(sv, columns=coins),
            pd.DataFrame(sd, columns=coins),
            covered)


def build():
    """Perp panel from book.py (unchanged), spot panel aligned to it."""
    idx, PX, R, VOL, DN = book.panel()
    coins = list(R.columns)
    SVOL, SDN, covered = spot_panel(idx, coins)
    FUND, fcov = funding_panel(idx, coins)

    # normalised flow, both tapes. Divide-by-zero -> NaN, never 0: a day with
    # no volume has no flow reading, and pretending it is neutral flow would
    # put that coin in the middle of the cross-section every quiet day.
    PFN = DN / VOL.replace(0, np.nan)
    SFN = SDN / SVOL.replace(0, np.nan)
    return (idx, PX, R, VOL, DN, SVOL, SDN, PFN, SFN, covered, FUND, fcov)


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def sig_div(SFN, PFN, k):
    """The divergence signal. z-score each leg cross-sectionally, subtract."""
    def f(t):
        s = Z(SFN.iloc[t - k:t].sum(min_count=1))
        p = Z(PFN.iloc[t - k:t].sum(min_count=1))
        return s - p
    return f


def sig_spot_only(SFN, k):
    """Control: is spot flow just a better `delta`? If this scores as well as
    divergence, there is no divergence edge -- only a cleaner tape."""
    return lambda t: Z(SFN.iloc[t - k:t].sum(min_count=1))


def sig_perp_only(PFN, k):
    """Control: reproduces the existing `delta` sleeve's construction. Any
    divergence result has to be read against THIS, not against zero."""
    return lambda t: Z(PFN.iloc[t - k:t].sum(min_count=1))


# --------------------------------------------------------------------------
# FUNDING-CONDITIONED VARIANTS -- a PRE-REGISTERED grid, not a free search
# --------------------------------------------------------------------------
# The plain divergence above throws away the most informative fact: which
# side is CROWDED and paying to stay there. Funding is exactly that.
#
#   spot selling + POSITIVE funding -> holders distributing into leveraged
#                                      longs who are paying to hold. Short.
#   spot buying  + NEGATIVE funding -> cash accumulating while shorts pay
#                                      to stay short. Long.
#
# Sign convention: positive funding = longs pay shorts. So a LONG candidate
# wants high spot flow and LOW (negative) funding, hence `- z(funding)`
# everywhere below.
#
# DISCIPLINE: exactly three variants x two lookbacks = six configs, fixed
# before the data arrived. Confluence multiplies the search space, and the
# measured null says best-of-21 reaches +1.1 Sharpe on pure noise; a free
# search over spot x funding x OI x lookback x hold would push that past
# 1.8 and make any result meaningless. Every config here is counted in the
# Bonferroni correction.
#
# CONTEXT YOU SHOULD NOT FORGET: funding already FAILED standalone in both
# directions (fade and momentum both lose; costs dominate). A conditioning
# variable does not need standalone directional edge -- that is a real
# distinction -- but this is not a strong starting position.


def sig_sfund_add(SFN, FUND, k):
    """Additive confluence: spot flow plus crowding. Keeps the full
    cross-section, so nothing is thrown away."""
    def f(t):
        s = Z(SFN.iloc[t - k:t].sum(min_count=1))
        fu = Z(FUND.iloc[t - k:t].sum(min_count=1))
        return s - fu
    return f


def sig_sfund_div(SFN, PFN, FUND, k):
    """All three tapes: spot-vs-perp divergence, plus crowding."""
    def f(t):
        s = Z(SFN.iloc[t - k:t].sum(min_count=1))
        p = Z(PFN.iloc[t - k:t].sum(min_count=1))
        fu = Z(FUND.iloc[t - k:t].sum(min_count=1))
        return (s - p) - fu
    return f


def sig_sfund_mask(SFN, FUND, k):
    """Dante's version, literally: score ONLY the coins where spot flow and
    funding genuinely oppose each other -- spot selling into positive
    funding, or spot buying into negative funding. Coins without that
    confluence get NaN and drop out of the cross-section.

    This is the most faithful to the economic story and the most fragile:
    it thins the cross-section, and a thin cross-section has a much wider
    null, which is exactly how a high-variance config wins a sweep by luck.
    --audit reports how many coins it actually scores."""
    def f(t):
        s = SFN.iloc[t - k:t].sum(min_count=1)
        fu = FUND.iloc[t - k:t].sum(min_count=1)
        score = Z(s) - Z(fu)
        return score.where(np.sign(s) != np.sign(fu))
    return f


def sig_spot_demean(SFN, k, win=252):
    """Spot flow MINUS each coin's own trailing average spot flow.

    WHY THIS EXISTS: --diag showed a fixed portfolio of spot_only's average
    weights earns Sharpe 0.62 on its own -- 38% of the 1.65. That static
    part is a bet on coin CHARACTERISTICS: a coin's average spot-buy ratio
    over years reflects its float, holder base and which venues its volume
    sits on, not anything timely. Ranking on a static characteristic is the
    component most likely to vanish out of sample.

    Subtracting each coin's own trailing mean removes it, leaving only
    "is this coin's spot flow unusual FOR THIS COIN right now". If the
    Sharpe survives, it is genuine timing. If it collapses to ~0.6, the
    original result was mostly the tilt.

    The baseline window is TRAILING and ends at t, never including bar t,
    so this stays causal. An expanding or full-sample mean would leak the
    future into the baseline -- a subtle lookahead that would look like a
    free improvement."""
    def f(t):
        lo = max(0, t - win)
        cw = SFN.iloc[t - k:t]
        bw = SFN.iloc[lo:t]
        cur = cw.mean().where(cw.notna().sum() >= 1)
        # a "long-run average" built from a handful of points is not one;
        # require real history or the coin drops out of the cross-section
        base = bw.mean().where(bw.notna().sum() >= MIN_BASE_OBS)
        return Z(cur - base)
    return f


def sig_spot_std(SFN, k, win=252):
    """Per-coin STANDARDISED spot flow: subtract each coin's own trailing
    mean AND divide by its own trailing standard deviation.

    WHY: --diag found that dropping ZEC takes spot_only from 1.65 to 0.98.
    One coin should not be able to decide a 23-coin cross-sectional result.
    The reason it can is that a cross-sectional z-score ranks coins against
    EACH OTHER, so a coin whose flow reading is structurally more dispersed
    (thin spot book, concentrated holder base) sits in the top or bottom 5
    almost every rebalance regardless of whether anything happened.

    Demeaning alone (sig_spot_demean) removes a coin's average LEVEL but
    not its SCALE, so the most volatile reader still dominates the
    extremes. Dividing by its own std puts every coin on the same footing:
    the question becomes "how unusual is this for THIS coin", which is the
    only form in which the comparison across coins is fair.

    Trailing windows end at t and never include it."""
    def f(t):
        lo = max(0, t - win)
        cw = SFN.iloc[t - k:t]
        bw = SFN.iloc[lo:t]
        cur = cw.mean().where(cw.notna().sum() >= 1)
        nobs = bw.notna().sum()
        mu = bw.mean().where(nobs >= MIN_BASE_OBS)
        sd = bw.std().where(nobs >= MIN_BASE_OBS)
        return Z((cur - mu) / sd.replace(0, np.nan))
    return f


def sig_svp(SVOL, VOL, k, win=252):
    """SPOT-to-PERP VOLUME ratio -- not direction, but WHO is trading.
    A high spot share means cash-driven activity; a low one means the
    action is in leverage. Nothing in the book reads this.

    THE SCALE TRAP, and it would have silently ruined this: perps use 1000x
    contracts for small-unit coins, so 1000PEPE's spot base volume is 1000x
    its perp base volume purely as a unit convention. A cross-sectional
    z-score CANNOT remove a per-coin constant offset -- it would park every
    1000x coin permanently at one end of the ranking and the sleeve would
    be a bet on which coins have a 1000x contract.

    So the log ratio is demeaned PER COIN against its own trailing average
    before ranking. That removes any constant unit offset exactly, because
    a constant in log space is a constant to subtract."""
    def f(t):
        lo = max(0, t - win)
        sv = SVOL.iloc[t - k:t].sum(min_count=1)
        pv = VOL.iloc[t - k:t].sum(min_count=1)
        cur = np.log((sv / pv.replace(0, np.nan)).replace(0, np.nan))

        bs = SVOL.iloc[lo:t]
        bp = VOL.iloc[lo:t]
        base = np.log((bs / bp.replace(0, np.nan)).replace(0, np.nan))
        nobs = base.notna().sum()
        mu = base.mean().where(nobs >= MIN_BASE_OBS)
        sd = base.std().where(nobs >= MIN_BASE_OBS)
        return Z((cur - mu) / sd.replace(0, np.nan))
    return f


def sig_spot_level(SFN, win=252):
    """The static tilt made EXPLICIT: rank purely on each coin's trailing
    average spot flow, with no recency at all. This is a control, not a
    candidate -- if it scores near the 0.62 that --diag's static test
    found, the decomposition is confirmed and we know exactly how much of
    spot_only is characteristic rather than signal."""
    def f(t):
        lo = max(0, t - win)
        bw = SFN.iloc[lo:t]
        return Z(bw.mean().where(bw.notna().sum() >= MIN_BASE_OBS))
    return f


def sig_disagree(SFN, PFN, k):
    """Only score coins where the two tapes genuinely disagree in SIGN.
    Agreement is just directional consensus, which `delta` already ranks.
    Coins in agreement get NaN -> excluded from the cross-section."""
    def f(t):
        s = SFN.iloc[t - k:t].sum(min_count=1)
        p = PFN.iloc[t - k:t].sum(min_count=1)
        d = Z(s) - Z(p)
        return d.where(np.sign(s) != np.sign(p))
    return f


# --------------------------------------------------------------------------
# audit: the checks that would have caught the previous five bugs
# --------------------------------------------------------------------------
def audit() -> int:
    log("  AUDIT -- data integrity and signal timing\n")
    worst = 0

    if SPOT is None or not os.path.isdir(SPOT):
        log("  FAIL cannot find the spot panel. Paths tried:")
        for p_ in _SPOT_TRIED:
            log(f"      {p_}")
        log("  fix: python3 -u fetch_spot.py --run"
            "   (or set HYRO_SPOT_DATA_DIR)")
        return 2

    (idx, PX, R, VOL, DN, SVOL, SDN, PFN, SFN, covered,
     FUND, fcov) = build()
    missing = [c for c in R.columns if c not in covered]

    log(f"  spot dir   : {SPOT}")
    log(f"  perp panel : {len(R.columns)} coins, {len(idx)} days "
        f"({idx[0].date()} -> {idx[-1].date()})")
    log(f"  spot cover : {len(covered)}/{len(R.columns)} coins")
    if FUND_DIR is None:
        log(f"  funding    : NOT FOUND -- confluence variants will be SKIPPED")
        log(f"               paths tried: {', '.join(_FUND_TRIED)}")
        log(f"               fix: python3 -u fetch_funding.py --run")
        worst = max(worst, 1)
    else:
        log(f"  funding    : {len(fcov)}/{len(R.columns)} coins from {FUND_DIR}")
        if not fcov:
            log(f"               no usable funding series -- confluence "
                f"variants SKIPPED")
            worst = max(worst, 1)
    if missing:
        log(f"  no spot    : {', '.join(missing)}")
        if len(covered) < 2 * 5 + 2:
            log(f"  FAIL only {len(covered)} coins with spot -- cannot fill "
                f"both sides of a 5-per-side cross-section")
            worst = 2
        elif len(covered) < 14:
            log(f"  WARN {len(covered)} coins is a thin cross-section; a "
                f"result here is fragile by construction")
            worst = max(worst, 1)

    # ---- 1. normalised flow must sit in [-1, 1]. Outside that range means
    #         delta and volume came from different venues (the clean_panel bug)
    log("")
    for name, M in (("perp", PFN), ("spot", SFN)):
        v = M.values[np.isfinite(M.values)]
        if v.size == 0:
            log(f"  FAIL {name} normalised flow is entirely NaN")
            worst = 2
            continue
        lo, hi = float(v.min()), float(v.max())
        bad = int(((v < -1.0000001) | (v > 1.0000001)).sum())
        tag = "OK" if bad == 0 else f"FAIL {bad} values outside [-1,1]"
        log(f"  {name} flow range [{lo:+.3f}, {hi:+.3f}]  {tag}")
        if bad:
            worst = 2

    # ---- 2. the two tapes must not be the same series. If corr ~ 1.0 there
    #         is no divergence to trade and the whole premise is dead.
    log("")
    cors = []
    for c in covered:
        a, b = SFN[c], PFN[c]
        m = a.notna() & b.notna()
        if m.sum() > 200:
            cors.append((c, float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())))
    if not cors:
        log("  FAIL no coin has 200+ overlapping spot/perp days")
        return 2
    vals = [c[1] for c in cors]
    log(f"  spot-vs-perp flow correlation, per coin:")
    log(f"    median {np.median(vals):+.3f}   "
        f"min {min(vals):+.3f} ({min(cors, key=lambda x: x[1])[0]})   "
        f"max {max(vals):+.3f} ({max(cors, key=lambda x: x[1])[0]})")
    if np.median(vals) > 0.95:
        log("  FAIL the two tapes are the same series -- nothing to diverge")
        worst = 2
    elif np.median(vals) > 0.85:
        log("  WARN tapes are very similar; expect a weak residual")
        worst = max(worst, 1)
    else:
        log("  OK   tapes are distinct -- there is a residual to trade")

    # ---- 3. TIMING. Every signal must be blind to bar t. Feed it a panel
    #         whose last row is poisoned and confirm the score does not move.
    log("")
    t = len(idx) - 1
    builders = dict(_grid(have_funding=bool(fcov), SVOL=SVOL, VOL=VOL))
    # one representative per family is enough; the lookback does not change
    # which bars a slice touches
    fam, seen = {}, set()
    for nm, (b, _n, _h) in builders.items():
        key = nm.split(" k=")[0]
        if key not in seen:
            seen.add(key)
            fam[key] = b
    for name, mk in fam.items():
        clean = mk(SFN, PFN, FUND)(t)
        Sp, Pp, Fp = SFN.copy(), PFN.copy(), FUND.copy()
        Sp.iloc[t] = 0.99          # poison bar t only
        Pp.iloc[t] = -0.99
        Fp.iloc[t] = 0.99
        dirty = mk(Sp, Pp, Fp)(t)
        a = clean.reindex(sorted(clean.index)).astype(float)
        b = dirty.reindex(sorted(dirty.index)).astype(float)
        same = np.allclose(a.fillna(-999), b.fillna(-999), atol=1e-12)
        log(f"  {name:<10} blind to bar t: {'OK' if same else 'FAIL LOOKAHEAD'}")
        if not same:
            worst = 2

    # ---- 4. the signal must actually fire, on enough coins
    log("")
    for name, mk in fam.items():
        n_valid = [int(mk(SFN, PFN, FUND)(tt).notna().sum())
                   for tt in range(60, len(idx), 50)]
        log(f"  {name:<10} median coins scored per rebalance: "
            f"{int(np.median(n_valid))}")
        if np.median(n_valid) < 12:
            log(f"             WARN under 12 -- cross-section too thin to "
                f"fill 5 long + 5 short reliably")
            worst = max(worst, 1)

    log("")
    log({0: "  CLEAN -- proceed to --sweep",
         1: "  WARNINGS -- read them before trusting any Sharpe below",
         2: "  DO NOT PROCEED -- fix the FAILs first"}[worst])
    return worst


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def row(label, s, extra=""):
    h = len(s) // 2
    s1 = stats(s.iloc[:h])[0]
    s2 = stats(s.iloc[h:])[0]
    sh, ann = stats(s)
    both = "yes" if s1 > 0 and s2 > 0 else "NO"
    log(f"  {label:<22}{sh:>8.2f}{ann:>8.1f}%{s1:>8.2f}{s2:>8.2f}"
        f"{both:>6}  {extra}")
    return sh


def _roll(M: pd.DataFrame, off: int) -> pd.DataFrame:
    """Circular time shift. Index and columns unchanged, values rotated."""
    return pd.DataFrame(np.roll(M.values, off, axis=0),
                        index=M.index, columns=M.columns)


def null_dist(R, builder, SFN, PFN, FUND, n, hold, seeds=60, seed0=0,
              runner=None):
    """Circular-time-shift null for ONE config.

    `builder(S, P)` must return a signal function, so the same construction
    is used for the null as for the real run -- no second implementation to
    drift. Rolling the signal panels preserves the signal's autocorrelation
    and cross-sectional shape, and destroys only its time alignment with
    forward returns.

    Offsets avoid small shifts, which would barely decorrelate the signal
    from returns and so produce an optimistically HIGH null."""
    T = len(R)
    lo = max(30, int(0.05 * T))
    rng = np.random.default_rng(seed0)
    out = []
    for _ in range(seeds):
        off = int(rng.integers(lo, T - lo))
        run = runner if runner is not None else xs
        s = run(R, builder(_roll(SFN, off), _roll(PFN, off),
                           _roll(FUND, off)), n=n, hold=hold)
        out.append(stats(s)[0])
    return np.array(out)


# config name -> (builder, n_per_side, hold). One table, so the sweep, the
# null and the verdict can never be computing different things.
def _grid(have_funding: bool = False, SVOL=None, VOL=None):
    """Every builder takes (SFN, PFN, FUND) whether it uses all three or not,
    so null_dist can roll all three panels through one call signature. If
    funding is rolled while the rest is not, the funding leg stays aligned
    with forward returns and the null is silently too easy."""
    g = {}
    for k in (1, 2, 3, 5, 7):
        for hold in (5, 7, 10):
            g[f"div k={k} h={hold}"] = (
                (lambda kk: (lambda S, P, F: sig_div(S, P, kk)))(k), 5, hold)
    for k in (3, 5):
        g[f"spot_only k={k}"] = (
            (lambda kk: (lambda S, P, F: sig_spot_only(S, kk)))(k), 5, 7)
        g[f"perp_only k={k}"] = (
            (lambda kk: (lambda S, P, F: sig_perp_only(P, kk)))(k), 5, 7)
        g[f"disagree k={k}"] = (
            (lambda kk: (lambda S, P, F: sig_disagree(S, P, kk)))(k), 5, 7)

    # The pre-registered confluence grid: 3 variants x 2 lookbacks = 6.
    # Skipped entirely when there is no funding data, so a missing input
    # narrows the test rather than silently changing it.
    # Decomposition of the spot_only result: demeaned (pure timing) and
    # level-only (pure characteristic). Pre-registered with a stated
    # purpose, and counted in the Bonferroni like everything else.
    for k in (3, 5):
        g[f"spot_dm k={k}"] = (
            (lambda kk: (lambda S, P, F: sig_spot_demean(S, kk)))(k), 5, 7)
    g["spot_level"] = (lambda S, P, F: sig_spot_level(S), 5, 7)
    # per-coin STANDARDISED (scale as well as level), so no single coin's
    # dispersion can own the extremes
    for k in (3, 5):
        g[f"spot_std k={k}"] = (
            (lambda kk: (lambda S, P, F: sig_spot_std(S, kk)))(k), 5, 7)
    # spot-vs-perp VOLUME ratio: who is trading, not which way. Needs the
    # raw volume panels, so it is only offered when they are supplied.
    if SVOL is not None and VOL is not None:
        for k in (3, 5):
            g[f"svp k={k}"] = (
                (lambda kk: (lambda S, P, F: sig_svp(SVOL, VOL, kk)))(k), 5, 7)

    if have_funding:
        for k in (3, 5):
            g[f"sfund_add k={k}"] = (
                (lambda kk: (lambda S, P, F: sig_sfund_add(S, F, kk)))(k), 5, 7)
            g[f"sfund_div k={k}"] = (
                (lambda kk: (lambda S, P, F: sig_sfund_div(S, P, F, kk)))(k),
                5, 7)
            g[f"sfund_mask k={k}"] = (
                (lambda kk: (lambda S, P, F: sig_sfund_mask(S, F, kk)))(k),
                5, 7)
    return g


def sweep(n_null=200, panels=None, quiet=False, runner=None) -> int:
    if panels is None:
        rc = audit()
        if rc == 2:
            return 2
        panels = build()
    (idx, PX, R, VOL, DN, SVOL, SDN, PFN, SFN, covered,
     FUND, fcov) = panels

    have_f = bool(fcov)
    grid = _grid(have_funding=have_f, SVOL=SVOL, VOL=VOL)
    run = runner if runner is not None else xs
    if runner is not None and not quiet:
        log("  WEIGHTS: equal RISK inside the sleeve (inverse trailing vol),")
        log("  not equal dollars. book.xs gives every position 0.5/n, so the")
        log("  highest-vol holding dominates sleeve PnL -- which is why one")
        log("  coin can decide the result.")
    if not quiet:
        log("\n" + "=" * 78)
        log("  SWEEP")
        log("=" * 78)
        log(f"  {'config':<22}{'Sharpe':>8}{'ann':>9}{'1st':>8}{'2nd':>8}"
            f"{'both':>6}")

    results, series, eligible = {}, {}, {}
    order = [k for k in grid if k.startswith("div")]
    rest = [k for k in grid if not k.startswith("div")]

    def run_one(name):
        b, n, hold = grid[name]
        sig = b(SFN, PFN, FUND)
        # How many coins does this config actually score? book.xs needs
        # 2n+2 valid names or it SKIPS the rebalance and holds the previous
        # position. A config scoring fewer is not running the strategy on
        # the label -- it is a low-turnover stale-carry variant whose Sharpe
        # is not comparable to the others, and must not be crowned winner.
        # (`disagree` scores a median of 8 of 24 on the real panel.)
        need = 2 * n + 2
        scored = [int(sig(tt).notna().sum())
                  for tt in range(60, len(R), 25)]
        med = float(np.median(scored)) if scored else 0.0
        frac_ok = float(np.mean([x >= need for x in scored])) if scored else 0.0
        series[name] = run(R, sig, n=n, hold=hold)
        eligible[name] = (frac_ok >= 0.80, med, frac_ok, need)
        tag = "" if frac_ok >= 0.80 else \
            f"<- SKIPS {100*(1-frac_ok):.0f}% of rebalances (scores {med:.0f}/{need})"
        results[name] = row(name, series[name], tag) if not quiet \
            else stats(series[name])[0]

    for name in order:
        run_one(name)

    if not quiet:
        log("")
        log("  CONTROLS -- divergence must beat BOTH single tapes, or there")
        log("  is no divergence edge, only a cleaner tape or old `delta`.")
    for name in rest:
        run_one(name)

    # -------- null for the WINNING config, not an arbitrary one -----------
    # Winner chosen only among configs that actually execute their
    # rebalances. A disqualified config is still shown above, flagged.
    runnable = [k for k in results if eligible[k][0]]
    dq = [k for k in results if not eligible[k][0]]
    if not runnable:
        if not quiet:
            log("")
            log("  ABORT: every config skips >20% of its rebalances. There is")
            log("  no strategy here to measure -- the cross-section is too")
            log("  thin for n=5 per side. Lower n or widen the universe.")
        return 2, None, 0.0, 1.0, series, np.array([0.0])
    if dq and not quiet:
        log("")
        log(f"  DISQUALIFIED from winning (skip >20% of rebalances, so they")
        log(f"  hold stale positions instead of the stated hold):")
        for k in dq:
            _, med, frac, need = eligible[k]
            log(f"    {k:<22} scores {med:.0f}/{need} needed, "
                f"executes {100*frac:.0f}%")
    best = max(runnable, key=results.get)
    bsh = results[best]
    b, n, hold = grid[best]

    if not quiet:
        log("")
        log(f"  NULL for the winning config ({best}), {n_null} circular")
        log(f"  time shifts. Turnover and fee drag preserved; only the")
        log(f"  alignment with forward returns is destroyed.")
    null = null_dist(R, b, SFN, PFN, FUND, n, hold, seeds=n_null,
                     runner=run)

    n_cfg = len(results)

    # ---- p-values. Two estimators, because each one alone misleads here.
    #
    # Empirical: (1 + #{null >= real}) / (1 + n). The +1s are not cosmetic --
    # a plain mean() returns p=0.000 when no null draw beats the real value,
    # and a finite permutation test can never justify p=0. That exact bug let
    # a NOISE config through this verdict during development (it scored +0.89
    # against a null of -0.26+-0.62 with 0 of 40 draws above, printing
    # p=0.000); only the both-halves rule caught it.
    #
    # The empirical estimator's floor is 1/(1+n), so with n=200 the smallest
    # Bonferroni-corrected p it can express is 21/201 = 0.10 -- too coarse to
    # ever clear 0.05. Resolving that empirically needs n >~ n_cfg/0.05 = 420+
    # draws. So the verdict uses a Gaussian tail fitted to the null, and the
    # empirical p is reported alongside as a check ON THAT ASSUMPTION: if the
    # two disagree badly, the null is not Gaussian and the Gaussian p is void.
    n_above = int((null >= bsh).sum())
    p_emp = (1.0 + n_above) / (1.0 + len(null))
    p_emp_floor = 1.0 / (1.0 + len(null))
    sd = float(null.std(ddof=1))
    z = (bsh - float(null.mean())) / sd if sd > 0 else 0.0
    # normal survival function without scipy
    from math import erfc, sqrt
    p_gauss = 0.5 * erfc(z / sqrt(2.0))

    p_adj = min(1.0, p_gauss * n_cfg)
    p_adj_emp = min(1.0, p_emp * n_cfg)
    e_max = float(np.percentile(null, 100 * (1 - 1.0 / n_cfg))) \
        if n_cfg > 1 else float(null.max())

    if not quiet:
        log(f"    null mean {null.mean():+.2f}  std {sd:.2f}  "
            f"min {null.min():+.2f}  max {null.max():+.2f}  (n={len(null)})")
        log(f"    real {bsh:+.2f}   z vs null {z:+.2f}")
        log(f"    p Gaussian  {p_gauss:.4f}  -> Bonferroni x{n_cfg} "
            f"{p_adj:.3f}   <- verdict uses this")
        log(f"    p empirical {p_emp:.4f}  -> Bonferroni x{n_cfg} "
            f"{p_adj_emp:.3f}   ({n_above}/{len(null)} draws >= real)")
        if n_above == 0:
            log(f"    NOTE empirical p is AT ITS FLOOR ({p_emp_floor:.4f}); it"
                f" cannot resolve further.")
            log(f"         for an empirical verdict re-run with "
                f"--nulls {int(np.ceil(n_cfg / 0.05))}")
        if p_adj_emp <= 0.05 and p_adj > 0.05:
            log(f"    NOTE the two estimators DISAGREE -- treat the Gaussian "
                f"p as void and trust the empirical one")
        log(f"    E[max of {n_cfg} nulls] ~ {e_max:+.2f}")

        log("")
        log("=" * 78)
        log(f"  best config: {best}   Sharpe {bsh:.2f}")
        h = len(series[best]) // 2
        s1 = stats(series[best].iloc[:h])[0]
        s2 = stats(series[best].iloc[h:])[0]
        if p_adj > 0.05:
            log(f"  VERDICT: NOISE.")
            log(f"  After correcting for the {n_cfg} configs tried, p={p_adj:.2f}."
                f" Do not wire this sleeve.")
        elif not (s1 > 0 and s2 > 0):
            log(f"  VERDICT: REJECT -- halves {s1:.2f} / {s2:.2f}.")
            log(f"  Significant overall but not in both halves, which is the")
            log(f"  signature of one regime carrying the whole result.")
        else:
            log(f"  VERDICT: survives. p(Bonferroni)={p_adj:.3f}, "
                f"halves {s1:.2f} / {s2:.2f}.")
            log(f"  STILL REQUIRED before wiring:")
            log(f"    1. --corr   (< 0.3 against all nine live sleeves)")
            log(f"    2. beat spot_only AND perp_only above, or it is not a")
            log(f"       divergence edge -- just a cleaner or older tape")
            log(f"    3. re-run on a coin set chosen out of sample")
        log("=" * 78)

    return 0, best, bsh, p_adj, series, null


# --------------------------------------------------------------------------
# selftest: the verdict must reject a panel with no edge in it
# --------------------------------------------------------------------------
def _synth(days, seed, edge=0.0):
    """Synthetic panel. `edge` plants a REAL relationship: next-day return
    is partly driven by today's spot-minus-perp flow divergence, which is
    exactly the hypothesis under test. edge=0 is pure noise."""
    rng = np.random.default_rng(seed)
    coins = CORE24[:]
    idx = pd.date_range("2023-01-01", periods=days, freq="1D")
    C = len(coins)

    def persist(a, rho=0.5):
        out = np.zeros_like(a)
        out[0] = a[0]
        for i in range(1, len(a)):
            out[i] = rho * out[i - 1] + np.sqrt(1 - rho ** 2) * a[i]
        return out

    pf = np.clip(persist(rng.normal(0, .3, (days, C))), -1, 1)
    sf = np.clip(persist(rng.normal(0, .3, (days, C))), -1, 1)

    # returns: idiosyncratic noise + (optionally) a real signal component.
    # The signal uses day t's divergence to drive day t+1's return, so a
    # harness reading .iloc[t-k:t] and earning R[t] can legitimately find it.
    rr = rng.normal(0, 0.03, (days, C))
    if edge:
        div = sf - pf
        rr[1:] += edge * div[:-1]

    PX = pd.DataFrame(100 * np.exp(np.cumsum(rr, axis=0)),
                      index=idx, columns=coins)
    R = PX.pct_change()
    PFN = pd.DataFrame(pf, index=idx, columns=coins)
    SFN = pd.DataFrame(sf, index=idx, columns=coins)
    VOL = pd.DataFrame(1.0, index=idx, columns=coins)
    # synthetic funding: pure noise even in the edge=1 case, so the
    # confluence variants are exercised for CRASHES and LOOKAHEAD without
    # being handed an edge they did not earn
    FUND = pd.DataFrame(persist(rng.normal(0, 3e-4, (days, C))),
                        index=idx, columns=coins)
    return (idx, PX, R, VOL, PFN * VOL, VOL, SFN * VOL, PFN, SFN, coins,
            FUND, coins)


def _verdict(best, bsh, p_adj, series):
    h = len(series[best]) // 2
    s1 = stats(series[best].iloc[:h])[0]
    s2 = stats(series[best].iloc[h:])[0]
    return (p_adj <= 0.05 and s1 > 0 and s2 > 0), s1, s2


def selftest(days=700, n_null=60, seeds=(12345, 777, 20260101)) -> int:
    """Two-sided validation of the verdict logic.

    NEGATIVE control: pure-noise panels must be REJECTED. A harness that
    cannot reject noise manufactures confidence, which is worse than having
    no harness. An earlier version of this file failed exactly this test.

    POSITIVE control: a panel with a real planted divergence edge must be
    ACCEPTED. A harness that rejects everything is equally useless -- it
    would have thrown away the real sleeve along with the seven bad ones.

    Both must pass across several seeds, because a single pass is luck.

    n_null is 60 here, not the 200 --sweep uses. This test checks that the
    verdict LOGIC separates signal from noise, which 60 draws resolve fine;
    it does not need a precise p. At 200 x 6 panels this would be an hour of
    compute on a 1-vCPU box for no extra information.
    """
    log(f"  SELFTEST -- verdict logic, {days}d panels, {len(seeds)} seeds\n")
    fails = []

    log("  NEGATIVE control: pure noise, must be rejected")
    for sd in seeds:
        _, best, bsh, p_adj, series, null = sweep(
            n_null=n_null, panels=_synth(days, sd, edge=0.0), quiet=True)
        accepted, s1, s2 = _verdict(best, bsh, p_adj, series)
        log(f"    seed {sd:<9} best {best:<18} Sharpe {bsh:+.2f}  "
            f"null {null.mean():+.2f}+-{null.std():.2f}  "
            f"p {p_adj:.3f}  halves {s1:+.2f}/{s2:+.2f}  "
            f"-> {'ACCEPTED (BAD)' if accepted else 'rejected (good)'}")
        if accepted:
            fails.append(f"noise seed {sd} was accepted")

    log("")
    log("  POSITIVE control: real planted edge, must be accepted")
    for sd in seeds:
        _, best, bsh, p_adj, series, null = sweep(
            n_null=n_null, panels=_synth(days, sd, edge=0.02), quiet=True)
        accepted, s1, s2 = _verdict(best, bsh, p_adj, series)
        log(f"    seed {sd:<9} best {best:<18} Sharpe {bsh:+.2f}  "
            f"null {null.mean():+.2f}+-{null.std():.2f}  "
            f"p {p_adj:.3f}  halves {s1:+.2f}/{s2:+.2f}  "
            f"-> {'accepted (good)' if accepted else 'REJECTED (BAD)'}")
        if not accepted:
            fails.append(f"planted edge seed {sd} was rejected")

    log("")
    if not fails:
        log("  PASS -- rejects noise, detects a real edge. --sweep is usable.")
        return 0
    for f in fails:
        log(f"  FAIL: {f}")
    log("  Do not trust --sweep until this passes.")
    return 2


# --------------------------------------------------------------------------
# correlation against the live book
# --------------------------------------------------------------------------
def corr(k=3, hold=7) -> int:
    (idx, PX, R, VOL, DN, SVOL, SDN, PFN, SFN, covered,
     FUND, fcov) = build()
    # Compare the sleeve that actually scored, not the dead divergence one.
    # The sweep found every div config flat-to-negative while spot_only k=3
    # scored 1.65, so correlating `div` against the book would be answering
    # a question nobody is asking.
    cand = {
        "spot_only": xs(R, sig_spot_only(SFN, k), n=5, hold=hold),
        "spot_dm": xs(R, sig_spot_demean(SFN, k), n=5, hold=hold),
        "spot_div": xs(R, sig_div(SFN, PFN, k), n=5, hold=hold),
    }

    log(f"  building the live sleeves from book.py for comparison...")
    sl, _, _, _ = book.build()
    try:
        import sr2
        got = sr2.run(6, "break_res", oi_filter=True)
        if got is not None:
            sl["sr"] = got[0]
    except Exception as e:
        log(f"  ! could not build sr sleeve: {e}")

    # Drop anything that is not a usable series. sr2.run() can return None
    # WITHOUT raising, which slipped past the try/except above and crashed
    # on v.index -- a failed build must narrow the comparison, not kill it.
    bad = [kk for kk, v in sl.items()
           if not isinstance(v, pd.Series) or v.dropna().empty]
    for kk in bad:
        log(f"  ! sleeve {kk!r} unavailable ({type(sl[kk]).__name__}) "
            f"-- excluded from the comparison")
        sl.pop(kk)
    if not sl:
        log("  FAIL no live sleeve could be built -- nothing to compare against")
        return 2

    sl.update(cand)
    common = None
    for v in sl.values():
        common = v.index if common is None else common.intersection(v.index)
    if common is None or len(common) < 100:
        log(f"  FAIL only {0 if common is None else len(common)} shared days")
        return 2
    S = {kk: v.reindex(common).fillna(0.0) for kk, v in sl.items()}

    log(f"\n  {len(common)} shared days "
        f"({common[0].date()} -> {common[-1].date()})\n")
    live = [kk for kk in S if kk not in cand]
    log("  CORRELATION of each spot candidate against each live sleeve")
    log("  " + " " * 12 + "".join(f"{kk[:9]:>10}" for kk in live))
    for cname in cand:
        row_ = []
        worst_c = 0.0
        for kk in live:
            c = float(np.corrcoef(S[cname], S[kk])[0, 1])
            worst_c = max(worst_c, abs(c))
            row_.append(f"{c:>+10.2f}")
        flag = "" if worst_c < 0.30 else f"   max |r|={worst_c:.2f} TOO HIGH"
        log(f"    {cname:<10}" + "".join(row_) + flag)

    log("")
    log("  A candidate needs max |r| < 0.30 against every live sleeve to add")
    log("  diversification. Note that `spot_only` is the same construction as")
    log("  `delta` on a different tape, so a high correlation there is")
    log("  expected and means it is a REPLACEMENT, not a tenth sleeve.")
    return 0


# --------------------------------------------------------------------------
# --diag : is the winner a TIMING signal, or a static tilt that paid?
# --------------------------------------------------------------------------
# The failure mode this is built to catch: spot books are thin for alts, so
# delta/volume runs to extremes on illiquid names, and the SAME names land in
# the top/bottom 5 every rebalance. That is not a flow signal -- it is a fixed
# long/short portfolio that happened to work, which is how ZEC's 2.19x risk
# share and "how did RATS make the cut" happened.
#
# This forks xs()'s weight loop to collect the weights, which this file
# otherwise warns against. The fork is therefore PROVEN against xs(): its
# reconstructed PnL must match xs() to 1e-12 or the diagnostic refuses to
# report. Fork for introspection, never for a performance number.

def _collect(R, sig, n, hold):
    """Byte-for-byte replica of book.xs's loop, additionally recording the
    weight vector at every executed rebalance."""
    idx = R.index
    acc = np.zeros(len(idx))
    hist = []                                  # (phase, t, weights)
    for ph in range(hold):
        wp = pd.Series(0.0, index=R.columns)
        out = np.zeros(len(idx))
        for t in range(60, len(idx)):
            r = R.iloc[t].fillna(0.0)
            carry = float((wp * r).sum())
            if (t - ph) % hold != 0:
                out[t] = carry
                continue
            s = sig(t)
            v = s.dropna()
            if len(v) < 2 * n + 2:
                out[t] = carry
                continue
            o = v.sort_values()
            w = pd.Series(0.0, index=R.columns)
            w[o.index[-n:]] = 0.5 / n
            w[o.index[:n]] = -0.5 / n
            out[t] = float((w * r).sum()) - float((w - wp).abs().sum()) * FEE
            wp = w
            hist.append((ph, t, w.copy()))
        acc += out
    return pd.Series(acc[60:] / hold, index=idx[60:]), hist


def diag(cfgname=None) -> int:
    (idx, PX, R, VOL, DN, SVOL, SDN, PFN, SFN, covered,
     FUND, fcov) = build()
    grid = _grid(have_funding=bool(fcov), SVOL=SVOL, VOL=VOL)
    if cfgname is None:
        cfgname = "spot_only k=3"
    if cfgname not in grid:
        log(f"  unknown config {cfgname!r}")
        log(f"  choose one of: {', '.join(sorted(grid))}")
        return 2

    b, n, hold = grid[cfgname]
    sig = b(SFN, PFN, FUND)
    log(f"  DIAGNOSTIC for {cfgname}  (n={n} per side, hold={hold})\n")

    mine, hist = _collect(R, sig, n, hold)
    ref = xs(R, sig, n=n, hold=hold)
    gap = float((mine - ref).abs().max())
    if gap > 1e-12:
        log(f"  ABORT: the instrumented loop does not reproduce book.xs "
            f"(max diff {gap:.2e}).")
        log(f"  Every number below would be measuring a different strategy.")
        return 2
    log(f"  fork matches book.xs exactly (max diff {gap:.1e})  OK")
    sh, ann = stats(ref)
    log(f"  Sharpe {sh:.2f}   ann {ann:.1f}%   {len(hist)} rebalances\n")

    # ---- 1. persistence: is it the same names every time? ----------------
    W = pd.DataFrame([h[2] for h in hist])
    nreb = len(W)
    pct_long = (W > 0).mean() * 100
    pct_short = (W < 0).mean() * 100
    held = pct_long + pct_short
    log("  POSITION PERSISTENCE -- % of rebalances each coin is held")
    log(f"  {'coin':<10}{'long%':>8}{'short%':>8}{'any%':>8}")
    for c in held.sort_values(ascending=False).index[:12]:
        log(f"  {c:<10}{pct_long[c]:>8.1f}{pct_short[c]:>8.1f}"
            f"{held[c]:>8.1f}")
    log(f"\n  a coin held >70% of the time on ONE side is a static tilt,")
    log(f"  not a timing signal. {int((pct_long>70).sum())} coins are "
        f"long >70%, {int((pct_short>70).sum())} short >70%.")
    log(f"  mean coins used at all: {int((held>0).sum())} of {len(held)}")

    # ---- 2. THE KILLER TEST: fixed portfolio of the average weights ------
    # If a static book earns most of the Sharpe, the signal is not timing.
    wbar = W.mean()
    static = (R.fillna(0.0) * wbar).sum(axis=1).reindex(ref.index).fillna(0.0)
    s_sh, s_ann = stats(static)
    log("")
    log("  STATIC-TILT TEST -- hold the AVERAGE weight vector, never retrade")
    log(f"    dynamic signal   Sharpe {sh:>6.2f}   ann {ann:>6.1f}%")
    log(f"    static average   Sharpe {s_sh:>6.2f}   ann {s_ann:>6.1f}%")
    if sh > 0:
        share = 100.0 * max(s_sh, 0.0) / sh
        log(f"    the static book reproduces {share:.0f}% of the Sharpe")
        if share > 70:
            log(f"    VERDICT: this is a STATIC TILT, not a flow signal. The")
            log(f"    edge is which coins it is permanently long/short, which")
            log(f"    is a bet on 23 specific coins over one 3.7-year sample.")
        elif share > 40:
            log(f"    VERDICT: MIXED -- a large part is a static tilt. Treat")
            log(f"    the timing component as the real, smaller result.")
        else:
            log(f"    VERDICT: mostly genuine timing -- the static book does "
                f"not explain it.")

    # ---- 3. leave-one-coin-out: is one coin carrying it? -----------------
    log("")
    log("  LEAVE-ONE-COIN-OUT -- Sharpe with each coin dropped entirely")
    base = sh
    rows = []
    for c in list(R.columns):
        keep = [x for x in R.columns if x != c]
        sig_c = b(SFN[keep], PFN[keep], FUND[keep])
        s_c = xs(R[keep], sig_c, n=n, hold=hold)
        rows.append((c, stats(s_c)[0]))
    rows.sort(key=lambda x: x[1])
    log(f"  {'dropped':<10}{'Sharpe':>8}{'delta':>8}")
    for i_, (c, v) in enumerate(rows[:6]):
        tag = "   <- most damaging to drop" if i_ == 0 else ""
        log(f"  {c:<10}{v:>8.2f}{v-base:>+8.2f}{tag}")
    log(f"  {'(none)':<10}{base:>8.2f}{0.0:>+8.2f}")
    for c, v in rows[-3:]:
        log(f"  {c:<10}{v:>8.2f}{v-base:>+8.2f}")
    worst_drop = base - rows[0][1]
    log("")
    if worst_drop > 0.5:
        log(f"  CONCENTRATED: dropping {rows[0][0]} costs {worst_drop:.2f}"
            f" Sharpe. One coin is carrying a large share of the result.")
    else:
        log(f"  BROAD: no single coin is worth more than {worst_drop:.2f}"
            f" Sharpe. The result does not rest on one name.")
    return 0


# --------------------------------------------------------------------------
# EQUAL-RISK WEIGHTS INSIDE THE SLEEVE
# --------------------------------------------------------------------------
# book.xs assigns every position 0.5/n -- equal DOLLARS. A high-vol coin
# therefore contributes far more variance to sleeve PnL than a low-vol one,
# so whichever name is most volatile dominates the result. ZEC is high-vol
# and has now surfaced as the concentration problem in four separate places;
# that is consistent with a WEIGHTING artifact rather than four independent
# coincidences.
#
# The portfolio layer already equalises risk ACROSS sleeves
# (scale[k] = median(vol)/vol[k]) and then each sleeve internally lets its
# most volatile holding take over. This closes that inconsistency.
#
# `force_equal=True` reproduces book.xs exactly and is used as the parity
# test: identical loop, identical fees, identical carry, only the weights
# differ. Without that proof, a difference in result could be the loop
# rather than the weighting.

def xs_rw(R, sig, n=5, hold=7, vol_win=60, cap=4.0, force_equal=False):
    idx = R.index
    acc = np.zeros(len(idx))
    # trailing per-coin vol. min_periods guards the warmup; the window ends
    # at t-1 at use time, so this never reads the bar being traded.
    V = R.rolling(vol_win, min_periods=30).std()

    for ph in range(hold):
        wp = pd.Series(0.0, index=R.columns)
        out = np.zeros(len(idx))
        for t in range(60, len(idx)):
            r = R.iloc[t].fillna(0.0)
            carry = float((wp * r).sum())
            if (t - ph) % hold != 0:
                out[t] = carry
                continue
            s = sig(t)
            v = s.dropna()
            if len(v) < 2 * n + 2:
                out[t] = carry
                continue
            o = v.sort_values()
            w = pd.Series(0.0, index=R.columns)
            vv = V.iloc[t - 1]                 # causal: vol through t-1

            for names, sign in ((list(o.index[-n:]), 1.0),
                                (list(o.index[:n]), -1.0)):
                if force_equal:
                    for c in names:
                        w[c] = sign * 0.5 / n
                    continue
                iv = {}
                for c in names:
                    x = vv.get(c, np.nan)
                    if np.isfinite(x) and x > 0:
                        iv[c] = 1.0 / x
                if not iv:
                    # no vol estimate for any name on this side -- fall back
                    # to equal weight rather than silently dropping the side
                    for c in names:
                        w[c] = sign * 0.5 / n
                    continue
                ivs = pd.Series(iv)
                # cap relative to the median so one unusually quiet coin
                # cannot absorb the whole side's risk budget
                ivs = ivs.clip(upper=float(ivs.median()) * cap)
                ivs = ivs / ivs.sum() * 0.5
                for c, x in ivs.items():
                    w[c] = sign * float(x)

            out[t] = float((w * r).sum()) - float((w - wp).abs().sum()) * FEE
            wp = w
        acc += out
    return pd.Series(acc[60:] / hold, index=idx[60:])


if __name__ == "__main__":
    if "--audit" in sys.argv:
        raise SystemExit(audit())
    nn = 200
    if "--nulls" in sys.argv:
        try:
            nn = int(sys.argv[sys.argv.index("--nulls") + 1])
        except (IndexError, ValueError):
            log("  ! --nulls needs an integer")
            raise SystemExit(2)
    if "--selftest" in sys.argv:
        raise SystemExit(selftest(n_null=nn))
    if "--sweep" in sys.argv:
        rw = xs_rw if "--rw" in sys.argv else None
        raise SystemExit(sweep(n_null=nn, runner=rw)[0])
    if "--corr" in sys.argv:
        raise SystemExit(corr())
    if "--diag" in sys.argv:
        i = sys.argv.index("--diag")
        cfg = sys.argv[i + 1] if len(sys.argv) > i + 1 \
            and not sys.argv[i + 1].startswith("--") else None
        raise SystemExit(diag(cfg))
    log(__doc__)
    log("  pick one: --audit | --selftest | --sweep | --corr | --diag")
    log("  add --rw to --sweep for equal-RISK weights inside the sleeve")
    raise SystemExit(0)

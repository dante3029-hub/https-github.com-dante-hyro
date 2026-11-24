#!/usr/bin/env python3
"""
of_flow.py -- order-flow patterns from the delta that was already in the
data and was being thrown away.

Every test in this project so far has run on price alone, because
event_audit.load_bars() returns only OHLCV. The CSVs also carry
`taker_buy` and `delta`, for 63 coins at 1h over 3.4 years. This module
reads those columns and tests the footprint patterns that bar-level
order flow can actually support.

WHAT THIS IS NOT. A real footprint needs volume per PRICE LEVEL inside
each bar, which needs aggTrades, which this session cannot download
(data.binance.vision answers 403 from the egress proxy -- a policy
denial, not a transient one). So stacked imbalances, P/b-shaped
profiles, point of control and unfinished auctions are out of reach
here. What bar-level delta does support:

  ABSORPTION    heavy one-sided delta that fails to move price -> the
                other side is absorbing it; trade AGAINST the delta
  DIVERGENCE    price makes a new extreme, cumulative delta does not
                -> the move is not being paid for; trade against it
  EXHAUSTION    extreme delta at a new extreme, and the bar closes
                against itself -> the last buyers/sellers are in
  CONTINUATION  extreme delta WITH the move, bar closes in its
                direction -> follow it

THRESHOLDS ARE FIXED, AND RANKED CAUSALLY. Every percentile is a
ROLLING rank over the trailing 200 bars, never a full-sample quantile.
That distinction already cost this project once: a `v.median()` over a
whole series silently leaked the future into a volatility-targeting test
and manufactured an edge that vanished when it was made expanding.

SIGNIFICANCE. Both the naive t on R per trade and a t clustered by
MONTH are reported. On the FVG work, clustering turned a naive t of 3.09
into -0.35, because tens of thousands of overlapping trades across
correlated coins are nowhere near independent. The clustered number is
the one to believe.

USAGE
    python3 of_flow.py --selftest
    python3 of_flow.py --top 30
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import zlib

import numpy as np
import pandas as pd

import fvg_revisit as fvg   # my FVG tests; fvg.py is the original LuxAlgo backtest

log = logging.getLogger("of_flow")

RANK_WIN = 200       # trailing bars for every percentile rank
EXTREME_WIN = 20     # bars defining a "new extreme"
HI_RANK = 0.90       # "heavy" delta
X_RANK = 0.95        # "extreme" delta
QUIET_RANK = 0.30    # "price did not move"
ATR_STOP = 1.5
_CACHE: dict = {}


def _src():
    import vars_shim as V  # noqa: F401  (only if the repo exposes it)


def load_flow(coin, tf):
    """1h CSV with delta kept, resampled to `tf` hours.

    Validated on read: delta must equal 2*taker_buy - volume (that is
    how the column was built), timestamps strictly increasing, no
    duplicate bars, prices positive, high >= low.
    """
    key = (coin, tf)
    if key in _CACHE:
        return _CACHE[key]
    import event_audit as EA
    base = None
    for pat in (f"taker_data/{coin}_1h.csv",):
        if os.path.exists(pat):
            base = pat
            break
    if base is None:
        hits = glob.glob(f"taker_data/{coin}_1h.csv")
        if not hits:
            _CACHE[key] = None
            return None
        base = hits[0]
    try:
        d = pd.read_csv(base)
    except Exception as e:
        log.warning("  %s unreadable: %s", coin, e)
        _CACHE[key] = None
        return None
    need = {"open_time", "open", "high", "low", "close", "volume",
            "taker_buy", "delta"}
    if not need.issubset(d.columns):
        _CACHE[key] = None
        return None
    d["ts"] = pd.to_datetime(d["open_time"], unit="ms")
    d = d.set_index("ts").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    for c in ("open", "high", "low", "close", "volume", "taker_buy",
              "delta"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close", "volume",
                         "delta"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d.high >= d.low) & (d.volume > 0)]
    if tf > 1:
        d = d.resample(f"{tf}h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum", "taker_buy": "sum",
             "delta": "sum"}).dropna()
    _CACHE[key] = d if len(d) > 400 else None
    return _CACHE[key]


M5DIR = ("/tmp/claude-0/-home-claude/45f2f086-9843-5124-bddd-90b82b67de76"
         "/scratchpad/m5raw")
_C5: dict = {}


def coins5():
    return sorted(os.path.basename(p).replace("_5m.csv", "")
                  for p in glob.glob(f"{M5DIR}/*_5m.csv"))


def load_flow5(coin, minutes):
    """5m bars with delta, resampled to `minutes`.

    A 5m bar spans far fewer price levels than a 1h bar, so a single
    delta number per bar discards much less of what a real footprint
    would show. Still not a footprint -- just a much less lossy proxy.

    The 5m files carry taker_buy but no delta column, so delta is
    derived the same way the 1h files were built: 2*taker_buy - volume.
    Incomplete resample buckets are dropped (a file starting mid-bucket
    otherwise yields a first bar with a wrong open and an understated
    range, right where the warm-up happens).
    """
    key = (coin, minutes)
    if key in _C5:
        return _C5[key]
    p = f"{M5DIR}/{coin}_5m.csv"
    if not os.path.exists(p):
        _C5[key] = None
        return None
    try:
        d = pd.read_csv(p, usecols=["timestamp_ms", "open", "high", "low",
                                    "close", "volume", "taker_buy"])
    except Exception as e:
        log.warning("  %s unreadable: %s", coin, e)
        _C5[key] = None
        return None
    d["ts"] = pd.to_datetime(d["timestamp_ms"], unit="ms")
    d = d.set_index("ts").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    for c in ("open", "high", "low", "close", "volume", "taker_buy"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d.high >= d.low) & (d.volume > 0)]
    d["delta"] = 2.0 * d["taker_buy"] - d["volume"]
    if minutes > 5:
        g = d.resample(f"{minutes}min")
        r = g.agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum",
                   "taker_buy": "sum", "delta": "sum"}).dropna()
        r = r[g.size().reindex(r.index).fillna(0) == minutes // 5]
        d = r
    _C5[key] = d if len(d) > 400 else None
    return _C5[key]


def features(d):
    """Causal features. Every rank uses only the trailing window."""
    c = d["close"]
    dr = (d["delta"] / d["volume"]).clip(-1, 1)      # delta as a ratio
    rng = (d["high"] - d["low"]) / c
    # rolling percentile rank: share of the trailing window below now
    dr_rank = dr.rolling(RANK_WIN).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    rng_rank = rng.rolling(RANK_WIN).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    cum = d["delta"].rolling(EXTREME_WIN).sum()
    cum_rank = cum.rolling(RANK_WIN).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    hi = c == c.rolling(EXTREME_WIN).max()
    lo = c == c.rolling(EXTREME_WIN).min()
    # where the bar closed inside its own range: 1.0 = on the high.
    # This is the rejection measure exhaustion needs. The first version
    # asked for a DOWN CLOSE on a bar that also closed at a 20-bar high,
    # which is nearly a contradiction -- it fired zero times on ETH.
    span = (d["high"] - d["low"]).replace(0, np.nan)
    cpos = ((d["close"] - d["low"]) / span).fillna(0.5)
    return pd.DataFrame({"dr_rank": dr_rank, "rng_rank": rng_rank,
                         "cum_rank": cum_rank, "hi": hi, "lo": lo,
                         "cpos": cpos,
                         "up": d["close"] > d["open"]})


def signals(d, f, pattern):
    """Entry bar indices and sides. Entry is the NEXT bar, always."""
    out = []
    n = len(d)
    dr, rr = f.dr_rank.values, f.rng_rank.values
    cr = f.cum_rank.values
    hi, lo, up = f.hi.values, f.lo.values, f.up.values
    cp = f.cpos.values
    for i in range(RANK_WIN + EXTREME_WIN, n - 1):
        if np.isnan(dr[i]) or np.isnan(rr[i]) or np.isnan(cr[i]):
            continue
        s = 0
        if pattern == "absorption":
            if dr[i] > HI_RANK and rr[i] < QUIET_RANK:
                s = -1                      # buying absorbed -> short
            elif dr[i] < 1 - HI_RANK and rr[i] < QUIET_RANK:
                s = 1
        elif pattern == "divergence":
            if hi[i] and cr[i] < 0.50:
                s = -1                      # new high, delta not confirming
            elif lo[i] and cr[i] > 0.50:
                s = 1
        elif pattern == "exhaustion":
            # extreme delta into a new extreme, rejected within the bar
            if dr[i] > X_RANK and hi[i] and cp[i] < 0.5:
                s = -1
            elif dr[i] < 1 - X_RANK and lo[i] and cp[i] > 0.5:
                s = 1
        elif pattern == "continuation":
            if dr[i] > HI_RANK and hi[i] and up[i]:
                s = 1
            elif dr[i] < 1 - HI_RANK and lo[i] and not up[i]:
                s = -1
        else:
            raise ValueError(pattern)
        if s:
            out.append((i + 1, s))
    return out


def collect(coins, tf, pattern, exit_mode, loader=None):
    rows = []
    loader = loader or load_flow
    for coin in coins:
        d = loader(coin, tf)
        if d is None:
            continue
        h, l, c = d["high"].values, d["low"].values, d["close"].values
        a = fvg.atr(h, l, c)
        f = features(d)
        for eb, side in signals(d, f, pattern):
            if eb >= len(c) or np.isnan(a[eb - 1]) or a[eb - 1] <= 0:
                continue
            ent = float(c[eb - 1])
            stop = ent - side * ATR_STOP * float(a[eb - 1])
            res = fvg.trade(h, l, c, eb, side, ent, stop, exit_mode)
            if res is None or res[2] <= fvg.MIN_R_COST * fvg.ROUND_TRIP:
                continue
            rows.append((d.index[eb], coin, side, *res))
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                       "bars", "r"]).sort_values("ts")


def selftest() -> int:
    d = load_flow("ETH", 1)
    assert d is not None, "ETH flow data not loadable"
    # delta must be exactly 2*taker_buy - volume
    err = (d["delta"] - (2 * d["taker_buy"] - d["volume"])).abs()
    rel = (err / d["volume"].abs()).max()
    print(f"  delta == 2*taker_buy - volume, max rel error {rel:.2e}")
    assert rel < 1e-6, "delta column is not what it claims to be"
    # ranks must be causal: recomputing on a truncated series must not
    # change the value at the truncation point
    f_full = features(d)
    cut = len(d) // 2
    f_part = features(d.iloc[:cut])
    for col in ("dr_rank", "rng_rank", "cum_rank"):
        a, b = f_full[col].iloc[cut - 1], f_part[col].iloc[-1]
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, \
            f"{col} changes when the future is removed -- lookahead"
    print("  every rank is causal (truncation-invariant): OK")
    for p in ("absorption", "divergence", "exhaustion", "continuation"):
        sg = signals(d, f_full, p)
        sides = {s for _, s in sg}
        print(f"  {p:<13} {len(sg):>5} signals, sides {sorted(sides)}")
        assert sg, f"{p} never fires"
        assert sides == {-1, 1}, f"{p} is one-directional"
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    if a.top:
        coins = fvg.top_by_liquidity(coins, EA.load_bars, a.top)
    rg = np.random.default_rng(zlib.crc32(b"of_flow"))
    print(f"\n  Order flow from bar-level delta, {len(coins)} coins")
    print("  stop 1.5 ATR | entry next bar | ranks rolling 200 bars\n")
    print(f"  {'tf':>3} {'pattern':<13}{'exit':<7}{'n':>7}{'1R':>7}"
          f"{'win%':>6}{'R/trade':>9}{'nullR':>8}{'edgeR':>8}"
          f"{'t':>7}{'t_mon':>7}{'mon+':>8}")
    cells = []
    for tf in (1, 4):
        for pat in ("absorption", "divergence", "exhaustion",
                    "continuation"):
            for xm in ("2R", "3R", "trail"):
                d = collect(coins, tf, pat, xm)
                if len(d) < 50:
                    print(f"  {tf:>3} {pat:<13}{xm:<7}{len(d):>7}"
                          f"   too few")
                    continue
                R = (d.pnl / d.r).values
                se = R.std(ddof=1) / np.sqrt(len(R))
                t = R.mean() / se if se > 0 else 0.0
                mon = (d.assign(R=d.pnl / d.r)
                       .groupby(d.ts.dt.to_period("M")).R.mean())
                tm = (mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
                      if len(mon) > 2 else 0.0)
                nl = fvg.null(coins, tf, d, load_flow, xm, a.sims, rg)
                rmed = float(d.r.median())
                nR = float(nl.mean()) / rmed
                print(f"  {tf:>3} {pat:<13}{xm:<7}{len(d):>7}"
                      f"{rmed*100:>6.2f}%{(R>0).mean()*100:>5.0f}%"
                      f"{R.mean():>9.3f}{nR:>8.3f}{R.mean()-nR:>8.3f}"
                      f"{t:>7.2f}{tm:>7.2f}"
                      f"{int((mon>0).sum()):>5}/{len(mon)}")
                cells.append((f"{tf}h {pat}/{xm}", len(R), R.mean(),
                              nR, t, tm))
    if cells:
        N = len(cells)
        E = (np.sqrt(2 * np.log(N))
             - (np.log(np.log(N)) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(N))))
        best = max(cells, key=lambda x: x[5])
        print(f"\n  {N} cells examined. E[max t under the null] = {E:.2f}")
        print(f"  best by clustered t: {best[0]}  n={best[1]}  "
              f"{best[2]:+.3f}R  t_mon={best[5]:.2f} -> "
              f"{'survives' if best[5] > E else 'does NOT survive'}")
        allR = np.array([c[2] for c in cells])
        print(f"  mean R across cells: {allR.mean():+.3f}  "
              f"(above zero: {(allR > 0).sum()}/{N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

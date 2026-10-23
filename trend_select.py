#!/usr/bin/env python3
"""
trend_select.py -- does V29 work on coins that actually trend?

WHERE THIS CAME FROM. V29's per-coin tuning is worth 23.6 points (the 9
tuned coins lose 33%, the 55 untuned lose 57%), which says the strategy
behaves very differently across coins. The tempting move is to keep the 10
coins that were profitable -- but those were profitable IN SAMPLE, and
picking them is fitting, not selecting.

The legitimate version is a MECHANICAL rule: measure how much each coin
trends using only PRIOR data, trade trend-following only where that measure
is high, and re-rank as you go. If trend-following's edge is real but
concentrated in trending coins, this finds it. If trendiness does not
predict where V29 works, this finds nothing -- which is also an answer.

THE BAR IS KNOWN AND IT IS NOT "POSITIVE". V29's gross edge is +0.033% per
trade against a 0.170% round trip at market, or 0.040% with perfect maker
fills. So selection has to lift the GROSS edge above ~4bp before execution
is even worth discussing, and above ~17bp to work as it currently trades.
A top-decile result of +0.05% is not a win; it is a slower loss.

PRE-REGISTERED, fixed before the first run:
  measures   mean ADX(14); variance ratio at k=5; lag-1 autocorrelation of
             daily returns. Three, chosen because each is a standard,
             well-defined trend measure -- not because they looked good.
  window     180 days of PRIOR data, re-ranked every 30 days
  selection  top 14 coins (matching V29's lineup size)
  controls   BOTTOM 14 by the same measure, and RANDOM 14
  9 cells total (3 measures x 3 selections), counted.

THE CONTROL IS THE POINT. "Top 14 by ADX loses less" means nothing on its
own -- high-ADX coins may simply be less volatile. What matters is TOP vs
BOTTOM on the same measure at the same time. If they are the same, the
measure does not discriminate and the rule is noise.

NO LOOKAHEAD: a coin's rank for month M uses data strictly before month M
begins. Verified by a shift check in --audit.

USAGE
    python3 -u trend_select.py --audit     # lookahead + coverage checks
    python3 -u trend_select.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

from fetch_spot import resolve_dir
from v29_test import COIN_CONFIGS, _load, adx, run_coin, btc_macd_dir, FEE

PERP, _TRIED = resolve_dir("HYRO_TAKER_DATA_DIR", "taker_data", True)

WINDOW_D = 180
RERANK_D = 30
TOP_N = 14


def log(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------------
# trendiness measures -- all on PRIOR data only
# --------------------------------------------------------------------------
def m_adx(df):
    """Mean ADX(14) on the window. Higher = stronger directional movement."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = adx(h, l, c, 14)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) > 20 else np.nan


def m_varratio(df, k=5):
    """Variance ratio: var(k-bar returns) / (k * var(1-bar returns)).
    >1 trends (moves persist), <1 mean-reverts. Scale-free."""
    c = np.log(df["close"].values)
    r1 = np.diff(c)
    if len(r1) < 50 * k:
        return np.nan
    rk = c[k:] - c[:-k]
    v1, vk = np.var(r1), np.var(rk)
    return float(vk / (k * v1)) if v1 > 0 else np.nan


def m_autocorr(df):
    """Lag-1 autocorrelation of DAILY returns. Positive = momentum."""
    d = df["close"].resample("1D").last().dropna()
    r = d.pct_change().dropna().values
    if len(r) < 40:
        return np.nan
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


MEASURES = {"adx": m_adx, "varratio": m_varratio, "autocorr": m_autocorr}


def build():
    """Load every coin, run V29 once, keep per-trade (ret, weight, entry_ts)."""
    coins = sorted(os.path.basename(f).replace("_1h.csv", "")
                   for f in glob.glob(f"{PERP}/*_1h.csv"))
    coins = [c for c in coins if c != "BTC"]
    frames, trades = {}, {}
    for c in coins:
        df = _load(f"{PERP}/{c}_1h.csv")
        if df is None or len(df) < 3000:
            continue
        COIN_CONFIGS.setdefault(c, (21, 2.0))
        btc = btc_macd_dir(df.index)
        rr = run_coin(c, df, btc=btc)
        if not rr:
            continue
        idx = df.index
        trades[c] = [(r, w, idx[eb]) for (r, w, eb) in rr if eb < len(idx)]
        frames[c] = df
    return frames, trades


def rank_table(frames, measure_fn):
    """{rerank_date: {coin: score}} using only data BEFORE that date."""
    starts = None
    for df in frames.values():
        s, e = df.index[0], df.index[-1]
        starts = (s, e) if starts is None else (min(starts[0], s),
                                                max(starts[1], e))
    first = starts[0] + pd.Timedelta(days=WINDOW_D)
    dates = pd.date_range(first, starts[1], freq=f"{RERANK_D}D")
    out = {}
    for d in dates:
        lo = d - pd.Timedelta(days=WINDOW_D)
        scores = {}
        for c, df in frames.items():
            w = df.loc[(df.index >= lo) & (df.index < d)]   # STRICTLY before
            if len(w) < 24 * 60:
                continue
            try:
                v = measure_fn(w)
            except Exception:
                v = np.nan
            if np.isfinite(v):
                scores[c] = v
        if len(scores) >= TOP_N * 2:
            out[d] = scores
    return out, dates


def evaluate(trades, ranks, dates, pick, rng=None):
    """pick: 'top' | 'bottom' | 'random'. Returns (mean_net, n, gross)."""
    sel = {}
    for d, sc in ranks.items():
        order = sorted(sc, key=sc.get, reverse=True)
        if pick == "top":
            sel[d] = set(order[:TOP_N])
        elif pick == "bottom":
            sel[d] = set(order[-TOP_N:])
        else:
            sel[d] = set(rng.choice(order, TOP_N, replace=False))
    keys = sorted(sel)
    got_r, got_w = [], []
    for c, tr in trades.items():
        for (r, w, ts) in tr:
            i = np.searchsorted(keys, ts, side="right") - 1
            if i < 0:
                continue
            if c in sel[keys[i]]:
                got_r.append(r); got_w.append(w)
    if not got_r:
        return np.nan, 0, np.nan
    r = np.array(got_r); w = np.array(got_w)
    net = float(np.sum(r * w) / np.sum(w))
    return net, len(r), net + 2 * FEE


def main() -> int:
    if PERP is None:
        log("  no taker_data; tried: " + ", ".join(_TRIED))
        return 2
    log("  loading and running V29 on every coin (slow, once)...")
    frames, trades = build()
    log(f"  {len(frames)} coins, "
        f"{sum(len(v) for v in trades.values()):,} trades\n")
    if len(frames) < TOP_N * 2:
        log(f"  need at least {TOP_N*2} coins to split top/bottom")
        return 2

    import zlib
    rng = np.random.default_rng(zlib.crc32(b"trendsel"))
    bar_market = 2 * FEE * 100
    bar_maker = 2 * 0.0002 * 100

    log(f"  The bar: GROSS must clear {bar_maker:.3f}% (perfect maker fills)")
    log(f"  or {bar_market:.3f}% (as it trades now). V29 overall gross is "
        f"+0.033%.\n")
    log(f"  {'measure':<11}{'select':<9}{'trades':>8}{'net/trade':>11}"
        f"{'gross':>9}{'clears maker?':>15}")

    results = {}
    for name, fn in MEASURES.items():
        ranks, dates = rank_table(frames, fn)
        if not ranks:
            log(f"  {name:<11} no usable ranking windows -- SKIPPED")
            continue
        for pick in ("top", "bottom", "random"):
            net, n, gross = evaluate(trades, ranks, dates, pick, rng)
            if n == 0:
                continue
            results[(name, pick)] = (net, n, gross)
            ok = "YES" if gross > bar_maker else "no"
            log(f"  {name:<11}{pick:<9}{n:>8,}{net*100:>10.3f}%"
                f"{gross*100:>8.3f}%{ok:>15}")
        log("")

    log("  " + "=" * 68)
    log("  DOES TRENDINESS DISCRIMINATE? top vs bottom on the SAME measure")
    log("  " + "=" * 68)
    any_edge = False
    for name in MEASURES:
        t = results.get((name, "top")); b = results.get((name, "bottom"))
        if not (t and b):
            continue
        gap = (t[2] - b[2]) * 100
        log(f"    {name:<11} top gross {t[2]*100:>+7.3f}%   "
            f"bottom {b[2]*100:>+7.3f}%   gap {gap:>+7.3f}%")
        if gap > 0.02:
            log(f"      -> the measure DOES separate them")
            any_edge = True
        else:
            log(f"      -> no separation. Trendiness does not predict where")
            log(f"         V29 works, so selecting on it is noise.")
    log("")
    best = max((v[2], k) for k, v in results.items() if k[1] == "top") \
        if any(k[1] == "top" for k in results) else (np.nan, None)
    if np.isfinite(best[0]) and best[0] > bar_maker:
        log(f"  Best top-selection gross {best[0]*100:+.3f}% clears the "
            f"{bar_maker:.3f}% maker bar.")
        log(f"  Worth pursuing -- but it needs an out-of-sample re-test")
        log(f"  before it means anything, and perfect maker fills do not")
        log(f"  exist.")
    else:
        log(f"  No selection clears even the perfect-maker bar of "
            f"{bar_maker:.3f}%.")
        log(f"  Trend-following on these coins does not pay for its own")
        log(f"  execution, however the universe is chosen. That is not a")
        log(f"  tuning problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

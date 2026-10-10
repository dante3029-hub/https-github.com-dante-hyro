#!/usr/bin/env python3
"""
tp_test.py -- test the fixed-take-profit idea on the REAL panel.

THE IDEA, as specified by the account owner, before any results were seen:
    Go long every coin in the universe. Take profit at exactly +1% (plus
    fees). Use a structural stop loss to bound risk. Once the account is
    +5% for the week, flatten and stand down until next week. Reasoning:
    out of 60 coins, they will certainly move 1% in a week.

The first half of that reasoning is correct -- almost every coin does touch
+1% within a week. The question this file answers is what happens in the
cases where it does not, measured on 1,344 days of real data rather than on
a simulation, because simulated crypto is better behaved than actual crypto.

PRE-REGISTERED SPEC (fixed before the first run, no sweeping afterwards):

  entry        the open of the first hourly bar of each week, every coin
  take profit  +1.0% from entry, limit order, fills intrabar on high
  stop loss    pre-registered set {none, 2%, 5%, 10%}, fills intrabar on low
  max hold     7 days, exit at that bar's close
  weekly halt  flatten and stand down once account PnL >= +5% for the week
  costs        0.085% per side (repo FEE), charged on entry and exit
  sizing       equal weight, leverage L spread over N coins (L/N each)

  SAME-BAR AMBIGUITY: when an hourly bar's high clears the take-profit AND
  its low breaks the stop, the order they occurred is unknowable from OHLC.
  This assumes the STOP filled first. That is the conservative choice and it
  is not optional -- assuming the take-profit filled first is the single most
  common way a backtest of this strategy lies about itself.

  BENCHMARK, and it is the point of the whole exercise: equal-weight
  buy-and-hold of the SAME universe at the SAME leverage with the SAME fees.
  Crypto rose over most of this sample. A long-only strategy will therefore
  make money, and the only question that matters is whether it makes more
  than simply holding. If it does not, the profit is beta, not edge -- and a
  capped upside with an uncapped downside is a strictly worse way to own
  beta than owning it.

  REPORTED REGARDLESS OF OUTCOME: both halves, leave-one-coin-out, worst
  week, and the leverage implied by the +5%/week goal.

USAGE
    python3 -u tp_test.py                  # default stops, L=1
    python3 -u tp_test.py --leverage 5     # the leverage +5%/wk actually needs
    python3 -u tp_test.py --coins 60
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from fetch_spot import resolve_dir

FEE = 0.00085
TP = 0.01
HOLD_HOURS = 7 * 24
WEEKLY_HALT = 0.05
STOPS = (None, 0.02, 0.05, 0.10)

PERP, _TRIED = resolve_dir("HYRO_TAKER_DATA_DIR", "taker_data", True)


def log(m=""):
    print(m, flush=True)


def load(coin):
    p = f"{PERP}/{coin}_1h.csv"
    try:
        df = pd.read_csv(p, usecols=["open_time", "open", "high", "low", "close"])
    except Exception:
        return None
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("ts").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df if len(df) > 5000 else None


def run_coin(df, stop, idx_starts):
    """Returns per-week gross return for ONE coin (before leverage/fees)."""
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    n = len(o)
    out = []
    for s in idx_starts:
        if s >= n - 1:
            out.append(np.nan)
            continue
        e = o[s]
        if not np.isfinite(e) or e <= 0:
            out.append(np.nan)
            continue
        end = min(s + HOLD_HOURS, n - 1)
        tp_px = e * (1 + TP)
        sl_px = e * (1 - stop) if stop else None

        seg_h = h[s:end + 1]
        seg_l = lo[s:end + 1]
        t_hit = np.argmax(seg_h >= tp_px) if (seg_h >= tp_px).any() else None
        s_hit = (np.argmax(seg_l <= sl_px)
                 if (sl_px is not None and (seg_l <= sl_px).any()) else None)

        if t_hit is not None and s_hit is not None:
            # same bar -> assume the STOP filled first (conservative)
            r = -stop if s_hit <= t_hit else TP
        elif t_hit is not None:
            r = TP
        elif s_hit is not None:
            r = -stop
        else:
            r = c[end] / e - 1.0
        out.append(r)
    return np.array(out, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--coins", type=int, default=0, help="0 = all available")
    a = ap.parse_args()

    if PERP is None:
        log("  cannot find taker_data. tried: " + ", ".join(_TRIED))
        return 2

    names = sorted(os.path.basename(f).replace("_1h.csv", "")
                   for f in glob.glob(f"{PERP}/*_1h.csv"))
    if a.coins:
        names = names[:a.coins]
    frames = {}
    for cn in names:
        d = load(cn)
        if d is not None:
            frames[cn] = d
    if len(frames) < 5:
        log(f"  only {len(frames)} usable coins -- aborting")
        return 2

    # common weekly entry points, from the longest series
    ref = max(frames.values(), key=len)
    starts = list(range(0, len(ref) - HOLD_HOURS, HOLD_HOURS))
    log(f"  {len(frames)} coins, {len(starts)} weeks "
        f"({ref.index[0].date()} -> {ref.index[-1].date()})")
    log(f"  leverage {a.leverage:g}x, equal weight, "
        f"fees {FEE*100:.3f}%/side\n")

    # align every coin onto the reference index so weeks line up
    aligned = {}
    for cn, d in frames.items():
        aligned[cn] = d.reindex(ref.index).ffill(limit=3)

    # ---- benchmark: equal-weight buy and hold, same weeks ---------------
    bench = []
    for s in starts:
        e = min(s + HOLD_HOURS, len(ref) - 1)
        rs = []
        for cn, d in aligned.items():
            o = d["open"].values[s]
            cc = d["close"].values[e]
            if np.isfinite(o) and np.isfinite(cc) and o > 0:
                rs.append(cc / o - 1.0)
        bench.append(np.nanmean(rs) if rs else np.nan)
    bench = np.array(bench) * a.leverage - 2 * FEE * a.leverage

    log(f"  {'stop':>7}{'wk mean':>10}{'wk med':>9}{'ann%':>9}"
        f"{'Sharpe':>8}{'1st':>7}{'2nd':>7}{'worst wk':>10}{'win%':>7}")

    results = {}
    for stop in STOPS:
        per = {cn: run_coin(d, stop, starts) for cn, d in aligned.items()}
        M = np.vstack([per[cn] for cn in sorted(per)])
        wk = np.nanmean(M, axis=0) * a.leverage - 2 * FEE * a.leverage
        # weekly halt: cap the week's gain at +5%
        wk_h = np.minimum(wk, WEEKLY_HALT)
        good = np.isfinite(wk_h)
        w = wk_h[good]
        if len(w) < 20:
            continue
        hm = len(w) // 2
        sh = w.mean() / w.std() * np.sqrt(52) if w.std() > 0 else 0.0
        s1 = (w[:hm].mean() / w[:hm].std() * np.sqrt(52)
              if w[:hm].std() > 0 else 0.0)
        s2 = (w[hm:].mean() / w[hm:].std() * np.sqrt(52)
              if w[hm:].std() > 0 else 0.0)
        lbl = "none" if stop is None else f"{stop*100:.0f}%"
        log(f"  {lbl:>7}{w.mean()*100:>9.3f}%{np.median(w)*100:>8.3f}%"
            f"{w.mean()*52*100:>8.1f}%{sh:>8.2f}{s1:>7.2f}{s2:>7.2f}"
            f"{w.min()*100:>9.2f}%{(w>0).mean()*100:>6.0f}%")
        results[lbl] = (w, M)

    bg = bench[np.isfinite(bench)]
    bs = bg.mean() / bg.std() * np.sqrt(52) if bg.std() > 0 else 0.0
    hm = len(bg) // 2
    b1 = bg[:hm].mean() / bg[:hm].std() * np.sqrt(52) if bg[:hm].std() > 0 else 0
    b2 = bg[hm:].mean() / bg[hm:].std() * np.sqrt(52) if bg[hm:].std() > 0 else 0
    log(f"  {'HOLD':>7}{bg.mean()*100:>9.3f}%{np.median(bg)*100:>8.3f}%"
        f"{bg.mean()*52*100:>8.1f}%{bs:>8.2f}{b1:>7.2f}{b2:>7.2f}"
        f"{bg.min()*100:>9.2f}%{(bg>0).mean()*100:>6.0f}%   <- just holding")

    # ---- the leverage the +5%/week goal actually implies ----------------
    log("")
    log("  " + "=" * 68)
    log("  WHAT +5%/WEEK REQUIRES")
    log("  " + "=" * 68)
    log(f"    Every position is capped at +{TP*100:.0f}%. If EVERY coin hits")
    log(f"    its take-profit, the book makes {TP*100:.0f}% x leverage.")
    log(f"    So +5%/week needs {WEEKLY_HALT/TP:.0f}x leverage MINIMUM --")
    log(f"    and that assumes a perfect week where nothing goes wrong.")
    for lev in (1, 5, 10):
        for lbl, (w, M) in results.items():
            if lbl != "5%":
                continue
            scaled = w / a.leverage * lev
            log(f"      at {lev:>2}x: best week {scaled.max()*100:>6.2f}%  "
                f"worst week {scaled.min()*100:>7.2f}%"
                + ("   <-- account gone" if scaled.min() * 100 < -10 else ""))

    # ---- leave-one-coin-out on the best stop ----------------------------
    if results:
        best = max(results, key=lambda k: results[k][0].mean())
        w, M = results[best]
        log("")
        log(f"  LEAVE-ONE-COIN-OUT (stop={best}): is it a few names?")
        base = np.nanmean(M, axis=0)
        cn_sorted = sorted(aligned)
        drops = []
        for i, cn in enumerate(cn_sorted):
            keep = np.delete(M, i, axis=0)
            v = np.nanmean(keep, axis=0)
            drops.append((cn, np.nanmean(v) - np.nanmean(base)))
        drops.sort(key=lambda x: x[1])
        log("    most damaging to drop: " + ", ".join(
            f"{c} ({d*100:+.3f}%/wk)" for c, d in drops[:3]))
        log("    least:                 " + ", ".join(
            f"{c} ({d*100:+.3f}%/wk)" for c, d in drops[-3:]))

    log("")
    log("  " + "=" * 68)
    log("  READ IT THIS WAY: the strategy is long-only over a sample where")
    log("  crypto mostly rose, so a positive number is EXPECTED and is not")
    log("  evidence of edge. The only comparison that means anything is the")
    log("  HOLD row. Beating it would be edge. Losing to it means the profit")
    log("  is beta bought with a capped upside and an uncapped downside.")
    log("  " + "=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

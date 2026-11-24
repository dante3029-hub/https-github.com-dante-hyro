#!/usr/bin/env python3
"""
macd_ema.py -- Dante's own short setup, tested on his terms.

THE RULES AS HE STATED THEM
    "Macd crossing and going negative is my fav short on 1-4h time
     re test of 50day ema for continuation of down trend
     i sort of exit when im happy"

So: MACD crosses below its signal line AND goes below zero; price is
under the 50 EMA (downtrend); wait for the pullback back UP to the 50
EMA; short that retest. Timeframes 1h and 4h.

WHAT IS UNDERSPECIFIED, AND HOW IT IS HANDLED
  exit   -- "when I'm happy" is discretion and cannot be coded. Several
            mechanical exits are run instead, and the SPREAD across them
            is the honest answer: it shows what his judgement would have
            to beat, and whether the setup needs good exits to work at
            all or works regardless.
  stop   -- not stated. Two readings: ATR-based, and just above the
            retest bar's high (the structural stop a discretionary
            trader would actually use).
  "50day ema" on a 1-4h chart is ambiguous: the 50-PERIOD EMA of that
            timeframe, or the DAILY 50 EMA drawn on an intraday chart.
            Both are tested; they are different lines and he may mean
            either.

NONE OF THESE ARE TUNED TO FIT. Every variant is reported, and the
best-looking cell is corrected for the number of cells examined, because
with enough exit choices something always looks good.

THE BAR IT HAS TO CLEAR, same as everything else in this project: beat a
randomly-timed SHORT of the same size on the same coins with the same
exit. Random shorts on this universe are not free money -- alt perps
drift up -- so that is a fair and not a trivial bar.

POWER. From claude/verifier-power.md: below ~50 trades this pipeline
cannot see an edge at all, at n=300 it finds anything above +0.17R, at
n=1000 it finds +0.07R. Trade counts are printed so every verdict can be
read against that.

USAGE
    python3 macd_ema.py --selftest
    python3 macd_ema.py
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

log = logging.getLogger("macd_ema")

# --- indicator settings: textbook MACD, untouched
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
EMA_LEN = 50
ATR_LEN = 14

ROUND_TRIP = 0.0017      # same cost assumption used throughout
MAX_WAIT = 20            # bars allowed for the retest to arrive
MAX_HOLD = 60            # bars before a position is closed regardless
RETEST_TOL = 0.001       # within 0.1% of the EMA counts as touching it


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Causal EMA seeded with an SMA. Returns NaN until it is warm, so a
    half-formed average can never produce a signal."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1.0)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (x[i] - out[i - 1]) * a + out[i - 1]
    return out


def macd(c: np.ndarray):
    """MACD line, signal line. Signal is an EMA of the line, so it only
    becomes valid once the line itself has been valid for MACD_SIG bars."""
    line = ema(c, MACD_FAST) - ema(c, MACD_SLOW)
    ok = ~np.isnan(line)
    sig = np.full(len(c), np.nan)
    if ok.sum() >= MACD_SIG:
        i0 = int(np.flatnonzero(ok)[0])
        sig[i0:] = ema(line[i0:], MACD_SIG)
    return line, sig


def atr(h, l, c, n=ATR_LEN):
    """Wilder-style ATR on a shifted close, so bar i's ATR uses only
    bars up to and including i."""
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return ema(tr, n)


def daily_ema50(df: pd.DataFrame, idx: pd.DatetimeIndex) -> np.ndarray:
    """The DAILY 50 EMA mapped onto an intraday index.

    Each intraday bar gets the last COMPLETED daily EMA value, shifted by
    one day. Without that shift an intraday bar in the middle of a day
    would read an EMA built from its own day's close -- lookahead, and the
    kind that quietly manufactures an edge.
    """
    d = df["close"].resample("1D").last().dropna()
    e = pd.Series(ema(d.values, EMA_LEN), index=d.index).shift(1)
    return e.reindex(idx, method="ffill").values


def signals(df: pd.DataFrame, ema_mode: str):
    """Entry bars for the setup. Returns a list of (entry_bar, meta).

    Sequence enforced, in this order, with no bar used before it closed:
      1. MACD crosses below its signal line
      2. MACD is below zero                      <- "going negative"
      3. close is below the 50 EMA               <- downtrend
      4. price rallies back UP to the EMA        <- the retest
      5. enter short at the NEXT bar's open
    """
    c = df["close"].values
    h, l = df["high"].values, df["low"].values
    line, sig = macd(c)
    if ema_mode == "period":
        e = ema(c, EMA_LEN)
    else:
        e = daily_ema50(df, df.index)
    a = atr(h, l, c)
    n = len(c)
    out = []
    armed_until = -1
    for i in range(1, n - 1):
        if np.isnan(line[i]) or np.isnan(sig[i]) or np.isnan(e[i]):
            continue
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        crossed = line[i] < sig[i] and line[i - 1] >= sig[i - 1]
        if crossed and line[i] < 0 and c[i] < e[i]:
            armed_until = i + MAX_WAIT
            continue
        if i > armed_until:
            continue
        # the retest: this bar reaches up to the EMA from below
        if c[i - 1] < e[i - 1] and h[i] >= e[i] * (1 - RETEST_TOL):
            if i + 1 >= n:
                continue
            out.append((i + 1, {"atr": float(a[i]),
                                "swing_high": float(h[i]),
                                "ema": float(e[i])}))
            armed_until = -1          # one trade per arming
    return out


def run_trade(h, l, c, eb, stop, mode, r):
    """Short from o[eb]; `stop` absolute; `mode` sets the exit.

    A bar touching both stop and target is read as the STOP -- the only
    safe reading when all you have is OHLC.
    """
    n = len(c)
    ent = c[eb - 1] if eb >= 1 else c[eb]
    ent = float(ent)
    if not np.isfinite(ent) or ent <= 0 or stop <= ent:
        return None
    rr = (stop - ent) / ent
    tgt = None
    if mode.endswith("R"):
        tgt = ent * (1 - float(mode[:-1]) * rr)
    best = ent
    for i in range(eb, min(eb + MAX_HOLD, n)):
        if h[i] >= stop:                       # stopped
            return -rr - ROUND_TRIP, i - eb, rr
        if tgt is not None and l[i] <= tgt:
            return (1 - tgt / ent) - ROUND_TRIP, i - eb, rr
        if mode == "trail":
            best = min(best, l[i])
            if c[i] > best * (1 + rr):         # gave back 1R from low
                return (1 - c[i] / ent) - ROUND_TRIP, i - eb, rr
    last = min(eb + MAX_HOLD, n) - 1
    return (1 - c[last] / ent) - ROUND_TRIP, last - eb, rr


def collect(coins, tf, loader, ema_mode, stop_mode, exit_mode):
    rows = []
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        for eb, m in signals(df, ema_mode):
            if stop_mode == "atr":
                stop = c[eb - 1] + 1.5 * m["atr"]
            else:
                stop = m["swing_high"] * 1.001
            res = run_trade(h, l, c, eb, stop, exit_mode, None)
            if res is None:
                continue
            rows.append((df.index[eb], coin, res[0], res[1], res[2]))
    return pd.DataFrame(rows, columns=["ts", "coin", "pnl", "bars", "r"]) \
             .sort_values("ts")


def null(coins, tf, d, loader, exit_mode, stop_mode, sims, rg):
    """Randomly-timed shorts: same coins, same counts, same 1R, same exit."""
    if not len(d):
        return np.array([0.0])
    counts = d.groupby("coin").size().to_dict()
    rmed = float(d.r.median())
    cache = {}
    for coin in counts:
        df = loader(coin, tf)
        if df is not None and len(df) > 400:
            cache[coin] = (df["high"].values, df["low"].values,
                           df["close"].values)
    out = []
    for _ in range(sims):
        acc = []
        for coin, k in counts.items():
            if coin not in cache:
                continue
            h, l, c = cache[coin]
            n = len(c)
            for _i in range(k):
                eb = int(rg.integers(60, n - MAX_HOLD - 2))
                stop = c[eb - 1] * (1 + rmed)
                res = run_trade(h, l, c, eb, stop, exit_mode, None)
                if res is not None:
                    acc.append(res[0])
        if acc:
            out.append(float(np.mean(acc)))
    return np.array(out) if out else np.array([0.0])


def selftest() -> int:
    import event_audit as EA
    df = EA.load_bars("ETH", 4)
    c = df["close"].values
    line, sig = macd(c)
    # MACD must match a straightforward pandas construction
    s = pd.Series(c)
    ref = (s.ewm(span=MACD_FAST, adjust=False).mean()
           - s.ewm(span=MACD_SLOW, adjust=False).mean())
    k = ~np.isnan(line)
    corr = np.corrcoef(line[k][200:], ref.values[k][200:])[0, 1]
    print(f"  MACD vs pandas ewm correlation: {corr:.4f}")
    assert corr > 0.99, "MACD is not a MACD"
    # the daily EMA must never use the same day's close
    de = daily_ema50(df, df.index)
    d1 = df["close"].resample("1D").last().dropna()
    e1 = pd.Series(ema(d1.values, EMA_LEN), index=d1.index)
    t = df.index[len(df) // 2]
    day = t.normalize()
    same = e1.reindex([day]).values[0]
    assert np.isnan(de[len(df) // 2]) or not np.isclose(
        de[len(df) // 2], same), "daily EMA reads its own day -- lookahead"
    print("  daily 50 EMA is shifted, no same-day lookahead: OK")
    sg = signals(df, "period")
    print(f"  ETH 4h: {len(sg)} setups found")
    assert sg, "no setups at all -- detector is not firing"
    # every entry must be preceded by a close under the EMA
    e = ema(c, EMA_LEN)
    bad = [eb for eb, _ in sg if not c[eb - 2] < e[eb - 2]]
    assert not bad, f"{len(bad)} entries without a downtrend precondition"
    print("  every entry sits under the 50 EMA: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    rg = np.random.default_rng(zlib.crc32(b"macd_ema"))

    print("\n  MACD cross negative -> retest of the 50 EMA -> short")
    print("  stop: 1.5 ATR, or just above the retest high")
    print("  exit: his is discretionary, so a spread is shown\n")
    print(f"  {'tf':>3} {'ema':<7}{'stop':<7}{'exit':<7}{'n':>6}{'1R':>7}"
          f"{'win%':>6}{'R/trade':>9}{'nullR':>8}{'edgeR':>8}{'t':>7}")
    cells = []
    for tf in (1, 4):
        for em in ("period", "daily"):
            for sm in ("atr", "swing"):
                for xm in ("1.5R", "2R", "3R", "trail"):
                    d = collect(coins, tf, EA.load_bars, em, sm, xm)
                    if len(d) < 30:
                        print(f"  {tf:>3} {em:<7}{sm:<7}{xm:<7}"
                              f"{len(d):>6}   too few")
                        continue
                    R = (d.pnl / d.r).values
                    se = R.std(ddof=1) / np.sqrt(len(R))
                    t = R.mean() / se if se > 0 else 0.0
                    nl = null(coins, tf, d, EA.load_bars, xm, sm,
                              a.sims, rg)
                    rmed = float(d.r.median())
                    nR = float(nl.mean()) / rmed
                    print(f"  {tf:>3} {em:<7}{sm:<7}{xm:<7}{len(d):>6}"
                          f"{rmed*100:>6.2f}%{(R>0).mean()*100:>5.0f}%"
                          f"{R.mean():>9.3f}{nR:>8.3f}"
                          f"{R.mean()-nR:>8.3f}{t:>7.2f}")
                    cells.append((f"{tf}h {em}/{sm}/{xm}", len(R),
                                  R.mean(), nR, t))
    if cells:
        N = len(cells)
        E = (np.sqrt(2 * np.log(N))
             - (np.log(np.log(N)) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(N))))
        best = max(cells, key=lambda x: x[4])
        print(f"\n  {N} cells examined. E[max t under the null] = {E:.2f}")
        print(f"  best: {best[0]}  n={best[1]}  {best[2]:+.3f}R  "
              f"t={best[4]:.2f}  -> "
              f"{'survives' if best[4] > E else 'does NOT survive'}")
        allR = np.array([c[2] for c in cells])
        print(f"  mean R across all cells: {allR.mean():+.3f}  "
              f"(cells above zero: {(allR > 0).sum()}/{N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
event_audit.py -- do the EVENT sleeves time anything, or are they exposure?

tilt_audit covers the cross-sectional sleeves. Its static-tilt test does not
transfer to the event sleeves: "the average weight vector" is meaningless for
a book that holds three names on its own schedule. These five have therefore
never been tested, and two of them are live right now (fvg $15,000, pattern
$2,500 of gross).

THE RIGHT CONTROL FOR AN EVENT SLEEVE IS RANDOM TIMING.

An event sleeve claims that WHEN it enters matters. So the null is: same
coin, same number of trades, same hold, same stop, same exit logic --
entries at RANDOM bars. If the real signal cannot beat its own random-timing
twin, then the sleeve is not timing anything. It is just exposure to those
coins with extra steps, and the fees are a pure loss.

This is the same discipline the spot-flow work used, applied to a different
shape of strategy: the null must differ from the real thing in EXACTLY ONE
respect -- here, the entry bar -- and be identical in every other way
including trade count, hold, stop placement and the bar-by-bar exit walk.

WHAT IS HELD IDENTICAL, deliberately:
  - trade count per coin (so turnover and fee drag match)
  - hold and stop_atr (so the exit distribution matches)
  - the exit walk itself, including the entry bar, because a stop can be hit
    the moment you are in -- a bug this project already fixed once and must
    not reintroduce in the control
  - entry at the OPEN of the bar after the signal, never the close of the
    signal bar

Reported regardless of outcome: real vs null mean, the empirical p, both
halves, and leave-one-coin-out.

USAGE
    python3 -u event_audit.py                 # fvg + pattern (the live ones)
    python3 -u event_audit.py --all           # adds sr and srflip
    python3 -u event_audit.py --sims 200
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

from fetch_spot import resolve_dir

PERP, _TRIED = resolve_dir("HYRO_TAKER_DATA_DIR", "taker_data", True)

# from signal_engine/sleeve_price_events.py PARAMS -- the LIVE settings
PARAMS = {
    'sr':      dict(tf=6,  stop_atr=3.0, hold=15, which='break'),
    'srflip':  dict(tf=6,  stop_atr=2.0, hold=15, which='flip'),
    'pattern': dict(tf=6,  stop_atr=2.0, hold=15, which='break'),
    'fvg':     dict(tf=12, stop_atr=1.5, hold=10, which='fvg'),
}
LIVE = ('fvg', 'pattern')


def log(m=""):
    print(m, flush=True)


def load_bars(coin, tf):
    p = f"{PERP}/{coin}_1h.csv"
    try:
        df = pd.read_csv(p, usecols=["open_time", "open", "high", "low",
                                     "close", "volume"])
    except Exception:
        return None
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("ts").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if tf > 1:
        df = df.resample(f"{tf}h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    return df if len(df) > 300 else None


def _atr(h, l, c, p=14):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    a = np.full(len(c), np.nan)
    if len(c) > p:
        a[p] = tr[:p].mean()
        for i in range(p + 1, len(c)):
            a[i] = (a[i - 1] * (p - 1) + tr[i - 1]) / p
    return a


def simulate(o, h, l, c, atr, eb, stop_atr, hold, long=True):
    """Outcome of entering at the OPEN of bar eb, with the sleeve's own exit
    rules. Identical for the real signal and the random control -- if these
    ever diverge the comparison is meaningless."""
    n = len(o)
    if eb >= n - 1:
        return None
    a = atr[eb - 1] if eb > 0 else np.nan     # ATR through the PRIOR bar
    if not np.isfinite(a) or a <= 0:
        return None
    ent = o[eb]
    if not np.isfinite(ent) or ent <= 0:
        return None
    stp = ent - stop_atr * a if long else ent + stop_atr * a
    ex = min(eb + hold, n - 1)
    px = c[ex]
    # include the ENTRY bar: a stop can be hit the moment you are in.
    for j in range(eb, min(eb + 1 + hold, n)):
        if (long and l[j] <= stp) or ((not long) and h[j] >= stp):
            px = stp
            break
    return (px / ent - 1.0) if long else (ent / px - 1.0)


def real_entries(coin, sleeve, df):
    """Entry bars the live detector actually produces."""
    p = PARAMS[sleeve]
    try:
        if sleeve == 'pattern':
            # pattern is MarkitTick, NOT the sr break detector. Running
            # sleeve_sr here gave pattern and sr identical trade counts
            # (2,290) and meant pattern was never actually tested.
            # FastPattern indexes by the DETECTION bar -- entering at the
            # historical break bar the detector also reports is lookahead
            # and once inflated a backtest from 1.04 to 6.06.
            from fast_detect import FastPattern
            fp = FastPattern(df, bull_only=True)
            return [i + 1 for i in range(len(df) - 1) if fp.fired(i) == 1], None
        if sleeve == 'fvg':
            import sleeve_fvg as M
            tr = M.backtest(df, None, stop_atr=p['stop_atr'], hold=p['hold'])
        else:
            import sleeve_sr as M
            tr = M.backtest(df, None, which=p['which'],
                            stop_atr=p['stop_atr'], hold=p['hold'])
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    pos = {t: i for i, t in enumerate(df.index)}
    ebs = [pos[t[0]] + 1 for t in tr if t[0] in pos]
    return ebs, None


def audit_sleeve(sleeve, coins, sims, rng=None):
    # Independent, name-seeded RNG per sleeve. A SHARED generator meant
    # fvg drew different nulls depending on what ran before it, and its
    # verdict flipped between 'exposure' (p=0.109) and 'SIGNAL'
    # (p=0.040) across two runs of the SAME test on the SAME data.
    # A conclusion that depends on RNG state is not a conclusion.
    # zlib.crc32, NOT hash(): Python randomises string hashing per process
    # (PYTHONHASHSEED), so hash() gave a different seed every run and fvg
    # still drifted 0.707 -> 0.756 across two invocations. crc32 is stable.
    import zlib
    rng = np.random.default_rng(zlib.crc32(sleeve.encode()))
    p = PARAMS[sleeve]
    real_all, null_all = [], [[] for _ in range(sims)]
    per_coin, skipped = {}, 0
    for coin in coins:
        df = load_bars(coin, p['tf'])
        if df is None:
            skipped += 1
            continue
        ebs, err = real_entries(coin, sleeve, df)
        if ebs is None or not ebs:
            skipped += 1
            continue
        o = df["open"].values; h = df["high"].values
        l = df["low"].values;  c = df["close"].values
        atr = _atr(h, l, c)
        lo, hi = 30, len(o) - p['hold'] - 2
        if hi <= lo:
            skipped += 1
            continue
        rr = [simulate(o, h, l, c, atr, eb, p['stop_atr'], p['hold'])
              for eb in ebs]
        rr = [x for x in rr if x is not None]
        if not rr:
            skipped += 1
            continue
        real_all += rr
        per_coin[coin] = rr
        # the control: SAME coin, SAME trade count, SAME hold/stop, random bar
        for s in range(sims):
            picks = rng.integers(lo, hi, size=len(rr))
            vv = [simulate(o, h, l, c, atr, int(e), p['stop_atr'], p['hold'])
                  for e in picks]
            null_all[s] += [x for x in vv if x is not None]
    if not real_all:
        return None
    real = float(np.mean(real_all))
    nulls = np.array([np.mean(x) for x in null_all if x])
    p_emp = (1.0 + int((nulls >= real).sum())) / (1.0 + len(nulls))
    return dict(sleeve=sleeve, n=len(real_all), coins=len(per_coin),
                skipped=skipped, real=real, nulls=nulls, p=p_emp,
                per_coin=per_coin)


def main() -> int:
    if PERP is None:
        log("  cannot find taker_data; tried: " + ", ".join(_TRIED))
        return 2
    sims = 500
    if "--sims" in sys.argv:
        sims = int(sys.argv[sys.argv.index("--sims") + 1])
    want = list(PARAMS) if "--all" in sys.argv else list(LIVE)
    coins = sorted(os.path.basename(f).replace("_1h.csv", "")
                   for f in glob.glob(f"{PERP}/*_1h.csv"))
    log(f"  {len(coins)} coins, {sims} random-timing draws per sleeve")
    log(f"  auditing: {', '.join(want)}\n")

    floor = 1.0 / (1.0 + sims)
    log(f"  smallest p this many draws can express: {floor:.4f}\n")
    out = []
    for s in want:
        log(f"  {s}: running...")
        r = audit_sleeve(s, coins, sims)
        if r is None:
            log(f"  {s}: no trades produced -- SKIPPED (not 'passed')\n")
            continue
        out.append(r)
        nm, ns = float(r["nulls"].mean()), float(r["nulls"].std())
        z = (r["real"] - nm) / ns if ns > 0 else 0.0
        log(f"    {r['n']:,} trades on {r['coins']} coins "
            f"({r['skipped']} coins skipped)")
        log(f"    real mean return per trade   {r['real']*100:>+7.3f}%")
        log(f"    random-timing null           {nm*100:>+7.3f}% "
            f"+- {ns*100:.3f}%   (z {z:+.2f})")
        log(f"    p(random >= real)            {r['p']:.3f}")
        if abs(r["p"] - 0.05) < 2 * floor:
            log(f"    BORDERLINE: p={r['p']:.3f} is within the resolution of")
            log(f"               {sims} draws. Re-run with --sims 2000 before")
            log(f"               treating this as either answer.")
        if r["p"] > 0.05:
            log(f"    VERDICT: does NOT beat random timing. This sleeve is")
            log(f"             exposure, not a signal, and its fees are a")
            log(f"             pure loss.")
        else:
            log(f"    VERDICT: beats random timing. The entry rule adds value.")
        log("")

    if out:
        log("  " + "=" * 62)
        log(f"  {'sleeve':<10}{'trades':>8}{'real':>10}{'null':>10}"
            f"{'p':>8}{'verdict':>16}")
        for r in out:
            v = "exposure" if r["p"] > 0.05 else "SIGNAL"
            log(f"  {r['sleeve']:<10}{r['n']:>8,}{r['real']*100:>9.3f}%"
                f"{r['nulls'].mean()*100:>9.3f}%{r['p']:>8.3f}{v:>16}")
        log("")
        log("  The null holds trade count, hold, stop and the exit walk")
        log("  IDENTICAL. The only difference is the entry bar -- which is")
        log("  the sleeve's entire claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
dominance.py -- does BTC dominance rising mean alts are weak, and does
the book care?

THE HYPOTHESIS, stated by the account owner BEFORE any result was seen:
"BTC dominance definitely dictates alt strength -- if it rises alts are
weak."

That is a better test than the five regime variables I scanned, and the
reason is methodological rather than a matter of taste. I searched a
list and reported whichever looked best, which is how a false positive
is manufactured. This hypothesis specifies the VARIABLE, the DIRECTION
and the MECHANISM in advance, so there is one prediction to confirm or
refute and nothing to select after the fact.

THE PREDICTION, written out so it cannot be reinterpreted later:
  dominance RISING  ->  alts WEAK  ->  a SHORT alt book does BETTER.
So the short book's return should be HIGHER in the top quintile of
rising dominance than in the bottom. A result in the opposite direction
refutes it; a result near zero fails to support it. Neither gets
rewritten into a success.

MEASURING DOMINANCE WITHOUT MARKET CAPS. True dominance is BTC market
cap over total market cap. Over horizons of days, changes in it are
dominated by relative PRICE moves rather than supply, so

    dominance change  ~=  BTC return  -  alt return

is a close proxy and needs no cap data. Where BTC price is present that
is what gets used. Where it is not -- this snapshot has no BTC file --
the fallback is the same rotation measured inside the universe:

    MAJORS (ETH, BNB, SOL, XRP, ADA, TRX, BCH, LTC, DOT, LINK, AVAX)
    minus
    SMALLS (the memes and the long tail)

Capital rotating out of the tail and into the large caps is the same
phenomenon dominance is measuring. It is a PROXY and the output says so
every time, because quietly substituting a different variable and
reporting it under the original name is how a hypothesis gets credited
for something it did not predict.

LEVEL vs CHANGE. The claim is about dominance RISING, not about it being
high, so the variable is the trailing change. Two horizons, 5 and 20
days, fixed in advance -- that is two tests, not a sweep.

RESULT. The owner's prediction was about the SHORT book and it is not
supported: 0.055 / 0.099 across halves at 5 days, i.e. nothing. But the
mechanism is real and lands on the other book. The four cross-sectional
sleeves lose 0.395 and 0.414 of a standard deviation in the top
quintile of rising dominance -- across INDEPENDENT halves, the tightest
agreement anything in this project has produced.

That makes sense mechanically: relvol, oirank, pattern and sr all
harvest DISPERSION among alts. When capital concentrates in the large
caps the tail stops dispersing in a tradeable way and bleeds together.

So he was right that dominance dictates alt behaviour, and wrong about
which book it hits. Both halves of that belong in the record.

AND IT SURVIVES AS A FILTER, threshold fitted on the first half and
applied to the second:

    skip   sleeves full   1st    2nd      combined 60/40    2nd
      0%            198   136    218                 122    119
     10%            122   111    121                 104     95
     20%            116    98    115                  94     85
     30%            119    95    125                  96     91

The second-half column is out of sample. 218 -> 115 on the sleeves and
119 -> 85 combined. The shape is a plateau peaking at 20% rather than a
spike, which is what a real effect looks like rather than a fitted one.

The filter applies to the SLEEVES ONLY. The short book shows no
dominance effect, so filtering it would cost trades and buy nothing.

USAGE
    python3 -u dominance.py                 # uses BTC if present
    python3 -u dominance.py --filter        # does skipping those days help?
    python3 -u dominance.py --horizon 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import book

MAJORS = ["ETH", "BNB", "SOL", "XRP", "ADA", "TRX", "BCH", "LTC",
          "DOT", "LINK", "AVAX"]


def log(m=""):
    print(m, flush=True)


def dominance_series(R, index):
    """(series, label, is_real_btc). Daily change in dominance."""
    import v29_test as V
    p = f"{V.PERP}/BTC_1h.csv"
    if os.path.exists(p):
        df = V._load(p)
        if df is not None:
            btc = df["close"].resample("1D").last().dropna().pct_change()
            btc = btc.reindex(index)
            alt = R.mean(axis=1)
            return (btc - alt), "BTC return - alt return (REAL)", True
    maj = [c for c in MAJORS if c in R.columns]
    sml = [c for c in R.columns if c not in maj]
    if len(maj) < 4 or len(sml) < 4:
        return None, "insufficient coins to split", False
    return (R[maj].mean(axis=1) - R[sml].mean(axis=1)),\
           f"majors({len(maj)}) - smalls({len(sml)}) PROXY, no BTC file", False


def quintile_effect(x, sig, lo, hi):
    """Mean of x in the top quintile of sig minus the bottom, over a
    slice. Returns None when the slice is too thin to quintile."""
    m = pd.Series(False, index=x.index)
    m.iloc[lo:hi] = True
    ok = m & sig.notna() & x.notna()
    if ok.sum() < 200:
        return None
    q = pd.qcut(sig[ok], 5, labels=False, duplicates="drop")
    if q.max() < 4:
        return None
    return float(x[ok][q == q.max()].mean() - x[ok][q == 0].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", action="store_true",
                    help="test standing the sleeves aside on those days")
    ap.add_argument("--horizon", type=int, default=0,
                    help="0 = run both pre-registered horizons (5 and 20)")
    a = ap.parse_args()

    import event_daily as ED
    import short_risk as SR
    import survivors as SV

    idx, PX, R, VOL, DN = book.panel()
    dom, label, real = dominance_series(R, idx)
    if dom is None:
        log(f"  cannot build a dominance series: {label}")
        return 2
    log(f"  dominance proxy: {label}")
    if not real:
        log("  NOTE: this is NOT BTC dominance. It is the same rotation")
        log("  measured inside the universe. On a box with BTC_1h.csv this")
        log("  script uses the real thing instead -- run it there too.")

    # the books
    sl, R2, VOL2, DN2 = book.build()
    sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
    oi = SV.build_oirank(R2, R2.index)
    if oi is not None:
        sl["oirank"] = oi
    for s_ in ("pattern", "sr"):
        v, _ = ED.sleeve_series(s_, quiet=True)
        if v is not None:
            sl[s_] = v
    ver = [k for k in ("relvol", "oirank", "pattern", "sr") if k in sl]
    j = None
    for k in ver:
        j = sl[k].index if j is None else j.intersection(sl[k].index)
    sh, _, _ = SR.build(SR.universe(), 12, (-1,), 10)
    j = j.intersection(sh.index).intersection(dom.index)

    def nz(s):
        s = s.reindex(j).fillna(0.0)
        return s / s.std() if s.std() > 0 else s
    sleeves = pd.Series(np.mean([nz(sl[k]).values for k in ver], axis=0),
                        index=j)
    books = {"short book": nz(sh), "4 verified sleeves": nz(sleeves)}
    books["combined 60/40"] = 0.6 * books["4 verified sleeves"] \
        + 0.4 * books["short book"]

    if a.filter:
        sig = dom.rolling(a.horizon or 5).sum().shift(1).reindex(j)
        hh = len(j) // 2
        A_, B_ = books["4 verified sleeves"], books["short book"]
        log("")
        log("  Threshold fitted on the FIRST half, applied to the second,")
        log("  so the '2nd' columns are out of sample.")
        log("")
        log(f"  {'skip':>6}{'sleeves':>9}{'1st':>7}{'2nd':>7}"
            f"{'combined':>11}{'2nd':>7}")
        for pct in (0.0, 0.10, 0.20, 0.30):
            thr = sig.iloc[:hh].quantile(1 - pct) if pct else np.inf
            mask = (sig > thr).fillna(False)
            Af = A_.copy()
            Af[mask] = 0.0
            C = 0.6 * Af + 0.4 * B_
            d_f = SV.days_of(Af)[0]
            d_1 = SV.days_of(Af.iloc[:hh])[0]
            d_2 = SV.days_of(Af.iloc[hh:])[0]
            dC = SV.days_of(C)[0]
            dC2 = SV.days_of(C.iloc[hh:])[0]
            log(f"  {pct*100:>5.0f}%{d_f:>9.0f}{d_1:>7.0f}{d_2:>7.0f}"
                f"{dC:>11.0f}{dC2:>7.0f}")
        log("")
        log("  The short book is NOT filtered: it shows no dominance")
        log("  effect, so skipping its days costs trades and buys nothing.")
        return 0

    horizons = [a.horizon] if a.horizon else [5, 20]
    h = len(j) // 2
    log("")
    log("  PREDICTION: dominance rising -> alts weak -> SHORT book does")
    log("  BETTER, so the short book's Q5-Q1 should be POSITIVE.")
    log("")
    log(f"  {'book':<20}{'horizon':>9}{'1st half':>11}{'2nd half':>11}"
        f"{'verdict':>12}")
    for hz in horizons:
        sig = dom.rolling(hz).sum().shift(1).reindex(j)
        for nm, x in books.items():
            e1 = quintile_effect(x, sig, 0, h)
            e2 = quintile_effect(x, sig, h, len(j))
            if e1 is None or e2 is None:
                log(f"  {nm:<20}{hz:>8}d        thin slice")
                continue
            agree = np.sign(e1) == np.sign(e2)
            strong = min(abs(e1), abs(e2)) > 0.10
            v = ("SUPPORTED" if (agree and strong and
                                 (e1 > 0 or nm != "short book"))
                 else "holds" if agree and strong else "no")
            if nm == "short book" and agree and strong and e1 < 0:
                v = "REFUTED"
            log(f"  {nm:<20}{hz:>8}d{e1:>11.3f}{e2:>11.3f}{v:>12}")
    log("")
    log("  Effects are in units of each book's own daily sd. 'holds' means")
    log("  the sign agrees across independent halves and both exceed 0.10;")
    log("  it is not a p-value and it is not permission to trade it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

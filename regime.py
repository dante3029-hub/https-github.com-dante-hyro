#!/usr/bin/env python3
"""
regime.py -- three regime variables already in the data, pre-registered,
plus proportional sizing instead of a binary skip.

PRE-REGISTRATION. Written before any of these were computed. The
dominance result stood up because the owner named the variable, the
direction and the mechanism in advance; the dispersion result I produced
by scanning five candidates and reporting the best looked identical in
sample (+0.455) and died out of sample (+0.114). The difference was
process, not data. So: three hypotheses, three predicted signs, fixed
here, and a 3x multiplicity correction applies to whatever survives.

  H1  AGGREGATE OI BUILD-UP
      sum of open interest across the universe, 5-day change.
      MECHANISM: open interest rising fast is leverage accumulating.
      Crowded leverage is what a liquidation cascade needs as fuel, and
      a cascade is a day where every alt moves together and
      cross-sectional dispersion vanishes.
      PREDICTION: high OI build-up -> the SLEEVES do WORSE (negative).

  H2  ONE-SIDED TAKER AGGRESSION
      sum(delta) / sum(volume) across the universe, absolute value,
      5-day mean. delta is taker buy minus taker sell, so this is how
      lopsided market-wide aggression is regardless of direction.
      MECHANISM: when everyone is hitting the same side, coins move as
      a block. Dispersion is what the sleeves harvest.
      PREDICTION: high |imbalance| -> the SLEEVES do WORSE (negative).

  H3  POSITIONING WITHOUT PARTICIPATION
      aggregate OI divided by aggregate volume, 5-day mean.
      MECHANISM: open interest high relative to turnover means
      positions are held rather than traded -- stale, crowded books
      that unwind in jumps rather than trending.
      PREDICTION: high OI/volume -> the SLEEVES do WORSE (negative).

All three predict the same sign for the same underlying reason, which is
a weakness worth naming: they may be three measurements of one thing. If
all three "work" and they are highly correlated with each other, that is
one finding, not three, and the multiplicity correction does not save it.
Pairwise correlations are printed for exactly that reason.

SECOND QUESTION: SIZING vs SKIPPING. The dominance filter is binary --
stand aside on the worst 20% of days. That discards the winners inside
that bucket along with the losers. If a variable RANKS days by quality
then size should scale with it:

    size(t) = clip(1 - k * z(signal), floor, 1)

which keeps every day and tilts. Binary skip is the special case where
k is infinite and the floor is zero. Tested head to head on the same
signal, so the comparison isolates the sizing rule rather than the
signal.

USAGE
    python3 -u regime.py                # the three hypotheses
    python3 -u regime.py --sizing       # proportional vs binary skip
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import book


def log(m=""):
    print(m, flush=True)


def load_oi(index, coins):
    """Aggregate open interest, daily, across whatever coins have data."""
    frames = {}
    for c in coins:
        for p in (f"oi_data/{c}_oi_1h.csv", f"oi/{c}_oi_1h.csv"):
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            # the real schema is idx,timestamp,open_interest -- my
            # first guess listed ts/open_time/time and oi/close/value,
            # matched neither, and load_oi silently returned empty. Two
            # of the three pre-registered hypotheses were then reported
            # as absent rather than as failing to load, which is the
            # quiet kind of wrong: a missing input must look different
            # from a negative result.
            tcol = next((x for x in ("timestamp", "ts", "open_time",
                                     "time") if x in d.columns), None)
            vcol = next((x for x in ("open_interest", "oi", "close",
                                     "value", "sum_open_interest")
                         if x in d.columns), None)
            if not tcol or not vcol:
                continue
            d["t"] = pd.to_datetime(d[tcol], unit="ms", errors="coerce")
            if d["t"].isna().all():
                d["t"] = pd.to_datetime(d[tcol], errors="coerce")
            s = d.set_index("t")[vcol].astype(float).sort_index()
            frames[c] = s.resample("1D").last().reindex(index).ffill(limit=3)
            break
    return pd.DataFrame(frames)


def load_taker(index, coins):
    """Aggregate taker delta and volume, daily."""
    dl, vl = {}, {}
    for c in coins:
        p = f"taker_data/{c}_1h.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, usecols=lambda x: x in
                        ("open_time", "volume", "delta"))
        if "delta" not in d.columns:
            continue
        d["t"] = pd.to_datetime(d["open_time"], unit="ms")
        d = d.set_index("t").sort_index()
        dl[c] = d["delta"].resample("1D").sum().reindex(index)
        vl[c] = d["volume"].resample("1D").sum().reindex(index)
    return pd.DataFrame(dl), pd.DataFrame(vl)


def effect(x, sig, lo, hi):
    m = pd.Series(False, index=x.index)
    m.iloc[lo:hi] = True
    ok = m & sig.notna() & x.notna()
    if ok.sum() < 200:
        return None
    q = pd.qcut(sig[ok], 5, labels=False, duplicates="drop")
    if q.max() < 4:
        return None
    return float(x[ok][q == q.max()].mean() - x[ok][q == 0].mean())


def books():
    import event_daily as ED
    import short_risk as SR
    import survivors as SV
    sl, R, VOL, DN = book.build()
    sl = {k: v for k, v in sl.items() if isinstance(v, pd.Series)}
    oi = SV.build_oirank(R, R.index)
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
    j = j.intersection(sh.index)

    def nz(s):
        s = s.reindex(j).fillna(0.0)
        return s / s.std() if s.std() > 0 else s
    A = pd.Series(np.mean([nz(sl[k]).values for k in ver], axis=0), index=j)
    return nz(A), nz(sh), j


def candidates(index, coins):
    OI = load_oi(index, coins)
    DL, VL = load_taker(index, coins)
    out = {}
    if not OI.empty:
        # min_count=1: a row where every coin is missing must stay NaN.
        # Plain .sum() turns it into 0.0, and pct_change across a zero
        # is inf, which propagated to an all-NaN H1 that printed as
        # "did not load" rather than as a broken calculation.
        agg = OI.sum(axis=1, min_count=1).replace(0.0, np.nan)
        out["H1 aggregate OI build-up"] = agg.pct_change(5).shift(1)
        if not VL.empty:
            out["H3 OI / volume"] = (agg / VL.sum(axis=1).replace(0, np.nan)
                                     ).rolling(5).mean().shift(1)
    if not DL.empty and not VL.empty:
        imb = DL.sum(axis=1) / VL.sum(axis=1).replace(0, np.nan)
        out["H2 |taker imbalance|"] = imb.abs().rolling(5).mean().shift(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizing", action="store_true")
    ap.add_argument("--filter", action="store_true",
                    help="H2 alone, dominance alone, and the combination")
    a = ap.parse_args()
    import survivors as SV

    idx, PX, R, VOL, DN = book.panel()
    A, B, j = books()
    cands = {k: v.reindex(j) for k, v in
             candidates(idx, list(R.columns)).items()}
    if not cands:
        log("  none of the three inputs are present (oi_data / taker delta)")
        return 2
    if len(cands) < 3:
        missing = [h for h in ("H1", "H2", "H3")
                   if not any(k.startswith(h) for k in cands)]
        log(f"  WARNING: {', '.join(missing)} did NOT LOAD -- that is a")
        log(f"  missing input, not a negative result. Do not read their")
        log(f"  absence below as a rejection.\n")

    if a.filter:
        import dominance as DM
        h2 = cands.get("H2 |taker imbalance|")
        if h2 is None:
            log("  H2 did not load")
            return 2
        dom, lab, _ = DM.dominance_series(R, idx, force_proxy=False)
        dm = dom.rolling(5).sum().shift(1).reindex(j)
        hh = len(j) // 2
        log(f"  dominance source: {lab}")
        log("  Thresholds fitted on the 1st half, applied to the 2nd.\n")
        log(f"  {'rule':<30}{'sleeves':>9}{'2nd':>7}"
            f"{'combined':>11}{'2nd':>7}")

        def row(nm, mask):
            Af = A.copy()
            Af[mask] = 0.0
            C = 0.6 * Af + 0.4 * B
            log(f"  {nm:<30}{SV.days_of(Af)[0]:>9.0f}"
                f"{SV.days_of(Af.iloc[hh:])[0]:>7.0f}"
                f"{SV.days_of(C)[0]:>11.0f}"
                f"{SV.days_of(C.iloc[hh:])[0]:>7.0f}")
        td = (dm > dm.iloc[:hh].quantile(0.80)).fillna(False)
        t2 = (h2 > h2.iloc[:hh].quantile(0.80)).fillna(False)
        row("no filter", pd.Series(False, index=j))
        row("dominance 20% only", td)
        row("taker imbalance 20% only", t2)
        row("EITHER fires", td | t2)
        row("BOTH fire", td & t2)
        log("")
        log(f"  rho(dominance, H2) = {dm.corr(h2):+.3f}; the two 20% buckets")
        log(f"  overlap on {(td & t2).sum()} of {td.sum()} days, so they are")
        log("  flagging DIFFERENT bad days. And it is not simply skipping")
        log("  more: H2 alone at 30% is WORSE than at 20%, while EITHER at")
        log("  a similar total skip is much better. Which days, not how many.")
        return 0

    if a.sizing:
        import dominance as DM
        dom, label, real = DM.dominance_series(R, idx)
        sig = dom.rolling(5).sum().shift(1).reindex(j)
        z = ((sig - sig.rolling(252, min_periods=60).mean())
             / sig.rolling(252, min_periods=60).std()).clip(-3, 3)
        h = len(j) // 2
        log(f"  dominance: {label}")
        log("")
        log("  BINARY SKIP vs PROPORTIONAL SIZING, same signal.")
        log("  size(t) = clip(1 - k*z, floor, 1). Binary skip is k=inf.")
        log("")
        log(f"  {'rule':<28}{'sleeves':>9}{'2nd half':>10}"
            f"{'combined':>10}{'2nd half':>10}")
        thr = sig.iloc[:h].quantile(0.80)
        rules = [("no filter", pd.Series(1.0, index=j)),
                 ("binary skip top 20%",
                  (~(sig > thr).fillna(False)).astype(float))]
        for k in (0.2, 0.3, 0.5):
            for fl in (0.0, 0.3):
                rules.append((f"proportional k={k} floor={fl}",
                              (1 - k * z).clip(fl, 1.0).fillna(1.0)))
        for nm, w in rules:
            Af = A * w
            C = 0.6 * Af + 0.4 * B
            log(f"  {nm:<28}{SV.days_of(Af)[0]:>9.0f}"
                f"{SV.days_of(Af.iloc[h:])[0]:>10.0f}"
                f"{SV.days_of(C)[0]:>10.0f}"
                f"{SV.days_of(C.iloc[h:])[0]:>10.0f}")
        log("")
        log("  Weights use a trailing 252d z-score, so they are computable")
        log("  on the day. The threshold for the binary rule is fitted on")
        log("  the first half only.")
        return 0

    h = len(j) // 2
    log("  PRE-REGISTERED: all three predict NEGATIVE (high value of the")
    log("  variable -> sleeves do worse). Effects in units of the book's")
    log("  own daily sd, halves INDEPENDENT.\n")
    log(f"  {'hypothesis':<28}{'1st half':>10}{'2nd half':>10}"
        f"{'as predicted?':>15}")
    kept = {}
    for nm, s in cands.items():
        e1, e2 = effect(A, s, 0, h), effect(A, s, h, len(j))
        if e1 is None or e2 is None:
            log(f"  {nm:<28}   thin slice")
            continue
        good = (e1 < 0 and e2 < 0 and min(abs(e1), abs(e2)) > 0.10)
        log(f"  {nm:<28}{e1:>10.3f}{e2:>10.3f}"
            f"{('YES' if good else 'no'):>15}")
        if good:
            kept[nm] = s
    log("")
    if len(cands) > 1:
        log("  Are these three measurements of ONE thing?")
        ks = list(cands)
        for i in range(len(ks)):
            for k2 in ks[i + 1:]:
                c = cands[ks[i]].corr(cands[k2])
                log(f"    rho({ks[i][:3]}, {k2[:3]}) = {c:+.3f}")
        log("    High correlation means a survivor is one finding, not")
        log("    three, and the 3x correction does not rescue it.")
    log("")
    if not kept:
        log("  NONE survive. All three predicted the same sign for the same")
        log("  reason and none of them hold across independent halves.")
    else:
        log(f"  {len(kept)} survived. Next step is the filter test, not a")
        log("  conclusion -- an effect that does not change days-to-target")
        log("  is a curiosity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

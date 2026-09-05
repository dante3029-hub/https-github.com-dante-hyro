#!/usr/bin/env python3
"""
divergence_detector.py — Multi-Oscillator Divergence Scanner [Quantum Algo],
ported after reading the source (docs/indicators.txt lines 1162-1537) in full.

SOURCE DEFAULTS, all reproduced here:
    pivL = pivR = 5          minConf = 2         strongTh = 4
    RSI 14 · MACD 12/26/9 (histogram) · Stochastic 14 SMOOTHED BY 3
    CCI 20 on hlc3 · OBV · MFI 14 on hlc3 · Momentum 10
    normLen = 100 (composite only)

DETECTION, exactly as the source scans it:
    A pivot low at bar `pb = bar_index - pivR` is compared against the PREVIOUS
    pivot low (running state, not an array lookup). Oscillator values are read
    at [pivR] bars back -- i.e. AT the pivot bar, not at the confirmation bar.
      regular bullish : plNow < pLowPrev  AND  osc > oscLowPrev   (price LL, osc HL)
      hidden  bullish : plNow > pLowPrev  AND  osc < oscLowPrev   (price HL, osc LL)
      regular bearish : phNow > pHighPrev AND  osc < oscHighPrev  (price HH, osc LH)
      hidden  bearish : phNow < pHighPrev AND  osc > oscHighPrev  (price LH, osc HH)
    Fires when `cnt >= minConf`; upgraded to "Strong" at `cnt >= strongTh`.
    prev-state updates on EVERY confirmed pivot, whether or not a signal fired.

ERROR IN MY EARLIER BUILD: stochastic used raw %K. The source uses
    ta.sma(ta.stoch(...), 3)  -- i.e. %D. Fixed.

TESTING CAVEAT, stated plainly: every divergence configuration LOST money on
24 coins of daily data (best 0.22 Sharpe on 130 trades, and its own control
scored higher on some settings). OBV-alone looked promising at 0.70 until an
out-of-sample check on 4h and 12h returned -0.71 and -0.63. Treat these as
things to LOOK AT, not signals to size positions from.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PIV_L = PIV_R = 5
MIN_CONF = 2
STRONG_TH = 4
RSI_LEN, MACD_F, MACD_S, MACD_SIG = 14, 12, 26, 9
STOCH_LEN, STOCH_SMOOTH = 14, 3
CCI_LEN, MFI_LEN, MOM_LEN = 20, 14, 10

NAMES = ["RSI", "MACD", "Stochastic", "CCI", "OBV", "MFI", "Momentum"]


def _rma(x: pd.Series, n: int) -> pd.Series:
    """Pine's RMA (Wilder smoothing)."""
    return x.ewm(alpha=1/n, adjust=False).mean()


def oscillators(d: pd.DataFrame) -> dict:
    c, h, l = d["close"], d["high"], d["low"]
    v = d["volume"] if "volume" in d.columns else pd.Series(1.0, index=d.index)
    hlc3 = (h + l + c) / 3
    out = {}

    # RSI(14) -- Wilder
    delta = c.diff()
    up = _rma(delta.clip(lower=0), RSI_LEN)
    dn = _rma((-delta.clip(upper=0)), RSI_LEN)
    out["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # MACD histogram (12,26,9)
    macd = c.ewm(span=MACD_F, adjust=False).mean() - c.ewm(span=MACD_S, adjust=False).mean()
    out["MACD"] = macd - macd.ewm(span=MACD_SIG, adjust=False).mean()

    # Stochastic: ta.sma(ta.stoch(close, high, low, 14), 3)  <-- SMOOTHED
    ll = l.rolling(STOCH_LEN).min()
    hh = h.rolling(STOCH_LEN).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["Stochastic"] = k.rolling(STOCH_SMOOTH).mean()

    # CCI(20) on hlc3
    sma = hlc3.rolling(CCI_LEN).mean()
    md = (hlc3 - sma).abs().rolling(CCI_LEN).mean()
    out["CCI"] = (hlc3 - sma) / (0.015 * md.replace(0, np.nan))

    # OBV
    out["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()

    # MFI(14) on hlc3
    mf = hlc3 * v
    pos = mf.where(hlc3 > hlc3.shift(), 0.0).rolling(MFI_LEN).sum()
    neg = mf.where(hlc3 < hlc3.shift(), 0.0).rolling(MFI_LEN).sum()
    out["MFI"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    # Momentum(10)
    out["Momentum"] = c - c.shift(MOM_LEN)
    return out


def pivots(h: np.ndarray, l: np.ndarray, left: int, right: int):
    """ta.pivothigh / ta.pivotlow. A pivot at bar i is CONFIRMED at i+right,
    which is why divergences print `right` bars after the swing."""
    N = len(h)
    ph, pl = [], []
    for i in range(left, N - right):
        # Pine's ta.pivothigh is ASYMMETRIC: strictly lower on the LEFT,
        # ties permitted on the RIGHT. A symmetric strict test (what I had
        # first) misses pivots wherever an equal high/low sits to the right,
        # which shifted every downstream comparison.
        hi_ok = all(h[j] < h[i] for j in range(i-left, i)) and \
                all(h[j] <= h[i] for j in range(i+1, i+right+1))
        lo_ok = all(l[j] > l[i] for j in range(i-left, i)) and \
                all(l[j] >= l[i] for j in range(i+1, i+right+1))
        if hi_ok:
            ph.append(i)
        if lo_ok:
            pl.append(i)
    return ph, pl


def detect(d: pd.DataFrame, enabled=None, min_conf=MIN_CONF,
           strong_th=STRONG_TH, piv_l=PIV_L, piv_r=PIV_R,
           show_regular=True, show_hidden=True):
    """Returns list of dicts, one per fired divergence:
        {bar, time, dir(+1/-1), kind('regular'|'hidden'), count, strong,
         price, oscillators[list of names]}
    """
    if enabled is None:
        enabled = set(NAMES)
    osc = oscillators(d)
    h, l = d["high"].values, d["low"].values
    N = len(d)
    ph_idx, pl_idx = pivots(h, l, piv_l, piv_r)
    out = []

    # ── running previous-pivot state, exactly as the source keeps it ──
    p_low_prev = None; osc_low_prev = {}
    p_high_prev = None; osc_high_prev = {}

    events = ([(i, "low") for i in pl_idx] + [(i, "high") for i in ph_idx])
    events.sort(key=lambda x: x[0] + piv_r)      # order by CONFIRMATION bar

    for pb, kind in events:
        conf_bar = pb + piv_r
        if conf_bar >= N:
            continue
        cv = {n: osc[n].iloc[pb] for n in NAMES}

        if kind == "low":
            if p_low_prev is not None:
                cnt_r, cnt_h, names_r, names_h = 0, 0, [], []
                for n in NAMES:
                    if n not in enabled:
                        continue
                    prev = osc_low_prev.get(n)
                    now = cv[n]
                    if prev is None or not (np.isfinite(prev) and np.isfinite(now)):
                        continue
                    if l[pb] < p_low_prev and now > prev:
                        cnt_r += 1; names_r.append(n)
                    if l[pb] > p_low_prev and now < prev:
                        cnt_h += 1; names_h.append(n)
                if show_regular and cnt_r >= min_conf:
                    out.append(dict(bar=pb, time=d.index[pb], dir=1, kind="regular",
                                    count=cnt_r, strong=cnt_r >= strong_th,
                                    price=l[pb], oscillators=names_r,
                                    confirmed=d.index[conf_bar]))
                if show_hidden and cnt_h >= min_conf:
                    out.append(dict(bar=pb, time=d.index[pb], dir=1, kind="hidden",
                                    count=cnt_h, strong=False,
                                    price=l[pb], oscillators=names_h,
                                    confirmed=d.index[conf_bar]))
            p_low_prev = l[pb]
            osc_low_prev = dict(cv)
        else:
            if p_high_prev is not None:
                cnt_r, cnt_h, names_r, names_h = 0, 0, [], []
                for n in NAMES:
                    if n not in enabled:
                        continue
                    prev = osc_high_prev.get(n)
                    now = cv[n]
                    if prev is None or not (np.isfinite(prev) and np.isfinite(now)):
                        continue
                    if h[pb] > p_high_prev and now < prev:
                        cnt_r += 1; names_r.append(n)
                    if h[pb] < p_high_prev and now > prev:
                        cnt_h += 1; names_h.append(n)
                if show_regular and cnt_r >= min_conf:
                    out.append(dict(bar=pb, time=d.index[pb], dir=-1, kind="regular",
                                    count=cnt_r, strong=cnt_r >= strong_th,
                                    price=h[pb], oscillators=names_r,
                                    confirmed=d.index[conf_bar]))
                if show_hidden and cnt_h >= min_conf:
                    out.append(dict(bar=pb, time=d.index[pb], dir=-1, kind="hidden",
                                    count=cnt_h, strong=False,
                                    price=h[pb], oscillators=names_h,
                                    confirmed=d.index[conf_bar]))
            p_high_prev = h[pb]
            osc_high_prev = dict(cv)

    out.sort(key=lambda x: x["bar"])
    return out


def describe(sig: dict) -> str:
    side = "Bullish" if sig["dir"] > 0 else "Bearish"
    kind = "Hidden " if sig["kind"] == "hidden" else ""
    strong = "Strong " if sig["strong"] else ""
    marks = ("◆" if sig["kind"] == "regular" else "◇") * sig["count"]
    return (f"{'▲' if sig['dir'] > 0 else '▼'} {strong}{kind}{side} Divergence "
            f"{marks}  ({sig['count']}x: {' · '.join(sig['oscillators'])})")

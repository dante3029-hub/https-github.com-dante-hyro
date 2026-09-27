"""
Price-event sleeves — sr, srflip, pattern, fvg behind ONE interface.

All four are event-driven: they answer "is there a fresh trigger on the latest
CLOSED bar", and slot management is the tracker's job (bot/event_sleeves.py),
exactly as sleeve_short and sleeve_bos already work.

THE INTERFACE
    latest_signal(bars, oi_expanding=..., state=...) -> Signal | None

    Signal carries everything the tracker needs to open and later close a slot:
    side, the reference price, the stop, and how many bars to hold.

WHY ONE MODULE
Four separate files would duplicate the ATR, the entry convention and the exit
loop four times. Every one of those was a source of a real bug during
research -- the stop was placed off the entry bar's CLOSE while entering at its
OPEN (worth -0.21 on fvg), and the exit loop skipped the entry bar entirely,
which made tight stops look monotonically better (fvg at 0.25 ATR "scored"
2.84; it is really 0.06). Fixing those in one place fixes them everywhere.

ENTRY CONVENTION, applied uniformly
    signal detected on bar i (the latest CLOSED bar)
    entry at the OPEN of bar i+1  -- never bar i, never the historical break
    stop  = entry -/+ stop_atr * ATR(14) measured AT BAR i
    exit  on stop (checked FROM the entry bar) or after hold_bars

That "never the historical break bar" matters: the pattern detector reports a
break bar up to 20 bars in the past. Entering there is lookahead and once
inflated a backtest from 1.04 to 6.06.

PER-SLEEVE PROVENANCE (hourly simulator, 24-coin core)
    sr       6h   3.0 ATR  hold 15  +OI   long only   1.95
    srflip   6h   2.0 ATR  hold 15        long only   1.69
    pattern  6h   2.0 ATR  hold 15  +OI   long only   1.87
    fvg     12h   1.5 ATR  hold 10  +OI   both ways   2.96
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ── per-sleeve parameters, single source of truth ───────────────────────────
PARAMS = {
    'sr':      dict(tf=6,  stop_atr=3.0, hold=15, needs_oi=True,  long_only=True),
    'srflip':  dict(tf=6,  stop_atr=2.0, hold=15, needs_oi=False, long_only=True),
    'pattern': dict(tf=6,  stop_atr=2.0, hold=15, needs_oi=True,  long_only=True),
    'fvg':     dict(tf=12, stop_atr=1.5, hold=10, needs_oi=True,  long_only=False),
}
MAX_CONCURRENT = 6


@dataclass
class Signal:
    """A fresh trigger on the latest closed bar.

    `needs_oi` tells the tracker whether to gate this on open interest when it
    places the order. See latest_signal's docstring for why that gate belongs
    at entry rather than at detection.
    """
    sleeve: str
    side: int                 # +1 long, -1 short
    entry_ref: float          # last close; the real entry is the next open
    stop_atr: float
    atr: float
    hold_bars: int
    needs_oi: bool = False

    def stop_from(self, entry_price: float) -> float:
        """Stop placed off the ACTUAL entry price, not the signal bar's close.
        Placing it off the close while entering at the open is lookahead."""
        return entry_price - self.side * self.stop_atr * self.atr

    def to_dict(self) -> dict:
        return asdict(self)


def atr14(bars: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (bars['high'].to_numpy(float), bars['low'].to_numpy(float),
               bars['close'].to_numpy(float))
    m = len(c)
    tr = np.zeros(m)
    tr[0] = h[0] - l[0]
    for i in range(1, m):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    out = np.full(m, np.nan)
    a = np.nan
    for i in range(m):
        a = tr[i] if not np.isfinite(a) else (a*(n-1) + tr[i]) / n
        if i >= n:
            out[i] = a
    return out


# ── the four detectors, each returning "did it fire on the last bar" ────────

def _fired_sr(bars: pd.DataFrame) -> int:
    """Resistance BREAK. Levels from volume-confirmed pivots; the break is the
    LOW crossing above the outer box edge, per the source (`ta.crossover(low,
    resistanceLevel_1)`). Using the close instead broke parity and turned the
    second half negative."""
    import sr2
    L, S = sr2.signals(bars, 'break_res')
    i = len(bars) - 1
    return 1 if (i < len(L) and L[i]) else 0


def _fired_srflip(bars: pd.DataFrame, window: int = 20) -> int:
    """Resistance turned SUPPORT. After the low clears the outer edge, wait for
    price to return to the inner level and hold above it."""
    import sr2
    sup, sup1, res, res1 = sr2.levels(bars)
    l = bars['low'].to_numpy(float)
    c = bars['close'].to_numpy(float)
    n = len(bars)
    broke, lvl = None, np.nan
    for i in range(1, n):
        if np.isfinite(res1[i]) and l[i] > res1[i] and l[i-1] <= res1[i-1]:
            broke, lvl = i, res[i]
        if (broke is not None and i > broke and i - broke <= window
                and np.isfinite(lvl) and l[i] <= lvl and c[i] > lvl):
            if i == n - 1:
                return 1
            broke = None
    return 0


def _fired_pattern(bars: pd.DataFrame) -> int:
    """MarkitTick, bullish patterns only. Keyed on the DETECTION bar, never the
    historical break bar the detector also reports."""
    from markittick_detector import detect
    last = len(bars) - 1
    for rec in detect(bars):
        if not rec[2]:                       # bullish only
            continue
        det_bar = rec[7] if len(rec) > 7 else rec[0]
        if det_bar == last:
            return 1
    return 0


def _fired_fvg(bars: pd.DataFrame) -> int:
    """Fair value gap on the last three bars. Returns +1 bull, -1 bear, 0 none.
    Verified an exact match to the LuxAlgo Pine (575 of 575 gaps identical)."""
    h = bars['high'].to_numpy(float)
    l = bars['low'].to_numpy(float)
    c = bars['close'].to_numpy(float)
    i = len(bars) - 1
    if i < 2:
        return 0
    if h[i-2] > 0 and l[i] > h[i-2] and c[i-1] > h[i-2]:
        return 1
    if h[i] > 0 and h[i] < l[i-2] and c[i-1] < l[i-2]:
        return -1
    return 0


_DETECTORS = {'sr': _fired_sr, 'srflip': _fired_srflip,
              'pattern': _fired_pattern, 'fvg': _fired_fvg}


def latest_signal(sleeve: str, bars: pd.DataFrame,
                  oi_expanding: Optional[bool] = None) -> Optional[Signal]:
    """Is there a fresh trigger on the latest CLOSED bar of `bars`?

    sleeve: one of PARAMS.
    bars:   that sleeve's native timeframe, newest last, last bar CLOSED.

    oi_expanding: OI state AT ORDER TIME, not at signal time. Pass None to
                  detect only and let the caller gate on OI when it places
                  the order.

    ** THE OI GATE IS APPLIED AT THE ENTRY BAR, NOT THE SIGNAL BAR. **

    That is not a detail. On SOL 6h there are 60 raw breaks; gating at the
    SIGNAL bar keeps 31, gating at the ENTRY bar keeps 23, and the two sets
    overlap by only 12 -- 29%. The backtest gates at entry, so gating at
    signal live would trade a materially different strategy.

    Both are causal: OI is a live snapshot, so at the moment the order goes in
    (the open of the next bar) current OI is observable. The tracker should
    call this at the close of bar i, then query OI when it actually places the
    order and drop the signal if OI is not expanding.

    Returns a Signal, or None if nothing fired.
    """
    if sleeve not in PARAMS:
        raise ValueError(f"unknown sleeve {sleeve!r}")
    p = PARAMS[sleeve]
    if bars is None or len(bars) < 700:
        return None                       # detectors need history; do not guess
    side = _DETECTORS[sleeve](bars)
    if side == 0:
        return None
    if p['long_only'] and side < 0:
        return None
    # oi_expanding=None means "detect only, the caller gates at order time"
    if p['needs_oi'] and oi_expanding is False:
        return None
    a = atr14(bars)[len(bars) - 1]
    if not np.isfinite(a) or a <= 0:
        return None
    return Signal(sleeve=sleeve, side=int(side),
                  entry_ref=float(bars['close'].iloc[-1]),
                  stop_atr=float(p['stop_atr']), atr=float(a),
                  hold_bars=int(p['hold']), needs_oi=bool(p['needs_oi']))

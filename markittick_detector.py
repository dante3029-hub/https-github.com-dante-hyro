#!/usr/bin/env python3
"""
mt2.py — MarkitTick "Auto Pattern Detector Targets", ported after reading all
886 lines of the source rather than grepping fragments.

ERRORS IN MY EARLIER ATTEMPTS, ALL FIXED HERE:
  1. req_close_brk defaults to FALSE -> breaks use HIGH/LOW, not close.
     I used close, which suppressed most signals.
  2. Entry = f_get_entry_price(b) = open[bar_index-b] = the OPEN AT THE BREAK
     BAR. I used the next bar's open.
  3. Stop = f_sl_pat with use_sl_pct_tgt=TRUE -> entry -/+ |target-entry| * 0.25.
     I used an ATR buffer, which failed R:R validation constantly.
     (Consequence: risk is always 25% of reward, so R:R is always ~4:1 and
      min_rr=1.0 ALWAYS passes. Validation is effectively a no-op at defaults.)
  4. _gate = bar_index > 600 -> NOTHING fires before bar 600.
  5. Arrays use unshift, so get(0) is the NEWEST pivot.
  6. f_pivotHigh uses STRICT inequality (src[i] >= candidate invalidates), and
     is symmetric on lb_left for both sides.
  7. Detection order is double -> triple -> HS -> flag/pennant -> wedge ->
     triangle -> rectangle -> cup, FIRST MATCH WINS.
  8. sym_tol is input 10.0 / 100 = 0.10, and f_isNear multiplies by it.
     So it is a real 10% -- my "symmetry off" test was the wrong reading.
  9. Break scan runs OLDEST-FIRST over max(min_brk_bar, lb_right)+10 = 20 bars,
     and uses atr_val[i] -- the ATR AT THE SCANNED BAR.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ── defaults, exactly as the source declares them ──
COOLDOWN = 5
SYM_TOL = 10.0 / 100          # 0.10
LVL_TOL = 3.0 / 100           # 0.03
MIN_SIZE_PCT = 0.5 / 100      # 0.005
MIN_ATR_MULT = 1.0
LB_LEFT = LB_RIGHT = 10
REQ_CLOSE_BRK = False         # <-- breaks on high/low
REQ_BODY_BRK = False
USE_ATR_BRK = True
BREAK_ATR = 1.0
BREAK_PCT = 0.3 / 100
BRK_CONFIRM_BARS = 1
MIN_BRK_BAR = 1
MIN_RR = 1.0
STOP_BUFFER = 0.5
USE_SL_PCT_TGT = True
SL_PCT_TGT = 25.0 / 100       # 0.25
GATE_BAR = 600                # source gate; override per timeframe
SCAN_LIMIT = max(MIN_BRK_BAR, LB_RIGHT) + 10   # 20


def atr14(h, l, c):
    N = len(c); tr = np.zeros(N); out = np.full(N, np.nan)
    for i in range(1, N):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    # Pine ta.atr is RMA-smoothed
    a = np.nan
    for i in range(1, N):
        a = tr[i] if not np.isfinite(a) else (a*13 + tr[i]) / 14
        if i >= 14:
            out[i] = a
    return out


def is_near(v1, v2, tol):
    """f_isNear -- tol is already a fraction (sym_tol = 0.10)."""
    if not (np.isfinite(v1) and np.isfinite(v2)):
        return False
    return abs(v1 - v2) <= ((v1 + v2) / 2) * tol


def project(x1, y1, x2, y2, tx):
    if x2 == x1:
        return y1
    return y1 + ((y2 - y1) / (x2 - x1)) * (tx - x1)


def find_pivots_strict(h, l, length):
    """f_pivotHigh/f_pivotLow: candidate = src[length], invalid if ANY of the
    2*length neighbours is >= (high) or <= (low). Detected at bar i+length,
    stored with index i (= bar_index - lb_right)."""
    N = len(h)
    highs, lows = [], []
    for det in range(2*length, N):          # detection bar
        cand_i = det - length
        ok_h = ok_l = True
        for k in range(1, 2*length + 1):
            if k == length:
                continue
            j = det - k
            if j < 0:
                ok_h = ok_l = False
                break
            if h[j] >= h[cand_i]:
                ok_h = False
            if l[j] <= l[cand_i]:
                ok_l = False
        if ok_h:
            highs.append((det, cand_i, h[cand_i]))   # (known_at, index, price)
        if ok_l:
            lows.append((det, cand_i, l[cand_i]))
    return highs, lows


class MT:
    def __init__(self, d):
        self.h = d["high"].values
        self.l = d["low"].values
        self.c = d["close"].values
        self.o = d["open"].values
        self.N = len(self.c)
        self.A = atr14(self.h, self.l, self.c)
        self.ph_all, self.pl_all = find_pivots_strict(self.h, self.l, LB_LEFT)

    def pivots_at(self, bar):
        """Newest-first, as unshift() produces."""
        ph = [(i, p) for (k, i, p) in self.ph_all if k <= bar][::-1]
        pl = [(i, p) for (k, i, p) in self.pl_all if k <= bar][::-1]
        return ph, pl

    def trigger_level(self, bar, x1, y1, x2, y2):
        """Where the boundary line sits at the current bar -- the level price
        must cross for the pattern to complete."""
        return project(x1, y1, x2, y2, bar)

    def break_idx(self, bar, x1, y1, x2, y2, is_up):
        """f_get_break_idx_precise -- oldest-first over the last 20 bars."""
        m = 0.0 if x2 == x1 else (y2 - y1) / (x2 - x1)
        for i in range(SCAN_LIMIT, 0, -1):
            b = bar - i
            if b < 0:
                continue
            proj = y1 if x2 == x1 else y1 + m * (b - x1)
            a = self.A[b]
            if not np.isfinite(a):
                continue
            margin = a * BREAK_ATR if USE_ATR_BRK else proj * BREAK_PCT
            up_t, dn_t = proj + margin, proj - margin
            val = (self.c[b] if REQ_CLOSE_BRK
                   else (self.h[b] if is_up else self.l[b]))
            broken = val > up_t if is_up else val < dn_t
            if broken:
                return b
        return None

    def _brk(self, forming, bar, x1, y1, x2, y2, is_up):
        """Breakout bar, or -- in forming mode -- the current bar, so the
        pattern is reported the moment its geometry is valid."""
        # NOTE: `forming` is retained for diagnostics only. Setting it True
        # makes every break "succeed", which lets Double (first in dispatch)
        # pre-empt every other pattern -- five of six coins returned Double
        # Bottom on the same bar. The source requires a real break before
        # res.detected is set, so breakout-only is the faithful behaviour.
        return self.break_idx(bar, x1, y1, x2, y2, is_up)

    def entry_at(self, b):
        """f_get_entry_price: open AT the break bar."""
        return self.o[b] if 0 <= b < self.N else np.nan

    def stop_for(self, entry, target, is_bull):
        """f_sl_pat with use_sl_pct_tgt = True."""
        if not (np.isfinite(entry) and np.isfinite(target)):
            return np.nan
        dist = abs(target - entry)
        return entry - dist*SL_PCT_TGT if is_bull else entry + dist*SL_PCT_TGT

    def valid(self, entry, stop, target):
        if not np.isfinite(entry):
            return False
        risk = abs(entry - stop)
        if risk == 0:
            return False
        return (abs(target - entry) / risk) >= MIN_RR

    def valid_size(self, height, price, bar):
        a = self.A[bar]
        if not np.isfinite(a):
            return False
        return height > max(price * MIN_SIZE_PCT, a * MIN_ATR_MULT)

    # ── detectors, in the source's dispatch order ──
    def detect_at(self, bar, forming=False):
        """forming=False -> emit only on a confirmed breakout (what alert() does).
        forming=True  -> emit as soon as the SHAPE is valid, which is when the
        indicator DRAWS it. Earlier, more alerts, and the level to watch is
        reported instead of a filled entry."""
        ph, pl = self.pivots_at(bar)
        px = self.c[bar]

        # 1. DOUBLE
        if len(ph) >= 2 and len(pl) >= 1:
            p1, p2 = ph[0], ph[1]           # p1 newest
            tr = pl[0]
            if is_near(p1[1], p2[1], LVL_TOL) and p1[0] > tr[0] > p2[0]:
                avg_top = (p1[1] + p2[1]) / 2
                height = avg_top - tr[1]      # source: _avg_top - _mid.p
                if self.valid_size(height, px, bar):
                    b = self._brk(forming, bar, tr[0], tr[1], tr[0], tr[1], False)
                    if b is not None:
                        e = self.entry_at(b); t = tr[1] - height
                        s = self.stop_for(e, t, False)
                        if t < e and self.valid(e, s, t):
                            return b, False, "Double Top", e, s, t, p2[0]
        if len(pl) >= 2 and len(ph) >= 1:
            p1, p2 = pl[0], pl[1]
            pk = ph[0]
            if is_near(p1[1], p2[1], LVL_TOL) and p1[0] > pk[0] > p2[0]:
                avg_bot = (p1[1] + p2[1]) / 2
                height = pk[1] - avg_bot      # source: _mid_b.p - _avg_bot
                if self.valid_size(height, px, bar):
                    b = self._brk(forming, bar, pk[0], pk[1], pk[0], pk[1], True)
                    if b is not None:
                        e = self.entry_at(b); t = pk[1] + height
                        s = self.stop_for(e, t, True)
                        if t > e and self.valid(e, s, t):
                            return b, True, "Double Bottom", e, s, t, p2[0]

        # 2. TRIPLE
        if len(ph) >= 3 and len(pl) >= 2:
            a, bb, cc = ph[0], ph[1], ph[2]      # h1, h2, h3 newest-first
            l1t, l2t = pl[0], pl[1]
            # source: _h1.i > _l1t.i > _h2.i > _l2t.i > _h3.i
            ordered = a[0] > l1t[0] > bb[0] > l2t[0] > cc[0]
            if ordered and is_near(a[1], bb[1], LVL_TOL) and is_near(bb[1], cc[1], LVL_TOL):
                lo = min(l1t[1], l2t[1])         # _neck_tt = min of the two lows
                height = max(a[1], bb[1], cc[1]) - lo
                if self.valid_size(height, px, bar):
                    b = self._brk(forming, bar, l2t[0], l2t[1], l1t[0], l1t[1], False)
                    if b is not None:
                        e = self.entry_at(b); t = lo - height
                        s = self.stop_for(e, t, False)
                        if t < e and self.valid(e, s, t):
                            return b, False, "Triple Top", e, s, t, cc[0]
        if len(pl) >= 3 and len(ph) >= 2:
            a, bb, cc = pl[0], pl[1], pl[2]      # l1b, l2b, l3b newest-first
            h1b, h2b = ph[0], ph[1]
            # source: _l1b.i > _h1b.i > _l2b.i > _h2b.i > _l3b.i
            ordered = a[0] > h1b[0] > bb[0] > h2b[0] > cc[0]
            if ordered and is_near(a[1], bb[1], LVL_TOL) and is_near(bb[1], cc[1], LVL_TOL):
                hi = max(h1b[1], h2b[1])         # _neck_tb = MAX of the two highs
                height = hi - min(a[1], bb[1], cc[1])
                if self.valid_size(height, px, bar):
                    # source break line is HORIZONTAL at _neck_tb
                    b = self._brk(forming, bar, h2b[0], hi, h1b[0], hi, True)
                    if b is not None:
                        e = self.entry_at(b); t = hi + height
                        s = self.stop_for(e, t, True)
                        if t > e and self.valid(e, s, t):
                            return b, True, "Triple Bottom", e, s, t, cc[0]

        # 3. HEAD & SHOULDERS
        if len(ph) >= 3 and len(pl) >= 2:
            rs, head, ls = ph[0], ph[1], ph[2]
            nr, nl = pl[0], pl[1]
            # source: _rs.i > _l_neck_r.i > _head.i > _l_neck_l.i > _ls.i
            ordered = rs[0] > nr[0] > head[0] > nl[0] > ls[0]
            if (ordered and head[1] > rs[1] and head[1] > ls[1]
                    and is_near(ls[1], rs[1], SYM_TOL)):
                neck = (nr[1] + nl[1]) / 2
                height = head[1] - neck
                if self.valid_size(height, px, bar):
                    b = self._brk(forming, bar, nl[0], nl[1], nr[0], nr[1], False)
                    if b is not None:
                        nb = project(nl[0], nl[1], nr[0], nr[1], b)
                        e = self.entry_at(b); t = nb - height
                        s = self.stop_for(e, t, False)
                        if t < e and self.valid(e, s, t):
                            return b, False, "Head & Shoulders", e, s, t, ls[0]
        if len(pl) >= 3 and len(ph) >= 2:
            rs, head, ls = pl[0], pl[1], pl[2]
            nr, nl = ph[0], ph[1]
            # source: _rs_inv.i > _h_neck_r.i > _head_inv.i > _h_neck_l.i > _ls_inv.i
            ordered = rs[0] > nr[0] > head[0] > nl[0] > ls[0]
            if (ordered and head[1] < rs[1] and head[1] < ls[1]
                    and is_near(ls[1], rs[1], SYM_TOL)):
                neck = (nr[1] + nl[1]) / 2
                height = neck - head[1]
                if self.valid_size(height, px, bar):
                    b = self._brk(forming, bar, nl[0], nl[1], nr[0], nr[1], True)
                    if b is not None:
                        nb = project(nl[0], nl[1], nr[0], nr[1], b)
                        e = self.entry_at(b); t = nb + height
                        s = self.stop_for(e, t, True)
                        if t > e and self.valid(e, s, t):
                            return b, True, "Inv Head & Shoulders", e, s, t, ls[0]

        # 4. CUP & HANDLE  (source: detectCupHandle, dispatched LAST but the
        #    engine order is double->triple->hs->fp->wedge->tri->rect->cup;
        #    placed here only for code locality -- guarded so it cannot pre-empt
        #    the slope family, which the source checks first.)
        #    std : _l_handle.i > _h_rim.i > _l_bot.i > _h_left.i
        #          isNear(h_rim, h_left, sym_tol), l_handle > l_bot, l_handle < h_rim
        #    inv : _h_handle.i > _l_rim.i > _h_top.i > _l_left.i
        self._cup_pending = None
        if len(ph) >= 2 and len(pl) >= 2:
            h_rim, h_left = ph[0], ph[1]
            l_handle, l_bot = pl[0], pl[1]
            if (l_handle[0] > h_rim[0] > l_bot[0] > h_left[0]
                    and is_near(h_rim[1], h_left[1], SYM_TOL)
                    and l_handle[1] > l_bot[1] and l_handle[1] < h_rim[1]):
                cup_h = h_rim[1] - l_bot[1]
                if self.valid_size(cup_h, px, bar):
                    b = self._brk(forming, bar, h_rim[0], h_rim[1], bar, h_rim[1], True)
                    if b is not None:
                        na_cup = project(h_left[0], h_left[1], h_rim[0], h_rim[1], b)
                        e = self.entry_at(b); t = na_cup + cup_h
                        st = self.stop_for(e, t, True)
                        if t > e and self.valid(e, st, t):
                            self._cup_pending = (b, True, "Cup & Handle", e, st, t, h_left[0])
            l_rim, l_left = pl[0], pl[1]
            h_handle, h_top = ph[0], ph[1]
            if (self._cup_pending is None
                    and h_handle[0] > l_rim[0] > h_top[0] > l_left[0]
                    and is_near(l_rim[1], l_left[1], SYM_TOL)
                    and h_handle[1] < h_top[1] and h_handle[1] > l_rim[1]):
                cup_h = h_top[1] - l_rim[1]
                if self.valid_size(cup_h, px, bar):
                    b = self._brk(forming, bar, l_rim[0], l_rim[1], bar, l_rim[1], False)
                    if b is not None:
                        na_inv = project(l_left[0], l_left[1], l_rim[0], l_rim[1], b)
                        e = self.entry_at(b); t = na_inv - cup_h
                        st = self.stop_for(e, t, False)
                        if t < e and self.valid(e, st, t):
                            self._cup_pending = (b, False, "Inv Cup & Handle", e, st, t, l_left[0])

        # 4-7. SLOPE FAMILY
        if len(ph) >= 2 and len(pl) >= 2:
            h1, h2 = ph[0], ph[1]
            l1, l2 = pl[0], pl[1]
            su = 0.0 if h1[0] == h2[0] else (h1[1]-h2[1]) / max(1, h1[0]-h2[0])
            sl = 0.0 if l1[0] == l2[0] else (l1[1]-l2[1]) / max(1, l1[0]-l2[0])
            start = min(h2[0], l2[0])
            plen = min(200, bar - start + 1)
            lo_w = max(0, bar - plen + 1)
            pole = self.h[lo_w:bar+1].max() - self.l[lo_w:bar+1].min()
            a = self.A[bar]
            par = is_near(su, sl, 0.2)

            # FLAG / PENNANT
            if np.isfinite(a) and pole > a * 3:
                if su < 0 and sl < 0:
                    b = self._brk(forming, bar, h2[0], h2[1], h1[0], h1[1], True)
                    if b is not None:
                        ub = project(h2[0], h2[1], h1[0], h1[1], b)
                        e = self.entry_at(b); t = ub + pole
                        s = self.stop_for(e, t, True)
                        if t > e and self.valid(e, s, t):
                            nm = "Bullish Flag" if par else "Bullish Pennant"
                            return b, True, nm, e, s, t, start
                elif su > 0 and sl > 0:
                    b = self._brk(forming, bar, l2[0], l2[1], l1[0], l1[1], False)
                    if b is not None:
                        lb = project(l2[0], l2[1], l1[0], l1[1], b)
                        e = self.entry_at(b); t = lb - pole
                        s = self.stop_for(e, t, False)
                        if t < e and self.valid(e, s, t):
                            nm = "Bearish Flag" if par else "Bearish Pennant"
                            return b, False, nm, e, s, t, start

            # WEDGE -- source guards: if _has and _height_now > 0
            _pu = project(h2[0], h2[1], h1[0], h1[1], bar)
            _pl = project(l2[0], l2[1], l1[0], l1[1], bar)
            _height_now = _pu - _pl
            if _height_now > 0 and su < 0 and sl < 0 and sl < su:
                b = self._brk(forming, bar, h2[0], h2[1], h1[0], h1[1], True)
                if b is not None:
                    e = self.entry_at(b); t = h2[1]
                    s = self.stop_for(e, t, True)
                    if t > e and self.valid(e, s, t):
                        return b, True, "Falling Wedge", e, s, t, h2[0]
            elif _height_now > 0 and su > 0 and sl > 0 and sl > su:
                b = self._brk(forming, bar, l2[0], l2[1], l1[0], l1[1], False)
                if b is not None:
                    e = self.entry_at(b); t = l2[1]
                    s = self.stop_for(e, t, False)
                    if t < e and self.valid(e, s, t):
                        return b, False, "Rising Wedge", e, s, t, l2[0]

            # ── TRIANGLES (source: detectTriangles) ──
            # _flat_tol   = close * 0.0005
            # _converging = proj(upper, bar) > proj(lower, bar)   <-- I MISSED THIS
            # _base_height= |proj(upper,start) - proj(lower,start)|  measured at
            #               START, not at the current bar          <-- AND THIS
            flat_tol = px * 0.0005
            start_tri = min(h2[0], l2[0])
            base_h = abs(project(h2[0], h2[1], h1[0], h1[1], start_tri)
                         - project(l2[0], l2[1], l1[0], l1[1], start_tri))
            up_now = project(h2[0], h2[1], h1[0], h1[1], bar)
            lo_now = project(l2[0], l2[1], l1[0], l1[1], bar)
            converging = up_now > lo_now

            if converging and self.valid_size(base_h, px, bar):
                # BULL leg first, exactly as the source orders it
                if (su < 0 and sl > 0) or (abs(su) < flat_tol and sl > 0):
                    b = self._brk(forming, bar, h2[0], h2[1], h1[0], h1[1], True)
                    if b is not None:
                        ub = project(h2[0], h2[1], h1[0], h1[1], b)
                        e = self.entry_at(b); t = ub + base_h
                        st = self.stop_for(e, t, True)
                        if self.valid(e, st, t):
                            nm = ("Ascending Triangle" if abs(su) < flat_tol
                                  else "Symmetrical Triangle")
                            return b, True, nm, e, st, t, h2[0]
                # BEAR leg only if the bull leg did not fire
                if (su < 0 and sl > 0) or (su < 0 and abs(sl) < flat_tol):
                    b = self._brk(forming, bar, l2[0], l2[1], l1[0], l1[1], False)
                    if b is not None:
                        lb = project(l2[0], l2[1], l1[0], l1[1], b)
                        e = self.entry_at(b); t = lb - base_h
                        st = self.stop_for(e, t, False)
                        if self.valid(e, st, t):
                            nm = ("Descending Triangle" if abs(sl) < flat_tol
                                  else "Symmetrical Triangle")
                            return b, False, nm, e, st, t, h2[0]

            # ── RECTANGLES (source: detectRectangles) ──
            # _flat_tol_r = atr_val * 0.05   <-- ATR-based, NOT price-based
            # plus f_isNear(slope_u, slope_l, 0.5). With opposite-sign slopes the
            # RHS of isNear goes negative, so that check can never pass -- which
            # is exactly what should have rejected SOL.
            a_bar = self.A[bar]
            if np.isfinite(a_bar):
                flat_r = a_bar * 0.05
                is_flat = abs(su) < flat_r and abs(sl) < flat_r
                avg_top = (h1[1] + h2[1]) / 2
                avg_bot = (l1[1] + l2[1]) / 2
                height_r = avg_top - avg_bot
                start_r = min(h2[0], l2[0])
                if (is_flat and is_near(su, sl, 0.5)
                        and self.valid_size(height_r, px, bar)):
                    b = self._brk(forming, bar, start_r, avg_top, bar, avg_top, True)
                    if b is not None:
                        e = self.entry_at(b); t = avg_top + height_r
                        st = self.stop_for(e, t, True)
                        if self.valid(e, st, t):
                            return b, True, "Rectangle", e, st, t, h2[0]
                    b = self._brk(forming, bar, start_r, avg_bot, bar, avg_bot, False)
                    if b is not None:
                        e = self.entry_at(b); t = avg_bot - height_r
                        st = self.stop_for(e, t, False)
                        if self.valid(e, st, t):
                            return b, False, "Rectangle", e, st, t, h2[0]
        if self._cup_pending is not None:
            return self._cup_pending
        return None


def detect(d, gate=None, forming=False):
    """Returns (bar_index, timestamp, is_bull, name, entry, stop, target).

    Dedupe follows the source's _is_new_pattern: a detection is NEW only if its
    START_IDX (the pattern's origin pivot) is more than `cooldown` bars from the
    last emitted pattern's start_idx. Gating on the SCAN bar -- what I did
    before -- let one formation emit several times from different scans."""
    m = MT(d)
    out = []
    frozen_start = None
    g = GATE_BAR if gate is None else gate
    for bar in range(max(g, 2*LB_LEFT+2), m.N):
        r = m.detect_at(bar, forming=forming)
        if r is None:
            continue
        b, is_bull, name, e, s, t, start_idx = r
        if frozen_start is not None and abs(start_idx - frozen_start) <= COOLDOWN:
            continue
        out.append((b, d.index[b], is_bull, name, e, s, t))
        frozen_start = start_idx
    return out

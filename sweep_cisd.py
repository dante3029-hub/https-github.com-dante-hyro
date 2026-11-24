#!/usr/bin/env python3
"""
sweep_cisd.py -- Python port of "HTF Sweep CISD Setups [WillyAlgoTrader] v1.7.1",
default settings, tested across many coins. Pre-registered; one run per cell.

PORTED (defaults; the user's chart: Breaker / Breaker Close, Limit, stop at sweep extreme)
  HTF            auto-paired: 30m -> 1D, 1h -> 1D (UTC days)
  step 1         live sweeps of the previous HTF high/low, and of any untaken HTF
                 high/low from the last 10 HTF candles
  structure      LTF pivots (length 3); after the sweep extreme, a close through the
                 most recent pivot formed BEFORE the extreme with a displacement body
                 >= 0.5 ATR(14) = breaker
  entry          limit at the breaker (pivot) candle's close; market at the close if
                 price is already through it; skip if the level already traded since
  stop           the sweep extreme;  risk must be 0.25 - 3.0 ATR
  targets        TP1 1R (stop -> break-even), TP3 3R closes; opposite signal closes
  orders         cancel after 60 bars, or if TP1 trades before the fill
  context        expires 36 bars after the last trigger / 120 bars after creation
  one position at a time, same-bar long+short skipped

THREE ENGINES, reported side by side
  as-shown   the indicator's own defaults: Optimistic intrabar, cost 0, touch fills
             (what the TradingView panel shows)
  honest     Conservative intrabar, 0.15% round trip (fees + slippage) charged in R,
             limit fills only if price trades 0.05% THROUGH the entry
  random     honest engine, same coin, side and stop distance (% of price),
             market entry at a random bar's open -- isolates the setup's timing

CELLS
  A  30m -> 1D   25 coins with 5m data, Feb 2025 - Aug 2026
  B  1h  -> 1D   top-30 coins by volume, Feb 2023 - Sep 2026

VERDICT  honest mean R > 0, beats random, monthly t > 2, both halves > 0
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

PIVOT_LEN = 3
MAX_WAIT = 36
PRE_BARS = 30
FVG_LIFE = 120
MAX_CTX_AGE = 120
LIQ_LOOKBACK = 10
ATR_LEN = 14
MIN_RISK_ATR = 0.25
MAX_RISK_ATR = 3.0
DISP_ATR = 0.5
MIN_FVG_ATR = 0.1
PEND_EXPIRY = 60
TP = (1.0, 2.0, 3.0)
MAX_REPLAY = 450
WARMUP = 50

COST_PCT = 0.0015          # honest: round-trip fees + slippage, as a fraction of price
PENETRATION = 0.0005       # honest: limit fills only 0.05% through
SIMS = 20


def rma_atr(h, l, c, n=ATR_LEN):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n - 1] = np.nanmean(tr[:n])
        for i in range(n, len(c)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return np.nan_to_num(out)


class Ctx:
    """Bullish-space context (dir=1 raw prices; dir=-1 prices negated, high/low swapped)."""

    def __init__(self, d):
        self.dir = d
        self.active = False
        self.pv = []                       # (p, bar, low, close) pivots in this space
        self.reset_all()

    def reset_all(self):
        self.trigBar = self.createdBar = None
        self.level = self.levelBar = self.anchor = None
        self.ext = self.extBar = None
        self.inRun = False
        self.rOpen = self.rHigh = self.rLow = self.rMinC = self.rStart = None
        self.rPH = self.rPL = self.rPC = self.rPBar = None
        self.kOpen = self.kHigh = self.kLow = self.kMinC = self.kStart = None
        self.kPH = self.kPL = self.kPC = self.kPBar = None
        self.oTop, self.oBot, self.oBar = [], [], []
        self.reset_found()

    def reset_found(self):
        self.obX = self.fvg = self.ifvg = self.brk = False
        self.brTop = self.brBot = self.brC = self.brXBar = None
        self.kLowX = None

    def add_pivot(self, p, b, lw, cl):
        self.pv.append((p, b, lw, cl))
        if len(self.pv) > 40:
            self.pv.pop(0)

    def begin(self, lvT, lvlBar, anch, bar):
        pv = self.pv
        self.reset_all()
        self.pv = pv
        self.active = True
        self.trigBar = self.createdBar = bar
        self.level, self.levelBar, self.anchor = lvT, lvlBar, anch

    def advance(self, bi, o, h, l, c, h1, l1, c1, h2, l2, cnt, a):
        disp = c - o >= DISP_ATR * a
        newExt = False
        if cnt and (self.ext is None or l < self.ext):
            self.ext, self.extBar, newExt = l, bi, True
        if c < o:
            if not self.inRun:
                self.inRun = True
                self.rOpen, self.rHigh, self.rLow, self.rMinC, self.rStart = o, h, l, c, bi
                self.rPH, self.rPL, self.rPC, self.rPBar = h1, l1, c1, bi - 1
            else:
                self.rHigh = max(self.rHigh, h); self.rLow = min(self.rLow, l); self.rMinC = min(self.rMinC, c)
        else:
            self.inRun = False
        if newExt:
            self.kOpen, self.kHigh, self.kLow, self.kMinC, self.kStart = self.rOpen, self.rHigh, self.rLow, self.rMinC, self.rStart
            self.kPH, self.kPL, self.kPC, self.kPBar = self.rPH, self.rPL, self.rPC, self.rPBar
            self.reset_found()
        elif self.inRun and self.rStart == self.kStart:
            self.kMinC, self.kLow = self.rMinC, self.rLow
        # opposing FVGs (only matter for IFVG entries; kept for faithfulness)
        if l2 is not None and h < l2 and l2 - h >= MIN_FVG_ATR * a:
            self.oTop.append(l2); self.oBot.append(h); self.oBar.append(bi - 1)
        for j in range(len(self.oTop) - 1, -1, -1):
            tp, ob = self.oTop[j], self.oBar[j]
            inWin = cnt and self.extBar is not None and bi >= self.extBar
            if (c > tp) or (bi - ob > FVG_LIFE):
                del self.oTop[j], self.oBot[j], self.oBar[j]
        if cnt and self.extBar is not None and bi >= self.extBar:
            if self.brk and self.brBot is not None and c < self.brBot:
                self.brk = False
            if not self.brk:
                lv = None
                for (p, pb, lw, cl) in reversed(self.pv):
                    if pb < self.extBar and pb + PIVOT_LEN <= bi:
                        lv, zb, zc, zx = p, lw, cl, pb
                        break
                if lv is not None and c > lv and disp:
                    self.brk, self.brXBar, self.brTop, self.brBot, self.brC = True, bi, lv, zb, zc


def run_coin(d, engine, rng=None, plan_override=None):
    """engine: 'shown' | 'honest'. Returns list of (ts, dir, R, stop_pct)."""
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    atr = rma_atr(h, l, c)
    day = d.index.floor("D").to_numpy()
    honest = engine == "honest"
    cost_frac = COST_PCT if honest else 0.0

    bull, bear = Ctx(1), Ctx(-1)
    htfs, liqH, liqL = [], [], []          # liq: [price, bar, idx]
    cur = None
    htfCount = 0
    first_done = False
    bPrevDone = sPrevDone = False
    tr = None                               # dict
    pend = None
    out = []

    def tview(ctx, i):
        """bar i values in ctx space, with i-1, i-2."""
        if ctx.dir == 1:
            g = lambda k: (h[k], l[k], c[k]) if k >= 0 else (None, None, None)
            h1, l1, c1 = g(i - 1); h2, l2, _ = g(i - 2)
            return o[i], h[i], l[i], c[i], h1, l1, c1, h2, l2
        g = lambda k: (-l[k], -h[k], -c[k]) if k >= 0 else (None, None, None)
        h1, l1, c1 = g(i - 1); h2, l2, _ = g(i - 2)
        return -o[i], -l[i], -h[i], -c[i], h1, l1, c1, h2, l2

    def close_trade(px, i):
        nonlocal tr
        cost_r = cost_frac * tr["entry"] / tr["risk"]
        r = tr["dir"] * (px - tr["entry"]) / tr["risk"] - cost_r
        out.append((d.index[tr["bar"]], tr["dir"], r, tr["risk"] / tr["entry"]))
        tr = None

    for i in range(n):
        # ---- pivots (confirmed PIVOT_LEN bars later) ----
        k = i - PIVOT_LEN
        if k - PIVOT_LEN >= 0:
            win_h = h[k - PIVOT_LEN:i + 1]; win_l = l[k - PIVOT_LEN:i + 1]
            if h[k] == win_h.max() and (win_h == h[k]).sum() == 1:
                bull.add_pivot(h[k], k, l[k], c[k])
            if l[k] == win_l.min() and (win_l == l[k]).sum() == 1:
                bear.add_pivot(-l[k], k, -h[k], -c[k])

        # ---- HTF aggregation ----
        bt = st = None                       # [lvl, lvlBar, anchor]
        def put(t, lvl, lb, anch, dirn):
            if t is None:
                return [lvl, lb, anch]
            if (dirn == 1 and lvl < t[0]) or (dirn == -1 and lvl > t[0]):
                t[0], t[1] = lvl, lb
            t[2] = min(t[2], anch)
            return t
        newHtf = cur is None or day[i] != day[i - 1]
        if newHtf:
            if cur is not None and first_done:
                if htfs:
                    p = htfs[-1]
                htfs.append(cur)
                htfCount += 1
                liqH.append([cur["h"], cur["hb"], htfCount])
                liqL.append([cur["l"], cur["lb"], htfCount])
            if cur is not None:
                first_done = True
            cur = {"o": o[i], "h": h[i], "l": l[i], "c": c[i], "b": i, "hb": i, "lb": i}
            bPrevDone = sPrevDone = False
        else:
            if h[i] > cur["h"]:
                cur["h"], cur["hb"] = h[i], i
            if l[i] < cur["l"]:
                cur["l"], cur["lb"] = l[i], i
            cur["c"] = c[i]
        if htfs:
            p = htfs[-1]
            if not bPrevDone and l[i] < p["l"]:
                bt = put(bt, p["l"], p["lb"], i, 1); bPrevDone = True
            if not sPrevDone and h[i] > p["h"]:
                st = put(st, p["h"], p["hb"], i, -1); sPrevDone = True
        for arr, isHigh in ((liqL, False), (liqH, True)):
            sw = swb = None
            for j in range(len(arr) - 1, -1, -1):
                q = arr[j]
                if htfCount - q[2] >= LIQ_LOOKBACK:
                    del arr[j]
                elif (h[i] > q[0]) if isHigh else (l[i] < q[0]):
                    if sw is None or (q[0] > sw if isHigh else q[0] < sw):
                        sw, swb = q[0], q[1]
                    del arr[j]
            if sw is not None:
                if isHigh:
                    st = put(st, sw, swb, i, -1)
                else:
                    bt = put(bt, sw, swb, i, 1)

        # ---- contexts ----
        for ctx, tg in ((bull, bt), (bear, st)):
            started = False
            if tg is not None:
                lvT = ctx.dir * tg[0]
                if not ctx.active:
                    ctx.begin(lvT, tg[1], tg[2], i); started = True
                else:
                    ctx.trigBar = i
                    if lvT < ctx.level:
                        ctx.level, ctx.levelBar = lvT, tg[1]
            if ctx.active:
                m = 0
                if started:
                    m = max(0, min(max(i - ctx.anchor, 0) + PRE_BARS, MAX_REPLAY, i - 2))
                for bi in range(i - m, i + 1):
                    v = tview(ctx, bi)
                    ctx.advance(bi, *v, bi >= ctx.anchor, atr[bi])
            if ctx.active and (i - ctx.trigBar > MAX_WAIT or i - ctx.createdBar > MAX_CTX_AGE):
                ctx.active = False

        # ---- trade management ----
        if tr is not None and i > tr["bar"]:
            dn, sl = tr["dir"], tr["sl"]
            slHit = (l[i] <= sl) if dn == 1 else (h[i] >= sl)
            tps = [(h[i] >= t) if dn == 1 else (l[i] <= t) for t in (tr["tp1"], tr["tp2"], tr["tp3"])]
            gapSl = (o[i] <= sl) if dn == 1 else (o[i] >= sl)
            gapTp = (o[i] >= tr["tp1"]) if dn == 1 else (o[i] <= tr["tp1"])
            slFirst = slHit and (gapSl or (honest and not gapTp))
            tp1First = tps[0] and not tr["hit1"] and not slFirst
            if tp1First:
                tr["hit1"] = True
                if not tr["be"]:
                    tr["sl"], tr["be"] = tr["entry"], True
            tp3Close = tps[2] and not slFirst
            closeBeyond = (c[i] <= sl) if dn == 1 else (c[i] >= sl)
            closeBySl = slHit and not tp3Close and (slFirst or not tp1First or closeBeyond)
            if tp3Close:
                close_trade(max(o[i], tr["tp3"]) if dn == 1 else min(o[i], tr["tp3"]), i)
            elif closeBySl:
                close_trade(min(o[i], sl) if dn == 1 else max(o[i], sl), i)

        # ---- pending order ----
        if pend is not None and tr is None and i > pend["bar"]:
            dn, en = pend["dir"], pend["en"]
            thr = en * (1 - PENETRATION) if (honest and dn == 1) else en * (1 + PENETRATION) if honest else en
            touched = (l[i] <= thr) if dn == 1 else (h[i] >= thr)
            tp1Before = (h[i] >= pend["tp1"]) if dn == 1 else (l[i] <= pend["tp1"])
            if i - pend["bar"] > PEND_EXPIRY:
                pend = None
            elif touched:
                fill = min(o[i], en) if dn == 1 else max(o[i], en)
                tr = {"dir": dn, "entry": fill, "sl": pend["sl"], "tp1": pend["tp1"], "tp2": pend["tp2"],
                      "tp3": pend["tp3"], "risk": abs(fill - pend["sl"]), "bar": i, "hit1": False, "be": False}
                pend = None
                gapStop = dn * (fill - tr["sl"]) <= 0
                stopTouch = (l[i] <= tr["sl"]) if dn == 1 else (h[i] >= tr["sl"])
                stopClose = (c[i] <= tr["sl"]) if dn == 1 else (c[i] >= tr["sl"])
                if gapStop or (stopTouch if honest else stopClose):
                    close_trade(fill if gapStop else tr["sl"], i)
            elif tp1Before:
                pend = None

        # ---- signals (bar close) ----
        if i < WARMUP:
            continue
        plans = []
        for ctx in (bull, bear):
            if not ctx.active or not ctx.brk:
                continue
            dd = ctx.dir
            e, s = ctx.brC, ctx.ext
            cT = c[i] if dd == 1 else -c[i]
            stale = False
            for k2 in range(ctx.brXBar + 1, i):
                if (l[k2] if dd == 1 else -h[k2]) <= e:
                    stale = True; break
            if stale or s is None or not (e > s and cT > s):
                continue
            mkt = cT <= e
            en = cT if mkt else e
            risk = en - s
            if not (risk > 0 and cT > s) or atr[i] <= 0:
                continue
            ra = risk / atr[i]
            if ra < MIN_RISK_ATR or ra > MAX_RISK_ATR:
                continue
            ctx.active = False
            fresh = tr is not None and tr["dir"] == -dd and tr["bar"] == i
            acc = (tr is None or tr["dir"] != dd) and (pend is None or pend["dir"] != dd) and not fresh
            plans.append((dd, acc, mkt, dd * en, dd * s, [dd * (en + risk * m) for m in TP]))
        acc_plans = [p for p in plans if p[1]]
        if len(acc_plans) == 2:
            acc_plans = []
        if not acc_plans:
            continue
        dd, _, mkt, en, sl, tps = acc_plans[0]
        if tr is not None and tr["dir"] == -dd:
            close_trade(c[i], i)
        if pend is not None and pend["dir"] == -dd:
            pend = None
        if tr is None and dd * (en - sl) > 0:
            if mkt:
                tr = {"dir": dd, "entry": en, "sl": sl, "tp1": tps[0], "tp2": tps[1], "tp3": tps[2],
                      "risk": abs(en - sl), "bar": i, "hit1": False, "be": False}
            else:
                pend = {"dir": dd, "en": en, "sl": sl, "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "bar": i}
    return out


def random_control(d, trades, rg):
    """Honest engine mechanics, random market entries: same side and stop % as each real trade."""
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n, res = len(c), []
    for _, dn, _, sp in trades:
        i = int(rg.integers(WARMUP, n - 2))
        en = o[i]; risk = en * sp; sl = en - dn * risk
        tp1, tp3 = en + dn * risk, en + dn * 3 * risk
        hit1, j = False, i
        r = None
        while j < n:
            slHit = (l[j] <= sl) if dn == 1 else (h[j] >= sl)
            if slHit:                                       # conservative: stop first
                px = (min(o[j], sl) if dn == 1 else max(o[j], sl)) if j > i else sl
                r = dn * (px - en) / risk; break
            if ((h[j] >= tp1) if dn == 1 else (l[j] <= tp1)) and not hit1:
                hit1, sl = True, en
            if (h[j] >= tp3) if dn == 1 else (l[j] <= tp3):
                r = 3.0; break
            j += 1
        if r is None:
            r = dn * (c[-1] - en) / risk
        res.append(r - COST_PCT / sp)
    return res


def report(label, per_coin, data, rg):
    def stats(eng):
        rows = [(ts, coin, dn, r, sp) for coin, tr in per_coin[eng].items() for ts, dn, r, sp in tr]
        return pd.DataFrame(rows, columns=["ts", "coin", "dir", "R", "sp"]).sort_values("ts")
    shown, hon = stats("shown"), stats("honest")
    print(f"  {label}")
    print(f"    as shown on TradingView : {len(shown):5d} trades  avg {shown.R.mean():+.3f}R  total {shown.R.sum():+.1f}R"
          f"  win(TP1) n/a  PF {shown.R[shown.R>0].sum()/max(-shown.R[shown.R<0].sum(),1e-9):.2f}")
    if len(hon) < 50:
        print(f"    honest: only {len(hon)} trades\n"); return
    ctrl = []
    for _ in range(SIMS):
        acc = []
        for coin, tr in per_coin["honest"].items():
            acc += random_control(data[coin], tr, rg)
        ctrl.append(np.mean(acc))
    cm = float(np.mean(ctrl))
    mon = hon.groupby(hon.ts.dt.to_period("M")).R.mean()
    tm = mon.mean() / (mon.std(ddof=1) / math.sqrt(len(mon)))
    mid = hon.ts.iloc[len(hon) // 2]
    h1, h2 = hon.R[hon.ts < mid].mean(), hon.R[hon.ts >= mid].mean()
    pf = hon.R[hon.R > 0].sum() / max(-hon.R[hon.R < 0].sum(), 1e-9)
    ok = hon.R.mean() > 0 and hon.R.mean() > cm and tm > 2 and h1 > 0 and h2 > 0
    pos = sum(1 for _, g in hon.groupby("coin") if g.R.sum() > 0)
    print(f"    honest (fees, fills)    : {len(hon):5d} trades  avg {hon.R.mean():+.3f}R  total {hon.R.sum():+.1f}R"
          f"  PF {pf:.2f}  median stop {hon.sp.median()*100:.2f}%  cost/trade {(COST_PCT/hon.sp).median():.2f}R")
    print(f"    random entries, same stop: avg {cm:+.3f}R   excess {hon.R.mean()-cm:+.3f}R")
    print(f"    monthly t {tm:+.2f} ({int((mon>0).sum())}/{len(mon)} months up)   halves {h1:+.3f} / {h2:+.3f}"
          f"   coins net positive {pos}/{hon.coin.nunique()}")
    print(f"    long {hon.R[hon.dir>0].mean():+.3f}R ({(hon.dir>0).sum()})  short {hon.R[hon.dir<0].mean():+.3f}R ({(hon.dir<0).sum()})")
    print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")


def run_cell(label, coins, loader, tf, rg):
    data, per = {}, {"shown": {}, "honest": {}}
    for coin in coins:
        d = loader(coin, tf)
        if d is None or len(d) < 2000:
            continue
        data[coin] = d
        for eng in ("shown", "honest"):
            per[eng][coin] = run_coin(d, eng)
    report(label, per, data, rg)


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    import of_flow as OF
    import fvg_revisit as FV
    rg = np.random.default_rng(zlib.crc32(b"sweep_cisd"))
    print("\n  HTF Sweep CISD (WillyAlgoTrader v1.7.1), default settings -- pre-registered\n")
    run_cell("A  30m -> 1D, 25 coins, Feb 2025 - Aug 2026", OF.coins5(), OF.load_flow5, 30, rg)
    top30 = FV.top_by_liquidity(SR.universe(), EA.load_bars, 30)
    run_cell("B  1h -> 1D, top 30 coins, Feb 2023 - Sep 2026", top30, EA.load_bars, 1, rg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

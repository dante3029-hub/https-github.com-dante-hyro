#!/usr/bin/env python3
"""
================================================================================
 THREE-SLEEVE MARKET-NEUTRAL STRATEGY  —  REFERENCE IMPLEMENTATION
================================================================================
 Verified on a CLEAN fixed-universe panel:
     27 coins, all with identical coverage from 2024-01-20, all >$5M/day volume
     2.42 years of daily data, 884 days
 VERIFIED RESULTS
     core (DELTA+RELVOL, 60% stop)          Sharpe 2.09
     + BOS sleeve at 33% risk               Sharpe 2.38   (halves 1.82 / 3.01)
     correlation core vs BOS                       -0.04
     worst single day                              -3.73 x daily vol
     max drawdown (vol-normalised)                 10.8
 THE THREE SLEEVES
     DELTA   cross-sectional taker-delta ranking      (long top N / short bottom N)
     RELVOL  cross-sectional relative-volume ranking  (long top N / short bottom N)
     BOS     per-coin market-structure short, 4h bars (short only, 3 ATR TP)
 FOUR METHODOLOGY RULES THAT ARE NOT OPTIONAL
   1. FIXED UNIVERSE. One coin set, one start date. An expanding universe
      silently changes the strategy mid-backtest and inflates results ~15%.
   2. PHASE-AVERAGE THE REBALANCE. Run all 7 weekday offsets and average.
      Anchoring to any single weekday is a lottery: the same signal scored
      1.58 phase-averaged vs 0.98 Tuesday-only (and -0.05 in the first half).
   3. DENSE DAILY SERIES. Sparse sleeves (BOS trades ~33% of days) must be
      ZERO-FILLED onto the full calendar. Intersecting calendars measures only
      on trade days and inflates Sharpe ~4x on sparse sleeves.
   4. PER-LEG STOP-LOSS. Alt shorts have unbounded loss. Without a stop, one
      squeeze (e.g. ORDI +149% in a day) produces a -11 sigma day that alone
      dictates position size. The 60% stop cuts the tail to -3.7 sigma and
      more than doubles safe size.
================================================================================
"""
import numpy as np, glob, os, csv, datetime as dt
from datetime import datetime, timezone
# ------------------------------------------------------------------ CONFIG
HIST, TAKER, OI = 'clean_panel/hist', 'clean_panel/taker', 'clean_panel/oi'
FEE          = 0.00055 + 0.0003   # taker fee + slippage per side (verified vs live fills)
START        = dt.date(2024, 1, 20)
MIN_LIQ      = 5_000_000          # median $/day; peak Sharpe at this threshold
N_PER_SIDE   = 6                  # long N / short N per cross-sectional sleeve
HOLD_DAYS    = 7
PHASES       = 7                  # rebalance-phase averaging (RULE 2)
STOP_FRAC    = 0.24               # per-leg stop (RULE 4). 2026-08-10: account owner instructed
                                   # tightening from 0.60 to 0.24. NOTE this sits OUTSIDE the
                                   # documented stable plateau (40-80%) that was the original basis
                                   # for choosing 0.60 -- the stop-tightening sweep found the sharpest,
                                   # most non-monotonic Sharpe/CAGR gains below 0.30, which is the
                                   # signature of curve-fitting to noise, not a real edge. The intrabar
                                   # 1h test also found real execution would be materially worse than
                                   # this daily-close backtest assumes at this stop level: among 129
                                   # stop-outs, median 3.75% extra adverse slippage vs the assumed
                                   # daily-close fill, 42% would fire a day earlier intrabar, and 4.11%
                                   # of trades this model counts as full wins actually breached -24%
                                   # intrabar and recovered by the close (a real stop order would have
                                   # exited those at a loss). Do not treat the backtested Sharpe/CAGR
                                   # improvement from this change as validated live edge.
DELTA_LB     = 3                  # taker-delta lookback (days)
RELVOL_LB    = 20                 # relative-volume baseline (days)
BOS_PIVOT_K  = 5                  # swing pivot confirmation bars (4h)
BOS_ATR_STOP = 2.0
BOS_TP_ATR   = 3.0                # TP plateau 2.0-5.0; this is not a fitted point
BOS_EXIT_BAR = 30
BOS_MAXPOS   = 6
W_CORE, W_BOS = 2/3, 1/3          # risk weights
D = lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc).date()
zs = lambda x: (x - np.nanmean(x)) / np.nanstd(x) if np.nanstd(x) > 0 else x*0.0
nz = lambda x: x / x.std() if x.std() > 0 else x
sharpe = lambda x: x.mean()/x.std()*np.sqrt(365) if len(x) > 10 and x.std() > 0 else 0.0
# ------------------------------------------------------------------ LOADERS
def _read(path, ts_col, val_col, accumulate=False):
    out = {}
    if not os.path.exists(path): return out
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                d = D(int(row[ts_col])); v = float(row[val_col])
                out[d] = out.get(d, 0.0) + v if accumulate else v
            except (ValueError, IndexError):
                continue
    return out
def _taker(coin):
    out = {}; p = f"{TAKER}/{coin}_taker_1h.csv"
    if not os.path.exists(p): return out
    with open(p) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                d = D(int(row[1]))
                a, b = out.get(d, (0.0, 0.0))
                out[d] = (a + float(row[6]), b + float(row[3]))   # delta, volume
            except (ValueError, IndexError):
                continue
    return out
def select_universe():
    """Fixed universe: identical coverage + liquidity floor (RULE 1)."""
    keep = []
    for f in sorted(glob.glob(f"{HIST}/*_1h.csv")):
        c = os.path.basename(f).replace("_1h.csv", "")
        if c == 'BTC': continue                       # index/hedge reference only
        qv, first = [], None
        with open(f) as fh:
            r = csv.reader(fh); next(r, None)
            for row in r:
                try:
                    if first is None: first = D(int(row[1]))
                    qv.append(float(row[7]))          # quote (USD) volume — NOT col 6
                except (ValueError, IndexError):
                    continue
        if qv and first and first <= START and np.median(qv)*24 >= MIN_LIQ:
            keep.append(c)
    return sorted(keep)
def build_matrices(coins):
    # 2026-08-10 FIX: an empty coin list previously produced a cryptic
    # "TypeError: unbound method set.intersection() needs an argument" from
    # set.intersection(*[]) below. A silently-empty universe is the exact bug
    # class that invalidated an earlier data panel, so fail LOUDLY and early.
    if not coins:
        raise RuntimeError(
            f"no coins selected -- check that price data exists under {HIST!r} "
            f"and that START/MIN_LIQ are not excluding everything. "
            f"(This package does not ship market data; point HIST/TAKER/OI at "
            f"your own panel.)"
        )
    CL = {c: _read(f"{HIST}/{c}_1h.csv", 1, 5)              for c in coins}
    VO = {c: _read(f"{HIST}/{c}_1h.csv", 1, 6, True)        for c in coins}
    FU = {c: _read(f"{OI}/{c}_funding.csv", 0, 1, True)     for c in coins}
    TK = {c: _taker(c)                                      for c in coins}
    dates = sorted(d for d in set.intersection(*[set(CL[c]) for c in coins]) if d >= START)
    T, C = len(dates), len(coins); idx = {d: i for i, d in enumerate(dates)}
    PX = np.full((T, C), np.nan); VL = np.full((T, C), np.nan)
    FN = np.zeros((T, C)); DE = np.zeros((T, C)); TV = np.zeros((T, C))
    for j, c in enumerate(coins):
        for d, v in CL[c].items():
            if d in idx: PX[idx[d], j] = v
        for d, v in VO[c].items():
            if d in idx: VL[idx[d], j] = v
        for d, v in FU[c].items():
            if d in idx: FN[idx[d], j] = v
        for d, (dl, vo) in TK[c].items():
            if d in idx: DE[idx[d], j] = dl; TV[idx[d], j] = vo
    R  = np.full((T, C), np.nan); R[1:] = PX[1:]/PX[:-1] - 1
    DN = np.divide(DE, np.where(TV > 0, TV, np.nan))    # normalised taker flow [-1,1]
    RV = np.full((T, C), np.nan)
    for t in range(RELVOL_LB+1, T):
        RV[t] = VL[t-1] / np.nanmean(VL[t-RELVOL_LB:t], axis=0)
    return dates, PX, R, FN, DN, RV
# ------------------------------------------------- CROSS-SECTIONAL SLEEVES
def run_sleeve(signal_fn, dates, PX, R, FN, stop=STOP_FRAC,
               n=N_PER_SIDE, hold=HOLD_DAYS, phases=PHASES):
    """
    Long top-n / short bottom-n, dollar-neutral, weekly, PHASE-AVERAGED (RULE 2)
    with a per-leg stop-loss (RULE 4). Returns a DENSE daily array.
    """
    T, C = PX.shape
    acc = np.zeros(T)
    for ph in range(phases):
        w_prev = np.zeros(C); entry = np.full(C, np.nan); out = np.zeros(T)
        for t in range(45, T):
            # --- per-leg stop: flatten any position beyond `stop` against entry
            if stop is not None:
                for j in range(C):
                    if w_prev[j] != 0 and not np.isnan(entry[j]) and not np.isnan(PX[t, j]):
                        move = (PX[t, j]/entry[j] - 1) * np.sign(w_prev[j])
                        if move < -stop:
                            w_prev[j] = 0.0; entry[j] = np.nan
            carry = np.nansum(w_prev*np.nan_to_num(R[t])) - np.nansum(w_prev*FN[t])
            if (t - ph) % hold != 0:
                out[t] = carry; continue
            s = signal_fn(t); valid = ~np.isnan(s)
            if valid.sum() < 2*n + 2:
                out[t] = carry; continue
            order = np.argsort(np.where(valid, s, -1e9))
            w = np.zeros(C); w[order[-n:]] = 0.5/n; w[order[:n]] = -0.5/n
            out[t] = (np.nansum(w*np.nan_to_num(R[t]))
                      - np.abs(w - w_prev).sum()*FEE
                      - np.nansum(w*FN[t]))
            w_prev = w; entry = np.where(w != 0, PX[t], np.nan)
        acc += out
    return acc[45:] / phases
def sig_delta(DN):
    """Rank by recent net taker flow. NOTE: subtracting a cross-sectional mean
    here is a NO-OP because zscore() removes any constant shift — the original
    'flow divergence' formulation was mathematically identical to raw delta."""
    return lambda t: zs(np.nansum(DN[t-DELTA_LB:t], axis=0))
def sig_relvol(RV):
    """Rank by volume vs its own 20-day baseline. Correlation with delta ~ +0.3."""
    return lambda t: zs(RV[t])
# --------------------------------------------------------- BOS SHORT SLEEVE
def _atr(h, l, c, p=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    a = np.full(len(c), np.nan)
    if len(c) > p:
        a[p] = tr[:p].mean()
        for i in range(p+1, len(c)):
            a[i] = (a[i-1]*(p-1) + tr[i-1]) / p
    return a
def _pivots(h, l, k):
    """CAUSAL swing pivots: a pivot at bar i is only CONFIRMED at bar i+k."""
    n = len(h); ph = np.full(n, np.nan); pl = np.full(n, np.nan)
    for i in range(k, n-k):
        if h[i] == max(h[i-k:i+k+1]): ph[i+k] = h[i]
        if l[i] == min(l[i-k:i+k+1]): pl[i+k] = l[i]
    return ph, pl
def _load_4h(coin):
    ts=[];o=[];h=[];l=[];c=[];v=[];tb=[]
    with open(f"{HIST}/{coin}_1h.csv") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                if D(int(row[1])) < START: continue
                ts.append(int(row[1])); o.append(float(row[2])); h.append(float(row[3]))
                l.append(float(row[4]));  c.append(float(row[5])); v.append(float(row[6]))
                tb.append(float(row[9]))                      # taker buy base
            except (ValueError, IndexError):
                continue
    if len(c) < 500: return None
    A = [np.array(x) for x in (ts,o,h,l,c,v,tb)]; m = len(A[0])//4
    agg = lambda x, how: {'f': x[:m*4].reshape(m,4)[:,0], 'x': x[:m*4].reshape(m,4)[:,-1],
                          'mx': x[:m*4].reshape(m,4).max(1), 'mn': x[:m*4].reshape(m,4).min(1),
                          's': x[:m*4].reshape(m,4).sum(1)}[how]
    return (agg(A[0],'f'), agg(A[1],'f'), agg(A[2],'mx'), agg(A[3],'mn'),
            agg(A[4],'x'), agg(A[5],'s'), agg(A[6],'s'))
def run_bos(coins, calendar):
    """
    Market-structure SHORT, 4h bars. Trend state from confirmed swing pivots:
        state +1 = HH & HL (uptrend), -1 = LH & LL (downtrend)
    BOS short = in a DOWNTREND, price closes below the last confirmed swing low
                (continuation). CHoCH — the reversal version — tested at -0.38
                and is NOT used.
    Confirmed by negative taker delta. Exit: 3 ATR take-profit, 2 ATR stop, or
    a close above the last swing high. The TP is what makes it work
    (0.26 -> 1.00) and is a plateau across 2.0-5.0 ATR.
    Returns a DENSE daily array (RULE 3: zero-filled, trades only ~33% of days).
    """
    trades = []
    for coin in coins:
        d4 = _load_4h(coin)
        if d4 is None: continue
        ts, o, hh, ll, cl, vv, tb = d4
        n = len(cl); A = _atr(hh, ll, cl); ph, pl = _pivots(hh, ll, BOS_PIVOT_K)
        dn = np.divide(2*tb - vv, np.where(vv > 0, vv, np.nan))
        lastH = lastL = prevH = prevL = np.nan; state = 0
        i = BOS_PIVOT_K + 20
        while i < n-1:
            if not np.isnan(ph[i]): prevH, lastH = lastH, ph[i]
            if not np.isnan(pl[i]): prevL, lastL = lastL, pl[i]
            if not any(np.isnan(v) for v in (lastH, prevH, lastL, prevL)):
                if   lastH > prevH and lastL > prevL: state = 1
                elif lastH < prevH and lastL < prevL: state = -1
            fresh_break = (not np.isnan(lastL)) and cl[i] < lastL and cl[i-1] >= lastL
            trig = fresh_break and state == -1
            trig = trig and (not np.isnan(dn[i])) and dn[i] < 0
            if trig and not np.isnan(A[i]) and A[i] > 0:
                ent  = o[i+1]
                stop = ent + BOS_ATR_STOP*A[i]
                tp   = ent - BOS_TP_ATR*A[i]
                j = i+1; px = None
                while j < min(i+1+BOS_EXIT_BAR, n):
                    if hh[j] >= stop: px = stop; break
                    if ll[j] <= tp:   px = tp;   break
                    if not np.isnan(lastH) and cl[j] > lastH:
                        px = o[min(j+1, n-1)]; break
                    j += 1
                if px is None: px = cl[min(j, n-1)]
                trades.append((ts[i+1], ts[min(j, n-1)], (ent-px)/ent - 2*FEE))
                i = j+1; continue
            i += 1
    trades.sort()
    live, taken, daily = [], [], {}
    for entry_ts, exit_ts, pnl in trades:
        live = [x for x in live if x > entry_ts]
        if len(live) < BOS_MAXPOS:
            live.append(exit_ts); taken.append((exit_ts, pnl))
    for exit_ts, pnl in taken:
        d = D(exit_ts); daily[d] = daily.get(d, 0.0) + pnl
    return np.array([daily.get(d, 0.0) for d in calendar])   # ZERO-FILL (RULE 3)
# ------------------------------------------------------------------- MAIN
def main():
    coins = select_universe()
    print(f"universe: {len(coins)} coins >= ${MIN_LIQ/1e6:.0f}M/day, from {START}")
    dates, PX, R, FN, DN, RV = build_matrices(coins)
    delta  = run_sleeve(sig_delta(DN),  dates, PX, R, FN)
    relvol = run_sleeve(sig_relvol(RV), dates, PX, R, FN)
    core   = (nz(delta) + nz(relvol)) / 2
    cal = []; d = dates[45]
    while d <= dates[-1]: cal.append(d); d += dt.timedelta(days=1)
    bos = run_bos(coins, cal)
    L = min(len(core), len(bos)); c2, b2 = core[-L:], bos[-L:]
    blend = W_CORE*nz(c2) + W_BOS*nz(b2)
    u = blend / blend.std(); half = L//2; yrs = L/365
    print(f"\n{'sleeve':<24}{'Sharpe':>8}")
    print(f"{'DELTA':<24}{sharpe(delta):>8.2f}")
    print(f"{'RELVOL':<24}{sharpe(relvol):>8.2f}")
    print(f"{'BOS':<24}{sharpe(bos):>8.2f}")
    print(f"{'core (D+RV)':<24}{sharpe(core):>8.2f}")
    print(f"{'BLEND 2/3 core + 1/3 BOS':<24}{sharpe(blend):>8.2f}")
    print(f"\ncorr(core, BOS)  {np.corrcoef(c2, b2)[0,1]:+.2f}")
    print(f"halves           {sharpe(u[:half]):.2f} / {sharpe(u[half:]):.2f}")
    print(f"SE on Sharpe     {np.sqrt((1+sharpe(u)**2/2)/yrs):.2f}   ({yrs:.2f} years)")
    print(f"worst single day {u.min():.2f} x vol")
    print(f"max drawdown     {(np.maximum.accumulate(np.cumsum(u))-np.cumsum(u)).max():.1f}")
    print(f"max safe vol     ${10000/abs(u.min()):,.0f}/day  (vs $10k daily loss limit)")
    print(f"\n{'size':<13}{'ann $':>11}{'pass%':>7}{'fail%':>7}{'medDays':>9}{'E[days]':>9}")
    for V in (1500, 2000, 2500, 3000):
        dl = u*V; n = len(dl); rng = np.random.default_rng(11)
        P = F = 0; days = []; fdays = []
        for st in rng.integers(0, max(1, int(n*0.6)), size=12000):
            eq = 0.0
            for k in range(int(st), min(int(st)+400, n)):
                if dl[k] <= -10000: F += 1; fdays.append(k-st); break
                eq += dl[k]
                if eq <= -20000:    F += 1; fdays.append(k-st); break
                if eq >= 20000 and (k-st) >= 10: P += 1; days.append(k-st); break
        p = P/12000
        md = np.median(days) if days else 0
        mf = np.median(fdays) if fdays else 0
        E  = md + (1-p)/p*(mf+7) if p > 0.05 else float('nan')
        print(f"  ${V}/day{'':<3}${dl.mean()*365:>10,.0f}{p*100:>7.0f}%{F/120:>7.0f}%{md:>9.0f}{E:>9.0f}")
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OPTION 1 -- REFERENCE IMPLEMENTATION, reconstructed verbatim from the attached
PDF (option1-reference.pdf) with data paths pointed at run/hist, run/taker,
run/oi (symlinked to the user's actual coins_history / taker_data / oi_data).

Lines cut off by PDF pagination are completed with the only sensible
continuation and flagged inline with `# [completed from context]`.

sleeve_S is NOT reconstructed here -- run/short_engine_backtest.py does not
exist anywhere in the uploaded files (checked ex/botcode, ex/botcode-1: no
Ichimoku/chandelier/BOS/EMA200 cascade logic present; strategy.py implements
an unrelated Supertrend engine). See sleeve_S_reconstructed.py for a best-
effort rebuild from the prose spec, run separately and clearly labeled.
"""
import numpy as np, glob, os, csv, datetime as dt, sys
from datetime import datetime, timezone

HIST_DIR, TAKER_DIR, OI_DIR = 'run/hist', 'run/taker', 'run/oi'
EXCLUDE      = ("BTC", "ETHBTC")
FEE          = 0.00055 + 0.0003
N_PER_SIDE   = 5
HOLD_DAYS    = 7
MOM_LOOKBACK = 14
CARRY_WINDOW = 7
MOM_WEIGHT   = 0.5
FLOW_LOOKBACKS = [2, 3, 5]
SHORT_MAX_POS  = 6
TREND_SMA      = 200
TREND_WEIGHT   = 0.20
WARMUP         = 40
D = lambda ms: datetime.fromtimestamp(ms/1000, tz=timezone.utc).date()

def _read(path, ts_col, val_col, accumulate=False):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                d = D(int(row[ts_col])); v = float(row[val_col])
                out[d] = out.get(d, 0.0) + v if accumulate else v
            except (ValueError, IndexError):
                continue
    return out

def load_close(c):   return _read(f"{HIST_DIR}/{c}_1h.csv", 1, 5)
def load_funding(c): return _read(f"{OI_DIR}/{c}_funding.csv", 0, 1, accumulate=True)

def load_taker(c):
    out = {}; p = f"{TAKER_DIR}/{c}_taker_1h.csv"
    if not os.path.exists(p): return out
    with open(p) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            try:
                d = D(int(row[1])); dl, vo = float(row[6]), float(row[3])
                a, b = out.get(d, (0.0, 0.0)); out[d] = (a+dl, b+vo)
            except (ValueError, IndexError):
                continue
    return out

def build_matrices():
    coins = [os.path.basename(f).replace("_1h.csv", "")
             for f in glob.glob(f"{HIST_DIR}/*_1h.csv")]
    coins = sorted(c for c in coins if c not in EXCLUDE)
    CL  = {c: load_close(c)   for c in coins}
    FU  = {c: load_funding(c) for c in coins}
    TK  = {c: load_taker(c)   for c in coins}
    dates = sorted(set().union(*[set(CL[c]) for c in coins]))
    T, C  = len(dates), len(coins)
    idx   = {d: i for i, d in enumerate(dates)}
    PX = np.full((T, C), np.nan); FN = np.zeros((T, C))
    DE = np.zeros((T, C));        VO = np.zeros((T, C))
    for j, c in enumerate(coins):
        for d, v in CL[c].items():
            if d in idx: PX[idx[d], j] = v
        for d, v in FU[c].items():
            if d in idx: FN[idx[d], j] = v
        for d, (dl, vo) in TK[c].items():
            if d in idx: DE[idx[d], j] = dl; VO[idx[d], j] = vo
    R = np.full((T, C), np.nan); R[1:] = PX[1:] / PX[:-1] - 1
    DN = np.divide(DE, np.where(VO > 0, VO, np.nan))
    BF = np.nanmean(DN, axis=1)
    return coins, dates, PX, R, FN, DN, BF

def zscore(x):
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * 0.0

def run_sleeve(signal_fn, dates, PX, R, FN, T, C,
               hold=HOLD_DAYS, n=N_PER_SIDE, fee=FEE):
    prev_w = np.zeros(C); out = {}
    for t in range(WARMUP, T):
        carry_pnl = (np.nansum(prev_w * np.nan_to_num(R[t]))
                     - np.nansum(prev_w * FN[t]))          # [completed from context]
        if t % hold != 0:
            out[dates[t]] = carry_pnl; continue
        s  = signal_fn(t)
        vd = ~np.isnan(s)
        if vd.sum() < 2*n + 2:
            out[dates[t]] = carry_pnl; continue
        order = np.argsort(np.where(vd, s, -1e9))
        w = np.zeros(C); w[order[-n:]] = 0.5/n; w[order[:n]] = -0.5/n
        out[dates[t]] = (np.nansum(w * np.nan_to_num(R[t]))
                         - np.abs(w - prev_w).sum() * fee
                         - np.nansum(w * FN[t]))
        prev_w = w
    return out

def sleeve_A(PX, FN):
    def sig(t):
        mom   = PX[t-2] / PX[t-2-MOM_LOOKBACK] - 1
        carry = -np.nanmean(FN[t-CARRY_WINDOW:t], axis=0)
        return MOM_WEIGHT*zscore(mom) + (1-MOM_WEIGHT)*zscore(carry)
    return sig

def sleeve_F(DN, BF, lookback):
    def sig(t):
        return zscore(np.nansum(DN[t-lookback:t], axis=0) - np.nansum(BF[t-lookback:t]))
    return sig

def btc_trend_overlay(dates):
    bt = _read(f"{HIST_DIR}/BTC_1h.csv", 1, 5)
    bd = sorted(bt); bc = np.array([bt[d] for d in bd])
    sma = np.array([np.mean(bc[max(0, i-TREND_SMA):i]) if i >= TREND_SMA else np.nan
                    for i in range(len(bc))])
    ret = np.zeros(len(bc)); ret[1:] = bc[1:]/bc[:-1] - 1
    tr, bull = {}, {}
    for i, d in enumerate(bd):
        up = i > 0 and not np.isnan(sma[i-1]) and bc[i-1] > sma[i-1]
        tr[d] = ret[i] if up else 0.0; bull[d] = up
    return tr, bull

def sharpe(x):
    x = np.asarray(x, float)
    return x.mean()/x.std()*np.sqrt(365) if len(x) > 5 and x.std() > 0 else 0.0

def max_dd(x):
    eq = np.cumsum(x / (x.std() or 1))
    return float((np.maximum.accumulate(eq) - eq).max())

def blend(dates_axis, sleeves_dense, sparse_sleeve):
    cols = []
    for s in sleeves_dense:
        cols.append(np.array([s[d] for d in dates_axis]))
    cols.append(np.array([sparse_sleeve.get(d, 0.0) for d in dates_axis]))
    norm = [c / (c.std() or 1) for c in cols]
    return np.mean(norm, axis=0), cols

if __name__ == '__main__':
    coins, dates, PX, R, FN, DN, BF = build_matrices()
    T, C = PX.shape
    print(f"universe: {len(coins)} coins | {len(dates)} days | {dates[0]} -> {dates[-1]}")
    print(f"coins: {coins}\n")

    A  = run_sleeve(sleeve_A(PX, FN), dates, PX, R, FN, T, C)
    Fs = [run_sleeve(sleeve_F(DN, BF, lb), dates, PX, R, FN, T, C) for lb in FLOW_LOOKBACKS]
    axis = sorted(set(A) & set(Fs[0]))
    F_ens = {d: float(np.mean([f[d] for f in Fs])) for d in axis}

    a_col = np.array([A[d] for d in axis])
    f_col = np.array([F_ens[d] for d in axis])
    print(f"sleeve A solo (momentum+carry): Sharpe {sharpe(a_col):.3f}")
    print(f"sleeve F solo (flow ensemble {FLOW_LOOKBACKS}): Sharpe {sharpe(f_col):.3f}")
    for lb, f in zip(FLOW_LOOKBACKS, Fs):
        fc = np.array([f[d] for d in axis])
        print(f"  F[L={lb}] solo: Sharpe {sharpe(fc):.3f}")
    print(f"corr(A,F): {np.corrcoef(a_col, f_col)[0,1]:+.3f}")

    core_AF = 0.5*(a_col/(a_col.std() or 1) + f_col/(f_col.std() or 1))
    print(f"\nA+F equal-risk blend (no S -- S engine unavailable, see below):")
    print(f"  Sharpe {sharpe(core_AF):.3f}")
    cut = axis[-1] - dt.timedelta(days=548)
    recent_mask = np.array([d >= cut for d in axis])
    if recent_mask.sum() > 30:
        print(f"  last-18mo Sharpe: {sharpe(core_AF[recent_mask]):.3f}  "
              f"(n={recent_mask.sum()} days -- 18mo window barely fits the "
              f"{ (axis[-1]-axis[0]).days } total days available)")

    tr_map, bull_map = btc_trend_overlay(dates)
    axis2 = [d for d in axis if d in tr_map]
    ci = np.array([core_AF[axis.index(d)] for d in axis2])
    tr = np.array([tr_map[d] for d in axis2])
    bl = np.array([bull_map[d] for d in axis2])
    w  = np.where(bl, TREND_WEIGHT, 0.0)
    funded = (1-w)*(ci/(ci.std() or 1)) + w*(tr/(tr.std() or 1))
    print(f"\n+20% BTC trend overlay (A+F core, no S): Sharpe {sharpe(funded):.3f} "
          f"| bull-days {sharpe(funded[bl]):.3f} | bull regime {bl.mean()*100:.0f}% of days")

    print(f"\nDATA WINDOW CHECK:")
    print(f"  alt-coin history spans: {dates[0]} -> {dates[-1]} "
          f"= {(dates[-1]-dates[0]).days/365.25:.2f} years")
    print(f"  spec claims: 3.5 years")

    tue = sum(1 for i, d in enumerate(dates) if i % HOLD_DAYS == 0 and i >= WARMUP)
    tue_actual = sum(1 for i, d in enumerate(dates)
                      if i % HOLD_DAYS == 0 and i >= WARMUP and d.weekday() == 1)
    print(f"\nREBALANCE-DAY CHECK: of {tue} rebalance events (t % 7 == 0), "
          f"{tue_actual} actually fall on a Tuesday ({tue_actual/max(tue,1)*100:.0f}%)")

    print(f"\n=== COVERAGE BUG CHECK ===")
    cov = (~np.isnan(PX)).sum(axis=1)
    print(f"coins with live price data on first day ({dates[0]}): {cov[0]}/25")
    print(f"coins with live price data on {dates[365]} (~1yr in): {cov[365]}/25")
    idx_2025 = next(i for i, d in enumerate(dates) if d >= dt.date(2025,1,1))
    print(f"date index where coverage reaches 25/25: {idx_2025} ({dates[idx_2025]})")
    print(f"coverage the day before: {cov[idx_2025-1]}/25 -> day of: {cov[idx_2025]}/25")

    print(f"\n=== CLEAN-WINDOW RE-RUN: restrict to dates where all 25 coins have data ===")
    clean_start = idx_2025
    def rerun_clean(signal_fn_factory, *args):
        prev_w = np.zeros(C); out = {}
        for t in range(max(WARMUP, clean_start), T):
            s  = signal_fn_factory(*args)(t) if callable(signal_fn_factory) else None
        return out
    # simplest correct approach: just re-slice PX/R/FN/DN/BF to start at clean_start
    # and re-run WARMUP relative to that slice so no pre-2025 coin ever enters the window
    PXc, Rc, FNc, DNc = PX[clean_start-WARMUP:], R[clean_start-WARMUP:], FN[clean_start-WARMUP:], DN[clean_start-WARMUP:]
    BFc = BF[clean_start-WARMUP:]
    datesc = dates[clean_start-WARMUP:]
    Tc = len(datesc)
    Ac  = run_sleeve(sleeve_A(PXc, FNc), datesc, PXc, Rc, FNc, Tc, C)
    Fsc = [run_sleeve(sleeve_F(DNc, BFc, lb), datesc, PXc, Rc, FNc, Tc, C) for lb in FLOW_LOOKBACKS]
    axisc = sorted(set(Ac) & set(Fsc[0]))
    F_ensc = {d: float(np.mean([f[d] for f in Fsc])) for d in axisc}
    ac = np.array([Ac[d] for d in axisc]); fc = np.array([F_ensc[d] for d in axisc])
    print(f"clean window: {datesc[WARMUP]} -> {datesc[-1]}  ({len(axisc)} rebalanced-day rows)")
    print(f"  sleeve A solo: Sharpe {sharpe(ac):.3f}")
    print(f"  sleeve F solo: Sharpe {sharpe(fc):.3f}")
    print(f"  corr(A,F): {np.corrcoef(ac, fc)[0,1]:+.3f}")
    core_clean = 0.5*(ac/(ac.std() or 1) + fc/(fc.std() or 1))
    print(f"  A+F equal-risk blend: Sharpe {sharpe(core_clean):.3f}")

    print(f"\n=== SPLIT-HALF ROBUSTNESS CHECK on clean window (small-sample check) ===")
    mid = axisc[len(axisc)//2]
    m1 = np.array([d < mid for d in axisc]); m2 = ~m1
    print(f"n rebalance events total: {len(axisc)} (~{len(axisc)//7} independent weekly bets)")
    print(f"  A  first half {sharpe(ac[m1]):.3f}  second half {sharpe(ac[m2]):.3f}")
    print(f"  F  first half {sharpe(fc[m1]):.3f}  second half {sharpe(fc[m2]):.3f}")
    for lb, f in zip(FLOW_LOOKBACKS, Fsc):
        fcl = np.array([f[d] for d in axisc])
        print(f"  F[L={lb}] first half {sharpe(fcl[m1]):.3f}  second half {sharpe(fcl[m2]):.3f}")
    n_weekly = len(axisc)//7
    print(f"\n  noise ceiling check: expected max Sharpe from spurious weekly strategies, "
          f"n~{n_weekly} independent bets over {len(axisc)/365.25:.2f}y ~ "
          f"{np.sqrt(2*np.log(max(n_weekly,2)))/np.sqrt(len(axisc)/365.25):.2f}")

    # Save clean-window A, F for blend with reconstructed sleeve S
    import pandas as pd
    pd.Series(ac, index=pd.to_datetime(axisc)).to_csv('sleeve_A_clean.csv', header=['pnl'])
    pd.Series(fc, index=pd.to_datetime(axisc)).to_csv('sleeve_F_clean.csv', header=['pnl'])
    print(f"\nsaved sleeve_A_clean.csv, sleeve_F_clean.csv ({len(axisc)} rows, {datesc[WARMUP]} -> {datesc[-1]})")

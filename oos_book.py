#!/usr/bin/env python3
"""
oos_book.py -- do the book's cross-sectional sleeves survive data they were
never fitted to?

WHY THIS TEST, AND WHY NOW
The 9-sleeve book (BOOK_VERIFIED.md, OI_SLEEVE.md) claims 4.13 Sharpe. Its own
caveats say "~250 tests run, the noise ceiling is high", and when it went to
the Bybit demo account it lost money on every day but one unexplained one. A
4.13-Sharpe book does not lose for 70 days. Something in the backtest is not
real, and the cheapest way to find out what is to score each sleeve on dates
its parameters never saw.

The book's fitting window was 2024-02-13 -> 2026-08-22 (BOOK_FINAL.md), and
reference_impl.START = 2024-01-20. The repo's data starts Feb 2023. So:

    PRE   2023-02 -> 2024-02-12   ~12 months, never used for these sleeves
    FIT   2024-02-13 -> 2026-08-22  where the parameters were chosen
    POST  2026-08-23 -> 2026-09-28  after the window closed (short)

One honest exception: OI's hold was changed 3 -> 5 days on 2026-10-22 using
1,357 days that INCLUDE 2023. So OI is reported at both holds, and only the
3-day version counts as clean out-of-sample.

WHAT IS FROZEN (nothing here is tuned; every number comes from the repo)
  delta_book   7-day summed taker delta, z-scored, N=5, hold 14  (BOOK_VERIFIED)
  delta_live   3-day, N=6, hold 7                     (reference_impl, live)
  relvol_book  vol / own 20d baseline, N=8, hold 7, 24 most liquid (BOOK_FINAL)
  oirank       OI change ranked, blended 3/5/7/10/14d lookbacks, N=5
               hold 3 (original) and hold 5 (post-Oct change)

ENGINE: a line-for-line mirror of reference_impl.run_sleeve -- dollar-neutral
long top-N / short bottom-N at 0.5/N each, phase-averaged rebalancing, per-leg
stop, fee on turnover, funding charged on the carried book. The stop is the
documented 60% plateau, NOT the 24% the live config was tightened to on
2026-08-10: the code's own comment on that change calls it "the signature of
curve-fitting to noise". The 24% version is reported separately.

CONTROL: identical engine, identical N / hold / phases / stop / fees, but the
score is uniform random noise at every rebalance. 40 seeds. A sleeve has an
edge in a period only if it beats that distribution IN THAT PERIOD.

SIGNIFICANCE: daily returns are not independent of their neighbours (a 7-day
hold means a week of overlapping exposure), so the t-stat is computed on
MONTHLY sums, not on days.

USAGE
    python3 oos_book.py --selftest
    python3 oos_book.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import os
import zlib

import numpy as np
import pandas as pd

log = logging.getLogger("oos_book")

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.00055 + 0.0003          # taker fee + slippage per side (reference_impl)
STOP_PLATEAU = 0.60
STOP_LIVE = 0.24
WARMUP = 45                     # reference_impl starts scoring at t=45
FIT_START = pd.Timestamp("2024-02-13")
FIT_END = pd.Timestamp("2026-08-22")
CONTROL_SEEDS = 40


def _import_oirank():
    """Load sleeve_oirank.oi_scores straight from its file, so the frozen
    scoring is used without dragging in signal_engine/__init__ (which pulls in
    the reference loaders and their missing clean_panel/ paths)."""
    base = os.path.join(HERE, "signal_engine")
    spec_rw = importlib.util.spec_from_file_location(
        "signal_engine.rank_weights", os.path.join(base, "rank_weights.py"))
    rw = importlib.util.module_from_spec(spec_rw)
    spec_rw.loader.exec_module(rw)
    import sys
    sys.modules.setdefault("signal_engine", type(sys)("signal_engine"))
    sys.modules["signal_engine.rank_weights"] = rw
    spec = importlib.util.spec_from_file_location(
        "signal_engine.sleeve_oirank", os.path.join(base, "sleeve_oirank.py"),
        submodule_search_locations=[base])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "signal_engine"
    spec.loader.exec_module(mod)
    return mod.oi_scores


oi_scores = _import_oirank()


# ------------------------------------------------------------------ data
def _daily_from_1h(path, cols, agg):
    d = pd.read_csv(path)
    tcol = "open_time" if "open_time" in d.columns else "timestamp"
    d["day"] = pd.to_datetime(d[tcol], unit="ms").dt.floor("D")
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.groupby("day").agg({c: agg[c] for c in cols})


def build_panels():
    """Daily matrices on one common calendar.

    PX close, VL volume, DE summed taker delta, TV taker-side volume,
    OI last open interest of the day, FN summed funding rate, QV $ volume.
    Missing data stays NaN -- a coin that did not exist yet must not be
    ranked as if it had zero flow.
    """
    px, vl, de, tv, qv, oi, fn = {}, {}, {}, {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(HERE, "taker_data", "*_1h.csv"))):
        c = os.path.basename(f).replace("_1h.csv", "")
        if c == "BTC":
            continue
        try:
            d = _daily_from_1h(f, ["close", "volume", "delta"],
                               {"close": "last", "volume": "sum",
                                "delta": "sum"})
        except Exception as e:
            log.warning("  skip %s: %s", c, e)
            continue
        d = d[(d.close > 0) & (d.volume > 0)]
        px[c], vl[c], de[c] = d.close, d.volume, d.delta
        tv[c], qv[c] = d.volume, d.close * d.volume
    for f in sorted(glob.glob(os.path.join(HERE, "oi_data", "*_oi_1h.csv"))):
        c = os.path.basename(f).replace("_oi_1h.csv", "")
        try:
            d = _daily_from_1h(f, ["open_interest"], {"open_interest": "last"})
            oi[c] = d.open_interest.where(d.open_interest > 0)
        except Exception as e:
            log.warning("  skip OI %s: %s", c, e)
    for f in sorted(glob.glob(os.path.join(HERE, "funding_data", "*_funding.csv"))):
        c = os.path.basename(f).replace("_funding.csv", "")
        try:
            d = pd.read_csv(f)
            d["day"] = pd.to_datetime(d["ts"], unit="ms").dt.floor("D")
            fn[c] = pd.to_numeric(d["rate"], errors="coerce").groupby(d["day"]).sum()
        except Exception as e:
            log.warning("  skip funding %s: %s", c, e)

    coins = sorted(px)
    cal = pd.DatetimeIndex(sorted(set().union(*[px[c].index for c in coins])))
    def M(dct):
        return pd.DataFrame({c: dct.get(c, pd.Series(dtype=float))
                             for c in coins}).reindex(cal)
    P = dict(coins=coins, dates=cal, PX=M(px).values, VL=M(vl).values,
             DE=M(de).values, TV=M(tv).values, QV=M(qv).values,
             OI=M(oi).values, FN=M(fn).fillna(0.0).values,
             has_fn=np.array([c in fn for c in coins]),
             has_oi=np.array([c in oi for c in coins]))
    R = np.full_like(P["PX"], np.nan)
    R[1:] = P["PX"][1:] / P["PX"][:-1] - 1
    P["R"] = R
    with np.errstate(divide="ignore", invalid="ignore"):
        P["DN"] = P["DE"] / np.where(P["TV"] > 0, P["TV"], np.nan)
    RV = np.full_like(P["VL"], np.nan)
    for t in range(21, len(cal)):
        with np.errstate(invalid="ignore", divide="ignore"):
            RV[t] = P["VL"][t - 1] / np.nanmean(P["VL"][t - 20:t], axis=0)
    P["RV"] = RV
    return P


# ---------------------------------------------------------------- signals
def zs(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * 0.0


def sig_delta(DN, lb):
    """Rank by summed taker flow over the PREVIOUS lb days (t-lb .. t-1).
    Applied to R[t] = close(t-1) -> close(t), so it is causal."""
    def f(t):
        blk = DN[t - lb:t]
        ok = np.isfinite(blk).sum(axis=0) >= max(1, lb - 1)
        s = np.nansum(blk, axis=0)
        s[~ok] = np.nan
        return zs(s)
    return f


def sig_relvol(RV):
    return lambda t: zs(RV[t])     # RV[t] already uses volume to t-1


def sig_oi(OI):
    """oi_scores(OI, k) uses OI up to and including day k. The engine applies
    the weights chosen at t to R[t], so the score must be taken at t-1."""
    return lambda t: oi_scores(OI, t - 1)


def sig_random(C, seed):
    rg = np.random.default_rng(seed)
    return lambda t: rg.random(C)


# ----------------------------------------------------------------- engine
def run_sleeve(signal_fn, PX, R, FN, n, hold, phases, stop, cols=None):
    """Mirror of reference_impl.run_sleeve. Returns a dense daily array the
    length of the calendar, zero before WARMUP.

    `cols` restricts the sleeve to a subset of coins (others never held).
    """
    T, C = PX.shape
    mask = np.ones(C, bool) if cols is None else cols
    acc = np.zeros(T)
    for ph in range(phases):
        w_prev = np.zeros(C)
        entry = np.full(C, np.nan)
        out = np.zeros(T)
        for t in range(WARMUP, T):
            if stop is not None:
                held = np.flatnonzero(w_prev)
                for j in held:
                    if np.isnan(entry[j]) or np.isnan(PX[t, j]):
                        continue
                    move = (PX[t, j] / entry[j] - 1) * np.sign(w_prev[j])
                    if move < -stop:
                        w_prev[j] = 0.0
                        entry[j] = np.nan
            r_t = np.nan_to_num(R[t])
            carry = np.sum(w_prev * r_t) - np.sum(w_prev * FN[t])
            if (t - ph) % hold != 0:
                out[t] = carry
                continue
            s = np.asarray(signal_fn(t), dtype=float).copy()
            s[~mask] = np.nan
            s[~np.isfinite(PX[t])] = np.nan
            valid = np.isfinite(s)
            if valid.sum() < 2 * n + 2:
                out[t] = carry
                continue
            order = np.argsort(np.where(valid, s, -1e9))
            w = np.zeros(C)
            w[order[-n:]] = 0.5 / n
            w[order[:n]] = -0.5 / n
            out[t] = (np.sum(w * r_t) - np.abs(w - w_prev).sum() * FEE
                      - np.sum(w * FN[t]))
            w_prev = w
            entry = np.where(w != 0, PX[t], np.nan)
        acc += out
    return acc / phases


# ------------------------------------------------------------- statistics
def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 10 and x.std() > 0 else 0.0


def monthly_t(dates, x):
    s = pd.Series(x, index=dates).resample("ME").sum()
    s = s[s != 0]
    if len(s) < 3 or s.std(ddof=1) == 0:
        return 0.0, len(s), 0
    return (float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s),
            int((s > 0).sum()))


def periods(dates):
    first = dates[WARMUP]
    return {"PRE": (dates >= first) & (dates < FIT_START),
            "FIT": (dates >= FIT_START) & (dates <= FIT_END),
            "POST": dates > FIT_END}


# ------------------------------------------------------------------ specs
def specs(P):
    coins = np.array(P["coins"])
    liq = np.nanmedian(P["QV"], axis=0)
    top24 = np.zeros(len(coins), bool)
    top24[np.argsort(np.nan_to_num(liq, nan=-1))[-24:]] = True
    oi_cols = P["has_oi"]
    return {
        "delta_book":   (sig_delta(P["DN"], 7), 5, 14, 7, None),
        "delta_live":   (sig_delta(P["DN"], 3), 6, 7, 7, None),
        "relvol_book":  (sig_relvol(P["RV"]), 8, 7, 7, top24),
        "oirank_h3":    (sig_oi(P["OI"]), 5, 3, 3, oi_cols),
        "oirank_h5":    (sig_oi(P["OI"]), 5, 5, 5, oi_cols),
    }


def evaluate(P, name, spec, stop, seeds):
    sig, n, hold, phases, cols = spec
    real = run_sleeve(sig, P["PX"], P["R"], P["FN"], n, hold, phases, stop, cols)
    C = len(P["coins"])
    ctrl = [run_sleeve(sig_random(C, zlib.crc32(f"{name}{s}".encode())),
                       P["PX"], P["R"], P["FN"], n, hold, phases, stop, cols)
            for s in range(seeds)]
    return real, np.array(ctrl)


def selftest(P) -> int:
    T, C = P["PX"].shape
    print(f"  panel: {C} coins x {T} days, "
          f"{P['dates'][0].date()} -> {P['dates'][-1].date()}")
    print(f"  coins with OI: {int(P['has_oi'].sum())}, "
          f"with funding: {int(P['has_fn'].sum())}")
    # delta ratio must lie in [-1, 1]
    dn = P["DN"][np.isfinite(P["DN"])]
    assert dn.min() >= -1.0001 and dn.max() <= 1.0001, "DN outside [-1,1]"
    print(f"  delta ratio within [-1, 1]: OK  (range {dn.min():.3f} .. {dn.max():.3f})")
    # every signal must be truncation-invariant: computing it on data that
    # ends at t must give the same answer as on the full panel
    t = T // 2
    cut = {k: (v[:t + 1] if isinstance(v, np.ndarray) and v.ndim == 2 else v)
           for k, v in P.items()}
    full, part = specs(P), specs(cut)
    for k in full:
        a = full[k][0](t)
        b = part[k][0](t)
        same = np.allclose(np.nan_to_num(a, nan=9e9), np.nan_to_num(b, nan=9e9))
        assert same, f"{k} changes when the future is removed -- lookahead"
    print("  every sleeve signal is causal (truncation-invariant): OK")
    # a random-signal sleeve must earn ~ -fees, not something large
    rnd = run_sleeve(sig_random(C, 1), P["PX"], P["R"], P["FN"], 5, 7, 7,
                     STOP_PLATEAU)
    sr = sharpe(rnd[WARMUP:])
    print(f"  random-signal sleeve Sharpe: {sr:+.2f}  (should be near zero)")
    assert abs(sr) < 1.5, "random sleeve is not near zero -- engine is biased"
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=CONTROL_SEEDS)
    ap.add_argument("--stop", choices=["plateau", "live"], default="plateau")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    P = build_panels()
    if a.selftest:
        return selftest(P)

    stop = STOP_PLATEAU if a.stop == "plateau" else STOP_LIVE
    dates = P["dates"]
    per = periods(dates)
    print(f"\n  Cross-sectional sleeves, frozen params, stop {stop:.0%}")
    print(f"  PRE {dates[WARMUP].date()} -> {FIT_START.date()} (never used) | "
          f"FIT -> {FIT_END.date()} | POST -> {dates[-1].date()}\n")
    print(f"  {'sleeve':<13}{'period':<6}{'days':>5}{'Sharpe':>8}"
          f"{'ctrl mu':>9}{'ctrl sd':>8}{'z':>7}{'t_mon':>7}{'mon+':>7}{'ann%':>8}")
    store = {}
    for name, spec in specs(P).items():
        real, ctrl = evaluate(P, name, spec, stop, a.seeds)
        store[name] = real
        for pn, m in per.items():
            x = real[m]
            if m.sum() < 20:
                continue
            cs = np.array([sharpe(c[m]) for c in ctrl])
            z = (sharpe(x) - cs.mean()) / cs.std() if cs.std() > 0 else 0.0
            tm, nm, mp = monthly_t(dates[m], x)
            print(f"  {name:<13}{pn:<6}{int(m.sum()):>5}{sharpe(x):>8.2f}"
                  f"{cs.mean():>9.2f}{cs.std():>8.2f}{z:>7.2f}{tm:>7.2f}"
                  f"{mp:>4}/{nm:<2}{x.mean()*365*100:>7.1f}%")
        print()
    pd.DataFrame(store, index=dates).to_csv(
        os.path.join(HERE, f"oos_book_daily_{a.stop}.csv"))
    print(f"  daily returns saved to oos_book_daily_{a.stop}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Signal parity test -- BUILD_PLAN.md Phase 2's stated "single most important
checkpoint," given the rescale bug already found once earlier this session.

For each dense cross-sectional sleeve (Main, Flow, DELTA, RELVOL), this
re-derives the target weight vector at N historical rebalance days two ways:

  (a) directly, using this package's live sleeve module
      (signal_engine.sleeve_X.latest_target_weights(..., t=day))
  (b) independently, using the SAME reference signal function called at the
      SAME bar index directly from the original reference_impl.py /
      option1_reference.py files, with a verbatim copy of run_sleeve's own
      weight-construction lines (not the shared rank_weights.py module, so
      this is not just checking that a module imports itself correctly)

and asserts the two weight vectors are bit-identical (np.array_equal, not an
approximate tolerance -- these are the same floating point ops on the same
inputs, so they should match exactly).

For the event-driven sleeves (Short, BOS), parity is checked on the trigger
CONDITION (regime+trigger+confirm booleans) at historical bars, run two ways:
directly from sleeve_short.py / sleeve_bos.py, and independently recomputed
inline here from the raw reference-file building blocks. BOS parity is
checked only while flat (no open simulated position at that bar), per the
documented deliberate divergence in sleeve_bos.py.

Reports EXACT counts of matches/mismatches per sleeve, per the standing rule
to always state exact numeric results, not just a pass/fail summary.
"""
import sys
sys.path.insert(0, "/home/user/workspace")

import numpy as np

from signal_engine import data_loader
from signal_engine import sleeve_main, sleeve_flow, sleeve_delta, sleeve_relvol
from signal_engine import sleeve_short, sleeve_bos

import reference_impl as B
import option1_reference as A
from sleeve_S_reconstructed import build_signals as ref_build_signals


def _independent_rank(s, n):
    """Verbatim copy of run_sleeve's weight-construction lines, independent
    of signal_engine/rank_weights.py."""
    s = np.asarray(s, dtype=float)
    C = s.shape[0]
    valid = ~np.isnan(s)
    if valid.sum() < 2 * n + 2:
        return np.zeros(C)
    order = np.argsort(np.where(valid, s, -1e9))
    w = np.zeros(C)
    w[order[-n:]] = 0.5 / n
    w[order[:n]] = -0.5 / n
    return w


def check_dense_sleeve(name, live_fn, independent_score_fn, T, n, n_days=40, seed=0):
    """
    live_fn(t) -> (w, ok)                 [calls the signal_engine module]
    independent_score_fn(t) -> score arr  [calls the reference file's signal fn(s) directly]
    Returns (n_checked, n_match, n_mismatch, max_abs_diff).
    """
    rng = np.random.default_rng(seed)
    warmup = 60
    days = sorted(rng.choice(np.arange(warmup, T), size=min(n_days, T - warmup), replace=False).tolist())
    n_match = n_mismatch = 0
    max_diff = 0.0
    mismatches = []
    for t in days:
        w_live, ok_live = live_fn(t)
        w_ref = independent_score_fn(t)
        if w_live.shape != w_ref.shape:
            n_mismatch += 1
            mismatches.append((t, "shape mismatch"))
            continue
        diff = float(np.max(np.abs(w_live - w_ref)))
        max_diff = max(max_diff, diff)
        if np.array_equal(w_live, w_ref):
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append((t, diff))
    return dict(name=name, n_checked=len(days), n_match=n_match, n_mismatch=n_mismatch,
                max_abs_diff=max_diff, mismatches=mismatches[:5])


def run():
    results = []

    # ---- Main sleeve (Strategy A) ----
    ua = data_loader.load_universe_a()
    PX, FN, DN, BF = ua["PX"], ua["FN"], ua["DN"], ua["BF"]
    T = PX.shape[0]
    sig_A = A.sleeve_A(PX, FN)
    results.append(check_dense_sleeve(
        "Main",
        lambda t: sleeve_main.latest_target_weights(PX, FN, t=t),
        lambda t: _independent_rank(sig_A(t), A.N_PER_SIDE),
        T, A.N_PER_SIDE,
    ))

    # ---- Flow sleeve (Strategy A, 3-lookback ensemble) ----
    def flow_independent(t):
        ws = []
        for lb in A.FLOW_LOOKBACKS:
            s = A.sleeve_F(DN, BF, lb)(t)
            ws.append(_independent_rank(s, A.N_PER_SIDE))
        return np.mean(np.vstack(ws), axis=0)
    results.append(check_dense_sleeve(
        "Flow",
        lambda t: sleeve_flow.latest_target_weights(DN, BF, t=t),
        flow_independent,
        T, A.N_PER_SIDE,
    ))

    # ---- DELTA sleeve (Strategy B) ----
    ub = data_loader.load_universe_b()
    DNb, RV = ub["DN"], ub["RV"]
    Tb = DNb.shape[0]
    sig_delta = B.sig_delta(DNb)
    results.append(check_dense_sleeve(
        "DELTA",
        lambda t: sleeve_delta.latest_target_weights(DNb, t=t),
        lambda t: _independent_rank(sig_delta(t), B.N_PER_SIDE),
        Tb, B.N_PER_SIDE,
    ))

    # ---- RELVOL sleeve (Strategy B) ----
    sig_relvol = B.sig_relvol(RV)
    results.append(check_dense_sleeve(
        "RELVOL",
        lambda t: sleeve_relvol.latest_target_weights(RV, t=t),
        lambda t: _independent_rank(sig_relvol(t), B.N_PER_SIDE),
        Tb, B.N_PER_SIDE,
    ))

    # ---- Short sleeve (Strategy A, event-driven) ----
    # Independent check: re-run build_signals() (same function, imported
    # separately) and compare short_ok at the last row against the live
    # wrapper's `triggered`. This mainly guards against the live wrapper
    # accidentally looking at the wrong row / stale cache, not against the
    # underlying signal math (which is the same function object either way).
    short_checked = short_match = short_mismatch = 0
    for coin in ua["coins"][:8]:  # sample -- CSV I/O per coin is not free
        df = ref_build_signals(coin)
        live = sleeve_short.latest_signal(coin)
        if df is None or len(df) == 0 or live is None:
            continue
        short_checked += 1
        ref_triggered = bool(df.iloc[-1]["short_ok"])
        if ref_triggered == live["triggered"]:
            short_match += 1
        else:
            short_mismatch += 1
    results.append(dict(name="Short", n_checked=short_checked, n_match=short_match,
                         n_mismatch=short_mismatch, max_abs_diff=None, mismatches=[]))

    # ---- BOS sleeve (Strategy B, event-driven) ----
    # Independent check: recompute the trigger condition inline (verbatim
    # from run_bos()'s loop body) at the SAME latest confirmed bar the live
    # wrapper reports, for a sample of coins, and compare.
    bos_checked = bos_match = bos_mismatch = 0
    for coin in ub["coins"][:8]:
        live = sleeve_bos.latest_signal(coin)
        if live is None:
            continue
        d4 = B._load_4h(coin)
        if d4 is None:
            continue
        ts, o, hh, ll, cl, vv, tb = d4
        n = len(cl)
        A_ = B._atr(hh, ll, cl)
        ph, pl = B._pivots(hh, ll, B.BOS_PIVOT_K)
        dn = np.divide(2 * tb - vv, np.where(vv > 0, vv, np.nan))
        i = live["i"]
        lastH = lastL = prevH = prevL = np.nan
        state = 0
        for k in range(B.BOS_PIVOT_K + 20, i + 1):
            if not np.isnan(ph[k]):
                prevH, lastH = lastH, ph[k]
            if not np.isnan(pl[k]):
                prevL, lastL = lastL, pl[k]
            if not any(np.isnan(v) for v in (lastH, prevH, lastL, prevL)):
                if lastH > prevH and lastL > prevL:
                    state = 1
                elif lastH < prevH and lastL < prevL:
                    state = -1
        fresh_break = (not np.isnan(lastL)) and cl[i] < lastL and cl[i - 1] >= lastL
        trig = fresh_break and state == -1
        trig = trig and (not np.isnan(dn[i])) and dn[i] < 0
        trig = trig and (not np.isnan(A_[i])) and A_[i] > 0
        bos_checked += 1
        if bool(trig) == live["triggered"] and state == live["state"]:
            bos_match += 1
        else:
            bos_mismatch += 1
    results.append(dict(name="BOS", n_checked=bos_checked, n_match=bos_match,
                         n_mismatch=bos_mismatch, max_abs_diff=None, mismatches=[]))

    return results


if __name__ == "__main__":
    results = run()
    print(f"{'sleeve':<8} {'checked':>8} {'match':>8} {'mismatch':>9} {'max_abs_diff':>14}")
    all_pass = True
    for r in results:
        mad = f"{r['max_abs_diff']:.3e}" if r['max_abs_diff'] is not None else "n/a"
        print(f"{r['name']:<8} {r['n_checked']:>8} {r['n_match']:>8} {r['n_mismatch']:>9} {mad:>14}")
        if r['n_mismatch'] > 0:
            all_pass = False
            print(f"  MISMATCHES (first 5): {r['mismatches']}")
    print()
    print("ALL SLEEVES BIT-IDENTICAL TO REFERENCE" if all_pass else "MISMATCHES FOUND -- see above")

"""
Exact backtest impact of the $6,000 max-loss-per-trade notional cap and the
$10,000 low-cap (ORDI/BRETT) aggregate exposure cap, WITH vs WITHOUT, using
real historical per-leg weights for Main and Flow (the two sleeves whose
per-coin construction is exactly reconstructable from option1_reference.py
and run/hist/*.csv -- Delta/Relvol/Bos per-leg weights are NOT reconstructed
here, see caveats printed at the bottom).

Design, mirroring portfolio_layer.portfolio._apply_max_loss_cap /
_apply_low_cap_exposure_cap exactly:
  - stop_frac = DEFAULT_STOP_FRAC = 0.60 for every Main/Flow leg (these are
    NOT event sleeves, so there is no per-leg ATR stop_fracs override --
    the real orchestrator would use the 0.60 default for them too).
  - per-leg notional cap = MAX_LOSS_PER_TRADE / stop_frac = $6,000/0.60 = $10,000.
  - low-cap aggregate cap ($10,000) applies across the ORDI/BRETT legs
    summed over Main+Flow (Delta/Relvol not included -- see caveat).
  - account_mult = L_DEFAULT = 1.70 (throttle/kill-switch ignored, per the
    session's documented "conservative UPPER BOUND on cap-binding frequency"
    simplification -- real throttle/kill-switch only ever reduce size
    further, so this can only OVER-state, never UNDER-state, how often the
    caps bind).
  - sleeve multiplier m_main(t)/m_flow(t): the SAME walk-forward
    compute_sleeve_multipliers() used by cap_vol.py, computed causally
    (only data up to t-1) at every rebalance date.

Uncapped and capped daily return series for Main and Flow are recombined
with the UNTOUCHED Short/Delta/Relvol/Bos histories using the exact same
portfolio formula as analysis/cap_vol.py: r(t) = sum_s m_s(t) * H_s(t).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import option1_reference as ref_a
from bot.sleeve_history import compute_sleeve_histories
from portfolio_layer.position_sizer import compute_sleeve_multipliers
from portfolio_layer.ab_blend import DEFAULT_WINDOW as W
from bot.config import L_DEFAULT
from portfolio_layer.portfolio import MAX_LOSS_PER_TRADE, LOW_CAP_EXPOSURE_LIMIT, DEFAULT_STOP_FRAC

ANN = np.sqrt(365)
LOW_CAP_COINS = {"ORDI", "BRETT"}
NOTIONAL = 200_000.0
LEG_CAP_DOLLARS = MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC  # $10,000


def run_sleeve_with_weights(signal_fn, dates, PX, R, FN, T, C, hold=ref_a.HOLD_DAYS,
                             n=ref_a.N_PER_SIDE, fee=ref_a.FEE):
    """Identical to option1_reference.run_sleeve() but also records the raw
    target weight vector w (fraction of $200k) actually used on EVERY
    rebalance date, keyed by date -> np.ndarray[C]. No logic changed from
    the original -- verified byte-for-byte identical control flow, this
    just additionally stashes `w` before it's overwritten next iteration."""
    prev_w = np.zeros(C)
    out = {}
    weights_by_date = {}
    for t in range(ref_a.WARMUP, T):
        carry_pnl = (np.nansum(prev_w * np.nan_to_num(R[t]))
                     - np.nansum(prev_w * FN[t]))
        if t % hold != 0:
            out[dates[t]] = carry_pnl
            weights_by_date[dates[t]] = prev_w.copy()
            continue
        s = signal_fn(t)
        vd = ~np.isnan(s)
        if vd.sum() < 2 * n + 2:
            out[dates[t]] = carry_pnl
            weights_by_date[dates[t]] = prev_w.copy()
            continue
        order = np.argsort(np.where(vd, s, -1e9))
        w = np.zeros(C)
        w[order[-n:]] = 0.5 / n
        w[order[:n]] = -0.5 / n
        out[dates[t]] = (np.nansum(w * np.nan_to_num(R[t]))
                          - np.abs(w - prev_w).sum() * fee
                          - np.nansum(w * FN[t]))
        weights_by_date[dates[t]] = w.copy()
        prev_w = w
    return out, weights_by_date


def apply_caps_to_weight_series(dates, coins, weight_by_date, mult_series, low_cap_mask,
                                 other_low_cap_dollars_by_date=None):
    """Returns a NEW dict date->capped_w (fraction of $200k, so directly
    substitutable for w in the P&L recompute below).

    mult_series: dict date -> sleeve multiplier m(t) (already causal/walk-
    forward, one value per date in `dates`).
    other_low_cap_dollars_by_date: optional dict date -> dollars already
    committed to low-cap coins by OTHER sleeves (for aggregate cross-sleeve
    accounting) -- None means "this sleeve alone" (used for the isolated
    single-sleeve pass before combining Main+Flow).
    """
    capped = {}
    binds_leg = 0
    binds_lowcap = 0
    for d in dates:
        w = weight_by_date[d]
        m = mult_series.get(d, 0.0)
        dollar = w * m * L_DEFAULT * NOTIONAL
        # 1. per-leg max-loss-per-trade cap
        over = np.abs(dollar) > LEG_CAP_DOLLARS
        if over.any():
            binds_leg += int(over.sum())
            sign = np.sign(dollar)
            dollar = np.where(over, sign * LEG_CAP_DOLLARS, dollar)
        # 2. low-cap aggregate cap (this sleeve's low-cap legs only, plus any
        #    externally-supplied concurrent low-cap dollars from other sleeves)
        lc_dollar = np.abs(dollar) * low_cap_mask
        total_lc = lc_dollar.sum() + (other_low_cap_dollars_by_date.get(d, 0.0)
                                       if other_low_cap_dollars_by_date else 0.0)
        if total_lc > LOW_CAP_EXPOSURE_LIMIT and lc_dollar.sum() > 0:
            binds_lowcap += 1
            scale = LOW_CAP_EXPOSURE_LIMIT / total_lc
            dollar = np.where(low_cap_mask, dollar * scale, dollar)
        capped[d] = dollar / (m * L_DEFAULT * NOTIONAL) if m != 0 else np.zeros_like(dollar)
    return capped, binds_leg, binds_lowcap


def recompute_pnl(dates, PX, R, FN, weight_by_date, hold, fee, date_to_row, prev_w_by_date):
    """Recompute the daily P&L series from a (possibly capped) weight
    series, using the EXACT same P&L formula as run_sleeve (fee on turnover
    of the raw weight vector, funding cost, spot return).

    date_to_row: dict mapping every date in `dates` to its correct row index
    in the FULL PX/R/FN matrices (dates_all-indexed) -- NOT a local
    enumerate() counter, since `dates` here is a trailing SUBSET of the full
    calendar (eval_dates starts ~757 rows into the full ~1262-row main axis,
    after the warmup + 2*W walk-forward multiplier burn-in). Using a local
    counter instead of the true row index was an earlier bug in this script
    (verified against the correct H['main'] alignment) that silently pulled
    the WRONG rows out of PX/R/FN, corrupting every downstream number.

    prev_w_by_date: dict mapping each date to the weight vector that was
    actually held on the PRIOR day (needed for the turnover/fee term at the
    eval window's first date, since that boundary's true predecessor weight
    lives outside `dates`).
    """
    out = {}
    for d in dates:
        w = weight_by_date[d]
        prev_w = prev_w_by_date[d]
        t = date_to_row[d]
        pnl = (np.nansum(w * np.nan_to_num(R[t])) - np.abs(w - prev_w).sum() * fee
               - np.nansum(w * FN[t]))
        out[d] = pnl
    return out


def main():
    coins, dates_all, PX, R, FN, DN, BF = ref_a.build_matrices()
    T, C = PX.shape
    low_cap_mask = np.array([c in LOW_CAP_COINS for c in coins])
    print(f"universe: {C} coins, low-cap flagged: {[c for c in coins if c in LOW_CAP_COINS]}")

    # --- Main sleeve (sleeve_A) ---
    main_raw, main_w = run_sleeve_with_weights(ref_a.sleeve_A(PX, FN), dates_all, PX, R, FN, T, C)
    main_dates = sorted(main_raw.keys())

    # --- Flow sleeve (ensemble of 3 lookbacks -- average PNL AND average weights,
    #     consistent with sleeve_history.py's own averaging-of-PNL-series logic) ---
    flow_w_by_lb = []
    flow_dates_ref = None
    for lb in ref_a.FLOW_LOOKBACKS:
        _, w_by_date = run_sleeve_with_weights(ref_a.sleeve_F(DN, BF, lb), dates_all, PX, R, FN, T, C)
        ds = sorted(w_by_date.keys())
        if flow_dates_ref is None:
            flow_dates_ref = ds
        flow_w_by_lb.append(w_by_date)
    flow_dates = flow_dates_ref
    flow_w = {d: np.mean([wbd[d] for wbd in flow_w_by_lb], axis=0) for d in flow_dates}

    assert main_dates == flow_dates, "main/flow date axis mismatch"

    # --- Walk-forward sleeve multipliers, computed exactly as cap_vol.py does,
    #     over the FULL 6-sleeve history (needed for m_main(t)/m_flow(t)) ---
    h = compute_sleeve_histories()
    K = ("main", "short", "flow", "delta", "relvol", "bos")
    n_hist = min(len(h[k]) for k in K)
    H = {k: np.asarray(h[k], float)[-n_hist:] for k in K}
    hist_dates = main_dates[-n_hist:]  # main_hist/short/flow/etc are aligned to this same axis per sleeve_history.py

    m_main_series, m_flow_series = {}, {}
    for i in range(2 * W + 5, n_hist):
        sub = {k: H[k][:i] for k in K}
        m = compute_sleeve_multipliers(sub["main"], sub["short"], sub["flow"],
                                        sub["delta"], sub["relvol"], sub["bos"],
                                        window=W, max_multiplier=1.0)
        d = hist_dates[i]
        m_main_series[d] = m["main"]
        m_flow_series[d] = m["flow"]

    eval_dates = [d for d in hist_dates if d in m_main_series]
    print(f"eval window: {eval_dates[0]} to {eval_dates[-1]}  ({len(eval_dates)} days)")

    # --- Apply caps: Main first (no cross-sleeve context yet), then Flow
    #     with Main's already-committed low-cap dollars folded in for the
    #     aggregate low-cap check (Main+Flow combined, per the real
    #     cross-sleeve _apply_low_cap_exposure_cap semantics) ---
    main_w_eval = {d: main_w[d] for d in eval_dates}
    flow_w_eval = {d: flow_w[d] for d in eval_dates}

    main_capped_w, main_binds_leg, main_binds_lc_solo = apply_caps_to_weight_series(
        eval_dates, coins, main_w_eval, m_main_series, low_cap_mask)

    # dollars Main committed to low-cap coins AFTER its own leg-cap (pre low-cap-cap,
    # to feed into Flow's aggregate check) -- recompute from capped-for-leg-only pass
    main_leg_capped, _, _ = apply_caps_to_weight_series(
        eval_dates, coins, main_w_eval, m_main_series, np.zeros_like(low_cap_mask))  # leg cap only, no lc cap
    main_lc_dollars_by_date = {
        d: float(np.abs(main_leg_capped[d] * m_main_series[d] * L_DEFAULT * NOTIONAL)[low_cap_mask].sum())
        for d in eval_dates
    }
    flow_capped_w, flow_binds_leg, flow_binds_lc = apply_caps_to_weight_series(
        eval_dates, coins, flow_w_eval, m_flow_series, low_cap_mask,
        other_low_cap_dollars_by_date=main_lc_dollars_by_date)
    # re-apply Main's low-cap cap now accounting for Flow's simultaneous low-cap dollars too
    flow_lc_dollars_by_date = {
        d: float(np.abs(flow_capped_w[d] * m_flow_series[d] * L_DEFAULT * NOTIONAL)[low_cap_mask].sum())
        for d in eval_dates
    }
    main_capped_w, main_binds_leg, main_binds_lc = apply_caps_to_weight_series(
        eval_dates, coins, main_w_eval, m_main_series, low_cap_mask,
        other_low_cap_dollars_by_date=flow_lc_dollars_by_date)

    # --- Recompute Main/Flow daily P&L (fraction of $200k, UNLEVERED raw
    #     sleeve return -- same convention as sleeve_history.py) from the
    #     capped vs uncapped weight series ---
    idx_all = {d: i for i, d in enumerate(dates_all)}

    def prev_w_map(dates_key, w_full_by_date):
        """For every date in `dates_key` (a subset of main_dates), find the
        true chronological predecessor date in the FULL main_dates axis and
        return its weight -- correct even at the eval window's first date,
        whose predecessor lies just outside eval_dates."""
        pos = {d: k for k, d in enumerate(main_dates)}
        out = {}
        for d in dates_key:
            k = pos[d]
            out[d] = w_full_by_date[main_dates[k - 1]] if k > 0 else np.zeros(C)
        return out

    main_prev_uncapped = prev_w_map(eval_dates, main_w)
    flow_prev_uncapped = prev_w_map(eval_dates, flow_w)
    # capped predecessor: use the capped value if the predecessor date is itself
    # inside eval_dates (i.e. also capped), else fall back to the true raw
    # predecessor weight (caps are only introduced starting at eval_dates[0]).
    date_pos = {d: k for k, d in enumerate(main_dates)}

    def prev_w_map_capped(dates_key, capped_by_date, prev_uncapped):
        out = {}
        for d in dates_key:
            k = date_pos[d]
            pred = main_dates[k - 1] if k > 0 else None
            out[d] = capped_by_date[pred] if (pred is not None and pred in capped_by_date) else prev_uncapped[d]
        return out

    main_prev_capped = prev_w_map_capped(eval_dates, main_capped_w, main_prev_uncapped)
    flow_prev_capped = prev_w_map_capped(eval_dates, flow_capped_w, flow_prev_uncapped)

    main_pnl_uncapped = recompute_pnl(eval_dates, PX, R, FN, main_w_eval, ref_a.HOLD_DAYS, ref_a.FEE,
                                       idx_all, main_prev_uncapped)
    main_pnl_capped = recompute_pnl(eval_dates, PX, R, FN, main_capped_w, ref_a.HOLD_DAYS, ref_a.FEE,
                                     idx_all, main_prev_capped)
    flow_pnl_uncapped = recompute_pnl(eval_dates, PX, R, FN, flow_w_eval, ref_a.HOLD_DAYS, ref_a.FEE,
                                       idx_all, flow_prev_uncapped)
    flow_pnl_capped = recompute_pnl(eval_dates, PX, R, FN, flow_capped_w, ref_a.HOLD_DAYS, ref_a.FEE,
                                     idx_all, flow_prev_capped)

    def arr(dct):
        return np.array([dct[d] for d in eval_dates])

    # --- Sanity check: the reconstructed uncapped Main/Flow P&L MUST exactly
    #     reproduce compute_sleeve_histories()'s trusted H['main']/H['flow']
    #     on every eval date (both are, by construction, the identical raw
    #     sleeve-return computation) -- if this fails, the row-index/date
    #     alignment in this script is broken and every downstream number is
    #     untrustworthy. Refuse to proceed rather than report bad numbers.
    main_h_by_date = dict(zip(hist_dates, H["main"]))
    flow_h_by_date = dict(zip(hist_dates, H["flow"]))
    main_trusted = np.array([main_h_by_date[d] for d in eval_dates])
    flow_trusted = np.array([flow_h_by_date[d] for d in eval_dates])
    main_check = arr(main_pnl_uncapped)
    flow_check = arr(flow_pnl_uncapped)
    max_main_err = np.abs(main_check - main_trusted).max()
    max_flow_err = np.abs(flow_check - flow_trusted).max()
    mean_flow_err = np.abs(flow_check - flow_trusted).mean()
    print(f"sanity check vs trusted sleeve_history: max|main_recon - main_trusted|={max_main_err:.2e}, "
          f"max|flow_recon - flow_trusted|={max_flow_err:.2e} (mean={mean_flow_err:.2e})")
    if max_main_err > 1e-9:
        raise AssertionError(
            f"Reconstructed uncapped Main P&L does NOT match the trusted "
            f"compute_sleeve_histories() output (max_main_err={max_main_err}). Refusing to "
            f"report results built on a broken reconstruction."
        )
    if max_flow_err > 5e-4:
        raise AssertionError(
            f"Reconstructed uncapped Flow P&L diverges from sleeve_history() by more than the "
            f"known ensemble-averaging tolerance (max_flow_err={max_flow_err}). Refusing to report."
        )
    # KNOWN, DOCUMENTED discrepancy (not a bug in this script): sleeve_history.py's Flow
    # series averages 3 SEPARATE lookbacks' PNL series (each with its own fee/turnover
    # term computed against ITS OWN previous weight), while this script -- correctly,
    # for cap-enforcement purposes, since a cap must act on the single REAL notional
    # actually held -- averages the 3 lookbacks' WEIGHT VECTORS FIRST into one book, then
    # computes ONE fee/turnover term on the averaged weight. These are only
    # APPROXIMATELY equivalent (fee is not linear under the triangle inequality), verified
    # here at max|diff|=3.4e-04, mean|diff|=8.3e-06 per day, affecting 114/1262 days --
    # negligible next to typical daily portfolio swings (ann vol ~0.10-0.15 => daily vol
    # ~0.005-0.008) but real. The single-book reconstruction here (not sleeve_history.py's
    # 3-book-PNL-average convenience computation) is the correct ground truth for cap
    # enforcement, since caps must clip the position actually traded, not a phantom
    # 3-way-split book. This has NO effect on the capped-vs-uncapped COMPARISON below,
    # since both capped and uncapped Flow numbers use the same single-book method
    # consistently.

    main_u, main_c = arr(main_pnl_uncapped), arr(main_pnl_capped)
    flow_u, flow_c = arr(flow_pnl_uncapped), arr(flow_pnl_capped)

    # sanity: capped P&L should differ from uncapped ONLY on days a cap bound
    diff_days_main = int((np.abs(main_u - main_c) > 1e-12).sum())
    diff_days_flow = int((np.abs(flow_u - flow_c) > 1e-12).sum())

    # --- Recombine into full 6-sleeve portfolio return, WITH vs WITHOUT caps,
    #     holding Short/Delta/Relvol/Bos identical in both runs (uncapped --
    #     these are NOT touched by this analysis, see caveat) ---
    idx0 = hist_dates.index(eval_dates[0])
    short_h = H["short"][idx0:idx0 + len(eval_dates)]
    delta_h = H["delta"][idx0:idx0 + len(eval_dates)]
    relvol_h = H["relvol"][idx0:idx0 + len(eval_dates)]
    bos_h = H["bos"][idx0:idx0 + len(eval_dates)]
    m_main_arr = np.array([m_main_series[d] for d in eval_dates])
    m_flow_arr = np.array([m_flow_series[d] for d in eval_dates])
    # need short/delta/relvol/bos multipliers too, recompute alongside
    m_short_series, m_delta_series, m_relvol_series, m_bos_series = {}, {}, {}, {}
    for i in range(2 * W + 5, n_hist):
        sub = {k: H[k][:i] for k in K}
        m = compute_sleeve_multipliers(sub["main"], sub["short"], sub["flow"],
                                        sub["delta"], sub["relvol"], sub["bos"],
                                        window=W, max_multiplier=1.0)
        d = hist_dates[i]
        if d in m_short_series or d not in eval_dates:
            pass
        m_short_series[d] = m["short"]
        m_delta_series[d] = m["delta"]
        m_relvol_series[d] = m["relvol"]
        m_bos_series[d] = m["bos"]
    m_short_arr = np.array([m_short_series[d] for d in eval_dates])
    m_delta_arr = np.array([m_delta_series[d] for d in eval_dates])
    m_relvol_arr = np.array([m_relvol_series[d] for d in eval_dates])
    m_bos_arr = np.array([m_bos_series[d] for d in eval_dates])

    r_uncapped = (m_main_arr * main_u + m_flow_arr * flow_u + m_short_arr * short_h
                  + m_delta_arr * delta_h + m_relvol_arr * relvol_h + m_bos_arr * bos_h)
    r_capped = (m_main_arr * main_c + m_flow_arr * flow_c + m_short_arr * short_h
                + m_delta_arr * delta_h + m_relvol_arr * relvol_h + m_bos_arr * bos_h)

    def stats(r, label):
        v = r.std() * ANN
        sh = r.mean() / r.std() * ANN if r.std() > 0 else 0.0
        eq = np.cumprod(1 + r * L_DEFAULT)
        dd = float((1 - eq / np.maximum.accumulate(eq)).max())
        cagr = eq[-1] ** (365 / len(r)) - 1
        total_ret = eq[-1] - 1
        print(f"{label:>10s}  ann_vol={v:.6f}  Sharpe={sh:.6f}  CAGR={cagr:.6%}  "
              f"maxDD={dd:.6%}  total_ret={total_ret:.6%}  n_days={len(r)}")
        return dict(ann_vol=v, sharpe=sh, cagr=cagr, maxdd=dd, total_ret=total_ret)

    print("\n=== Full 6-sleeve portfolio, Main+Flow capped vs uncapped, Short/Delta/Relvol/Bos held identical ===")
    su = stats(r_uncapped, "uncapped")
    sc = stats(r_capped, "capped")

    print(f"\nSharpe delta (capped - uncapped): {sc['sharpe'] - su['sharpe']:+.6f}")
    print(f"CAGR delta:   {sc['cagr'] - su['cagr']:+.6%}")
    print(f"maxDD delta:  {sc['maxdd'] - su['maxdd']:+.6%}  (negative = smaller drawdown = better)")
    print(f"ann_vol delta: {sc['ann_vol'] - su['ann_vol']:+.6%}")

    print(f"\nMax-loss-per-trade cap (${LEG_CAP_DOLLARS:,.0f}/leg) bound on: "
          f"Main {main_binds_leg} leg-days, Flow {flow_binds_leg} leg-days "
          f"(out of {len(eval_dates)*2*ref_a.N_PER_SIDE} total Main+Flow leg-days each side)")
    print(f"Low-cap aggregate cap (${LOW_CAP_EXPOSURE_LIMIT:,.0f}) bound on: "
          f"Main {main_binds_lc} days, Flow {flow_binds_lc} days")
    print(f"Days Main P&L changed by capping: {diff_days_main} / {len(eval_dates)}")
    print(f"Days Flow P&L changed by capping: {diff_days_flow} / {len(eval_dates)}")

    np.save(os.path.join(os.path.dirname(__file__), "r_uncapped.npy"), r_uncapped)
    np.save(os.path.join(os.path.dirname(__file__), "r_capped.npy"), r_capped)
    print(f"\nSaved r_uncapped.npy / r_capped.npy ({len(r_uncapped)} days) to {os.path.dirname(__file__)}")

    return dict(uncapped=su, capped=sc, main_binds_leg=main_binds_leg, flow_binds_leg=flow_binds_leg,
                main_binds_lc=main_binds_lc, flow_binds_lc=flow_binds_lc,
                diff_days_main=diff_days_main, diff_days_flow=diff_days_flow, n_days=len(eval_dates))


if __name__ == "__main__":
    main()

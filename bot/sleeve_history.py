"""
Reconstructs each of the 6 sleeves' trailing daily-return history (as a
fraction of $200k REFERENCE_NOTIONAL, assuming that sleeve's raw target
weight vector was run at ~1.0 gross notional every day) as of a given
as_of_date, by re-running the EXACT SAME validated backtest engine functions
already used throughout Phase 0-3 (reference_impl.py, option1_reference.py,
sleeve_S_reconstructed.py) on as_of_date-trimmed data.

This is the "sleeve_return_history" input required by
portfolio_layer.portfolio.size_portfolio(). No sleeve logic is reimplemented
here -- every series is produced by calling the same function already used
to build the reference backtest series (raw_delta_ret.npy / raw_relvol_ret.npy
/ raw_bos_ret.npy came from exactly these calls in backtest_causal_b.py;
build_blend.py's main/short/flow series came from the option1_reference /
sleeve_S_reconstructed equivalents), just re-run on a trimmed window ending
the day BEFORE as_of_date so as_of_date's own not-yet-realized return is
never included -- this matches size_portfolio()'s documented contract.

CAVEATS carried forward, not silently dropped:
  - Short sleeve uses sleeve_S_reconstructed.backtest_sleeve_S(), whose own
    docstring says "RECONSTRUCTION, NOT VERIFICATION": built from prose spec
    only, several parameters are ASSUMED, NOT validated against the spec's
    claimed 0.78 solo Sharpe.
  - All underlying data is the same static local CSVs data_loader.py reads
    (NOT live feeds) -- see AUDIT_FINDINGS.md / BUILD_PLAN.md Phase 4 for the
    confirmed live-data-ingestion gap. Calling this with as_of_date=None (or
    any date beyond the CSVs' last row) trims to whatever data actually
    exists -- it does NOT fetch anything new.
  - A full re-run of all 6 sleeves' backtest engines takes roughly 15-30s
    (BOS's per-coin 4h loop dominates) -- a per-rebalance-cycle cost, not a
    per-tick one. Acceptable given cadences are 72h/weekly for the
    weight-bearing sleeves and event-driven (4h-checked) for Short/BOS.
  - universe-A (Main/Flow/Short) and universe-B (DELTA/RELVOL/BOS) each have
    their OWN last-available data row, and can legitimately differ by days
    (confirmed: 15 days stale for universe A, 4 days for universe B, as of
    2026-08-09). Callers MUST check the returned as_of_a/as_of_b/as_of_s
    dates against the nominal as_of_date and surface a flag on a large gap,
    not silently proceed as if the data were current.
"""
import sys
import os
import glob
import datetime as dt
import numpy as np

WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import reference_impl as ref_b          # Strategy B: DELTA / RELVOL / BOS
import option1_reference as ref_a       # Strategy A: Main / Flow
import sleeve_S_reconstructed as ref_s  # Short (reconstruction, see caveat above)


def _cut_before(dates, as_of_date):
    """Index of the last row strictly BEFORE as_of_date (no lookahead)."""
    if as_of_date is None:
        return len(dates)
    cutoff = [i for i, d in enumerate(dates) if d < as_of_date]
    if not cutoff:
        raise ValueError(f"no data strictly before {as_of_date}")
    return cutoff[-1] + 1


def _main_flow_histories(as_of_date):
    coins, dates, PX, R, FN, DN, BF = ref_a.build_matrices()
    last = _cut_before(dates, as_of_date)
    dates, PX, R, FN, DN, BF = dates[:last], PX[:last], R[:last], FN[:last], DN[:last], BF[:last]
    T, C = PX.shape

    main_out = ref_a.run_sleeve(ref_a.sleeve_A(PX, FN), dates, PX, R, FN, T, C)
    main_dates = sorted(main_out.keys())
    main_hist = np.array([main_out[d] for d in main_dates])

    # Flow -- ensemble of 3 lookbacks: average the 3 independently-ranked
    # return series elementwise (see signal_engine/sleeve_flow.py docstring
    # for why this is mathematically equivalent to averaging the PNL series,
    # which is what option1_reference.py's own __main__ backtest does).
    flow_series = []
    flow_dates_ref = None
    for lb in ref_a.FLOW_LOOKBACKS:
        out = ref_a.run_sleeve(ref_a.sleeve_F(DN, BF, lb), dates, PX, R, FN, T, C)
        ds = sorted(out.keys())
        if flow_dates_ref is None:
            flow_dates_ref = ds
        elif ds != flow_dates_ref:
            raise ValueError("flow lookback date axes diverged -- unexpected data gap")
        flow_series.append(np.array([out[d] for d in ds]))
    flow_hist = np.mean(np.vstack(flow_series), axis=0)

    if main_dates != flow_dates_ref:
        raise ValueError("main and flow sleeve date axes diverged -- unexpected data gap, "
                          "refusing to combine misaligned series")

    return main_hist, flow_hist, main_dates, (dates[-1] if len(dates) else None)


def _short_history(as_of_date, align_dates):
    """
    align_dates: the EXACT date axis (list of datetime.date, ascending) the
    Short sleeve's output must be reindexed onto, so that it can be summed
    elementwise against main_hist/flow_hist in sleeve_combiner.strategy_a_combo().

    Requesting the full available history (start=None) rather than a fixed
    lookback window and then reindexing onto align_dates is deliberate: a
    fixed lookback_days window produced a DIFFERENT length/date range than
    main/flow's own warmup-trimmed axis (confirmed bug caught by the first
    end-to-end smoke test -- main_hist had 1262 rows starting 2023-02-10,
    the old fixed-180-day short_hist had a totally different start/end and
    could not be summed against it; ValueError: operands could not be
    broadcast together with shapes (1262,) (1302,)). Reindexing explicitly
    onto align_dates is the fix, and it fails loudly (raises) rather than
    silently truncating/padding if any requested date is actually missing
    from the Short sleeve's own output -- a real gap there would be a data
    problem worth surfacing, not smoothing over.
    """
    import pandas as pd
    coins = sorted(os.path.basename(f).replace('_1h.csv', '')
                    for f in glob.glob(f"{ref_s.HIST_DIR}/*_1h.csv"))
    coins = [c for c in coins if c not in ref_s.EXCLUDE]
    end = (as_of_date - dt.timedelta(days=1)) if as_of_date else None
    daily = ref_s.backtest_sleeve_S(coins, start=None, end=str(end) if end else None)
    as_of_s = daily.index[-1].date() if len(daily) else None

    daily.index = pd.DatetimeIndex(daily.index).normalize()
    want_index = pd.DatetimeIndex([pd.Timestamp(d) for d in align_dates])
    reindexed = daily.reindex(want_index)
    missing = reindexed[reindexed.isna()]
    if len(missing) > 0:
        raise ValueError(
            f"Short sleeve history is missing {len(missing)} date(s) required to align with "
            f"main/flow (e.g. {list(missing.index[:3].date)}) -- as_of_s={as_of_s}. This means "
            f"the Short sleeve's own data (run/hist 1h CSVs) does not cover the full main/flow "
            f"calendar. Refusing to fill/interpolate -- treat this sleeve as unavailable for this "
            f"cycle rather than fabricate its missing history."
        )
    return reindexed.values.astype(float), as_of_s


def _delta_relvol_bos_histories(as_of_date):
    coins = ref_b.select_universe()
    dates, PX, R, FN, DN, RV = ref_b.build_matrices(coins)
    last = _cut_before(dates, as_of_date)
    dates, PX, R, FN, DN, RV = dates[:last], PX[:last], R[:last], FN[:last], DN[:last], RV[:last]

    delta_hist = ref_b.run_sleeve(ref_b.sig_delta(DN), dates, PX, R, FN)
    relvol_hist = ref_b.run_sleeve(ref_b.sig_relvol(RV), dates, PX, R, FN)

    cal = []
    d = dates[45]
    while d <= dates[-1]:
        cal.append(d)
        d += dt.timedelta(days=1)
    bos_hist = ref_b.run_bos(coins, cal)

    # DELTA/RELVOL/BOS are all confirmed (empirically, and by re-deriving
    # reference_impl.py's own warmup logic) to come out dense over the SAME
    # 45-day-warmup-trimmed calendar (cal), ending on the same last date --
    # asserted here rather than silently trusted, since a length mismatch
    # here would corrupt strategy_b_combo() the same way the Short-sleeve
    # mismatch corrupted strategy_a_combo() (caught by the first smoke test).
    if not (len(delta_hist) == len(relvol_hist) == len(bos_hist) == len(cal)):
        raise ValueError(
            f"universe-B sleeve histories diverged in length: delta={len(delta_hist)} "
            f"relvol={len(relvol_hist)} bos={len(bos_hist)} cal={len(cal)} -- refusing to "
            f"combine misaligned series"
        )

    return delta_hist, relvol_hist, bos_hist, (dates[-1] if len(dates) else None)


def compute_sleeve_histories(as_of_date: "dt.date | None" = None) -> dict:
    """
    Returns:
        {"main": arr, "short": arr, "flow": arr, "delta": arr, "relvol": arr,
         "bos": arr, "as_of_a": date, "as_of_b": date, "as_of_s": date}

    *_hist arrays are each sleeve's dense daily-return history (fraction of
    $200k notional), ending the day BEFORE as_of_date (or the last date in
    the underlying CSVs if as_of_date is None / beyond available data).
    """
    # 2026-08-12: Strategy A disabled -- see portfolio_layer/position_sizer.py.
    # Universe A has NO coin-selection logic (option1_reference.build_matrices
    # globs every *_1h.csv in run/hist), so it would trade a universe that was
    # never backtested. Its histories are returned as ZEROS rather than computed:
    # compute_sleeve_multipliers() forces wA=0, so main/short/flow multipliers
    # are discarded regardless -- but computing them here pulls in a run/hist
    # calendar reaching back to 2020 and then hard-fails aligning Short to it.
    from portfolio_layer.position_sizer import STRATEGY_A_ENABLED

    delta_hist, relvol_hist, bos_hist, as_of_b = _delta_relvol_bos_histories(as_of_date)

    if STRATEGY_A_ENABLED:
        main_hist, flow_hist, main_dates, as_of_a = _main_flow_histories(as_of_date)
        short_hist, as_of_s = _short_history(as_of_date, main_dates)
    else:
        # zero-length-matched to universe B so any downstream len() check aligns
        zeros = np.zeros(len(delta_hist))
        main_hist = flow_hist = short_hist = zeros
        as_of_a = as_of_s = as_of_b

    return dict(
        main=main_hist, short=short_hist, flow=flow_hist,
        delta=delta_hist, relvol=relvol_hist, bos=bos_hist,
        as_of_a=as_of_a, as_of_b=as_of_b, as_of_s=as_of_s,
    )


if __name__ == "__main__":
    import time
    t0 = time.time()
    h = compute_sleeve_histories()
    for k in ("main", "short", "flow", "delta", "relvol", "bos"):
        arr = h[k]
        tail = np.round(arr[-5:], 5) if len(arr) >= 5 else arr
        print(f"{k:8s} n={len(arr):4d}  last5={tail}")
    print(f"as_of_a={h['as_of_a']}  as_of_b={h['as_of_b']}  as_of_s={h['as_of_s']}")
    print(f"elapsed {time.time()-t0:.1f}s")

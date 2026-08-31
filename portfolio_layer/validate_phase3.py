"""
Phase 3 validation harness -- numeric parity check against the cached
backtest reference series, analogous to Phase 2's signal_engine/parity_test.py.

Reports EXACT numbers, not pass/fail summaries, per standing instruction.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/user/workspace')
from portfolio_layer.sleeve_combiner import strategy_a_combo
from portfolio_layer.ab_blend import inv_vol_weighted


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(365)


print("=" * 78)
print("TEST 1: strategy_a_combo() vs backtest's eq_thirds / combo_a_correct")
print("=" * 78)
main = pd.read_csv('/home/user/workspace/main_realfill_series.csv', index_col=0, parse_dates=True)['pnl']
short = pd.read_csv('/home/user/workspace/short_realfill_series.csv', index_col=0, parse_dates=True)['pnl']
flow = pd.read_csv('/home/user/workspace/flow_lb5_phaseavg_realfill.csv', index_col=0, parse_dates=True)['pnl']
common = main.index.intersection(short.index).intersection(flow.index)
m, s, f = main.loc[common].values, short.loc[common].values, flow.loc[common].values

my_combo_a = strategy_a_combo(m, s, f)
saved_eq_thirds = np.load('/home/user/workspace/eq_thirds_ret.npy')
saved_combo_a_correct = np.load('/tmp/combo_a_correct.npy')

print(f"n days (recomputed from raw CSVs): {len(my_combo_a)}")
print(f"n days (eq_thirds_ret.npy):        {len(saved_eq_thirds)}")
print(f"n days (combo_a_correct.npy):       {len(saved_combo_a_correct)}")

if len(my_combo_a) == len(saved_eq_thirds):
    diff1 = np.abs(my_combo_a - saved_eq_thirds)
    print(f"vs eq_thirds_ret.npy:        max abs diff = {diff1.max():.10e}   "
          f"exact match = {np.array_equal(my_combo_a, saved_eq_thirds)}")
else:
    print(f"vs eq_thirds_ret.npy:        LENGTH MISMATCH, cannot diff elementwise "
          f"({len(my_combo_a)} vs {len(saved_eq_thirds)})")

if len(my_combo_a) == len(saved_combo_a_correct):
    diff2 = np.abs(my_combo_a - saved_combo_a_correct)
    print(f"vs combo_a_correct.npy:      max abs diff = {diff2.max():.10e}   "
          f"exact match = {np.array_equal(my_combo_a, saved_combo_a_correct)}")
else:
    print(f"vs combo_a_correct.npy:      LENGTH MISMATCH, cannot diff elementwise "
          f"({len(my_combo_a)} vs {len(saved_combo_a_correct)})")

print()
print("=" * 78)
print("TEST 2: ab_blend.inv_vol_weighted() vs backtest's combo_invvol.npy")
print("=" * 78)
a = np.load('/tmp/combo_a_correct.npy')
b = np.load('/tmp/combo_b_correct.npy')  # NOTE: this is the ORIGINAL (look-ahead nz()) combo_b, used
                                          # here only to test that MY COPY of inv_vol_weighted() itself
                                          # is bit-identical to the backtest's -- not to validate combo_b.
saved_invvol = np.load('/tmp/combo_invvol.npy')

my_combo_iv, my_wA = inv_vol_weighted(a, b, W=30)
diff3 = np.abs(my_combo_iv - saved_invvol)
print(f"n days: {len(my_combo_iv)} (matches saved: {len(my_combo_iv) == len(saved_invvol)})")
print(f"vs combo_invvol.npy:         max abs diff = {diff3.max():.10e}   "
      f"exact match = {np.array_equal(my_combo_iv, saved_invvol)}")
print(f"my recomputed Sharpe (full):  {sharpe(my_combo_iv):.6f}")
print(f"saved combo_invvol Sharpe:    {sharpe(saved_invvol):.6f}")
print(f"mean wA (full period):        {my_wA.mean():.6f}")

print()
print("=" * 78)
print("TEST 3: sleeve_combiner.strategy_b_combo() vs combo_b_correct.npy")
print("=" * 78)
print("UPDATED -- real DELTA/RELVOL/BOS leg series were later recovered by")
print("re-running reference_impl.py's own pipeline against live clean_panel data")
print("(see backtest_causal_b_v2.py; legs cached at raw_delta_ret.npy /")
print("raw_relvol_ret.npy / raw_bos_ret.npy). Two findings from that pass, both")
print("still open / NOT resolved to an exact match:")
print()
print("  1. combo_b_correct.npy is NOT the raw nz()-blend -- it is that blend")
print("     RESCALED to a target daily vol of ~0.963%/day. This rescale was")
print("     undocumented anywhere in the workspace before this pass; it is now")
print("     implemented in strategy_b_combo() as a causal (trailing-window)")
print("     version -- see sleeve_combiner.TARGET_DAILY_VOL docstring.")
print()
print("  2. Even with that rescale applied, reconstructing combo_b_correct.npy")
print("     from the CURRENT clean_panel data under the exact look-ahead nz()")
print("     formula gives only corr=0.986, R^2=0.973 (NOT exact) -- and the")
print("     reconstructed series' Sharpe (2.59 on the correct 568-day window)")
print("     is ~24% higher than the real cached combo_b_correct.npy's Sharpe")
print("     (2.09). This means combo_b_correct.npy embeds at least one more")
print("     undisclosed difference (most likely a frozen data snapshot or a")
print("     different universe selection at build time) that could NOT be")
print("     tracked down in this pass. strategy_b_combo()'s live behavior")
print("     remains UNVALIDATED against the original backtest to full precision --")
print("     see backtest_causal_b_v2.py output for exact numbers.")
print()
print("  Causal-vs-look-ahead comparison (now on the PROPERLY rescaled series,")
print("  apples-to-apples within this reconstruction): causal Sharpe 2.5749 vs")
print("  reconstructed look-ahead Sharpe 2.5895 -- a small (~0.6% relative) gap.")
print("  Do NOT read this as 'causal costs ~0% vs the real backtest' -- the")
print("  reconstruction itself already diverges materially (point 2 above) from")
print("  the real cached ground truth, for reasons independent of causality.")

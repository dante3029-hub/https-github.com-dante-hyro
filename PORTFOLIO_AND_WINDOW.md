# Portfolio simulator, and where 3.44 actually comes from

## The simulator

`portfolio.py` nets positions ACROSS sleeves before costing, which is what the
live portfolio layer does and what the blend does not:

1. every sleeve emits a weight per coin per day
2. weights summed per coin -> net desired position
3. per-coin cap applied to the NET (two sleeves at 0.15 on one coin is 0.30 real)
4. gross scaled if it exceeds the leverage limit
5. ONE set of costs on the net change, not one per sleeve

## It agrees with the blend

From IDENTICAL weight matrices:

| method | Sharpe |
|---|---|
| blend (nz each sleeve, average) | 1.88 |
| average raw, no nz | 1.71 |
| **sum weights, one cost (netted)** | **1.76** |
| sum weights, separate costs | 1.71 |

The `nz()` normalisation is worth **+0.17**, not the +1.7 I briefly suspected.
Netting is worth **+0.05** in Sharpe -- but saves **34% of gross exposure**
(0.721 -> 0.475), which is a capital-efficiency gain, not a return gain.

Per-coin caps at 0.10 / 0.15 / 0.25 give IDENTICAL results -- sleeves rarely
pile into the same coin hard enough for the cap to bind.

Trailing-vol risk parity (60d) gives 1.67 vs 1.69 equal-notional. No benefit.
Phase-averaging is worth +0.07 at book level.

## So where does 3.44 come from? THE WINDOW.

| sleeve | full history (1,315d) | blend window (922d) |
|---|---|---|
| delta | 1.26 | 1.28 |
| **relvol** | **1.02** | **1.40** |
| **skew** | **0.43** | **0.89** |
| **cascade** | **0.77** | **1.17** |

The blend ran on 922 days because it intersects with S/R, FVG and BOS, which
need more warm-up. **That window starts Feb 2024 and excludes all of 2023.**

The sleeves genuinely performed better in the later period. That is the
opposite of the usual decay story and is consistent with the half-split results
throughout -- second halves are stronger nearly everywhere in this book.

## Which number to use

| basis | Sharpe |
|---|---|
| Feb 2024 -> Aug 2026 (922d), 7 sleeves | 3.44 |
| full history (1,315d), 4 cross-sectional sleeves | 1.76 - 1.88 |
| **plan on** | **the lower** |

Both are defensible. 3.44 is real for the recent regime; 1.88 is what you get
if 2023 is representative of what comes next. The manuals' guidance -- halve
the backtest -- lands around the same place either way.

## Still not done

- maker vs taker fee sensitivity (1.5bp vs 8.5bp)
- vol targeting (30% band)
- CPCV across partitions
- event sleeves (sr, fvg, bos) not yet expressed as weights, so the netted
  simulator currently covers only the 4 cross-sectional sleeves

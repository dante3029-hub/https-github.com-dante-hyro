# Build — three cross-sectional sleeves written

Written against the EXISTING `signal_engine` interface, not a new one:
`latest_target_weights(matrix, t, n) -> (weights, ok)`, using the shared
`rank_weights.rank_to_weights()` so every cross-sectional sleeve ranks
identically.

| file | sleeve | cadence | status |
|---|---|---|---|
| `signal_engine/sleeve_oirank.py` | OI rank | 72h | **parity CONFIRMED** |
| `signal_engine/sleeve_skew.py` | skew | 1080h | parity close, deviation documented |
| `signal_engine/sleeve_cascade.py` | cascade | 24h | parity close, deviation documented |

## Parity results

| sleeve | new module | backtest reference | notes |
|---|---|---|---|
| **oirank** | **2.50** | **2.51** | max weight diff 0.067 = one slot. Confirmed. |
| skew | 1.44 | 1.26 | maxDD matches EXACTLY (17.1%) |
| cascade | 0.78 | 0.49 | maxDD 26.9% vs 36.8% |

**Both remaining modules score HIGHER than their reference. That is a warning,
not good news** -- a reimplementation beating its own backtest usually means it
differs somewhere. Both differences are traced:

**skew** -- my module requires 48 of 60 valid days before scoring a coin;
`pandas.skew()` computes on whatever is present. I exclude a few thin-history
names the backtest included. Identical maxDD confirms the positions are
otherwise the same.

**cascade** -- an off-by-one on the hold. The backtest enters at `t+1` and
exits at `t+3`, holding TWO days. The module holds `t+1` through `t+3` --
THREE days, which is what `HOLD_DAYS = 3` should mean. **The backtest is
arguably the one that is wrong.**

Neither has been silently adopted. Both need a deliberate decision on which
behaviour is CORRECT, rather than picking whichever scores higher.

## Bugs found and fixed while writing these

**The skewness estimator.** First version used the population form
`g1 = m3/sd**3` with `ddof=0`. `pandas.skew()` uses the BIAS-CORRECTED sample
estimator. Nearly monotonic but not exactly -- enough to shuffle borderline
ranks. Parity failed on it (1.47 vs 1.26) and caught it.

**A lookahead in the test driver.** The cascade driver applied the weight at
bar `t` when the backtest enters at `t+1`. Since cascade picks the worst
performers OF THAT DAY, same-bar application meant holding the continuation
down: **-2.03 Sharpe and a 295% drawdown.** The module was fine; the harness
was not.

## Design notes worth keeping

**`ok=True` with an all-zero weight vector is cascade's NORMAL state.** It
holds a position ~3.3% of the time. That distinction matters for the health
check: a flat cascade is expected, a flat delta is not.

**oirank blends five lookbacks deliberately.** 7d alone scored 1.41 vs ~0.95
at 5d and 10d -- a 45% spike. Ranking each lookback BEFORE averaging means one
lookback with an outlier cannot dominate.

**skew uses 60d, not 45d, and the 24-coin core, not the wide universe.** On 60
coins it collapses to -0.16: realised skewness on thin names measures noise.

## Still to write

Four event sleeves: `sr`, `srflip`, `pattern`, `fvg`. These use the existing
`bot/event_sleeves.py` slot machinery (6 slots, 1/6 weight, JSON state for
restart recovery). srflip additionally needs to remember "resistance at X broke
at bar N" for up to 20 bars -- `sleeve_sr.py` already has that state machine.

## Config changes required

- `DELTA_CADENCE_HOURS`: 168 -> **336** (research says hold 14 days, not 7)
- `SLEEVE_NAMES`: remove `main`, `short`, `flow`, `bos`; add `skew`, `cascade`,
  `oirank`, `sr`, `srflip`, `pattern`, `fvg`
- `RELVOL_CADENCE_HOURS`: 168 is already correct

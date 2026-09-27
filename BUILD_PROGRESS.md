# Build — seven sleeve modules written, ALL AT PARITY

Written against the EXISTING `signal_engine` interface, using the shared
`rank_weights.rank_to_weights()` so no cross-sectional sleeve can drift from
delta and relvol.

## Status

| module | sleeve | cadence | parity |
|---|---|---|---|
| `sleeve_oirank.py` | OI rank | 72h | **2.50 vs 2.51** |
| `sleeve_cascade.py` | cascade | 24h | **0.43 vs 0.49** |
| `sleeve_skew.py` | skew | 1080h | maxDD and bear half identical |
| `sleeve_price_events.py` | sr | 6h | **23 vs 23, EXACT** |
| | srflip | 6h | **12/4 vs 12/4, EXACT** |
| | pattern | 6h | **51/51 bars agree** |
| | fvg | 12h | **196 vs 196, EXACT** |

Plus `sleeve_health.py` -- self-diagnosing checks that catch all three
historical silent failures.

## THE BUG THAT MATTERED: the OI gate belongs at ENTRY, not at detection

The event sleeves initially matched the backtest only **29%** of the time. The
levels were identical (0 of 6 sampled bars differed), so formation was causal
and correct. The cause was WHICH BAR the OI filter was applied on.

On SOL 6h:

| gate | signals kept |
|---|---|
| no OI filter | 60 |
| OI at the SIGNAL bar `i` | 31 |
| **OI at the ENTRY bar `i+1`** | **23**  <- what the backtest does |
| overlap of the two | **12 (29%)** |

Both are causal -- OI is a live snapshot, so at the moment the order goes in
(the open of bar i+1) current OI is observable. But they select almost
different strategies.

**Fix:** `latest_signal()` now DETECTS only. `oi_expanding=None` means "the
caller gates at order time". The `Signal` carries `needs_oi` so the tracker
knows to query OI when it actually places the order.

After the fix: exact parity on every sleeve.

## Other bugs caught by parity, before they reached live

**The skewness estimator.** First version used the population form
`g1 = m3/sd**3` with `ddof=0`. `pandas.skew()` uses the BIAS-CORRECTED sample
estimator. Nearly monotonic, but enough to shuffle borderline ranks.

**A lookahead in the cascade test driver.** Applied the weight at bar `t` when
the backtest enters at `t+1`. Since cascade buys the worst performers OF THAT
DAY, same-bar application meant holding the continuation down: **-2.03 Sharpe,
295% drawdown.** The module was fine; the harness was not.

## Design decisions worth keeping

**One module for all four price events, not four files.** The ATR, the entry
convention and the exit loop would otherwise be duplicated four times -- and
each was a real bug source in research (stop placed off the entry bar's CLOSE
while entering at its OPEN, worth -0.21 on fvg; exit loop skipping the entry
bar, which made fvg at 0.25 ATR "score" 2.84 when it is really 0.06).

**`ok=True` with an all-zero weight vector is cascade's NORMAL state.** It
holds a position ~3.3% of the time. The health check knows a flat cascade is
expected and a flat delta is not.

**oirank blends five lookbacks.** 7d alone scored 1.41 vs ~0.95 at 5d and 10d.
Ranking each lookback BEFORE averaging stops one outlier dominating.

**skew uses 60d and the 24-coin core.** On 60 coins it collapses to -0.16 --
realised skewness on thin names measures noise.

## Remaining work

1. **A generic long-event tracker.** `bot/event_sleeves.py` has
   `ShortSleeveTracker` and `BOSSleeveTracker`; the four new sleeves need the
   same slot machinery (6 slots, 1/6 weight, JSON state) but long-biased, and
   must apply the OI gate at order time.
2. **Config:** `DELTA_CADENCE_HOURS` 168 -> **336**. `SLEEVE_NAMES`: drop
   `main`/`short`/`flow`/`bos`, add the seven new ones.
3. **Wire `sleeve_health.run_health_check()`** into the orchestrator and refuse
   to place orders while any CRITICAL is open.
4. **Performance:** the pattern detector re-scans full history on every call.
   Fine at one call per coin per 6h close; it was the slow part of the parity
   test at ~4,000 calls.

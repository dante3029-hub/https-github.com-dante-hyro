# bot/price_event_tracker.py — slot management for the four event sleeves

ONE class parameterised by sleeve name, not four near-identical ones.
`latest_signal()` answers "is there a fresh trigger"; slot management, exits
and restart recovery live here -- the same split as ShortSleeveTracker and
BOSSleeveTracker already use.

## Verified

| check | result |
|---|---|
| slots opened over 1,500 replayed bars | 69 |
| max concurrent | **4** (cap 6) |
| max gross | **0.667** (cap 1.000) |
| bars holding >=1 slot | 46% |
| JSON round-trip | OK (234 bytes) |
| **restart recovery** | resumed with 1 open slot, gross 0.167 |
| protective stops exposed | yes |

Driven bar by bar exactly as the live loop will: at step `i` the tracker sees
only `bars.iloc[:i+1]`.

## Three things it gets right that cost real money to learn

**1. The OI gate is at ORDER time, not detection.**
60 raw sr breaks on SOL 6h. Gating at the SIGNAL bar keeps 31; at the ENTRY
bar keeps 23; overlap is 12 -- **29%**. Both are causal (OI is a live snapshot
observable when the order goes in) but they select almost different
strategies. `latest_signal(..., oi_expanding=None)` DETECTS only; the tracker
queries OI when it opens the slot.

**2. The exit walk resumes from `last_checked_ts`, never from entry.**
ShortSleeveTracker carried a documented bug where the walk restarted at entry
every cycle while trailing state persisted -- bars already survived were
re-tested against a ratcheted stop and a slot was force-exited with ZERO new
market data. Live that would have flushed the sleeve every cycle.

**3. One position per coin per sleeve.**
Without it, sr stacked 4 of 6 slots into one coin across 5,304 coin-hours and
fvg 3 slots across 16,525. That concentration earned nothing -- capping it was
free and improved three of four sleeves.

## Exits

**stop** -- checked from the ENTRY BAR onward, not entry+1. Skipping the entry
bar made tight stops look monotonically better (fvg at 0.25 ATR "scored" 2.84;
it is really 0.06).

**time** -- after `hold_bars` on the sleeve's own timeframe.

## `set_actual_entry(coin, fill_price)`

Recomputes the stop from the REAL fill. The stop must sit a fixed ATR distance
from where you actually got in, not from the previous close -- placing it off
the signal bar's close while entering at the next open is lookahead, worth
-0.21 on fvg before it was fixed.

## Remaining work

1. **Config**: `DELTA_CADENCE_HOURS` 168 -> **336**; `SLEEVE_NAMES` drop
   `main`/`short`/`flow`/`bos`, add the seven new sleeves with their cadences.
2. **Wire `sleeve_health.run_health_check()`** into the orchestrator; refuse to
   place orders while any CRITICAL is open.
3. **End-to-end parity**: run the whole book through the orchestrator on
   history and confirm the portfolio-level output matches `hourly_sim`. The
   individual sleeves match; the COMBINATION through `size_portfolio()` has
   not been verified.
4. **Nothing here has touched a live API.** The sleeves are proven against the
   backtest, not against partial fills, rejected orders, or whether the perp
   is tradeable in the size wanted. That is what the paper month is for.

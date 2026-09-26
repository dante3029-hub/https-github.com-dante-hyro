# Second audit — four real bugs, and the verified book

Triggered by building `sleeve_sr.py` as ONE implementation for live and
backtest. Running it against the existing code exposed bugs that had been
sitting in numbers we were about to trade on.

## THE BOOK — 7 sleeves, 3.87

| sleeve | Sharpe | bull | bear | ann% | maxDD |
|---|---|---|---|---|---|
| fvg | 3.28 | 3.76 | 2.85 | 204.2% | 28.9% |
| sr | 1.93 | 2.02 | 1.83 | 59.6% | 21.0% |
| pattern | 1.83 | 1.85 | 1.85 | 57.3% | 23.3% |
| delta | 1.41 | 0.77 | 2.04 | 25.8% | 32.9% |
| skew | 1.26 | 1.12 | 1.38 | 20.9% | 17.1% |
| relvol | 1.17 | 1.02 | 1.31 | 15.5% | 14.8% |
| cascade | 0.49 | -0.32 | 1.15 | 13.4% | 36.8% |

**BOOK: 3.87 Sharpe, bull 3.44, bear 4.27, 5.9% maxDD. Daily Sharpe ~4.0.**

srflip excluded. Corrected it is 1.37 (3ATR) and costs 0.07 in the book. It
does cut maxDD 5.9% -> 4.5%, so it is worth revisiting, but not yet.

## Bug 1 — S/R used the wrong volume series

`upAndDownVolume()` in the Pine signs volume by **candle direction**:

```pine
close > open => isBuyVolume := true
if isBuyVolume: posVol += volume else: negVol -= volume
```

My `sleeve_sr.py` preferred real taker delta when available. **Different data
entirely.** 631 trades instead of 573.

## Bug 2 — break detected on close, not low

```pine
brekout_res := ta.crossover(low, resistanceLevel_1)
```

It is the **LOW**. The whole candle must clear the box. `sleeve_sr.py` used the
close; `hourly_sim.srflip_hourly` used the close. `sr2.py` had it right.

Fixing bugs 1+2 turned **negative second halves positive** -- break went from
1.16 (halves 2.50 / -0.44) to 1.70 (halves 2.03 / 1.33). That sign flip is
strong evidence the fixes were correct.

For srflip it cut trades 450 -> 335 and Sharpe 1.71 -> 1.37.

## Bug 3 — stop-placement lookahead in THREE sleeves

```python
stp = c[eb] - side*stop_atr*A[eb]     # entry is at o[eb]
```

Entry at the bar's OPEN, stop placed off that same bar's CLOSE. You do not know
the close when you enter at the open. `A[eb]` includes the entry bar too.

It favours the strategy: when the entry bar moves against you, the stop lands
further away.

| sleeve | before | after | change |
|---|---|---|---|
| sr | 2.02 | 1.93 | -0.09 |
| **fvg** | 3.49 | **3.28** | **-0.21** |
| pattern | 1.81 | 1.83 | +0.02 |

breakout and srflip already used `o[eb]` and were clean.

## Bug 4 — a crash I introduced fixing bug 3

`sr_hourly` and `fvg_hourly` never unpacked the `open` column. Fixed.

## What passed

**FVG detection is an exact match to the Pine** -- `low > high[2] and
close[1] > high[2]`, same threshold, same bear mirror.

**Parity holds in `sleeve_sr.py`**: `live_signal` and `backtest` fire on the
same bar. That is structural -- one `step()` function, swept for backtest,
called once for live.

**Note on FVG**: the LuxAlgo indicator only DRAWS gaps. "Enter in the gap
direction, 1xATR stop, 10-bar hold" is MY trading rule, not the source's. It
has been swept and controlled, but the Pine does not validate it.

## sleeve_sr.py — the pattern to follow for the rest

One `step(P, i, state)` advancing a state machine by one bar using only data up
to `i`. Backtest sweeps it; live calls it once. The 403 / 572 / 631 trade-count
disagreements cannot happen with one code path.

`SRState` is JSON-serialisable so the live bot survives restarts -- srflip needs
to remember "resistance at X broke at bar N" for up to 20 bars.

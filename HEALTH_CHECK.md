# sleeve_health.py — the bot tells YOU what is wrong

## Why

Three real failures on this project, every one SILENT:

| failure | how long it hid | why nothing caught it |
|---|---|---|
| **BOS produced zero weights** | 300+ live cycles | the log said `rebalanced=['short','bos']` every time |
| **`run/` went 408h stale** | weeks | cycles kept "succeeding" on frozen data |
| **taker_buy > total volume on 60% of bars** | months | nothing ever validated the data |

Each was findable in ONE cycle by a check that knew what normal looks like.

## Design

Every sleeve declares an `Expectation` -- how often it should fire, what weight
range is sane, which files it needs and how fresh. Each cycle the checker
compares reality against that and emits specific, actionable findings.

**A check never silently passes on missing information.** If it cannot
evaluate, that is a WARN, not an OK.

| severity | meaning |
|---|---|
| CRITICAL | stop trading -- data is wrong or a sleeve is structurally broken |
| ERROR | this sleeve is not working; others may continue |
| WARN | outside expectation, but could be a quiet market |
| INFO | normal |

## Verified against the real failures

```
[CRITICAL] sr      NO WEIGHTS for far too long  400 cycles flat; expected a
                   position every ~3. THIS IS THE BOS FAILURE -- check the
                   entry condition can ever be true

[CRITICAL] delta   taker_buy EXCEEDS total volume  455/500 recent bars --
                   this is the spliced-venue bug, delta is meaningless

[CRITICAL] delta   delta is never negative  only 0.0% of bars -- any sleeve
                   needing delta<0 can NEVER fire (the BOS bug)

[CRITICAL] sr      data STALE  SOL_1h.csv last written 250h ago (limit 8h)

[ERROR   ] fvg     gross above cap  2.200 vs 1.000 -- the sleeve is running
                   levered (the FVG 2.1x bug)

[CRITICAL] sr      SHORT position in a long-only sleeve  {'ETH': -0.16}
```

The delta check is the important one: **it catches the corrupt column two
different ways**, either of which would have found the BOS bug on day one.

## What each check covers

**`check_data_freshness`** -- file exists, is readable, and was written
recently. Catches the 408h staleness.

**`check_data_sanity`** -- NaN, non-positive prices, taker>volume, delta that
is never negative or never positive, gaps in recent bars.

**`check_weights`** -- flat for longer than the sleeve's expected gap, per-coin
cap breached, gross cap breached, wrong-side positions in a directional sleeve.

**`check_cadence`** -- rebalance overdue against the sleeve's own schedule.

## Expectations for the book

`trades_per_month` and `active_frac` come from the hourly backtest. A live
sleeve deviating sharply is not proof of a bug, but is always worth knowing --
and ZERO when 20 are expected is the BOS failure repeating.

| sleeve | trades/mo | active | cadence | direction |
|---|---|---|---|---|
| delta | -- | 85% | 336h | both |
| relvol | -- | 65% | 168h | both |
| skew | -- | 84% | 1080h | both |
| oirank | -- | 90% | 72h | both |
| cascade | 2.5 | 3.3% | 24h | LONG only |
| sr | 13 | 24% | 6h | LONG only |
| srflip | 8 | 16% | 6h | LONG only |
| pattern | 19 | 27% | 6h | LONG only |
| fvg | 126 | 95% | 12h | both |

## Wire it in

Call `run_health_check()` at the end of every orchestrator cycle, log
`format_report()`, and refuse to place orders while any CRITICAL is open.

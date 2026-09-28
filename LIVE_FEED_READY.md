# The live feed exists now

## What was blocking live trading

`LiveDataFeed` in `data_feed.py` is a deliberate hard stop that RAISES. Its
docstring listed the reasons, and they were correct at the time:

> live open-interest ingestion (only static run/oi/*.csv, up to 15 days stale)
> live taker-flow ingestion (same staleness issue)
> a live 4h/1h kline poller

**All three were about STALENESS.** `topup.py` on an hourly cron fixes all
three: `taker_data/` and `oi_data/` are current to the hour, and the klines
they hold ARE the poller's output.

The stub's other objections -- order book depth, cross-venue basis -- were
spec'd for a different design. **No sleeve in the nine-sleeve book uses them.**
All nine need only OHLCV, taker delta and open interest.

This also explains the demo account: everything to date ran on
`ReplayDataFeed` reading static CSVs, which is why equity moved $76 across
many cycles with 24 positions supposedly open.

## Verified

```
  freshness:   24/24 coins, worst bar age 0.88h
  snapshot:    LiveSnapshot, data_source='live'
  weights:     7 sleeves, 2 with positions this bar
  stops:       exposed for sync_protective_stops()
```

**And the gate actually refuses:**

```
  24/24 coins truncated to 20 days old
  -> StaleDataError: only 0/24 coins have taker data under 8h
     (worst 482h). Is topup.py running?
```

A stale feed is a HARD STOP, not a warning. The failure it prevents is a
nine-sleeve book trading week-old prices -- which no downstream check catches,
because every sleeve computes cleanly on old data.

**Freshness is measured on the LAST BAR, never file mtime.** `git checkout`
once rewrote every data file, refreshing mtimes while reverting content to 17
days old, and an mtime check called that healthy.

## Separation of fetch and read

`live_feed.py` does NOT fetch. It reads what `topup.py` maintains, and refuses
if that is stale. A dead fetcher therefore shows up as a hard stop rather than
as silently frozen prices -- which is exactly the 408-hour failure the old bot
had.

## A robustness bug found while wiring book_v2

**oirank returned FLAT whenever OI lagged price by one bar.** OI and taker are
topped up by separate fetchers; if OI is behind, the union index leaves the
last row all-NaN, `oi_scores` finds nothing, and the sleeve reports zero
positions -- indistinguishable from a quiet market.

Fixed with `ffill(limit=3)` plus an explicit error when fewer than 12 of 24
coins have current OI. The LIMIT matters: without it the sleeve would quietly
trade on arbitrarily stale positioning if the OI fetcher died.

Confirmed both ways -- with 10-day-old OI it refuses and logs; with current OI
it ranks 5 long / 5 short at gross 1.000.

## Remaining

1. Wire `LiveDataFeed.get_snapshot()` into `orchestrator.py` alongside the
   existing delta/relvol engine call
2. `burn_in.py --mode offline`
3. `burn_in.py --mode demo` (the same demo account already in use)
4. mainnet at $300/day with `DISCORD_WEBHOOK_BOT` set and the watchdog cron
